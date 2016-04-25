/*
 * bfs_kernel.hpp
 *
 *  Created on: Mar 27, 2012
 *      Author: koji
 */

#ifndef BFS_KERNEL_HPP_
#define BFS_KERNEL_HPP_

#include "gpu.hpp"

namespace cuda {

// Initialize
void cu_initialize_memory(
	int num_local_vertices,
	int bitmap_size_visited,
	int bitmap_size_v0,
	int bitmap_size_v1,
	int summary_size,
	int64_t* pred,
	uint32_t* nq_bitmap,
	uint32_t* nq_sorted_bitmap,
	uint32_t* visited,
	uint32_t* cq_summary,
	uint32_t* cq_bitmap,
	uint32_t* shared_visited,
	cudaStream_t* streams,
	int num_streams)
{
	memset_gpu<int64_t>(pred, -1, num_local_vertices, streams[0 % num_streams]);
	memset_gpu<uint32_t>(nq_bitmap, 0, bitmap_size_visited, streams[1 % num_streams]);
	memset_gpu<uint32_t>(nq_sorted_bitmap, 0, bitmap_size_visited, streams[2 % num_streams]);
	memset_gpu<uint32_t>(visited, 0, bitmap_size_visited, streams[3 % num_streams]);
	memset_gpu<uint32_t>(shared_visited, 0, bitmap_size_v1, streams[4 % num_streams]);
	if(cq_summary && cq_bitmap) {
		memset_gpu<uint32_t>(cq_summary, 0, summary_size, streams[5 % num_streams]);
		memset_gpu<uint32_t>(cq_bitmap, 0, bitmap_size_v0, streams[6 % num_streams]);
	}
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void initialize_create_column_list(
	int summary_size,
	int* __restrict__ const count)
{
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	if(gid <  summary_size) count[gid] = 0;
}

// Number of elements processed by a thread-block
// summary_size : WARPS_PER_BLOCK
// requirements : sizeof(uint32_t)*8 == WARP_SIZE
// blockDim: (WARP_SIZE, WARPS_PER_BLOCK)
__global__ void create_column_list_count_kernel(
	uint32_t* __restrict__ const bitmap,
	uint32_t* __restrict__ const summary,
	int summary_size,
	int* __restrict__ const count)
{
	if(WARP_IDX >= summary_size) return;

	__shared__ uint s_flags_buffer[WARPS_PER_BLOCK*sizeof(uint32_t)*8];
	__shared__ int s_reduce_buffer[WARPS_PER_BLOCK*(WARP_SIZE+1)];
	volatile uint* s_flags = &s_flags_buffer[threadIdx.y*sizeof(uint32_t)*8];
	volatile int* s_reduce = &s_reduce_buffer[threadIdx.y*(WARP_SIZE+1)];

	// Processing Thread Group: Warp
	const uint32_t i_summary = summary[WARP_IDX];
	s_flags[threadIdx.x] = 0;
	for(uint i = 0; i < sizeof(i_summary)*8; ++i) {
		if(i_summary & ((uint32_t)1 << i)) {
			const uint idx = (WARP_IDX*sizeof(i_summary)*8 + i) * WARP_SIZE + threadIdx.x;
			const uint32_t i_bitmap = bitmap[idx];
			const uint cq_flags = __ballot(i_bitmap);
			if(threadIdx.x == 0) {
				s_flags[i] = cq_flags;
			}
		}
	}

	const uint cq_flags = s_flags[threadIdx.x];
	s_reduce[threadIdx.x] = __popc(cq_flags);
	// We don't need to synchronize because the next reduction is performed within a warp.
//	__syncthreads();
	dev_reduce<int, 32>(32, s_reduce + 0);
	if(threadIdx.x == 0) {
		count[WARP_IDX] = s_reduce[0];
	}
}

// Number of elements processed by a thread-block
// summary_size : WARPS_PER_BLOCK
__global__ void create_column_list_set_kernel(
	uint32_t* __restrict__ const bitmap,
	volatile uint32_t* __restrict__ const summary,
	int summary_size,
	int* __restrict__ const offset,
	uint2* __restrict__ const columns)
{
	if(WARP_IDX >= summary_size) return;

	const uint32_t summary_flags = summary[WARP_IDX];
	const uint mask = (1U << threadIdx.x) - 1U;

	int base_offset = offset[WARP_IDX];

	// Processing Thread Group: Warp
	for(int i = 0; i < sizeof(uint32_t)*8; ++i) {
		if(summary_flags & ((uint32_t)1 << i)) {
			const int idx = (WARP_IDX*sizeof(uint32_t)*8 + i) * WARP_SIZE + threadIdx.x;
			const uint32_t i_bitmap = bitmap[idx];
			// clear CQ
			bitmap[idx] = 0;
			const uint cq_flags = __ballot(i_bitmap);
			const int dst_offset = __popc(mask & cq_flags);
			if(i_bitmap) {
				columns[base_offset + dst_offset] = make_uint2(idx, i_bitmap);
			}
			base_offset += __popc(cq_flags);
		}
	}

	// clear summary
	if(threadIdx.x == 0) {
		summary[WARP_IDX] = 0;
	}
}

__global__ void fill_cq_list_last_block(
	int c,
	const int* const offset,
	uint2* const columns)
{
	__shared__ uint2* base;
	__shared__ int length;
	if(threadIdx.x == 0) {
		int last = *offset;
		base = columns + last;
		length = ((last + THREADS_PER_BLOCK - 1) & (-THREADS_PER_BLOCK)) - last;
	}
	__syncthreads();
	if(threadIdx.x < length) base[threadIdx.x] = make_uint2(c, 0);
}

void create_cq_list(
	uint32_t* bitmap,
	uint32_t* summary,
	int summary_size,
	PrefixSumGPU<int>* cq_count,
	uint2* column_buffer,
	int empty_column,
	cudaStream_t stream)
{
	int nblocks;

	nblocks = nblocks2n<int>(summary_size, THREADS_PER_BLOCK);
	initialize_create_column_list<<<nblocks, THREADS_PER_BLOCK, 0, stream>>>
		(summary_size, cq_count->get_buffer() + 1);

	nblocks = nblocks2n<int>(summary_size, WARPS_PER_BLOCK);
	create_column_list_count_kernel<<<nblocks, dim3(WARP_SIZE, WARPS_PER_BLOCK), 0, stream>>>
		(bitmap, summary, summary_size, cq_count->get_buffer() + 1);

	(*cq_count)(&stream, 1);

	create_column_list_set_kernel<<<nblocks, dim3(WARP_SIZE, WARPS_PER_BLOCK), 0, stream>>>
		(bitmap, summary, summary_size, cq_count->get_buffer(), column_buffer);

	fill_cq_list_last_block<<<1, THREADS_PER_BLOCK, 0, stream>>>
		(empty_column, cq_count->get_buffer() + summary_size, column_buffer);
}

__device__ int read_graph_is_visited(
	uint32_t* __restrict__ const shared_visited,
	int64_t e1)
{
	const int64_t c1 = (e1 & INT64_C(0xFFFFFFFFFFFF)) / NUMBER_PACKING_EDGE_LISTS;
	const int64_t word_idx = c1 / NUMBER_PACKING_EDGE_LISTS;
	const int bit_idx = c1 % NUMBER_PACKING_EDGE_LISTS;
	const uint mask = 1U << bit_idx;

#if SV_ATOMIC_LEVEL == 2
	return !(atomicOr(&shared_visited[word_idx], mask) & mask);
#elif SV_ATOMIC_LEVEL == 1
	// Is this technique effective on GPU ???
	if((shared_visited[word_idx] & mask) == 0) {
		if((atomicOr(&shared_visited[word_idx], mask) & mask) == 0) {
			return 1;
		}
	}
	return 0;
#else
	// Is this technique effective on GPU ???
	if((shared_visited[word_idx] & mask) == 0) {
		shared_visited[word_idx] |= mask;
		return 1;
	}
	return 0;
#endif
}

__global__ void initialize_read_graph_input(
	ReadGraphInput* input)
{
	input->reserve_offset = 0;
	input->intermid_offset = 0;
	input->output_offset = 0;
	input->skipped = false;
}

__global__ void initialize_filter_edges_input(
	ReadGraphInput* input,
	int num_edges)
{
	input->intermid_offset = num_edges;
	input->output_offset = 0;
	input->skipped = false;
}

/*
#if KD_PRINT
#define TRV_V0_SORTED
#define TRV_V0_PRIG
#endif
*/

// blockDim: dim3(WARP_SIZE, WARPS_PER_BLOCK)
__global__ void read_graph_and_store(
	ReadGraphInput* in_param,
	const int64_t* row_starts,
	const int32_t* index_array_high,
	const uint16_t* index_array_low,
	uint2* columns,
	const int column_start, const int column_end,
	const int empty_column,
	int* out_columns,
	int64_t* out_indices)
{
	__shared__ uint32_t s_bitmap[THREADS_PER_BLOCK];
	__shared__ int s_columns[THREADS_PER_BLOCK];
	__shared__ int s_r_offset[THREADS_PER_BLOCK];
	__shared__ int s_out_offset[THREADS_PER_BLOCK*2];

	const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
	int th_length;

	{
		const int column_offset = column_start + THREADS_PER_BLOCK * blockIdx.x + tid;
		int column = (int)columns[column_offset].x;
		s_bitmap[tid] = columns[column_offset].y;
		s_columns[tid] = column;
		int r_start = (int)row_starts[column];
		int r_end = (int)row_starts[column + 1];
		th_length = r_end - r_start;
		s_r_offset[tid] = r_start;
		s_out_offset[tid] = th_length;
#if 0
			if(column == 253941 / NUMBER_PACKING_EDGE_LISTS
					&& (s_bitmap[tid] & (1 << (253941 % NUMBER_PACKING_EDGE_LISTS)))) {
				printf("===Debug Info: K:%d:r:%d Found pred==253941, NumEdges(%d), tid=%d,\n", __LINE__, mpig.rank_2d, th_length, tid);
				for(int i = 0; i < th_length; ++i) {
					int64_t e1 = (int64_t(index_array_high[r_start + i]) << 16) | index_array_low[r_start + i];
					if(e1 == 462017) {
						printf("===Debug Info: index[%d + %d] = %"PRId64"\n", r_start, i, (int64_t(index_array_high[r_start + i]) << 16) | index_array_low[r_start + i]);
					}
				}
			}
#endif
	}

	__syncthreads();

	dev_reduce<int, THREADS_PER_BLOCK>(THREADS_PER_BLOCK, s_out_offset, tid);

	__shared__ int out_base;
	__shared__ bool s_skip_task;

	if(tid == 0) {
		int total_length = s_out_offset[0];
		if(total_length == 0) {
			// for speed up of retry turn
			s_skip_task = true;
		}
		else if(atomicAdd(&in_param->reserve_offset, total_length) + total_length
				> READ_GRAPH_OUTBUF_SIZE) {
			atomicAdd(&in_param->reserve_offset, - total_length);
			s_skip_task = true;
			in_param->skipped = true;
		}
		else {
			s_skip_task = false;
			out_base = atomicAdd(&in_param->intermid_offset, total_length);
		}
		s_out_offset[0] = 0;
	}

	__syncthreads();

	if(s_skip_task) {
		return ;
	}

	if(th_length >= WARP_SIZE)
		s_out_offset[tid + 1] = th_length;
	else
		s_out_offset[tid + 1] = 0;

	__syncthreads();

	dev_prefix_sum<int, LOG_THREADS_PER_BLOCK>(s_out_offset, tid);
	// dev_prefix_sum include __syncthreads() at last of it

	const int th_attack_out_offset = s_out_offset[THREADS_PER_BLOCK];
	int* out_columns_base = out_columns + out_base;
	int64_t* out_indices_base = out_indices + out_base;

	// block attack
	{
		__shared__ int s_control;
		__shared__ int s_length;
		bool vote = (th_length >= THREADS_PER_BLOCK);
		while(__syncthreads_or(vote)) {
			// vie for control of CTA
			if(vote) s_control = tid;
			__syncthreads();
			// winner describes adjlist
			if(tid == s_control) {
				s_length = th_length;
				th_length = 0;
				vote = false;
			}
			__syncthreads();
			const int winner = s_control;
			const int r_offset = s_r_offset[winner];
			const int out_offset = s_out_offset[winner];
			const uint32_t bitmap = s_bitmap[winner];
			const int column = s_columns[winner];

			const int32_t* idx_high_ptr = index_array_high + r_offset;
			const uint16_t* idx_low_ptr = index_array_low + r_offset;
			int* out_columns_ptr = out_columns_base + out_offset;
			int64_t* out_indices_ptr = out_indices_base + out_offset;
			const int length = s_length;

			for(int i = tid; i < length; i += THREADS_PER_BLOCK) {
				const int64_t e1 = ((int64_t)idx_high_ptr[i] << 16) | idx_low_ptr[i];
				const int v0_lowbits = e1 % NUMBER_PACKING_EDGE_LISTS;
				int c_value = column;
				if((bitmap & (1U << v0_lowbits)) == 0) {
					c_value = -1;
				}
#if KD_PRINT
				else if(c_value < 0 || c_value >= empty_column) {
					printf("K:%d:c_value(%d) is out of range\n", __LINE__, c_value);
				}
#endif
#if 0
			if(r_offset + i == 7370793 + 10879) {
				printf("===Debug Info: K:%d:r:%d Found pred==7268\n", __LINE__, mpig.rank_2d);
			}
#endif
#if 0
			if(e1 / NUMBER_PACKING_EDGE_LISTS == 115504 && c_value == 253941 / NUMBER_PACKING_EDGE_LISTS
					&& (e1 % NUMBER_PACKING_EDGE_LISTS) == (253941 % NUMBER_PACKING_EDGE_LISTS)) {
				printf("===Debug Info: K:%d:r:%d Found pred==22504\n", __LINE__, mpig.rank_2d);
			}
#endif
				out_columns_ptr[i] = c_value;
				out_indices_ptr[i] = e1;
			}
		}
	}

	// warp attack
	{
		__shared__ int16_t s_control_buffer[WARPS_PER_BLOCK];
		__shared__ int s_length_buffer[WARPS_PER_BLOCK];
		volatile int16_t *s_control = &s_control_buffer[threadIdx.y];
		volatile int *s_length = &s_length_buffer[threadIdx.y];
		bool vote = (th_length >= WARP_SIZE);
		while(__any(vote)) {
			if(vote) *s_control = tid;
			// winner describes adjlist
			if(tid == *s_control) {
				*s_length = th_length;
				th_length = 0;
				vote = false;
			}

			const int winner = *s_control;
			const int r_offset = s_r_offset[winner];
			const int out_offset = s_out_offset[winner];
			const uint32_t bitmap = s_bitmap[winner];
			const int column = s_columns[winner];

			const int32_t* idx_high_ptr = index_array_high + r_offset;
			const uint16_t* idx_low_ptr = index_array_low + r_offset;
			int* out_columns_ptr = out_columns_base + out_offset;
			int64_t* out_indices_ptr = out_indices_base + out_offset;
			const int length = *s_length;
#if 0
			if(r_offset == 207757 && (tid%32) == 0) {
				printf("===Debug Info: K:%d:r:%d Hit!! pred==7268, length=%d, tid=%d\n", __LINE__, mpig.rank_2d, length, tid);
			}
#endif

			for(int i = threadIdx.x; i < length; i += WARP_SIZE) {
				const int64_t e1 = ((int64_t)idx_high_ptr[i] << 16) | idx_low_ptr[i];
				const int v0_lowbits = e1 % NUMBER_PACKING_EDGE_LISTS;
				int c_value = column;
				if((bitmap & (1U << v0_lowbits)) == 0) {
					c_value = -1;
				}
#if KD_PRINT
				else if(c_value < 0 || c_value >= empty_column) {
					printf("K:%d:c_value(%d) is out of range\n", __LINE__, c_value);
				}
#endif
#if 0
			if(r_offset + i == 207800) {
				printf("===Debug Info: K:%d:r:%d Found pred==7268\n", __LINE__, mpig.rank_2d);
			}
#endif
#if 0
			if(e1 / NUMBER_PACKING_EDGE_LISTS == 45510 && c_value == 22504 / NUMBER_PACKING_EDGE_LISTS
					&& (e1 % NUMBER_PACKING_EDGE_LISTS) == (22504 % NUMBER_PACKING_EDGE_LISTS)) {
				printf("===Debug Info: K:%d:r:%d Found pred==22504\n", __LINE__, mpig.rank_2d);
			}
#endif
				out_columns_ptr[i] = c_value;
				out_indices_ptr[i] = e1;
			}
		}
	}

	__syncthreads();

	out_columns_base += th_attack_out_offset;
	out_indices_base += th_attack_out_offset;

	// thread attack
	{
		// s_out_offset[0] is already initialized (before block attack)
		s_out_offset[tid + 1] = th_length;
		// move from register
		int th_r_offset = s_r_offset[tid];
#if 0
		int r_start = th_r_offset;
#endif

		__syncthreads();

		dev_prefix_sum<int, LOG_THREADS_PER_BLOCK>(s_out_offset, tid);
		// dev_prefix_sum include __syncthreads() at last of it

		__shared__ int s_control[THREADS_PER_BLOCK];

		int th_offset = s_out_offset[tid];
		int th_end = s_out_offset[tid + 1];
		const int th_total_length = s_out_offset[THREADS_PER_BLOCK];

		int progress = 0;
		while(progress < th_total_length) {
			int remain = th_total_length - progress;
			int progress_th_end = min(th_end, progress + THREADS_PER_BLOCK);
			for( ; th_offset < progress_th_end; ++th_offset) {
				s_control[th_offset - progress] = tid;
				s_r_offset[th_offset - progress] = th_r_offset++;
#if 0
			if((s_bitmap[tid] & (1 << (22504 % NUMBER_PACKING_EDGE_LISTS))) &&
					r_start == 657321) {
				printf("===Debug Info: K:%d:r:%d Hit r_start(%d),th_r_offset(%d),offset(%d),remain(%d)\n", __LINE__, mpig.rank_2d,
						r_start, th_r_offset, th_offset - progress, remain);
			}
#endif
			}

			__syncthreads();

			if(tid < remain) {
				const int winner = s_control[tid];
				const int r_offset = s_r_offset[tid];
#if 0
				if(winner < 0 || winner >= THREADS_PER_BLOCK) {
					printf("K:%d:B(%d)T(%d,%d)Winner(%d) is out of range\n", __LINE__, blockIdx.x, threadIdx.x, threadIdx.y, winner);
				}
#endif
				const int out_offset = progress + tid; // can be opt ?
				const uint32_t bitmap = s_bitmap[winner];
				const int column = s_columns[winner];

				const int32_t idx_high = index_array_high[r_offset];
				const uint16_t idx_low = index_array_low[r_offset];

				const int64_t e1 = ((int64_t)idx_high << 16) | idx_low;
				const int v0_lowbits = e1 % NUMBER_PACKING_EDGE_LISTS;
				int c_value = column;
				if((bitmap & (1U << v0_lowbits)) == 0) {
					c_value = -1;
				}
#if KD_PRINT
				else if(c_value < 0 || c_value >= empty_column) {
					printf("K:%d:B(%d)T(%d,%d)c_value(%d) is out of range\n", __LINE__, blockIdx.x, threadIdx.x, threadIdx.y, c_value);
				}
#endif
#if 0
			if((bitmap & (1 << (22504 % NUMBER_PACKING_EDGE_LISTS))) &&
					r_offset == 657344) {
				printf("===Debug Info: K:%d:r:%d Hit r_offset==657344\n", __LINE__, mpig.rank_2d);
			}
#endif
#if 0
			if(e1 / NUMBER_PACKING_EDGE_LISTS == 44000 && c_value == 7268 / NUMBER_PACKING_EDGE_LISTS
					&& (e1 % NUMBER_PACKING_EDGE_LISTS) == (7268 % NUMBER_PACKING_EDGE_LISTS)) {
				printf("===Debug Info: K:%d:r:%d Found pred==7268\n", __LINE__, mpig.rank_2d);
			}
#endif
#if 0
			if(r_offset == 657344) {
				printf("===Debug Info: K:%d:r:%d Found pred==7268\n", __LINE__, mpig.rank_2d);
			}
#endif
				out_columns_base[out_offset] = c_value;
				out_indices_base[out_offset] = e1;
			}

			__syncthreads();
			progress += THREADS_PER_BLOCK;
		}
	}

	const int column_offset = column_start + THREADS_PER_BLOCK * blockIdx.x + tid;
	columns[column_offset].x = (uint)empty_column;
}

// blockDim: dim3(WARP_SIZE, WARPS_PER_BLOCK)
__global__ void read_edges_and_filter(
	ReadGraphInput* in_param,
	const int* columns,
	const int64_t* indices,
	uint32_t* shared_visited,
	long2* out_edges)
{
	const int tid = threadIdx.y*WARP_SIZE + threadIdx.x;
	const uint mask = (1U << threadIdx.x) - 1U;
	const int num_in_edges = in_param->intermid_offset;
	const int i_start = THREADS_PER_BLOCK * blockIdx.x * LOOPS_PER_THREAD;
	const int i_end = min(i_start + THREADS_PER_BLOCK * LOOPS_PER_THREAD, num_in_edges);

	__shared__ int s_sum_buffer[WARPS_PER_BLOCK*2];
	int* s_sum = s_sum_buffer + WARPS_PER_BLOCK;
	int* s_offset = s_sum - 1;
#if KD_PRINT
			if(blockIdx.x == 0 && tid == 0) {
				printf("K:%d:num_in_edges=%d\n", __LINE__, num_in_edges);
			}
#endif

	for(int i = i_start; i < i_end; i += THREADS_PER_BLOCK) {
		int valid = 0;
		int64_t e1;
		int c_value;

		if(i + tid < i_end) {
			c_value = columns[i + tid];
			if(c_value != -1) {
				e1 = indices[i + tid];
				valid = read_graph_is_visited(shared_visited, e1);
			}
		}

		const uint valid_flags = __ballot(valid);
		s_sum[threadIdx.y] = __popc(valid_flags);

		__syncthreads();

		if(tid < WARPS_PER_BLOCK) dev_simd_kogge_stone<int, LOG_WARPS_PER_BLOCK>(s_sum, tid);

		// TODO: if s_offset[WARPS_PER_BLOCK] == 0 then continue!!

		__shared__ int out_base;
		if(tid == 0) { // The warp 0 does not need to call __synchthreads()
			int total_length = s_offset[WARPS_PER_BLOCK];
			out_base = atomicAdd(&in_param->output_offset, total_length);
#if KD_PRINT
			if(out_base == 0 && total_length != 0) printf("K:%d:output %d edges from %d.\n", __LINE__, total_length, out_base);
		//	printf("K:%d:output %d edges from %d.\n", __LINE__, total_length, out_base);
#endif
		}

		__syncthreads();

		if(valid) {
			const int64_t c1 = e1 / NUMBER_PACKING_EDGE_LISTS;
			const int64_t v0 = ((int64_t)c_value * NUMBER_PACKING_EDGE_LISTS) |
						 (e1 % NUMBER_PACKING_EDGE_LISTS);
			const int out_offset = out_base + s_offset[threadIdx.y] + __popc(mask & valid_flags);
#if KD_PRINT
			if(out_offset < 0 || out_offset >= out_base + s_offset[WARPS_PER_BLOCK]) {
				printf("K:%d:B(%d)T(%d,%d)Error:out_offset(%d),out_base(%d),warp_base(%d),my_offset(%d),total_length(%d),.\n",
						__LINE__, blockIdx.x, threadIdx.x, threadIdx.y, out_offset, out_base, s_offset[threadIdx.y], __popc(mask & valid_flags), s_offset[WARPS_PER_BLOCK]);
			}
#endif
#if 0
			if(v0 >= (1 << 16)) {
				printf("K:%d:Error v0(%"PRId64") is our of range, num_in_edges=%d, out_offset=%d.\n", __LINE__, v0, num_in_edges, out_offset);
			}
#endif
#if 0
			printf("K:%d:out_edges[%d] = make_long2(%"PRId64", %"PRId64").\n", __LINE__, out_offset, v0, c1);
#endif
#if 0
			if(c1 == 115504 && v0 == 253941) {
				printf("===Debug Info: K:%d:r:%d Found pred==7268\n", __LINE__, mpig.rank_2d);
			}
#endif
			out_edges[out_offset] = make_long2(v0, c1);
		}

		__syncthreads();
	}
}

void read_graph_1(
	ReadGraphInput* in_param,
	const int64_t* row_starts,
	const int32_t* index_array_high,
	const uint16_t* index_array_low,
	uint2* columns,
	const int column_start, const int column_end,
	const int empty_column,
	uint32_t* shared_visited,
	int* out_columns,
	int64_t* out_indices,
	long2* out_edges,
	cudaStream_t stream)
{
	initialize_read_graph_input<<<1, 1, 0, stream>>>(in_param);

	int num_ctas = nblocks2n<int>(column_end - column_start, THREADS_PER_BLOCK);
	read_graph_and_store<<<num_ctas, dim3(WARP_SIZE, WARPS_PER_BLOCK), 0, stream>>>
		(in_param, row_starts, index_array_high, index_array_low, columns, column_start, column_end,
				empty_column, out_columns, out_indices);

}

void read_graph_2(
	ReadGraphInput* in_param,
	const int64_t* row_starts,
	const int32_t* index_array_high,
	const uint16_t* index_array_low,
	uint2* columns,
	const int column_start, const int column_end,
	const int empty_column,
	uint32_t* shared_visited,
	int* out_columns,
	int64_t* out_indices,
	long2* out_edges,
	cudaStream_t stream)
{
	int num_ctas = nblocks2n<int>(READ_GRAPH_OUTBUF_SIZE, THREADS_PER_BLOCK*LOOPS_PER_THREAD);
	read_edges_and_filter<<<num_ctas, dim3(WARP_SIZE, WARPS_PER_BLOCK), 0, stream>>>
		(in_param, out_columns, out_indices, shared_visited, out_edges);

}

void filter_edges(
	ReadGraphInput* in_param,
	int num_edges,
	const int* in_columns,
	const int64_t* in_indices,
	uint32_t* shared_visited,
	long2* out_edges,
	cudaStream_t stream)
{
	initialize_filter_edges_input<<<1, 1, 0, stream>>>(in_param, num_edges);

	int num_ctas = nblocks2n<int>(num_edges, THREADS_PER_BLOCK*LOOPS_PER_THREAD);
	read_edges_and_filter<<<num_ctas, dim3(WARP_SIZE, WARPS_PER_BLOCK), 0, stream>>>
		(in_param, in_columns, in_indices, shared_visited, out_edges);
}

template<typename T>
__global__ void decode_varint_kernel(
	const int4* packet_list, // (stream_offset, num_int, v_offset, *)
	const uint8_t* stream,
	const int num_packet,
	T* output)
{
	__shared__ int s_offset_buffer[WARPS_PER_BLOCK];
	volatile int* s_offset = &s_offset_buffer[threadIdx.y];

	if(WARP_IDX >= num_packet) return;

	const int4& packet = packet_list[WARP_IDX];
	const int stream_offset = packet.x;
	const int num_int = packet.y;
	const int v_offset = packet.z;

	decode_varint_packet(stream + stream_offset, num_int, output + v_offset, *s_offset, threadIdx.x);
}

template<typename T>
__global__ void decode_varint_signed_kernel(
	const int4* packet_list, // (stream_offset, num_int, v_offset, *)
	const uint8_t* stream,
	const int num_packet,
	T* output)
{
	__shared__ int s_offset_buffer[WARPS_PER_BLOCK];
	volatile int* s_offset = &s_offset_buffer[threadIdx.y];

	if(WARP_IDX >= num_packet) return;

	const int4& packet = packet_list[WARP_IDX];
	const int stream_offset = packet.x;
	const int num_int = packet.y;
	const int v_offset = packet.z;

	decode_varint_packet_signed(stream + stream_offset, num_int, output + v_offset, *s_offset, threadIdx.x);
}

template<typename T>
void decode_varint_stream(
	const int4* packet_list, // (stream_offset, num_int, v_offset, *)
	const uint8_t* byte_stream,
	const int num_packet,
	T* output,
	cudaStream_t stream)
{
	int num_ctas = nblocks2n<int>(num_packet, WARPS_PER_BLOCK);
	if(num_ctas == 0) return ;
	decode_varint_kernel<T> <<<num_ctas, dim3(WARP_SIZE, WARPS_PER_BLOCK), 0, stream>>>
			(packet_list, byte_stream, num_packet, output);
}

template<typename T>
void decode_varint_stream_signed(
	const int4* packet_list, // (stream_offset, num_int, v_offset, *)
	const uint8_t* byte_stream,
	const int num_packet,
	T* output,
	cudaStream_t stream)
{
	int num_ctas = nblocks2n<int>(num_packet, WARPS_PER_BLOCK);
	if(num_ctas == 0) return ;
	decode_varint_signed_kernel<T> <<<num_ctas, dim3(WARP_SIZE, WARPS_PER_BLOCK), 0, stream>>>
			(packet_list, byte_stream, num_packet, output);
}

template void decode_varint_stream_signed<int32_t>(
		const int4* packet_list, // (stream_offset, num_int, v_offset, *)
		const uint8_t* byte_stream,
		const int num_packet,
		int32_t* output,
		cudaStream_t stream);
template void decode_varint_stream_signed<int64_t>(
		const int4* packet_list, // (stream_offset, num_int, v_offset, *)
		const uint8_t* byte_stream,
		const int num_packet,
		int64_t* output,
		cudaStream_t stream);

// blockDim: (PACKET_LENGTH/2, THREADS_PER_BLOCK*2/PACKET_LENGTH)
__global__ void receiver_processing_kernel(
	const int4* const input_packets, // (stream_offset, num_edges, v1_offset, stream_length)
	const int num_packets,
	const int64_t* const v0_list,
	const uint32_t* const v1_list,
	uint32_t* const nq_bitmap,
	uint32_t* const nq_sorted_bitmap,
	uint32_t* const visited,
	int64_t* const pred,
	const uint32_t* const v1_map,

	// for parent vertex conversion
	const int log_local_verts,
	const int64_t log_size,
	const int64_t local_verts_mask,
	const int current_level,

	int* const nq_count_ptr)
{
	enum {
		HARF_PACKET = BFS_PARAMS::PACKET_LENGTH / 2,
	};
	__shared__ int64_t s_v0_buffer[THREADS_PER_BLOCK*4];
	int64_t* s_v0 = &s_v0_buffer[BFS_PARAMS::PACKET_LENGTH*2*threadIdx.y];
	int count = 0;

	const int warp_idx = WARP_IDX;
	const int4& packet = input_packets[warp_idx];
	int num_edges;
	int v1_offset;

	if(warp_idx < num_packets) {
		num_edges = packet.y;
		v1_offset = packet.z;
	}
	else {
		num_edges = 0;
		v1_offset = 0;
	}

	const int64_t* v0_start = v0_list + v1_offset;
	const int tidx = threadIdx.x;

#if 0
	if(tidx == 0) printf("K:%d:receive %d edges from %d.\n", __LINE__, num_edges, v1_offset);
#endif
	if(HARF_PACKET*0 + tidx < num_edges) s_v0[HARF_PACKET*0 + tidx] = v0_start[HARF_PACKET*0 + tidx];
	if(HARF_PACKET*1 + tidx < num_edges) s_v0[HARF_PACKET*1 + tidx] = v0_start[HARF_PACKET*1 + tidx];

	__syncthreads();

#if KD_PRINT
			if(HARF_PACKET*0 + tidx < num_edges && s_v0[tidx] >= (int64_t(1) << 28)) {
				printf("K:%d:R(%d)B(%d)T(%d,%d)Error:  s_v0[tidx] (%"PRId64") is too big!!!\n", __LINE__, mpig.rank_2d, blockIdx.x, threadIdx.x, threadIdx.y,
						s_v0[tidx]);
			}
			if(HARF_PACKET*1 + tidx < num_edges && s_v0[tidx + HARF_PACKET] >= (int64_t(1) << 28)) {
				printf("K:%d:R(%d)B(%d)T(%d,%d)Error:  s_v0[tidx + HARF_PACKET] (%"PRId64") is too big!!!\n", __LINE__, mpig.rank_2d, blockIdx.x, threadIdx.x, threadIdx.y,
						s_v0[tidx + HARF_PACKET]);
			}
#endif
	dev_prefix_sum<int64_t, BFS_PARAMS::LOG_PACKET_LENGTH>(s_v0, tidx);
	// dev_prefix_sum include __syncthreads() at last of it

	for(int k = tidx; k < num_edges; k += HARF_PACKET) {
		const uint32_t v1_local = v1_list[v1_offset + k];
		const int word_idx = v1_local / NUMBER_PACKING_EDGE_LISTS;
		const int bit_idx = v1_local % NUMBER_PACKING_EDGE_LISTS;
		const uint32_t mask = uint32_t(1) << bit_idx;

#if 0
			if(v1_local == 30830 && mpig.rank_2d == 3) {
				printf("Debug Info: r:%d Found v1_local == 30830\n", mpig.rank_2d);
			}
#endif
		// Is this technique effective on GPU ???
		if((visited[word_idx] & mask) == 0) {
			if((atomicOr(&visited[word_idx], mask) & mask) == 0) {
				atomicOr(&nq_sorted_bitmap[word_idx], mask);

				const uint32_t v1_orig = v1_map[v1_local];
				const int64_t v0_swizzled = s_v0[k];
				pred[v1_orig] = (v0_swizzled >> log_local_verts) |
								((v0_swizzled & local_verts_mask) << log_size) |
								(int64_t(current_level) << 48);

				const int word_idx = v1_orig / NUMBER_PACKING_EDGE_LISTS;
				const int bit_idx = v1_orig % NUMBER_PACKING_EDGE_LISTS;
				const uint32_t mask = uint32_t(1) << bit_idx;

				atomicOr(&nq_bitmap[word_idx], mask);

#if 0
				printf("K:%d:new vertices(%d:orig) found (%d:sorted).\n", __LINE__, v1_orig, v1_local);
#endif
#if KD_PRINT
			if((pred[v1_orig] & 0xFFFFFFFFFFFF) >= (int64_t(1) << 28)) {
				printf("Error: r:%d pred[v1_orig] (%"PRId64") is too big!!!\n", mpig.rank_2d, (pred[v1_orig] & 0xFFFFFFFFFFFF));
				printf("K:%d:B(%d)T(%d,%d):DumpInfo:v0_swizzled(%"PRId64"),log_local_verts(%d),local_verts_mask(%"PRId64"),log_size(%d),current_level(%d)\n", __LINE__, blockIdx.x, threadIdx.x, threadIdx.y,
						v0_swizzled, log_local_verts, local_verts_mask, log_size, current_level);
			}
#endif
#if 0
			if(v1_orig == 93982 && (pred[v1_orig] & 0xFFFFFFFF) == 253941) {
				printf("===Debug Info: K:%d:r:%d Found pred==253941\n", __LINE__, mpig.rank_2d);
			}
#endif
				++count;
			}
		}
	}

	__shared__ int s_count[THREADS_PER_BLOCK];
	const int tid = threadIdx.y * blockDim.x + threadIdx.x;
	s_count[tid] = count;
	__syncthreads();
	dev_reduce<int, THREADS_PER_BLOCK>(THREADS_PER_BLOCK, s_count, tid);
	if(tid == 0) {
		atomicAdd(nq_count_ptr, s_count[0]);
	}
}

// num_packet: multiply of THREADS_PER_BLOCK*2/PACKET_LENGTH
void receiver_processing(
	const int4* input_packets, // (stream_offset, num_edges, v1_offset, stream_length)
	int num_packets,
	const int64_t* v0_list,
	const uint32_t* v1_list,
	uint32_t* nq_bitmap,
	uint32_t* nq_sorted_bitmap,
	uint32_t* visited,
	int64_t* pred,
	const uint32_t* v1_map,
	int log_local_verts,
	int64_t log_size,
	int64_t local_verts_mask,
	int current_level,
	int* nq_count_ptr,
	cudaStream_t stream)
{
	const dim3 blkDim(BFS_PARAMS::PACKET_LENGTH/2, THREADS_PER_BLOCK*2/BFS_PARAMS::PACKET_LENGTH);
	assert (blkDim.x * blkDim.y == THREADS_PER_BLOCK);
	assert (blkDim.x * 2 == BFS_PARAMS::PACKET_LENGTH);
	int num_ctas = nblocks2n<int>(num_packets, blkDim.y);
	if(num_ctas == 0) return ;
	receiver_processing_kernel<<<num_ctas, blkDim, 0, stream>>>
		(input_packets, num_packets, v0_list, v1_list, nq_bitmap, nq_sorted_bitmap,
		 visited, pred, v1_map, log_local_verts, log_size, local_verts_mask, current_level, nq_count_ptr);
}

void cu_clear_nq(
	int bitmap_size_visited,
	uint32_t* nq_bitmap,
	uint32_t* nq_sorted_bitmap,
	cudaStream_t* streams,
	int num_streams)
{
	memset_gpu<uint32_t>(nq_bitmap, 0, bitmap_size_visited, streams[0 % num_streams]);
	memset_gpu<uint32_t>(nq_sorted_bitmap, 0, bitmap_size_visited, streams[1 % num_streams]);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void update_bitmap_kernel_with_summary(
	const int4* input_packets, // (stream_offset, num_vertices, v_offset, src_num)
	int num_packets,
	const uint32_t* v_list,
	uint32_t* bitmap,
	uint32_t* summary,
	int log_src_factor)
{
	enum {
		HARF_PACKET = BFS_PARAMS::PACKET_LENGTH / 2,
	};
	__shared__ uint32_t s_v0_buffer[THREADS_PER_BLOCK*4];
	uint32_t* s_v0 = &s_v0_buffer[BFS_PARAMS::PACKET_LENGTH*2*threadIdx.y];

	const int warp_idx = WARP_IDX;
	const int4& packet = input_packets[warp_idx];
	int num_vertices;
	int v_offset;
	int src_num;

	if(warp_idx < num_packets) {
		num_vertices = packet.y;
		v_offset = packet.z;
		src_num = packet.w;
	}
	else {
		num_vertices = 0;
		v_offset = 0;
		src_num = 0;
	}

	const uint32_t* v_start = v_list + v_offset;
	const int tidx = threadIdx.x;
	const uint32_t high_bit = uint32_t(src_num) << log_src_factor;

	if(HARF_PACKET*0 + tidx < num_vertices) s_v0[HARF_PACKET*0 + tidx] = v_start[HARF_PACKET*0 + tidx];
	if(HARF_PACKET*1 + tidx < num_vertices) s_v0[HARF_PACKET*1 + tidx] = v_start[HARF_PACKET*1 + tidx];

#if 0
			if(HARF_PACKET*0 + tidx < num_vertices && s_v0[tidx] >= (1 << 26)) {
				printf("K:%d:R(%d)B(%d)T(%d,%d)Error:  s_v0[tidx] (%"PRId64") is too big!!!\n", __LINE__, mpig.rank_2d, blockIdx.x, threadIdx.x, threadIdx.y,
						s_v0[tidx]);
			}
			if(HARF_PACKET*1 + tidx < num_vertices && s_v0[tidx + HARF_PACKET] >= (1 << 26)) {
				printf("K:%d:R(%d)B(%d)T(%d,%d)Error:  s_v0[tidx + HARF_PACKET] (%"PRId64") is too big!!!\n", __LINE__, mpig.rank_2d, blockIdx.x, threadIdx.x, threadIdx.y,
						s_v0[tidx + HARF_PACKET]);
			}
#endif
	__syncthreads();
#if 0
	if(tid == 0) printf("K:%d:B(%d)T(%d,%d)num_vertices=%d,vertex[0]=%d,src_num=%d\n", __LINE__, blockIdx.x, threadIdx.x, threadIdx.y, num_vertices,s_v0[0], src_num);
#endif
	dev_prefix_sum<uint32_t, BFS_PARAMS::LOG_PACKET_LENGTH>(s_v0, tidx);
	// dev_prefix_sum include __syncthreads() at last of it

	for(int k = tidx; k < num_vertices; k += HARF_PACKET) {
		const uint32_t v = s_v0[k] | high_bit;
		const int word_idx = v / NUMBER_PACKING_EDGE_LISTS;
		const int bit_idx = v % NUMBER_PACKING_EDGE_LISTS;
		const uint32_t mask = uint32_t(1) << bit_idx;
#if 0
		if(word_idx < 0 || word_idx >= (1 << (16 - LOG_PACKING_EDGE_LISTS))) {
			printf("K:%d:B(%d)T(%d,%d)Error: Out of bound. word_idx=%d\n", __LINE__, blockIdx.x, threadIdx.x, threadIdx.y, word_idx);
		}
#endif
		// Is this technique effective on GPU ???
		if((bitmap[word_idx] & mask) == 0) {
			if((atomicOr(&bitmap[word_idx], mask) & mask) == 0) {
				const int bit_offset = word_idx / NUMBER_CQ_SUMMARIZING;
				const int summary_word_idx = bit_offset / (sizeof(summary[0]) * 8);
				const int summary_bit_idx = bit_offset % (sizeof(summary[0]) * 8);
				uint32_t summary_mask = uint32_t(1) << summary_bit_idx;

#if 0
	printf("K:%d:B(%d)T(%d,%d)update bitmap: word_idx=%d\n", __LINE__, blockIdx.x, threadIdx.x, threadIdx.y, word_idx);
#endif
				// Is this technique effective on GPU ???
				if((summary[summary_word_idx] & summary_mask) == 0) {
					atomicOr(&summary[summary_word_idx], summary_mask);
#if 0
	printf("K:%d:B(%d)T(%d,%d)update summary: summary_word_idx=%d\n", __LINE__, blockIdx.x, threadIdx.x, threadIdx.y, summary_word_idx);
#endif
				}
			}
		}
	}
}

// num_packet: multiply of THREADS_PER_BLOCK*2/PACKET_LENGTH
__global__ void update_bitmap_kernel(
	const int4* input_packets, // (stream_offset, num_vertices, v_offset, src_num)
	int num_packets,
	const uint32_t* v_list,
	uint32_t* bitmap,
	int log_src_factor)
{
	enum {
		HARF_PACKET = BFS_PARAMS::PACKET_LENGTH / 2,
	};
	__shared__ uint32_t s_v0_buffer[THREADS_PER_BLOCK*4];
	uint32_t* s_v0 = &s_v0_buffer[BFS_PARAMS::PACKET_LENGTH*2*threadIdx.y];

	const int warp_idx = WARP_IDX;
	const int4& packet = input_packets[warp_idx];
	int num_vertices;
	int v_offset;
	int src_num;

	if(warp_idx < num_packets) {
		num_vertices = packet.y;
		v_offset = packet.z;
		src_num = packet.w;
	}
	else {
		num_vertices = 0;
		v_offset = 0;
		src_num = 0;
	}

	const uint32_t* v_start = v_list + v_offset;
	const int tidx = threadIdx.x;
	const uint32_t high_bit = uint32_t(src_num) << log_src_factor;

	if(HARF_PACKET*0 + tidx < num_vertices) s_v0[HARF_PACKET*0 + tidx] = v_start[HARF_PACKET*0 + tidx];
	if(HARF_PACKET*1 + tidx < num_vertices) s_v0[HARF_PACKET*1 + tidx] = v_start[HARF_PACKET*1 + tidx];

	__syncthreads();

	dev_prefix_sum<uint32_t, BFS_PARAMS::LOG_PACKET_LENGTH>(s_v0, tidx);
	// dev_prefix_sum include __syncthreads() at last of it

	for(int k = tidx; k < num_vertices; k += HARF_PACKET) {
		const uint32_t v = s_v0[k] | high_bit;
		const int word_idx = v / NUMBER_PACKING_EDGE_LISTS;
		const int bit_idx = v % NUMBER_PACKING_EDGE_LISTS;
		const uint32_t mask = uint32_t(1) << bit_idx;

		// Is this technique effective on GPU ???
		if((bitmap[word_idx] & mask) == 0) {
			atomicOr(&bitmap[word_idx], mask);
		}
	}
}

// num_packet: multiply of THREADS_PER_BLOCK*2/PACKET_LENGTH
void update_bitmap(
	const int4* input_packets, // (stream_offset, num_vertices, src_num, stream_length)
	int num_packets,
	const uint32_t* v_list,
	uint32_t* bitmap,
	uint32_t* summary,
	int log_src_factor,
	cudaStream_t stream)
{
	const dim3 blkDim(BFS_PARAMS::PACKET_LENGTH/2, THREADS_PER_BLOCK*2/BFS_PARAMS::PACKET_LENGTH);
	assert (blkDim.x * blkDim.y == THREADS_PER_BLOCK);
	assert (blkDim.x * 2 == BFS_PARAMS::PACKET_LENGTH);
	int num_ctas = nblocks2n<int>(num_packets, blkDim.y);
	if(num_ctas == 0) return ;
	if(summary) {
		update_bitmap_kernel_with_summary<<<num_ctas, blkDim, 0, stream>>>
			(input_packets, num_packets, v_list, bitmap, summary, log_src_factor);
	}
	else {
		update_bitmap_kernel<<<num_ctas, blkDim, 0, stream>>>
			(input_packets, num_packets, v_list, bitmap, log_src_factor);

	}
}

void fill_1_summary(
	uint32_t* cq_summary,
	int summary_size)
{
	memset_gpu<uint32_t>(cq_summary, uint32_t(-1), summary_size, NULL);
}

} // namespace cuda {

#endif /* BFS_KERNEL_HPP_ */
