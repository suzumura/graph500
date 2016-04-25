/*
 * gpu_core.h
 *
 *  Created on: Feb 29, 2012
 *      Author: koji
 */

#ifndef GPU_CORE_H_
#define GPU_CORE_H_

#include <stdint.h>

#include "parameters.h"

#define PREFIX_SUM_OPT 1

//-------------------------------------------------------------//
// Prefix Sum
//-------------------------------------------------------------//

template <typename T>
class PrefixSumGPU
{
#if PREFIX_SUM_OPT
	enum {
		THRESHOLD = 12, // 4096
		LOG_MAX = 10,
		MAX = 1 << LOG_MAX,
		CTA_DIM_X = MAX / 2,
		LOG_MAX_ACTIVE_CTAS = 7,
		MAX_ACTIVE_CTAS = 1 << LOG_MAX_ACTIVE_CTAS,
		PADDING = (sizeof(T) <= 4) ? 32
				: (sizeof(T) <= 8) ? 16
				: 					  8,
	};
#else
	enum {
		THRESHOLD1 = 11,
		THRESHOLD2 = 14,
		LOG_MAX9 = 9,
		LOG_MAX10 = 10,
		MAX9 = 1 << LOG_MAX9,
		MAX10 = 1 << LOG_MAX10,
		PADDING = (sizeof(T) <= 4) ? 32
				: (sizeof(T) <= 8) ? 16
				: 					  8,
	};
#endif
public:
	PrefixSumGPU(int64_t max_length, int num_partitions = 1);
	~PrefixSumGPU();

	T* get_buffer(int partition_index = 0)
	{
#if PREFIX_SUM_OPT
		return dev_array_ + partition_index * (max_length_ + PADDING + MAX_ACTIVE_CTAS);
#else
		return dev_array_ + partition_index * (max_length_ + PADDING);
#endif
	}

	void operator()(cudaStream_t* streams = NULL, int num_streams = 0);

private:
#if !PREFIX_SUM_OPT
	void process_sum_block(int nblocks, int stride, cudaStream_t* streams, int num_streams);
#endif

#if PREFIX_SUM_OPT
	int num_launch_ctas_;
#endif
	int64_t max_length_;
	int log_max_length_;
	int num_partitions_;
	T* dev_array_;
};

namespace cuda {

/////////////////

struct ReadGraphInput
{
	int reserve_offset;
	int intermid_offset;
	int output_offset;
	bool skipped;
};

struct EdgeIOBuffer {
	ReadGraphInput input;
	int columns[GPU_PARAMS::READ_GRAPH_OUTBUF_SIZE];
	int64_t indices[GPU_PARAMS::READ_GRAPH_OUTBUF_SIZE];
};

struct ReadGraphBuffer {
	ReadGraphInput input;
	long2 edges[GPU_PARAMS::READ_GRAPH_OUTBUF_SIZE]; // 64MB
};

struct RecvProcBuffer {
	int4 packet_list[GPU_PARAMS::GPU_BLOCK_MAX_PACKTES];
	uint8_t v0_stream[GPU_PARAMS::GPU_BLOCK_V0_LEGNTH]; // 6MB
	uint32_t v1_list[GPU_PARAMS::GPU_BLOCK_V1_LENGTH]; // 12MB
};

// 64MB
union FoldIOBuffer
{
	ReadGraphBuffer read_graph; // for graph on GPU
	EdgeIOBuffer edge_io; // for semi-GPU computation
	RecvProcBuffer recv_proc;
};

// 48MB
union FoldGpuBuffer
{
	struct {
		int out_columns[GPU_PARAMS::READ_GRAPH_OUTBUF_SIZE]; // 16MB
		int64_t out_indices[GPU_PARAMS::READ_GRAPH_OUTBUF_SIZE]; // 32MB
	} read_graph;
	struct {
		int64_t v0_list[GPU_PARAMS::GPU_BLOCK_V1_LENGTH]; // 24MB
	} recv_proc;
};

// about 100MB
struct UpdateProcBuffer
{
	int4 packet_list[GPU_PARAMS::EXPAND_PACKET_LIST_LENGTH];
	uint8_t v_stream[GPU_PARAMS::EXPAND_STREAM_BLOCK_LENGTH];
	uint32_t v_list[GPU_PARAMS::EXPAND_DECODE_BLOCK_LENGTH];
};

struct BfsGPUContext {
	int num_non_zero_columns;
	int nq_count;
	int64_t root;
	uint32_t mask;
};

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
	int num_streams);

void create_cq_list(
	uint32_t* bitmap,
	uint32_t* summary,
	int summary_size,
	PrefixSumGPU<int>* cq_count,
	uint2* column_buffer,
	int empty_column,
	cudaStream_t stream);

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
	cudaStream_t stream);

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
	cudaStream_t stream);

void filter_edges(
	ReadGraphInput* in_param,
	int num_edges,
	const int* in_columns,
	const int64_t* in_indices,
	uint32_t* shared_visited,
	long2* out_edges,
	cudaStream_t stream);

#if 0
void receiver_processing(
	const int4* input_packets, // (stream_offset, num_edges, v1_offset, stream_length)
	int num_packets,
	const uint8_t* v0_stream,
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
	RECV_PROC_TEMP_BUFFER* tmp_buffer,
	cudaStream_t stream);
#endif
void cu_clear_nq(
	int bitmap_size_visited,
	uint32_t* nq_bitmap,
	uint32_t* nq_sorted_bitmap,
	cudaStream_t* streams,
	int num_streams);

void update_bitmap(
	const int4* input_packets, // (stream_offset, num_vertices, src_num, stream_length)
	int num_packets,
	const uint32_t* v_list,
	uint32_t* bitmap,
	uint32_t* summary,
	int log_src_factor,
	cudaStream_t stream);

void fill_1_summary(
	uint32_t* cq_summary,
	int summary_size);

// simplified version

template<typename T>
void decode_varint_stream(
	const int4* packet_list, // (stream_offset, num_int, v_offset, *)
	const uint8_t* byte_stream,
	const int num_packet,
	T* output,
	cudaStream_t stream);

template<typename T>
void decode_varint_stream_signed(
	const int4* packet_list, // (stream_offset, num_int, v_offset, *)
	const uint8_t* byte_stream,
	const int num_packet,
	T* output,
	cudaStream_t stream);

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
	cudaStream_t stream);

} // namespace cuda {

#endif /* GPU_CORE_H_ */
