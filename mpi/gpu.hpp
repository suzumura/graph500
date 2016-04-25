/*
 * gpu.hpp
 *
 *  Created on: Feb 29, 2012
 *      Author: koji
 */

#ifndef GPU_HPP_
#define GPU_HPP_

#include "gpu_host.h"
#include "gpu_core.h"

using namespace BFS_PARAMS;
using namespace GPU_PARAMS;
using namespace cuda;

typedef int gsize_t;

template <typename T>
inline T nblocks2n(T value, T unit)
{
	return (value + unit - 1) / unit;
}

/*
    --- Parallel Reduction ---
	This function compute sum of sdata and store result to sdata[0].
	The data of sdata is destroyed when return.
	n : length of sdata
	sdata : [shared memory] target data
	requirements
	blockDim.x >= n
	max >= 32 , max = pow(2, r) (r : integer)
	max/2 <= n <= max
	All threads of the block must call this function if max >= 64.

	approach : parallel reduction from CUDA sample
*/
template <typename T, int max>
__device__ void dev_reduce(int n, volatile T* sdata)
{
	int right;
	if( max >= 64 ){
		right = threadIdx.x + max/2;
		if( right < n ){
			sdata[threadIdx.x] += sdata[right];
		}
	}
	if( max >= 128 ){
		__syncthreads();
	}
	if( max >= 512 ){
		if( threadIdx.x < 128 ) sdata[threadIdx.x] += sdata[threadIdx.x + 128];
		__syncthreads();
	}
	if( max >= 256 ){
		if( threadIdx.x < 64 ) sdata[threadIdx.x] += sdata[threadIdx.x + 64];
		__syncthreads();
	}
	if( max >= 128 ){
		if( threadIdx.x < 32 ) sdata[threadIdx.x] += sdata[threadIdx.x + 32];
	}
	// if WARP_SIZE != 32, rewrite here!
	if (threadIdx.x < 16) {
		sdata[threadIdx.x] += sdata[threadIdx.x + 16];
		sdata[threadIdx.x] += sdata[threadIdx.x + 8];
		sdata[threadIdx.x] += sdata[threadIdx.x + 4];
		sdata[threadIdx.x] += sdata[threadIdx.x + 2];
		sdata[threadIdx.x] += sdata[threadIdx.x + 1];
	}
}

template <typename T, int max>
__device__ void dev_reduce(int n, volatile T* sdata, int tid)
{
	int right;
	if( max >= 64 ){
		right = tid + max/2;
		if( right < n ){
			sdata[tid] += sdata[right];
		}
	}
	if( max >= 128 ){
		__syncthreads();
	}
	if( max >= 512 ){
		if( tid < 128 ) sdata[tid] += sdata[tid + 128];
		__syncthreads();
	}
	if( max >= 256 ){
		if( tid < 64 ) sdata[tid] += sdata[tid + 64];
		__syncthreads();
	}
	if( max >= 128 ){
		if( tid < 32 ) sdata[tid] += sdata[tid + 32];
	}
	// if WARP_SIZE != 32, rewrite here!
	if (tid < 16) {
		sdata[tid] += sdata[tid + 16];
		sdata[tid] += sdata[tid + 8];
		sdata[tid] += sdata[tid + 4];
		sdata[tid] += sdata[tid + 2];
		sdata[tid] += sdata[tid + 1];
	}
}

#include "prefix_sum.hpp"

/*
    --- Parallel VarInt Encoding ---
    These functions encode or decode integers
    using VarInt Encoding (Variable length Integer Encoding).
    Requirements:
    Threads: A Warp. (tid: 0...31)
*/

// 64 bit version //

__device__ int encode_varint_packet(
	const uint64_t* encode_data,	// data to be encoded
	int num_int,					// the number of integer values to be encoded
	uint8_t* encode_buffer,			// a buffer encoded data will be stored
	volatile int& s_offset,			// [shared memory] buffer used in this function
	int tid)						// thread identifier (require 0...31 thread)
{
	typedef uint64_t T;
	const uint mask = (1U << threadIdx.x) - 1U;
	int offset = 0;
	s_offset = 0;
	for(int i = tid; i < num_int; i += WARP_SIZE) {
		int flags, t_offset;
		T data = encode_data[i];
		int w_offset = offset + tid;
		flags = __ballot(1);
		offset += __popc(flags);
		s_offset = offset;
		if(data >= 128) {
			encode_buffer[w_offset] = data | 0x80;
			flags = __ballot(1);
			t_offset = __popc(mask & flags);
			w_offset = offset + t_offset;
			offset += __popc(flags);
			s_offset = offset;
			data >>= 7;
			if(data >= 128) {
				encode_buffer[w_offset] = data | 0x80;
				flags = __ballot(1);
				t_offset = __popc(mask & flags);
				w_offset = offset + t_offset;
				offset += __popc(flags);
				s_offset = offset;
				data >>= 7;
				if(data >= 128) {
					encode_buffer[w_offset] = data | 0x80;
					flags = __ballot(1);
					t_offset = __popc(mask & flags);
					w_offset = offset + t_offset;
					offset += __popc(flags);
					s_offset = offset;
					data >>= 7;
					if(data >= 128) {
						encode_buffer[w_offset] = data | 0x80;
						flags = __ballot(1);
						t_offset = __popc(mask & flags);
						w_offset = offset + t_offset;
						offset += __popc(flags);
						s_offset = offset;
						data >>= 7;
						if(data >= 128) {
							encode_buffer[w_offset] = data | 0x80;
							flags = __ballot(1);
							t_offset = __popc(mask & flags);
							w_offset = offset + t_offset;
							offset += __popc(flags);
							s_offset = offset;
							data >>= 7;
							if(data >= 128) {
								encode_buffer[w_offset] = data | 0x80;
								flags = __ballot(1);
								t_offset = __popc(mask & flags);
								w_offset = offset + t_offset;
								offset += __popc(flags);
								s_offset = offset;
								data >>= 7;
							}
						}
					}
				}
			}
		}
		encode_buffer[w_offset] = data;
		if(offset < s_offset)
			offset = s_offset;
	}
	offset = s_offset;
	return offset;
}

__device__ int decode_varint_packet(
	const uint8_t* packet_start,	// data to be decoded
	int num_int,					// the number of integer values to be decoded
	uint64_t* decode_buffer,		// a buffer decoded data will be stored
	volatile int& s_offset,			// [shared memory] buffer used in thid function
	int tid)						// thread identifier (require 0...31 thread)
{
	typedef uint64_t T;
	const uint mask = (1U << threadIdx.x) - 1U;
	int offset = 0;
	s_offset = 0;
	for(int i = tid; i < num_int; i += WARP_SIZE) {
		int flags, t_offset;
		uint8_t v = packet_start[offset + tid];
		T decode_v = T(v) & 0x7F;
		flags = __ballot(1);
		offset += __popc(flags);
		s_offset = offset;
		if(v >= 128) {
			flags = __ballot(1);
			t_offset = __popc(mask & flags);
			v = packet_start[offset + t_offset];
			decode_v |= (T(v) & 0x7F) << 7;
			offset += __popc(flags);
			s_offset = offset;
			if(v >= 128) {
				flags = __ballot(1);
				t_offset = __popc(mask & flags);
				v = packet_start[offset + t_offset];
				decode_v |= (T(v) & 0x7F) << 14;
				offset += __popc(flags);
				s_offset = offset;
				if(v >= 128) {
					flags = __ballot(1);
					t_offset = __popc(mask & flags);
					v = packet_start[offset + t_offset];
					decode_v |= (T(v) & 0x7F) << 21;
					offset += __popc(flags);
					s_offset = offset;
					if(v >= 128) {
						flags = __ballot(1);
						t_offset = __popc(mask & flags);
						v = packet_start[offset + t_offset];
						decode_v |= (T(v) & 0x7F) << 28;
						offset += __popc(flags);
						s_offset = offset;
						if(v >= 128) {
							flags = __ballot(1);
							t_offset = __popc(mask & flags);
							v = packet_start[offset + t_offset];
							decode_v |= (T(v) & 0x7F) << 35;
							offset += __popc(flags);
							s_offset = offset;
							if(v >= 128) {
								flags = __ballot(1);
								t_offset = __popc(mask & flags);
								v = packet_start[offset + t_offset];
								decode_v |= (T(v) & 0x7F) << 49;
								offset += __popc(flags);
								s_offset = offset;
								if(v >= 128) {
									flags = __ballot(1);
									t_offset = __popc(mask & flags);
									v = packet_start[offset + t_offset];
									decode_v |= (T(v) & 0x7F) << 56;
									offset += __popc(flags);
									s_offset = offset;
								}
							}
						}
					}
				}
			}
		}
		decode_buffer[i] = decode_v;
		if(offset < s_offset)
			offset = s_offset;
	}
	offset = s_offset;
	return offset;
}

__device__ int decode_varint_packet_signed(
	const uint8_t* packet_start,	// data to be decoded
	int num_int,					// the number of integer values to be decoded
	int64_t* decode_buffer,		// a buffer decoded data will be stored
	volatile int& s_offset,			// [shared memory] buffer used in thid function
	int tid)						// thread identifier (require 0...31 thread)
{
	typedef int64_t T;
	const uint mask = (1U << threadIdx.x) - 1U;
	int offset = 0;
	s_offset = 0;
	for(int i = tid; i < num_int; i += WARP_SIZE) {
		int flags, t_offset;
		uint8_t v = packet_start[offset + tid];
		T decode_v = T(v) & 0x7F;
		flags = __ballot(1);
		offset += __popc(flags);
		s_offset = offset;
		if(v >= 128) {
			flags = __ballot(1);
			t_offset = __popc(mask & flags);
			v = packet_start[offset + t_offset];
			decode_v |= (T(v) & 0x7F) << 7;
			offset += __popc(flags);
			s_offset = offset;
			if(v >= 128) {
				flags = __ballot(1);
				t_offset = __popc(mask & flags);
				v = packet_start[offset + t_offset];
				decode_v |= (T(v) & 0x7F) << 14;
				offset += __popc(flags);
				s_offset = offset;
				if(v >= 128) {
					flags = __ballot(1);
					t_offset = __popc(mask & flags);
					v = packet_start[offset + t_offset];
					decode_v |= (T(v) & 0x7F) << 21;
					offset += __popc(flags);
					s_offset = offset;
					if(v >= 128) {
						flags = __ballot(1);
						t_offset = __popc(mask & flags);
						v = packet_start[offset + t_offset];
						decode_v |= (T(v) & 0x7F) << 28;
						offset += __popc(flags);
						s_offset = offset;
						if(v >= 128) {
							flags = __ballot(1);
							t_offset = __popc(mask & flags);
							v = packet_start[offset + t_offset];
							decode_v |= (T(v) & 0x7F) << 35;
							offset += __popc(flags);
							s_offset = offset;
							if(v >= 128) {
								flags = __ballot(1);
								t_offset = __popc(mask & flags);
								v = packet_start[offset + t_offset];
								decode_v |= (T(v) & 0x7F) << 49;
								offset += __popc(flags);
								s_offset = offset;
								if(v >= 128) {
									flags = __ballot(1);
									t_offset = __popc(mask & flags);
									v = packet_start[offset + t_offset];
									decode_v |= (T(v) & 0x7F) << 56;
									offset += __popc(flags);
									s_offset = offset;
								}
							}
						}
					}
				}
			}
		}
#if 0
		decode_buffer[i] = (decode_v >> 1) ^ ((decode_v << 63) >> 63);
#else
		int64_t v_abs = decode_v >> 1;
		decode_buffer[i] = (decode_v&1) ? ~v_abs : v_abs;
#endif
		if(offset < s_offset)
			offset = s_offset;
	}
	offset = s_offset;
	return offset;
}

// 32 bit version //

__device__ int encode_varint_packet(
	const uint32_t* encode_data,	// data to be encoded
	int num_int,					// the number of integer values to be encoded
	uint8_t* encode_buffer,			// a buffer encoded data will be stored
	volatile int& s_offset,			// [shared memory] buffer used in this function
	int tid)						// thread identifier (require 0...31 thread)
{
	typedef uint32_t T;
	const uint mask = (1U << threadIdx.x) - 1U;
	int offset = 0;
	s_offset = 0;
	for(int i = tid; i < num_int; i += WARP_SIZE) {
		int flags, t_offset;
		T data = encode_data[i];
		int w_offset = offset + tid;
		flags = __ballot(1);
		offset += __popc(flags);
		s_offset = offset;
		if(data >= 128) {
			encode_buffer[w_offset] = data | 0x80;
			flags = __ballot(1);
			t_offset = __popc(mask & flags);
			w_offset = offset + t_offset;
			offset += __popc(flags);
			s_offset = offset;
			data >>= 7;
			if(data >= 128) {
				encode_buffer[w_offset] = data | 0x80;
				flags = __ballot(1);
				t_offset = __popc(mask & flags);
				w_offset = offset + t_offset;
				offset += __popc(flags);
				s_offset = offset;
				data >>= 7;
				if(data >= 128) {
					encode_buffer[w_offset] = data | 0x80;
					flags = __ballot(1);
					t_offset = __popc(mask & flags);
					w_offset = offset + t_offset;
					offset += __popc(flags);
					s_offset = offset;
					data >>= 7;
					if(data >= 128) {
						encode_buffer[w_offset] = data | 0x80;
						flags = __ballot(1);
						t_offset = __popc(mask & flags);
						w_offset = offset + t_offset;
						offset += __popc(flags);
						s_offset = offset;
					}
				}
			}
		}
		encode_buffer[w_offset] = data;
		if(offset < s_offset)
			offset = s_offset;
	}
	offset = s_offset;
	return offset;
}

__device__ int decode_varint_packet(
	const uint8_t* packet_start,	// data to be decoded
	int num_int,					// the number of integer values to be decoded
	uint32_t* decode_buffer,		// a buffer decoded data will be stored
	volatile int& s_offset,			// [shared memory] buffer used in thid function
	int tid)						// thread identifier (require 0...31 thread)
{
	typedef uint32_t T;
	const uint mask = (1U << threadIdx.x) - 1U;
	int offset = 0;
	s_offset = 0;
	for(int i = tid; i < num_int; i += WARP_SIZE) {
		int flags, t_offset;
		uint8_t v = packet_start[offset + tid];
		T decode_v = T(v) & 0x7F;
		flags = __ballot(1);
		offset += __popc(flags);
		s_offset = offset;
		if(v >= 128) {
			flags = __ballot(1);
			t_offset = __popc(mask & flags);
			v = packet_start[offset + t_offset];
			decode_v |= (T(v) & 0x7F) << 7;
			offset += __popc(flags);
			s_offset = offset;
			if(v >= 128) {
				flags = __ballot(1);
				t_offset = __popc(mask & flags);
				v = packet_start[offset + t_offset];
				decode_v |= (T(v) & 0x7F) << 14;
				offset += __popc(flags);
				s_offset = offset;
				if(v >= 128) {
					flags = __ballot(1);
					t_offset = __popc(mask & flags);
					v = packet_start[offset + t_offset];
					decode_v |= (T(v) & 0x7F) << 21;
					offset += __popc(flags);
					s_offset = offset;
					if(v >= 128) {
						flags = __ballot(1);
						t_offset = __popc(mask & flags);
						v = packet_start[offset + t_offset];
						decode_v |= (T(v) & 0x7F) << 28;
						offset += __popc(flags);
						s_offset = offset;
					}
				}
			}
		}
		decode_buffer[i] = decode_v;
		if(offset < s_offset)
			offset = s_offset;
	}
	offset = s_offset;
	return offset;
}

__device__ int decode_varint_packet_signed(
	const uint8_t* packet_start,	// data to be decoded
	int num_int,					// the number of integer values to be decoded
	int32_t* decode_buffer,		// a buffer decoded data will be stored
	volatile int& s_offset,			// [shared memory] buffer used in thid function
	int tid)						// thread identifier (require 0...31 thread)
{
	typedef int32_t T;
	const uint mask = (1U << threadIdx.x) - 1U;
	int offset = 0;
	s_offset = 0;
	for(int i = tid; i < num_int; i += WARP_SIZE) {
		int flags, t_offset;
		uint8_t v = packet_start[offset + tid];
		T decode_v = T(v) & 0x7F;
		flags = __ballot(1);
		offset += __popc(flags);
		s_offset = offset;
		if(v >= 128) {
			flags = __ballot(1);
			t_offset = __popc(mask & flags);
			v = packet_start[offset + t_offset];
			decode_v |= (T(v) & 0x7F) << 7;
			offset += __popc(flags);
			s_offset = offset;
			if(v >= 128) {
				flags = __ballot(1);
				t_offset = __popc(mask & flags);
				v = packet_start[offset + t_offset];
				decode_v |= (T(v) & 0x7F) << 14;
				offset += __popc(flags);
				s_offset = offset;
				if(v >= 128) {
					flags = __ballot(1);
					t_offset = __popc(mask & flags);
					v = packet_start[offset + t_offset];
					decode_v |= (T(v) & 0x7F) << 21;
					offset += __popc(flags);
					s_offset = offset;
					if(v >= 128) {
						flags = __ballot(1);
						t_offset = __popc(mask & flags);
						v = packet_start[offset + t_offset];
						decode_v |= (T(v) & 0x7F) << 28;
						offset += __popc(flags);
						s_offset = offset;
					}
				}
			}
		}
		int32_t v_abs = decode_v >> 1;
		decode_buffer[i] = (decode_v&1) ? ~v_abs : v_abs;
#if 0
	if(tid == 0) printf("K:%d:decode_buffer[i]=%d\n", __LINE__, decode_buffer[i]);
#endif
		if(offset < s_offset)
			offset = s_offset;
	}
	offset = s_offset;
	return offset;
}

template <typename T>
__global__ void memset_kernel(
	T* ptr,
	T value,
	gsize_t length)
{
	const gsize_t warp_offset = WARP_IDX * WARP_SIZE * LOOPS_PER_THREAD;
	const gsize_t end_offset = min(warp_offset + WARP_SIZE * LOOPS_PER_THREAD, length);

	for(gsize_t i = warp_offset + threadIdx.x; i < end_offset; i += WARP_SIZE) {
		ptr[i] = value;
	}
}

template <typename T>
void memset_gpu(T* ptr, T value, gsize_t length, cudaStream_t stream = NULL)
{
	memset_kernel<<<nblocks2n<int>(length, THREADS_PER_BLOCK * LOOPS_PER_THREAD),
			dim3(WARP_SIZE, WARPS_PER_BLOCK), 0, stream>>>(ptr, value, length);
}

#endif /* GPU_HPP_ */
