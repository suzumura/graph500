/*
 * prefix_sum.h
 *
 *  Created on: Feb 28, 2012
 *      Author: koji
 */

#ifndef PREFIX_SUM_H_
#define PREFIX_SUM_H_

#include <stdint.h>
#include <inttypes.h>
#include <assert.h>

// 0: reducing compile time, 1: optimizing for speed
#define OPT_SPEED 0

// Forward Declaration //

// SIMD Kogge-Stone implementation
// LOG_MAX: <= 8
// s_data: offset: +1, [-(1 << (LOG_MAX - 1) ... 1 << (LOG_MAX)]
// tid: 0 ... 1 << (LOG_MAX - 1)
template <typename T, int LOG_MAX>
__device__ void dev_simd_kogge_stone(volatile T* s_data, const int tid);

// LOG_MAX : 7 <= LOG_MAX <= 11
// tid: 0 ... 1 << (LOG_MAX - 1)
// sdata : offset: +0, [0 ... 1 << (LOG_MAX + 1)]
template <typename T, int LOG_MAX>
__device__ void dev_prefix_sum(T* s_data, const int tid);

// LOG_MAX: recommendation: 10
// blockDim: (1 << (LOG_MAX - 1), 0, 0)
// idata: offset: +1
template <typename T, int LOG_MAX>
__global__ void prefix_sum_one_block(T* idata);

// LOG_MAX: recommendation: 10
// blockDim: (1 << (LOG_MAX - 1), 0, 0)
// idata: offset: +1
template <typename T, int LOG_MAX>
__global__ void prefix_sum_one_kernel(T* idata, const int num_blocks);

// Implementation//

namespace prefix_sum_detail {

template <typename T>
__global__ void prefix_sum_reference_kernel(T* odata, T* idata, const int n)
{
	extern __shared__ T temp[];
	const int tid = threadIdx.x;
	int offset = 1;
	const int ai = tid + 0;
	const int bi = tid + n/2;
	temp[ai] = idata[blockIdx.x*2*blockDim.x + ai];
	temp[bi] = idata[blockIdx.x*2*blockDim.x + bi];
	for(int d = n >> 1; d > 0; d >>= 1) {
		__syncthreads();
		if(tid < d) {
			int ai = offset*(2*tid+1)-1;
			int bi = offset*(2*tid+2)-1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if(tid == 0) { temp[n - 1] = 0; }
	for(int d = 1; d < n; d *= 2) {
		offset >>= 1;
		__syncthreads();
		if(tid < d) {
			int ai = offset*(2*tid+1)-1;
			int bi = offset*(2*tid+2)-1;
			T t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	odata[blockIdx.x*2*blockDim.x + ai] = temp[ai];
	odata[blockIdx.x*2*blockDim.x + bi] = temp[bi];
}

// LOG_MAX: recommendation: 10
// blockDim: (1 << (LOG_MAX - 2), 0, 0)
// idata: offset: +1
// itermid: offset: +1
template <typename T, int LOG_MAX>
__global__ void prefix_sum_up_phase(T* idata, T* intermid, const int num_blocks, const int num_loops)
{
	const int tid = threadIdx.x;
	const int MAX = 1U << LOG_MAX;
	const int HARF_MAX = MAX / 2;
	const int QUAT_MAX = MAX / 4; // == blockDim.x

	const int i_start = num_loops * blockIdx.x;
	const int i_end = min(i_start + num_loops, num_blocks);

	__shared__ T s_data[(1U << LOG_MAX)/2];
	T sum = T(0);

	for(int i = i_start; i < i_end; ++i) {
		s_data[tid + 0       ] = idata[i*MAX + QUAT_MAX*0 + tid]
		                       + idata[i*MAX + QUAT_MAX*1 + tid];
		s_data[tid + QUAT_MAX] = idata[i*MAX + QUAT_MAX*2 + tid]
		                       + idata[i*MAX + QUAT_MAX*3 + tid];
		__syncthreads();

		dev_reduce<T, HARF_MAX>(HARF_MAX, s_data, tid);
		if(tid == 0) sum += s_data[0];
		__syncthreads();
	}

	if(tid == 0) intermid[blockIdx.x] = sum;
}

// LOG_MAX: recommendation: 10
// blockDim: (1 << (LOG_MAX - 1), 0, 0)
// idata: offset: +1
// itermid: offset: +0
template <typename T, int LOG_MAX>
__global__ void prefix_sum_down_phase(T* idata, T* intermid, const int num_blocks, const int num_loops)
{
	const int tid = threadIdx.x;
	const int MAX = 1U << LOG_MAX;
	const int HARF_MAX = MAX / 2; // == blockDim.x

	const int i_start = num_loops * blockIdx.x;
	const int i_end = min(i_start + num_loops, num_blocks);

	__shared__ T s_data[(1U << LOG_MAX)*2];
	T* s_in = s_data + 1;
	T carry;
	if(tid == 0) carry = intermid[blockIdx.x];

	for(int i = i_start; i < i_end; ++i) {
		s_in[tid + HARF_MAX*0] = idata[i*MAX + HARF_MAX*0 + tid];
		s_in[tid + HARF_MAX*1] = idata[i*MAX + HARF_MAX*1 + tid];
		if(tid == 0) s_data[0] = carry;
		__syncthreads();

		dev_prefix_sum<T, LOG_MAX>(s_data, tid);

		idata[i*MAX + HARF_MAX*0 + tid] = s_in[tid + HARF_MAX*0];
		idata[i*MAX + HARF_MAX*1 + tid] = s_in[tid + HARF_MAX*1];
		if(tid == 0) carry = s_data[MAX];
		__syncthreads();
	}

	if(tid == 0 && blockIdx.x == 0) idata[-1] = 0;
}

} // namespace prefix_sum_detail {

// optimized version //

// SIMD Kogge-Stone implementation
// LOG_MAX: <= 8
// s_data: offset: +1, [-(1 << (LOG_MAX - 1) ... 1 << (LOG_MAX)]
// tid: 0 ... 1 << LOG_MAX
template <typename T, int LOG_MAX>
__device__ void dev_simd_kogge_stone(volatile T* s_data, const int tid)
{
	if(tid < (1 << (LOG_MAX - 1))) s_data[tid - (1 << (LOG_MAX - 1))] = T(0);
	if(LOG_MAX >= 1) s_data[tid] += s_data[tid - 1];
	if(LOG_MAX >= 2) s_data[tid] += s_data[tid - 2];
	if(LOG_MAX >= 3) s_data[tid] += s_data[tid - 4];
	if(LOG_MAX >= 4) s_data[tid] += s_data[tid - 8];
	if(LOG_MAX >= 5) s_data[tid] += s_data[tid - 16];
	if(LOG_MAX >= 6) s_data[tid] += s_data[tid - 32];
	if(LOG_MAX >= 7) s_data[tid] += s_data[tid - 64];
	if(LOG_MAX >= 8) s_data[tid] += s_data[tid - 128];
}

// LOG_MAX : 7 <= LOG_MAX <= 11
// tid: 0 ... 1 << (LOG_MAX - 1)
// sdata : offset: +0, [0 ... 1 << (LOG_MAX + 1)]
template <typename T, int LOG_MAX>
__device__ void dev_prefix_sum(T* s_data, const int tid)
{
	enum {
		OFFSET_2048 = (LOG_MAX >= 11) ? 2048 + 1 : 0,
		OFFSET_1024 = (LOG_MAX >= 10) ? OFFSET_2048 + 1024 + 1 : 0,
		OFFSET_512 = (LOG_MAX >= 9) ? OFFSET_1024 + 512 + 1 : 0,
		OFFSET_256 = (LOG_MAX >= 8) ? OFFSET_512 + 256 + 1 : 0,
		OFFSET_128 = (LOG_MAX >= 7) ? OFFSET_256 + 128 + 1 : 0,
		OFFSET_64 = OFFSET_128 + 64 + 1 + 16,
	};
	const int tid2 = 2 * tid;
#if DEBUG_PRINT
	const int HARF_SIZE = 1 << (LOG_MAX - 1);
	const T base_v = s_data[0];
	const T init_v1 = s_data[tid + 1];
	const T init_v2 = s_data[tid + 1 + HARF_SIZE];
#endif

	if(LOG_MAX >= 11) { if(tid < 1024) {
			s_data[OFFSET_2048 + tid] = s_data[tid2 + 0] +
									    s_data[tid2 + 1];
	} __syncthreads(); }
	if(LOG_MAX >= 10) { if(tid < 512) {
			s_data[OFFSET_1024 + tid] = s_data[OFFSET_2048 + tid2 + 0] +
									    s_data[OFFSET_2048 + tid2 + 1];
	} __syncthreads(); }
	if(LOG_MAX >= 9) { if(tid < 256) {
			s_data[OFFSET_512 + tid] = s_data[OFFSET_1024 + tid2 + 0] +
									   s_data[OFFSET_1024 + tid2 + 1];
	} __syncthreads(); }
	if(LOG_MAX >= 8) { if(tid < 128) {
		s_data[OFFSET_256 + tid] = s_data[OFFSET_512 + tid2 + 0] +
								   s_data[OFFSET_512 + tid2 + 1];
	} __syncthreads(); }
	if(LOG_MAX >= 7) { if(tid < 64) {
		s_data[OFFSET_128 + tid] = s_data[OFFSET_256 + tid2 + 0] +
								   s_data[OFFSET_256 + tid2 + 1];
	} __syncthreads(); }
	if(LOG_MAX >= 6) { if(tid < 32) {
		s_data[OFFSET_64 + tid] = s_data[OFFSET_128 + tid2 + 0] +
								  s_data[OFFSET_128 + tid2 + 1];
	} __syncthreads(); }

	if(tid < 32) { dev_simd_kogge_stone<T, 5>(s_data + OFFSET_64, tid); } __syncthreads();

	T* const s_out = s_data + 1;

	if(LOG_MAX >= 6) { if(tid < 32) {
		T tmp = s_data[OFFSET_64 + tid];
		s_out[OFFSET_128 + tid2 + 0] = tmp;
		s_out[OFFSET_128 + tid2 + 1] += tmp;
	} __syncthreads(); }
	if(LOG_MAX >= 7) { if(tid < 64) {
		T tmp = s_data[OFFSET_128 + tid];
		s_out[OFFSET_256 + tid2 + 0] = tmp;
		s_out[OFFSET_256 + tid2 + 1] += tmp;
	} __syncthreads(); }
	if(LOG_MAX >= 8) { if(tid < 128) {
		T tmp = s_data[OFFSET_256 + tid];
		s_out[OFFSET_512 + tid2 + 0] = tmp;
		s_out[OFFSET_512 + tid2 + 1] += tmp;
	} __syncthreads(); }
	if(LOG_MAX >= 9) { if(tid < 256) {
		T tmp = s_data[OFFSET_512 + tid];
		s_out[OFFSET_1024 + tid2 + 0] = tmp;
		s_out[OFFSET_1024 + tid2 + 1] += tmp;
	} __syncthreads(); }
	if(LOG_MAX >= 10) { if(tid < 512) {
		T tmp = s_data[OFFSET_1024 + tid];
		s_out[OFFSET_2048 + tid2 + 0] = tmp;
		s_out[OFFSET_2048 + tid2 + 1] += tmp;
	} __syncthreads(); }
	if(LOG_MAX >= 11) { if(tid < 1024) {
		T tmp = s_data[OFFSET_2048 + tid];
		s_out[tid2 + 0] = tmp;
		s_out[tid2 + 1] += tmp;
	} __syncthreads(); }
#if DEBUG_PRINT
	if(tid < HARF_SIZE) {
		const T diff_v1 = s_data[tid + 1] - s_data[tid];
		const T diff_v2 = s_data[tid + 1 + HARF_SIZE] - s_data[tid + HARF_SIZE];
		if(init_v1 != diff_v1) {
			printf("KU:%d:R(%d)B(%d)T(%d,%d):PrefixSumError1(%d!=%d)\n", __LINE__, mpig.rank_2d, blockIdx.x, threadIdx.x, threadIdx.y, init_v1, diff_v1);
		}
		if(init_v2 != diff_v2) {
			printf("KU:%d:R(%d)B(%d)T(%d,%d):PrefixSumError2(%d!=%d)\n", __LINE__, mpig.rank_2d, blockIdx.x, threadIdx.x, threadIdx.y, init_v2, diff_v2);
		}
		if(tid == 0 && base_v != s_data[0]) {
			printf("KU:%d:R(%d)B(%d)T(%d,%d):PrefixSumError3(%d!=%d)\n", __LINE__, mpig.rank_2d, blockIdx.x, threadIdx.x, threadIdx.y, base_v, s_data[0]);
		}
	}
#endif
}
#if 0
// LOG_MAX : 7 <= LOG_MAX <= 8
// tid: 0 ... 1 << (LOG_MAX - 1)
// sdata : offset: +0, [0 ... 1 << (LOG_MAX + 1)]
template <typename T, int LOG_MAX>
__device__ void dev_prefix_sum_warp(T* s_data, const int tid)
{
	enum {
		OFFSET_256 = (LOG_MAX >= 8) ? 256 + 1 : 0,
		OFFSET_128 = (LOG_MAX >= 7) ? OFFSET_256 + 128 + 1 : 0,
		OFFSET_64 = OFFSET_128 + 64 + 1 + 16,
	};
	const int tid2 = 2 * tid;

	if(LOG_MAX >= 10) if(tid < 512) {
			s_data[OFFSET_1024 + tid] = s_data[OFFSET_2048 + tid2 + 0] +
									    s_data[OFFSET_2048 + tid2 + 1];
	}
	if(LOG_MAX >= 9) { if(tid < 256) {
			s_data[OFFSET_512 + tid] = s_data[OFFSET_1024 + tid2 + 0] +
									   s_data[OFFSET_1024 + tid2 + 1];
	} __syncthreads(); }
	if(LOG_MAX >= 8) { if(tid < 128) {
		s_data[OFFSET_256 + tid] = s_data[OFFSET_512 + tid2 + 0] +
								   s_data[OFFSET_512 + tid2 + 1];
	} __syncthreads(); }
	if(LOG_MAX >= 7) { if(tid < 64) {
		s_data[OFFSET_128 + tid] = s_data[OFFSET_256 + tid2 + 0] +
								   s_data[OFFSET_256 + tid2 + 1];
	} __syncthreads(); }
	if(LOG_MAX >= 6) { if(tid < 32) {
		s_data[OFFSET_64 + tid] = s_data[OFFSET_128 + tid2 + 0] +
								  s_data[OFFSET_128 + tid2 + 1];
	} __syncthreads(); }

	if(tid < 32) { dev_simd_kogge_stone<T, 5>(s_data + OFFSET_64, tid); } __syncthreads();

	T* const s_out = s_data + 1;

	if(LOG_MAX >= 6) { if(tid < 32) {
		T tmp = s_data[OFFSET_64 + tid];
		s_out[OFFSET_128 + tid2 + 0] = tmp;
		s_out[OFFSET_128 + tid2 + 1] += tmp;
	} __syncthreads(); }
	if(LOG_MAX >= 7) { if(tid < 64) {
		T tmp = s_data[OFFSET_128 + tid];
		s_out[OFFSET_256 + tid2 + 0] = tmp;
		s_out[OFFSET_256 + tid2 + 1] += tmp;
	} __syncthreads(); }
	if(LOG_MAX >= 8) { if(tid < 128) {
		T tmp = s_data[OFFSET_256 + tid];
		s_out[OFFSET_512 + tid2 + 0] = tmp;
		s_out[OFFSET_512 + tid2 + 1] += tmp;
	} __syncthreads(); }
	if(LOG_MAX >= 9) { if(tid < 256) {
		T tmp = s_data[OFFSET_512 + tid];
		s_out[OFFSET_1024 + tid2 + 0] = tmp;
		s_out[OFFSET_1024 + tid2 + 1] += tmp;
	} __syncthreads(); }
	if(LOG_MAX >= 10) { if(tid < 512) {
		T tmp = s_data[OFFSET_1024 + tid];
		s_out[OFFSET_2048 + tid2 + 0] = tmp;
		s_out[OFFSET_2048 + tid2 + 1] += tmp;
	} __syncthreads(); }
	if(LOG_MAX >= 11) { if(tid < 1024) {
		T tmp = s_data[OFFSET_2048 + tid];
		s_out[tid2 + 0] = tmp;
		s_out[tid2 + 1] += tmp;
	} __syncthreads(); }
}
#endif
// LOG_MAX: recommendation: 10
// blockDim: (1 << (LOG_MAX - 1), 0, 0)
// idata: offset: +1
template <typename T, int LOG_MAX>
__global__ void prefix_sum_one_block(T* idata)
{
	const int tid = threadIdx.x;
	const int MAX = 1U << LOG_MAX;
	const int HARF_MAX = MAX / 2; // == blockDim.x

	__shared__ T s_data[(1U << LOG_MAX)*2];
	T* const s_in = s_data + 1;

	s_in[tid + 0] = idata[tid];
	s_in[tid + HARF_MAX] = idata[tid + HARF_MAX];
	if(tid == 0) s_in[-1] = T(0);
	__syncthreads();

	dev_prefix_sum<T, LOG_MAX>(s_data, tid);

	idata[tid + 0] = s_in[tid + 0];
	idata[tid + HARF_MAX] = s_in[tid + HARF_MAX];
	if(tid == 0) idata[-1] = 0;
}

// LOG_MAX: recommendation: 10
// blockDim: (1 << (LOG_MAX - 1), 0, 0)
// idata: offset: +1
template <typename T, int LOG_MAX>
__global__ void prefix_sum_one_kernel(T* idata, const int num_blocks)
{
	const int tid = threadIdx.x;
	const int MAX = 1U << LOG_MAX;
	const int HARF_MAX = MAX / 2; // == blockDim.x

	__shared__ T s_data[(1U << LOG_MAX)*2];
	T* s_in = s_data + 1;
	T carry = 0;

	for(int i = 0; i < num_blocks; ++i) {
		s_in[tid + HARF_MAX*0] = idata[i*MAX + HARF_MAX*0 + tid];
		s_in[tid + HARF_MAX*1] = idata[i*MAX + HARF_MAX*1 + tid];
		if(tid == 0) s_data[0] = carry;
		__syncthreads();

		dev_prefix_sum<T, LOG_MAX>(s_data, tid);

		idata[i*MAX + HARF_MAX*0 + tid] = s_in[tid + HARF_MAX*0];
		idata[i*MAX + HARF_MAX*1 + tid] = s_in[tid + HARF_MAX*1];
		if(tid == 0) carry = s_data[MAX];
		__syncthreads();
	}

	if(tid == 0 && blockIdx.x == 0) idata[-1] = 0;
}

template <typename T> class vector2
{
public:
	T x, y;

	__host__ __device__ vector2() { }
	__host__ __device__ vector2(T x__)
		: x(x__)
		, y(x__)
	{ }
	__host__ __device__ vector2(T x__, T y__)
		: x(x__)
		, y(y__)
	{ }
	__host__ __device__ vector2(const vector2<T>& r)
		: x(r.x)
		, y(r.y)
	{ }
	__host__ __device__ vector2(const volatile vector2<T>& r)
		: x(r.x)
		, y(r.y)
	{ }

	// +
	__host__ __device__ vector2<T>& operator+(const vector2<T> r) const {
		return vector2<T>(x + r.x, y + r.y);
	}
	__host__ __device__ vector2<T>& operator+=(const vector2<T> r) {
		x += r.x; y += r.y; return *this;
	}
	__host__ __device__ volatile vector2<T>& operator+=(const vector2<T> r) volatile {
		x += r.x; y += r.y; return *this;
	}

	// -
	__host__ __device__ vector2<T>& operator-(const vector2<T> r) const {
		return vector2<T>(x - r.x, y - r.y);
	}
	__host__ __device__ vector2<T>& operator-=(const vector2<T> r) {
		x -= r.x; y -= r.y; return *this;
	}
	__host__ __device__ volatile vector2<T>& operator-=(const vector2<T> r) volatile {
		x -= r.x; y -= r.y; return *this;
	}

	// *
	__host__ __device__ vector2<T>& operator*(const vector2<T> r) const {
		return vector2<T>(x * r.x, y * r.y);
	}
	__host__ __device__ vector2<T>& operator*=(const vector2<T> r) {
		x *= r.x; y *= r.y; return *this;
	}
	__host__ __device__ volatile vector2<T>& operator*=(const vector2<T> r) volatile {
		x *= r.x; y *= r.y; return *this;
	}

	// /
	__host__ __device__ vector2<T>& operator/(const vector2<T> r) const {
		return vector2<T>(x / r.x, y / r.y);
	}
	__host__ __device__ vector2<T>& operator/=(const vector2<T> r) {
		x /= r.x; y /= r.y; return *this;
	}
	__host__ __device__ volatile vector2<T>& operator/=(const vector2<T> r) volatile {
		x /= r.x; y /= r.y; return *this;
	}
};

//-------------------------------------------------------------//
// PrefixSumGPU
//-------------------------------------------------------------//

template <typename T>
PrefixSumGPU<T>::PrefixSumGPU(int64_t max_length, int num_partitions)
{
	assert (max_length > 0);
	num_partitions_ = num_partitions;
	log_max_length_ = get_msb_index(max_length - 1) + 1;
	cudaDeviceProp& dev_prop = CudaStreamManager::get_instance()->getDeviceProp();
	int num_ctas_per_sm = dev_prop.maxThreadsPerMultiProcessor / CTA_DIM_X;
	int num_active_ctas = num_ctas_per_sm * dev_prop.multiProcessorCount;
	num_launch_ctas_ = MAX_ACTIVE_CTAS / num_active_ctas * num_active_ctas;
	max_length_ = (max_length + MAX - 1) & (-MAX);
	CUDA_CHECK(cudaMalloc((void**)&dev_array_, sizeof(T)*num_partitions_*(max_length_ + PADDING + MAX_ACTIVE_CTAS)));
}
template <typename T>
PrefixSumGPU<T>::~PrefixSumGPU()
{
	CUDA_CHECK(cudaFree(dev_array_));
}

#define STREAM_I (streams ? streams[i % num_streams] : NULL)
#define BUFFER_WIDTH (max_length_ + PADDING + MAX_ACTIVE_CTAS)
#define ARRAY_I	(dev_array_ + i*BUFFER_WIDTH)
#define INTERMID_I (dev_array_ + i*BUFFER_WIDTH + max_length_ + PADDING)

template <typename T>
void PrefixSumGPU<T>::operator()(cudaStream_t* streams, int num_streams)
{
	using namespace prefix_sum_detail;
	if(log_max_length_ <= THRESHOLD) { // 4K
		for(int i = 0; i < num_partitions_; ++i)
			prefix_sum_one_kernel<T, LOG_MAX>
			<<<1, CTA_DIM_X, 0, STREAM_I>>>(ARRAY_I + 1, max_length_ / MAX);
	}
	else {
		int num_blocks = max_length_ / MAX;
		int num_ctas = std::min(num_launch_ctas_, num_blocks);
		int num_loops = nblocks2n(num_blocks, num_ctas);
		for(int i = 0; i < num_partitions_; ++i)
			prefix_sum_up_phase<T, LOG_MAX>
			<<<num_ctas, CTA_DIM_X/2, 0, STREAM_I>>>(ARRAY_I + 1, INTERMID_I, num_blocks, num_loops);
		for(int i = 0; i < num_partitions_; ++i)
			prefix_sum_one_block<T, LOG_MAX_ACTIVE_CTAS>
			<<<1, MAX_ACTIVE_CTAS/2, 0, STREAM_I>>>(INTERMID_I);
		for(int i = 0; i < num_partitions_; ++i)
			prefix_sum_down_phase<T, LOG_MAX>
			<<<num_ctas, CTA_DIM_X, 0, STREAM_I>>>(ARRAY_I + 1, INTERMID_I - 1, num_blocks, num_loops);
	}
}

template class PrefixSumGPU<int>;

#endif /* PREFIX_SUM_H_ */
