/*
 * graph_generator.hpp
 *
 *  Created on: Dec 9, 2011
 *      Author: koji
 */
/* Copyright (C) 2009-2010 The Trustees of Indiana University.             */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

#ifndef GRAPH_GENERATOR_HPP_
#define GRAPH_GENERATOR_HPP_

#include <stdint.h>
#include <assert.h>

#include <algorithm>

#include "../mpi/mpi_workarounds.h"

#define FAST_64BIT_ARITHMETIC
#include "splittable_mrg.h"

#include "../mpi/primitives.hpp"

//-------------------------------------------------------------//
// Edge List
//-------------------------------------------------------------//

template <typename EdgeType, int CHUNK_SIZE_>
class EdgeListStorage {
public:
	static const int CHUNK_SIZE = CHUNK_SIZE_;
	typedef EdgeType edge_type;

	EdgeListStorage(int64_t nLocalEdges, const char* filepath = NULL)
		: data_in_file_(false)
		, edge_memory_(NULL)
		, edge_file_(NULL)
		, num_local_edges_(nLocalEdges)
		, edge_memory_size_(0)
		, edge_filled_size_(0)
		, max_edge_size_among_all_procs_(0)

		, read_enabled_(false)
		, read_buffer_(NULL)

		, write_enabled_(false)
		, write_buffer_(NULL)
		, write_buffer_filled_size_(0)
		, write_buffer_size_(0)
	{
		edge_memory_size_ = nLocalEdges * 103 / 100 + CHUNK_SIZE; // add 3%
#if VERVOSE_MODE
		if(mpi.isMaster()) {
			print_with_prefix("Allocating edge list memory (%"PRId64" * %d bytes)", edge_memory_size_*sizeof(EdgeType), mpi.size_2d);
		}
#endif
		edge_memory_ = static_cast<EdgeType*>
			(cache_aligned_xmalloc(edge_memory_size_*sizeof(EdgeType)));

		if(filepath == NULL) {
			data_in_file_ = false;
		}
		else {
			data_in_file_ = true;
			sprintf(filepath_, "%s-%03d", filepath, mpi.rank_2d);
			MPI_File_open(MPI_COMM_SELF, const_cast<char*>(filepath_),
							MPI_MODE_RDWR |
							MPI_MODE_CREATE |
						//	MPI_MODE_EXCL |
							MPI_MODE_DELETE_ON_CLOSE |
							MPI_MODE_UNIQUE_OPEN,
							MPI_INFO_NULL, &edge_file_);
			MPI_File_set_atomicity(edge_file_, 0);
			MPI_File_set_view(edge_file_, 0,
					MpiTypeOf<EdgeType>::type, MpiTypeOf<EdgeType>::type, const_cast<char*>("native"), MPI_INFO_NULL);
		}
	}

	~EdgeListStorage()
	{
		if(edge_memory_ != NULL) { free(edge_memory_); edge_memory_ = NULL; }
		if(data_in_file_ == false) {
		}
		else {
			MPI_File_close(&edge_file_); edge_file_ = NULL;
			if(read_buffer_ != NULL) { free(read_buffer_); read_buffer_ = NULL; }
		}
	}

	// return the number of loops each process must do
	int beginRead(bool release_buffer)
	{
		assert (read_enabled_ == false);
		read_enabled_ = true;
		read_block_index_ = 0;
		if(data_in_file_) {
			if(release_buffer) {
				if(edge_memory_ != NULL) { free(edge_memory_); edge_memory_ = NULL; }
			}
			if(edge_memory_ == NULL) {
				read_buffer_ = static_cast<EdgeType*>(cache_aligned_xmalloc(CHUNK_SIZE*2*sizeof(EdgeType)));
				EdgeType *buffer_to_read, *buffer_for_user;
				getReadBuffer(&buffer_to_read, &buffer_for_user);
				if(edge_filled_size_ > 0) {
					int read_count = static_cast<int>(std::min<int64_t>(edge_filled_size_, CHUNK_SIZE));
					MPI_File_iread_at(edge_file_, 0,
							buffer_to_read, read_count, MpiTypeOf<EdgeType>::type, &read_request_);
				}
			}
		}
		return (max_edge_size_among_all_procs_ + CHUNK_SIZE - 1) / CHUNK_SIZE;
	}

	// return the number of elements filled in the buffer
	int read(EdgeType** pp_buffer)
	{
		assert (read_enabled_ == true);

		int64_t read_offset = read_block_index_*CHUNK_SIZE;
		if(edge_filled_size_ <= read_offset) {
			// no more data
			return 0;
		}
		else {
			int filled_count = static_cast<int>
				(std::min<int64_t>(edge_filled_size_ - read_offset, CHUNK_SIZE));
			if(edge_memory_ != NULL) {
				*pp_buffer = edge_memory_ + read_offset;
				++read_block_index_; read_offset += CHUNK_SIZE;
			}
			else {
				MPI_Status read_result;
				MPI_Wait(&read_request_, &read_result);
				++read_block_index_; read_offset += CHUNK_SIZE;
				EdgeType *buffer_to_read, *buffer_for_user;
				getReadBuffer(&buffer_to_read, &buffer_for_user);
				*pp_buffer = buffer_for_user;
				if(edge_filled_size_ > read_offset) {
					int read_count = static_cast<int>
						(std::min<int64_t>(edge_filled_size_ - read_offset, CHUNK_SIZE));
					MPI_File_iread_at(edge_file_, read_offset,
							buffer_to_read, read_count, MpiTypeOf<EdgeType>::type, &read_request_);
				}
			}
			return filled_count;
		}
	}

	void endRead()
	{
		assert (read_enabled_ == true);

		if(edge_memory_ == NULL) {
			if(edge_filled_size_ > read_block_index_*CHUNK_SIZE) {
				// break reading loop
				MPI_Wait(&read_request_, MPI_STATUS_IGNORE);
			}
			if(read_buffer_ != NULL) { free(read_buffer_); read_buffer_ = NULL; }
		}

		if(write_buffer_filled_size_ > 0) {
			while(write_buffer_filled_size_ > 0) {
				reduceWriteBuffer();
			}
			if(write_buffer_ != NULL) { free(write_buffer_); write_buffer_ = NULL; }
			write_buffer_size_ = 0;
			if(write_enabled_ == false) {
				edge_filled_size_ = write_offset_;
			}
		}

		read_enabled_ = false;
	}

	void beginWrite()
	{
		assert (write_enabled_ == false);
		write_enabled_ = true;
		write_offset_ = 0;
		write_buffer_filled_size_ = 0;
	}

	void write(EdgeType* edge_data, int count)
	{
		assert (write_enabled_ == true);

		if(read_enabled_) {
			int64_t read_offset = read_block_index_*CHUNK_SIZE;
			// this writing is concurrent with reading
			if(write_buffer_filled_size_ > 0) {
				reduceWriteBuffer();
			}
			if(write_offset_ + count <= read_offset) {
				writeInternal(edge_data, count);
			}
			else {
				if(write_offset_ <= read_offset) {
					int64_t write_count = read_offset - write_offset_;
					writeInternal(edge_data, write_count);
					edge_data += write_count;
					count -= write_count;
				}
				if(write_buffer_filled_size_ + count > write_buffer_size_) {
					// buffer size is not enough
					EdgeType* swap_buffer = write_buffer_;
					while(write_buffer_filled_size_ + count > write_buffer_size_) {
						write_buffer_size_ = std::max(INT64_C(4096), write_buffer_size_*2);
					}
					write_buffer_ = static_cast<EdgeType*>
						(cache_aligned_xmalloc(write_buffer_size_*sizeof(EdgeType)));
					if(swap_buffer != NULL) {
						memcpy(write_buffer_, swap_buffer,
								write_buffer_filled_size_*sizeof(EdgeType));
						free(swap_buffer);
					}
				}
				memcpy(write_buffer_ + write_buffer_filled_size_,
						edge_data, count*sizeof(EdgeType));
				write_buffer_filled_size_ += count;
			}
		}
		else {
			writeInternal(edge_data, count);
		}
	}

	void endWrite()
	{
		assert (write_enabled_ == true);
		write_enabled_ = false;
		if(write_buffer_filled_size_ == 0) {
			edge_filled_size_ = write_offset_;
		}
		int64_t written_size = write_offset_ + write_buffer_filled_size_;
		MPI_Allreduce(&written_size, &max_edge_size_among_all_procs_,
				1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
	}

	int64_t num_local_edges() { return num_local_edges_; }
	bool data_is_in_file() { return data_in_file_; }
	const char* get_filepath() { return filepath_; }

private:
	void getReadBuffer(EdgeType** pp_buffer_to_read, EdgeType** pp_buffer_for_user) {
		if(read_block_index_ % 2) {
			*pp_buffer_to_read = read_buffer_ + CHUNK_SIZE;
			*pp_buffer_for_user = read_buffer_;
		}
		else {
			*pp_buffer_to_read = read_buffer_;
			*pp_buffer_for_user = read_buffer_ + CHUNK_SIZE;
		}
	}

	void writeInternal(EdgeType* edge_data, int count) {
		if(edge_memory_ != NULL) {
			if(write_offset_ + count > edge_memory_size_) {
				fprintf(IMD_OUT, "Warning: re-allocation edge memory buffer !!");
				EdgeType* new_memory = static_cast<EdgeType*>(realloc(edge_memory_, (write_offset_ + count)*sizeof(EdgeType)));
				if(new_memory == NULL) {
					throw "OutOfMemoryException";
				}
				edge_memory_ = new_memory;
				edge_memory_size_ = write_offset_ + count;
			}
			memcpy(edge_memory_ + write_offset_, edge_data, count*sizeof(EdgeType));
		}
		if(data_in_file_) {
			MPI_Status write_result;
			MPI_File_write_at(edge_file_, write_offset_,
					edge_data, count, MpiTypeOf<EdgeType>::type, &write_result);
		}
		write_offset_ += count;
	}

	void reduceWriteBuffer()
	{
		int write_count = static_cast<int>
			(std::min<int64_t>(write_buffer_filled_size_, CHUNK_SIZE));
		writeInternal(write_buffer_, write_count);
		write_buffer_filled_size_ -= write_count;
		if(write_buffer_filled_size_ > 0) {
			// move remaining data
			memmove(write_buffer_, write_buffer_ + write_count,
					write_buffer_filled_size_ *sizeof(EdgeType));
		}
	}

	bool data_in_file_;
	EdgeType* edge_memory_;
	MPI_File edge_file_;
	int64_t num_local_edges_;
	int64_t edge_memory_size_;
	int64_t edge_filled_size_;
	int64_t max_edge_size_among_all_procs_;

	// read
	bool read_enabled_;
	// The offset from which we read next or we are reading now.
	MPI_Offset read_block_index_;
	MPI_Request read_request_;
	EdgeType* read_buffer_;

	// write
	bool write_enabled_;
	MPI_Offset write_offset_;
	EdgeType* write_buffer_;
	int64_t write_buffer_filled_size_;
	int64_t write_buffer_size_;

	char filepath_[256];
};

//-------------------------------------------------------------//
// Generator Implementations
//-------------------------------------------------------------//

/* Spread the two 64-bit numbers into five nonzero values in the correct
 * range. */
void make_mrg_seed(uint64_t userseed1, uint64_t userseed2, mrg_state* seed) {
	seed->z1 = (userseed1 & 0x3FFFFFFF) + 1;
	seed->z2 = ((userseed1 >> 30) & 0x3FFFFFFF) + 1;
	seed->z3 = (userseed2 & 0x3FFFFFFF) + 1;
	seed->z4 = ((userseed2 >> 30) & 0x3FFFFFFF) + 1;
	seed->z5 = ((userseed2 >> 60) << 4) + (userseed1 >> 60) + 1;
}

/* PRNG interface for implementations; takes seed in same format as given by
 * users, and creates a vector of doubles in a reproducible (and
 * random-access) way. */
void make_random_numbers(
       /* in */ int64_t nvalues    /* Number of values to generate */,
       /* in */ uint64_t userseed1 /* Arbitrary 64-bit seed value */,
       /* in */ uint64_t userseed2 /* Arbitrary 64-bit seed value */,
       /* in */ int64_t position   /* Start index in random number stream */,
       /* out */ double* result    /* Returned array of values */
) {
  int64_t i;
  mrg_state st;
  make_mrg_seed(userseed1, userseed2, &st);

  mrg_skip(&st, 2, 0, 2 * position); /* Each double takes two PRNG outputs */

  for (i = 0; i < nvalues; ++i) {
    result[i] = mrg_get_double_orig(&st);
  }
}

namespace InitialEdgeType {
	enum type {
		NONE,
		BINARY_TREE,
		HAMILTONIAN_CYCLE
	};
}

class GraphGeneratorBase
{
public:
	GraphGeneratorBase(int scale, int edge_factor, int max_weight, int userseed1, int userseed2, InitialEdgeType::type initial_edge_type)
		: scale_(scale)
		, edge_factor_(edge_factor)
		, max_weight_(max_weight)
		, initial_edge_type_(initial_edge_type)
	{
		make_mrg_seed(userseed1, userseed2, &mrg_seed_);

		{
			mrg_state new_state = mrg_seed_;
			mrg_skip(&new_state, 50, 7, 0);
			scramble_val0_ = mrg_get_uint_orig(&new_state);
			scramble_val0_ *= UINT64_C(0xFFFFFFFF);
			scramble_val0_ += mrg_get_uint_orig(&new_state);
			scramble_val1_ = mrg_get_uint_orig(&new_state);
			scramble_val1_ *= UINT64_C(0xFFFFFFFF);
			scramble_val1_ += mrg_get_uint_orig(&new_state);
		}

		switch(initial_edge_type_) {
			case InitialEdgeType::NONE:
				num_initial_edges_ = 0;
				break;
			case InitialEdgeType::BINARY_TREE:
				num_initial_edges_ = num_global_verts() - 1;
				break;
			case InitialEdgeType::HAMILTONIAN_CYCLE:
				num_initial_edges_ = num_global_verts();
				break;
		}
	}

	int64_t num_global_verts() const { return (INT64_C(1) << scale_); }
	int64_t num_global_edges() const { return num_global_verts()*edge_factor_ + num_initial_edges_; }

	/* Reverse bits in a number; this should be optimized for performance
	* (including using bit- or byte-reverse intrinsics if your platform has them).
	* */
	static inline uint64_t bitreverse(uint64_t x) {
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 3)
#define USE_GCC_BYTESWAP /* __builtin_bswap* are in 4.3 but not 4.2 */
#endif

#ifdef FAST_64BIT_ARITHMETIC

		/* 64-bit code */
#ifdef USE_GCC_BYTESWAP
		x = __builtin_bswap64(x);
#else
		x = (x >> 32) | (x << 32);
		x = ((x >> 16) & UINT64_C(0x0000FFFF0000FFFF)) | ((x & UINT64_C(0x0000FFFF0000FFFF)) << 16);
		x = ((x >>  8) & UINT64_C(0x00FF00FF00FF00FF)) | ((x & UINT64_C(0x00FF00FF00FF00FF)) <<  8);
#endif
		x = ((x >>  4) & UINT64_C(0x0F0F0F0F0F0F0F0F)) | ((x & UINT64_C(0x0F0F0F0F0F0F0F0F)) <<  4);
		x = ((x >>  2) & UINT64_C(0x3333333333333333)) | ((x & UINT64_C(0x3333333333333333)) <<  2);
		x = ((x >>  1) & UINT64_C(0x5555555555555555)) | ((x & UINT64_C(0x5555555555555555)) <<  1);
		return x;

#else

		/* 32-bit code */
		uint32_t h = (uint32_t)(x >> 32);
		uint32_t l = (uint32_t)(x & UINT32_MAX);
#ifdef USE_GCC_BYTESWAP
		h = __builtin_bswap32(h);
		l = __builtin_bswap32(l);
#else
		h = (h >> 16) | (h << 16);
		l = (l >> 16) | (l << 16);
		h = ((h >> 8) & UINT32_C(0x00FF00FF)) | ((h & UINT32_C(0x00FF00FF)) << 8);
		l = ((l >> 8) & UINT32_C(0x00FF00FF)) | ((l & UINT32_C(0x00FF00FF)) << 8);
#endif
		h = ((h >> 4) & UINT32_C(0x0F0F0F0F)) | ((h & UINT32_C(0x0F0F0F0F)) << 4);
		l = ((l >> 4) & UINT32_C(0x0F0F0F0F)) | ((l & UINT32_C(0x0F0F0F0F)) << 4);
		h = ((h >> 2) & UINT32_C(0x33333333)) | ((h & UINT32_C(0x33333333)) << 2);
		l = ((l >> 2) & UINT32_C(0x33333333)) | ((l & UINT32_C(0x33333333)) << 2);
		h = ((h >> 1) & UINT32_C(0x55555555)) | ((h & UINT32_C(0x55555555)) << 1);
		l = ((l >> 1) & UINT32_C(0x55555555)) | ((l & UINT32_C(0x55555555)) << 1);
		return ((uint64_t)l << 32) | h; /* Swap halves */

#endif
#ifdef USE_GCC_BYTESWAP
#undef USE_GCC_BYTESWAP
#endif
	}

protected:

	/* Apply a permutation to scramble vertex numbers; a randomly generated
	 * permutation is not used because applying it at scale is too expensive. */
	int64_t scramble(int64_t v0) const {
		uint64_t val0 = scramble_val0_;
		uint64_t val1 = scramble_val1_;
		uint64_t v = static_cast<uint64_t>(v0);
		v += val0 + val1;
		v *= (val0 | UINT64_C(0x4519840211493211));
		v = (bitreverse(v) >> (64 - scale_));
		assert ((v >> scale_) == 0);
		v *= (val1 | UINT64_C(0x3050852102C843A5));
		v = (bitreverse(v) >> (64 - scale_));
		assert ((v >> scale_) == 0);
		return static_cast<int64_t>(v);
	}

	mrg_state mrg_seed() const { return mrg_seed_; }
	int scale() const { return scale_; }
	int edge_factor() const { return edge_factor_; }

	int64_t num_initial_edges() const { return num_initial_edges_; }

	// using SFINAE
	// function #1
	template <typename EdgeType>
	void generateWeight(EdgeType* edge_buffer, int64_t start_edge, int64_t end_edge,
			typename EdgeType::has_weight dummy = 0) const {
#if 0
#pragma omp for
		for(int64_t edge_index = start_edge; edge_index < end_edge; ++edge_index) {
			mrg_state new_state = mrg_seed();
			mrg_skip(&new_state, 30, 46, edge_index);
			edge_buffer[edge_index - start_edge].weight_ = (mrg_get_uint_orig(&new_state) % max_weight_) + 1;
		}
#else
		const int num_threads = omp_get_num_threads();
		const int thread_num = omp_get_thread_num();
		const int64_t partition_size = ((end_edge - start_edge) + num_threads - 1) / num_threads;
		int64_t thread_start_edge = start_edge + partition_size * thread_num;
		int64_t thread_end_edge = std::min<int64_t>(thread_start_edge + partition_size, end_edge);
		mrg_state new_state = mrg_seed();
		mrg_skip(&new_state, 30, 46, thread_start_edge);

		for(int64_t edge_index = thread_start_edge; edge_index < thread_end_edge; ++edge_index) {
#if 1 // for debug
			edge_buffer[edge_index - start_edge].weight_ = 0xBEEF;
#else
			edge_buffer[edge_index - start_edge].weight_ = (mrg_get_uint_orig(&new_state) % max_weight_) + 1;
#endif
		}
#endif
	}
	// function #2
	template <typename EdgeType>
	void generateWeight(EdgeType* edge_buffer, int start_edge, int end_edge,
			typename EdgeType::no_weight dummy = 0) const { }

	template <typename EdgeType>
	void generateInitialEdge(EdgeType* edge_buffer, int64_t start_edge, int64_t end_edge) const {
		switch(initial_edge_type_) {
			case InitialEdgeType::BINARY_TREE:
#pragma omp for
				for(int64_t edge_index = start_edge; edge_index < end_edge; ++edge_index) {
					edge_buffer[edge_index - start_edge].set(
							this->scramble(edge_index + 1),
							this->scramble((edge_index + 1) / 2));
				}
				break;
			case InitialEdgeType::HAMILTONIAN_CYCLE:
#pragma omp for
				for(int64_t edge_index = start_edge;
						edge_index < std::min(end_edge, num_initial_edges_-1); ++edge_index) {
					edge_buffer[edge_index - start_edge].set(
							this->scramble(edge_index),
							this->scramble(edge_index + 1));
				}
#pragma omp master
				if(end_edge == num_initial_edges_) {
					// generate the last initial edge
					edge_buffer[end_edge - start_edge].set(
							this->scramble(end_edge),
							this->scramble(0));
				}
				break;
			case InitialEdgeType::NONE:
				break;
		}
	}

private:

	const int scale_;
	const int edge_factor_;
	const int max_weight_;

	// MRG: Multiple Recursive random number Generator
	mrg_state mrg_seed_;
	uint64_t scramble_val0_;
	uint64_t scramble_val1_;

	InitialEdgeType::type initial_edge_type_;
	int64_t num_initial_edges_;
};

template <typename EdgeType>
class GraphGenerator : public GraphGeneratorBase
{
public:
	GraphGenerator(int scale, int edge_factor, int max_weight, int userseed1, int userseed2, InitialEdgeType::type initial_edge_type)
		: GraphGeneratorBase(scale, edge_factor, max_weight, userseed1, userseed2, initial_edge_type)
	{ }
	virtual ~GraphGenerator() { }
	virtual void generateRange(EdgeType* edge_buffer, int64_t start_edge, int64_t end_edge) const = 0;
};

template <typename EdgeType>
class RandomGraphGenerator : public GraphGenerator<EdgeType>
{
	typedef GraphGenerator<EdgeType> BaseType;
public:
	RandomGraphGenerator(int scale, int edge_factor, int max_weight, int userseed1, int userseed2,
			InitialEdgeType::type initial_edge_type)
		: GraphGenerator<EdgeType>(scale, edge_factor, max_weight, userseed1, userseed2, initial_edge_type)
	{ }

	virtual void generateRange(EdgeType* edge_buffer, int64_t start_edge, int64_t end_edge) const
	{
		if(start_edge < this->num_initial_edges()) {
			BaseType::generateInitialEdge(edge_buffer, start_edge, std::min(end_edge, this->num_initial_edges()));
		}

		const int64_t num_global_verts_minus1 = this->num_global_verts() - 1;

#pragma omp for
		for(int64_t edge_index = std::max(start_edge, this->num_initial_edges());
				edge_index < end_edge; ++edge_index) {
			mrg_state new_state = this->mrg_seed();
			mrg_skip(&new_state, 0, edge_index, 0);
			edge_buffer[edge_index - start_edge].set(
					this->scramble(mrg_get_uint_orig(&new_state) & num_global_verts_minus1),
					this->scramble(mrg_get_uint_orig(&new_state) & num_global_verts_minus1));
		}

		BaseType::generateWeight(edge_buffer, start_edge, end_edge);
	}
};

template <typename EdgeType, int INITIATOR_A_NUMERATOR, int INITIATOR_BC_NUMERATOR>
class RmatGraphGenerator : public GraphGenerator<EdgeType>
{
	typedef GraphGenerator<EdgeType> BaseType;
public:
	RmatGraphGenerator(int scale, int edge_factor, int max_weight, int userseed1, int userseed2,
			InitialEdgeType::type initial_edge_type)
		: GraphGenerator<EdgeType>(scale, edge_factor, max_weight, userseed1, userseed2, initial_edge_type)
	{ }

	virtual void generateRange(EdgeType* edge_buffer, int64_t start_edge, int64_t end_edge) const
	{
		if(start_edge < this->num_initial_edges()) {
			BaseType::generateInitialEdge(edge_buffer, start_edge, std::min(end_edge, this->num_initial_edges()));
		}

#pragma omp for
		for(int64_t edge_index = std::max(start_edge, this->num_initial_edges());
				edge_index < end_edge; ++edge_index) {
			mrg_state new_state = this->mrg_seed();
			mrg_skip(&new_state, 0, edge_index, 0);
			make_one_edge(this->num_global_verts(), 0, &new_state, &edge_buffer[edge_index - start_edge]);
		}

		BaseType::generateWeight(edge_buffer, start_edge, end_edge);
	}
private:
	enum PARAMS {
		/* Initiator settings: for faster random number generation, the initiator
		* probabilities are defined as fractions (a = INITIATOR_A_NUMERATOR /
		* INITIATOR_DENOMINATOR, b = c = INITIATOR_BC_NUMERATOR /
		* INITIATOR_DENOMINATOR, d = 1 - a - b - c. */
	//	INITIATOR_A_NUMERATOR = 5700,
	//	INITIATOR_BC_NUMERATOR = 1900,
		INITIATOR_DENOMINATOR = 10000,

		limit = (UINT32_C(0xFFFFFFFF) % INITIATOR_DENOMINATOR),
	};

	/* If this macro is defined to a non-zero value, use SPK_NOISE_LEVEL /
	* INITIATOR_DENOMINATOR as the noise parameter to use in introducing noise
	* into the graph parameters.  The approach used is from "A Hitchhiker's Guide
	* to Choosing Parameters of Stochastic Kronecker Graphs" by C. Seshadhri, Ali
	* Pinar, and Tamara G. Kolda (http://arxiv.org/abs/1102.5046v1), except that
	* the adjustment here is chosen based on the current level being processed
	* rather than being chosen randomly. */
#define SPK_NOISE_LEVEL 0
	/* #define SPK_NOISE_LEVEL 1000, -- in INITIATOR_DENOMINATOR units */

	static int generate_4way_bernoulli(mrg_state* st, int level, int nlevels) {
		/* Generator a pseudorandom number in the range [0, INITIATOR_DENOMINATOR)
		* without modulo bias. */
		uint32_t val = mrg_get_uint_orig(st);
		if (/* Unlikely */ val < limit) {
			do {
				val = mrg_get_uint_orig(st);
			} while (val < limit);
		}
#if SPK_NOISE_LEVEL == 0
		int spk_noise_factor = 0;
#else
		int spk_noise_factor = 2 * SPK_NOISE_LEVEL * level / nlevels - SPK_NOISE_LEVEL;
#endif
		uint32_t adjusted_bc_numerator = INITIATOR_BC_NUMERATOR + spk_noise_factor;
		val %= INITIATOR_DENOMINATOR;
		if (val < adjusted_bc_numerator) return 1;
		val -= adjusted_bc_numerator;
		if (val < adjusted_bc_numerator) return 2;
		val -= adjusted_bc_numerator;
#if SPK_NOISE_LEVEL == 0
		if (val < INITIATOR_A_NUMERATOR) return 0;
#else
		if (val < INITIATOR_A_NUMERATOR * (INITIATOR_DENOMINATOR - 2 * INITIATOR_BC_NUMERATOR) / (INITIATOR_DENOMINATOR - 2 * adjusted_bc_numerator)) return 0;
#endif
		return 3;
	}
#undef SPK_NOISE_LEVEL

	/* Make a single graph edge using a pre-set MRG state. */
	void make_one_edge(int64_t nverts, int level, mrg_state* st, EdgeType* result) const {
		int64_t base_src = 0, base_tgt = 0;
		while (nverts > 1) {
			int square = generate_4way_bernoulli(st, level, this->scale());
			int src_offset = square / 2;
			int tgt_offset = square % 2;
			assert (base_src <= base_tgt);
			if (base_src == base_tgt) {
				/* Clip-and-flip for undirected graph */
				if (src_offset > tgt_offset) {
					int temp = src_offset;
					src_offset = tgt_offset;
					tgt_offset = temp;
				}
			}
			nverts /= 2;
			++level;
			base_src += nverts * src_offset;
			base_tgt += nverts * tgt_offset;
		}
		result->set(
				this->scramble(base_src),
				this->scramble(base_tgt));
	}
};

#undef FAST_64BIT_ARITHMETIC

#endif /* GRAPH_GENERATOR_HPP_ */
