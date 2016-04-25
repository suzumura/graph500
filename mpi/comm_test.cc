/*
 * comm_test.cc
 *
 *  Created on: 2014/05/19
 *      Author: ueno
 */

#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif

// C includes
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

#include "parameters.h"
#include "utils_core.h"
#include "utils.hpp"
#include "fiber.hpp"
#include "abstract_comm.hpp"
#include "mpi_comm.hpp"
#include "fjmpi_comm.hpp"

enum { BUF_SIZE = 16*1024 };

struct TestBuffer : public CommunicationBuffer {
	int mem[BUF_SIZE];
	virtual void add(void* ptr__, int offset, int length) {
		for(int i = offset; i < length; ++i) {
			mem[i] = i;
		}
	}
	virtual void* base_object() {
		return this;
	}
	virtual int element_size() {
		return sizeof(int);
	}
	virtual void* pointer() {
		return mem;
	}
};

struct DataChecker : public Runnable {
	DataChecker(TestBuffer* buf__, memory::Pool<TestBuffer>* pool__)
		: buf_(buf__), pool_(pool__) { }
	virtual void run() {
		int length = buf_->length_;
		int* mem = buf_->mem;
		for(int i = 0; i < length; ++i) {
			if(mem[i] != i) {
				throw_exception("invalid data");
			}
		}
		pool_->free(buf_);
		delete this;
		debug("data check OK");
	}
	TestBuffer* buf_;
	memory::Pool<TestBuffer>* pool_;
};

struct TestBufferHandler : public AlltoallBufferHandler {
	virtual void free_buffer(CommunicationBuffer* buf__) {
		pool_->free(static_cast<TestBuffer*>(buf__->base_object()));
	}
	virtual int buffer_length() {
		return BUF_SIZE;
	}
	virtual MPI_Datatype data_type() {
		return MPI_INT;
	}
	virtual void finished() {
		fiber_->end_processing();
	}
	virtual CommunicationBuffer* alloc_buffer() {
		return this->pool_->get();
	}
	virtual void received(CommunicationBuffer* buf_, int src) {
		fiber_->submit(new DataChecker(static_cast<TestBuffer*>(buf_->base_object()), pool_), 1);
	}
	FiberManager* fiber_;
	memory::Pool<TestBuffer>* pool_;
};

void* time_bomp(void*) {
	sleep(30);
	*((int*)0) = 1;
	return NULL;
}

int main(int argc, char **argv) {
	MPI_Init(&argc, &argv);
#if ENABLE_FJMPI_RDMA
	FJMPI_Rdma_init();
#endif
	MPI_Comm_size(MPI_COMM_WORLD, &mpi.size_);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
	double start = MPI_Wtime();

	int send_count = atoi(argv[1]);

	//pthread_t thread;
	//pthread_create(&thread, NULL, time_bomp, NULL);

	{
		TestBufferHandler handler;
		FiberManager fiber_man_;
		FJMpiAlltoallCommunicator<TestBuffer> alltoall_comm_;
		AsyncAlltoallManager comm_(&alltoall_comm_, &fiber_man_);
		AlltoallCommParameter parm(MPI_COMM_WORLD, 0, 4, &handler);
		if(mpi.isMaster()) printf("[%f] complete initialization\n", MPI_Wtime() - start);

		MPI_Barrier(MPI_COMM_WORLD);
		if(mpi.isMaster()) printf("[%f] after Barrier\n", MPI_Wtime() - start);

		handler.fiber_ = &fiber_man_;
		handler.pool_ = alltoall_comm_.get_allocator();
		int sub_comm = alltoall_comm_.reg_comm(parm);
		if(mpi.isMaster()) printf("[%f] after reg_comm\n", MPI_Wtime() - start);

		double s1, s2, s3;
		for(int i = 0; i < 10; ++i) {
			s1 = MPI_Wtime();
			comm_.prepare(sub_comm);
			fiber_man_.submit(&comm_, 0);

			s2 = MPI_Wtime();
			for(int p = 0; p < mpi.size_; ++p) {
				for(int c = 0; c < send_count; ++c) {
					comm_.send<false>(NULL, BUF_SIZE, p);
				}
				comm_.send_end(p);
			}
			fiber_man_.enter_processing();
			s3 = MPI_Wtime();
			MPI_Barrier(MPI_COMM_WORLD);
			if(mpi.isMaster()) printf("[%f] finished %d-th\n", MPI_Wtime() - start, i);
		}
		printf("[r:%d] finished. %f ms, %f ms\n", mpi.rank, (s2-s1)*1000, (s3-s2)*1000);
	}

#if ENABLE_FJMPI_RDMA
	FJMPI_Rdma_finalize();
#endif
	MPI_Finalize();
	return 0;
}

