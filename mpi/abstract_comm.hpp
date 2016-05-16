/*
 * abstract_comm.hpp
 *
 *  Created on: 2014/05/17
 *      Author: ueno
 */

#ifndef ABSTRACT_COMM_HPP_
#define ABSTRACT_COMM_HPP_

#include <limits.h>
#include "utils.hpp"
#include "fiber.hpp"

#define debug(...) debug_print(ABSCO, __VA_ARGS__)
class AlltoallBufferHandler {
public:
	virtual ~AlltoallBufferHandler() { }
	virtual void* get_buffer() = 0;
	virtual void add(void* buffer, void* data, int offset, int length) = 0;
	virtual void* clear_buffers() = 0;
	virtual void* second_buffer() = 0;
	virtual int max_size() = 0;
	virtual int buffer_length() = 0;
	virtual MPI_Datatype data_type() = 0;
	virtual int element_size() = 0;
	virtual void received(void* buf, int offset, int length, int from) = 0;
	virtual void finish() = 0;
};

class AsyncAlltoallManager {
	struct Buffer {
		void* ptr;
		int length;
	};

	struct PointerData {
		void* ptr;
		int length;
		int64_t header;
	};

	struct CommTarget {
		CommTarget()
			: reserved_size_(0)
			, filled_size_(0) {
			cur_buf.ptr = NULL;
			cur_buf.length = 0;
			pthread_mutex_init(&send_mutex, NULL);
		}
		~CommTarget() {
			pthread_mutex_destroy(&send_mutex);
		}

		pthread_mutex_t send_mutex;
		// monitor : send_mutex
		volatile int reserved_size_;
		volatile int filled_size_;
		Buffer cur_buf;
		std::vector<Buffer> send_data;
		std::vector<PointerData> send_ptr;
	};
public:
	AsyncAlltoallManager(MPI_Comm comm_, AlltoallBufferHandler* buffer_provider_)
		: comm_(comm_)
		, buffer_provider_(buffer_provider_)
		, scatter_(comm_)
	{
		CTRACER(AsyncA2A_construtor);
		MPI_Comm_size(comm_, &comm_size_);
		node_ = new CommTarget[comm_size_]();
		d_ = new DynamicDataSet();
		pthread_mutex_init(&d_->thread_sync_, NULL);
		buffer_size_ = buffer_provider_->buffer_length();
	}
	virtual ~AsyncAlltoallManager() {
		delete [] node_; node_ = NULL;
	}

	void prepare() {
		CTRACER(prepare);
		debug("prepare idx=%d", sub_comm);
		for(int i = 0; i < comm_size_; ++i) {
			node_[i].reserved_size_ = node_[i].filled_size_ = buffer_size_;
		}
	}

	/**
	 * Asynchronous send.
	 * When the communicator receive data, it will call fold_received(FoldCommBuffer*) function.
	 * To reduce the memory consumption, when the communicator detects stacked jobs,
	 * it also process the tasks in the fiber_man_ except the tasks that have the lowest priority (0).
	 * This feature realize the fixed memory consumption.
	 */
	void put(void* ptr, int length, int target)
	{
		CTRACER(comm_send);
		if(length == 0) {
			assert(length > 0);
			return ;
		}
		CommTarget& node = node_[target];

//#if ASYNC_COMM_LOCK_FREE
		do {
			int offset = __sync_fetch_and_add(&node.reserved_size_, length);
			if(offset > buffer_size_) {
				// wait
				while(node.reserved_size_ > buffer_size_) ;
				continue ;
			}
			else if(offset + length > buffer_size_) {
				// swap buffer
				assert (offset > 0);
				while(offset != node.filled_size_) ;
				flush(node);
				node.cur_buf.ptr = get_send_buffer(); // Maybe, this takes much time.
				// This order is important.
				offset = node.filled_size_ = 0;
				__sync_synchronize(); // membar
				node.reserved_size_ = length;
			}
			buffer_provider_->add(node.cur_buf.ptr, ptr, offset, length);
			__sync_fetch_and_add(&node.filled_size_, length);
			break;
		} while(true);
// #endif
	}

	void put_ptr(void* ptr, int length, int64_t header, int target) {
		CommTarget& node = node_[target];
		PointerData data = { ptr, length, header };

		pthread_mutex_lock(&node.send_mutex);
		node.send_ptr.push_back(data);
		pthread_mutex_unlock(&node.send_mutex);
	}

	void run_with_ptr() {
		PROF(profiling::TimeKeeper tk_all);
		int es = buffer_provider_->element_size();
		int max_size = buffer_provider_->max_size() / (es * comm_size_);
		VERVOSE(last_send_size_ = 0);
		VERVOSE(last_recv_size_ = 0);

		const int MINIMUM_POINTER_SPACE = 40;

		for(int loop = 0; ; ++loop) {
			USER_START(a2a_merge);
#pragma omp parallel
			{
				int* counts = scatter_.get_counts();
#pragma omp for schedule(static)
				for(int i = 0; i < comm_size_; ++i) {
					CommTarget& node = node_[i];
					flush(node);
					for(int b = 0; b < (int)node.send_data.size(); ++b) {
						counts[i] += node.send_data[b].length;
					}
					for(int b = 0; b < (int)node.send_ptr.size(); ++b) {
						PointerData& buffer = node.send_ptr[b];
						int length = buffer.length;
						if(length == 0) continue;

						int size = length + 3;
						if(counts[i] + size >= max_size) {
							counts[i] = max_size;
							break;
						}

						counts[i] += size;
						if(counts[i] + MINIMUM_POINTER_SPACE >= max_size) {
							// too small space
							break;
						}
					}
				} // #pragma omp for schedule(static)
			}

			scatter_.sum();

			if(loop > 0) {
				int has_data = (scatter_.get_send_count() > 0);
				MPI_Allreduce(MPI_IN_PLACE, &has_data, 1, MPI_INT, MPI_LOR, comm_);
				if(has_data == 0) break;
			}

#pragma omp parallel
			{
				int* offsets = scatter_.get_offsets();
				uint8_t* dst = (uint8_t*)buffer_provider_->second_buffer();
#pragma omp for schedule(static)
				for(int i = 0; i < comm_size_; ++i) {
					CommTarget& node = node_[i];
					int& offset = offsets[i];
					int count = 0;
					for(int b = 0; b < (int)node.send_data.size(); ++b) {
						Buffer buffer = node.send_data[b];
						void* ptr = buffer.ptr;
						int length = buffer.length;
						memcpy(dst + offset * es, ptr, length * es);
						offset += length;
						count += length;
					}
					for(int b = 0; b < (int)node.send_ptr.size(); ++b) {
						PointerData& buffer = node.send_ptr[b];
						int64_t* ptr = (int64_t*)buffer.ptr;
						int length = buffer.length;
						if(length == 0) continue;

						int size = length + 3;
						if(count + size >= max_size) {
							length = max_size - count - 3;
							count = max_size;
						}
						else {
							count += size;
						}
						uint32_t* dst_ptr = (uint32_t*)&dst[offset * es];
						dst_ptr[0] = (buffer.header >> 32) | 0x80000000u | 0x40000000u;
						dst_ptr[1] = (uint32_t)buffer.header;
						dst_ptr[2] = length;
						dst_ptr += 3;
						for(int i = 0; i < length; ++i) {
							dst_ptr[i] = ptr[i] & 0x7FFFFFFF;
						}
						offset += 3 + length;

						buffer.length -= length;
						buffer.ptr = (int64_t*)buffer.ptr + length;

						if(count + MINIMUM_POINTER_SPACE >= max_size) break;
					}
					node.send_data.clear();
				} // #pragma omp for schedule(static)
			} // #pragma omp parallel
			USER_END(a2a_merge);

			void* sendbuf = buffer_provider_->second_buffer();
			void* recvbuf = buffer_provider_->clear_buffers();
			MPI_Datatype type = buffer_provider_->data_type();
			int recvbufsize = buffer_provider_->max_size();
			PROF(merge_time_ += tk_all);
			USER_START(a2a_comm);
			VERVOSE(if(loop > 0 && mpi.isMaster()) print_with_prefix("Alltoall with pointer (Again)"));
			scatter_.alltoallv(sendbuf, recvbuf, type, recvbufsize);
			PROF(comm_time_ += tk_all);
			USER_END(a2a_comm);

			VERVOSE(last_send_size_ += scatter_.get_send_count() * es);
			VERVOSE(last_recv_size_ += scatter_.get_recv_count() * es);

			int* recv_offsets = scatter_.get_recv_offsets();

#pragma omp parallel for
			for(int i = 0; i < comm_size_; ++i) {
				int offset = recv_offsets[i];
				int length = recv_offsets[i+1] - offset;
				buffer_provider_->received(recvbuf, offset, length, i);
			}
			PROF(recv_proc_time_ += tk_all);

			buffer_provider_->finish();
			PROF(recv_proc_large_time_ += tk_all);
		}

		// clear
		for(int i = 0; i < comm_size_; ++i) {
			CommTarget& node = node_[i];
			node.send_ptr.clear();
		}

	}

	void run() {
		// merge
		PROF(profiling::TimeKeeper tk_all);
		int es = buffer_provider_->element_size();
		VERVOSE(last_send_size_ = 0);
		VERVOSE(last_recv_size_ = 0);
		USER_START(a2a_merge);
#pragma omp parallel
		{
			int* counts = scatter_.get_counts();
#pragma omp for schedule(static)
			for(int i = 0; i < comm_size_; ++i) {
				CommTarget& node = node_[i];
				flush(node);
				for(int b = 0; b < (int)node.send_data.size(); ++b) {
					counts[i] += node.send_data[b].length;
				}
			} // #pragma omp for schedule(static)
		}

		scatter_.sum();

#pragma omp parallel
		{
			int* offsets = scatter_.get_offsets();
			uint8_t* dst = (uint8_t*)buffer_provider_->second_buffer();
#pragma omp for schedule(static)
			for(int i = 0; i < comm_size_; ++i) {
				CommTarget& node = node_[i];
				int& offset = offsets[i];
				for(int b = 0; b < (int)node.send_data.size(); ++b) {
					Buffer buffer = node.send_data[b];
					void* ptr = buffer.ptr;
					int length = buffer.length;
					memcpy(dst + offset * es, ptr, length * es);
					offset += length;
				}
				node.send_data.clear();
			} // #pragma omp for schedule(static)
		} // #pragma omp parallel
		USER_END(a2a_merge);

		void* sendbuf = buffer_provider_->second_buffer();
		void* recvbuf = buffer_provider_->clear_buffers();
		MPI_Datatype type = buffer_provider_->data_type();
		int recvbufsize = buffer_provider_->max_size();
		PROF(merge_time_ += tk_all);
		USER_START(a2a_comm);
		scatter_.alltoallv(sendbuf, recvbuf, type, recvbufsize);
		PROF(comm_time_ += tk_all);
		USER_END(a2a_comm);

		VERVOSE(last_send_size_ = scatter_.get_send_count() * es);
		VERVOSE(last_recv_size_ = scatter_.get_recv_count() * es);

		int* recv_offsets = scatter_.get_recv_offsets();

#pragma omp parallel for schedule(dynamic,1)
		for(int i = 0; i < comm_size_; ++i) {
			int offset = recv_offsets[i];
			int length = recv_offsets[i+1] - offset;
			buffer_provider_->received(recvbuf, offset, length, i);
		}
		PROF(recv_proc_time_ += tk_all);
	}
#if PROFILING_MODE
	void submit_prof_info(int level, bool with_ptr) {
		merge_time_.submit("merge a2a data", level);
		comm_time_.submit("a2a comm", level);
		recv_proc_time_.submit("proc recv data", level);
		if(with_ptr) {
			recv_proc_large_time_.submit("proc recv large data", level);
		}
		VERVOSE(profiling::g_pis.submitCounter(last_send_size_, "a2a send data", level);)
		VERVOSE(profiling::g_pis.submitCounter(last_recv_size_, "a2a recv data", level);)
	}
#endif
#if VERVOSE_MODE
	int get_last_send_size() { return last_send_size_; }
#endif
private:

	struct DynamicDataSet {
		// lock topology
		// FoldNode::send_mutex -> thread_sync_
		pthread_mutex_t thread_sync_;
	} *d_;

	MPI_Comm comm_;

	int buffer_size_;
	int comm_size_;

	int node_list_length_;
	CommTarget* node_;
	AlltoallBufferHandler* buffer_provider_;
	ScatterContext scatter_;

	PROF(profiling::TimeSpan merge_time_);
	PROF(profiling::TimeSpan comm_time_);
	PROF(profiling::TimeSpan recv_proc_time_);
	PROF(profiling::TimeSpan recv_proc_large_time_);
	VERVOSE(int last_send_size_);
	VERVOSE(int last_recv_size_);

	void flush(CommTarget& node) {
		if(node.cur_buf.ptr != NULL) {
			node.cur_buf.length = node.filled_size_;
			node.send_data.push_back(node.cur_buf);
			node.cur_buf.ptr = NULL;
		}
	}

	void* get_send_buffer() {
		CTRACER(get_send_buffer);
		pthread_mutex_lock(&d_->thread_sync_);
		void* ret = buffer_provider_->get_buffer();
		pthread_mutex_unlock(&d_->thread_sync_);
		return ret;
	}
};

// Allgather

class MpiCompletionHandler {
public:
	virtual ~MpiCompletionHandler() { }
	virtual void complete(MPI_Status* status) = 0;
};

class MpiRequestManager {
public:
	MpiRequestManager(int MAX_REQUESTS)
		: MAX_REQUESTS(MAX_REQUESTS)
		, finish_count(0)
		, reqs(new MPI_Request[MAX_REQUESTS])
		, handlers(new MpiCompletionHandler*[MAX_REQUESTS])
	{
		for(int i = 0; i < MAX_REQUESTS; ++i) {
			reqs[i] = MPI_REQUEST_NULL;
			empty_list.push_back(i);
		}
	}
	~MpiRequestManager() {
		delete [] reqs; reqs = NULL;
		delete [] handlers; handlers = NULL;
	}
	MPI_Request* submit_handler(MpiCompletionHandler* handler) {
		if(empty_list.size() == 0) {
			fprintf(IMD_OUT, "No more empty MPI requests...\n");
			throw "No more empty MPI requests...";
		}
		int empty = empty_list.back();
		empty_list.pop_back();
		handlers[empty] = handler;
		return &reqs[empty];
	}
	void finished() {
		--finish_count;
	}
	void run(int finish_count__) {
		finish_count += finish_count__;

		while(finish_count > 0) {
			if(empty_list.size() == MAX_REQUESTS) {
				fprintf(IMD_OUT, "Error: No active request\n");
				throw "Error: No active request";
			}
			int index;
			MPI_Status status;
			MPI_Waitany(MAX_REQUESTS, reqs, &index, &status);
			if(index == MPI_UNDEFINED) {
				fprintf(IMD_OUT, "MPI_Waitany returns MPI_UNDEFINED ...\n");
				throw "MPI_Waitany returns MPI_UNDEFINED ...";
			}
			MpiCompletionHandler* handler = handlers[index];
			reqs[index] = MPI_REQUEST_NULL;
			empty_list.push_back(index);

			handler->complete(&status);
		}
	}

private:
	int MAX_REQUESTS;
	int finish_count;
	MPI_Request *reqs;
	MpiCompletionHandler** handlers;
	std::vector<int> empty_list;
};

template <typename T>
class AllgatherHandler : public MpiCompletionHandler {
public:
	AllgatherHandler() { }
	virtual ~AllgatherHandler() { }

	void start(MpiRequestManager* req_man_, T *buffer_, int* count_, int* offset_, MPI_Comm comm_,
			int rank_, int size_, int left_, int right_, int tag_)
	{
		req_man = req_man_;
		buffer = buffer_;
		count = count_;
		offset = offset_;
		comm = comm_;
		rank = rank_;
		size = size_;
		left = left_;
		right = right_;
		tag = tag_;

		current = 1;
		l_sendidx = rank;
		l_recvidx = (rank + size + 1) % size;
		r_sendidx = rank;
		r_recvidx = (rank + size - 1) % size;

		next();
	}

	virtual void complete(MPI_Status* status) {
		if(++complete_count == 4) {
			next();
		}
	}

private:
	MpiRequestManager* req_man;

	T *buffer;
	int *count;
	int *offset;
	MPI_Comm comm;
	int rank;
	int size;
	int left;
	int right;
	int tag;

	int current;
	int l_sendidx;
	int l_recvidx;
	int r_sendidx;
	int r_recvidx;
	int complete_count;

	void next() {
		if(current >= size) {
			req_man->finished();
			return ;
		}

		if(l_sendidx >= size) l_sendidx -= size;
		if(l_recvidx >= size) l_recvidx -= size;
		if(r_sendidx < 0) r_sendidx += size;
		if(r_recvidx < 0) r_recvidx += size;

		int l_send_off = offset[l_sendidx];
		int l_send_cnt = count[l_sendidx] / 2;
		int l_recv_off = offset[l_recvidx];
		int l_recv_cnt = count[l_recvidx] / 2;

		int r_send_off = offset[r_sendidx] + count[r_sendidx] / 2;
		int r_send_cnt = count[r_sendidx] - count[r_sendidx] / 2;
		int r_recv_off = offset[r_recvidx] + count[r_recvidx] / 2;
		int r_recv_cnt = count[r_recvidx] - count[r_recvidx] / 2;

		MPI_Irecv(&buffer[l_recv_off], l_recv_cnt, MpiTypeOf<T>::type,
				right, tag, comm, req_man->submit_handler(this));
		MPI_Irecv(&buffer[r_recv_off], r_recv_cnt, MpiTypeOf<T>::type,
				left, tag, comm, req_man->submit_handler(this));
		MPI_Isend(&buffer[l_send_off], l_send_cnt, MpiTypeOf<T>::type,
				left, tag, comm, req_man->submit_handler(this));
		MPI_Isend(&buffer[r_send_off], r_send_cnt, MpiTypeOf<T>::type,
				right, tag, comm, req_man->submit_handler(this));

		++current;
		++l_sendidx;
		++l_recvidx;
		--r_sendidx;
		--r_recvidx;

		complete_count = 0;
	}
};

template <typename T>
class AllgatherStep1Handler : public MpiCompletionHandler {
public:
	AllgatherStep1Handler() { }
	virtual ~AllgatherStep1Handler() { }

	void start(MpiRequestManager* req_man_, T *buffer_, int* count_, int* offset_,
			COMM_2D comm_, int unit_x_, int unit_y_, int steps_, int tag_)
	{
		req_man = req_man_;
		buffer = buffer_;
		count = count_;
		offset = offset_;
		comm = comm_;
		unit_x = unit_x_;
		unit_y = unit_y_;
		steps = steps_;
		tag = tag_;

		current = 1;

		send_to = get_rank(-1);
		recv_from = get_rank(1);

		next();
	}

	virtual void complete(MPI_Status* status) {
		if(++complete_count == 2) {
			next();
		}
	}

private:
	MpiRequestManager* req_man;

	T *buffer;
	int *count;
	int *offset;
	COMM_2D comm;
	int unit_x;
	int unit_y;
	int steps;
	int tag;

	int send_to;
	int recv_from;

	int current;
	int complete_count;

	int get_rank(int diff) {
		int pos_x = (comm.rank_x + unit_x * diff + comm.size_x) % comm.size_x;
		int pos_y = (comm.rank_y + unit_y * diff + comm.size_y) % comm.size_y;
		return comm.rank_map[pos_x + pos_y * comm.size_x];
	}

	void next() {
		if(current >= steps) {
			req_man->finished();
			return ;
		}

		int sendidx = get_rank(current - 1);
		int recvidx = get_rank(current);

		int send_off = offset[sendidx];
		int send_cnt = count[sendidx];
		int recv_off = offset[recvidx];
		int recv_cnt = count[recvidx];

		MPI_Irecv(&buffer[recv_off], recv_cnt, MpiTypeOf<T>::type,
				recv_from, tag, comm.comm, req_man->submit_handler(this));
		MPI_Isend(&buffer[send_off], send_cnt, MpiTypeOf<T>::type,
				send_to, tag, comm.comm, req_man->submit_handler(this));

		++current;
		complete_count = 0;
	}
};

template <typename T>
class AllgatherStep2Handler : public MpiCompletionHandler {
public:
	AllgatherStep2Handler() { }
	virtual ~AllgatherStep2Handler() { }

	void start(MpiRequestManager* req_man_, T *buffer_, int* count_, int* offset_,
			COMM_2D comm_, int unit_x_, int unit_y_, int steps_, int width_, int tag_)
	{
		req_man = req_man_;
		buffer = buffer_;
		count = count_;
		offset = offset_;
		comm = comm_;
		unit_x = unit_x_;
		unit_y = unit_y_;
		steps = steps_;
		width = width_;
		tag = tag_;

		current = 1;

		send_to = get_rank(-1, 0);
		recv_from = get_rank(1, 0);

		next();
	}

	virtual void complete(MPI_Status* status) {
		if(++complete_count == width*2) {
			next();
		}
	}

private:
	MpiRequestManager* req_man;

	T *buffer;
	int *count;
	int *offset;
	COMM_2D comm;
	int unit_x;
	int unit_y;
	int steps;
	int width;
	int tag;

	int send_to;
	int recv_from;

	int current;
	int complete_count;

	int get_rank(int step_diff, int idx) {
		int pos_x = (comm.rank_x + unit_x * step_diff + (!unit_x * idx) + comm.size_x) % comm.size_x;
		int pos_y = (comm.rank_y + unit_y * step_diff + (!unit_y * idx) + comm.size_y) % comm.size_y;
		return comm.rank_map[pos_x + pos_y * comm.size_x];
	}

	void next() {
		if(current >= steps) {
			req_man->finished();
			return ;
		}

		for(int i = 0; i < width; ++i) {
			int sendidx = get_rank(current - 1, i);
			int recvidx = get_rank(current, i);

			int send_off = offset[sendidx];
			int send_cnt = count[sendidx];
			int recv_off = offset[recvidx];
			int recv_cnt = count[recvidx];

			MPI_Irecv(&buffer[recv_off], recv_cnt, MpiTypeOf<T>::type,
					recv_from, tag, comm.comm, req_man->submit_handler(this));
			MPI_Isend(&buffer[send_off], send_cnt, MpiTypeOf<T>::type,
					send_to, tag, comm.comm, req_man->submit_handler(this));
		}

		++current;
		complete_count = 0;
	}
};

template <typename T>
void my_allgatherv_2d(T *sendbuf, int send_count, T *recvbuf, int* recv_count, int* recv_offset, COMM_2D comm)
{
	// copy own data
	memcpy(&recvbuf[recv_offset[comm.rank]], sendbuf, sizeof(T) * send_count);

	if(mpi.isMultiDimAvailable == false) {
		MpiRequestManager req_man(8);
		AllgatherHandler<T> handler;
		int size; MPI_Comm_size(comm.comm, &size);
		int rank; MPI_Comm_rank(comm.comm, &rank);
		int left = (rank + size - 1) % size;
		int right = (rank + size + 1) % size;
		handler.start(&req_man, recvbuf, recv_count, recv_offset, comm.comm,
				rank, size, left, right, PRM::MY_EXPAND_TAG1);
		req_man.run(1);
		return ;
	}

	//MPI_Allgatherv(sendbuf, send_count, MpiTypeOf<T>::type, recvbuf, recv_count, recv_offset, MpiTypeOf<T>::type, comm.comm);
	//return;

	MpiRequestManager req_man((comm.size_x + comm.size_y)*4);
	int split_count[4][comm.size];
	int split_offset[4][comm.size];

	for(int s = 0; s < 4; ++s) {
		for(int i = 0; i < comm.size; ++i) {
			int max = recv_count[i];
			int split = (max + 3) / 4;
			int start = recv_offset[i] + std::min(max, split * s);
			int end = recv_offset[i] + std::min(max, split * (s+1));
			split_count[s][i] = end - start;
			split_offset[s][i] = start;
		}
	}

	{
		AllgatherStep1Handler<T> handler[4];
		handler[0].start(&req_man, recvbuf, split_count[0], split_offset[0], comm, 1, 0, comm.size_x, PRM::MY_EXPAND_TAG1);
		handler[1].start(&req_man, recvbuf, split_count[1], split_offset[1], comm,-1, 0, comm.size_x, PRM::MY_EXPAND_TAG1);
		handler[2].start(&req_man, recvbuf, split_count[2], split_offset[2], comm, 0, 1, comm.size_y, PRM::MY_EXPAND_TAG2);
		handler[3].start(&req_man, recvbuf, split_count[3], split_offset[3], comm, 0,-1, comm.size_y, PRM::MY_EXPAND_TAG2);
		req_man.run(4);
	}
	{
		AllgatherStep2Handler<T> handler[4];
		handler[0].start(&req_man, recvbuf, split_count[0], split_offset[0], comm, 0, 1, comm.size_y, comm.size_x, PRM::MY_EXPAND_TAG1);
		handler[1].start(&req_man, recvbuf, split_count[1], split_offset[1], comm, 0,-1, comm.size_y, comm.size_x, PRM::MY_EXPAND_TAG1);
		handler[2].start(&req_man, recvbuf, split_count[2], split_offset[2], comm, 1, 0, comm.size_x, comm.size_y, PRM::MY_EXPAND_TAG2);
		handler[3].start(&req_man, recvbuf, split_count[3], split_offset[3], comm,-1, 0, comm.size_x, comm.size_y, PRM::MY_EXPAND_TAG2);
		req_man.run(4);
	}
}

template <typename T>
void my_allgather_2d(T *sendbuf, int count, T *recvbuf, COMM_2D comm)
{
	memcpy(&recvbuf[count * comm.rank], sendbuf, sizeof(T) * count);
	int recv_count[comm.size];
	int recv_offset[comm.size+1];
	recv_offset[0] = 0;
	for(int i = 0; i < comm.size; ++i) {
		recv_count[i] = count;
		recv_offset[i+1] = recv_offset[i] + count;
	}
	my_allgatherv_2d(sendbuf, count, recvbuf, recv_count, recv_offset, comm);
}

#undef debug

#endif /* ABSTRACT_COMM_HPP_ */
