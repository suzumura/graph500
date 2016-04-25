/*
 * mpi_comm.hpp
 *
 *  Created on: 2014/05/17
 *      Author: ueno
 */

#ifndef MPI_COMM_HPP_
#define MPI_COMM_HPP_

#include "abstract_comm.hpp"

#if 0
#define debug(...) debug_print(MPICO, __VA_ARGS__)
class MpiAlltoallCommunicatorBase : public AlltoallCommunicator {
	struct CommTarget {
		CommTarget()
			: send_buf(NULL)
			, recv_buf(NULL) {
		}

		std::deque<CommunicationBuffer*> send_queue;
		CommunicationBuffer* send_buf;
		CommunicationBuffer* recv_buf;
	};
public:
	MpiAlltoallCommunicatorBase() {
		CTRACER(MPIA2A_constructor);
		node_list_length_ = 0;
		node_ = NULL;
		mpi_reqs_ = NULL;
		num_pending_send = 0;
	}
	virtual ~MpiAlltoallCommunicatorBase() {
		CTRACER(MPIA2A_destructor);
		delete [] node_;
		delete [] mpi_reqs_;
	}
	virtual void send(CommunicationBuffer* data, int target) {
		CTRACER(MPIA2A_send);
		node_[target].send_queue.push_back(data);
		++num_pending_send;
		set_send_buffer(target);
	}
	virtual AlltoallSubCommunicator reg_comm(AlltoallCommParameter parm) {
		CTRACER(MPIA2A_reg_comm);
		int idx = handlers_.size();
		handlers_.push_back(parm);
		int comm_size;
		MPI_Comm_size(parm.base_communicator, &comm_size);
		node_list_length_ = std::max(comm_size, node_list_length_);
		return idx;
	}
	virtual AlltoallBufferHandler* begin(AlltoallSubCommunicator sub_comm) {
		CTRACER(MPIA2A_begin);
		AlltoallCommParameter active = handlers_[sub_comm];
		comm_ = active.base_communicator;
		tag_ = active.tag;
		handler_ = active.handler;
		data_type_ = handler_->data_type();
		MPI_Comm_size(comm_, &comm_size_);
		initialized_ = false;
		num_recv_active = num_send_active = comm_size_;
		paused = false;

		if(node_ == NULL) {
			node_ = new CommTarget[node_list_length_]();
			mpi_reqs_ = new MPI_Request[node_list_length_*REQ_TOTAL]();
			for(int i = 0; i < node_list_length_; ++i) {
				for(int k = 0; k < REQ_TOTAL; ++k) {
					mpi_reqs_[REQ_TOTAL*i + k] = MPI_REQUEST_NULL;
				}
			}
		}

		PROF(num_send_count = 0);

		debug("begin idx=%d", sub_comm);
		return handler_;
	}
	//! @return finished
	virtual void* probe() {
		CTRACER(MPIA2A_begin_probe);
		if(num_recv_active == 0 && num_send_active == 0) {
			return NULL;
		}

		if(initialized_ == false) {
			CTRACER(MPIA2A_initialize);
			initialized_ = true;
			for(int i = 0; i < comm_size_; ++i) {
				CTRACER(MpiAlltoallCommunicatorBase);
				CommTarget& node = node_[i];
				assert (node.recv_buf == NULL);
				node.recv_buf = handler_->alloc_buffer();
				assert (node.recv_buf != NULL);
				set_recv_buffer(node.recv_buf, i, &mpi_reqs_[REQ_TOTAL*i + REQ_RECV]);
			}
		}

		int index;
		int flag;
		MPI_Status status = {0};
		MPI_Testany(comm_size_ * (int)REQ_TOTAL, mpi_reqs_, &index, &flag, &status);

		if(flag != 0 && index != MPI_UNDEFINED) {
			CTRACER(MPIA2A_request_completed);
			const int src_c = index/REQ_TOTAL;
			const MPI_REQ_INDEX req_kind = (MPI_REQ_INDEX)(index%REQ_TOTAL);
			const bool b_send = (req_kind == REQ_SEND);

			CommTarget& node = node_[src_c];
			CommunicationBuffer* buf;
			if(b_send) {
				buf = node.send_buf; node.send_buf = NULL;
			}
			else {
				buf = node.recv_buf; node.recv_buf = NULL;
			}

			assert (mpi_reqs_[index] == MPI_REQUEST_NULL);
			mpi_reqs_[index] = MPI_REQUEST_NULL;

			if(req_kind == REQ_RECV) {
				MPI_Get_count(&status, data_type_, &buf->length_);
			}

			bool completion_message = (buf->length_ == 0);
			// complete
			if(b_send) {
				// send buffer
				handler_->free_buffer(buf);
				if(completion_message) {
					// sent fold completion
					debug("send complete to=%d (finished)", src_c);
					--num_send_active;
				}
				else {
					set_send_buffer(src_c);
				}
			}
			else {
				// recv buffer
				if(completion_message) {
					// received fold completion
					debug("recv complete from=%d (finished)", src_c);
					--num_recv_active;
					handler_->free_buffer(buf);
				}
				else {
					// set new buffer for next receiving
					recv_stv.push_back(src_c);

					handler_->received(buf, src_c);
				}
			}

			// process recv starves
			while(recv_stv.size() > 0) {
				CTRACER(MPIA2A_process_recv_starves);
				int target = recv_stv.front();
				CommTarget& node = node_[target];
				assert (node.recv_buf == NULL);
				node.recv_buf = handler_->alloc_buffer();
				if(node.recv_buf == NULL) break;
				set_recv_buffer(node.recv_buf, target, &mpi_reqs_[REQ_TOTAL*target + REQ_RECV]);
				recv_stv.pop_front();
			}
		}

		if(num_recv_active == 0 && num_send_active == 0) {
			CTRACER(MpiAlltoallCommunicatorBase);
			// finished
			debug("finished");
			handler_->finished();
		}

		return NULL;
	}
	virtual bool is_finished() {
		return (num_recv_active == 0 && num_send_active == 0);
	}
	virtual int get_comm_size() {
		return comm_size_;
	}
	virtual void pause() {
		paused = true;
	}
	virtual void restart() {
		paused = false;
		for(int i = 0; i < comm_size_; ++i) {
			set_send_buffer(i);
		}
	}
#ifndef NDEBUG
	bool check_num_send_buffer() { return (num_pending_send == 0); }
#endif
protected:
	PROF(int num_send_count);

private:

	enum MPI_REQ_INDEX {
		REQ_SEND = 0,
		REQ_RECV = 1,
		REQ_TOTAL = 2,
	};

	std::vector<AlltoallCommParameter> handlers_;
	MPI_Comm comm_;
	int tag_;
	AlltoallBufferHandler* handler_;
	MPI_Datatype data_type_;
	int comm_size_;
	bool initialized_;

	int node_list_length_;
	CommTarget* node_;
	std::deque<int> recv_stv;
	MPI_Request* mpi_reqs_;

	int num_recv_active;
	int num_send_active;
	int num_pending_send;
	bool paused;

	void set_send_buffer(int target) {
		CTRACER(MPIA2A_set_send_buffer);
		// do not send when this is paused
		if(paused) return;
		CommTarget& node = node_[target];
		if(node.send_buf) {
			// already sending
			return ;
		}
		if(node.send_queue.size() > 0) {
			CommunicationBuffer* buf = node.send_buf = node.send_queue.front();
			node.send_queue.pop_front();
			MPI_Request* req = &mpi_reqs_[REQ_TOTAL*target + REQ_SEND];

			MPI_Isend(buf->pointer(), buf->length_, data_type_,
					target, tag_, comm_, req);

			--num_pending_send;
		}
	}

	void set_recv_buffer(CommunicationBuffer* buf, int target, MPI_Request* req) {
		CTRACER(MPIA2A_set_recv_buffer);
		MPI_Irecv(buf->pointer(), handler_->buffer_length(), data_type_,
				target, tag_, comm_, req);
	}

};

template <typename T>
class MpiAlltoallCommunicator : public MpiAlltoallCommunicatorBase {
public:
	MpiAlltoallCommunicator() : MpiAlltoallCommunicatorBase() { }
	memory::Pool<T>* get_allocator() {
		return &pool_;
	}
#if PROFILING_MODE
	void submit_prof_info(int number) {
		profiling::g_pis.submitCounter(pool_.num_extra_buffer_, "num_extra_buffer_", number);
		profiling::g_pis.submitCounter(num_send_count, "num_send_count", number);
	}
#endif
private:

	class CommBufferPool : public memory::ConcurrentPool<T> {
	public:
		CommBufferPool()
			: memory::ConcurrentPool<T>()
		{
			num_extra_buffer_ = 0;
		}

		int num_extra_buffer_;
	protected:
		virtual T* allocate_new() {
			PROF(__sync_fetch_and_add(&num_extra_buffer_, 1));
			return new (page_aligned_xmalloc(sizeof(T))) T();
		}
	};

	CommBufferPool pool_;
};
#undef debug
#endif // #if 0

#if 0
#define debug(...) debug_print(MPIBU, __VA_ARGS__)
template <int NBUF>
class MpiBottomUpSubstepComm :  public AsyncCommHandler {
public:
	MpiBottomUpSubstepComm(MPI_Comm mpi_comm__) {
		mpi_comm = mpi_comm__;
		int size, rank;
		MPI_Comm_size(mpi_comm__, &size);
		MPI_Comm_rank(mpi_comm__, &rank);
		int size_cmask = size - 1;
		send_to = (rank - 1) & size_cmask;
		recv_from = (rank + 1) & size_cmask;
		total_phase = size*2;
		for(int i = 0; i < DNBUF+4; ++i) {
			req[i] = MPI_REQUEST_NULL;
		}
	}
	virtual ~MpiBottomUpSubstepComm() {
	}
	void register_memory(void* memory, int64_t size) {
	}
	template <typename T>
	void begin(T** buffers__, T** last_data, int buffer_width__, int recv_start) {
		type = MpiTypeOf<T>::type;
		buffer_width = buffer_width__;
		for(int i = 0; i < NBUF; ++i) {
			buffers[i] = buffers__[i];
		}
		for(int i = 0; i < 2; ++i) {
			buffers[NBUF+i] = last_data[i];
		}
		recv_offset = recv_start;
		next_recv = 0;
		proc_cnt = 0;
		finished = false;
		next_send = 0;
		send_complete = 0;
		initialized = false;
		debug("begin");
	}
	int advance(int send_size) {
		VT_TRACER("bu_comm_adv");
		debug("advance begin <- %d", send_size);
		data_size[proc_cnt & BUFMASK] = send_size;
		__sync_synchronize();
		++proc_cnt;
		__sync_synchronize();
		while(next_recv + 1 < proc_cnt) sched_yield();
		__sync_synchronize();
		int recv_buf_idx = (proc_cnt + recv_offset - 2) & BUFMASK;
		debug("advance finished -> %d", data_size[recv_buf_idx]);
		return data_size[recv_buf_idx];
	}
	void finish(int* last_recv_size) {
		VT_TRACER("bu_comm_fin_wait");
		while(!finished) sched_yield();
		__sync_synchronize();
		if(last_recv_size) {
			last_recv_size[0] = data_size[NBUF+0];
			last_recv_size[1] = data_size[NBUF+1];
		}
	}
	virtual void probe(void* comm_data) {
		if(finished) return;

		if(initialized == false) {
			initialized = true;
			int n = std::min<int>(NBUF, recv_offset+total_phase-2);
			for(int i = recv_offset; i < n; ++i) {
				MPI_Request* req_ptr = this->req + i * 2 + 1;
				MPI_Irecv(buffers[i], buffer_width, type, this->recv_from, 0, this->mpi_comm, req_ptr);
			}
			// for the last data
			for(int i = 0; i < 2; ++i) {
				MPI_Request* req_ptr = this->req + DNBUF + i * 2 + 1;
				MPI_Irecv(buffers[NBUF+i], buffer_width, type, this->recv_from, 1, this->mpi_comm, req_ptr);
			}
			debug("initialized");
		}

		while(next_send < proc_cnt) {
			assert (next_send < total_phase);
			int buf_idx = next_send & BUFMASK;
			MPI_Request* req_ptr = this->req + buf_idx * 2;
			int tag = (next_send < total_phase-2) ? 0 : 1; // last data?
			MPI_Isend(buffers[buf_idx], data_size[buf_idx], type, this->send_to, tag, this->mpi_comm, req_ptr);
			VERVOSE(g_bu_list_comm += data_size[buf_idx] * sizeof(BitmapType));
			debug("send %d", data_size[buf_idx]);
			++next_send;
		}

		if(send_complete < next_send) { // is there any sending data?
			int buf_idx = send_complete & BUFMASK;
			MPI_Request* req_ptr = req + buf_idx * 2;
			int flag; MPI_Status status;
			MPI_Test(req_ptr, &flag, &status);
			if(flag) {
				// complete -> set recv buffer
				if(send_complete + NBUF < recv_offset+total_phase-2) {
					MPI_Irecv(buffers[buf_idx], buffer_width, type, this->recv_from, 0, this->mpi_comm, req_ptr+1);
				}
				debug("send complete");
				++send_complete;
			}
		}

		if(recv_offset+next_recv < send_complete + NBUF) {
			if(next_recv < total_phase) {
				int buf_idx;
				MPI_Request* req_ptr;
				int flag; MPI_Status status;
				if(next_recv < total_phase-2) {
					buf_idx = (next_recv + recv_offset) & BUFMASK;
				}
				else {
					// last data
					buf_idx = NBUF + next_recv - (total_phase-2);
				}
				req_ptr = req + buf_idx * 2 + 1;
				MPI_Test(req_ptr, &flag, &status);
				if(flag) {
					MPI_Get_count(&status, type, (int*)&data_size[buf_idx]);
					debug("recv complete %d", data_size[buf_idx]);
					++next_recv;
				}
			}
		}

		if(send_complete == total_phase && next_recv == total_phase) {
			debug("finished");
			finished = true;
		}
	}

protected:
	enum { DNBUF = NBUF*2, BUFMASK = NBUF-1 };
	MPI_Comm mpi_comm;
	int send_to, recv_from;
	int total_phase;
	void* buffers[NBUF+2];
	MPI_Request req[DNBUF+4];

	MPI_Datatype type;
	int recv_offset;
	int buffer_width;
	bool initialized;
	volatile int data_size[NBUF+2];
	volatile int next_recv;
	volatile int proc_cnt;
	volatile bool finished;
	int next_send;
	int send_complete;
};
#undef debug
#endif

#endif /* MPI_COMM_HPP_ */
