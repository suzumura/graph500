/*
 * fjmpi_comm.hpp
 *
 *  Created on: 2014/05/18
 *      Author: ueno
 */

#ifndef FJMPI_COMM_HPP_
#define FJMPI_COMM_HPP_
#if ENABLE_FJMPI_RDMA

#include "abstract_comm.hpp"
#include "utils.hpp"

#define debug(...) debug_print(FJA2A, __VA_ARGS__)
struct FJMPI_CQ : FJMPI_Rdma_cq {
	int nic;
	FJMPI_CQ(FJMPI_Rdma_cq& base__, int nic__) : FJMPI_Rdma_cq(base__), nic(nic__) { }
};

static int FJMPI_Local_nic[] = {
	FJMPI_RDMA_LOCAL_NIC0,
	FJMPI_RDMA_LOCAL_NIC1,
	FJMPI_RDMA_LOCAL_NIC2,
	FJMPI_RDMA_LOCAL_NIC3
};

static int FJMPI_Remote_nic[] = {
	FJMPI_RDMA_REMOTE_NIC0,
	FJMPI_RDMA_REMOTE_NIC1,
	FJMPI_RDMA_REMOTE_NIC2,
	FJMPI_RDMA_REMOTE_NIC3
};

class FJMpiAlltoallCommunicatorBase : public AlltoallCommunicator {
public:
	enum {
		MAX_SYSTEM_MEMORY = 16ll*1024*1024*1024,
	//	MAX_SYSTEM_MEMORY = 4ll*1024*1024*1024,
		MAX_RDMA_BUFFER = 256,
		RDMA_BUF_SIZE = MAX_SYSTEM_MEMORY / MAX_RDMA_BUFFER
	};
private:
	enum STATE_FLAG {
		MAX_FLYING_REQ = 4, // maximum is 14 due to the limitation of FJMPI
		INITIAL_RADDR_TABLE_SIZE = 4,
		SYSTEM_TAG = 0,
		FIRST_DATA_TAG = 1,
		FIRST_USER_TAG = FIRST_DATA_TAG + MAX_FLYING_REQ,

		INVALIDATED = 0,
		READY = 1,
		COMPLETE = 2,
	};

	struct BufferState {
		uint16_t state; // current state of the buffer
		uint16_t memory_id; // memory id of the RDMA buffer
		union {
			uint64_t offset; // offset to the buffer starting address
			uint64_t length; // length of the received data in bytes
		};
	};

	class RankMap {
		typedef std::pair<int,int> ElementType;
		struct Compare {
			bool operator()(const ElementType& l, int r) {
				return l.first < r;
			}
		};
	public:
		typedef std::vector<ElementType>::iterator iterator;

		iterator lower_bound(int pid) {
			return std::lower_bound(vec.begin(), vec.end(), pid, Compare());
		}

		bool match(iterator it, int pid) {
			if(it == vec.end() || it->first != pid) return false;
			return true;
		}

		void add(iterator it, int pid, int rank) {
			vec.insert(it, ElementType(pid, rank));
		}

	private:
		std::vector<ElementType> vec;
	};

	struct CommTarget {
		int proc_index;
		int put_flag;
		uint64_t remote_buffer_state;

		std::deque<CommunicationBuffer*> send_queue;

		CommunicationBuffer* send_buf[MAX_FLYING_REQ];
		CommunicationBuffer* recv_buf[MAX_FLYING_REQ];

		// take care the overflow of these counters
		unsigned int send_count; // the next buffer index is calculated from this value
		unsigned int recv_count; //   "
		unsigned int recv_complete_count;

		PROF(unsigned int start_send_count);

		CommTarget() {
			proc_index = put_flag = 0;
			remote_buffer_state = 0;
			for(int i = 0; i < MAX_FLYING_REQ; ++i) {
				send_buf[i] = NULL;
				recv_buf[i] = NULL;
			}
			send_count = recv_count = recv_complete_count = 0;
		}
	};

	struct InternalCommunicator {
		MPI_Comm base_communicator;
		AlltoallBufferHandler* handler;
		int size, rank;
		int num_nics_to_use;
		//! pid -> rank
		RankMap rank_map;
		//! index = rank in the base communicator
		std::vector<CommTarget> proc_info;
		volatile BufferState* local_buffer_state;
		int num_recv_active;
		int num_send_active;
		int num_pending_send;

		volatile BufferState& send_buffer_state(int rank, int idx) {
			return local_buffer_state[offset_of_send_buffer_state(rank, idx)];
		}

		volatile BufferState& recv_buffer_state(int rank, int idx) {
			return local_buffer_state[offset_of_recv_buffer_state(rank, idx)];
		}
	};

public:
	FJMpiAlltoallCommunicatorBase() {
		CTRACER(FJMpiAlltoallCommunicatorBase);
		c_ = NULL;
		fix_system_memory_ = false;
		remote_address_table_ = NULL;
		num_procs_ = 0;
		num_address_per_proc_ = 0;
		for(int i = 0; i < MAX_RDMA_BUFFER; ++i) {
			rdma_buffer_pointers_[i] = NULL;
			free_memory_ids_[i] = MAX_RDMA_BUFFER;
			local_address_table_[i] = 0;
		}
		MPI_Comm_group(MPI_COMM_WORLD, &world_group_);
		system_rdma_mem_size_ = 0;
	}
	virtual ~FJMpiAlltoallCommunicatorBase() {
		CTRACER(~FJMpiAlltoallCommunicatorBase);
		free(remote_address_table_); remote_address_table_ = NULL;
		for(int i = 0; i < MAX_RDMA_BUFFER; ++i) {
			if(rdma_buffer_pointers_[i] != NULL) {
				FJMPI_Rdma_dereg_mem(i);
				free(rdma_buffer_pointers_[i]); rdma_buffer_pointers_[i] = NULL;
			}
		}
		MPI_Group_free(&world_group_);
	}
	virtual void send(CommunicationBuffer* data, int target) {
		CTRACER(FJMpiAlltoallCommunicatorBase::send);
		c_->proc_info[target].send_queue.push_back(data);
		debug("send data queued to=%d, length=%d", pid_from_rank(target), data->length_);
		c_->num_pending_send++;
		set_send_buffer(target);
	}
	virtual AlltoallSubCommunicator reg_comm(AlltoallCommParameter parm) {
		CTRACER(FJMpiAlltoallCommunicatorBase::AlltoallSubCommunicator);
		if(fix_system_memory_) {
			throw_exception("reg_comm is called after data memory allocation");
		}
		int idx = internal_communicators_.size();
		internal_communicators_.push_back(InternalCommunicator());
		InternalCommunicator& c = internal_communicators_.back();

		// initialize c
		c.base_communicator = parm.base_communicator;
		c.handler = parm.handler;
		MPI_Comm_rank(c.base_communicator, &c.rank);
		MPI_Comm_size(c.base_communicator, &c.size);
		c.num_nics_to_use = parm.num_nics_to_use;
		c.proc_info.resize(c.size, CommTarget());

		if(c.num_nics_to_use > 4) {
			c.num_nics_to_use = 4;
		}

		int remote_nic = c.rank % c.num_nics_to_use;
		MPI_Group comm_group;
		MPI_Comm_group(c.base_communicator, &comm_group);
		int* ranks1 = new int[c.size];
		int* ranks2 = new int[c.size];
		for(int i = 0; i < c.size; ++i) {
			ranks1[i] = i;
		}
		MPI_Group_translate_ranks(comm_group, c.size, ranks1, world_group_, ranks2);

		for(int i = 0; i < c.size; ++i) {
			int pid = ranks2[i];
			int proc_idx;
			RankMap::iterator it = proc_index_map_.lower_bound(pid);
			if(proc_index_map_.match(it, pid)) {
				proc_idx = it->second;
			}
			else {
				proc_idx = proc_info_.size();
				proc_index_map_.add(it, pid, proc_idx);
				proc_info_.push_back(pid);
			}

			c.rank_map.add(c.rank_map.lower_bound(pid), pid, i);
			c.proc_info[i].proc_index = proc_idx;
			c.proc_info[i].put_flag = FJMPI_Local_nic[i % c.num_nics_to_use] |
					FJMPI_Remote_nic[remote_nic] | FJMPI_RDMA_IMMEDIATE_RETURN | FJMPI_RDMA_PATH0;
		}

		delete [] ranks1; ranks1 = NULL;
		delete [] ranks2; ranks2 = NULL;
		MPI_Group_free(&comm_group);

		c.local_buffer_state = NULL;
		c.num_recv_active = c.num_send_active = c.num_pending_send = 0;

		// allocate buffer state memory
		if(rdma_buffer_pointers_[0] == NULL) {
			initialize_rdma_buffer();
		}
		int offset = system_rdma_mem_size_;
		c.local_buffer_state = get_pointer<BufferState>(rdma_buffer_pointers_[0], offset);
		system_rdma_mem_size_ += sizeof(BufferState) * buffer_state_offset(c.size);

		if(system_rdma_mem_size_ > RDMA_BUF_SIZE) {
			throw_exception("no system RDMA memory");
		}

		// initialize buffer state
		for(int p = 0; p < buffer_state_offset(c.size); ++p) {
			c.local_buffer_state[p].state = INVALIDATED;
		}

		// get the remote buffer state address
		int* state_offset = new int[c.size];
		MPI_Allgather(&offset, 1, MpiTypeOf<int>::type,
				state_offset, 1, MpiTypeOf<int>::type, c.base_communicator);

		for(int i = 0; i < c.size; ++i) {
			int proc_index = c.proc_info[i].proc_index;
			c.proc_info[i].remote_buffer_state = get_remote_address(proc_index, 0, state_offset[i]);
		}

		delete [] state_offset; state_offset = NULL;

		debug("reg_comm idx=%d finished", idx);
		return idx;
	}
	virtual AlltoallBufferHandler* begin(AlltoallSubCommunicator sub_comm) {
		CTRACER(FJMpiAlltoallCommunicatorBase::begin);
		c_ = &internal_communicators_[sub_comm];
		c_->num_recv_active = c_->num_send_active = c_->size;
		paused = false;
#if PROFILING_MODE
		for(int i = 0; i < c_->size; ++i) {
			CommTarget& target = c_->proc_info[i];
			target.start_send_count = target.send_count;
		}
#endif
		debug("begin idx=%d", sub_comm);
		return c_->handler;
	}
	//! @return finished
	virtual void* probe() {
		CTRACER(FJMpiAlltoallCommunicatorBase::probe);

		user_cq.clear();

		if(c_->num_recv_active == 0 && c_->num_send_active == 0) {
			// already finished
			return &user_cq;
		}

		// process receive completion
		for(int p = 0; p < c_->size; ++p) {
			check_recv_completion(p);
			set_recv_buffer(p);
			set_send_buffer(p);
		}

		// process send completion
		int nics[4] = {
			FJMPI_RDMA_NIC0,
			FJMPI_RDMA_NIC1,
			FJMPI_RDMA_NIC2,
			FJMPI_RDMA_NIC3
		};
		for(int nic = 0; nic < c_->num_nics_to_use; ++nic) {
			CTRACER(probe_foreach_nic);
			while(true) {
				struct FJMPI_Rdma_cq cq;
				int ret = FJMPI_Rdma_poll_cq(nics[nic], &cq);
				if(ret == FJMPI_RDMA_NOTICE) {
					if(cq.tag == SYSTEM_TAG) {
						// nothing to do
						debug("SYSTEM_TAG completion");
					}
					else if(cq.tag < FIRST_USER_TAG){
						RankMap::iterator it = c_->rank_map.lower_bound(cq.pid);
						assert(c_->rank_map.match(it, cq.pid));
						int rank = it->second;
						int buf_idx = cq.tag - FIRST_DATA_TAG;
						CommunicationBuffer*& comm_buf = c_->proc_info[rank].send_buf[buf_idx];
						bool completion_message = (comm_buf->length_ == 0);
						c_->handler->free_buffer(comm_buf);
						debug("send complete to=%d, buf_idx=%d, length=%d%s", pid_from_rank(rank), buf_idx, comm_buf->length_,
								completion_message ? " (finished)" : "");
						comm_buf = NULL;
						if(completion_message) {
							c_->num_send_active--;
						}
					}
					else {
						// user tag
						user_cq.push_back(FJMPI_CQ(cq, nic));
					}
				}
				else if(ret == FJMPI_RDMA_HALFWAY_NOTICE) {
					// impossible because we do not use notify flag at all
		//			debug("impossible because we do not use notify flag at all");
				}
				else if(ret == 0) {
					// no completion
				//	debug("break"); usleep(200*1000);
					break;
				}
				else {
					// ????
		//			debug("???????");
				}
			}
		}

		if(c_->num_recv_active == 0 && c_->num_send_active == 0) {
			// finished
			debug("finished");
			c_->handler->finished();
		}

	//	debug("probe finished"); usleep(200*1000);
		return &user_cq;
	}
	virtual bool is_finished() {
		return (c_->num_recv_active == 0 && c_->num_send_active == 0);
	}
	virtual int get_comm_size() {
		return c_->size;
	}
	virtual void pause() {
		paused = true;
	}
	virtual void restart() {
		paused = false;
		for(int i = 0; i < c_->size; ++i) {
			set_send_buffer(i);
		}
	}
#ifndef NDEBUG
	bool check_num_send_buffer() { return (c_->num_pending_send == 0); }
#endif
#if PROFILING_MODE
	void submit_prof_info(int number) {
		int num_rdma_buffers = MAX_RDMA_BUFFER - num_free_memory_;
		profiling::g_pis.submitCounter(num_rdma_buffers, "num_rdma_buffers", number);
		int num_send_count = 0;
		for(int i = 0; i < c_->size; ++i) {
			CommTarget& target = c_->proc_info[i];
			num_send_count += target.send_count - target.start_send_count;
		}
		profiling::g_pis.submitCounter(num_send_count, "num_send_count", number);
	}
#endif

	virtual int memory_id_of(CommunicationBuffer* comm_buf) = 0;

	template <typename T>
	static T* get_pointer(void* base_address, int64_t offset) {
		return reinterpret_cast<T*>(static_cast<uint8_t*>(base_address) + offset);
	}

	template <typename T>
	static T* get_pointer(uint64_t base_address, int64_t offset) {
		return reinterpret_cast<T*>(static_cast<uint8_t*>(base_address) + offset);
	}

	static int buffer_state_offset(int rank) {
		return offset_of_send_buffer_state(rank, 0);
	}

	static int offset_of_send_buffer_state(int rank, int idx) {
		return rank * MAX_FLYING_REQ * 2 + idx;
	}

	static int offset_of_recv_buffer_state(int rank, int idx) {
		return rank * MAX_FLYING_REQ * 2 + MAX_FLYING_REQ + idx;
	}

	template <typename T>
	uint64_t offset_from_pointer(T* pionter, int memory_id) const {
		return ((const uint8_t*)pionter - (const uint8_t*)rdma_buffer_pointers_[memory_id]);
	}

	template <typename T>
	uint64_t local_address_from_pointer(T* pionter, int memory_id) const {
		return local_address_table_[memory_id] +
				offset_from_pointer(pionter, memory_id);
	}
protected:
	void* fix_system_memory(int64_t* data_mem_size) {
		CTRACER(fix_system_memory);
		if(fix_system_memory_ == false) {
			fix_system_memory_ = true;
			debug("fix system memory system_rdma_mem_size=%d", system_rdma_mem_size_);
			*data_mem_size = RDMA_BUF_SIZE - system_rdma_mem_size_;
			return get_pointer<void>(
					rdma_buffer_pointers_[0], system_rdma_mem_size_);
		}
		*data_mem_size = 0;
		return NULL;
	}
	void* allocate_new_rdma_buffer(int* memory_id) {
		CTRACER(allocate_new_rdma_buffer);
		if(num_free_memory_ == 0) {
			throw_exception("Out of memory");
		}
		std::pop_heap(free_memory_ids_, free_memory_ids_ + num_free_memory_, std::greater<int>());
		int next_memory_id = free_memory_ids_[--num_free_memory_];
		allocate_rdma_buffer(next_memory_id);
		*memory_id = next_memory_id;
		return rdma_buffer_pointers_[next_memory_id];
	}

private:
	std::vector<InternalCommunicator> internal_communicators_;
	InternalCommunicator* c_;
	MPI_Group world_group_;

	//! pid -> index for the info
	RankMap proc_index_map_;
	std::vector<int> proc_info_;

	// these are maintained by grow_remote_address_table()
	uint64_t* remote_address_table_;
	int num_procs_;
	int num_address_per_proc_;

	// local RDMA buffers
	void* rdma_buffer_pointers_[MAX_RDMA_BUFFER];
	int free_memory_ids_[MAX_RDMA_BUFFER];
	int num_free_memory_;
	uint64_t local_address_table_[MAX_RDMA_BUFFER];

	bool fix_system_memory_;
	int system_rdma_mem_size_;
	bool paused;

	std::vector<FJMPI_CQ> user_cq;

	int pid_from_rank(int rank) {
		return proc_info_[c_->proc_info[rank].proc_index];
	}

	void initialize_rdma_buffer() {
		CTRACER(initialize_rdma_buffer);
		if(rdma_buffer_pointers_[0] == NULL) {
			debug("initialize_rdma_buffer");
			allocate_rdma_buffer(0);
			num_free_memory_ = MAX_RDMA_BUFFER-1;
			for(int i = 0; i < num_free_memory_; ++i) {
				free_memory_ids_[i] = i+1;
			}
			std::make_heap(free_memory_ids_, free_memory_ids_ + num_free_memory_, std::greater<int>());
		}
	}

	void allocate_rdma_buffer(int memory_id) {
		CTRACER(allocate_rdma_buffer);
		void*& pointer = rdma_buffer_pointers_[memory_id];
		assert (pointer == NULL);
		pointer = page_aligned_xmalloc(RDMA_BUF_SIZE);
		uint64_t dma_address = FJMPI_Rdma_reg_mem(memory_id, pointer, RDMA_BUF_SIZE);
		if(dma_address == FJMPI_RDMA_ERROR) {
			throw_exception("error on FJMPI_Rdma_reg_mem");
		}
		debug("allocate_rdma_buffer mem_id=%d, memory=0x%x, size=0x%x", memory_id, pointer, RDMA_BUF_SIZE);
		local_address_table_[memory_id] = dma_address;
	}

	uint64_t get_remote_address(int proc_index, int memory_id, int64_t offset) {
		CTRACER(get_remote_address);
		int new_num_entry = num_address_per_proc_;
		bool need_to_grow = false;
		// at first, grow the table if needed
		if(memory_id >= new_num_entry) {
			do {
				new_num_entry = std::max<int>(INITIAL_RADDR_TABLE_SIZE, new_num_entry*2);
			} while(memory_id >= new_num_entry);
			need_to_grow = true;
		}
		if(proc_index >= num_procs_) {
			assert (proc_index < proc_info_.size());
			need_to_grow = true;
		}
		if(need_to_grow) {
			grow_remote_address_table(proc_info_.size(), new_num_entry);
		}
		assert(memory_id < num_address_per_proc_);
		assert(remote_address_table_ != NULL);
		uint64_t& address = remote_address_table_[proc_index*num_address_per_proc_ + memory_id];
		if(address == FJMPI_RDMA_ERROR) {
			// we have not stored this address -> get the address
			address = FJMPI_Rdma_get_remote_addr(proc_info_[proc_index], memory_id);
			if(address == FJMPI_RDMA_ERROR) {
				throw_exception("buffer is not registered on the remote host");
			}
			debug("FJMPI_Rdma_get_remote_addr rank=%d, mem_id=%d, address=0x%"PRIx64, proc_info_[proc_index], memory_id, address);
		}
		return address + offset;
	}

	void grow_remote_address_table(int num_procs, int num_address_per_proc) {
		CTRACER(grow_remote_address_table);
		assert(num_procs > 0);
		uint64_t* new_table = (uint64_t*)cache_aligned_xmalloc(
				num_procs*num_address_per_proc*sizeof(uint64_t));
		// initialize table
		for(int i = 0; i < num_procs*num_address_per_proc; ++i) {
			new_table[i] = FJMPI_RDMA_ERROR;
		}
		// copy to the new table
		if(remote_address_table_ != NULL) {
			for(int p = 0; p < num_procs_; ++p) {
				for(int i = 0; i < num_address_per_proc_; ++i) {
					new_table[p*num_address_per_proc + i] =
							remote_address_table_[p*num_address_per_proc_ + i];
				}
			}
			free(remote_address_table_); remote_address_table_ = NULL;
		}
		debug("grow_remote_address_table");
		remote_address_table_ = new_table;
		num_procs_ = num_procs;
		num_address_per_proc_ = num_address_per_proc;
	}

	void set_send_buffer(int target) {
		CTRACER(set_send_buffer);
		// do not send when paused
		if(paused) return ;
		CommTarget& node = c_->proc_info[target];
		while(node.send_queue.size() > 0) {
			int buf_idx = node.send_count % MAX_FLYING_REQ;
			CommunicationBuffer*& comm_buf = node.send_buf[buf_idx];
			if(comm_buf != NULL || c_->send_buffer_state(target, buf_idx).state != READY) {
				debug("not ready to=%d, idx=%d(%d), state=%d, offset=%d", pid_from_rank(target),
						node.send_count, buf_idx, c_->send_buffer_state(target, buf_idx).state,
						offset_from_pointer(&c_->send_buffer_state(target, buf_idx), 0));
				break;
			}
			// To force loading state before loading other information
			__sync_synchronize();
			comm_buf = node.send_queue.front();
			node.send_queue.pop_front();
			volatile BufferState& buf_state = c_->send_buffer_state(target, buf_idx);
			debug("set_send_buffer to=%d, memory_id=%d, idx=%d(%d), length=%d",
					pid_from_rank(target), buf_state.memory_id, node.send_count, buf_idx, comm_buf->length_);
			int pid = proc_info_[node.proc_index];
			{
				// input RDMA command to send data
				int memory_id = memory_id_of(comm_buf);
				int tag = FIRST_DATA_TAG + buf_idx;
				uint64_t raddr = get_remote_address(node.proc_index, buf_state.memory_id, buf_state.offset);
				uint64_t laddr = local_address_from_pointer(comm_buf->pointer(), memory_id);
				int64_t length = comm_buf->element_size() * comm_buf->length_;
				FJMPI_Rdma_put(pid, tag, raddr, laddr, length, node.put_flag);
			}
			{
				// input RDMA command to notify completion
				// make buffer state for the statement of completion
				buf_state.state = COMPLETE;
				buf_state.memory_id = 0;
				buf_state.length = comm_buf->length_;
				int tag = SYSTEM_TAG;
				uint64_t raddr = node.remote_buffer_state +
						sizeof(BufferState) * offset_of_recv_buffer_state(c_->rank, buf_idx);
				uint64_t laddr = local_address_from_pointer(&buf_state, 0);
				FJMPI_Rdma_put(pid, tag, raddr, laddr, sizeof(BufferState), node.put_flag);
			}
			// increment counter
			node.send_count++;
			c_->num_pending_send--;
		}
	}

	void set_recv_buffer(int target) {
		CTRACER(set_recv_buffer);
		CommTarget& node = c_->proc_info[target];
		while(true) {
			int buf_idx = node.recv_count % MAX_FLYING_REQ;
			CommunicationBuffer*& comm_buf = node.recv_buf[buf_idx];
			if(comm_buf != NULL) {
				break;
			}
			// set new receive buffer
			assert (c_->recv_buffer_state(target, buf_idx).state != READY);
			comm_buf = c_->handler->alloc_buffer();
			int memory_id = memory_id_of(comm_buf);
			volatile BufferState& buf_state = c_->recv_buffer_state(target, buf_idx);
			buf_state.state = READY;
			buf_state.memory_id = memory_id;
			buf_state.offset = offset_from_pointer(comm_buf->pointer(), memory_id);
			// notify buffer info to the remote process
			int pid = proc_info_[node.proc_index];
			int tag = SYSTEM_TAG;
			uint64_t raddr = node.remote_buffer_state +
					sizeof(BufferState) * offset_of_send_buffer_state(c_->rank, buf_idx);
			uint64_t laddr = local_address_from_pointer(&buf_state, 0);
			debug("set_recv_buffer to=%d, memory_id=%d, idx=%d(%d), state_address=0x%x",
					pid_from_rank(target), memory_id, node.recv_count, buf_idx, raddr);
			FJMPI_Rdma_put(pid, tag, raddr, laddr, sizeof(BufferState), node.put_flag);
			// increment counter
			node.recv_count++;
		}
	}

	void check_recv_completion(int target) {
		CTRACER(check_recv_completion);
		CommTarget& node = c_->proc_info[target];
		while(true) {
			int buf_idx = node.recv_complete_count % MAX_FLYING_REQ;
			CommunicationBuffer*& comm_buf = node.recv_buf[buf_idx];
			volatile BufferState& buf_state = c_->recv_buffer_state(target, buf_idx);
			if(comm_buf == NULL || buf_state.state != COMPLETE) {
				break;
			}
			// To force loading state before loading length
			__sync_synchronize();
			// receive completed
			if(buf_state.length == 0) {
				// received fold completion
				debug("recv complete rank=%d, memory_id=%d, idx=%d(%d), length=0 (finished)",
						pid_from_rank(target), memory_id_of(comm_buf), node.recv_complete_count, buf_idx);
				c_->num_recv_active--;
				c_->handler->free_buffer(comm_buf);
			}
			else {
				// set new buffer for the next receiving
				comm_buf->length_ = buf_state.length;
				debug("recv complete rank=%d, memory_id=%d, idx=%d(%d), length=%d",
						pid_from_rank(target), memory_id_of(comm_buf), node.recv_complete_count, buf_idx, comm_buf->length_);
				c_->handler->received(comm_buf, target);
			}
			comm_buf = NULL;
			// increment counter
			node.recv_complete_count++;
		}
	}

};

template <typename T>
class FJMpiAlltoallCommunicator
	: public FJMpiAlltoallCommunicatorBase
	, private memory::ConcurrentPool<T>
{
public:
	FJMpiAlltoallCommunicator() : FJMpiAlltoallCommunicatorBase() { }

	~FJMpiAlltoallCommunicator() {
		// clear at this point to prevent the base class to do a wrong method to release memory
		// do not call clear() because it is a virtual function and lead to a confusing behavior.
		clear_pool();
	}

	memory::Pool<T>* get_allocator() {
		return this;
	}

	virtual int memory_id_of(CommunicationBuffer* comm_buf) {
		return static_cast<MemoryBlock*>(
				static_cast<T*>(comm_buf->base_object()))->memory_id;
	}

private:
	virtual T* allocate_new() {
		CTRACER(FJMpiA2A::allocate_new);
		bool ok = false;
		pthread_mutex_lock(&this->thread_sync_);
		// maybe, someone already allocated new buffer
		if(this->free_list_.size() > 0) {
			ok = true;
		}
		if(ok == false) {
			// fix system memory and add rest region to the buffers
			assert (sizeof(MemoryBlock) <= RDMA_BUF_SIZE);
			int64_t data_mem_size;
			void* ptr = this->fix_system_memory(&data_mem_size);
			// is there an empty region ?
			if(ptr != NULL) {
				add_memory_blocks(ptr, data_mem_size, 0);
				if(this->free_list_.size() > 0) {
					ok = true;
				}
			}
		}
		if(ok == false) {
			// allocate new memory and add them to the buffers
			int memory_id;
			void* ptr = this->allocate_new_rdma_buffer(&memory_id);
			add_memory_blocks(ptr, RDMA_BUF_SIZE, memory_id);
			if(this->free_list_.size() > 0) {
				ok = true;
			}
		}
		T* ret = NULL;
		if(ok) {
			ret = this->free_list_.back();
			this->free_list_.pop_back();
		}
		pthread_mutex_unlock(&this->thread_sync_);
		return ret;
	}
	virtual void clear() {
		clear_pool();
	}
	void clear_pool() {
		this->free_list_.clear();
	}

	struct MemoryBlock : public T {
		int memory_id;
		MemoryBlock(int memory_id__) : memory_id(memory_id__) { }
	};

	void add_memory_blocks(void* ptr, int64_t mem_size, int memory_id) {
		CTRACER(FJMpiA2A_add_memory_blocks);
		MemoryBlock* buf = (MemoryBlock*)ptr;
		int64_t num_blocks = mem_size / sizeof(MemoryBlock);
		// initialize blocks
		for(int64_t i = 0; i < num_blocks; ++i) {
			new (&buf[i]) MemoryBlock(memory_id);
		}
		// add to free list
		for(int64_t i = 0; i < num_blocks; ++i) {
			this->free_list_.push_back(&buf[i]);
		}
	}
};
#undef debug

#endif // #ifdef ENABLE_FJMPI_RDMA
#endif /* FJMPI_COMM_HPP_ */
