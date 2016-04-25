/*
 * bottom_up_comm.hpp
 *
 *  Created on: 2014/06/04
 *      Author: ueno
 */

#ifndef BOTTOM_UP_COMM_HPP_
#define BOTTOM_UP_COMM_HPP_

#include "parameters.h"
#include "abstract_comm.hpp"
#include "utils.hpp"

#define debug(...) debug_print(BUCOM, __VA_ARGS__)

struct BottomUpSubstepTag {
	int64_t length;
	int region_id; // < 1024
	int routed_count; // <= 1024
	int route; // <= 1
};

struct BottomUpSubstepData {
	BottomUpSubstepTag tag;
	void* data;
};

class BottomUpSubstepCommBase {
public:
	BottomUpSubstepCommBase() { }
	virtual ~BottomUpSubstepCommBase() {
#if OVERLAP_WAVE_AND_PRED
		MPI_Comm_free(&mpi_comm);
#endif
	}
	void init(MPI_Comm mpi_comm__) {
		mpi_comm = mpi_comm__;
#if OVERLAP_WAVE_AND_PRED
		MPI_Comm_dup(mpi_comm__, &mpi_comm);
#endif
		int size, rank;
		MPI_Comm_size(mpi_comm__, &size);
		MPI_Comm_rank(mpi_comm__, &rank);
		// compute route
		int right_rank = (rank + 1) % size;
		int left_rank = (rank + size - 1) % size;
		nodes(0).rank = left_rank;
		nodes(1).rank = right_rank;
		debug("left=%d, right=%d", left_rank, right_rank);
	}
	void send_first(BottomUpSubstepData* data) {
		data->tag.routed_count = 0;
		data->tag.route = send_filled % 2;
		debug("send_first length=%d, send_filled=%d", data->tag.length, send_filled);
		send_pair[send_filled++] = *data;
		if(send_filled == 2) {
			send_recv();
			send_filled = 0;
		}
	}
	void send(BottomUpSubstepData* data) {
		debug("send length=%d, send_filled=%d", data->tag.length, send_filled);
		send_pair[send_filled++] = *data;
		if(send_filled == 2) {
			send_recv();
			send_filled = 0;
		}
	}
	void recv(BottomUpSubstepData* data) {
		if(recv_tail >= recv_filled) {
			next_recv();
			if(recv_tail >= recv_filled) {
				fprintf(IMD_OUT, "recv_tail >= recv_filled\n");
				throw "recv_filled >= recv_tail";
			}
		}
		*data = recv_pair[recv_tail++ % NBUF];
		debug("recv length=%d, recv_tail=%d", data->tag.length, recv_tail - 1);
	}
	void finish() {
	}

	virtual void print_stt() {
#if VERVOSE_MODE
		int steps = compute_time_.size();
		int64_t sum_compute[steps];
		int64_t sum_wait_comm[steps];
		int64_t max_compute[steps];
		int64_t max_wait_comm[steps];
		MPI_Reduce(&compute_time_[0], sum_compute, steps, MpiTypeOf<int64_t>::type, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&comm_wait_time_[0], sum_wait_comm, steps, MpiTypeOf<int64_t>::type, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(&compute_time_[0], max_compute, steps, MpiTypeOf<int64_t>::type, MPI_MAX, 0, MPI_COMM_WORLD);
		MPI_Reduce(&comm_wait_time_[0], max_wait_comm, steps, MpiTypeOf<int64_t>::type, MPI_MAX, 0, MPI_COMM_WORLD);
		if(mpi.isMaster()) {
			for(int i = 0; i < steps; ++i) {
				double comp_avg = (double)sum_compute[i] / mpi.size_2d / 1000.0;
				double comm_wait_avg = (double)sum_wait_comm[i] / mpi.size_2d / 1000.0;
				double comp_max = (double)max_compute[i] / 1000.0;
				double comm_wait_max = (double)max_wait_comm[i] / 1000.0;
				print_with_prefix("step, %d, max-step, %d, avg-compute, %f, max-compute, %f, avg-wait-comm, %f, max-wait-comm, %f, (ms)",
						i+1, steps, comp_avg, comp_max, comm_wait_avg, comm_wait_max);
			}
		}
#endif
	}
protected:
	enum {
		NBUF = 4,
		BUFMASK = NBUF-1,
	};

	struct CommTargetBase {
		int rank;
	};

	MPI_Comm mpi_comm;
	std::vector<void*> free_list;
	BottomUpSubstepData send_pair[NBUF];
	BottomUpSubstepData recv_pair[NBUF];
	VERVOSE(profiling::TimeKeeper tk_);
	VERVOSE(std::vector<int64_t> compute_time_);
	VERVOSE(std::vector<int64_t> comm_wait_time_);

	int element_size;
	int buffer_width;
	int send_filled;
	int recv_filled;
	int recv_tail;

	virtual CommTargetBase& nodes(int target) = 0;
	virtual void send_recv() = 0;
	virtual void next_recv() = 0;

	int buffers_available() {
		return (int)free_list.size();
	}

	void* get_buffer() {
		assert(buffers_available());
		void* ptr = free_list.back();
		free_list.pop_back();
		return ptr;
	}

	void free_buffer(void* buffer) {
		free_list.push_back(buffer);
	}

	template <typename T>
	void begin(T** recv_buffers__, int buffer_count__, int buffer_width__) {
		element_size = sizeof(T);
		buffer_width = buffer_width__;

		free_list.clear();
		for(int i = 0; i < buffer_count__; ++i) {
			free_list.push_back(recv_buffers__[i]);
		}
		send_filled = recv_tail = recv_filled = 0;

		debug("begin buffer_count=%d, buffer_width=%d",
				buffer_count__, buffer_width__);
#if VERVOSE_MODE
		if(mpi.isMaster()) print_with_prefix("Bottom-up substep buffer count: %d", buffer_count__);
#endif
		VERVOSE(tk_.getSpanAndReset());
		VERVOSE(compute_time_.clear());
		VERVOSE(comm_wait_time_.clear());
	}
};

class MpiBottomUpSubstepComm : public BottomUpSubstepCommBase {
	typedef BottomUpSubstepCommBase super__;
public:
	MpiBottomUpSubstepComm(MPI_Comm mpi_comm__)
	{
		init(mpi_comm__);
	}
	virtual ~MpiBottomUpSubstepComm() {
	}
	void register_memory(void* memory, int64_t size) {
	}
	template <typename T>
	void begin(T** recv_buffers__, int buffer_count__, int buffer_width__) {
		super__::begin(recv_buffers__, buffer_count__, buffer_width__);
		type = MpiTypeOf<T>::type;
		recv_top = 0;
		is_active = false;
	}
	void probe() {
		next_recv_probe(false);
	}
	void finish() {
	}

protected:
	struct CommTarget : public CommTargetBase {
	};

	CommTarget nodes_[2];
	MPI_Datatype type;
	MPI_Request req[4];
	int recv_top;
	bool is_active;

	virtual CommTargetBase& nodes(int target) { return nodes_[target]; }

	int make_tag(BottomUpSubstepTag& tag) {
		return (1 << 30) | (tag.route << 24) |
				(tag.routed_count << 12) | tag.region_id;
	}

	BottomUpSubstepTag make_tag(MPI_Status& status) {
		BottomUpSubstepTag tag;
		int length;
		int raw_tag = status.MPI_TAG;
		MPI_Get_count(&status, type, &length);
		tag.length = length;
		tag.region_id = raw_tag & 0xFFF;
		tag.routed_count = (raw_tag >> 12) & 0xFFF;
		tag.route = (raw_tag >> 24) & 1;
		return tag;
	}

	void next_recv_probe(bool blocking) {
		if(is_active) {
			MPI_Status status[4];
			if(blocking) {
				MPI_Waitall(4, req, status);
			}
			else {
				int flag;
				MPI_Testall(4, req, &flag, status);
				if(flag == false) {
					return ;
				}
			}
			int recv_0 = recv_filled++ % NBUF;
			int recv_1 = recv_filled++ % NBUF;
			recv_pair[recv_0].tag = make_tag(status[0]);
			recv_pair[recv_1].tag = make_tag(status[1]);
			free_buffer(send_pair[2].data);
			free_buffer(send_pair[3].data);
			is_active = false;
		}
	}

	virtual void next_recv() {
		next_recv_probe(true);
	}

	virtual void send_recv() {
		VERVOSE(compute_time_.push_back(tk_.getSpanAndReset()));
		next_recv_probe(true);
		VERVOSE(comm_wait_time_.push_back(tk_.getSpanAndReset()));
		int recv_0 = recv_top++ % NBUF;
		int recv_1 = recv_top++ % NBUF;
		recv_pair[recv_0].data = get_buffer();
		recv_pair[recv_1].data = get_buffer();
		MPI_Irecv(recv_pair[recv_0].data, buffer_width,
				type, nodes_[0].rank, MPI_ANY_TAG, mpi_comm, &req[0]);
		MPI_Irecv(recv_pair[recv_1].data, buffer_width,
				type, nodes_[1].rank, MPI_ANY_TAG, mpi_comm, &req[1]);
		MPI_Isend(send_pair[0].data, send_pair[0].tag.length,
				type, nodes_[1].rank, make_tag(send_pair[0].tag), mpi_comm, &req[2]);
		MPI_Isend(send_pair[1].data, send_pair[1].tag.length,
				type, nodes_[0].rank, make_tag(send_pair[1].tag), mpi_comm, &req[3]);

		send_pair[2] = send_pair[0];
		send_pair[3] = send_pair[1];
		is_active = true;
#if !BOTTOM_UP_OVERLAP_PFS // if overlapping is disabled
		next_recv_probe(true);
#endif
	}
};

//#if ENABLE_FJMPI_RDMA
#if 0
#include "fjmpi_comm.hpp"

class FJMpiBottomUpSubstepComm : public BottomUpSubstepCommBase {
public:
	FJMpiBottomUpSubstepComm(MPI_Comm mpi_comm__, int Z2, int rank_z1) {
		init(mpi_comm__, Z2, rank_z1);
		MPI_Group world_group;
		MPI_Group comm_group;
		MPI_Comm_group(MPI_COMM_WORLD, &world_group);
		MPI_Comm_group(mpi_comm__, &comm_group);
		int ranks1[] = { left_rank, right_rank };
		int ranks2[2];
		MPI_Group_translate_ranks(comm_group, 2, ranks1, world_group, ranks2);
		MPI_Group_free(&world_group);
		MPI_Group_free(&comm_group);

		nodes[0].pid = ranks2[0];
		nodes[0].put_flag = FJMPI_Local_nic[0] | FJMPI_Remote_nic[2] |
				FJMPI_RDMA_IMMEDIATE_RETURN | FJMPI_RDMA_PATH0;
		nodes[1].pid = ranks2[1];
		nodes[1].put_flag = FJMPI_Local_nic[1] | FJMPI_Remote_nic[3] |
				FJMPI_RDMA_IMMEDIATE_RETURN | FJMPI_RDMA_PATH0;

		current_step = 0;
	}
	virtual ~FJMpiBottomUpSubstepComm() {
	}
	void register_memory(void* memory, int64_t size) {
		rdma_buffer_pointers[0] = buffer_state;
		rdma_buffer_pointers[1] = memory;
		local_address[0] = FJMPI_Rdma_reg_mem(SYS_MEM_ID, buffer_state, sizeof(buffer_state));
		local_address[1] = FJMPI_Rdma_reg_mem(DATA_MEM_ID, memory, size);
		for(int i = 0; i < TNBUF*4; ++i) {
			buffer_state[i].state = INVALIDATED;
		}
		MPI_Barrier(mpi_comm);
		for(int i = 0; i < 2; ++i) {
			nodes[i].address[0] = FJMPI_Rdma_get_remote_addr(node[i].pid, SYS_MEM_ID);
			nodes[i].address[1] = FJMPI_Rdma_get_remote_addr(node[i].pid, DATA_MEM_ID);
		}
	}
	template <typename T>
	void begin(T** recv_buffers__, int buffer_count__, int buffer_width__) {
		begin(recv_buffers__, buffer_count__, buffer_width__);
		++current_step;
		debug("begin");
	}
	virtual void begin_comm() {
		debug("initialized");
	}
	virtual void probe_comm(void* comm_data) {
		// process receive completion
		for(int p = 0; p < 2; ++p) {
			check_recv_completion(p);
			set_recv_buffer(p);
			set_send_buffer(p);
		}

		// process completion
		std::vector<FJMPI_CQ>& cqs = *(std::vector<FJMPI_CQ>*)comm_data;
		for(int i = 0; i < cqs.size(); ++i) {
			FJMPI_CQ cq = cqs[i];
			if(cq.tag >= FIRST_USER_TAG && cq.tag < USER_TAG_END){
				int buf_idx = cq.tag - FIRST_USER_TAG;
				CommTarget& node = (cq.pid == nodes[0].pid) ? nodes[0] : nodes[1];
				BottomUpSubstepData& comm_buf = node.send_buf[buf_idx];
				free_list.push_back(comm_buf.data);
				debug("send complete to=%d, buf_idx=%d, length=%d", node.pid, buf_idx, comm_buf.tag.length);
				comm_buf.data = NULL;
				node.send_complete_count++;
			}
		}
	}
	virtual void end_comm(void* comm_data) {
		//
	}

protected:
	enum {
		FIRST_USER_TAG = 5,
		USER_TAG_END = FIRST_USER_TAG + NBUF,

		SYSTEM = 0,
		DATA = 1,

		SYS_MEM_ID = 300,
		DATA_MEM_ID = 301,

		INVALIDATED = 0,
		READY = 1,
		COMPLETE = 2,
	};

	struct BufferState {
		uint16_t state; // current state of the buffer
		uint16_t step;
		union {
			uint64_t offset; // offset to the buffer starting address
			BottomUpSubstepTag tag; // length of the received data in bytes
		};
	};

	struct CommTarget : public CommTargetBase {
		int pid;
		int put_flag;
		uint64_t address[2];

		CommTarget() : CommTargetBase() {
			pid = put_flag = 0;
			address[0] = 0;
			address[1] = 0;
		}

		uint64_t remote_address(int memory_id, uint64_t offset) {
			return address[memory_id] + offset;
		}
	};
	volatile BufferState buffer_state[NBUF*4];
	void* rdma_buffer_pointers[2];
	uint64_t local_address[2];
	CommTarget nodes_[2];
	int current_step;

	virtual CommTargetBase& nodes(int target) { return nodes_[target]; }

	template <typename T>
	uint64_t offset_from_pointer(T* pionter, int memory_id) const {
		return ((const uint8_t*)pionter - (const uint8_t*)rdma_buffer_pointers[memory_id]);
	}

	template <typename T>
	uint64_t local_address_from_pointer(T* pionter, int memory_id) const {
		return local_address[memory_id] +
				offset_from_pointer(pionter, memory_id);
	}

	volatile BufferState& send_buffer_state(int rank, int idx) {
		return buffer_state[offset_of_send_buffer_state(rank, idx)];
	}

	volatile BufferState& recv_buffer_state(int rank, int idx) {
		return buffer_state[offset_of_recv_buffer_state(rank, idx)];
	}

	static int offset_of_send_buffer_state(int rank, int idx) {
		return rank * NBUF * 2 + idx;
	}

	static int offset_of_recv_buffer_state(int rank, int idx) {
		return rank * NBUF * 2 + NBUF + idx;
	}

	void set_send_buffer(int target) {
		MY_TRACE;
		CommTarget& node = nodes_[target];
		while(node.send_queue.size() > 0) {
			int buf_idx = node.send_count % NBUF;
			BottomUpSubstepData& comm_buf = node.send_buf[buf_idx];
			volatile BufferState& bs = send_buffer_state(target, buf_idx);
			if(comm_buf.data != NULL || bs.state != READY || bs.step != current_step) {
				debug("not ready to=%d, idx=%d(%d), state=%d", nodes[target].pid,
						node.send_count, buf_idx, send_buffer_state(target, buf_idx).state);
				break;
			}
			// To force loading state before loading other information
			__sync_synchronize();
			comm_buf = node.send_queue.front();
			node.send_queue.pop_front();
			volatile BufferState& buf_state = send_buffer_state(target, buf_idx);
			debug("set_send_buffer to=%d, idx=%d(%d), length=%d",
					nodes_[target].pid, node.send_count, buf_idx, comm_buf.tag.length);
			int pid = node.pid;
			{
				// input RDMA command to send data
				int memory_id = DATA_MEM_ID;
				int tag = FIRST_USER_TAG + buf_idx;
				uint64_t raddr = node.remote_address(DATA_MEM_ID, buf_state.offset);
				uint64_t laddr = local_address_from_pointer(comm_buf.data, DATA_MEM_ID);
				int64_t length = element_size * comm_buf.tag.length;
				FJMPI_Rdma_put(pid, tag, raddr, laddr, length, node.put_flag);
			}
			{
				// input RDMA command to notify completion
				// make buffer state for the statement of completion
				buf_state.state = COMPLETE;
				buf_state.step = current_step;
				buf_state.tag = comm_buf.tag;
				int tag = 0;
				uint64_t raddr = node.remote_address(SYS_MEM_ID,
						sizeof(BufferState) * offset_of_recv_buffer_state(target, buf_idx));
				uint64_t laddr = local_address_from_pointer(&buf_state, 0);
				FJMPI_Rdma_put(pid, tag, raddr, laddr, sizeof(BufferState), node.put_flag);
			}
			// increment counter
			node.send_count++;
		}
	}

	void set_recv_buffer(int target) {
		MY_TRACE;
		CommTarget& node = nodes_[target];
		while(true) {
			int buf_idx = node.recv_count % NBUF;
			BottomUpSubstepData& comm_buf = node.recv_buf[buf_idx];
			if(comm_buf.data != NULL) {
				break;
			}
			assert (recv_buffer_state(target, buf_idx).state != READY ||
					recv_buffer_state(target, buf_idx).step != current_step);
			if(free_list.size() == 0) {
				// no buffer
				break;
			}
			// set new receive buffer
			comm_buf.data = free_list.back(); free_list.pop_back();
			int memory_id = DATA_MEM_ID;
			volatile BufferState& buf_state = recv_buffer_state(target, buf_idx);
			buf_state.state = READY;
			buf_state.step = current_step;
			buf_state.offset = offset_from_pointer(comm_buf.data, memory_id);
			// notify buffer info to the remote process
			int pid = node.pid;
			int tag = 0;
			uint64_t raddr = node.remote_address(SYS_MEM_ID,
					sizeof(BufferState) * offset_of_send_buffer_state(target, buf_idx));
			uint64_t laddr = local_address_from_pointer(&buf_state, SYS_MEM_ID);
			debug("set_recv_buffer to=%d, idx=%d(%d), state_address=0x%x",
					pid, node.recv_count, buf_idx, raddr);
			FJMPI_Rdma_put(pid, tag, raddr, laddr, sizeof(BufferState), node.put_flag);
			// increment counter
			node.recv_count++;
		}
	}

	void check_recv_completion(int target) {
		MY_TRACE;
		CommTarget& node = nodes_[target];
		while(true) {
			int buf_idx = node.recv_complete_count % NBUF;
			BottomUpSubstepData& comm_buf = node.recv_buf[buf_idx];
			volatile BufferState& buf_state = recv_buffer_state(target, buf_idx);
			if(comm_buf.data == NULL || buf_state.state != COMPLETE) {
				break;
			}
			// To force loading state before loading length
			__sync_synchronize();

			// receive completed
			comm_buf.tag = buf_state.tag;
			debug("recv complete rank=%d, idx=%d(%d), length=%d",
					node.pid, node.recv_complete_count, buf_idx, comm_buf.tag.length);
			recv_data(&comm_buf);
			comm_buf.data = NULL;
			// increment counter
			node.recv_complete_count++;
		}
	}

};
#endif // #ifdef ENABLE_FJMPI_RDMA
#undef debug

#endif /* BOTTOM_UP_COMM_HPP_ */
