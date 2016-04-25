/*
 * bfs.hpp
 *
 *  Created on: Mar 5, 2012
 *      Author: koji
 */

#ifndef BFS_HPP_
#define BFS_HPP_

#include <pthread.h>

#include <deque>

#if CUDA_ENABLED
#include "gpu_host.hpp"
#endif

#include "utils.hpp"
#include "fiber.hpp"
#include "abstract_comm.hpp"
#include "mpi_comm.hpp"
#include "fjmpi_comm.hpp"
#include "bottom_up_comm.hpp"

#include "low_level_func.h"

#define debug(...) debug_print(BFSMN, __VA_ARGS__)
class BfsBase
{
	typedef BfsBase ThisType;
	typedef Graph2DCSR GraphType;

public:
	enum {
		// Number of CQ bitmap entries represent as 1 bit in summary.
		// Since the type of bitmap entry is int32_t and 1 cache line is composed of 32 bitmap entries,
		// 32 is effective value.
		ENABLE_WRITING_DEPTH = 1,

		BUCKET_UNIT_SIZE = 1024,

		// non-parameters
		NBPE = PRM::NBPE,
		LOG_NBPE = PRM::LOG_NBPE,
		NBPE_MASK = PRM::NBPE_MASK,

		BFELL_SORT = PRM::BFELL_SORT,
		LOG_BFELL_SORT = PRM::LOG_BFELL_SORT,
		BFELL_SORT_MASK = PRM::BFELL_SORT_MASK,
		BFELL_SORT_IN_BMP = BFELL_SORT / NBPE,

		BU_SUBSTEP = PRM::NUM_BOTTOM_UP_STREAMS,
	};

	class QueuedVertexes {
	public:
		TwodVertex v[BUCKET_UNIT_SIZE];
		int length;
		enum { SIZE = BUCKET_UNIT_SIZE };

		QueuedVertexes() : length(0) { }
		bool append(TwodVertex val) {
			if(length == SIZE) return false;
			v[length++] = val; return true;
		}
		void append_nocheck(TwodVertex val) {
			v[length++] = val;
		}
		void append_nocheck(TwodVertex pred, TwodVertex tgt) {
			v[length+0] = pred;
			v[length+1] = tgt;
			length += 2;
		}
		bool full() { return (length == SIZE); }
		int size() { return length; }
		void clear() { length = 0; }
	};

	struct ThreadLocalBuffer {
		QueuedVertexes* cur_buffer;
		LocalPacket fold_packet[1];
	};

	BfsBase()
		: bottom_up_substep_(NULL)
		, top_down_comm_(this)
		, bottom_up_comm_(this)
		, td_comm_(mpi.comm_2dc, &top_down_comm_)
		, bu_comm_(mpi.comm_2dr, &bottom_up_comm_)
		, denom_to_bottom_up_(DENOM_TOPDOWN_TO_BOTTOMUP)
		, denom_bitmap_to_list_(DENOM_BITMAP_TO_LIST)
		, thread_sync_(omp_get_max_threads())
	{
	}

	virtual ~BfsBase()
	{
		delete bottom_up_substep_; bottom_up_substep_ = NULL;
	}

	template <typename EdgeList>
	void construct(EdgeList* edge_list)
	{
		// minimun requirement of CQ
		// CPU: MINIMUN_SIZE_OF_CQ_BITMAP words -> MINIMUN_SIZE_OF_CQ_BITMAP * NUMBER_PACKING_EDGE_LISTS * mpi.size_2dc
		// GPU: THREADS_PER_BLOCK words -> THREADS_PER_BLOCK * NUMBER_PACKING_EDGE_LISTS * mpi.size_2dc

		int log_local_verts_unit = get_msb_index(std::max<int>(BFELL_SORT, NBPE) * 8);

		detail::GraphConstructor2DCSR<EdgeList> constructor;
		constructor.construct(edge_list, log_local_verts_unit, graph_);
	}

	void prepare_bfs() {
		printInformation();
		allocate_memory();
	}

	void run_bfs(int64_t root, int64_t* pred);

	void get_pred(int64_t* pred) {
	//	comm_.release_extra_buffer();
	}

	void end_bfs() {
		deallocate_memory();
	}

	GraphType graph_;

// protected:

	int64_t get_bitmap_size_src() const {
		return graph_.num_local_verts_ / NBPE * mpi.size_2dr;
	}
	int64_t get_bitmap_size_tgt() const {
		return graph_.num_local_verts_ / NBPE * mpi.size_2dc;
	}
	int64_t get_bitmap_size_local() const {
		return graph_.num_local_verts_ / NBPE;
	}

	template <typename T>
	void get_shared_mem_pointer(void*& ptr, int64_t width, T** local_ptr, T** orig_ptr) {
		if(orig_ptr) *orig_ptr = (T*)ptr;
		if(local_ptr) *local_ptr = (T*)ptr + width*mpi.rank_z;
		ptr = (uint8_t*)ptr + width*sizeof(T)*mpi.size_z;
	}

	void allocate_memory()
	{
		const int max_threads = omp_get_max_threads();
		const int max_comm_size = std::max(mpi.size_2dc, mpi.size_2dr);
		int64_t bitmap_width = get_bitmap_size_local();

		/**
		 * Buffers for computing BFS
		 * - next queue: This is vertex list in the top-down phase, and <pred, target> tuple list in the bottom-up phase.
		 * - thread local buffer (includes local packet)
		 * - two visited memory (for double buffering)
		 * - working memory: This is used
		 * 	1) to store steaming visited in the bottom-up search phase
		 * 		required size: half_bitmap_width * BOTTOM_UP_BUFFER (for each process)
		 * 	2) to store next queue vertices in the bottom-up expand phase
		 * 		required size: half_bitmap_width * 2 (for each process)
		 * - shared visited:
		 * - shared visited update (to store the information to update shared visited)
		 * - current queue extra memory (is allocated dynamically when the it is required)
		 * - communication buffer for asynchronous communication:
		 */

		a2a_comm_buf_.allocate_memory(graph_.num_local_verts_ * sizeof(int32_t) * 50); // TODO: accuracy

		top_down_comm_.max_num_rows = graph_.num_local_verts_ * 16 / PRM::TOP_DOWN_PENDING_WIDTH + 1000;
		top_down_comm_.tmp_rows = (TopDownRow*)cache_aligned_xmalloc(
				top_down_comm_.max_num_rows*2*sizeof(TopDownRow)); // for debug

		thread_local_buffer_ = (ThreadLocalBuffer**)cache_aligned_xmalloc(sizeof(thread_local_buffer_[0])*max_threads);

		const int bottom_up_vertex_count_per_thread = (bitmap_width/BU_SUBSTEP + max_threads - 1) / max_threads * NBPE;
		const int packet_buffer_length = std::max(
				sizeof(LocalPacket) * max_comm_size, // for the top-down and bottom-up list search
				sizeof(TwodVertex) * 2 * bottom_up_vertex_count_per_thread);
		const int buffer_width = roundup<int>(
				sizeof(ThreadLocalBuffer) + packet_buffer_length, CACHE_LINE);
		buffer_.thread_local_ = cache_aligned_xcalloc(buffer_width*max_threads);
		for(int i = 0; i < max_threads; ++i) {
			ThreadLocalBuffer* tlb = (ThreadLocalBuffer*)
							((uint8_t*)buffer_.thread_local_ + buffer_width*i);
			tlb->cur_buffer = NULL;
			thread_local_buffer_[i] = tlb;
		}
		packet_buffer_is_dirty_ = true;

		enum { NBUF = PRM::BOTTOM_UP_BUFFER };
		//work_buf_size_ = half_bitmap_width * PRM::BOTTOM_UP_BUFFER * sizeof(BitmapType);
		work_buf_size_ = std::max<int64_t>(
				bitmap_width * sizeof(BitmapType) * mpi.size_2dr / mpi.size_z, // space to receive NQ
				bitmap_width * NBUF * sizeof(BitmapType)); // space for working buffer
		int shared_offset_length = (max_threads * mpi.size_z * BU_SUBSTEP + 1);
		int64_t total_size_of_shared_memory =
				bitmap_width * 3 * sizeof(BitmapType) * mpi.size_z + // new and old visited and buffer
				work_buf_size_ * mpi.size_z + // work_buf_
				bitmap_width * sizeof(BitmapType) * max_comm_size + // shared visited memory
				sizeof(memory::SpinBarrier) + sizeof(int) * shared_offset_length;
		VERVOSE(if(mpi.isMaster()) print_with_prefix("Allocating shared memory: %f GB per node.", to_giga(total_size_of_shared_memory)));

		void* smem_ptr = buffer_.shared_memory_ = shared_malloc(total_size_of_shared_memory);

		get_shared_mem_pointer<BitmapType>(smem_ptr, bitmap_width, (BitmapType**)&new_visited_, NULL);
		get_shared_mem_pointer<BitmapType>(smem_ptr, bitmap_width, (BitmapType**)&old_visited_, NULL);
		get_shared_mem_pointer<BitmapType>(smem_ptr, bitmap_width, (BitmapType**)&visited_buffer_,
				(BitmapType**)&visited_buffer_orig_);
		get_shared_mem_pointer<int8_t>(smem_ptr, work_buf_size_, (int8_t**)&work_buf_,
				(int8_t**)&nq_recv_buf_);
		shared_visited_ = (BitmapType*)smem_ptr; smem_ptr = (BitmapType*)smem_ptr + bitmap_width * max_comm_size;
		s_.sync = new (smem_ptr) memory::SpinBarrier(mpi.size_z); smem_ptr = s_.sync + 1;
		s_.offset = (int*)smem_ptr; smem_ptr = (int*)smem_ptr + shared_offset_length;

		assert (smem_ptr == (int8_t*)buffer_.shared_memory_ + total_size_of_shared_memory);

		bottom_up_substep_ = new MpiBottomUpSubstepComm(mpi.comm_2dr);
		bottom_up_substep_->register_memory(buffer_.shared_memory_, total_size_of_shared_memory);

		cq_list_ = NULL;
		global_nq_size_ = max_nq_size_ = nq_size_ = cq_size_ = 0;
		bitmap_or_list_ = false;
		work_extra_buf_ = NULL;
		work_extra_buf_size_ = 0;
	}

	void deallocate_memory()
	{
		free(buffer_.thread_local_); buffer_.thread_local_ = NULL;
		shared_free(buffer_.shared_memory_); buffer_.shared_memory_ = NULL;
		free(thread_local_buffer_); thread_local_buffer_ = NULL;
		a2a_comm_buf_.deallocate_memory();
	}

	void initialize_memory(int64_t* pred)
	{
		int64_t num_orig_local_vertices = graph_.pred_size();
		int64_t bitmap_width = get_bitmap_size_local();
		int64_t shared_vis_width = bitmap_width * mpi.size_2dr;

		int64_t shared_vis_block = shared_vis_width / mpi.size_z;
		int64_t shared_vis_off = shared_vis_block * mpi.rank_z;

		BitmapType* visited = (BitmapType*)new_visited_;
		BitmapType* shared_visited = shared_visited_;

#pragma omp parallel
		{
#if !INIT_PRED_ONCE	// Only Spec2010 needs this initialization
#pragma omp for nowait
			for(int64_t i = 0; i < num_orig_local_vertices; ++i) {
				pred[i] = -1;
			}
#endif
			// clear NQ and visited
#pragma omp for nowait
			for(int64_t i = 0; i < bitmap_width; ++i) {
				visited[i] = ~(graph_.has_edge_bitmap_[i]);
			}
			// clear shared visited
#pragma omp for nowait
			for(int64_t i = 0; i < shared_vis_block; ++i) {
				shared_visited[shared_vis_off + i] = 0;
			}
		}

		assert (nq_.stack_.size() == 0);

		if(mpi.isYdimAvailable()) s_.sync->barrier();
	}

	//-------------------------------------------------------------//
	// Async communication
	//-------------------------------------------------------------//

	class CommBufferPool {
	public:
		void allocate_memory(int size) {
			first_buffer_ = cache_aligned_xmalloc(size);
			second_buffer_ = cache_aligned_xmalloc(size);
			current_index_ = 0;
			pool_buffer_size_ = size;
			num_buffers_ = size / PRM::COMM_BUFFER_SIZE;
		}

		void deallocate_memory() {
			free(first_buffer_); first_buffer_ = NULL;
			free(second_buffer_); second_buffer_ = NULL;
		}

		void* get_next() {
			int idx = current_index_++;
			if(num_buffers_ <= idx) {
				fprintf(IMD_OUT, "num_buffers_ <= idx (num_buffers=%d)\n", num_buffers_);
				throw "Error: buffer size not enough";
			}
			return (uint8_t*)first_buffer_ + PRM::COMM_BUFFER_SIZE * idx;
		}

		void* clear_buffers() {
			current_index_ = 0;
			return first_buffer_;
		}

		void* second_buffer() {
			return second_buffer_;
		}

		int pool_buffer_size() {
			return pool_buffer_size_;
		}

	protected:
		int pool_buffer_size_;
		void* first_buffer_;
		void* second_buffer_;
		int current_index_;
		int num_buffers_;
	};

	template <typename T>
	class CommHandlerBase : public AlltoallBufferHandler {
	public:
		enum { BUF_SIZE = PRM::COMM_BUFFER_SIZE / sizeof(T) };
		CommHandlerBase(ThisType* this__)
			: this_(this__)
			, pool_(&this__->a2a_comm_buf_)
		{ }
		virtual ~CommHandlerBase() { }
		virtual void* get_buffer() {
			return this->pool_->get_next();
		}
		virtual void add(void* buffer, void* ptr__, int offset, int length) {
			assert (offset >= 0);
			assert (offset + length <= BUF_SIZE);
			memcpy((T*)buffer + offset, ptr__, length*sizeof(T));
		}
		virtual void* clear_buffers() {
			return this->pool_->clear_buffers();
		}
		virtual void* second_buffer() {
			return this->pool_->second_buffer();
		}
		virtual int max_size() {
			return this->pool_->pool_buffer_size();
		}
		virtual int buffer_length() {
			return BUF_SIZE;
		}
		virtual MPI_Datatype data_type() {
			return MpiTypeOf<T>::type;
		}
		virtual int element_size() {
			return sizeof(T);
		}
		virtual void finish() { }
	protected:
		ThisType* this_;
		CommBufferPool* pool_;
	};

	struct TopDownRow {
		int64_t src;
		int length;
		uint32_t* ptr;
	};

	class TopDownCommHandler : public CommHandlerBase<uint32_t> {
	public:
		TopDownCommHandler(ThisType* this__)
			: CommHandlerBase<uint32_t>(this__)
			, tmp_rows(NULL)
			, max_num_rows(0)
			, num_rows(0)
			  { }

		~TopDownCommHandler() {
			if(tmp_rows != NULL) { free(tmp_rows); tmp_rows = NULL; }
		}

		virtual void received(void* buf, int offset, int length, int src) {
			VERVOSE(g_tp_comm += length * sizeof(uint32_t));
			if(this_->growing_or_shrinking_) {
				this->this_->top_down_receive<true>((uint32_t*)buf + offset, length, tmp_rows, &num_rows);
			}
			else {
				this->this_->top_down_receive<false>((uint32_t*)buf + offset, length, tmp_rows, &num_rows);
			}
			assert (num_rows < max_num_rows);
		}

		virtual void finish() {
			VERVOSE(if(mpi.isMaster()) print_with_prefix("num_rows= %d / %d", num_rows, max_num_rows));
			if(num_rows == 0) return ;
			if(num_rows > max_num_rows) {
				fprintf(IMD_OUT, "Insufficient temporary rows buffer\n");
				throw "Insufficient temporary rows buffer";
			}
			if(this_->growing_or_shrinking_) {
				this->this_->top_down_row_receive<true>(tmp_rows, num_rows);
			}
			else {
				this->this_->top_down_row_receive<false>(tmp_rows, num_rows);
			}
			num_rows = 0;
		}

		TopDownRow* tmp_rows;
		int max_num_rows;
		volatile int num_rows;
	};

	class BottomUpCommHandler : public CommHandlerBase<int64_t> {
	public:
		BottomUpCommHandler(ThisType* this__)
			: CommHandlerBase<int64_t>(this__)
			  { }

		virtual void received(void* buf, int offset, int length, int src) {
			VERVOSE(g_bu_pred_comm += length * sizeof(int64_t));
			BottomUpReceiver recv(this->this_, (int64_t*)buf + offset, length, src);
			recv.run();
		}
	};

	/*
	void do_in_parallel(Runnable* main, Runnable* sub, bool inverse) {
		if(inverse) {
			std::swap(main, sub);
		}
#if OPENMP_SUB_THREAD
#pragma omp parallel num_threads(2)
		{
			SET_AFFINITY;
			if(omp_get_thread_num() == 0) {
				main->run();
			}
			else {
				sub->run();
			}
		}
#else
		compute_thread_.do_in_parallel(main, sub);
#endif
	}
	*/

	//-------------------------------------------------------------//
	// expand phase
	//-------------------------------------------------------------//

	template <typename T>
	void get_visited_pointers(T** ptrs, int num_ptrs, void* visited_buf, int split_count) {
		int step_bitmap_width = get_bitmap_size_local() / split_count;
		for(int i = 0; i < num_ptrs; ++i) {
			ptrs[i] = (T*)((BitmapType*)visited_buf + step_bitmap_width*i);
		}
	}

	void clear_nq_stack() {
		int num_buffers = nq_.stack_.size();
		for(int i = 0; i < num_buffers; ++i) {
			// Since there are no need to lock pool in this case,
			// we invoke Pool::free method explicitly.
			nq_.stack_[i]->length = 0;
			nq_empty_buffer_.memory::Pool<QueuedVertexes>::free(nq_.stack_[i]);
		}
		nq_.stack_.clear();
	}

	void first_expand(int64_t root) {
		// !!root is UNSWIZZLED and ORIGINAL ID!!
		int root_owner = vertex_owner(root);
		int root_r = root_owner % mpi.size_2dr;
		int root_c = root_owner / mpi.size_2dr;

		cq_list_ = (TwodVertex*)work_buf_;

		if(root_r == mpi.rank_2dr) {
			// get reordered vertex id
			int64_t root_reordered = 0;
			if(root_owner == mpi.rank_2d) {
				// yes, we have reordered id
				int64_t root_local = vertex_local(root);
				int64_t reordered = graph_.reorder_map_[root_local];
				root_reordered = (reordered * mpi.size_2d) + root_owner;

				MPI_Bcast(&root_reordered, 1, MpiTypeOf<int64_t>::type,
						root_c, mpi.comm_2dr);

				// update pred
				pred_[root_local] = root;

				// update visited
				int64_t word_idx = reordered >> LOG_NBPE;
				int bit_idx = reordered & NBPE_MASK;
				((BitmapType*)new_visited_)[word_idx] |= BitmapType(1) << bit_idx;
			}
			else {
				MPI_Bcast(&root_reordered, 1, MpiTypeOf<int64_t>::type,
						root_c, mpi.comm_2dr);
			}

			// update CQ
			SeparatedId root_src = graph_.VtoS(root_reordered);
			cq_list_[0] = root_src.value;
			cq_size_ = 1;
		}
		else {
			cq_size_ = 0;
		}
	}

	// expand visited bitmap and receive the shared visited
	void expand_visited_bitmap() {
		TRACER(expand_vis_bmp);
		int bitmap_width = get_bitmap_size_local();
		if(mpi.isYdimAvailable()) s_.sync->barrier();
		if(mpi.rank_z == 0 && mpi.comm_y != MPI_COMM_NULL) {
			BitmapType* const bitmap = (BitmapType*)new_visited_;
			BitmapType* recv_buffer = shared_visited_;
			// TODO: asymmetric size for z. (MPI_Allgather -> MPI_Allgatherv or MpiCol::allgatherv ?)
			int shared_bitmap_width = bitmap_width * mpi.size_z;
#if ENABLE_MY_ALLGATHER
			if(mpi.isYdimAvailable()) {
				if(mpi.isMaster()) print_with_prefix("Error: MY_ALLGATHER does not support shared memory Y dimension.");
			}
#if ENABLE_MY_ALLGATHER == 1
		MpiCol::my_allgather(bitmap, shared_bitmap_width, recv_buffer, mpi.comm_c);
#else
		my_allgather_2d(bitmap, shared_bitmap_width, recv_buffer, mpi.comm_c);
#endif // #if ENABLE_MY_ALLGATHER == 1
#else
			MPI_Allgather(bitmap, shared_bitmap_width, get_mpi_type(bitmap[0]),
					recv_buffer, shared_bitmap_width, get_mpi_type(bitmap[0]), mpi.comm_y);
#endif
#if VERVOSE_MODE
			g_expand_bitmap_comm += shared_bitmap_width * mpi.size_y * sizeof(BitmapType);
#endif
		}
		if(mpi.isYdimAvailable()) s_.sync->barrier();
	}

	// expand visited bitmap and receive the current queue
	void expand_nq_bitmap() {
		TRACER(expand_vis_bmp);
		int bitmap_width = get_bitmap_size_local();
		BitmapType* const bitmap = (BitmapType*)new_visited_;
		BitmapType* recv_buffer = shared_visited_;
#if ENABLE_MY_ALLGATHER
		if(mpi.isYdimAvailable()) {
			if(mpi.isMaster()) print_with_prefix("Error: MY_ALLGATHER does not support shared memory Y dimension.");
		}
#if ENABLE_MY_ALLGATHER == 1
	MpiCol::my_allgather(bitmap, bitmap_width, recv_buffer, mpi.comm_r);
#else
	my_allgather_2d(bitmap, bitmap_width, recv_buffer, mpi.comm_r);
#endif // #if ENABLE_MY_ALLGATHER == 1
#else
		MPI_Allgather(bitmap, bitmap_width, get_mpi_type(bitmap[0]),
				recv_buffer, bitmap_width, get_mpi_type(bitmap[0]), mpi.comm_2dr);
#endif
#if VERVOSE_MODE
		g_expand_bitmap_comm += bitmap_width * mpi.size_2dc * sizeof(BitmapType);
#endif
	}

	int expand_visited_list(int node_nq_size) {
		TRACER(expand_vis_list);
		if(mpi.rank_z == 0 && mpi.comm_y != MPI_COMM_NULL) {
			s_.offset[0] = MpiCol::allgatherv((TwodVertex*)visited_buffer_orig_,
					 nq_recv_buf_, node_nq_size, mpi.comm_y, mpi.size_y);
			VERVOSE(g_expand_list_comm += s_.offset[0] * sizeof(TwodVertex));
		}
		if(mpi.isYdimAvailable()) s_.sync->barrier();
		return s_.offset[0];
	}

	int top_down_make_nq_list(bool with_z, TwodVertex shifted_rc, int bitmap_width, TwodVertex* outbuf) {
		TRACER(td_make_nq_list);
		int size_z = with_z ? mpi.size_z : 1;
		int rank_z = with_z ? mpi.rank_z : 0;

		const int max_threads = omp_get_max_threads();
		const int node_threads = max_threads * size_z;

		int th_offset_storage[max_threads+1];
		int *th_offset = with_z ? s_.offset : th_offset_storage;

		int result_size = 0;
		int num_buffers = nq_.stack_.size();
#pragma omp parallel
		{
			int tid = omp_get_thread_num() + max_threads * rank_z;
			int count = 0;
#pragma omp for schedule(static) nowait
			for(int i = 0; i < num_buffers; ++i) {
				count += nq_.stack_[i]->length;
			}
			th_offset[tid+1] = count;
#pragma omp barrier
#pragma omp single
			{
				if(with_z) s_.sync->barrier();
				if(rank_z == 0) {
					th_offset[0] = 0;
					for(int i = 0; i < node_threads; ++i) {
						th_offset[i+1] += th_offset[i];
					}
					assert (th_offset[node_threads] <= int(bitmap_width*sizeof(BitmapType)/sizeof(TwodVertex)*size_z));
				}
				if(with_z) s_.sync->barrier();
				result_size = th_offset[node_threads];
			} // implicit barrier

			int offset = th_offset[tid];
#pragma omp for schedule(static) nowait
			for(int i = 0; i < num_buffers; ++i) {
				int len = nq_.stack_[i]->length;
				TwodVertex* src = nq_.stack_[i]->v;
				TwodVertex* dst = outbuf + offset;
				for(int c = 0; c < len; ++c) {
					dst[c] = src[c] | shifted_rc;
				}
				offset += len;
			}
			assert (offset == th_offset[tid+1]);
		} // implicit barrier
		if(with_z) s_.sync->barrier();

		return result_size;
	}

	void top_down_expand_nq_list(TwodVertex* nq, int nq_size) {
		TRACER(td_expand_nq_list);
		int comm_size = mpi.comm_r.size;
		int recv_size[comm_size];
		int recv_off[comm_size+1];
		MPI_Allgather(&nq_size, 1, MPI_INT, recv_size, 1, MPI_INT, mpi.comm_r.comm);
		recv_off[0] = 0;
		for(int i = 0; i < comm_size; ++i) {
			recv_off[i+1] = recv_off[i] + recv_size[i];
		}
		cq_size_ = recv_off[comm_size];
		if(work_extra_buf_ != NULL) { free(work_extra_buf_); work_extra_buf_ = NULL; }
		TwodVertex* recv_buf = (TwodVertex*)((int64_t(cq_size_)*int64_t(sizeof(TwodVertex)) > work_buf_size_) ?
				(work_extra_buf_ = cache_aligned_xmalloc(cq_size_*sizeof(TwodVertex))) :
				work_buf_);
#if ENABLE_MY_ALLGATHER == 1
		MpiCol::my_allgatherv(nq, nq_size, recv_buf, recv_size, recv_off, mpi.comm_r);
#elif ENABLE_MY_ALLGATHER == 2
		my_allgatherv_2d(nq, nq_size, recv_buf, recv_size, recv_off, mpi.comm_r);
#else
		MPI_Allgatherv(nq, nq_size, MpiTypeOf<TwodVertex>::type,
				recv_buf, recv_size, recv_off, MpiTypeOf<TwodVertex>::type, mpi.comm_r.comm);
#endif
		VERVOSE(g_expand_list_comm += cq_size_ * sizeof(TwodVertex));
		cq_list_ = recv_buf;
	}

	void top_down_expand() {
		TRACER(td_expand);
		// expand NQ within a processor column
		// convert NQ to a SRC format
		TwodVertex shifted_c = TwodVertex(mpi.rank_2dc) << graph_.local_bits_;
		int bitmap_width = get_bitmap_size_local();
		// old_visited is used as a temporal buffer
		TwodVertex* nq_list = (TwodVertex*)old_visited_;
		int nq_size = top_down_make_nq_list(false, shifted_c, bitmap_width, nq_list);
		top_down_expand_nq_list(nq_list, nq_size);
	}

	void top_down_switch_expand(bool bitmap_or_list) {
		TRACER(td_sw_expand);
		// expand NQ within a processor row
		if(bitmap_or_list) {
			// bitmap
			expand_visited_bitmap();
		}
		else {
			throw "Not implemented";
		}
	}

	template <typename BitmapF>
	int make_list_from_bitmap(bool with_z, TwodVertex shifted_rc, BitmapF bmp, int bitmap_width, TwodVertex* outbuf) {
		int size_z = with_z ? mpi.size_z : 1;
		int rank_z = with_z ? mpi.rank_z : 0;

		const int max_threads = omp_get_max_threads();
		const int node_threads = max_threads * size_z;

		int th_offset_storage[max_threads+1];
		int *th_offset = with_z ? s_.offset : th_offset_storage;

		int result_size = 0;
#pragma omp parallel
		{
			int tid = omp_get_thread_num() + max_threads * rank_z;
			int count = 0;
#pragma omp for schedule(static) nowait
			for(int i = 0; i < bitmap_width; ++i) {
				count += __builtin_popcountl(bmp(i));
			}
			th_offset[tid+1] = count;
#pragma omp barrier
#pragma omp single
			{
				if(with_z) s_.sync->barrier();
				if(rank_z == 0) {
					th_offset[0] = 0;
					for(int i = 0; i < node_threads; ++i) {
						th_offset[i+1] += th_offset[i];
					}
					assert (th_offset[node_threads] <= int(bitmap_width*sizeof(BitmapType)/sizeof(TwodVertex)*size_z));
				}
				if(with_z) s_.sync->barrier();
				result_size = th_offset[node_threads];
			} // implicit barrier

			TwodVertex* dst = outbuf + th_offset[tid];
#pragma omp for schedule(static) nowait
			for(int i = 0; i < bitmap_width; ++i) {
				BitmapType bmp_i = bmp(i);
				while(bmp_i != 0) {
					TwodVertex bit_idx = __builtin_ctzl(bmp_i);
					*(dst++) = (i * NBPE + bit_idx) | shifted_rc;
					bmp_i &= bmp_i - 1;
				}
			}
			assert ((dst - outbuf) == th_offset[tid+1]);
		} // implicit barrier
		if(with_z) s_.sync->barrier();

		return result_size;
	}

	struct NQBitmapCombiner {
		BitmapType* new_visited;
		BitmapType* old_visited;
		NQBitmapCombiner(ThisType* this__)
			: new_visited((BitmapType*)this__->new_visited_)
			, old_visited((BitmapType*)this__->old_visited_) { }
		BitmapType operator ()(int i) { return new_visited[i] & ~(old_visited[i]); }
	};

	int bottom_up_make_nq_list(bool with_z, TwodVertex shifted_rc, TwodVertex* outbuf) {
		TRACER(bu_make_nq_list);
		const int bitmap_width = get_bitmap_size_local();
		int node_nq_size;

		if(bitmap_or_list_) {
			NQBitmapCombiner NQBmp(this);
			node_nq_size = make_list_from_bitmap(with_z, shifted_rc, NQBmp, bitmap_width, outbuf);
		}
		else {
			int size_z = with_z ? mpi.size_z : 1;
			int rank_z = with_z ? mpi.rank_z : 0;
			TwodVertex* new_vis_p[BU_SUBSTEP];
			get_visited_pointers(new_vis_p, BU_SUBSTEP, new_visited_, BU_SUBSTEP);
			TwodVertex* old_vis_p[BU_SUBSTEP];
			get_visited_pointers(old_vis_p, BU_SUBSTEP, old_visited_, BU_SUBSTEP);
			const int num_parts = BU_SUBSTEP * size_z;
			int part_offset_storage[num_parts+1];
			int *part_offset = with_z ? s_.offset : part_offset_storage;

			for(int i = 0; i < BU_SUBSTEP; ++i) {
				part_offset[rank_z*BU_SUBSTEP+1+i] = old_visited_list_size_[i] - new_visited_list_size_[i];
			}
			if(with_z) s_.sync->barrier();
			if(rank_z == 0) {
				part_offset[0] = 0;
				for(int i = 0; i < num_parts; ++i) {
					part_offset[i+1] += part_offset[i];
				}
				assert (part_offset[num_parts] <= int(bitmap_width*sizeof(BitmapType)/sizeof(TwodVertex)*2));
			}
			if(with_z) s_.sync->barrier();
			node_nq_size = part_offset[num_parts];

			bool mt = int(node_nq_size*sizeof(TwodVertex)) > 16*1024*mpi.size_z;
#ifndef NDEBUG
			int dbg_inc0 = 0, dbg_exc0 = 0, dbg_inc1 = 0, dbg_exc1 = 0;
#pragma omp parallel if(mt) reduction(+:dbg_inc0, dbg_exc0, dbg_inc1, dbg_exc1)
#else
#pragma omp parallel if(mt)
#endif
			for(int i = 0; i < BU_SUBSTEP; ++i) {
				int max_threads = omp_get_num_threads(); // Place here because this region may be executed sequential.
				int tid = omp_get_thread_num();
				TwodVertex substep_base = graph_.num_local_verts_ / BU_SUBSTEP * i;
				TwodVertex* dst = outbuf + part_offset[rank_z*BU_SUBSTEP+i];
				TwodVertex *new_vis = new_vis_p[i], *old_vis = old_vis_p[i];
				int start, end, old_size = old_visited_list_size_[i], new_size = new_visited_list_size_[i];
				get_partition(old_size, max_threads, tid, start, end);
				int new_vis_start = std::lower_bound(
						new_vis, new_vis + new_size, old_vis[start]) - new_vis;
				int new_vis_off = new_vis_start;
				int dst_off = start - new_vis_start;

				for(int c = start; c < end; ++c) {
					if(new_vis[new_vis_off] == old_vis[c]) {
						++new_vis_off;
					}
					else {
						dst[dst_off++] = (old_vis[c] + substep_base) | shifted_rc;
					}
				}

#ifndef NDEBUG
				if(i == 0) {
					dbg_inc0 += dst_off - (start - new_vis_start);
					dbg_exc0 += new_vis_off - new_vis_start;
				}
				else if(i == 1) {
					dbg_inc1 += dst_off - (start - new_vis_start);
					dbg_exc1 += new_vis_off - new_vis_start;
				}
#endif
			}
#ifndef NDEBUG
			assert(dbg_inc0 == old_visited_list_size_[0] - new_visited_list_size_[0]);
			assert(dbg_exc0 == new_visited_list_size_[0]);
			assert(dbg_inc1 == old_visited_list_size_[1] - new_visited_list_size_[1]);
			assert(dbg_exc1 == new_visited_list_size_[1]);
#endif
			if(with_z) s_.sync->barrier();
		}

		return node_nq_size;
	}

	void bottom_up_expand_nq_list() {
		TRACER(bu_expand_nq_list);
		assert (mpi.isYdimAvailable() || (visited_buffer_orig_ == visited_buffer_));
		int lgl = graph_.local_bits_;
		int L = graph_.num_local_verts_;

		int node_nq_size = bottom_up_make_nq_list(
				mpi.isYdimAvailable(), TwodVertex(mpi.rank_2dr) << lgl, (TwodVertex*)visited_buffer_orig_);

		int recv_nq_size = expand_visited_list(node_nq_size);

#pragma omp parallel if(recv_nq_size > 1024*16)
		{
			const int max_threads = omp_get_num_threads(); // Place here because this region may be executed sequential.
			const int node_threads = max_threads * mpi.size_z;
			int tid = omp_get_thread_num() + max_threads * mpi.rank_z;
			int64_t begin, end;
			get_partition(recv_nq_size, nq_recv_buf_, get_msb_index(NBPE), node_threads, tid, begin, end);

			for(int i = begin; i < end; ++i) {
				SeparatedId dst(nq_recv_buf_[i]);
				TwodVertex compact = dst.compact(lgl, L);
				TwodVertex word_idx = compact >> LOG_NBPE;
				int bit_idx = compact & NBPE_MASK;
				shared_visited_[word_idx] |= BitmapType(1) << bit_idx;
			}
		} // implicit barrier
		if(mpi.isYdimAvailable()) s_.sync->barrier();
	}

	void bottom_up_expand(bool bitmap_or_list) {
		TRACER(bu_expand);
		if(bitmap_or_list) {
			// bitmap
			assert (bitmap_or_list_);
			expand_visited_bitmap();
		}
		else {
			// list
			bottom_up_expand_nq_list();
		}
	}

	void bottom_up_switch_expand(bool bitmap_or_list) {
		TRACER(bu_sw_expand);
		if(bitmap_or_list) {
			// new_visited - old_visited = nq
			const int bitmap_width = get_bitmap_size_local();
			BitmapType* new_visited = (BitmapType*)new_visited_;
			BitmapType* old_visited = (BitmapType*)old_visited_;
#pragma omp parallel for
			for(int i = 0; i < bitmap_width; ++i) {
				new_visited[i] &= ~(old_visited[i]);
			}
			// expand nq
			expand_nq_bitmap();
		}
		else {
			// visited_buffer_ is used as a temporal buffer
			TwodVertex* nq_list = (TwodVertex*)visited_buffer_;
			int nq_size = bottom_up_make_nq_list(
					false, TwodVertex(mpi.rank_2dc) << graph_.local_bits_, (TwodVertex*)nq_list);
			top_down_expand_nq_list(nq_list, nq_size);
		}
	}

	//-------------------------------------------------------------//
	// top-down search
	//-------------------------------------------------------------//

	void top_down_send(int64_t tgt, int lgl, int r_mask,
			LocalPacket* packet_array, int64_t src
#if PROFILING_MODE
			, profiling::TimeSpan& ts_commit
#endif
	) {
		int dest = (tgt >> lgl) & r_mask;
		LocalPacket& pk = packet_array[dest];
		if(pk.length > LocalPacket::TOP_DOWN_LENGTH-3) { // low probability
			PROF(profiling::TimeKeeper tk_commit);
			td_comm_.put(pk.data.t, pk.length, dest);
			PROF(ts_commit += tk_commit);
			pk.src = -1;
			pk.length = 0;
		}
		if(pk.src != src) { // TODO: use conditional branch
			pk.src = src;
			pk.data.t[pk.length++] = (src >> 32) | 0x80000000u;
			pk.data.t[pk.length++] = (uint32_t)src;
		}
		pk.data.t[pk.length++] = tgt & ((uint32_t(1) << lgl) - 1);
	}

	void top_down_send_large(int64_t* edge_array, int64_t start, int64_t end,
			int lgl, int r_mask, int64_t src)
	{
		assert (end > start);
		for(int i = 0; i < mpi.size_2dr; ++i) {
			if(start >= end) break;
			int s_dest = (edge_array[start] >> lgl) & r_mask;
			if(s_dest > i) continue;

			// search the destination change point with binary search
			int64_t left = start;
			int64_t right = end;
			int64_t next = std::min(left + (right - left) / (mpi.size_2dr - i) * 2, end - 1);
			do {
				int dest = (edge_array[next] >> lgl) & r_mask;
				if(dest > i) {
					right = next;
				}
				else {
					left = next;
				}
				next = (left + right) / 2;
			} while(left < next);
			// start ... right -> i
			td_comm_.put_ptr(edge_array + start, right - start, src, i);
			start = right;
		}
		assert(start == end);
	}

	void top_down_parallel_section(bool bitmap_or_list) {
		TRACER(td_par_sec);
		PROF(profiling::TimeKeeper tk_all);
		bool clear_packet_buffer = packet_buffer_is_dirty_;
		packet_buffer_is_dirty_ = false;

#if TOP_DOWN_SEND_LB == 2
#define IF_LARGE_EDGE if(e_end - e_start > PRM::TOP_DOWN_PENDING_WIDTH/10)
#define ELSE else
#else
#define IF_LARGE_EDGE
#define ELSE
#endif

		debug("begin parallel");
#pragma omp parallel
		{
			SET_OMP_AFFINITY;
			PROF(profiling::TimeKeeper tk_all);
			PROF(profiling::TimeSpan ts_commit);
			VERVOSE(int64_t num_edge_relax = 0);
			VERVOSE(int64_t num_large_edge = 0);
			//int max_threads = omp_get_num_threads();
			int64_t* edge_array = graph_.edge_array_;
			LocalPacket* packet_array =
					thread_local_buffer_[omp_get_thread_num()]->fold_packet;
			if(clear_packet_buffer) {
				for(int target = 0; target < mpi.size_2dr; ++target) {
					packet_array[target].src = -1;
					packet_array[target].length = 0;
				}
			}
			int lgl = graph_.local_bits_;
			//int vertex_bits = graph_.r_bits_ + lgl;
			int r_mask = (1 << graph_.r_bits_) - 1;
			int P = mpi.size_2d;
			int R = mpi.size_2dr;
			int r = mpi.rank_2dr;
			uint32_t local_mask = (uint32_t(1) << lgl) - 1;
			int64_t L = graph_.num_local_verts_;

			// count total edges
			if(bitmap_or_list) {
				BitmapType* cq_bitmap = shared_visited_;
				int64_t bitmap_size = get_bitmap_size_local() * mpi.size_2dc;
	#pragma omp for
				for(int64_t word_idx = 0; word_idx < bitmap_size; ++word_idx) {
					BitmapType cq_bit_i = cq_bitmap[word_idx];
					if(cq_bit_i == BitmapType(0)) continue;

					BitmapType row_bitmap_i = graph_.row_bitmap_[word_idx];
					BitmapType bit_flags = cq_bit_i & row_bitmap_i;
					TwodVertex bmp_row_sums = graph_.row_sums_[word_idx];
					while(bit_flags != BitmapType(0)) {
						BitmapType cq_bit = bit_flags & (-bit_flags);
						BitmapType low_mask = cq_bit - 1;
						bit_flags &= ~cq_bit;
						int bit_idx = __builtin_popcountl(low_mask);
						TwodVertex compact = word_idx * NBPE + bit_idx;
						TwodVertex src_c = word_idx / get_bitmap_size_local(); // TODO:
						TwodVertex non_zero_off = bmp_row_sums + __builtin_popcountl(row_bitmap_i & low_mask);
						int64_t src_orig =
								int64_t(graph_.orig_vertexes_[non_zero_off]) * P + src_c * R + r;
	#if ISOLATE_FIRST_EDGE
						top_down_send(graph_.isolated_edges_[non_zero_off], lgl,
								r_mask, packet_array, src_orig
	#if PROFILING_MODE
						, ts_commit
	#endif
							);
	#endif // #if ISOLATE_FIRST_EDGE
						int64_t e_start = graph_.row_starts_[non_zero_off];
						int64_t e_end = graph_.row_starts_[non_zero_off+1];
						IF_LARGE_EDGE
#if TOP_DOWN_SEND_LB > 0
						{
							top_down_send_large(edge_array, e_start, e_end, lgl, r_mask, src_orig);
							VERVOSE(num_large_edge += e_end - e_start);
						}
#endif // #if TOP_DOWN_SEND_LB > 0
						ELSE
#if TOP_DOWN_SEND_LB != 1
						{
							for(int64_t e = e_start; e < e_end; ++e) {
								top_down_send(edge_array[e], lgl,
										r_mask, packet_array, src_orig
#if PROFILING_MODE
								, ts_commit
#endif
									);
							}
						}
#endif // #if TOP_DOWN_SEND_LB != 1
						VERVOSE(num_edge_relax += e_end - e_start + 1);
					} // while(bit_flags != BitmapType(0)) {
				} // #pragma omp for // implicit barrier
			}
			else {
				TwodVertex* cq_list = (TwodVertex*)cq_list_;
	#pragma omp for
				for(int64_t i = 0; i < int64_t(cq_size_); ++i) {
					SeparatedId src(cq_list[i]);
					TwodVertex src_c = src.value >> lgl;
					TwodVertex compact = src_c * L + (src.value & local_mask);
					TwodVertex word_idx = compact >> LOG_NBPE;
					int bit_idx = compact & NBPE_MASK;
					BitmapType row_bitmap_i = graph_.row_bitmap_[word_idx];
					BitmapType mask = BitmapType(1) << bit_idx;
					if(row_bitmap_i & mask) {
						BitmapType low_mask = (BitmapType(1) << bit_idx) - 1;
						TwodVertex non_zero_off = graph_.row_sums_[word_idx] +
								__builtin_popcountl(graph_.row_bitmap_[word_idx] & low_mask);
						int64_t src_orig =
								int64_t(graph_.orig_vertexes_[non_zero_off]) * P + src_c * R + r;
	#if ISOLATE_FIRST_EDGE
						top_down_send(graph_.isolated_edges_[non_zero_off], lgl,
								r_mask, packet_array, src_orig
	#if PROFILING_MODE
						, ts_commit
	#endif
							);
	#endif // #if ISOLATE_FIRST_EDGE
						int64_t e_start = graph_.row_starts_[non_zero_off];
						int64_t e_end = graph_.row_starts_[non_zero_off+1];
						IF_LARGE_EDGE
#if TOP_DOWN_SEND_LB > 0
						{
							top_down_send_large(edge_array, e_start, e_end, lgl, r_mask, src_orig);
							VERVOSE(num_large_edge += e_end - e_start);
						}
#endif // #if TOP_DOWN_SEND_LB > 0
						ELSE
#if TOP_DOWN_SEND_LB != 1
						{
							for(int64_t e = e_start; e < e_end; ++e) {
								top_down_send(edge_array[e], lgl,
										r_mask, packet_array, src_orig
#if PROFILING_MODE
								, ts_commit
#endif
									);
							}
						}
#endif // #if TOP_DOWN_SEND_LB != 1
						VERVOSE(num_edge_relax += e_end - e_start + 1);
					} // if(row_bitmap_i & mask) {
				} // #pragma omp for // implicit barrier
			}

			// flush buffer
#pragma omp for
			for(int target = 0; target < mpi.size_2dr; ++target) {
				for(int i = 0; i < omp_get_num_threads(); ++i) {
					LocalPacket* packet_array =
							thread_local_buffer_[i]->fold_packet;
					LocalPacket& pk = packet_array[target];
					if(pk.length > 0) {
						PROF(profiling::TimeKeeper tk_commit);
						td_comm_.put(pk.data.t, pk.length, target);
						PROF(ts_commit += tk_commit);
						pk.src = -1;
						pk.length = 0;
					}
				}
			} // #pragma omp for
			PROF(profiling::TimeSpan ts_all; ts_all += tk_all; ts_all -= ts_commit);
			PROF(extract_edge_time_ += ts_all);
			PROF(commit_time_ += ts_commit);
			VERVOSE(__sync_fetch_and_add(&num_edge_top_down_, num_edge_relax));
			VERVOSE(__sync_fetch_and_add(&num_td_large_edge_, num_large_edge));
		} // #pragma omp parallel reduction(+:num_edge_relax)
#undef IF_LARGE_EDGE
#undef ELSE
		PROF(parallel_reg_time_ += tk_all);
		debug("finished parallel");
	}

	void top_down_search() {
		TRACER(td);

		td_comm_.prepare();
		top_down_parallel_section(bitmap_or_list_);
		td_comm_.run_with_ptr();

		PROF(profiling::TimeKeeper tk_all);
		// flush NQ buffer and count NQ total
		int max_threads = omp_get_max_threads();
		nq_size_ = nq_.stack_.size() * QueuedVertexes::SIZE;
		for(int tid = 0; tid < max_threads; ++tid) {
			ThreadLocalBuffer* tlb = thread_local_buffer_[tid];
			QueuedVertexes* buf = tlb->cur_buffer;
			if(buf != NULL) {
				nq_size_ += buf->length;
				nq_.push(buf);
			}
			tlb->cur_buffer = NULL;
		}
		int64_t send_nq_size = nq_size_;
		PROF(seq_proc_time_ += tk_all);
		PROF(MPI_Barrier(mpi.comm_2d));
		PROF(fold_competion_wait_ += tk_all);
		MPI_Allreduce(&nq_size_, &max_nq_size_, 1, MPI_INT, MPI_MAX, mpi.comm_2d);
		MPI_Allreduce(&send_nq_size, &global_nq_size_, 1, MpiTypeOf<int64_t>::type, MPI_SUM, mpi.comm_2d);
		PROF(gather_nq_time_ += tk_all);
	}


	template <bool growing>
	void top_down_row_receive(TopDownRow* rows, int num_rows) {
		int num_threads = omp_get_max_threads();
		int num_splits = num_threads * 8;
		// process from tail because computation cost is higher in tail
		volatile int procces_counter = num_splits - 1;
		//volatile int procces_counter = 0;

#pragma omp parallel
		{
			TRACER(td_recv);
			PROF(profiling::TimeKeeper tk_all);
			int tid = omp_get_thread_num();
			ThreadLocalBuffer* tlb = thread_local_buffer_[tid];
			QueuedVertexes* buf = tlb->cur_buffer;
			if(buf == NULL) buf = nq_empty_buffer_.get();
			BitmapType* visited = (BitmapType*)new_visited_;
			int64_t* restrict const pred = pred_;
			LocalVertex* invert_map = graph_.invert_map_;

			// for id converter //
			int lgl = graph_.local_bits_;
			LocalVertex lmask = (LocalVertex(1) << lgl) - 1;
			// ------------------- //

			while(true) {
				int split = __sync_fetch_and_add(&procces_counter, -1);
				if(split < 0) break;
				//int split = __sync_fetch_and_add(&procces_counter, 1);
				//if(split >= num_splits) break;

				for(int r = 0; r < num_rows; ++r) {
					uint32_t* ptr = rows[r].ptr;
					int length = rows[r].length;
					int64_t pred_v = rows[r].src | (int64_t(current_level_) << 48);

					int width_per_split = (length + num_splits - 1) / num_splits;
					int off_start = std::min(length, width_per_split * split);
					int off_end = std::min(length, off_start + width_per_split);

					for(int i = off_start; i < off_end; ++i) {
						LocalVertex tgt_local = ptr[i] & lmask;
						if(growing) {
							// TODO: which is better ?
							//LocalVertex tgt_orig = invert_map[tgt_local];
							const TwodVertex word_idx = tgt_local >> LOG_NBPE;
							const int bit_idx = tgt_local & NBPE_MASK;
							const BitmapType mask = BitmapType(1) << bit_idx;
							if((visited[word_idx] & mask) == 0) { // if this vertex has not visited
								if((__sync_fetch_and_or(&visited[word_idx], mask) & mask) == 0) {
									LocalVertex tgt_orig = invert_map[tgt_local];
									assert (pred[tgt_orig] == -1);
									pred[tgt_orig] = pred_v;
									if(buf->full()) {
										nq_.push(buf); buf = nq_empty_buffer_.get();
									}
									buf->append_nocheck(tgt_local);
								}
							}
						}
						else {
							LocalVertex tgt_orig = invert_map[tgt_local];
							if(pred[tgt_orig] == -1) {
								if(__sync_bool_compare_and_swap(&pred[tgt_orig], -1, pred_v)) {
									if(buf->full()) {
										nq_.push(buf); buf = nq_empty_buffer_.get();
									}
									buf->append_nocheck(tgt_local);
								}
							}
						}
					}
				}
			}
			tlb->cur_buffer = buf;
			PROF(recv_proc_thread_large_time_ += tk_all);
		} // #pragma omp parallel
	}

	template <bool growing>
	void top_down_receive(uint32_t* stream, int length, TopDownRow* rows, volatile int* num_rows) {
		TRACER(td_recv);
		PROF(profiling::TimeKeeper tk_all);

		ThreadLocalBuffer* tlb = thread_local_buffer_[omp_get_thread_num()];
		QueuedVertexes* buf = tlb->cur_buffer;
		if(buf == NULL) buf = nq_empty_buffer_.get();
		BitmapType* visited = (BitmapType*)new_visited_;
		int64_t* restrict const pred = pred_;
		const int cur_level = current_level_;
		int64_t pred_v = -1;
		LocalVertex* invert_map = graph_.invert_map_;

		// for id converter //
		int lgl = graph_.local_bits_;
		LocalVertex lmask = (LocalVertex(1) << lgl) - 1;
		// ------------------- //

		for(int i = 0; i < length; ++i) {
			uint32_t v = stream[i];
			if(v & 0x80000000u) {
				int64_t src = (int64_t(v & 0xFFFF) << 32) | stream[i+1];
				pred_v = src | (int64_t(cur_level) << 48);
				if(v & 0x40000000u) {
					int length_i = stream[i+2];
#if TOP_DOWN_RECV_LB
					if(length_i < PRM::TOP_DOWN_PENDING_WIDTH)
#endif // #if TOP_DOWN_RECV_LB
					{
						assert (pred_v != -1);
						for(int c = 0; c < length_i; ++c) {
							LocalVertex tgt_local = stream[i+3+c] & lmask;
							if(growing) {
								// TODO: which is better ?
								//LocalVertex tgt_orig = invert_map[tgt_local];
								const TwodVertex word_idx = tgt_local >> LOG_NBPE;
								const int bit_idx = tgt_local & NBPE_MASK;
								const BitmapType mask = BitmapType(1) << bit_idx;
								if((visited[word_idx] & mask) == 0) { // if this vertex has not visited
									if((__sync_fetch_and_or(&visited[word_idx], mask) & mask) == 0) {
										LocalVertex tgt_orig = invert_map[tgt_local];
										assert (pred[tgt_orig] == -1);
										pred[tgt_orig] = pred_v;
										if(buf->full()) {
											nq_.push(buf); buf = nq_empty_buffer_.get();
										}
										buf->append_nocheck(tgt_local);
									}
								}
							}
							else {
								LocalVertex tgt_orig = invert_map[tgt_local];
								if(pred[tgt_orig] == -1) {
									if(__sync_bool_compare_and_swap(&pred[tgt_orig], -1, pred_v)) {
										if(buf->full()) {
											nq_.push(buf); buf = nq_empty_buffer_.get();
										}
										buf->append_nocheck(tgt_local);
									}
								}
							}
						}
					}
#if TOP_DOWN_RECV_LB
					else {
						int put_off = __sync_fetch_and_add(num_rows, 1);
						rows[put_off].length = length_i;
						rows[put_off].ptr = &stream[i+3];
						rows[put_off].src = src;
					}
#endif // #if TOP_DOWN_RECV_LB

					i += 2 + length_i;
				}
				else {
					i += 1;
				}
			}
			else {
				assert (pred_v != -1);

				LocalVertex tgt_local = v & lmask;
				if(growing) {
					// TODO: which is better ?
					//LocalVertex tgt_orig = invert_map[tgt_local];
					const TwodVertex word_idx = tgt_local >> LOG_NBPE;
					const int bit_idx = tgt_local & NBPE_MASK;
					const BitmapType mask = BitmapType(1) << bit_idx;
					if((visited[word_idx] & mask) == 0) { // if this vertex has not visited
						if((__sync_fetch_and_or(&visited[word_idx], mask) & mask) == 0) {
							LocalVertex tgt_orig = invert_map[tgt_local];
							assert (pred[tgt_orig] == -1);
							pred[tgt_orig] = pred_v;
							if(buf->full()) {
								nq_.push(buf); buf = nq_empty_buffer_.get();
							}
							buf->append_nocheck(tgt_local);
						}
					}
				}
				else {
					LocalVertex tgt_orig = invert_map[tgt_local];
					if(pred[tgt_orig] == -1) {
						if(__sync_bool_compare_and_swap(&pred[tgt_orig], -1, pred_v)) {
							if(buf->full()) {
								nq_.push(buf); buf = nq_empty_buffer_.get();
							}
							buf->append_nocheck(tgt_local);
						}
					}
				}
			}
		}
		tlb->cur_buffer = buf;
		PROF(recv_proc_thread_time_ += tk_all);
	}

	//-------------------------------------------------------------//
	// bottom-up search
	//-------------------------------------------------------------//

	void botto_up_print_stt(int64_t num_blocks, int64_t num_vertexes, int* nq_count) {
		int64_t send_stt[2] = { num_vertexes, num_blocks };
		int64_t sum_stt[2];
		int64_t max_stt[2];
		MPI_Reduce(send_stt, sum_stt, 2, MpiTypeOf<int64_t>::type, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(send_stt, max_stt, 2, MpiTypeOf<int64_t>::type, MPI_MAX, 0, MPI_COMM_WORLD);
		if(mpi.isMaster() && sum_stt[0] != 0) {
			print_with_prefix("Bottom-Up using List. Total %f M Vertexes / %f M Blocks = %f Max %f %%+ Vertexes %f %%+ Blocks",
					to_mega(sum_stt[0]), to_mega(sum_stt[1]), to_mega(sum_stt[0]) / to_mega(sum_stt[1]),
					diff_percent(max_stt[0], sum_stt[0], mpi.size_2d),
					diff_percent(max_stt[1], sum_stt[1], mpi.size_2d));
		}
		int count_length = mpi.size_2dc;
		int start_proc = mpi.rank_2dc;
		int size_mask = count_length - 1;
		int64_t phase_count[count_length];
		int64_t phase_recv[count_length];
		for(int i = 0; i < count_length; ++i) {
			phase_count[i] = nq_count[(start_proc + i) & size_mask];
		}
		MPI_Reduce(phase_count, phase_recv, count_length, MpiTypeOf<int64_t>::type, MPI_SUM, 0, MPI_COMM_WORLD);
		if(mpi.isMaster()) {
			int64_t total_nq = 0;
			for(int i = 0; i < count_length; ++i) {
				total_nq += phase_recv[i];
			}
			print_with_prefix("Bottom-Up: %"PRId64" vertexes found. Break down ...", total_nq);
			for(int i = 0; i < count_length; ++i) {
				print_with_prefix("step %d / %d  %f M Vertexes ( %f %% )",
						i+1, count_length, to_mega(phase_recv[i]), (double)phase_recv[i] / (double)total_nq * 100.0);
			}
		}
	}

	struct UnvisitedBitmapFunctor {
		BitmapType* visited;
		UnvisitedBitmapFunctor(BitmapType* visited__)
			: visited(visited__) { }
		BitmapType operator()(int i) { return ~(visited[i]); }
	};

	void swap_visited_memory(bool prev_bitmap_or_list) {
		// ----> Now, new_visited_ has current VIS
		if(bitmap_or_list_) { // bitmap ?
			// Since the VIS bitmap is modified in the search function,
			// we copy the VIS to the working memory to avoid corrupting the VIS bitmap.
			// bottom_up_bitmap search function begins with work_buf_ that contains current VIS bitmap.
			int64_t bitmap_width = get_bitmap_size_local();
			memory::copy_mt(work_buf_, new_visited_, bitmap_width*sizeof(BitmapType));
		}
		std::swap(new_visited_, old_visited_);
		// ----> Now, old_visited_ has current VIS
		if(!bitmap_or_list_) { // list ?
			if(prev_bitmap_or_list) { // previous level is performed with bitmap ?
				// make list from bitmap
				int step_bitmap_width = get_bitmap_size_local() / BU_SUBSTEP;
				BitmapType* bmp_vis_p[4]; get_visited_pointers(bmp_vis_p, 4, old_visited_, BU_SUBSTEP);
				TwodVertex* list_vis_p[4]; get_visited_pointers(list_vis_p, 4, new_visited_, BU_SUBSTEP);
				//int64_t shifted_c = int64_t(mpi.rank_2dc) << graph_.lgl_;
				for(int i = 0; i < BU_SUBSTEP; ++i)
					new_visited_list_size_[i] = make_list_from_bitmap(
							false, 0, UnvisitedBitmapFunctor(bmp_vis_p[i]),
							step_bitmap_width, list_vis_p[i]);
				std::swap(new_visited_, old_visited_);
				// ----> Now, old_visited_ has current VIS in the list format.
			}
			for(int i = 0; i < BU_SUBSTEP; ++i)
				std::swap(new_visited_list_size_[i], old_visited_list_size_[i]);
		}
	}

	void flush_bottom_up_send_buffer(LocalPacket* buffer, int target_rank) {
		TRACER(flush);
		int bulk_send_size = BottomUpCommHandler::BUF_SIZE;
		for(int offset = 0; offset < buffer->length; offset += bulk_send_size) {
			int length = std::min(buffer->length - offset, bulk_send_size);
			bu_comm_.put(buffer->data.b + offset, length, target_rank);
		}
		buffer->length = 0;
	}

#if BFELL
	// returns the number of vertices found in this step.
	int bottom_up_search_bitmap_process_step(
			BitmapType* phase_bitmap,
			TwodVertex phase_bmp_off,
			TwodVertex half_bitmap_width)
	{
		USER_START(bu_bmp_step);
		struct BottomUpRow {
			SortIdx orig, sorted;
		};
		int target_rank = phase_bmp_off / half_bitmap_width / 2;
		int visited_count = 0;
		VERVOSE(int tmp_edge_relax = 0);
		PROF(profiling::TimeKeeper tk_all);
		PROF(profiling::TimeSpan ts_commit);
		ThreadLocalBuffer* tlb = thread_local_buffer_[omp_get_thread_num()];
#if STREAM_UPDATE
		LocalPacket& packet = tlb->fold_packet[target_rank];
		visited_count -= packet.length;
#else
		QueuedVertexes* buf = tlb->cur_buffer;
		if(buf == NULL) buf = nq_empty_buffer_.get();
		visited_count -= buf->length;
#endif

		TwodVertex num_blks = half_bitmap_width * NBPE / BFELL_SORT;
#pragma omp for nowait
		for(int64_t blk_idx = 0; blk_idx < int64_t(num_blks); ++blk_idx) {

			TwodVertex blk_bmp_off = blk_idx * BFELL_SORT_IN_BMP;
			BitmapType* blk_row_bitmap = graph_.row_bitmap_ + phase_bmp_off + blk_bmp_off;
			BitmapType* blk_bitmap = phase_bitmap + blk_bmp_off;
			TwodVertex* blk_row_sums = graph_.row_sums_ + phase_bmp_off + blk_bmp_off;
			SortIdx* sorted_idx = graph_.sorted_idx_;
			BottomUpRow rows[BFELL_SORT];
			int num_active_rows = 0;

			for(int bmp_idx = 0; bmp_idx < BFELL_SORT_IN_BMP; ++bmp_idx) {
				BitmapType row_bmp_i = blk_row_bitmap[bmp_idx];
				BitmapType unvis_i = ~(blk_bitmap[bmp_idx]) & row_bmp_i;
				if(unvis_i == BitmapType(0)) continue;
				TwodVertex bmp_row_sums = blk_row_sums[bmp_idx];
				do {
					uint32_t visb_idx = __builtin_ctzl(unvis_i);
					BitmapType mask = (BitmapType(1) << visb_idx) - 1;
					TwodVertex non_zero_idx = bmp_row_sums + __builtin_popcountl(row_bmp_i & mask);
					rows[num_active_rows].orig = bmp_idx * NBPE + visb_idx;
					rows[num_active_rows].sorted = sorted_idx[non_zero_idx];
					++num_active_rows;
					unvis_i &= unvis_i - 1;
				} while(unvis_i != BitmapType(0));
			}

			TwodVertex phase_blk_off = phase_bmp_off * NBPE / BFELL_SORT;
			int64_t edge_offset = graph_.blk_off[phase_blk_off + blk_idx].edge_start;
			int64_t length_start = graph_.blk_off[phase_blk_off + blk_idx].length_start;
			TwodVertex* col_edge_array = graph_.edge_array_ + edge_offset;
			SortIdx* col_len = graph_.col_len_ + length_start;
			TwodVertex blk_vertex_base = phase_bmp_off * NBPE + blk_idx * BFELL_SORT;

			int c = 0;
			for( ; num_active_rows > 0; ++c) {
				SortIdx next_col_len = col_len[c + 1];
				int i = num_active_rows - 1;
#if 0
				for( ; i >= 7; i -= 8) {
					BottomUpRow* cur_rows = &rows[i-7];
					TwodVertex src = {
							col_edge_array[cur_rows[0].sorted],
							col_edge_array[cur_rows[1].sorted],
							col_edge_array[cur_rows[2].sorted],
							col_edge_array[cur_rows[3].sorted],
							col_edge_array[cur_rows[4].sorted],
							col_edge_array[cur_rows[5].sorted],
							col_edge_array[cur_rows[6].sorted],
							col_edge_array[cur_rows[7].sorted],
					};
					bool connected = {
							shared_visited_[src[0] >> LOG_NBPE] & (BitmapType(1) << (src[0] & NBPE_MASK)),
							shared_visited_[src[1] >> LOG_NBPE] & (BitmapType(1) << (src[1] & NBPE_MASK)),
							shared_visited_[src[2] >> LOG_NBPE] & (BitmapType(1) << (src[2] & NBPE_MASK)),
							shared_visited_[src[3] >> LOG_NBPE] & (BitmapType(1) << (src[3] & NBPE_MASK)),
							shared_visited_[src[4] >> LOG_NBPE] & (BitmapType(1) << (src[4] & NBPE_MASK)),
							shared_visited_[src[5] >> LOG_NBPE] & (BitmapType(1) << (src[5] & NBPE_MASK)),
							shared_visited_[src[6] >> LOG_NBPE] & (BitmapType(1) << (src[6] & NBPE_MASK)),
							shared_visited_[src[7] >> LOG_NBPE] & (BitmapType(1) << (src[7] & NBPE_MASK)),
					};
					for(int s = 7; s >= 0; --s) {
						if(connected[s]) {
							// add to next queue
							int orig = cur_rows[s].orig;
							blk_bitmap[orig >> LOG_NBPE] |= (BitmapType(1) << (orig & NBPE_MASK));
#if STREAM_UPDATE
							if(packet.length == LocalPacket::BOTTOM_UP_LENGTH) {
								visited_count += packet.length;
								PROF(profiling::TimeKeeper tk_commit);
								comm_.send<false>(packet.data.b, packet.length, target_rank);
								PROF(ts_commit += tk_commit);
								packet.length = 0;
							}
							packet.data.b[packet.length+0] = src[s];
							packet.data.b[packet.length+1] = blk_vertex_base + orig;
							packet.length += 2;
#else // #if STREAM_UPDATE
							if(buf->full()) {
								nq_.push(buf); buf = nq_empty_buffer_.get();
							}
							buf->append_nocheck(src[s], blk_vertex_base + orig);
#endif // #if STREAM_UPDATE
							// end this row
							VERVOSE(tmp_edge_relax += c + 1);
							cur_rows[s] = rows[--num_active_rows];
						}
						else if(cur_rows[s].sorted >= next_col_len) {
							// end this row
							VERVOSE(tmp_edge_relax += c + 1);
							cur_rows[s] = rows[--num_active_rows];
						}
					}
				}
#elif 0
				for( ; i >= 3; i -= 4) {
					BottomUpRow* cur_rows = &rows[i-3];
					TwodVertex src[4] = {
							col_edge_array[cur_rows[0].sorted],
							col_edge_array[cur_rows[1].sorted],
							col_edge_array[cur_rows[2].sorted],
							col_edge_array[cur_rows[3].sorted],
					};
					bool connected[4] = {
							shared_visited_[src[0] >> LOG_NBPE] & (BitmapType(1) << (src[0] & NBPE_MASK)),
							shared_visited_[src[1] >> LOG_NBPE] & (BitmapType(1) << (src[1] & NBPE_MASK)),
							shared_visited_[src[2] >> LOG_NBPE] & (BitmapType(1) << (src[2] & NBPE_MASK)),
							shared_visited_[src[3] >> LOG_NBPE] & (BitmapType(1) << (src[3] & NBPE_MASK)),
					};
					for(int s = 0; s < 4; ++s) {
						if(connected[s]) { // TODO: use conditional branch
							// add to next queue
							int orig = cur_rows[s].orig;
							blk_bitmap[orig >> LOG_NBPE] |= (BitmapType(1) << (orig & NBPE_MASK));
#if STREAM_UPDATE
							if(packet.length == LocalPacket::BOTTOM_UP_LENGTH) {
								visited_count += packet.length;
								PROF(profiling::TimeKeeper tk_commit);
								comm_.send<false>(packet.data.b, packet.length, target_rank);
								PROF(ts_commit += tk_commit);
								packet.length = 0;
							}
							packet.data.b[packet.length+0] = src[s];
							packet.data.b[packet.length+1] = blk_vertex_base + orig;
							packet.length += 2;
#else // #if STREAM_UPDATE
							if(buf->full()) { // TODO: remove this branch
								nq_.push(buf); buf = nq_empty_buffer_.get();
							}
							buf->append_nocheck(src[s], blk_vertex_base + orig);
#endif // #if STREAM_UPDATE
							// end this row
							VERVOSE(tmp_edge_relax += c + 1);
							cur_rows[s] = rows[--num_active_rows];
						}
						else if(cur_rows[s].sorted >= next_col_len) {
							// end this row
							VERVOSE(tmp_edge_relax += c + 1);
							cur_rows[s] = rows[--num_active_rows];
						}
					}
				}
#endif
				for( ; i >= 0; --i) {
					SortIdx row = rows[i].sorted;
					TwodVertex src = col_edge_array[row];
					if(shared_visited_[src >> LOG_NBPE] & (BitmapType(1) << (src & NBPE_MASK))) {
						// add to next queue
						int orig = rows[i].orig;
						blk_bitmap[orig >> LOG_NBPE] |= (BitmapType(1) << (orig & NBPE_MASK));
#if STREAM_UPDATE
						if(packet.length == LocalPacket::BOTTOM_UP_LENGTH) {
							visited_count += packet.length;
							PROF(profiling::TimeKeeper tk_commit);
							comm_.send<false>(packet.data.b, packet.length, target_rank);
							PROF(ts_commit += tk_commit);
							packet.length = 0;
						}
						packet.data.b[packet.length+0] = src;
						packet.data.b[packet.length+1] = blk_vertex_base + orig;
						packet.length += 2;
#else
						if(buf->full()) {
							nq_.push(buf); buf = nq_empty_buffer_.get();
						}
						buf->append_nocheck(src, blk_vertex_base + orig);
#endif // #if STREAM_UPDATE
						// end this row
						VERVOSE(tmp_edge_relax += c + 1);
						rows[i] = rows[--num_active_rows];
					}
					else if(row >= next_col_len) {
						// end this row
						VERVOSE(tmp_edge_relax += c + 1);
						rows[i] = rows[--num_active_rows];
					}
				}
				col_edge_array += col_len[c];
			}
		} // #pragma omp for nowait

#if STREAM_UPDATE
		visited_count += packet.length;
#else
		tlb->cur_buffer = buf;
		visited_count += buf->length;
#endif
		PROF(profiling::TimeSpan ts_all(tk_all); ts_all -= ts_commit);
		PROF(extract_edge_time_ += ts_all);
		PROF(commit_time_ += ts_commit);
#if STREAM_UPDATE
		visited_count >>= 1;
#else
		int num_bufs_nq_after = nq_.stack_.size();
		for(int i = num_bufs_nq_before; i < num_bufs_nq_after; ++i) {
			visited_count += nq_.stack_[i]->length;
		}
#endif
		VERVOSE(__sync_fetch_and_add(&num_edge_bottom_up_, tmp_edge_relax));
		USER_END(bu_bmp_step);
		thread_sync_.barrier();
		return visited_count;
	}

	// returns the number of vertices found in this step.
	TwodVertex bottom_up_search_list_process_step(
#if VERVOSE_MODE
			int64_t& num_blocks,
#endif
			TwodVertex* phase_list,
			TwodVertex phase_size,
			int8_t* vertex_enabled,
			TwodVertex* write_list,
			TwodVertex phase_bmp_off,
			TwodVertex half_bitmap_width,
			int* th_offset)
	{
		TRACER(bu_list_step);
		int max_threads = omp_get_num_threads();
		int target_rank = phase_bmp_off / half_bitmap_width / 2;

		int tmp_num_blocks = 0;
		int tmp_edge_relax = 0;
		PROF(profiling::TimeKeeper tk_all);
		PROF(profiling::TimeSpan ts_commit);
		int tid = omp_get_thread_num();
		ThreadLocalBuffer* tlb = thread_local_buffer_[tid];
#if STREAM_UPDATE
		LocalPacket& packet = tlb->fold_packet[target_rank];
#else
		QueuedVertexes* buf = tlb->cur_buffer;
		if(buf == NULL) buf = nq_empty_buffer_.get();
#endif
		int num_enabled = 0;

		struct BottomUpRow {
			SortIdx orig, sorted, orig_i;
		};

		int64_t begin, end;
		get_partition(phase_size, phase_list, LOG_BFELL_SORT, max_threads, tid, begin, end);
		USER_START(bu_list_proc);
		for(int i = begin; i < end; ) {
			int blk_i_start = i;
			TwodVertex blk_idx = phase_list[i] >> LOG_BFELL_SORT;
			int num_active_rows = 0;
			BottomUpRow rows[BFELL_SORT];
			SortIdx* sorted_idx = graph_.sorted_idx_;
			TwodVertex* phase_row_sums = graph_.row_sums_ + phase_bmp_off;
			BitmapType* phase_row_bitmap = graph_.row_bitmap_ + phase_bmp_off;
			VERVOSE(tmp_num_blocks++);

			do {
				TwodVertex tgt = phase_list[i];
				TwodVertex word_idx = tgt >> LOG_NBPE;
				int bit_idx = tgt & NBPE_MASK;
				BitmapType mask = (BitmapType(1) << bit_idx);
				BitmapType row_bitmap_i = phase_row_bitmap[word_idx];
				if(row_bitmap_i & mask) { // I have edges for this vertex ?
					TwodVertex non_zero_idx = phase_row_sums[word_idx] +
							__builtin_popcountl(row_bitmap_i & (mask-1));
					rows[num_active_rows].orig = tgt & BFELL_SORT_MASK;
					rows[num_active_rows].orig_i = i - blk_i_start;
					rows[num_active_rows].sorted = sorted_idx[non_zero_idx];
					++num_active_rows;
				}
				else { // No, I do not have. -> pass through
					++num_enabled;
				}
				vertex_enabled[i] = 1;
			} while((phase_list[++i] >> LOG_BFELL_SORT) == blk_idx);
			assert(i <= end);

			TwodVertex phase_blk_off = phase_bmp_off * NBPE / BFELL_SORT;
			int64_t edge_offset = graph_.blk_off[phase_blk_off + blk_idx].edge_start;
			int64_t length_start = graph_.blk_off[phase_blk_off + blk_idx].length_start;
			TwodVertex* col_edge_array = graph_.edge_array_ + edge_offset;
			SortIdx* col_len = graph_.col_len_ + length_start;
			TwodVertex blk_vertex_base = phase_bmp_off * NBPE + blk_idx * BFELL_SORT;

			int c = 0;
			for( ; num_active_rows > 0; ++c) {
				SortIdx next_col_len = col_len[c + 1];
				int i = num_active_rows - 1;
				for( ; i >= 0; --i) {
					SortIdx row = rows[i].sorted;
					TwodVertex src = col_edge_array[row];
					if(shared_visited_[src >> LOG_NBPE] & (BitmapType(1) << (src & NBPE_MASK))) {
						// add to next queue
						int orig = rows[i].orig;
						vertex_enabled[blk_i_start + rows[i].orig_i] = 0;
#if STREAM_UPDATE
						if(packet.length == LocalPacket::BOTTOM_UP_LENGTH) {
							PROF(profiling::TimeKeeper tk_commit);
							comm_.send<false>(packet.data.b, packet.length, target_rank);
							PROF(ts_commit += tk_commit);
							packet.length = 0;
						}
						packet.data.b[packet.length+0] = src;
						packet.data.b[packet.length+1] = blk_vertex_base + orig;
						packet.length += 2;
#else
						if(buf->full()) {
							nq_.push(buf); buf = nq_empty_buffer_.get();
						}
						buf->append_nocheck(src, blk_vertex_base + orig);
#endif
						// end this row
						VERVOSE(tmp_edge_relax += c + 1);
						rows[i] = rows[--num_active_rows];
					}
					else if(row >= next_col_len) {
						// end this row
						VERVOSE(tmp_edge_relax += c + 1);
						rows[i] = rows[--num_active_rows];
						++num_enabled;
					}
				}
				col_edge_array += col_len[c];
			}
		}
		th_offset[tid+1] = num_enabled;
		USER_END(bu_list_proc);
		// TODO: measure wait time
		thread_sync_.barrier();
#pragma omp master
		{
			TRACER(bu_list_single);
			th_offset[0] = 0;
			for(int i = 0; i < max_threads; ++i) {
				th_offset[i+1] += th_offset[i];
			}
			assert (th_offset[max_threads] <= int(phase_size));
		}
		thread_sync_.barrier();

		USER_START(bu_list_write);
		// make new list to send
		int offset = th_offset[tid];

		for(int i = begin; i < end; ++i) {
			if(vertex_enabled[i]) {
				write_list[offset++] = phase_list[i];
			}
		}

		USER_END(bu_list_write);
#if !STREAM_UPDATE
		tlb->cur_buffer = buf;
#endif
		PROF(profiling::TimeSpan ts_all(tk_all); ts_all -= ts_commit);
		PROF(extract_edge_time_ += ts_all);
		PROF(commit_time_ += ts_commit);
		VERVOSE(__sync_fetch_and_add(&num_blocks, tmp_num_blocks));
		VERVOSE(__sync_fetch_and_add(&num_edge_bottom_up_, tmp_edge_relax));
		thread_sync_.barrier();
		return phase_size - th_offset[max_threads];
	}

#else // #if BFELL

	void bottom_up_search_bitmap_process_block(
			BitmapType* __restrict__ phase_bitmap,
			int off_start,
			int off_end,
			int phase_bmp_off,
			LocalPacket* buffer)
	{
		VERVOSE(int tmp_edge_relax = 0);

		int lgl = graph_.local_bits_;
		TwodVertex L = graph_.num_local_verts_;
		int r_bits = graph_.r_bits_;
		int orig_lgl = graph_.orig_local_bits_;

		const BitmapType* __restrict__ row_bitmap = graph_.row_bitmap_;
		const BitmapType* __restrict__ shared_visited = shared_visited_;
		const TwodVertex* __restrict__ row_sums = graph_.row_sums_;
		const int64_t* __restrict__ isolated_edges = graph_.isolated_edges_;
		const int64_t* __restrict__ row_starts = graph_.row_starts_;
		const LocalVertex* __restrict__ orig_vertexes = graph_.orig_vertexes_;
		const int64_t* __restrict__ edge_array = graph_.edge_array_;

		//TwodVertex lmask = (TwodVertex(1) << lgl) - 1;
		int num_send = 0;
#if CONSOLIDATE_IFE_PROC
		for(int64_t blk_bmp_off = off_start; blk_bmp_off < off_end; ++blk_bmp_off) {
			BitmapType row_bmp_i = *(row_bitmap + phase_bmp_off + blk_bmp_off);
			BitmapType visited_i = *(phase_bitmap + blk_bmp_off);
			TwodVertex bmp_row_sums = *(row_sums + phase_bmp_off + blk_bmp_off);
			BitmapType bit_flags = (~visited_i) & row_bmp_i;
			while(bit_flags != BitmapType(0)) {
				BitmapType vis_bit = bit_flags & (-bit_flags);
				BitmapType mask = vis_bit - 1;
				bit_flags &= ~vis_bit;
				int idx = __builtin_popcountl(mask);
				TwodVertex non_zero_idx = bmp_row_sums + __builtin_popcountl(row_bmp_i & mask);
				LocalVertex tgt_orig = orig_vertexes[non_zero_idx];
				// short cut
				int64_t src = isolated_edges[non_zero_idx];
				TwodVertex bit_idx = SeparatedId(SeparatedId(src).low(r_bits + lgl)).compact(lgl, L);
				if(shared_visited[bit_idx >> PRM::LOG_NBPE] & (BitmapType(1) << (bit_idx & PRM::NBPE_MASK))) {
					// add to next queue
					visited_i |= vis_bit;
					buffer->data.b[num_send++] = ((src >> lgl) << orig_lgl) | tgt_orig;
					// end this row
					VERVOSE(tmp_edge_relax += 1);
					continue;
				}
				int64_t e_start = row_starts[non_zero_idx];
				int64_t e_end = row_starts[non_zero_idx+1];
				for(int64_t e = e_start; e < e_end; ++e) {
					int64_t src = edge_array[e];
					TwodVertex bit_idx = SeparatedId(SeparatedId(src).low(r_bits + lgl)).compact(lgl, L);
					if(shared_visited[bit_idx >> PRM::LOG_NBPE] & (BitmapType(1) << (bit_idx & PRM::NBPE_MASK))) {
						// add to next queue
						visited_i |= vis_bit;
						buffer->data.b[num_send++] = ((src >> lgl) << orig_lgl) | tgt_orig;
						// end this row
						VERVOSE(tmp_edge_relax += e - e_start + 1);
						break;
					}
				}
			} // while(bit_flags != BitmapType(0)) {
			// write back
			*(phase_bitmap + blk_bmp_off) = visited_i;
		} // #pragma omp for

#else // #if CONSOLIDATE_IFE_PROC
		for(int64_t blk_bmp_off = off_start; blk_bmp_off < off_end; ++blk_bmp_off) {
			BitmapType row_bmp_i = *(row_bitmap + phase_bmp_off + blk_bmp_off);
			BitmapType visited_i = *(phase_bitmap + blk_bmp_off);
			TwodVertex bmp_row_sums = *(row_sums + phase_bmp_off + blk_bmp_off);
			BitmapType bit_flags = (~visited_i) & row_bmp_i;
			while(bit_flags != BitmapType(0)) {
				BitmapType vis_bit = bit_flags & (-bit_flags);
				BitmapType mask = vis_bit - 1;
				bit_flags &= ~vis_bit;
				int idx = __builtin_popcountl(mask);
				TwodVertex non_zero_idx = bmp_row_sums + __builtin_popcountl(row_bmp_i & mask);
				// short cut
				TwodVertex separated_src = isolated_edges[non_zero_idx];
				TwodVertex bit_idx = (separated_src >> lgl) * L + (separated_src & lmask);
				if(shared_visited[bit_idx >> PRM::LOG_NBPE] & (BitmapType(1) << (bit_idx & PRM::NBPE_MASK))) {
					// add to next queue
					visited_i |= vis_bit;
					buffer->data.b[num_send+0] = separated_src;
					buffer->data.b[num_send+1] = (phase_bmp_off + blk_bmp_off) * PRM::NBPE + idx;
					num_send += 2;
				}
			} // while(bit_flags != BitmapType(0)) {
			// write back
			*(phase_bitmap + blk_bmp_off) = visited_i;
		} // #pragma omp for

		for(int64_t blk_bmp_off = off_start; blk_bmp_off < off_end; ++blk_bmp_off) {
			BitmapType row_bmp_i = *(row_bitmap + phase_bmp_off + blk_bmp_off);
			BitmapType visited_i = *(phase_bitmap + blk_bmp_off);
			TwodVertex bmp_row_sums = *(row_sums + phase_bmp_off + blk_bmp_off);
			BitmapType bit_flags = (~visited_i) & row_bmp_i;
			while(bit_flags != BitmapType(0)) {
				BitmapType vis_bit = bit_flags & (-bit_flags);
				BitmapType mask = vis_bit - 1;
				bit_flags &= ~vis_bit;
				int idx = __builtin_popcountl(mask);
				TwodVertex non_zero_idx = bmp_row_sums + __builtin_popcountl(row_bmp_i & mask);
				int64_t e_start = row_starts[non_zero_idx];
				int64_t e_end = row_starts[non_zero_idx+1];
				for(int64_t e = e_start; e < e_end; ++e) {
					TwodVertex separated_src = edge_array[e];
					TwodVertex bit_idx = (separated_src >> lgl) * L + (separated_src & lmask);
					if(shared_visited[bit_idx >> PRM::LOG_NBPE] & (BitmapType(1) << (bit_idx & PRM::NBPE_MASK))) {
						// add to next queue
						visited_i |= vis_bit;
						buffer->data.b[num_send+0] = separated_src;
						buffer->data.b[num_send+1] = (phase_bmp_off + blk_bmp_off) * PRM::NBPE + idx;
						num_send += 2;
						// end this row
						break;
					}
				}
			} // while(bit_flags != BitmapType(0)) {
			// write back
			*(phase_bitmap + blk_bmp_off) = visited_i;
		} // #pragma omp for
#endif // #if CONSOLIDATE_IFE_PROC

		buffer->length = num_send;
		VERVOSE(__sync_fetch_and_add(&num_edge_bottom_up_, tmp_edge_relax));
	}

	// returns the number of vertices found in this step.
	int bottom_up_search_bitmap_process_step(
			BottomUpSubstepData& data,
			int step_bitmap_width,
			volatile int* process_counter,
			int target_rank)
	{
		USER_START(bu_bmp_step);

		BitmapType* phase_bitmap = (BitmapType*)data.data;
		int phase_bmp_off = data.tag.region_id * step_bitmap_width;
		//TwodVertex L = graph_.num_local_verts_;
		//TwodVertex phase_vertex_off = L / BU_SUBSTEP * (data.tag.region_id % BU_SUBSTEP);
		ThreadLocalBuffer* tlb = thread_local_buffer_[omp_get_thread_num()];
		LocalPacket* buffer = tlb->fold_packet;

		PROF(profiling::TimeKeeper tk_all);
		PROF(profiling::TimeSpan ts_commit);

#ifndef NDEBUG
		assert (buffer->length == 0);
		assert (*process_counter == 0);
		thread_sync_.barrier();
#endif
		int tid = omp_get_thread_num();
		int num_threads = omp_get_num_threads();

#if 1 // dynamic partitioning
		int visited_count = 0;
		int block_width = step_bitmap_width / 40 + 1;
		while(true) {
			int off_start = __sync_fetch_and_add(process_counter, block_width);
			if(off_start >= step_bitmap_width) break; // finish
			int off_end = std::min(step_bitmap_width, off_start + block_width);

			bottom_up_search_bitmap_process_block(phase_bitmap,
					off_start, off_end, phase_bmp_off, buffer);
			PROF(extract_edge_time_ += tk_all);

			if(tid == 0) {
				// process async communication
				bottom_up_substep_->probe();
			}

			visited_count += buffer->length;
			flush_bottom_up_send_buffer(buffer, target_rank);

			PROF(commit_time_ += tk_all);
		}
#else // static partitioning
		int width_per_thread = (step_bitmap_width + num_threads - 1) / num_threads;
		int off_start = std::min(step_bitmap_width, width_per_thread * tid);
		int off_end = std::min(step_bitmap_width, off_start + width_per_thread);

		bottom_up_search_bitmap_process_block(phase_bitmap,
				off_start, off_end, phase_bmp_off, buffer);
		PROF(extract_edge_time_ += tk_all);

		int visited_count = buffer->length / 2;
		flush_bottom_up_send_buffer(buffer, target_rank);

		PROF(commit_time_ += tk_all);
#endif
		USER_END(bu_bmp_step);

		thread_sync_.barrier();

		PROF(extract_thread_wait_ += tk_all);
		return visited_count;
	}

	// returns the number of vertices found in this step.
	TwodVertex bottom_up_search_list_process_step(
#if VERVOSE_MODE
			int64_t& num_blocks,
#endif
			BottomUpSubstepData& data,
			int target_rank,
			int8_t* vertex_enabled,
			TwodVertex* write_list,
			TwodVertex step_bitmap_width,
			int* th_offset)
	{
		TRACER(bu_list_step);
		int max_threads = omp_get_num_threads();
		TwodVertex* phase_list = (TwodVertex*)data.data;
		TwodVertex phase_bmp_off = data.tag.region_id * step_bitmap_width;

		int lgl = graph_.local_bits_;
		int r_bits = graph_.r_bits_;
		TwodVertex L = graph_.num_local_verts_;
		int orig_lgl = graph_.orig_local_bits_;
		//TwodVertex phase_vertex_off = L / BU_SUBSTEP * (data.tag.region_id % BU_SUBSTEP);
		VERVOSE(int tmp_num_blocks = 0);
		VERVOSE(int tmp_edge_relax = 0);
		PROF(profiling::TimeKeeper tk_all);
		PROF(profiling::TimeSpan ts_commit);
		int tid = omp_get_thread_num();
		ThreadLocalBuffer* tlb = thread_local_buffer_[tid];
		LocalPacket* buffer = tlb->fold_packet;
		assert (buffer->length == 0);
		int num_send = 0;

		int64_t begin, end;
		get_partition(data.tag.length, phase_list, LOG_BFELL_SORT, max_threads, tid, begin, end);
		int num_enabled = end - begin;
		USER_START(bu_list_proc);
		for(int i = begin; i < end; ) {
			TwodVertex blk_idx = phase_list[i] >> LOG_BFELL_SORT;
			TwodVertex* phase_row_sums = graph_.row_sums_ + phase_bmp_off;
			BitmapType* phase_row_bitmap = graph_.row_bitmap_ + phase_bmp_off;
			VERVOSE(tmp_num_blocks++);

			do {
				vertex_enabled[i] = 1;
				TwodVertex tgt = phase_list[i];
				TwodVertex word_idx = tgt >> LOG_NBPE;
				int bit_idx = tgt & NBPE_MASK;
				BitmapType vis_bit = (BitmapType(1) << bit_idx);
				BitmapType row_bitmap_i = phase_row_bitmap[word_idx];
				if(row_bitmap_i & vis_bit) { // I have edges for this vertex ?
					TwodVertex non_zero_idx = phase_row_sums[word_idx] +
							__builtin_popcountl(row_bitmap_i & (vis_bit-1));
					LocalVertex tgt_orig = graph_.orig_vertexes_[non_zero_idx];
#if ISOLATE_FIRST_EDGE
					int64_t src = graph_.isolated_edges_[non_zero_idx];
					TwodVertex bit_idx = SeparatedId(SeparatedId(src).low(r_bits + lgl)).compact(lgl, L);
					if(shared_visited_[bit_idx >> LOG_NBPE] & (BitmapType(1) << (bit_idx & NBPE_MASK))) {
						// add to next queue
						vertex_enabled[i] = 0; --num_enabled;
						buffer->data.b[num_send++] = ((src >> lgl) << orig_lgl) | tgt_orig;
						// end this row
						VERVOSE(tmp_edge_relax += 1);
						continue;
					}
#endif // #if ISOLATE_FIRST_EDGE
					int64_t e_start = graph_.row_starts_[non_zero_idx];
					int64_t e_end = graph_.row_starts_[non_zero_idx+1];
					for(int64_t e = e_start; e < e_end; ++e) {
						int64_t src = graph_.edge_array_[e];
						TwodVertex bit_idx = SeparatedId(SeparatedId(src).low(r_bits + lgl)).compact(lgl, L);
						if(shared_visited_[bit_idx >> LOG_NBPE] & (BitmapType(1) << (bit_idx & NBPE_MASK))) {
							// add to next queue
							vertex_enabled[i] = 0; --num_enabled;
							buffer->data.b[num_send++] = ((src >> lgl) << orig_lgl) | tgt_orig;
							// end this row
							VERVOSE(tmp_edge_relax += e - e_start + 1);
							break;
						}
					}
				} // if(row_bitmap_i & vis_bit) {
			} while((phase_list[++i] >> LOG_BFELL_SORT) == blk_idx);
			assert(i <= end);
		}
		buffer->length = num_send;
		th_offset[tid+1] = num_enabled;
		PROF(extract_edge_time_ += tk_all);
		USER_END(bu_list_proc);
		thread_sync_.barrier();
		PROF(extract_thread_wait_ += tk_all);
#pragma omp master
		{
			TRACER(bu_list_single);
			th_offset[0] = 0;
			for(int i = 0; i < max_threads; ++i) {
				th_offset[i+1] += th_offset[i];
			}
			assert (th_offset[max_threads] <= int(data.tag.length));
		}
		thread_sync_.barrier();

		USER_START(bu_list_write);
		// make new list to send
		int offset = th_offset[tid];

		for(int i = begin; i < end; ++i) {
			if(vertex_enabled[i]) {
				write_list[offset++] = phase_list[i];
			}
		}

		USER_END(bu_list_write);
		flush_bottom_up_send_buffer(buffer, target_rank);
		PROF(commit_time_ += tk_all);
		VERVOSE(__sync_fetch_and_add(&num_blocks, tmp_num_blocks));
		VERVOSE(__sync_fetch_and_add(&num_edge_bottom_up_, tmp_edge_relax));
		thread_sync_.barrier();
		return data.tag.length - th_offset[max_threads];
	}
#endif // #if BFELL

	void bottom_up_gather_nq_size(int* visited_count) {
		TRACER(bu_gather_info);
		PROF(profiling::TimeKeeper tk_all);
		PROF(MPI_Barrier(mpi.comm_2d));
		PROF(fold_competion_wait_ += tk_all);
#if 1 // which one is faster ?
		int recv_count[mpi.size_2dc]; for(int i = 0; i < mpi.size_2dc; ++i) recv_count[i] = 1;
		MPI_Reduce_scatter(visited_count, &nq_size_, recv_count, MPI_INT, MPI_SUM, mpi.comm_2dr);
		MPI_Allreduce(&nq_size_, &max_nq_size_, 1, MPI_INT, MPI_MAX, mpi.comm_2d);
		int64_t nq_size = nq_size_;
		MPI_Allreduce(&nq_size, &global_nq_size_, 1, MpiTypeOf<int64_t>::type, MPI_SUM, mpi.comm_2d);
#else
		int red_nq_size[mpi.size_2dc];
		struct {
			int nq_size;
			int max_nq_size;
			int64_t global_nq_size;
		} scatter_buffer[mpi.size_2dc], recv_nq_size;
		// gather information within the processor row
		MPI_Reduce(visited_count, red_nq_size, mpi.size_2dc, MPI_INT, MPI_SUM, 0, mpi.comm_2dr);
		if(mpi.rank_2dr == 0) {
			int max_nq_size = 0, sum_nq_size = 0;
			int64_t global_nq_size;
			for(int i = 0; i < mpi.size_2dc; ++i) {
				sum_nq_size += red_nq_size[i];
				if(max_nq_size < red_nq_size[i]) max_nq_size = red_nq_size[i];
			}
			// compute global_nq_size by allreduce within the processor column
			MPI_Allreduce(&sum_nq_size, &global_nq_size, 1, MpiTypeOf<int64_t>::type, MPI_SUM, mpi.comm_2dc);
			for(int i = 0; i < mpi.size_2dc; ++i) {
				scatter_buffer[i].nq_size = red_nq_size[i];
				scatter_buffer[i].max_nq_size = max_nq_size;
				scatter_buffer[i].global_nq_size = global_nq_size;
			}
		}
		// scatter information within the processor row
		MPI_Scatter(scatter_buffer, sizeof(recv_nq_size), MPI_BYTE,
				&recv_nq_size, sizeof(recv_nq_size), MPI_BYTE, 0, mpi.comm_2dr);
		nq_size_ = recv_nq_size.nq_size;
		max_nq_size_ = recv_nq_size.max_nq_size;
		global_nq_size_ = recv_nq_size.global_nq_size;
#endif
		PROF(gather_nq_time_ += tk_all);
	}

	void bottom_up_bmp_parallel_section(int *visited_count) {
		PROF(profiling::TimeKeeper tk_all);
		int bitmap_width = get_bitmap_size_local();
		int step_bitmap_width = bitmap_width / BU_SUBSTEP;
		assert (work_buf_size_ >= bitmap_width * PRM::BOTTOM_UP_BUFFER);
		int buffer_count = work_buf_size_ / (step_bitmap_width * sizeof(BitmapType));
		BitmapType* bitmap_buffer[buffer_count];
		get_visited_pointers(bitmap_buffer, buffer_count, work_buf_, BU_SUBSTEP);
		BitmapType *new_vis[BU_SUBSTEP];
		get_visited_pointers(new_vis, BU_SUBSTEP, new_visited_, BU_SUBSTEP);
		int comm_size = mpi.size_2dc;
		volatile int process_counter = 0;

		// since the first 4 buffer contains initial data, skip this region
		bottom_up_substep_->begin(bitmap_buffer + BU_SUBSTEP, buffer_count - BU_SUBSTEP,
				step_bitmap_width);

		BottomUpSubstepData data;
		int total_steps = (comm_size+1)*BU_SUBSTEP;

#pragma omp parallel
		{
			SET_OMP_AFFINITY;
			int tid = omp_get_thread_num();
			for(int step = 0; step < total_steps; ++step) {
				if(tid == 0) {
					if(step < BU_SUBSTEP) {
						data.data = bitmap_buffer[step];
						data.tag.length = step_bitmap_width;
						data.tag.region_id = mpi.rank_2dc * BU_SUBSTEP + step;
						data.tag.routed_count = 0;
						// route is set by communication lib
					}
					else {
						// receive data
						PROF(profiling::TimeKeeper tk_all);
						bottom_up_substep_->recv(&data);
						PROF(comm_wait_time_ += tk_all);
					}
					process_counter = 0;
				}
				thread_sync_.barrier();

				int target_rank = data.tag.region_id / BU_SUBSTEP;
				if(step >= BU_SUBSTEP && target_rank == mpi.rank_2dc) {
					// This is rounded and came here.
					BitmapType* src = (BitmapType*)data.data;
					BitmapType* dst = new_vis[data.tag.region_id % BU_SUBSTEP];
#pragma omp for
					for(int64_t i = 0; i < step_bitmap_width; ++i) {
						dst[i] = src[i];
					}
					thread_sync_.barrier();
				}
				else {
					visited_count[data.tag.routed_count + tid * comm_size] +=
							bottom_up_search_bitmap_process_step(
									data, step_bitmap_width, &process_counter, target_rank);
					if(tid == 0) {
						if(step < BU_SUBSTEP) {
							bottom_up_substep_->send_first(&data);
						}
						else {
							bottom_up_substep_->send(&data);
						}
					}
				}
			}
			// wait for local_visited is received.
			if(tid == 0) {
				PROF(profiling::TimeKeeper tk_all);
				bottom_up_substep_->finish();
				PROF(comm_wait_time_ += tk_all);
			}
			thread_sync_.barrier();

		} // #pragma omp parallel
		PROF(parallel_reg_time_ += tk_all);
	}

	struct BottomUpBitmapParallelSection : public Runnable {
		ThisType* this_; int* visited_count;
		BottomUpBitmapParallelSection(ThisType* this__, int* visited_count_)
			: this_(this__), visited_count(visited_count_) { }
		virtual void run() { this_->bottom_up_bmp_parallel_section(visited_count); }
	};

	void bottom_up_search_bitmap() {
		TRACER(bu_bmp);
		int max_threads = omp_get_max_threads();
		int comm_size = mpi.size_2dc;
		int visited_count[comm_size*max_threads];
		for(int i = 0; i < comm_size*max_threads; ++i) visited_count[i] = 0;

		bu_comm_.prepare();
		bottom_up_bmp_parallel_section(visited_count);
		bu_comm_.run();

		// gather visited_count
		for(int tid = 1; tid < max_threads; ++tid)
			for(int i = 0; i < comm_size; ++i)
				visited_count[i + 0*comm_size] += visited_count[i + tid*comm_size];

		bottom_up_gather_nq_size(visited_count);
		VERVOSE(botto_up_print_stt(0, 0, visited_count));
		VERVOSE(bottom_up_substep_->print_stt());
	}

	void bottom_up_list_parallel_section(int *visited_count, int8_t* vertex_enabled,
			int64_t& num_blocks, int64_t& num_vertexes)
	{
		PROF(profiling::TimeKeeper tk_all);
		int bitmap_width = get_bitmap_size_local();
		int step_bitmap_width = bitmap_width / BU_SUBSTEP;
		assert (work_buf_size_ >= bitmap_width * PRM::BOTTOM_UP_BUFFER);
		int buffer_size = step_bitmap_width * sizeof(BitmapType) / sizeof(TwodVertex);
		int buffer_count = work_buf_size_ / (step_bitmap_width * sizeof(BitmapType));
		TwodVertex* list_buffer[buffer_count];
		get_visited_pointers(list_buffer, buffer_count, work_buf_, BU_SUBSTEP);
		TwodVertex *new_vis[BU_SUBSTEP];
		get_visited_pointers(new_vis, BU_SUBSTEP, new_visited_, BU_SUBSTEP);
		TwodVertex *old_vis[BU_SUBSTEP];
		get_visited_pointers(old_vis, BU_SUBSTEP, old_visited_, BU_SUBSTEP);
		int comm_size = mpi.size_2dc;

		// the first 5 buffers are working buffer
		bottom_up_substep_->begin(list_buffer + BU_SUBSTEP + 1, buffer_count - BU_SUBSTEP - 1,
				buffer_size);

		TwodVertex* back_buffer = list_buffer[BU_SUBSTEP];
		TwodVertex* write_buffer;
		BottomUpSubstepData data;
		TwodVertex sentinel_value = step_bitmap_width * NBPE;
		int total_steps = (comm_size+1)*BU_SUBSTEP;
		int max_threads = omp_get_max_threads();
		int th_offset[max_threads];

#pragma omp parallel
		{
			SET_OMP_AFFINITY;
			for(int step = 0; step < total_steps; ++step) {
#pragma omp master
				{
					if(step < BU_SUBSTEP) {
						data.data = old_vis[step];
						data.tag.length = old_visited_list_size_[step];
						write_buffer = list_buffer[step];
						data.tag.region_id = mpi.rank_2dc * BU_SUBSTEP + step;
						data.tag.routed_count = 0;
						// route is set by communication lib
					}
					else {
						// receive data
						PROF(profiling::TimeKeeper tk_all);
						bottom_up_substep_->recv(&data);
						PROF(comm_wait_time_ += tk_all);
						write_buffer = back_buffer;
					}
				}
				thread_sync_.barrier(); // TODO: compare with omp barrier

				int target_rank = data.tag.region_id / BU_SUBSTEP;
				TwodVertex* phase_list = (TwodVertex*)data.data;
				if(step >= BU_SUBSTEP && target_rank == mpi.rank_2dc) {
					// This is rounded and came here.
					int local_region_id = data.tag.region_id % BU_SUBSTEP;
					TwodVertex* dst = new_vis[local_region_id];
#pragma omp for
					for(int64_t i = 0; i < data.tag.length; ++i) {
						dst[i] = phase_list[i];
					}
#pragma omp master
					{
						new_visited_list_size_[local_region_id] = data.tag.length;
					}
					thread_sync_.barrier();
				}
				else {
					// write sentinel value to the last
					assert (data.tag.length < buffer_size - 1);
					phase_list[data.tag.length] = sentinel_value;
					int new_visited_cnt = bottom_up_search_list_process_step(
#if VERVOSE_MODE
						num_blocks,
#endif
						data, target_rank, vertex_enabled, write_buffer, step_bitmap_width, th_offset);

#pragma omp master
					{
						VERVOSE(num_vertexes += data.tag.length);
						visited_count[data.tag.routed_count] += new_visited_cnt;
						data.tag.length -= new_visited_cnt;

						if(step < BU_SUBSTEP) {
							data.data = write_buffer;
							bottom_up_substep_->send_first(&data);
						}
						else {
							back_buffer = (TwodVertex*)data.data;
							data.data = write_buffer;
							bottom_up_substep_->send(&data);
						}
					}
				}
			}
			// wait for local_visited is received.
#pragma omp master
			{
				PROF(profiling::TimeKeeper tk_all);
				bottom_up_substep_->finish();
				PROF(comm_wait_time_ += tk_all);
			}
			thread_sync_.barrier();

		} // #pragma omp parallel
		PROF(parallel_reg_time_ += tk_all);
	}
/*
	struct BottomUpListParallelSection : public Runnable {
		ThisType* this_; int* visited_count; int8_t* vertex_enabled;
		int64_t num_blocks; int64_t num_vertexes;
		BottomUpListParallelSection(ThisType* this__, int* visited_count_, int8_t* vertex_enabled_)
			: this_(this__), visited_count(visited_count_) , vertex_enabled(vertex_enabled_)
			, num_blocks(0), num_vertexes(0) { }
		virtual void run() {
			this_->b;
		}
	};
*/
	void bottom_up_search_list() {
		TRACER(bu_list);

		int half_bitmap_width = get_bitmap_size_local() / 2;
		int buffer_size = half_bitmap_width * sizeof(BitmapType) / sizeof(TwodVertex);
		// TODO: reduce memory allocation
		int8_t* vertex_enabled = (int8_t*)cache_aligned_xcalloc(buffer_size*sizeof(int8_t));

		int comm_size = mpi.size_2dc;
		int visited_count[comm_size];
		for(int i = 0; i < comm_size; ++i) visited_count[i] = 0;

		bu_comm_.prepare();
		int64_t num_blocks; int64_t num_vertexes;
		bottom_up_list_parallel_section(visited_count, vertex_enabled, num_blocks, num_vertexes);
		bu_comm_.run();

		bottom_up_gather_nq_size(visited_count);
		VERVOSE(botto_up_print_stt(num_blocks, num_vertexes, visited_count));
		VERVOSE(bottom_up_substep_->print_stt());

		free(vertex_enabled); vertex_enabled = NULL;
	}

	struct BottomUpReceiver : public Runnable {
		BottomUpReceiver(ThisType* this_, int64_t* buffer_, int length_, int src_)
			: this_(this_), buffer_(buffer_), length_(length_), src_(src_) { }
		virtual void run() {
			TRACER(bu_recv);
			PROF(profiling::TimeKeeper tk_all);
			int P = mpi.size_2d;
			int r_bits = this_->graph_.r_bits_;
			int64_t r_mask = ((1 << r_bits) - 1);
			int orig_lgl = this_->graph_.orig_local_bits_;
			LocalVertex lmask = (LocalVertex(1) << orig_lgl) - 1;
			int64_t cshifted = src_ * mpi.size_2dr;
			int64_t levelshifted = int64_t(this_->current_level_) << 48;
			int64_t* buffer = buffer_;
			int length = length_;
			int64_t* pred = this_->pred_;
			for(int i = 0; i < length; ++i) {
				int64_t v = buffer[i];
				int64_t pred_dst = v >> orig_lgl;
				LocalVertex tgt_local = v & lmask;
				int64_t pred_v = ((pred_dst >> r_bits) * P +
						cshifted + (pred_dst & r_mask)) | levelshifted;
				assert (this_->pred_[tgt_local] == -1);
				pred[tgt_local] = pred_v;
			}

			PROF(this_->recv_proc_thread_time_ += tk_all);
		}
		ThisType* const this_;
		int64_t* buffer_;
		int length_;
		int src_;
	};

	static void printInformation()
	{
		if(mpi.isMaster() == false) return ;
		using namespace PRM;
		print_with_prefix("===== Settings and Parameters. ====");

#define PRINT_VAL(fmt, val) print_with_prefix(#val " = " fmt ".", val)
		PRINT_VAL("%d", NUM_BFS_ROOTS);
		PRINT_VAL("%d", omp_get_max_threads());
		PRINT_VAL("%d", omp_get_nested());
		PRINT_VAL("%zd", sizeof(BitmapType));
		PRINT_VAL("%zd", sizeof(TwodVertex));

		PRINT_VAL("%d", NUMA_BIND);
		PRINT_VAL("%d", CPU_BIND_CHECK);
		PRINT_VAL("%d", PRINT_BINDING);
		PRINT_VAL("%d", SHARED_MEMORY);

		PRINT_VAL("%d", MPI_FUNNELED);
		PRINT_VAL("%d", OPENMP_SUB_THREAD);

		PRINT_VAL("%d", VERVOSE_MODE);
		PRINT_VAL("%d", PROFILING_MODE);
		PRINT_VAL("%d", DEBUG_PRINT);
		PRINT_VAL("%d", REPORT_GEN_RPGRESS);
		PRINT_VAL("%d", ENABLE_FJMPI_RDMA);
		PRINT_VAL("%d", ENABLE_FUJI_PROF);
		PRINT_VAL("%d", ENABLE_MY_ALLGATHER);
		PRINT_VAL("%d", ENABLE_INLINE_ATOMICS);

		PRINT_VAL("%d", BFELL);

		PRINT_VAL("%d", VERTEX_REORDERING);
		PRINT_VAL("%d", TOP_DOWN_SEND_LB);
		PRINT_VAL("%d", TOP_DOWN_RECV_LB);
		PRINT_VAL("%d", BOTTOM_UP_OVERLAP_PFS);

		PRINT_VAL("%d", ISOLATE_FIRST_EDGE);
		PRINT_VAL("%d", CONSOLIDATE_IFE_PROC);

		PRINT_VAL("%d", INIT_PRED_ONCE);

		PRINT_VAL("%d", PRE_EXEC_TIME);

		PRINT_VAL("%d", BACKTRACE_ON_SIGNAL);
#if BACKTRACE_ON_SIGNAL
		PRINT_VAL("%d", PRINT_BT_SIGNAL);
#endif
		PRINT_VAL("%d", PACKET_LENGTH);
		PRINT_VAL("%d", COMM_BUFFER_SIZE);
		PRINT_VAL("%d", SEND_BUFFER_LIMIT);
		PRINT_VAL("%d", BOTTOM_UP_BUFFER);
		PRINT_VAL("%d", NBPE);
		PRINT_VAL("%d", BFELL_SORT);
		PRINT_VAL("%f", DENOM_TOPDOWN_TO_BOTTOMUP);
		PRINT_VAL("%f", DEMON_BOTTOMUP_TO_TOPDOWN);
		PRINT_VAL("%f", DENOM_BITMAP_TO_LIST);

		PRINT_VAL("%d", VALIDATION_LEVEL);
		PRINT_VAL("%d", SGI_OMPLACE_BUG);
#undef PRINT_VAL

		if(NUM_BFS_ROOTS == 64 && VALIDATION_LEVEL == 2)
			print_with_prefix("===== Benchmark Mode OK ====");
		else
			print_with_prefix("===== Non Benchmark Mode ====");
	}
#if VERVOSE_MODE
	void printTime(const char* fmt, double* sum, double* max, int idx) {
		print_with_prefix(fmt, sum[idx] / mpi.size_2d * 1000.0,
				diff_percent(max[idx], sum[idx], mpi.size_2d));
	}
	void printCounter(const char* fmt, int64_t* sum, int64_t* max, int idx) {
		print_with_prefix(fmt, to_mega(sum[idx] / mpi.size_2d),
				diff_percent(max[idx], sum[idx], mpi.size_2d));
	}
#endif
	void prepare_sssp() { }
	void run_sssp(int64_t root, int64_t* pred) { }
	void end_sssp() { }

	// members
	MpiBottomUpSubstepComm* bottom_up_substep_;
	CommBufferPool a2a_comm_buf_;
	TopDownCommHandler top_down_comm_;
	BottomUpCommHandler bottom_up_comm_;
	AsyncAlltoallManager td_comm_;
	AsyncAlltoallManager bu_comm_;
	ThreadLocalBuffer** thread_local_buffer_;
	memory::ConcurrentPool<QueuedVertexes> nq_empty_buffer_;
	memory::ConcurrentStack<QueuedVertexes*> nq_;

	// switch parameters
	double denom_to_bottom_up_; // alpha
	double denom_bitmap_to_list_; // gamma

	// cq_list_ is a pointer to work_buf_ or work_extra_buf_
	TwodVertex* cq_list_;
	TwodVertex cq_size_;
	int nq_size_;
	int max_nq_size_;
	int64_t global_nq_size_;

	// size = local bitmap width
	// These two buffer is swapped at the beginning of every backward step
	void* new_visited_; // shared memory but point to the local portion
	void* old_visited_; // shared memory but point to the local portion
	void* visited_buffer_; // shared memory but point to the local portion
	void* visited_buffer_orig_; // shared memory
	// size = 2
	int new_visited_list_size_[BU_SUBSTEP];
	int old_visited_list_size_[BU_SUBSTEP];


	// 1. CQ at the top-down phase
	// 2. NQ receive buffer at the bottom-up expand phase
	// 3. Working memory at the bottom-up search phase
	void* work_buf_; // shared memory but point to the local portion
	void* work_extra_buf_; // for large CQ in the top down phase
	int64_t work_buf_size_; // in bytes
	int64_t work_extra_buf_size_;

	BitmapType* shared_visited_; // shared memory
	TwodVertex* nq_recv_buf_; // shared memory (memory space is shared with work_buf_)

	int64_t* pred_; // passed from main method

	struct SharedDataSet {
		memory::SpinBarrier *sync;
		int *offset; // max(max_threads*2+1, 2*mpi.size_z+1)
	} s_; // shared memory

	int current_level_;
	bool forward_or_backward_;
	bool bitmap_or_list_;
	bool growing_or_shrinking_;
	bool packet_buffer_is_dirty_;
	memory::SpinBarrier thread_sync_;

	VERVOSE(int64_t num_edge_top_down_);
	VERVOSE(int64_t num_td_large_edge_);
	VERVOSE(int64_t num_edge_bottom_up_);
	struct {
		void* thread_local_;
		void* shared_memory_;
	} buffer_;
	PROF(profiling::TimeSpan extract_edge_time_);
	PROF(profiling::TimeSpan isolated_edge_time_);
	PROF(profiling::TimeSpan extract_thread_wait_);
	PROF(profiling::TimeSpan parallel_reg_time_);
	PROF(profiling::TimeSpan seq_proc_time_);
	PROF(profiling::TimeSpan commit_time_);
	PROF(profiling::TimeSpan comm_wait_time_);
	PROF(profiling::TimeSpan fold_competion_wait_);
	PROF(profiling::TimeSpan recv_proc_thread_time_);
	PROF(profiling::TimeSpan recv_proc_thread_large_time_);
	PROF(profiling::TimeSpan gather_nq_time_);
};

void BfsBase::run_bfs(int64_t root, int64_t* pred)
{
	SET_AFFINITY;
#if ENABLE_FUJI_PROF
	fapp_start("initialize", 0, 0);
	start_collection("initialize");
#endif
	TRACER(run_bfs);
	pred_ = pred;
#if VERVOSE_MODE
	double tmp = MPI_Wtime();
	double start_time = tmp;
	double prev_time = tmp;
	double expand_time = 0.0, fold_time = 0.0;
	g_tp_comm = g_bu_pred_comm = g_bu_bitmap_comm = g_bu_list_comm = g_expand_bitmap_comm = g_expand_list_comm = 0;
	int64_t total_edge_top_down = 0;
	int64_t total_edge_bottom_up = 0;
#endif

	initialize_memory(pred);

#if VERVOSE_MODE
	if(mpi.isMaster()) print_with_prefix("Time of initialize memory: %f ms", (MPI_Wtime() - prev_time) * 1000.0);
	prev_time = MPI_Wtime();
#endif

	bool next_forward_or_backward = true; // begin with forward search
	bool next_bitmap_or_list = false;
	int64_t global_visited_vertices = 1; // count the root vertex

	// perform level 0
	current_level_ = 0;
	max_nq_size_ = 1;
	global_nq_size_ = 0;
	forward_or_backward_ = next_forward_or_backward;
	bitmap_or_list_ = next_bitmap_or_list;
	growing_or_shrinking_ = true;
	first_expand(root);

#if VERVOSE_MODE
	tmp = MPI_Wtime();
	double cur_expand_time = tmp - prev_time;
	expand_time += cur_expand_time; prev_time = tmp;
#endif

#if ENABLE_FUJI_PROF
	stop_collection("initialize");
	fapp_stop("initialize", 0, 0);
	char *prof_mes[] = { "bottom-up", "top-down" };
#endif

	while(true) {
		++current_level_;
#if VERVOSE_MODE
		num_edge_top_down_ = 0;
		num_td_large_edge_ = 0;
		num_edge_bottom_up_ = 0;
#endif // #if VERVOSE_MODE
#if ENABLE_FUJI_PROF
		fapp_start(prof_mes[(int)forward_or_backward_], 0, 0);
		start_collection(prof_mes[(int)forward_or_backward_]);
#endif
		TRACER(level);
		// search phase //
		int64_t prev_global_nq_size = global_nq_size_;
//		bool prev_forward_or_backward = forward_or_backward_;
		bool prev_bitmap_or_list = bitmap_or_list_;

		forward_or_backward_ = next_forward_or_backward;
		bitmap_or_list_ = next_bitmap_or_list;

		if(forward_or_backward_) { // forward
			top_down_search();
			// release extra buffer
			if(work_extra_buf_ != NULL) { free(work_extra_buf_); work_extra_buf_ = NULL; }
		}
		else { // backward
			swap_visited_memory(prev_bitmap_or_list);
			if(bitmap_or_list_) { // bitmap
				bottom_up_search_bitmap();
			}
			else { // list
				bottom_up_search_list();
			}
		}

		global_visited_vertices += global_nq_size_;

#if VERVOSE_MODE
		tmp = MPI_Wtime();
		double cur_fold_time = tmp - prev_time;
		fold_time += cur_fold_time; prev_time = tmp;
		AsyncAlltoallManager* a2a_comm = forward_or_backward_ ? &td_comm_ : &bu_comm_;
		total_edge_top_down += num_edge_top_down_;
		total_edge_bottom_up += num_edge_bottom_up_;
#if PROFILING_MODE
		if(forward_or_backward_) {
			extract_edge_time_.submit("forward edge", current_level_);
		}
		else {
#if ISOLATE_FIRST_EDGE
			isolated_edge_time_.submit("isolated edge", current_level_);
#endif
			extract_edge_time_.submit("backward edge", current_level_);
		}
		extract_thread_wait_.submit("extract thread wait", current_level_);

		parallel_reg_time_.submit("parallel region", current_level_);
		commit_time_.submit("extract commit", current_level_);
		if(!forward_or_backward_) { // backward
			comm_wait_time_.submit("bottom-up communication wait", current_level_);
		}
		fold_competion_wait_.submit("fold completion wait", current_level_);
		recv_proc_thread_time_.submit("recv proc thread", current_level_);
		if(forward_or_backward_) {
			recv_proc_thread_large_time_.submit("recv proc thread large", current_level_);
		}
		gather_nq_time_.submit("gather NQ info", current_level_);
		seq_proc_time_.submit("sequential processing", current_level_);
		a2a_comm->submit_prof_info(current_level_, forward_or_backward_);

		if(forward_or_backward_) {
			profiling::g_pis.submitCounter(num_edge_top_down_, "top-down edge relax", current_level_);
			profiling::g_pis.submitCounter(num_td_large_edge_, "top-down large edge", current_level_);
		}
		else
			profiling::g_pis.submitCounter(num_edge_bottom_up_, "bottom-up edge relax", current_level_);

		int64_t send_num_edges[] = { num_edge_top_down_, num_td_large_edge_, num_edge_bottom_up_ };
		int64_t recv_num_edges[3];
		MPI_Reduce(send_num_edges, recv_num_edges, 3, MpiTypeOf<int64_t>::type, MPI_SUM, 0, MPI_COMM_WORLD);
		num_edge_top_down_ = recv_num_edges[0];
		num_td_large_edge_ = recv_num_edges[1];
		num_edge_bottom_up_ = recv_num_edges[2];
#endif // #if PROFILING_MODE
#endif // #if VERVOSE_MODE
#if ENABLE_FUJI_PROF
		stop_collection(prof_mes[(int)forward_or_backward_]);
		fapp_stop(prof_mes[(int)forward_or_backward_], 0, 0);
#endif
#if !VERVOSE_MODE
		if(global_nq_size_ == 0)
			break;
#endif
#if ENABLE_FUJI_PROF
		fapp_start("expand", 0, 0);
		start_collection("expand");
#endif
		int64_t global_unvisited_vertices = graph_.num_global_verts_ - global_visited_vertices;
		next_bitmap_or_list = !forward_or_backward_;
		if(growing_or_shrinking_ && global_nq_size_ > prev_global_nq_size) { // growing
			if(forward_or_backward_ // forward ?
				&& global_nq_size_ > graph_.num_global_verts_ / denom_to_bottom_up_ // NQ is large ?
				) { // switch to backward
				next_forward_or_backward = false;
				packet_buffer_is_dirty_ = true;
			}
		}
		else { // shrinking
			if(!forward_or_backward_  // backward ?
				&& global_unvisited_vertices < int64_t(graph_.num_global_verts_ / DEMON_BOTTOMUP_TO_TOPDOWN) // NQ is small ?
				) { // switch to topdown
				next_forward_or_backward = true;
				growing_or_shrinking_ = false;

				// Enabled if we compress lists with VLQ
				//int max_capacity = vlq::BitmapEncoder::calc_capacity_of_values(
				//		bitmap_width, NBPE, bitmap_width*sizeof(BitmapType));
				//int threashold = std::min<int>(max_capacity, bitmap_width*sizeof(BitmapType)/2);
				int bitmap_width = get_bitmap_size_local();
				double threashold = bitmap_width*sizeof(BitmapType)/sizeof(TwodVertex)/denom_bitmap_to_list_;
				next_bitmap_or_list = (max_nq_size_ >= threashold);
			}
		}
		if(next_forward_or_backward == false) {
			// bottom-up only with bitmap
			// do not use top_down_switch_expand with list since it is very slow!!
			next_bitmap_or_list = true;
		}

#if VERVOSE_MODE
		int64_t send_num_bufs[2] = { a2a_comm->get_last_send_size(), nq_empty_buffer_.size() };
		int64_t sum_num_bufs[2], max_num_bufs[2];
		MPI_Reduce(send_num_bufs, sum_num_bufs, 2, MpiTypeOf<int64_t>::type, MPI_SUM, 0, MPI_COMM_WORLD);
		MPI_Reduce(send_num_bufs, max_num_bufs, 2, MpiTypeOf<int64_t>::type, MPI_MAX, 0, MPI_COMM_WORLD);
		if(mpi.isMaster()) {
			double nq_rate = (double)global_nq_size_ / graph_.num_global_verts_;
			double nq_unvis_rate = (double)global_nq_size_ / global_unvisited_vertices;
			double unvis_rate = (double)global_unvisited_vertices / graph_.num_global_verts_;
			double time_of_level = cur_expand_time + cur_fold_time;
			print_with_prefix("=== Level %d complete ===", current_level_);
			print_with_prefix("Direction %s", forward_or_backward_ ? "top-down" : "bottom-up");

			print_with_prefix("Expand Time: %f ms", cur_expand_time * 1000.0);
			print_with_prefix("Fold Time: %f ms", cur_fold_time * 1000.0);
			print_with_prefix("Level Total Time: %f ms", time_of_level * 1000.0);

			print_with_prefix("NQ %"PRId64", 1/ %f, %f %% of global, 1/ %f, %f %% of Unvisited",
						global_nq_size_, 1/nq_rate, nq_rate*100, 1/nq_unvis_rate, nq_unvis_rate*100);
			print_with_prefix("Unvisited %"PRId64", 1/ %f, %f %% of global",
					global_unvisited_vertices, 1/unvis_rate, unvis_rate*100);

			int64_t edge_relaxed = forward_or_backward_ ? num_edge_top_down_ : num_edge_bottom_up_;
			print_with_prefix("Edge relax: %"PRId64", %f %%, %f M/s (Level), %f M/s (Fold)",
					edge_relaxed, (double)edge_relaxed / graph_.num_global_edges_ * 100.0,
					to_mega(edge_relaxed) / time_of_level,
					to_mega(edge_relaxed) / cur_fold_time);

			int64_t total_qb_size = int64_t(sum_num_bufs[1]) * BUCKET_UNIT_SIZE;
			int64_t max_qb_size = int64_t(max_num_bufs[1]) * BUCKET_UNIT_SIZE;
			print_with_prefix("A2A Send: %f MB per node, Max %f %%+",
					to_mega(sum_num_bufs[0] / mpi.size_2d),
					diff_percent((int64_t)max_num_bufs[0], sum_num_bufs[0], mpi.size_2d));
			print_with_prefix("Queue buffer: %f MB per node, Max %f %%+",
					to_mega(total_qb_size / mpi.size_2d), diff_percent(max_qb_size, total_qb_size, mpi.size_2d));

			if(next_forward_or_backward != forward_or_backward_) {
				if(forward_or_backward_)
					print_with_prefix("Direction Change: top-down -> bottom-up");
				else
					print_with_prefix("Direction Change: bottom-up -> top-down");
			}
			if(next_bitmap_or_list != bitmap_or_list_) {
				if(bitmap_or_list_)
					print_with_prefix("Format Change: bitmap -> list");
				else
					print_with_prefix("Format Change: list -> bitmap");
			}
			print_with_prefix("Next expand format: %s",
					next_bitmap_or_list ? "Bitmap" : "List");

			print_with_prefix("=== === === === ===");
		}
		if(global_nq_size_ == 0) {
			break;
		}
#endif
		// expand //
		if(next_forward_or_backward == forward_or_backward_) {
			if(forward_or_backward_)
				top_down_expand();
			else
				bottom_up_expand(next_bitmap_or_list);
		} else {
			if(forward_or_backward_)
				top_down_switch_expand(next_bitmap_or_list);
			else
				bottom_up_switch_expand(next_bitmap_or_list);
		}
		clear_nq_stack(); // currently, this is required only in the top-down phase

#if ENABLE_FUJI_PROF
		stop_collection("expand");
		fapp_stop("expand", 0, 0);
#endif
#if VERVOSE_MODE
		tmp = MPI_Wtime();
		cur_expand_time = tmp - prev_time;
		expand_time += cur_expand_time; prev_time = tmp;
#endif
	} // while(true) {
	clear_nq_stack();
#if VERVOSE_MODE
	if(mpi.isMaster()) print_with_prefix("Time of BFS: %f ms", (MPI_Wtime() - start_time) * 1000.0);
	int64_t total_edge_relax = total_edge_top_down + total_edge_bottom_up;
	int time_cnt = 2, cnt_cnt = 9;
	double send_time[] = { fold_time, expand_time }, sum_time[time_cnt], max_time[time_cnt];
	int64_t send_cnt[] = { g_tp_comm, g_bu_pred_comm, g_bu_bitmap_comm,
			g_bu_list_comm, g_expand_bitmap_comm, g_expand_list_comm,
			total_edge_top_down, total_edge_bottom_up, total_edge_relax };
	int64_t sum_cnt[cnt_cnt], max_cnt[cnt_cnt];
	MPI_Reduce(send_time, sum_time, time_cnt, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(send_time, max_time, time_cnt, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(send_cnt, sum_cnt, cnt_cnt, MpiTypeOf<int64_t>::type, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(send_cnt, max_cnt, cnt_cnt, MpiTypeOf<int64_t>::type, MPI_MAX, 0, MPI_COMM_WORLD);
	if(mpi.isMaster()) {
		printTime("Avg time of fold: %f ms, %f %%+", sum_time, max_time, 0);
		printTime("Avg time of expand: %f ms, %f %%+", sum_time, max_time, 1);
		printCounter("Avg top-down fold recv: %f MiB, %f %%+", sum_cnt, max_cnt, 0);
		printCounter("Avg bottom-up pred update recv: %f MiB, %f %%+", sum_cnt, max_cnt, 1);
		printCounter("Avg bottom-up bitmap send: %f MiB, %f %%+", sum_cnt, max_cnt, 2);
		printCounter("Avg bottom-up list send: %f MiB, %f %%+", sum_cnt, max_cnt, 3);
		printCounter("Avg expand bitmap recv: %f MiB, %f %%+", sum_cnt, max_cnt, 4);
		printCounter("Avg expand list recv: %f MiB, %f %%+", sum_cnt, max_cnt, 5);
		printCounter("Avg top-down traversed edges: %f MiB, %f %%+", sum_cnt, max_cnt, 6);
		printCounter("Avg bottom-up traversed edges: %f MiB, %f %%+", sum_cnt, max_cnt, 7);
		printCounter("Avg total relaxed traversed: %f MiB, %f %%+", sum_cnt, max_cnt, 8);
	}
#endif
}
#undef debug

#endif /* BFS_HPP_ */
