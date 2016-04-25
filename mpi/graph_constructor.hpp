/*
 * graph_constructor.hpp
 *
 *  Created on: Dec 14, 2011
 *      Author: koji
 */

#ifndef GRAPH_CONSTRUCTOR_HPP_
#define GRAPH_CONSTRUCTOR_HPP_

#include "parameters.h"
#include "limits.h"

//-------------------------------------------------------------//
// 2D partitioning
//-------------------------------------------------------------//

int inline vertex_owner_r(int64_t v) { return v % mpi.size_2dr; }
int inline vertex_owner_c(int64_t v) { return (v / mpi.size_2dr) % mpi.size_2dc; }
int inline edge_owner(int64_t v0, int64_t v1) { return vertex_owner_r(v0) + vertex_owner_c(v1) * mpi.size_2dr; }
int inline vertex_owner(int64_t v) { return v % mpi.size_2d; }
int64_t inline vertex_local(int64_t v) { return v / mpi.size_2d; }

class Graph2DCSR
{
	enum {
		LOG_NBPE = PRM::LOG_NBPE,
		NBPE_MASK = PRM::NBPE_MASK
	};
public:
	Graph2DCSR()
	: row_bitmap_(NULL)
	, row_sums_(NULL)
	, has_edge_bitmap_(NULL)
	, reorder_map_(NULL)
	, orig_vertexes_(NULL)
	, edge_array_(NULL)
	, row_starts_(NULL)
	, isolated_edges_(NULL)
	, log_orig_global_verts_(0)
	, log_max_weight_(0)
	, max_weight_(0)
	, num_global_edges_(0)
	, num_global_verts_(0)
	{ }
	~Graph2DCSR()
	{
		clean();
	}

	void clean()
	{
		free(row_bitmap_); row_bitmap_ = NULL;
		free(row_sums_); row_sums_ = NULL;
		free(reorder_map_); reorder_map_ = NULL;
		free(invert_map_); invert_map_ = NULL;
		MPI_Free_mem(orig_vertexes_); orig_vertexes_ = NULL;
		free(has_edge_bitmap_); has_edge_bitmap_ = NULL;
		free(edge_array_); edge_array_ = NULL;
		free(row_starts_); row_starts_ = NULL;
		free(isolated_edges_); isolated_edges_ = NULL;
	}

	int pred_size() { return num_orig_local_verts_; }

	int log_orig_global_verts() const { return log_orig_global_verts_; }

	// Reference Functions
	static int rank(int r, int c) { return c * mpi.size_2dr + r; }
	int64_t swizzle_vertex(int64_t v) {
		return SeparatedId(vertex_owner(v), vertex_local(v), local_bits_).value;
	}
	int64_t unswizzle_vertex(int64_t v) {
		SeparatedId id(v);
		return id.high(local_bits_) + id.low(local_bits_) * num_local_verts_;
	}

	// vertex id converter
	SeparatedId VtoD(int64_t v) {
		return SeparatedId(vertex_owner_r(v), vertex_local(v), local_bits_);
	}
	SeparatedId VtoS(int64_t v) {
		return SeparatedId(vertex_owner_c(v), vertex_local(v), local_bits_);
	}
	int64_t DtoV(SeparatedId id, int c) {
		return id.low(local_bits_) * mpi.size_2d + rank(id.high(local_bits_), c);
	}
	int64_t StoV(SeparatedId id, int r) {
		return id.low(local_bits_) * mpi.size_2d + rank(r, id.high(local_bits_));
	}
	SeparatedId StoD(SeparatedId id, int r) {
		return SeparatedId(r, id.low(local_bits_), local_bits_);
	}
	SeparatedId DtoS(SeparatedId id, int c) {
		return SeparatedId(c, id.low(local_bits_), local_bits_);
	}
	int get_weight_from_edge(int64_t e) {
		return e & ((1 << log_max_weight_) - 1);
	}

	bool has_edge(int64_t v, bool has_weight = false) {
		if(vertex_owner(v) == mpi.rank_2d) {
			int64_t v_local = reorder_map_[v / mpi.size_2d];
			if(v_local > num_local_verts_) return false;
			int64_t word_idx = v_local >> LOG_NBPE;
			int bit_idx = v_local & NBPE_MASK;
			return has_edge_bitmap_[word_idx] & (BitmapType(1) << bit_idx);
		}
		return false;
	}
//private:

	// Array Indices:
	//  - Compressed Source Index (CSI) : source index skipping vertices with no edges in this rank
	//  - Source Bitmap Index (SBI) : source index / 64
	//   - Pred: same as Pred (original local vertices)

	// num_local_verts_ <= num_orig_local_verts_ <= length(reorder_map_)

	BitmapType* row_bitmap_; // Index: SBI
	TwodVertex* row_sums_; // Index: SBI
	BitmapType* has_edge_bitmap_; // for every local vertices, Index: SBI
	LocalVertex* reorder_map_; // Index: Pred
	LocalVertex* invert_map_; // Index: Reordered Pred
	LocalVertex* orig_vertexes_; // Index: CSI

	int64_t* edge_array_;
	int64_t* row_starts_; // Index: CSI
	int64_t* isolated_edges_; // Index: CSI

	int log_orig_global_verts_; // estimated SCALE parameter
	int log_max_weight_;
	int64_t num_orig_local_verts_; // number of local vertices for original graph

	int max_weight_;
	int64_t num_global_edges_; // number of edges after reduction of hyper edges
	int64_t num_global_verts_; // number of vertices that have at least one edge

	int local_bits_; // local bits for computation
	int orig_local_bits_; // local bits for original vertex id
	int r_bits_;
	int64_t num_local_verts_; // number of local vertices for computation
};

namespace detail {

enum {
	LOG_EDGE_PART_SIZE = 16,
//	LOG_EDGE_PART_SIZE = 12,
	EDGE_PART_SIZE = 1 << LOG_EDGE_PART_SIZE, // == UINT64_MAX + 1
	EDGE_PART_SIZE_MASK = EDGE_PART_SIZE - 1,

	NBPE = PRM::NBPE,
	LOG_NBPE = PRM::LOG_NBPE,
	NBPE_MASK = PRM::NBPE_MASK,

	BFELL_SORT = PRM::BFELL_SORT,
	LOG_BFELL_SORT = PRM::LOG_BFELL_SORT,

	BFELL_SORT_IN_BMP = PRM::BFELL_SORT / NBPE,
};

struct GraphConstructionData {
	int64_t num_local_verts_;

	LocalVertex* reordre_map_;
	LocalVertex* invert_map_;

	int64_t* wide_row_starts_;
	int64_t* row_starts_sup_;

	BitmapType* row_bitmap_;
	TwodVertex* row_sums_;
	LocalVertex* orig_vertexes_; // free with MPI_Free_mem
};

struct DegreeCalculation {
	typedef Graph2DCSR GraphType;

	// Wide row edge for degree calculation
	struct DWideRowEdge {
		uint16_t src_vertex;
		int16_t c;

		DWideRowEdge(uint16_t src_vertex, int16_t c)
			: src_vertex(src_vertex)
			, c(c)
		{ }
	};

	enum {
		LOG_BLOCK_SIZE = LOG_EDGE_PART_SIZE - 5,
		BLOCK_SIZE = 1 << LOG_BLOCK_SIZE,
	};

	int org_local_bits_;
	int log_local_verts_unit_;

	int64_t* wide_row_length_;
	BitmapType* row_bitmap_;
	TwodVertex* row_sums_;
	LocalVertex* orig_vertexes_;
	int* num_vertexes_;

	int64_t max_local_verts_;

	int num_rows_;
	int64_t* row_length_;
	int64_t* row_offset_;
	std::vector<DWideRowEdge>* dwide_row_data_;
	LocalVertex* vertexes_; // passed to ConstructionData

	DegreeCalculation(int orig_local_bits, int log_local_verts_unit) {
		org_local_bits_ = orig_local_bits;
		log_local_verts_unit_ = log_local_verts_unit;
		num_rows_ = num_orig_local_verts() / BLOCK_SIZE;
		if(num_rows_ == 0) {
			fprintf(IMD_OUT, "BLOCK_SIZE is too large");
			throw "Error";
		}
		dwide_row_data_ = new std::vector<DWideRowEdge>[num_rows_]();
		row_length_ = static_cast<int64_t*>(cache_aligned_xcalloc(num_rows_*sizeof(int64_t)));
		row_offset_ = static_cast<int64_t*>(cache_aligned_xcalloc(num_rows_*sizeof(int64_t)));
	}

	~DegreeCalculation() {
		if(dwide_row_data_ != NULL) { delete [] dwide_row_data_; dwide_row_data_ = NULL; }
		if(wide_row_length_ != NULL) { free(wide_row_length_); wide_row_length_ = NULL; }
		if(row_bitmap_ != NULL) { free(row_bitmap_); row_bitmap_ = NULL; }
		if(row_sums_ != NULL) { free(row_sums_); row_sums_ = NULL; }
		if(orig_vertexes_ != NULL) { free(orig_vertexes_); orig_vertexes_ = NULL; }
		if(row_length_ != NULL) { free(row_length_); row_length_ = NULL; }
		if(row_offset_ != NULL) { free(row_offset_); row_offset_ = NULL; }
	}

	int64_t num_orig_local_verts() const {
		return int64_t(1) << org_local_bits_;
	}

	int64_t local_wide_row_size() const {
		return max_local_verts_ / EDGE_PART_SIZE;
	}

	int64_t local_bitmap_size() const {
		return max_local_verts_ / NBPE;
	}

	// edges: high: v1's c, low: v0's vertex_local
	void add(int64_t* edges, int64_t num_edges) {

		// count edges
#pragma omp parallel for
		for(int64_t i = 0; i < num_edges; ++i) {
			SeparatedId id(edges[i]);
			TwodVertex local = id.low(org_local_bits_);
			int row = local >> LOG_BLOCK_SIZE;

			__sync_fetch_and_add(&row_length_[row], 1);
		}

		// resize data store
		for(int i = 0; i < num_rows_; ++i) {
			dwide_row_data_[i].resize(row_length_[i], DWideRowEdge(0,0));
		}

		// store data
#pragma omp parallel for
		for(int64_t i = 0; i < num_edges; ++i) {
			SeparatedId id(edges[i]);
			int c = id.high(org_local_bits_);
			TwodVertex local = id.low(org_local_bits_);
			int row = local >> LOG_BLOCK_SIZE;
			int src_vertex = local % BLOCK_SIZE;

			int64_t offset = __sync_fetch_and_add(&row_offset_[row], 1);
			dwide_row_data_[row][offset] = DWideRowEdge(src_vertex, c);
		}
	}

	GraphConstructionData process() {
		LocalVertex* reorder_map = calc_degree();
		make_construct_data(reorder_map);
		return gather_data(reorder_map);
	}

private:

	template <typename T> struct ZeroOrElseComparator {
		bool operator()(const T& x, const T& y) {
			int xx = (x != 0);
			int yy = (y != 0);
			return xx > yy;
		}
	};

	LocalVertex* calc_degree() {
		if(mpi.isMaster()) print_with_prefix("Counting degree.");

		int64_t num_verts = num_orig_local_verts();
		int64_t* degree = static_cast<int64_t*>(cache_aligned_xcalloc(num_verts*sizeof(int64_t)));
		vertexes_ = static_cast<LocalVertex*>(cache_aligned_xcalloc(num_verts*sizeof(LocalVertex)));

#pragma omp parallel for
		for(int r = 0; r < num_rows_; ++r) {
			std::vector<DWideRowEdge>& row_data = dwide_row_data_[r];
			for(int64_t c = 0; c < int64_t(row_data.size()); ++c) {
				DWideRowEdge& edge = row_data[c];
				TwodVertex local = r * BLOCK_SIZE + edge.src_vertex;
				__sync_fetch_and_add(&degree[local], 1);
			}
		}

		for(LocalVertex i = 0; i < num_verts; ++i) {
			vertexes_[i] = i;
		}

		// sort by degree
#if VERTEX_REORDERING == 2
		sort2(degree, vertexes_, num_verts, std::greater<int64_t>());
#elif VERTEX_REORDERING == 1
		sort2(degree, vertexes_, num_verts, ZeroOrElseComparator<int64_t>());
#endif

		max_local_verts_ = 0;
		for(int64_t i = num_verts-1; i >= 0; --i) {
			if(degree[i] != 0) {
				max_local_verts_ = i+1;
				break;
			}
		}
		// roundup
		int64_t local_verts_unit = int64_t(1) << log_local_verts_unit_;
		max_local_verts_ = roundup(max_local_verts_, local_verts_unit);
		// get global max
		MPI_Allreduce(
				MPI_IN_PLACE, &max_local_verts_, 1,
				MpiTypeOf<int64_t>::type, MPI_MAX, mpi.comm_2d);
		if(mpi.isMaster()) print_with_prefix("Max local vertex %f M / %f M = %f %%",
				to_mega(max_local_verts_), to_mega(num_verts), (double)max_local_verts_ / num_verts * 100.0);

		free(degree); degree = NULL;

		// store mapping to degree
		LocalVertex* reorde_map = static_cast<LocalVertex*>(
				cache_aligned_xcalloc(num_verts*sizeof(LocalVertex)));
#pragma omp parallel for
		for(int64_t i = 0; i < num_verts; ++i) {
			reorde_map[vertexes_[i]] = int(i);
		}

		return reorde_map;
	}

	void make_construct_data(LocalVertex* reorder_map) {
		int64_t src_bitmap_size = local_bitmap_size() * mpi.size_2dc;
		int64_t num_wide_rows = local_wide_row_size() * mpi.size_2dc;

		// allocate 1
		row_bitmap_ = static_cast<BitmapType*>(
				cache_aligned_xcalloc(src_bitmap_size*sizeof(BitmapType)));

		ParallelPartitioning<int64_t> row_length_counter(num_wide_rows);

#pragma omp parallel
		{
			int64_t* counts = row_length_counter.get_counts();

#pragma omp for
			for(int r = 0; r < num_rows_; ++r) {
				std::vector<DWideRowEdge>& row_data = dwide_row_data_[r];
				for(int64_t c = 0; c < int64_t(row_data.size()); ++c) {
					DWideRowEdge& edge = row_data[c];
					int c = edge.c;
					TwodVertex local = r * BLOCK_SIZE + edge.src_vertex;
					LocalVertex reordred = reorder_map[local];

					int64_t wide_row_offset = local_wide_row_size() * c + (reordred >> LOG_EDGE_PART_SIZE);
					counts[wide_row_offset]++;

					BitmapType& bitmap_v = row_bitmap_[local_bitmap_size() * c + reordred / NBPE];
					BitmapType add_mask = BitmapType(1) << (reordred % NBPE);
					if((bitmap_v & add_mask) == 0) {
						__sync_fetch_and_or(&bitmap_v, add_mask);
					}
				}
			}
		}

		// free memory
		delete [] dwide_row_data_; dwide_row_data_ = NULL;

		// allocate 2
		wide_row_length_ = static_cast<int64_t*>(
				cache_aligned_xcalloc(num_wide_rows*sizeof(int64_t)));
		row_sums_ = static_cast<TwodVertex*>(
				cache_aligned_xcalloc((src_bitmap_size + 1)*sizeof(TwodVertex)));
		num_vertexes_ = static_cast<int*>(
				cache_aligned_xcalloc(mpi.size_2dc*sizeof(int)));

		// compute sum
		row_length_counter.sum();
		memcpy(wide_row_length_, row_length_counter.get_partition_size(),
				num_wide_rows*sizeof(int64_t));

		row_sums_[0] = 0;
		for(int64_t i = 0; i < src_bitmap_size; ++i) {
			int num_rows = __builtin_popcountl(row_bitmap_[i]);
			row_sums_[i+1] = row_sums_[i] + num_rows;
		}

		int64_t total_vertexes_ = 0;
		for(int i = 0; i < mpi.size_2dc; ++i) {
			num_vertexes_[i] =
					row_sums_[local_bitmap_size() * (i+1)] -
					row_sums_[local_bitmap_size() * i];
			total_vertexes_ += num_vertexes_[i];
		}

#if VERVOSE_MODE
		int64_t send_rowbmp[3] = { total_vertexes_, src_bitmap_size*NBPE, 0 };
		int64_t max_rowbmp[3];
		int64_t sum_rowbmp[3];
		MPI_Reduce(send_rowbmp, sum_rowbmp, 3, MpiTypeOf<int64_t>::type, MPI_SUM, 0, mpi.comm_2d);
		MPI_Reduce(send_rowbmp, max_rowbmp, 3, MpiTypeOf<int64_t>::type, MPI_MAX, 0, mpi.comm_2d);
		if(mpi.isMaster()) {
			print_with_prefix("DC total vertexes. Total %f M / %f M = %f %% Avg %f M / %f M Max %f %%+",
					to_mega(sum_rowbmp[0]), to_mega(sum_rowbmp[1]), to_mega(sum_rowbmp[0]) / to_mega(sum_rowbmp[1]) * 100,
					to_mega(sum_rowbmp[0]) / mpi.size_2d, to_mega(sum_rowbmp[1]) / mpi.size_2d,
					diff_percent(max_rowbmp[0], sum_rowbmp[0], mpi.size_2d));
		}
#endif // #if VERVOSE_MODE

		orig_vertexes_ = static_cast<LocalVertex*>(
				cache_aligned_xcalloc(total_vertexes_*sizeof(LocalVertex)));

#pragma omp parallel for
		for(int c = 0; c < mpi.size_2dc; ++c) {
			for(int64_t i = 0; i < local_bitmap_size(); ++i) {
				int64_t word_idx = i + local_bitmap_size() * c;
				for(int bit_idx = 0; bit_idx < NBPE; ++bit_idx) {
					BitmapType mask = BitmapType(1) << bit_idx;
					if(row_bitmap_[word_idx] & mask) {
						BitmapType word = row_bitmap_[word_idx] & (mask - 1);
						TwodVertex offset = __builtin_popcountl(word) + row_sums_[word_idx];
						TwodVertex reordred = i * NBPE + bit_idx;
						orig_vertexes_[offset] = vertexes_[reordred];
					}
				}
			}
		}
	}

	GraphConstructionData gather_data(LocalVertex* reorder_map) {
		GraphConstructionData data = {0};

		int64_t num_wide_rows_ = local_wide_row_size() * mpi.size_2dc;
		int64_t src_bitmap_size = local_bitmap_size() * mpi.size_2dc;

		data.num_local_verts_ = max_local_verts_;
		data.reordre_map_ = reorder_map;
		data.invert_map_ = vertexes_;

		// allocate memory
		data.wide_row_starts_ = static_cast<int64_t*>
			(cache_aligned_xmalloc((num_wide_rows_+1)*sizeof(int64_t)));
		data.row_starts_sup_ = static_cast<int64_t*>(
				cache_aligned_xmalloc((num_wide_rows_+1)*sizeof(int64_t)));
		data.row_bitmap_ = static_cast<BitmapType*>(
				cache_aligned_xmalloc(src_bitmap_size*sizeof(BitmapType)));
		data.row_sums_ = static_cast<TwodVertex*>(
				cache_aligned_xmalloc((src_bitmap_size+1)*sizeof(TwodVertex)));

		if(mpi.isMaster()) print_with_prefix("Transferring wide row length.");
		MPI_Alltoall(
				wide_row_length_,
				local_wide_row_size(),
				MpiTypeOf<int64_t>::type,
				data.wide_row_starts_ + 1,
				local_wide_row_size(),
				MpiTypeOf<int64_t>::type,
				mpi.comm_2dr);

		if(mpi.isMaster()) print_with_prefix("Computing edge offset.");
		data.wide_row_starts_[0] = 0;
		for(int64_t i = 1; i < num_wide_rows_; ++i) {
			data.wide_row_starts_[i+1] += data.wide_row_starts_[i];
		}

#ifndef NDEBUG
		if(mpi.isMaster()) print_with_prefix("Copying edge_counts for debugging.");
		memcpy(data.row_starts_sup_, data.wide_row_starts_,
				(num_wide_rows_+1)*sizeof(data.row_starts_sup_[0]));
#endif

		if(mpi.isMaster()) print_with_prefix("Transferring row bitmap.");
		MPI_Alltoall(
				row_bitmap_,
				local_bitmap_size(),
				MpiTypeOf<BitmapType>::type,
				data.row_bitmap_,
				local_bitmap_size(),
				MpiTypeOf<BitmapType>::type,
				mpi.comm_2dr);

		if(mpi.isMaster()) print_with_prefix("Re making row sums bitmap.");
		data.row_sums_[0] = 0;
		for(int64_t i = 0; i < src_bitmap_size; ++i) {
			// TODO: deal with different BitmapType
			int num_rows = __builtin_popcountl(data.row_bitmap_[i]);
			data.row_sums_[i+1] = data.row_sums_[i] + num_rows;
		}

		data.orig_vertexes_ = gather_orig_vertexes(data.row_sums_[src_bitmap_size]);

		return data;
	}

	LocalVertex* gather_orig_vertexes(TwodVertex num_non_zero_rows) {
		int sendoffset[mpi.size_2dc+1];
		int recvcount[mpi.size_2dc];
		int recvoffset[mpi.size_2dc+1];
		return MpiCol::alltoallv(orig_vertexes_, num_vertexes_, sendoffset,
				recvcount, recvoffset, mpi.comm_2dr, mpi.size_2dc);
	}
};

template <typename EdgeList>
class GraphConstructor2DCSR
{
public:
	typedef Graph2DCSR GraphType;
	typedef typename EdgeList::edge_type EdgeType;

	GraphConstructor2DCSR()
		: log_local_verts_unit_(0)
		, num_wide_rows_(0)
		, org_local_bits_(0)
		, local_bits_(0)
		, degree_calc_(NULL)
		, src_vertexes_(NULL)
		, wide_row_starts_(NULL)
		, row_starts_sup_(NULL)
	{ }
	~GraphConstructor2DCSR()
	{
		// since the heap checker of FUJITSU compiler reports error on free(NULL) ...
		if(degree_calc_ != NULL) { delete degree_calc_; degree_calc_ = NULL; }
		if(src_vertexes_ != NULL) { free(src_vertexes_); src_vertexes_ = NULL; }
		if(wide_row_starts_ != NULL) { free(wide_row_starts_); wide_row_starts_ = NULL; }
	}

	void construct(EdgeList* edge_list, int log_local_verts_unit, GraphType& g)
	{
		TRACER(construction);
		log_local_verts_unit_ = std::max<int>(log_local_verts_unit, LOG_EDGE_PART_SIZE);
		g.log_orig_global_verts_ = 0;

		searchMaxVertex(edge_list, g);
		scatterAndScanEdges(edge_list, g);
		makeWideRowStarts(g);
		scatterAndStore(edge_list, g);
		sortEdges(g);
		if(row_starts_sup_ != NULL) { free(row_starts_sup_); row_starts_sup_ = NULL; }

		if(mpi.isMaster()) print_with_prefix("Wide CSR creation complete.");

		constructFromWideCSR(g);

		computeNumVertices(g);

		if(mpi.isMaster()) print_with_prefix("Graph construction complete.");
	}

	void copy_to_gpu(GraphType& g, bool graph_on_gpu_) {
#if CUDA_ENABLED
		// transfer data to GPU
		const int64_t num_columns = (int64_t(1) << g.log_edge_lists());
		const int64_t index_size = g.row_starts_[num_columns];
		const int64_t num_local_vertices = (int64_t(1) << g.log_local_verts());

		CudaStreamManager::begin_cuda();
		if(graph_on_gpu_) {
			CUDA_CHECK(cudaMalloc((void**)&g.dev_row_starts_,
					sizeof(g.dev_row_starts_[0])*(num_columns+2)));
			CUDA_CHECK(cudaMalloc((void**)&g.dev_edge_array_high_,
					sizeof(g.dev_edge_array_high_[0])*index_size));
			CUDA_CHECK(cudaMalloc((void**)&g.dev_edge_array_low_,
					sizeof(g.dev_edge_array_low_[0])*index_size));
		}
		else {
			g.dev_row_starts_ = NULL;
			g.dev_edge_array_high_ = NULL;
			g.dev_edge_array_low_ = NULL;
		}
		CUDA_CHECK(cudaMalloc((void**)&g.dev_invert_vertex_mapping_,
				sizeof(g.dev_invert_vertex_mapping_[0])*num_local_vertices));

		if(graph_on_gpu_) {
			CUDA_CHECK(cudaMemcpy(g.dev_row_starts_, g.row_starts_,
					sizeof(g.dev_row_starts_[0])*(num_columns+1), cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(g.dev_edge_array_high_, g.edge_array_.get_ptr_high(),
					sizeof(g.dev_edge_array_high_[0])*index_size, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaMemcpy(g.dev_edge_array_low_, g.edge_array_.get_ptr_low(),
					sizeof(g.dev_edge_array_low_[0])*index_size, cudaMemcpyHostToDevice));
			// add an empty column
			CUDA_CHECK(cudaMemcpy(g.dev_row_starts_ + num_columns + 1, &index_size,
					sizeof(g.dev_row_starts_[0]), cudaMemcpyHostToDevice));
		}
		CUDA_CHECK(cudaMemcpy(g.dev_invert_vertex_mapping_, g.invert_vertex_mapping_,
				sizeof(g.dev_invert_vertex_mapping_[0])*num_local_vertices, cudaMemcpyHostToDevice));
		CudaStreamManager::end_cuda();
#endif
	}

private:

	// step1: for computing degree order
	void initializeParameters(
		int log_max_vertex,
		int64_t num_global_edges,
		GraphType& g)
	{
		int64_t num_global_verts = int64_t(1) << log_max_vertex;
		int64_t local_verts_unit = int64_t(1) << log_local_verts_unit_;
		int64_t num_local_verts = roundup(num_global_verts / mpi.size_2d, local_verts_unit);

		// estimated SCALE parameter
		g.log_orig_global_verts_ = log_max_vertex;
		g.num_orig_local_verts_ = num_local_verts;
		g.orig_local_bits_ = org_local_bits_ = get_msb_index(num_local_verts - 1) + 1;

		degree_calc_ = new DegreeCalculation(org_local_bits_, log_local_verts_unit_);
	}

	// step2: for graph construction
	void makeWideRowStarts(GraphType& g) {

		// count degree
		GraphConstructionData data = degree_calc_->process();

		g.reorder_map_ = data.reordre_map_;
		g.invert_map_ = data.invert_map_;
		g.num_local_verts_ = data.num_local_verts_;
		local_bits_ = g.local_bits_ = get_msb_index(g.num_local_verts_ - 1) + 1;
		g.r_bits_ = (mpi.size_2dr == 1) ? 0 : (get_msb_index(mpi.size_2dr - 1) + 1);
		num_wide_rows_ = g.num_local_verts_ * mpi.size_2dc / EDGE_PART_SIZE;
		wide_row_starts_ = data.wide_row_starts_;
		row_starts_sup_ = data.row_starts_sup_;
		g.row_bitmap_ = data.row_bitmap_;
		g.row_sums_ = data.row_sums_;
		g.orig_vertexes_ = data.orig_vertexes_;

		delete degree_calc_; degree_calc_ = NULL;
	}

	// function #1
	template<typename EdgeType>
	void maxVertex(const EdgeType* edge_data, const int edge_data_length,
			uint64_t& max_vertex, int& max_weight, typename EdgeType::has_weight dummy = 0)
	{
#pragma omp for schedule(static)
		for(int i = 0; i < edge_data_length; ++i) {
			const int64_t v0 = edge_data[i].v0();
			const int64_t v1 = edge_data[i].v1();
			const int weight = edge_data[i].weight_;
			max_vertex |= (uint64_t)(v0 | v1);
			if(max_weight < weight) max_weight = weight;
		} // #pragma omp for schedule(static)
	}

	// function #2
	template<typename EdgeType>
	void maxVertex(const EdgeType* edge_data, const int edge_data_length,
			uint64_t& max_vertex, int& max_weight, typename EdgeType::no_weight dummy = 0)
	{
#pragma omp for schedule(static)
		for(int i = 0; i < edge_data_length; ++i) {
			const int64_t v0 = edge_data[i].v0();
			const int64_t v1 = edge_data[i].v1();
			max_vertex |= (uint64_t)(v0 | v1);
		} // #pragma omp for schedule(static)
	}

	// using SFINAE
	// function #1
	template<typename EdgeType>
	void scanEdges(const EdgeType* edge_data, const int edge_data_length,
			int* restrict counts, typename EdgeType::has_weight dummy = 0)
	{
#pragma omp for schedule(static)
		for(int i = 0; i < edge_data_length; ++i) {
			const int64_t v0 = edge_data[i].v0();
			const int64_t v1 = edge_data[i].v1();
			const int weight = edge_data[i].weight_;
			if (v0 == v1) continue;
			(counts[edge_owner(v0,v1)])++;
			(counts[edge_owner(v1,v0)])++;
		} // #pragma omp for schedule(static)
	}

	// function #2
	template<typename EdgeType>
	void scanEdges(const EdgeType* edge_data, const int edge_data_length,
			int* restrict counts, typename EdgeType::no_weight dummy = 0)
	{
#pragma omp for schedule(static)
		for(int i = 0; i < edge_data_length; ++i) {
			const int64_t v0 = edge_data[i].v0();
			const int64_t v1 = edge_data[i].v1();
			if (v0 == v1) continue;
			(counts[edge_owner(v0,v1)])++;
			(counts[edge_owner(v1,v0)])++;
		} // #pragma omp for schedule(static)
	}

	// using SFINAE
	// function #1
	template<typename EdgeType>
	void reduceMaxWeight(int max_weight, GraphType& g, typename EdgeType::has_weight dummy = 0)
	{
		int global_max_weight;
		MPI_Allreduce(&max_weight, &global_max_weight, 1, MPI_INT, MPI_MAX, mpi.comm_2d);
		g.max_weight_ = global_max_weight;
		g.log_max_weight_ = get_msb_index(global_max_weight);
	}

	// function #2
	template<typename EdgeType>
	void reduceMaxWeight(int max_weight, GraphType& g, typename EdgeType::no_weight dummy = 0)
	{
	}

	void searchMaxVertex(EdgeList* edge_list, GraphType& g) {
		TRACER(scan_vertex);
		uint64_t max_vertex = 0;
		int max_weight = 0;

		if(mpi.isMaster()) print_with_prefix("Searching max vertex id...");

		int num_loops = edge_list->beginRead(false);
		for(int loop_count = 0; loop_count < num_loops; ++loop_count) {
			EdgeType* edge_data;
			const int edge_data_length = edge_list->read(&edge_data);
#pragma omp parallel reduction(|:max_vertex) reduction(+:max_weight)
			{
				maxVertex(edge_data, edge_data_length, max_vertex, max_weight);
			} // #pragma omp parallel
		}
		edge_list->endRead();

		{
			uint64_t tmp_send = max_vertex;
			MPI_Allreduce(&tmp_send, &max_vertex, 1, MpiTypeOf<uint64_t>::type, MPI_BOR, mpi.comm_2d);

			const int log_max_vertex = get_msb_index(max_vertex) + 1;
			if(mpi.isMaster()) print_with_prefix("Estimated SCALE = %d.", log_max_vertex);

			initializeParameters(log_max_vertex, edge_list->num_local_edges()*mpi.size_2d, g);
			reduceMaxWeight<EdgeType>(max_weight, g);
		}
	}

	void scatterAndScanEdges(EdgeList* edge_list, GraphType& g) {
		TRACER(scan_edge);
		ScatterContext scatter(mpi.comm_2d);
		int64_t* edges_to_send = static_cast<int64_t*>(
				xMPI_Alloc_mem(2 * EdgeList::CHUNK_SIZE * sizeof(int64_t)));
		int num_loops = edge_list->beginRead(false);

		if(mpi.isMaster()) print_with_prefix("Begin counting degree. Number of iterations is %d.", num_loops);

		for(int loop_count = 0; loop_count < num_loops; ++loop_count) {
			EdgeType* edge_data;
			const int edge_data_length = edge_list->read(&edge_data);

#pragma omp parallel
			{
				int* restrict counts = scatter.get_counts();

#pragma omp for schedule(static)
				for(int i = 0; i < edge_data_length; ++i) {
					const int64_t v0 = edge_data[i].v0();
					const int64_t v1 = edge_data[i].v1();
					if (v0 == v1) continue;
					(counts[vertex_owner(v0)])++;
					(counts[vertex_owner(v1)])++;
				} // #pragma omp for schedule(static)
			} // #pragma omp parallel

			scatter.sum();

#if NETWORK_PROBLEM_AYALISYS
			if(mpi.isMaster()) print_with_prefix("MPI_Allreduce...");
#endif

#if NETWORK_PROBLEM_AYALISYS
			if(mpi.isMaster()) print_with_prefix("OK! ");
#endif

			const int local_bits = org_local_bits_;
#pragma omp parallel
			{
				int* restrict offsets = scatter.get_offsets();

#pragma omp for schedule(static)
				for(int i = 0; i < edge_data_length; ++i) {
					const int64_t v0 = edge_data[i].v0();
					const int64_t v1 = edge_data[i].v1();
					if (v0 == v1) continue;
					// high: v1's c, low: v0's vertex_local, lgl: local_bits
					const SeparatedId v0_swizzled(vertex_owner_c(v1), vertex_local(v0), local_bits);
					const SeparatedId v1_swizzled(vertex_owner_c(v0), vertex_local(v1), local_bits);
					//assert (offsets[edge_owner(v0,v1)] < 2 * FILE_CHUNKSIZE);
					edges_to_send[(offsets[vertex_owner(v0)])++] = v0_swizzled.value;
					//assert (offsets[edge_owner(v1,v0)] < 2 * FILE_CHUNKSIZE);
					edges_to_send[(offsets[vertex_owner(v1)])++] = v1_swizzled.value;
				} // #pragma omp for schedule(static)
			} // #pragma omp parallel

#if NETWORK_PROBLEM_AYALISYS
			if(mpi.isMaster()) print_with_prefix("MPI_Alltoall...");
#endif

			int64_t* recv_edges = scatter.scatter(edges_to_send);

#if NETWORK_PROBLEM_AYALISYS
			if(mpi.isMaster()) print_with_prefix("OK! ");
#endif

			const int64_t num_recv_edges = scatter.get_recv_count();
			degree_calc_->add(recv_edges, num_recv_edges);

			scatter.free(recv_edges);

			if(mpi.isMaster()) print_with_prefix("Iteration %d finished.", loop_count);
		}
		edge_list->endRead();
		MPI_Free_mem(edges_to_send);

		if(mpi.isMaster()) print_with_prefix("Finished scattering edges.");
	}

	void isolateFirstEdge(GraphType& g) {
		const int64_t num_local_verts = g.num_local_verts_;
		const int64_t local_bitmap_width = num_local_verts / NBPE;
		const int64_t row_bitmap_length = local_bitmap_width * mpi.size_2dc;
		const int64_t non_zero_rows = g.row_sums_[row_bitmap_length];
		g.isolated_edges_ = static_cast<int64_t*>(
				cache_aligned_xmalloc(non_zero_rows*sizeof(g.isolated_edges_[0])));

		if(mpi.isMaster()) print_with_prefix("Isolating first edges.");
		for(int64_t non_zero_idx = 0; non_zero_idx < non_zero_rows; ++non_zero_idx) {
			int64_t e_start = g.row_starts_[non_zero_idx];
			g.isolated_edges_[non_zero_idx] = g.edge_array_[e_start];
		}
		MPI_Barrier(mpi.comm_2d);

		if(mpi.isMaster()) print_with_prefix("Compacting edge array.");
		// This loop cannot be parallelized.
		for(int64_t non_zero_idx = 0; non_zero_idx < non_zero_rows; ++non_zero_idx) {
			int64_t e_start = g.row_starts_[non_zero_idx];
			int64_t e_end = g.row_starts_[non_zero_idx + 1];
			int64_t e_length = e_end - e_start;
			memmove(g.edge_array_ + e_start - non_zero_idx, g.edge_array_ + e_start + 1, sizeof(int64_t) * (e_length - 1));
			g.row_starts_[non_zero_idx] -= non_zero_idx;
		}

		// update the last entry of row_starts_ and compact the edge array memory
		g.row_starts_[non_zero_rows] -= non_zero_rows;
		g.edge_array_ = static_cast<int64_t*>(realloc(g.edge_array_,
				g.row_starts_[non_zero_rows] * sizeof(int64_t)));
		if(g.row_starts_[non_zero_rows] != 0 && g.edge_array_ == NULL) {
			throw_exception("Out of memory trying to re-allocate edge array");
		}

		if(mpi.isMaster()) print_with_prefix("Finished compacting edge array.");
	}

	void constructFromWideCSR(GraphType& g) {
		TRACER(form_csr);
		const int64_t num_local_verts = g.num_local_verts_;
		const int64_t src_region_length = num_local_verts * mpi.size_2dc;
		const int64_t row_bitmap_length = src_region_length >> LOG_NBPE;
		VERVOSE(const int64_t num_local_edges = wide_row_starts_[num_wide_rows_]);

		VERVOSE(if(mpi.isMaster()) {
			print_with_prefix("num_local_verts %f M", to_mega(num_local_verts));
			print_with_prefix("src_region_length %f M", to_mega(src_region_length));
			print_with_prefix("num_wide_rows %f M", to_mega(num_wide_rows_));
			print_with_prefix("row_bitmap_length %f M", to_mega(row_bitmap_length));
			print_with_prefix("local_bits=%d", local_bits_);
			print_with_prefix("correspond to %f M", to_mega(int64_t(1) << local_bits_));
		});

		const int64_t non_zero_rows = g.row_sums_[row_bitmap_length];
		int64_t* row_starts = static_cast<int64_t*>
			(cache_aligned_xmalloc((non_zero_rows+1)*sizeof(int64_t)));

		if(mpi.isMaster()) print_with_prefix("Computing row_starts.");
#pragma omp parallel for
		for(int64_t part_base = 0; part_base < src_region_length; part_base += EDGE_PART_SIZE) {
			int64_t part_idx = part_base >> LOG_EDGE_PART_SIZE;
			int64_t row_length[EDGE_PART_SIZE] = {0};
			int64_t edge_offset = wide_row_starts_[part_idx];
			for(int64_t i = wide_row_starts_[part_idx]; i < wide_row_starts_[part_idx+1]; ++i) {
				++(row_length[src_vertexes_[i] & EDGE_PART_SIZE_MASK]);
			}
			int part_end = (int)std::min<int64_t>(EDGE_PART_SIZE, src_region_length - part_base);
			for(int64_t i = 0; i < part_end; ++i) {
				int64_t word_idx = (part_base + i) >> LOG_NBPE;
				int bit_idx = i & NBPE_MASK;
				if(g.row_bitmap_[word_idx] & (BitmapType(1) << bit_idx)) {
					assert (row_length[i] > 0);
					BitmapType word = g.row_bitmap_[word_idx] & ((BitmapType(1) << bit_idx) - 1);
					TwodVertex row_offset = __builtin_popcountl(word) + g.row_sums_[word_idx];
					row_starts[row_offset] = edge_offset;
					edge_offset += row_length[i];
				}
				else {
					assert (row_length[i] == 0);
				}
			}
			assert (edge_offset == wide_row_starts_[part_idx+1]);
		}
		row_starts[non_zero_rows] = wide_row_starts_[num_wide_rows_];

#ifndef NDEBUG
		// check row_starts
#pragma omp parallel for
		for(int64_t part_base = 0; part_base < src_region_length; part_base += EDGE_PART_SIZE) {
			int64_t part_size = std::min<int64_t>(src_region_length - part_base, EDGE_PART_SIZE);
			int64_t part_idx = part_base >> LOG_EDGE_PART_SIZE;
			int64_t word_idx = part_base >> LOG_NBPE;
			int64_t nz_idx = g.row_sums_[word_idx];
			int num_rows = g.row_sums_[word_idx + part_size / NBPE] - nz_idx;

			for(int i = 0; i < part_size / NBPE; ++i) {
				int64_t diff = g.row_sums_[word_idx + i + 1] - g.row_sums_[word_idx + i];
				assert (diff == __builtin_popcountl(g.row_bitmap_[word_idx + i]));
			}

			assert (row_starts[nz_idx] == wide_row_starts_[part_idx]);
			for(int i = 0; i < num_rows; ++i) {
				assert (row_starts[nz_idx + i + 1] > row_starts[nz_idx + i]);
			}
		}
#endif // #ifndef NDEBUG

		// delete wide row structure
		free(wide_row_starts_); wide_row_starts_ = NULL;
		free(src_vertexes_); src_vertexes_ = NULL;

		g.row_starts_ = row_starts;
#if ISOLATE_FIRST_EDGE
		isolateFirstEdge(g);
#endif // #if ISOLATE_FIRST_EDGE || DEGREE_ORDER

#if VERVOSE_MODE
		int64_t send_rowbmp[5] = { non_zero_rows, row_bitmap_length*NBPE, num_local_edges, 0, 0 };
		int64_t max_rowbmp[5];
		int64_t sum_rowbmp[5];
		MPI_Reduce(send_rowbmp, sum_rowbmp, 5, MpiTypeOf<int64_t>::type, MPI_SUM, 0, mpi.comm_2d);
		MPI_Reduce(send_rowbmp, max_rowbmp, 5, MpiTypeOf<int64_t>::type, MPI_MAX, 0, mpi.comm_2d);
		if(mpi.isMaster()) {
			int64_t local_bits_max = int64_t(1) << local_bits_;
			print_with_prefix("non zero rows. Total %f M / %f M = %f %% Avg %f M / %f M Max %f %%+",
					to_mega(sum_rowbmp[0]), to_mega(sum_rowbmp[1]), to_mega(sum_rowbmp[0]) / to_mega(sum_rowbmp[1]) * 100,
					to_mega(sum_rowbmp[0]) / mpi.size_2d, to_mega(sum_rowbmp[1]) / mpi.size_2d,
					diff_percent(max_rowbmp[0], sum_rowbmp[0], mpi.size_2d));
			print_with_prefix("distributed edges. Total %f M Avg %f M Max %f %%+",
					to_mega(sum_rowbmp[2]), to_mega(sum_rowbmp[2]) / mpi.size_2d,
					diff_percent(max_rowbmp[2], sum_rowbmp[2], mpi.size_2d));
			print_with_prefix("Type requirements:");
			print_with_prefix("Global vertex id %s using %s", minimum_type(num_local_verts * mpi.size_2d), TypeName<int64_t>::value);
			print_with_prefix("Local vertex id %s using %s", minimum_type(num_local_verts), TypeName<uint32_t>::value);
			print_with_prefix("Index for local edges %s using %s", minimum_type(max_rowbmp[2]), TypeName<int64_t>::value);
			print_with_prefix("*Index for src local region %s using %s", minimum_type(local_bits_max * mpi.size_2dc), TypeName<TwodVertex>::value);
			print_with_prefix("*Index for dst local region %s using %s", minimum_type(local_bits_max * mpi.size_2dr), TypeName<TwodVertex>::value);
			print_with_prefix("Index for non zero rows %s using %s", minimum_type(max_rowbmp[0]), TypeName<TwodVertex>::value);
			print_with_prefix("*BFELL sort region size %s using %s", minimum_type(BFELL_SORT), TypeName<SortIdx>::value);
			print_with_prefix("Memory consumption:");
			print_with_prefix("row_bitmap %f MB", to_mega(row_bitmap_length*sizeof(BitmapType)));
			print_with_prefix("row_sums %f MB", to_mega((row_bitmap_length+1)*sizeof(TwodVertex)));
			print_with_prefix("edge_array %f MB", to_mega(max_rowbmp[2]*sizeof(TwodVertex)));
			print_with_prefix("row_starts %f MB", to_mega(max_rowbmp[0]*sizeof(int64_t)));
		}
#endif // #if VERVOSE_MODE
	}

	// using SFINAE
	// function #1
	template<typename EdgeType>
	void writeSendEdges(const EdgeType* edge_data, const int edge_data_length,
			int* restrict offsets, EdgeType* edges_to_send, typename EdgeType::has_weight dummy = 0)
	{
		const int local_bits = org_local_bits_;
#pragma omp for schedule(static)
		for(int i = 0; i < edge_data_length; ++i) {
			const int64_t v0 = edge_data[i].v0();
			const int64_t v1 = edge_data[i].v1();
			if (v0 == v1) continue;
			const SeparatedId v0_src(vertex_owner_c(v0), vertex_local(v0), local_bits);
			const SeparatedId v1_src(vertex_owner_c(v1), vertex_local(v1), local_bits);
			const SeparatedId v0_dst(vertex_owner_r(v0), vertex_local(v0), local_bits);
			const SeparatedId v1_dst(vertex_owner_r(v1), vertex_local(v1), local_bits);
			//assert (offsets[edge_owner(v0,v1)] < 2 * FILE_CHUNKSIZE);
			edges_to_send[(offsets[edge_owner(v0,v1)])++].set(v0_src.value, v1_dst.value, edge_data[i].weight_);
			//assert (offsets[edge_owner(v1,v0)] < 2 * FILE_CHUNKSIZE);
			edges_to_send[(offsets[edge_owner(v1,v0)])++].set(v1_src.value, v0_dst.value, edge_data[i].weight_);
		} // #pragma omp for schedule(static)
	}

	// function #2
	template<typename EdgeType>
	void writeSendEdges(const EdgeType* edge_data, const int edge_data_length,
			int* restrict offsets, EdgeType* edges_to_send, typename EdgeType::no_weight dummy = 0)
	{
		const int local_bits = org_local_bits_;
#pragma omp for schedule(static)
		for(int i = 0; i < edge_data_length; ++i) {
			const int64_t v0 = edge_data[i].v0();
			const int64_t v1 = edge_data[i].v1();
			if (v0 == v1) continue;
			const SeparatedId v0_src(vertex_owner_c(v0), vertex_local(v0), local_bits);
			const SeparatedId v1_src(vertex_owner_c(v1), vertex_local(v1), local_bits);
			const SeparatedId v0_dst(vertex_owner_r(v0), vertex_local(v0), local_bits);
			const SeparatedId v1_dst(vertex_owner_r(v1), vertex_local(v1), local_bits);
			//assert (offsets[edge_owner(v0,v1)] < 2 * FILE_CHUNKSIZE);
			edges_to_send[(offsets[edge_owner(v0,v1)])++].set(v0_src.value, v1_dst.value);
			//assert (offsets[edge_owner(v1,v0)] < 2 * FILE_CHUNKSIZE);
			edges_to_send[(offsets[edge_owner(v1,v0)])++].set(v1_src.value, v0_dst.value);
		} // #pragma omp for schedule(static)
	}
/*
	// using SFINAE
	// function #1
	template<typename EdgeType>
	void addEdges(EdgeType* edges, int num_edges, GraphType& g, typename EdgeType::has_weight dummy = 0)
	{
		const int64_t L = g.num_local_verts_;
		const int lgl = local_bits_;
		const int log_local_src = local_bits_ + get_msb_index(mpi.size_2dc-1) + 1;
#pragma omp parallel for schedule(static)
		for(int i = 0; i < num_edges; ++i) {
			const SeparatedId v0(edges[i].v0());
			const SeparatedId v1(edges[i].v1());
			const int weight = edges[i].weight_;

			const int src_high = v0.compact(lgl, L) >> LOG_EDGE_PART_SIZE;
			const uint16_t src_low = v0.compact(lgl, L) & EDGE_PART_SIZE_MASK;
			const int64_t pos = __sync_fetch_and_add(&wide_row_starts_[src_high], 1);

			// random access (write)
#ifndef NDEBUG
			assert( g.edge_array_[pos] == 0 );
#endif
			src_vertexes_[pos] = src_low;
			g.edge_array_[pos] = (weight << log_local_src) | v1.value;
		}
	}
*/
	// function #2
	void addEdges(int64_t* src_converted, int64_t* tgt_converted, int num_edges, GraphType& g)
	{
		const int64_t L = g.num_local_verts_;
		const int lgl = local_bits_;
		ParallelPartitioning<int64_t> row_length_counter(num_wide_rows_);

#pragma omp parallel
		{
			int64_t* counts = row_length_counter.get_counts();

#pragma omp for schedule(static)
			for(int i = 0; i < num_edges; ++i) {
				const SeparatedId v0(src_converted[i]);
				const int src_high = v0.compact(lgl, L) >> LOG_EDGE_PART_SIZE;
				counts[src_high]++;
			}
		}

		row_length_counter.sum(wide_row_starts_);

#pragma omp parallel
		{
			int64_t* offsets = row_length_counter.get_offsets();

#pragma omp for schedule(static)
			for(int i = 0; i < num_edges; ++i) {
				const SeparatedId v0(src_converted[i]);
				const SeparatedId v1(tgt_converted[i]);

				const int src_high = v0.compact(lgl, L) >> LOG_EDGE_PART_SIZE;
				const uint16_t src_low = v0.compact(lgl, L) & EDGE_PART_SIZE_MASK;
				const int64_t pos = offsets[src_high]++;

				// random access (write)
	#ifndef NDEBUG
				assert( g.edge_array_[pos] == 0 );
	#endif
				src_vertexes_[pos] = src_low;
				g.edge_array_[pos] = v1.value;
			}
		}
	}

	class SourceConverter
	{
	public:
		typedef TwodVertex send_type;
		typedef TwodVertex recv_type;

		SourceConverter(EdgeType* edges, int64_t* converted, LocalVertex* reorder_map, int org_local_bits, int local_bits)
			: edges_(edges)
			, reorder_map_(reorder_map)
			, converted_(converted)
			, org_local_bits_(org_local_bits)
			, local_bits_(local_bits)
		{ }
		int target(int i) const {
			const SeparatedId v0_swizzled(edges_[i].v0());
			assert (v0_swizzled.high(org_local_bits_) < mpi.size_2dc);
			return v0_swizzled.high(org_local_bits_);
		}
		TwodVertex get(int i) const {
			return SeparatedId(edges_[i].v0()).low(org_local_bits_);
		}
		TwodVertex map(TwodVertex v) const {
			return reorder_map_[v];
		}
		void set(int i, TwodVertex d) const {
			SeparatedId v0_swizzled(edges_[i].v0());
			converted_[i] = SeparatedId(v0_swizzled.high(org_local_bits_), d, local_bits_).value;
		}
	private:
		EdgeType* const edges_;
		LocalVertex* reorder_map_;
		int64_t* converted_;
		const int org_local_bits_;
		const int local_bits_;
	};

	class TargetConverter
	{
	public:
		typedef TwodVertex send_type;
		typedef TwodVertex recv_type;

		TargetConverter(EdgeType* edges, int64_t* converted, LocalVertex* reorder_map, int org_local_bits, int local_bits, int vertex_bits)
			: edges_(edges)
			, reorder_map_(reorder_map)
			, converted_(converted)
			, org_local_bits_(org_local_bits)
			, local_bits_(local_bits)
			, vertex_bits_(vertex_bits)
		{ }
		int target(int i) const {
			const SeparatedId v1_swizzled(edges_[i].v1());
			assert (v1_swizzled.high(org_local_bits_) < mpi.size_2dr);
			return v1_swizzled.high(org_local_bits_);
		}
		TwodVertex get(int i) const {
			return SeparatedId(edges_[i].v1()).low(org_local_bits_);
		}
		TwodVertex map(TwodVertex v) const {
			return reorder_map_[v];
		}
		void set(int i, TwodVertex d) const {
			SeparatedId v1_swizzled(edges_[i].v1());
			int64_t low_part = SeparatedId(v1_swizzled.high(org_local_bits_), d, local_bits_).value;
			converted_[i] = SeparatedId(v1_swizzled.low(org_local_bits_), low_part, vertex_bits_).value;
		}
	private:
		EdgeType* const edges_;
		LocalVertex* reorder_map_;
		int64_t* converted_;
		const int org_local_bits_;
		const int local_bits_;
		const int vertex_bits_;
	};

	void scatterAndStore(EdgeList* edge_list, GraphType& g) {
		TRACER(store_edge);
		ScatterContext scatter(mpi.comm_2d);
		EdgeType* edges_to_send = static_cast<EdgeType*>(
				xMPI_Alloc_mem(2 * EdgeList::CHUNK_SIZE * sizeof(EdgeType)));

		//const int64_t num_local_verts = g.num_local_verts_;
		g.edge_array_ = (int64_t*)cache_aligned_xcalloc(wide_row_starts_[num_wide_rows_]*sizeof(int64_t));
		src_vertexes_ = (uint16_t*)cache_aligned_xcalloc(wide_row_starts_[num_wide_rows_]*sizeof(uint16_t));

		int num_loops = edge_list->beginRead(true);

		if(mpi.isMaster()) print_with_prefix("Begin construction. Number of iterations is %d.", num_loops);

		for(int loop_count = 0; loop_count < num_loops; ++loop_count) {
			EdgeType* edge_data;
			const int edge_data_length = edge_list->read(&edge_data);

#pragma omp parallel
			{
				int* restrict counts = scatter.get_counts();

#pragma omp for schedule(static)
				for(int i = 0; i < edge_data_length; ++i) {
					const int64_t v0 = edge_data[i].v0();
					const int64_t v1 = edge_data[i].v1();
					if (v0 == v1) continue;
					(counts[edge_owner(v0,v1)])++;
					(counts[edge_owner(v1,v0)])++;
				} // #pragma omp for schedule(static)
			} // #pragma omp parallel

			scatter.sum();

#pragma omp parallel
			{
				int* offsets = scatter.get_offsets();
				writeSendEdges(edge_data, edge_data_length, offsets, edges_to_send);
			}

			if(mpi.isMaster()) print_with_prefix("Scatter edges...");

			EdgeType* recv_edges = scatter.scatter(edges_to_send);
			const int num_recv_edges = scatter.get_recv_count();

			int64_t* src_converted = (int64_t*)cache_aligned_xmalloc(num_recv_edges*sizeof(int64_t));
			int64_t* tgt_converted = (int64_t*)cache_aligned_xmalloc(num_recv_edges*sizeof(int64_t));

			if(mpi.isMaster()) print_with_prefix("Convert vertex id...");

			MpiCol::gather(
					SourceConverter(
							recv_edges,
							src_converted,
							g.reorder_map_,
							org_local_bits_,
							local_bits_),
					num_recv_edges,
					mpi.comm_2dr);

			MpiCol::gather(
					TargetConverter(
							recv_edges,
							tgt_converted,
							g.reorder_map_,
							org_local_bits_,
							local_bits_,
							g.r_bits_ + local_bits_),
					num_recv_edges,
					mpi.comm_2dc);

			if(mpi.isMaster()) print_with_prefix("Add edges...");
			addEdges(src_converted, tgt_converted, num_recv_edges, g);

			free(src_converted);
			free(tgt_converted);
			scatter.free(recv_edges);

			if(mpi.isMaster()) print_with_prefix("Iteration %d finished.", loop_count);
		}

		edge_list->endRead();
		MPI_Free_mem(edges_to_send);

		if(mpi.isMaster()) print_with_prefix("Refreshing edge offset.");
		memmove(wide_row_starts_+1, wide_row_starts_, num_wide_rows_*sizeof(wide_row_starts_[0]));
		wide_row_starts_[0] = 0;

#ifndef NDEBUG
#pragma omp parallel for
		for(int64_t i = 0; i <= num_wide_rows_; ++i) {
			if(row_starts_sup_[i] != wide_row_starts_[i]) {
				print_with_prefix("Error: Edge Counts: i=%"PRId64",1st=%"PRId64",2nd=%"PRId64"", i, row_starts_sup_[i], wide_row_starts_[i]);
			}
			assert(row_starts_sup_[i] == wide_row_starts_[i]);
		}
#endif
	}

	// using SFINAE
	// function #1
	template<typename EdgeType>
	void sortEdgesInner(GraphType& g, typename EdgeType::has_weight dummy = 0)
	{
		/*
		int64_t sort_buffer_length = 2*1024;
		int64_t* restrict sort_buffer = (int64_t*)cache_aligned_xmalloc(sizeof(int64_t)*sort_buffer_length);
		const int64_t num_local_verts = (int64_t(1) << g.log_local_verts());
		const int64_t src_region_length = num_local_verts * mpi.size_2dc;
		const int64_t num_wide_rows = std::max<int64_t>(1, src_region_length >> LOG_EDGE_PART_SIZE);

		const int64_t num_edge_lists = (int64_t(1) << g.log_edge_lists());
		const int log_weight_bits = g.log_packing_edge_lists_;
		const int log_packing_edge_lists = g.log_packing_edge_lists();
		const int index_bits = g.log_global_verts() - get_msb_index(mpi.size_2dr);
		const int64_t mask_packing_edge_lists = (int64_t(1) << log_packing_edge_lists) - 1;
		const int64_t mask_weight = (int64_t(1) << log_weight_bits) - 1;
		const int64_t mask_index = (int64_t(1) << index_bits) - 1;
		const int64_t mask_index_compare =
				(mask_index << (log_packing_edge_lists + log_weight_bits)) |
				mask_packing_edge_lists;

#define ENCODE(v) \
		(((((v & mask_packing_edge_lists) << log_weight_bits) | \
		((v >> log_packing_edge_lists) & mask_weight)) << index_bits) | \
		(v >> (log_packing_edge_lists + log_weight_bits)))
#define DECODE(v) \
		(((((v & mask_index) << log_weight_bits) | \
		((v >> index_bits) & mask_weight)) << log_packing_edge_lists) | \
		(v >> (index_bits + log_weight_bits)))

#pragma omp for
		for(int64_t i = 0; i < num_edge_lists; ++i) {
			const int64_t edge_count = wide_row_starts_[i];
			const int64_t rowstart_i = g.row_starts_[i];
			assert (g.row_starts_[i+1] - g.row_starts_[i] == wide_row_starts_[i]);

			if(edge_count > sort_buffer_length) {
				free(sort_buffer);
				while(edge_count > sort_buffer_length) sort_buffer_length *= 2;
				sort_buffer = (int64_t*)cache_aligned_xmalloc(sizeof(int64_t)*sort_buffer_length);
			}

			for(int64_t c = 0; c < edge_count; ++c) {
				const int64_t v = g.edge_array_(rowstart_i + c);
				sort_buffer[c] = ENCODE(v);
				assert(v == DECODE(ENCODE(v)));
			}
			// sort sort_buffer
			std::sort(sort_buffer, sort_buffer + edge_count);

			int64_t idx = rowstart_i;
			int64_t prev_v = -1;
			for(int64_t c = 0; c < edge_count; ++c) {
				const int64_t sort_v = sort_buffer[c];
				// TODO: now duplicated edges are not merged because sort order is
				// v0 row bits > weight > index
				// To reduce parallel edges, sort by the order of
				// v0 row bits > index > weight
				// and if you want to optimize SSSP, sort again by the order of
				// v0 row bits > weight > index
			//	if((prev_v & mask_index_compare) != (sort_v & mask_index_compare)) {
					assert (prev_v < sort_v);
					const int64_t v = DECODE(sort_v);
					g.edge_array_.set(idx, v);
			//		prev_v = sort_v;
					idx++;
			//	}
			}
		//	if(wide_row_starts_[i] > idx - rowstart_i) {
				wide_row_starts_[i] = idx - rowstart_i;
		//	}
		} // #pragma omp for

#undef ENCODE
#undef DECODE
		free(sort_buffer);
*/
	}

	struct SortEdgeCompair {
		typedef pointer_pair_value<uint16_t, int64_t> Val;
		SortEdgeCompair(int vertex_bits) : edge_mask((int64_t(1) << vertex_bits) - 1) { }
		int64_t edge_mask;

		bool operator ()(Val r1, Val r2) const {
			uint64_t r1_v = (uint64_t(r1.v1) << 48) | (r1.v2 & edge_mask);
			uint64_t r2_v = (uint64_t(r2.v1) << 48) | (r2.v2 & edge_mask);
			return r1_v < r2_v;
		}
	};

	// function #2
	template<typename EdgeType>
	void sortEdgesInner(GraphType& g, typename EdgeType::no_weight dummy = 0)
	{
#pragma omp for
		for(int64_t i = 0; i < num_wide_rows_; ++i) {
			const int64_t edge_offset = wide_row_starts_[i];
			const int64_t edge_count = wide_row_starts_[i+1] - edge_offset;
			int vertex_bits = g.r_bits_ + local_bits_;
			int64_t edge_mask = (int64_t(1) << vertex_bits) - 1;

			// sort
			sort2(src_vertexes_ + edge_offset, g.edge_array_ + edge_offset, edge_count,
					SortEdgeCompair(vertex_bits));

			// merge same edges
			int64_t idx = edge_offset;
			if(edge_count > 0) {
				// we can ignore the original vertex id
				uint64_t prev_v = (uint64_t(src_vertexes_[idx]) << 48) |
						(g.edge_array_[idx] & edge_mask);
				++idx;
				for(int64_t c = edge_offset+1; c < edge_offset + edge_count; ++c) {
					const uint64_t sort_v = (uint64_t(src_vertexes_[c]) << 48) |
							(g.edge_array_[c] & edge_mask);
					if(prev_v != sort_v) {
						assert (prev_v < sort_v);
						g.edge_array_[idx] = g.edge_array_[c];
						src_vertexes_[idx] = src_vertexes_[c];
						prev_v = sort_v;
						++idx;
					}
				}
			}
			row_starts_sup_[i] = idx - edge_offset;
		} // #pragma omp for

	}

	void sortEdges(GraphType& g) {
		TRACER(sort_edge);
		if(mpi.isMaster()) print_with_prefix("Sorting edges.");

#pragma omp parallel
		sortEdgesInner<EdgeType>(g);

		// this loop can't be parallel
		int64_t rowstart_new = 0;
		for(int64_t i = 0; i < num_wide_rows_; ++i) {
			const int64_t edge_count_new = row_starts_sup_[i];
			const int64_t rowstart_old = wide_row_starts_[i]; // read before write
			wide_row_starts_[i] = rowstart_new;
			if(rowstart_new != rowstart_old) {
				memmove(src_vertexes_ + rowstart_new, src_vertexes_ + rowstart_old, edge_count_new * sizeof(src_vertexes_[0]));
				memmove(g.edge_array_ + rowstart_new, g.edge_array_ + rowstart_old, edge_count_new * sizeof(g.edge_array_[0]));
			}
			rowstart_new += edge_count_new;
		}
		const int64_t old_num_edges = wide_row_starts_[num_wide_rows_];
		wide_row_starts_[num_wide_rows_] = rowstart_new;

		 int64_t num_edge_sum[2] = {0};
		 int64_t num_edge[2] = {old_num_edges, rowstart_new};
		MPI_Reduce(num_edge, num_edge_sum, 2, MPI_INT64_T, MPI_SUM, 0, mpi.comm_2d);
		if(mpi.isMaster()) print_with_prefix("# of edges is reduced. Total %zd -> %zd Diff %f %%",
				num_edge_sum[0], num_edge_sum[1], (double)(num_edge_sum[0] - num_edge_sum[1])/(double)num_edge_sum[0]*100.0);
		g.num_global_edges_ = num_edge_sum[1];
	}

	void computeNumVertices(GraphType& g) {
		TRACER(num_verts);
		const int64_t num_local_verts = g.num_local_verts_;
		const int64_t local_bitmap_width = num_local_verts / NBPE;
		int recvcounts[mpi.size_2dc];
		for(int i = 0; i < mpi.size_2dc; ++i) recvcounts[i] = local_bitmap_width;

		g.has_edge_bitmap_ = (BitmapType*)cache_aligned_xmalloc(local_bitmap_width*sizeof(BitmapType));
		MPI_Reduce_scatter(g.row_bitmap_, g.has_edge_bitmap_, recvcounts, MpiTypeOf<BitmapType>::type, MPI_BOR, mpi.comm_2dr);
		int64_t num_vertices = 0;
#pragma omp parallel for reduction(+:num_vertices)
		for(int i = 0; i < local_bitmap_width; ++i) {
			num_vertices += __builtin_popcountl(g.has_edge_bitmap_[i]);
		}
		int64_t tmp_send_num_vertices = num_vertices;
		MPI_Allreduce(&tmp_send_num_vertices, &num_vertices, 1, MpiTypeOf<int64_t>::type, MPI_SUM, mpi.comm_2d);
		VERVOSE(int64_t num_virtual_vertices = int64_t(1) << g.log_orig_global_verts_);
		VERVOSE(if(mpi.isMaster()) print_with_prefix("# of actual vertices %f G %f %%", to_giga(num_vertices),
				(double)num_vertices / (double)num_virtual_vertices * 100.0));
		g.num_global_verts_ = num_vertices;
	}

	//const int log_size_;
	//const int rmask_;
	//const int cmask_;
	int log_local_verts_unit_;
	int64_t num_wide_rows_;

	int org_local_bits_; // local bits for original vertex id
	int local_bits_; // local bits for reordered vertex id

	DegreeCalculation* degree_calc_;

	uint16_t* src_vertexes_;
	int64_t* wide_row_starts_;
	int64_t* row_starts_sup_;
};

} // namespace detail {


#endif /* GRAPH_CONSTRUCTOR_HPP_ */
