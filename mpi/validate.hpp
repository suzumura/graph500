/*
 * validate.hpp
 *
 *  Created on: Mar 2, 2012
 *      Author: koji
 */
/* Copyright (C) 2010-2011 The Trustees of Indiana University.             */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

#ifndef VALIDATE_HPP_
#define VALIDATE_HPP_

#include <mpi.h>
#include <assert.h>

#include <algorithm>

/* One-sided emulation since many MPI implementations don't have good
 * performance and/or fail when using many one-sided operations between fences.
 * Only the necessary operations are implemented, and only MPI_MODE_NOPRECEDE
 * and MPI_MODE_NOSUCCEED fences can be used. */

/* Gather from one array into another. */
struct gather {
  void* input;
  size_t input_count;
  size_t elt_size;
  void* output;
  size_t output_count;
  size_t nrequests_max;
  MPI_Datatype datatype;
  int valid;
  MPI_Comm comm;
  size_t* local_indices;
  int* remote_ranks;
  MPI_Aint* remote_indices;
  int comm_size;
  int* send_counts;
  int* send_offsets;
  int* recv_counts;
  int* recv_offsets;
};

gather* init_gather(void* input, size_t input_count, size_t elt_size, void* output, size_t output_count, size_t nrequests_max, MPI_Datatype dt) {
  gather* g = (gather*)cache_aligned_xmalloc(sizeof(gather));
  g->input = input;
  g->input_count = input_count;
  g->elt_size = elt_size;
  g->output = output;
  g->output_count = output_count;
  g->nrequests_max = nrequests_max;
  g->datatype = dt;
  g->valid = 0;
  MPI_Comm_dup(mpi.comm_2d, &g->comm);
  g->local_indices = (size_t*)page_aligned_xmalloc(nrequests_max * sizeof(size_t));
  g->remote_ranks = (int*)page_aligned_xmalloc(nrequests_max * sizeof(int));
  g->remote_indices = (MPI_Aint*)page_aligned_xmalloc(nrequests_max * sizeof(MPI_Aint));
  MPI_Comm_size(g->comm, &g->comm_size);
  int size = g->comm_size;
  g->send_counts = (int*)cache_aligned_xmalloc((size + 1) * sizeof(int));
  g->send_offsets = (int*)cache_aligned_xmalloc((size + 2) * sizeof(int));
  g->recv_counts = (int*)cache_aligned_xmalloc(size * sizeof(int));
  g->recv_offsets = (int*)cache_aligned_xmalloc((size + 1) * sizeof(int));
#ifndef NDEBUG
  int datasize;
  MPI_Type_size(dt, &datasize);
  assert (datasize == (int)elt_size);
#endif
  return g;
}

void destroy_gather(gather* g) {
  assert (!g->valid);
  free(g->local_indices); g->local_indices = NULL;
  free(g->remote_ranks); g->remote_ranks = NULL;
  free(g->remote_indices); g->remote_indices = NULL;
  MPI_Comm_free(&g->comm);
  free(g->send_counts); g->send_counts = NULL;
  free(g->send_offsets); g->send_offsets = NULL;
  free(g->recv_counts); g->recv_counts = NULL;
  free(g->recv_offsets); g->recv_offsets = NULL;
  g->input = NULL;
  g->output = NULL;
  g->input_count = 0;
  g->output_count = 0;
  free(g);
}

void begin_gather(gather* g) {
  assert (!g->valid);
  {size_t i, nr = g->nrequests_max; for (i = 0; i < nr; ++i) g->remote_ranks[i] = g->comm_size;}
  g->valid = 1;
}

void add_gather_request(gather* g, size_t local_idx, int remote_rank, size_t remote_idx, size_t req_id) {
  assert (g->valid);
  assert (remote_rank >= 0 && remote_rank < g->comm_size);
  assert (req_id < g->nrequests_max);
  assert (local_idx < g->output_count);
  g->local_indices[req_id] = local_idx;
  assert (g->remote_ranks[req_id] == g->comm_size);
  g->remote_ranks[req_id] = remote_rank;
  g->remote_indices[req_id] = (MPI_Aint)remote_idx;
}

/* Adapted from histogram_sort_inplace in boost/graph/detail/histogram_sort.hpp
 * */
void histogram_sort_size_tMPI_Aint
       (int* restrict keys,
        const int* restrict rowstart,
        int numkeys,
        size_t* restrict values1,
        MPI_Aint* restrict values2) {
  int* restrict insert_positions = (int*)cache_aligned_xmalloc(numkeys * sizeof(int));
  memcpy(insert_positions, rowstart, numkeys * sizeof(int));
  int i;
  for (i = 0; i < rowstart[numkeys]; ++i) {
    int key = keys[i];
    assert (key >= 0 && key < numkeys);
    // print_with_prefix("i = %d, key = %d", i, key);
    while (!(i >= rowstart[key] && i < insert_positions[key])) {
      int target_pos = insert_positions[key]++;
      // print_with_prefix("target_pos = %d from limit %d", target_pos, rowstart[key + 1]);
      if (target_pos == i) continue;
      assert (target_pos < rowstart[key + 1]);
      // print_with_prefix("swapping [%d] = (%d, %d, %d) with [%d] = (%d, %d, %d)", i, keys[i], (int)values1[i], (int)values2[i], target_pos, keys[target_pos], (int)values1[target_pos], (int)values2[target_pos]);
      {int t = keys[i]; key = keys[target_pos]; keys[i] = key; keys[target_pos] = t;}
      {size_t t = values1[i]; values1[i] = values1[target_pos]; values1[target_pos] = t;}
      {MPI_Aint t = values2[i]; values2[i] = values2[target_pos]; values2[target_pos] = t;}
      assert (key >= 0 && key < numkeys);
    }
    // print_with_prefix("done");
  }
  for (i = 1; i < rowstart[numkeys]; ++i) {
    assert (keys[i] >= keys[i - 1]);
  }
  free(insert_positions);
}

void end_gather(gather* g) {
  assert (g->valid);
  int size = g->comm_size;
  int* restrict send_counts = g->send_counts;
  int* restrict send_offsets = g->send_offsets;
  int* restrict recv_counts = g->recv_counts;
  int* restrict recv_offsets = g->recv_offsets;
  size_t* restrict local_indices = g->local_indices;
  int* restrict remote_ranks = g->remote_ranks;
  MPI_Aint* restrict remote_indices = g->remote_indices;
  const char* restrict input = (const char*)g->input;
  char* restrict output = (char*)g->output;
  size_t elt_size = g->elt_size;
  MPI_Comm comm = g->comm;
  MPI_Datatype datatype = g->datatype;
  size_t nrequests_max = g->nrequests_max;
#ifndef NDEBUG
  size_t input_count = g->input_count;
#endif
  memset(send_counts, 0, (size + 1) * sizeof(int));
  size_t i;
  for (i = 0; i < nrequests_max; ++i) {
    assert (remote_ranks[i] >= 0 && remote_ranks[i] < size + 1);
    ++send_counts[remote_ranks[i]];
  }
  send_offsets[0] = 0;
  for (i = 0; i < (size_t)size + 1; ++i) {
    assert (send_counts[i] >= 0);
    send_offsets[i + 1] = send_offsets[i] + send_counts[i];
  }
  assert (send_offsets[size + 1] == (int)nrequests_max);
  histogram_sort_size_tMPI_Aint(remote_ranks, send_offsets, size + 1, local_indices, remote_indices);
  assert (send_offsets[size] == send_offsets[size + 1] || remote_ranks[send_offsets[size]] == size);
  MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, comm);
  recv_offsets[0] = 0;
  for (i = 0; i < (size_t)size; ++i) {
    assert (recv_counts[i] >= 0);
    recv_offsets[i + 1] = recv_offsets[i] + recv_counts[i];
  }
  MPI_Aint* restrict recv_data = (MPI_Aint*)page_aligned_xmalloc(recv_offsets[size] * sizeof(MPI_Aint));
  MPI_Alltoallv(remote_indices, send_counts, send_offsets, MPI_AINT, recv_data, recv_counts, recv_offsets, MPI_AINT, comm);
  char* restrict reply_data = (char*)page_aligned_xmalloc(recv_offsets[size] * elt_size);
  for (i = 0; i < (size_t)recv_offsets[size]; ++i) {
    assert (recv_data[i] >= 0 && recv_data[i] < (MPI_Aint)input_count);
    memcpy(reply_data + i * elt_size, input + recv_data[i] * elt_size, elt_size);
  }
  free(recv_data);
  char* restrict recv_reply_data = (char*)page_aligned_xmalloc(send_offsets[size] * elt_size);
  MPI_Alltoallv(reply_data, recv_counts, recv_offsets, datatype, recv_reply_data, send_counts, send_offsets, datatype, comm);
  free(reply_data);
  for (i = 0; i < nrequests_max; ++i) {
    if (remote_ranks[i] >= 0 && remote_ranks[i] < size) {
      memcpy(output + local_indices[i] * elt_size, recv_reply_data + i * elt_size, elt_size);
    }
  }
  free(recv_reply_data);
  g->valid = 0;
}

struct scatter_constant {
  void* array;
  size_t array_count;
  size_t elt_size;
  void* constant;
  size_t nrequests_max;
  int valid;
  MPI_Comm comm;
  int* remote_ranks;
  MPI_Aint* remote_indices;
  int comm_size;
  int* send_counts;
  int* send_offsets;
  int* recv_counts;
  int* recv_offsets;
};

scatter_constant* init_scatter_constant(void* array, size_t array_count, size_t elt_size, void* constant, size_t nrequests_max, MPI_Datatype dt /* unused */) {
  (void)dt;
  scatter_constant* sc = (scatter_constant*)cache_aligned_xmalloc(sizeof(scatter_constant));
  sc->array = array;
  sc->array_count = array_count;
  sc->elt_size = elt_size;
  sc->constant = constant;
  sc->nrequests_max = nrequests_max;
  sc->valid = 0;
  MPI_Comm_dup(mpi.comm_2d, &sc->comm);
  sc->remote_ranks = (int*)cache_aligned_xmalloc(nrequests_max * sizeof(int));
  sc->remote_indices = (MPI_Aint*)page_aligned_xmalloc(nrequests_max * sizeof(MPI_Aint));
  MPI_Comm_size(sc->comm, &sc->comm_size);
  int size = sc->comm_size;
  sc->send_counts = (int*)cache_aligned_xmalloc((size + 1) * sizeof(int));
  sc->send_offsets = (int*)cache_aligned_xmalloc((size + 2) * sizeof(int));
  sc->recv_counts = (int*)cache_aligned_xmalloc(size * sizeof(int));
  sc->recv_offsets = (int*)cache_aligned_xmalloc((size + 1) * sizeof(int));
  return sc;
}

void destroy_scatter_constant(scatter_constant* sc) {
  assert (!sc->valid);
  free(sc->remote_ranks); sc->remote_ranks = NULL;
  free(sc->remote_indices); sc->remote_indices = NULL;
  MPI_Comm_free(&sc->comm);
  free(sc->send_counts); sc->send_counts = NULL;
  free(sc->send_offsets); sc->send_offsets = NULL;
  free(sc->recv_counts); sc->recv_counts = NULL;
  free(sc->recv_offsets); sc->recv_offsets = NULL;
  free(sc);
}

void begin_scatter_constant(scatter_constant* sc) {
  assert (!sc->valid);
  {size_t i, nr = sc->nrequests_max; for (i = 0; i < nr; ++i) sc->remote_ranks[i] = sc->comm_size;}
  sc->valid = 1;
}

void add_scatter_constant_request(scatter_constant* sc, int remote_rank, size_t remote_idx, size_t req_id) {
  assert (sc->valid);
  assert (remote_rank >= 0 && remote_rank < sc->comm_size);
  assert (req_id < sc->nrequests_max);
  assert (sc->remote_ranks[req_id] == sc->comm_size);
  sc->remote_ranks[req_id] = remote_rank;
  sc->remote_indices[req_id] = (MPI_Aint)remote_idx;
}

/* Adapted from histogram_sort_inplace in boost/graph/detail/histogram_sort.hpp
 * */
void histogram_sort_MPI_Aint
       (int* restrict keys,
        const int* restrict rowstart,
        int numkeys,
        MPI_Aint* restrict values1) {
  int* restrict insert_positions = (int*)cache_aligned_xmalloc(numkeys * sizeof(int));
  memcpy(insert_positions, rowstart, numkeys * sizeof(int));
  int i;
  for (i = 0; i < rowstart[numkeys]; ++i) {
    int key = keys[i];
    assert (key >= 0 && key < numkeys);
    while (!(i >= rowstart[key] && i < insert_positions[key])) {
      int target_pos = insert_positions[key]++;
      if (target_pos == i) continue;
      assert (target_pos < rowstart[key + 1]);
      {int t = keys[i]; key = keys[target_pos]; keys[i] = key; keys[target_pos] = t;}
      {MPI_Aint t = values1[i]; values1[i] = values1[target_pos]; values1[target_pos] = t;}
      assert (key >= 0 && key < numkeys);
    }
  }
  for (i = 1; i < rowstart[numkeys]; ++i) {
    assert (keys[i] >= keys[i - 1]);
  }
  free(insert_positions);
}

void end_scatter_constant(scatter_constant* sc) {
  assert (sc->valid);
  int size = sc->comm_size;
  int* restrict send_counts = sc->send_counts;
  int* restrict send_offsets = sc->send_offsets;
  int* restrict recv_counts = sc->recv_counts;
  int* restrict recv_offsets = sc->recv_offsets;
  int* restrict remote_ranks = sc->remote_ranks;
  MPI_Aint* restrict remote_indices = sc->remote_indices;
  char* restrict array = (char*)sc->array;
  const char* restrict constant = (const char*)sc->constant;
  size_t elt_size = sc->elt_size;
  MPI_Comm comm = sc->comm;
  size_t nrequests_max = sc->nrequests_max;
#ifndef NDEBUG
  size_t array_count = sc->array_count;
#endif
  memset(send_counts, 0, (size + 1) * sizeof(int));
  size_t i;
  for (i = 0; i < nrequests_max; ++i) {
    assert (remote_ranks[i] >= 0 && remote_ranks[i] < size + 1);
    ++send_counts[remote_ranks[i]];
  }
  send_offsets[0] = 0;
  for (i = 0; i < (size_t)size + 1; ++i) {
    send_offsets[i + 1] = send_offsets[i] + send_counts[i];
  }
  histogram_sort_MPI_Aint(remote_ranks, send_offsets, size + 1, remote_indices);
  MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, comm);
  recv_offsets[0] = 0;
  for (i = 0; i < (size_t)size; ++i) {
    recv_offsets[i + 1] = recv_offsets[i] + recv_counts[i];
  }
  MPI_Aint* restrict recv_data = (MPI_Aint*)page_aligned_xmalloc(recv_offsets[size] * sizeof(MPI_Aint));
  MPI_Alltoallv(remote_indices, send_counts, send_offsets, MPI_AINT, recv_data, recv_counts, recv_offsets, MPI_AINT, comm);
  for (i = 0; i < (size_t)recv_offsets[size]; ++i) {
    assert (recv_data[i] >= 0 && recv_data[i] < (MPI_Aint)array_count);
    memcpy(array + recv_data[i] * elt_size, constant, elt_size);
  }
  free(recv_data);
  sc->valid = 0;
}

struct scatter {
  void* array;
  size_t array_count;
  size_t elt_size;
  char* send_data;
  size_t nrequests_max;
  MPI_Datatype datatype;
  int valid;
  MPI_Comm comm;
  int* remote_ranks;
  MPI_Aint* remote_indices;
  int comm_size;
  int* send_counts;
  int* send_offsets;
  int* recv_counts;
  int* recv_offsets;
};

scatter* init_scatter(void* array, size_t array_count, size_t elt_size, size_t nrequests_max, MPI_Datatype dt) {
  scatter* sc = (scatter*)cache_aligned_xmalloc(sizeof(scatter));
  sc->array = array;
  sc->array_count = array_count;
  sc->elt_size = elt_size;
  sc->send_data = (char*)page_aligned_xmalloc(nrequests_max * elt_size);
  sc->nrequests_max = nrequests_max;
  sc->datatype = dt;
  sc->valid = 0;
  MPI_Comm_dup(mpi.comm_2d, &sc->comm);
  sc->remote_ranks = (int*)cache_aligned_xmalloc(nrequests_max * sizeof(int));
  sc->remote_indices = (MPI_Aint*)page_aligned_xmalloc(nrequests_max * sizeof(MPI_Aint));
  MPI_Comm_size(sc->comm, &sc->comm_size);
  int size = sc->comm_size;
  sc->send_counts = (int*)cache_aligned_xmalloc((size + 1) * sizeof(int));
  sc->send_offsets = (int*)cache_aligned_xmalloc((size + 2) * sizeof(int));
  sc->recv_counts = (int*)cache_aligned_xmalloc(size * sizeof(int));
  sc->recv_offsets = (int*)cache_aligned_xmalloc((size + 1) * sizeof(int));
  return sc;
}

void destroy_scatter(scatter* sc) {
  assert (!sc->valid);
  free(sc->send_data); sc->send_data = NULL;
  free(sc->remote_ranks); sc->remote_ranks = NULL;
  free(sc->remote_indices); sc->remote_indices = NULL;
  MPI_Comm_free(&sc->comm);
  free(sc->send_counts); sc->send_counts = NULL;
  free(sc->send_offsets); sc->send_offsets = NULL;
  free(sc->recv_counts); sc->recv_counts = NULL;
  free(sc->recv_offsets); sc->recv_offsets = NULL;
  free(sc);
}

void begin_scatter(scatter* sc) {
  assert (!sc->valid);
  {size_t i, nr = sc->nrequests_max; for (i = 0; i < nr; ++i) sc->remote_ranks[i] = sc->comm_size;}
  sc->valid = 1;
}

void add_scatter_request(scatter* sc, const char* local_data, int remote_rank, size_t remote_idx, size_t req_id) {
  assert (sc->valid);
  assert (req_id < sc->nrequests_max);
  assert (remote_rank >= 0 && remote_rank < sc->comm_size);
  memcpy(sc->send_data + req_id * sc->elt_size, local_data, sc->elt_size);
  assert (sc->remote_ranks[req_id] == sc->comm_size);
  sc->remote_ranks[req_id] = remote_rank;
  sc->remote_indices[req_id] = (MPI_Aint)remote_idx;
}

/* Adapted from histogram_sort_inplace in boost/graph/detail/histogram_sort.hpp
 * */
void histogram_sort_MPI_Aintcharblock
       (int* restrict keys,
        const int* restrict rowstart,
        int numkeys,
        MPI_Aint* restrict values1,
        char* restrict values2,
        size_t elt_size2) {
  int* restrict insert_positions = (int*)cache_aligned_xmalloc(numkeys * sizeof(int));
  memcpy(insert_positions, rowstart, numkeys * sizeof(int));
  int i;
  for (i = 0; i < rowstart[numkeys]; ++i) {
    int key = keys[i];
    assert (key >= 0 && key < numkeys);
    print_with_prefix("i = %d, key = %d", i, key);
    while (!(i >= rowstart[key] && i < insert_positions[key])) {
      int target_pos = insert_positions[key]++;
      print_with_prefix("target_pos = %d from limit %d", target_pos, rowstart[key + 1]);
      if (target_pos == i) continue;
      assert (target_pos < rowstart[key + 1]);
      {int t = keys[i]; key = keys[target_pos]; keys[i] = key; keys[target_pos] = t;}
      {MPI_Aint t = values1[i]; values1[i] = values1[target_pos]; values1[target_pos] = t;}
      {char t[elt_size2]; memcpy(t, values2 + i * elt_size2, elt_size2); memcpy(values2 + i * elt_size2, values2 + target_pos * elt_size2, elt_size2); memcpy(values2 + target_pos * elt_size2, t, elt_size2);}
      assert (key >= 0 && key < numkeys);
    }
    print_with_prefix("done");
  }
  for (i = 1; i < rowstart[numkeys]; ++i) {
    assert (keys[i] >= keys[i - 1]);
  }
  free(insert_positions);
}

void end_scatter(scatter* sc) {
  assert (sc->valid);
  int size = sc->comm_size;
  int* restrict send_counts = sc->send_counts;
  int* restrict send_offsets = sc->send_offsets;
  int* restrict recv_counts = sc->recv_counts;
  int* restrict recv_offsets = sc->recv_offsets;
  char* restrict send_data = sc->send_data;
  int* restrict remote_ranks = sc->remote_ranks;
  MPI_Aint* restrict remote_indices = sc->remote_indices;
  char* restrict array = (char*)sc->array;
  size_t elt_size = sc->elt_size;
  MPI_Comm comm = sc->comm;
  size_t nrequests_max = sc->nrequests_max;
#ifndef NDEBUG
  size_t array_count = sc->array_count;
#endif
  memset(send_counts, 0, (size + 1) * sizeof(int));
  size_t i;
  for (i = 0; i < nrequests_max; ++i) {
    assert (remote_ranks[i] >= 0 && remote_ranks[i] < size + 1);
    ++send_counts[remote_ranks[i]];
  }
  send_offsets[0] = 0;
  for (i = 0; i < (size_t)size + 1; ++i) {
    send_offsets[i + 1] = send_offsets[i] + send_counts[i];
  }
  histogram_sort_MPI_Aintcharblock(remote_ranks, send_offsets, size + 1, remote_indices, send_data, elt_size);
  MPI_Alltoall(send_counts, 1, MPI_INT, recv_counts, 1, MPI_INT, comm);
  recv_offsets[0] = 0;
  for (i = 0; i < (size_t)size; ++i) {
    recv_offsets[i + 1] = recv_offsets[i] + recv_counts[i];
  }
  MPI_Aint* restrict recv_indices = (MPI_Aint*)page_aligned_xmalloc(recv_offsets[size] * sizeof(MPI_Aint));
  char* restrict recv_data = (char*)cache_aligned_xmalloc(recv_offsets[size] * elt_size);
  MPI_Alltoallv(remote_indices, send_counts, send_offsets, MPI_AINT, recv_indices, recv_counts, recv_offsets, MPI_AINT, comm);
  MPI_Alltoallv(send_data, send_counts, send_offsets, sc->datatype, recv_data, recv_counts, recv_offsets, sc->datatype, comm);
  for (i = 0; i < (size_t)recv_offsets[size]; ++i) {
    assert (recv_indices[i] >= 0 && recv_indices[i] < (MPI_Aint)array_count);
    memcpy(array + recv_indices[i] * elt_size, recv_data + i * elt_size, elt_size);
  }
  free(recv_data);
  free(recv_indices);
  sc->valid = 0;
}

class BfsValidation {
	enum { MAX_OUTPUT = 10 };
public:
	BfsValidation(int64_t nglobalverts__, int64_t nlocalverts__, int64_t chunksize)
	: nglobalverts(nglobalverts__)
	, nlocalverts(nlocalverts__)
	, chunksize_(chunksize)
	{
		uint64_t maxlocalverts_ui = nlocalverts;
		MPI_Allreduce(MPI_IN_PLACE, &maxlocalverts_ui, 1, MPI_UINT64_T, MPI_MAX, mpi.comm_2d);
		maxlocalverts = maxlocalverts_ui;
	}

/* Returns true if result is valid.  Also, updates high 16 bits of each element
 * of pred to contain the BFS level number (or -1 if not visited) of each
 * vertex; this is based on the predecessor map if the user didn't provide it.
 * */
template <typename EdgeList>
bool validate(EdgeList* edge_list, const int64_t root, int64_t* const pred, int64_t* const edge_visit_count_ptr)
{
  assert (pred);
  *edge_visit_count_ptr = 0; /* Ensure it is a valid pointer */
  int64_t error_counts = 0;

  error_counts += check_value_ranges(nglobalverts, nlocalverts, pred);
  if (root < 0 || root >= nglobalverts) {
	print_with_prefix("Validation error: root vertex %" PRId64 " is invalid.", root);
  }
  MPI_Allreduce(MPI_IN_PLACE, &error_counts, 1, MPI_INT, MPI_SUM, mpi.comm_2d); // #1
  if (error_counts) return false; /* Fail */
  assert (pred);

  int root_owner = vertex_owner(root);
  int64_t root_local = vertex_local(root);
  int root_is_mine = (root_owner == mpi.rank_2d);

  //ptrdiff_t chunksize_ = std::min(HALF_chunksize_, chunksize_);

  assert (pred);

  /* Check that root is its own parent. */
  if (root_is_mine) {
	assert (root_local < nlocalverts);
	if (get_pred_from_pred_entry(pred[root_local]) != root) {
	  print_with_prefix("Validation error: parent of root vertex %" PRId64 " is %" PRId64 ", not the root itself.", root, get_pred_from_pred_entry(pred[root_local]));
	  ++error_counts;
	}
  }
  if (error_counts) return false; /* Fail */

  assert (pred);

  /* Check that nothing else is its own parent. */
  {
	int* restrict pred_owner = (int*)cache_aligned_xmalloc(std::min(chunksize_, nlocalverts) * sizeof(int));
	int64_t* restrict pred_local = (int64_t*)cache_aligned_xmalloc(std::min(chunksize_, nlocalverts) * sizeof(int64_t));
	for (int64_t ii = 0; ii < nlocalverts; ii += chunksize_) {
	  ptrdiff_t i_start = ii;
	  ptrdiff_t i_end = std::min(ii + chunksize_, nlocalverts);
	  assert (i_start >= 0 && i_start <= (ptrdiff_t)nlocalverts);
	  assert (i_end >= 0 && i_end <= (ptrdiff_t)nlocalverts);
#pragma omp parallel for
	  for (ptrdiff_t i = i_start; i < i_end; ++i) {
		int64_t v = get_pred_from_pred_entry(pred[i]);
		pred_owner[i - i_start] = vertex_owner(v);
		pred_local[i - i_start] = vertex_local(v);
	  }
#pragma omp parallel for
	  for (ptrdiff_t i = i_start; i < i_end; ++i) {
		if ((!root_is_mine || i != root_local) &&
			get_pred_from_pred_entry(pred[i]) != -1 &&
			pred_owner[i - i_start] == mpi.rank_2d &&
			pred_local[i - i_start] == i) {
			if(__sync_fetch_and_add(&error_counts, 1) < MAX_OUTPUT)
			  print_with_prefix("Validation error: parent of non-root vertex %" PRId64 " is itself.", i * mpi.size_2d + mpi.rank_2d);
		}
	  }
	}
	free(pred_owner);
	free(pred_local);
  }
  MPI_Allreduce(MPI_IN_PLACE, &error_counts, 1, MPI_INT, MPI_SUM, mpi.comm_2d); // #2
  if (error_counts) return false; /* Fail */

  assert (pred);

  if (true) { // TODO:
	  error_counts += check_bfs_depth_map_using_predecessors(root, pred);
  } else {
	/* Create a vertex depth map to use for later validation. */
	  error_counts += build_bfs_depth_map(root, pred);
  }
  MPI_Allreduce(MPI_IN_PLACE, &error_counts, 1, MPI_INT, MPI_SUM, mpi.comm_2d); // #3
  if (error_counts) return false; /* Fail */

  {
	/* Check that all edges connect vertices whose depths differ by at most
	 * one, and check that there is an edge from each vertex to its claimed
	 * predecessor.  Also, count visited edges (including duplicates and
	 * self-loops).  */
	ScatterContext scatter_r(mpi.comm_2dr);
	ScatterContext scatter_c(mpi.comm_2dc);
	unsigned char* restrict pred_valid = (unsigned char*)cache_aligned_xmalloc(nlocalverts * sizeof(unsigned char));
	memset(pred_valid, 0, nlocalverts * sizeof(unsigned char));
	int64_t edge_visit_count = 0;

	typedef typename EdgeList::edge_type EdgeType;
	int num_loops = edge_list->beginRead(false);

	for(int loop_count = 0; loop_count < num_loops && error_counts == 0; ++loop_count) {
		EdgeType* edge_data;
		const int bufsize = edge_list->read(&edge_data);
		assert (bufsize <= chunksize_);
		//begin_gather(pred_win);
		int* restrict local_indices_r = (int*)cache_aligned_xmalloc(chunksize_ * sizeof(int));
		int* restrict remote_indices_r = (int*)page_aligned_xmalloc(chunksize_ * sizeof(int));
		int* restrict local_indices_c = (int*)cache_aligned_xmalloc(chunksize_ * sizeof(int));
		int* restrict remote_indices_c = (int*)page_aligned_xmalloc(chunksize_ * sizeof(int));
#pragma omp parallel
		{
			int *count_r = scatter_r.get_counts();
			int *count_c = scatter_c.get_counts();

			// *** : schedule(static) is important.
#pragma omp for schedule(static) // ***
			for (int i = 0; i < bufsize; ++i) {
				int64_t v0 = edge_data[i].v0();
				int64_t v1 = edge_data[i].v1();
				(count_r[vertex_owner_c(v0)])++;
				assert (vertex_owner_r(v0) == mpi.rank_2dr);
				(count_c[vertex_owner_r(v1)])++;
				assert (vertex_owner_c(v1) == mpi.rank_2dc);
			} // #pragma omp for (there is implicit barrier on exit)
#if defined(__INTEL_COMPILER)
#pragma omp barrier
#endif

#pragma omp master
			{
				scatter_r.sum();
				scatter_c.sum();
				assert (scatter_r.get_send_count() == bufsize);
				assert (scatter_c.get_send_count() == bufsize);
			} // #pragma omp master
#pragma omp barrier
			;
			int* offsets_r = scatter_r.get_offsets();
			int* offsets_c = scatter_c.get_offsets();
#pragma omp for schedule(static) // ***
			for (int i = 0; i < bufsize; ++i) {
				int64_t v0 = edge_data[i].v0();
				int64_t v1 = edge_data[i].v1();
				int v0_pos = offsets_r[vertex_owner_c(v0)]++;
				local_indices_r[i] = v0_pos;
				remote_indices_r[v0_pos] = vertex_local(v0);
				int v1_pos = offsets_c[vertex_owner_r(v1)]++;
				local_indices_c[i] = v1_pos;
				remote_indices_c[v1_pos] = vertex_local(v1);
			  //add_gather_request(pred_win, i * 2 + 0, edge_owner[i * 2 + 0], edge_local[i * 2 + 0], i * 2 + 0);
			  //add_gather_request(pred_win, i * 2 + 1, edge_owner[i * 2 + 1], edge_local[i * 2 + 1], i * 2 + 1);
			}
		} // #pragma omp parallel
		int* restrict reply_indices_r = scatter_r.scatter(remote_indices_r);
		int recv_count_r = scatter_r.get_recv_count();
		int* restrict reply_indices_c = scatter_c.scatter(remote_indices_c);
		int recv_count_c = scatter_c.get_recv_count();
		free(remote_indices_r); remote_indices_r = NULL;
		free(remote_indices_c); remote_indices_c = NULL;
		int64_t* restrict reply_data_r = (int64_t*)page_aligned_xmalloc(recv_count_r * sizeof(int64_t));
		int64_t* restrict reply_data_c = (int64_t*)page_aligned_xmalloc(recv_count_c * sizeof(int64_t));

#pragma omp parallel for
		for (int i = 0; i < recv_count_r; ++i) {
			reply_data_r[i] = pred[reply_indices_r[i]];
		}
#pragma omp parallel for
		for (int i = 0; i < recv_count_c; ++i) {
			reply_data_c[i] = pred[reply_indices_c[i]];
		}

		scatter_r.free(reply_indices_r); reply_indices_r = NULL;
		scatter_c.free(reply_indices_c); reply_indices_c = NULL;
		assert (scatter_r.get_send_count() == bufsize);
		assert (scatter_c.get_send_count() == bufsize);
		int64_t* restrict recv_pred_r = scatter_r.gather(reply_data_r);
		int64_t* restrict recv_pred_c = scatter_c.gather(reply_data_c);
		free(reply_data_r); reply_data_r = NULL;
		free(reply_data_c); reply_data_c = NULL;

		int64_t* restrict edge_preds = (int64_t*)cache_aligned_xmalloc(2 * chunksize_ * sizeof(int64_t));

#pragma omp parallel for
		for (int i = 0; i < bufsize; ++i) {
			assert (local_indices_r[i] < bufsize);
			assert (local_indices_c[i] < bufsize);
			edge_preds[2*i+0] = recv_pred_r[local_indices_r[i]];
			edge_preds[2*i+1] = recv_pred_c[local_indices_c[i]];
		}
		//end_gather(pred_win);
		free(local_indices_r); local_indices_r = NULL;
		free(local_indices_c); local_indices_c = NULL;
		scatter_r.free(recv_pred_r); recv_pred_r = NULL;
		scatter_r.free(recv_pred_c); recv_pred_c = NULL;

		//begin_scatter_constant(pred_valid_win);
		MPI_Aint* restrict remote_valid_indices_r = (MPI_Aint*)page_aligned_xmalloc(chunksize_ * sizeof(MPI_Aint));
		MPI_Aint* restrict remote_valid_indices_c = (MPI_Aint*)page_aligned_xmalloc(chunksize_ * sizeof(MPI_Aint));

#pragma omp parallel
		{
			int *count_r = scatter_r.get_counts();
			int *count_c = scatter_c.get_counts();

			// *** : schedule(static) is important.
#pragma omp for schedule(static) reduction(+:edge_visit_count) // ***
			for (int i = 0; i < bufsize; ++i) {
			  int64_t src = edge_data[i].v0();
			  int64_t tgt = edge_data[i].v1();
			  uint16_t src_depth = get_depth_from_pred_entry(edge_preds[i * 2 + 0]);
			  uint16_t tgt_depth = get_depth_from_pred_entry(edge_preds[i * 2 + 1]);
			  if (src_depth != UINT16_MAX && tgt_depth == UINT16_MAX) {
				  if(__sync_fetch_and_add(&error_counts, 1) < MAX_OUTPUT)
					  print_with_prefix("Validation error: edge connects vertex %" PRId64 " in the BFS tree (depth %" PRIu16 ") to vertex %" PRId64 " outside the tree.", src, src_depth, tgt);
			  } else if (src_depth == UINT16_MAX && tgt_depth != UINT16_MAX) {
				  if(__sync_fetch_and_add(&error_counts, 1) < MAX_OUTPUT)
					  print_with_prefix("Validation error: edge connects vertex %" PRId64 " in the BFS tree (depth %" PRIu16 ") to vertex %" PRId64 " outside the tree.", tgt, tgt_depth, src);
			  } else if (src_depth - tgt_depth < -1 ||
						 src_depth - tgt_depth > 1) {
				  if(__sync_fetch_and_add(&error_counts, 1) < MAX_OUTPUT)
					  print_with_prefix("Validation error: depths of edge endpoints %" PRId64 " (depth %" PRIu16 ") and %" PRId64 " (depth %" PRIu16 ") are too far apart (abs. val. > 1).", src, src_depth, tgt, tgt_depth);
			  } else if (src_depth != UINT16_MAX) {
				++edge_visit_count;
			  }
			  if (get_pred_from_pred_entry(edge_preds[i * 2 + 0]) == tgt) {
				(count_r[vertex_owner_c(src)])++;
			  }
			  if (get_pred_from_pred_entry(edge_preds[i * 2 + 1]) == src) {
				(count_c[vertex_owner_r(tgt)])++;
			  }
			} // #pragma omp for (there is implicit barrier on exit)
#if defined(__INTEL_COMPILER)
#pragma omp barrier
#endif

#pragma omp master
			{
				scatter_r.sum();
				scatter_c.sum();
			} // #pragma omp master
#pragma omp barrier
			;
			int* offsets_r = scatter_r.get_offsets();
			int* offsets_c = scatter_c.get_offsets();
#pragma omp for schedule(static) // ***
			for (int i = 0; i < bufsize; ++i) {
			  int64_t src = edge_data[i].v0();
			  int64_t tgt = edge_data[i].v1();
			  if (get_pred_from_pred_entry(edge_preds[i * 2 + 0]) == tgt) {
				  remote_valid_indices_r[offsets_r[vertex_owner_c(src)]++] = vertex_local(src);
			  }
			  if (get_pred_from_pred_entry(edge_preds[i * 2 + 1]) == src) {
				  remote_valid_indices_c[offsets_c[vertex_owner_r(tgt)]++] = vertex_local(tgt);
			  }
			} // #pragma omp for schedule(static)
		} // #pragma omp parallel

		MPI_Aint* restrict recv_valid_indices_r = scatter_r.scatter(remote_valid_indices_r);
		recv_count_r = scatter_r.get_recv_count();
		MPI_Aint* restrict recv_valid_indices_c = scatter_c.scatter(remote_valid_indices_c);
		recv_count_c = scatter_c.get_recv_count();
		free(remote_valid_indices_r); remote_valid_indices_r = NULL;
		free(remote_valid_indices_c); remote_valid_indices_c = NULL;

#pragma omp parallel for
		for (int i = 0; i < recv_count_r; ++i) {
			pred_valid[recv_valid_indices_r[i]] = 1;
		}
#pragma omp parallel for
		for (int i = 0; i < recv_count_c; ++i) {
			pred_valid[recv_valid_indices_c[i]] = 1;
		}

		MPI_Free_mem(recv_valid_indices_r);
		MPI_Free_mem(recv_valid_indices_c);
		free(edge_preds); edge_preds = NULL;
		//end_scatter_constant(pred_valid_win);
	}
	edge_list->endRead();
	//destroy_gather(pred_win);

	//destroy_scatter_constant(pred_valid_win);
	ptrdiff_t i;
#pragma omp parallel for
	for (i = 0; i < (ptrdiff_t)nlocalverts; ++i) {
	  int64_t p = get_pred_from_pred_entry(pred[i]);
	  if (p == -1) continue;
	  int found_pred_edge = pred_valid[i];
	  if (root_owner == mpi.rank_2d && root_local == i) found_pred_edge = 1; /* Root vertex */
	  if (!found_pred_edge) {
		int64_t v = i * mpi.size_2d + mpi.rank_2d;
		if(__sync_fetch_and_add(&error_counts, 1) < MAX_OUTPUT)
			print_with_prefix("Validation error: no graph edge from vertex %" PRId64 " to its parent %" PRId64 ".", v, get_pred_from_pred_entry(pred[i]));
	  }
	}
	free(pred_valid);

	MPI_Allreduce(MPI_IN_PLACE, &edge_visit_count, 1, MPI_INT64_T, MPI_SUM, mpi.comm_2d);
	*edge_visit_count_ptr = edge_visit_count;
  }

  /* Collect the global validation result. */
  MPI_Allreduce(MPI_IN_PLACE, &error_counts, 1, MPI_INT, MPI_SUM, mpi.comm_2d); // #4
  return error_counts == 0;
}

private:

/* This code assumes signed shifts are arithmetic, which they are on
 * practically all modern systems but is not guaranteed by C. */

static inline int64_t get_pred_from_pred_entry(int64_t val) {
  return (val << 16) >> 16;
}

static inline uint16_t get_depth_from_pred_entry(int64_t val) {
  return (val >> 48) & 0xFFFF;
}

static inline void write_pred_entry_depth(int64_t* loc, uint16_t depth) {
  *loc = (*loc & INT64_C(0xFFFFFFFFFFFF)) | ((int64_t)(depth & 0xFFFF) << 48);
}

/* Returns true if all values are in range. */
int64_t check_value_ranges(const int64_t nglobalverts, const int64_t nlocalverts, const int64_t* const pred) {
  int64_t error_counts = 0;
    for (int64_t ii = 0; ii < nlocalverts; ii += chunksize_) {
      ptrdiff_t i_start = ii;
      ptrdiff_t i_end = std::min(ii + chunksize_, nlocalverts);
      assert (i_start >= 0 && i_start <= (ptrdiff_t)nlocalverts);
      assert (i_end >= 0 && i_end <= (ptrdiff_t)nlocalverts);
#pragma omp parallel for
      for (ptrdiff_t i = i_start; i < i_end; ++i) {
        int64_t p = get_pred_from_pred_entry(pred[i]);
        if (p < -1 || p >= nglobalverts) {
			if(__sync_fetch_and_add(&error_counts, 1) < MAX_OUTPUT)
        		print_with_prefix("Validation error: parent of vertex %" PRId64 " is out-of-range value %" PRId64 ".", i * mpi.size_2d + mpi.rank_2d, p);
        }
      }
    }
  return error_counts;
}

/* Use the predecessors in the given map to write the BFS levels to the high 16
 * bits of each element in pred; this also catches some problems in pred
 * itself.  Returns true if the predecessor map is valid. */
int64_t build_bfs_depth_map(const int64_t root, int64_t* const pred)
{
  (void)nglobalverts;
  int64_t error_counts = 0;
  int root_owner = vertex_owner(root);
  int64_t root_local = vertex_local(root);
  int root_is_mine = (root_owner == mpi.rank_2d);
  if (root_is_mine) assert (root_local < nlocalverts);

  {
    ptrdiff_t i;
#pragma omp parallel for
    for (i = 0; i < (ptrdiff_t)nlocalverts; ++i) write_pred_entry_depth(&pred[i], UINT16_MAX);
    if (root_is_mine) write_pred_entry_depth(&pred[root_local], 0);
  }
  int64_t* restrict pred_pred = (int64_t*)xMPI_Alloc_mem(std::min(chunksize_, nlocalverts) * sizeof(int64_t)); /* Predecessor info of predecessor vertex for each local vertex */
  gather* pred_win = init_gather((void*)pred, nlocalverts, sizeof(int64_t), pred_pred, std::min(chunksize_, nlocalverts), std::min(chunksize_, nlocalverts), MPI_INT64_T);
  int* restrict pred_owner = (int*)cache_aligned_xmalloc(std::min(chunksize_, nlocalverts) * sizeof(int));
  int64_t* restrict pred_local = (int64_t*)cache_aligned_xmalloc(std::min(chunksize_, nlocalverts) * sizeof(int64_t));
  int iter_number = 0;
  {
    /* Iteratively update depth[v] = min(depth[v], depth[pred[v]] + 1) [saturating at UINT16_MAX] until no changes. */
    while (1) {
      ++iter_number;
      int any_changes = 0;
      for (int64_t ii = 0; ii < maxlocalverts; ii += chunksize_) {
        ptrdiff_t i_start = std::min(ii, nlocalverts);
        ptrdiff_t i_end = std::min(ii + chunksize_, nlocalverts);
        begin_gather(pred_win);
        ptrdiff_t i;
        assert (i_start >= 0 && i_start <= (ptrdiff_t)nlocalverts);
        assert (i_end >= 0 && i_end <= (ptrdiff_t)nlocalverts);
#pragma omp parallel for
        for (i = i_start; i < i_end; ++i) {
          int64_t v = get_pred_from_pred_entry(pred[i]);
          pred_owner[i - i_start] = vertex_owner(v);
          pred_local[i - i_start] = vertex_local(v);
        }
#pragma omp parallel for
        for (i = i_start; i < i_end; ++i) {
          if (pred[i] != -1) {
            add_gather_request(pred_win, i - i_start, pred_owner[i - i_start], pred_local[i - i_start], i - i_start);
          } else {
            pred_pred[i - i_start] = -1;
          }
        }
        end_gather(pred_win);
#pragma omp parallel for reduction(||:any_changes)
        for (i = i_start; i < i_end; ++i) {
          if (mpi.rank_2d == root_owner && i == root_local) continue;
          if (get_depth_from_pred_entry(pred_pred[i - i_start]) != UINT16_MAX) {
            if (get_depth_from_pred_entry(pred[i]) != UINT16_MAX && get_depth_from_pred_entry(pred[i]) != get_depth_from_pred_entry(pred_pred[i - i_start]) + 1) {
				  if(__sync_fetch_and_add(&error_counts, 1) < MAX_OUTPUT)
            		print_with_prefix("Validation error: BFS predecessors do not form a tree; see vertices %" PRId64 " (depth %" PRIu16 ") and %" PRId64 " (depth %" PRIu16 ").", i * mpi.size_2d + mpi.rank_2d, get_depth_from_pred_entry(pred[i]), get_pred_from_pred_entry(pred[i]), get_depth_from_pred_entry(pred_pred[i - i_start]));
            } else if (get_depth_from_pred_entry(pred[i]) == get_depth_from_pred_entry(pred_pred[i - i_start]) + 1) {
              /* Nothing to do */
            } else {
              write_pred_entry_depth(&pred[i], get_depth_from_pred_entry(pred_pred[i - i_start]) + 1);
              any_changes = 1;
            }
          }
        }
      }
      MPI_Allreduce(MPI_IN_PLACE, &any_changes, 1, MPI_INT, MPI_LOR, mpi.comm_2d);
      if (!any_changes) break;
    }
  }
  destroy_gather(pred_win);
  MPI_Free_mem(pred_pred);
  free(pred_owner);
  free(pred_local);
  return error_counts;
}

/* Check the BFS levels in pred against the predecessors given there.  Returns
 * true if the maps are valid. */
int64_t check_bfs_depth_map_using_predecessors(const int64_t root, const int64_t* const pred) {
  assert (root >= 0 && root < nglobalverts);
  assert (nglobalverts >= 0);
  assert (pred);

  int64_t error_counts = 0;
  int root_owner = vertex_owner(root);
  int64_t root_local = vertex_local(root);
  int root_is_mine = (root_owner == mpi.rank_2d);
  if (root_is_mine) assert (root_local < nlocalverts);

  {
    if (root_is_mine && get_depth_from_pred_entry(pred[root_local]) != 0) {
      print_with_prefix("Validation error: depth of root vertex %" PRId64 " is %" PRIu16 ", not 0.", root, get_depth_from_pred_entry(pred[root_local]));
      ++error_counts;
    }
#pragma omp parallel for
    for (int64_t i = 0; i < nlocalverts; ++i) {
      if (get_pred_from_pred_entry(pred[i]) == -1 &&
          get_depth_from_pred_entry(pred[i]) != UINT16_MAX) {
		  if(__sync_fetch_and_add(&error_counts, 1) < MAX_OUTPUT)
    		  print_with_prefix("Validation error: depth of vertex %" PRId64 " with no predecessor is %" PRIu16 ", not UINT16_MAX.", i * mpi.size_2d + mpi.rank_2d, get_depth_from_pred_entry(pred[i]));
      } else if (get_pred_from_pred_entry(pred[i]) != -1 &&
                 get_depth_from_pred_entry(pred[i]) == UINT16_MAX) {
		  if(__sync_fetch_and_add(&error_counts, 1) < MAX_OUTPUT)
    		  print_with_prefix("Validation error: predecessor of claimed unreachable vertex %" PRId64 " is %" PRId64 ", not -1.", i * mpi.size_2d + mpi.rank_2d, get_pred_from_pred_entry(pred[i]));
      }
    }
  }
  int64_t* restrict pred_pred = (int64_t*)xMPI_Alloc_mem(std::min(chunksize_, nlocalverts) * sizeof(int64_t)); /* Predecessor info of predecessor vertex for each local vertex */
    gather* pred_win = init_gather((void*)pred, nlocalverts, sizeof(int64_t), pred_pred, std::min(chunksize_, nlocalverts), std::min(chunksize_, nlocalverts), MPI_INT64_T);
    int* restrict pred_owner = (int*)cache_aligned_xmalloc(std::min(chunksize_, nlocalverts) * sizeof(int));
    int64_t* restrict pred_local = (int64_t*)cache_aligned_xmalloc(std::min(chunksize_, nlocalverts) * sizeof(int64_t));
    for (int64_t ii = 0; ii < maxlocalverts; ii += chunksize_) {
      ptrdiff_t i_start = std::min(ii, nlocalverts);
      ptrdiff_t i_end = std::min(ii + chunksize_, nlocalverts);
      begin_gather(pred_win);
      assert (i_start >= 0 && i_start <= (ptrdiff_t)nlocalverts);
      assert (i_end >= 0 && i_end <= (ptrdiff_t)nlocalverts);
      assert (i_end >= i_start);
      assert (i_end - i_start >= 0 && i_end - i_start <= (ptrdiff_t)std::min(chunksize_, nlocalverts));
  #pragma omp parallel for
      for (ptrdiff_t i = i_start; i < i_end; ++i) {
        int64_t v = get_pred_from_pred_entry(pred[i]);
        pred_owner[i - i_start] = vertex_owner(v);
        pred_local[i - i_start] = vertex_local(v);
      }
  #pragma omp parallel for
      for (ptrdiff_t i = i_start; i < i_end; ++i) {
        if (pred[i] != -1) {
          add_gather_request(pred_win, i - i_start, pred_owner[i - i_start], pred_local[i - i_start], i - i_start);
        } else {
          pred_pred[i - i_start] = -1;
        }
      }
      end_gather(pred_win);
  #pragma omp parallel for
      for (ptrdiff_t i = i_start; i < i_end; ++i) {
        if (mpi.rank_2d == root_owner && i == root_local) continue;
        if (get_pred_from_pred_entry(pred[i]) == -1) continue; /* Already checked */
        if (get_depth_from_pred_entry(pred_pred[i - i_start]) == UINT16_MAX) {
			if(__sync_fetch_and_add(&error_counts, 1) < MAX_OUTPUT)
      		  print_with_prefix("Validation error: predecessor %" PRId64 " of vertex %" PRId64 " (depth %" PRIu16 ") is marked as unreachable.", get_pred_from_pred_entry(pred[i]), i * mpi.size_2d + mpi.rank_2d, get_depth_from_pred_entry(pred[i]));
        }
        if (get_depth_from_pred_entry(pred[i]) != get_depth_from_pred_entry(pred_pred[i - i_start]) + 1) {
			  if(__sync_fetch_and_add(&error_counts, 1) < MAX_OUTPUT)
			  print_with_prefix("Validation error: BFS predecessors do not form a tree; see vertices %" PRId64 " (depth %" PRIu16 ") and %" PRId64 " (depth %" PRIu16 ").", i * mpi.size_2d + mpi.rank_2d, get_depth_from_pred_entry(pred[i]), get_pred_from_pred_entry(pred[i]), get_depth_from_pred_entry(pred_pred[i - i_start]));
        }
      }
    }
    destroy_gather(pred_win);
    MPI_Free_mem(pred_pred);
    free(pred_owner);
    free(pred_local);
    return error_counts;
}

const int64_t nglobalverts;
const int64_t nlocalverts;
int64_t maxlocalverts;
int64_t chunksize_;

}; // class BfsValidation

/* Returns true if result is valid.  Also, updates high 16 bits of each element
 * of pred to contain the BFS level number (or -1 if not visited) of each
 * vertex; this is based on the predecessor map if the user didn't provide it.
 * */
template <typename EdgeList>
int validate_bfs_result(
	EdgeList* edge_list,
	const int64_t nglobalverts,
	const int64_t nlocalverts,
	const int64_t root,
	int64_t* const pred,
	int64_t* const edge_visit_count_ptr)
{
	BfsValidation validation(nglobalverts, nlocalverts, EdgeList::CHUNK_SIZE);
	return validation.validate(edge_list, root, pred, edge_visit_count_ptr);
}


#endif /* VALIDATE_HPP_ */
