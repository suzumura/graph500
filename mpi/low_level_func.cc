/*
 * low_level_func.c
 *
 *  Created on: 2012/10/17
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

#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include <sys/types.h>
#include <sys/time.h>
#include <omp.h>

#include <algorithm>

#include "utils_core.h"
#include "low_level_func.h"

#if LOW_LEVEL_FUNCTION

void backward_isolated_edge(
	int step_bitmap_width,
	int phase_bmp_off, // compact
	int phase_vertex_off, // separated
	int lgl, int L, int r_bits,
	BitmapType* __restrict__ phase_bitmap,
	const BitmapType* __restrict__ row_bitmap,
	const BitmapType* __restrict__ shared_visited,
	const TwodVertex* __restrict__ row_sums,
	const int64_t* __restrict__ isolated_edges,
	const int64_t* __restrict__ row_starts,
	const LocalVertex* __restrict__ orig_vertexes,
	const int64_t* __restrict__ edge_array,
	LocalPacket* buffer
) {
	int tid = omp_get_thread_num();
	int num_threads = omp_get_num_threads();
	int width_per_thread = (step_bitmap_width + num_threads - 1) / num_threads;
	int off_start = std::min(step_bitmap_width, width_per_thread * tid);
	int off_end = std::min(step_bitmap_width, off_start + width_per_thread);
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
				buffer->data.b[num_send+0] = src >> lgl;
				buffer->data.b[num_send+1] = tgt_orig;
				num_send += 2;
				// end this row
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
					buffer->data.b[num_send+0] = src >> lgl;
					buffer->data.b[num_send+1] = tgt_orig;
					num_send += 2;
					// end this row
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
}


#endif // #if LOW_LEVEL_FUNCTION

