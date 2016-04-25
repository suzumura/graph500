/*
 * low_level_func.h
 *
 *  Created on: 2012/10/17
 *      Author: ueno
 */

#ifndef LOW_LEVEL_FUNC_H_
#define LOW_LEVEL_FUNC_H_

#include "parameters.h"

struct LocalPacket {
	enum {
		TOP_DOWN_LENGTH = PRM::PACKET_LENGTH/sizeof(uint32_t),
		BOTTOM_UP_LENGTH = PRM::PACKET_LENGTH/sizeof(int64_t)
	};
	int length;
	int64_t src;
	union {
		uint32_t t[TOP_DOWN_LENGTH];
		int64_t b[BOTTOM_UP_LENGTH];
	} data;
};

void backward_isolated_edge(
	int half_bitmap_width,
	int phase_bmp_off,
	int phase_vertex_off,
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
);

#endif /* LOW_LEVEL_FUNC_H_ */
