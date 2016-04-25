/*
 * logreader.cc
 *
 *  Created on: 2012/10/27
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

#include "logfile.h"

#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#if GCC_VERSION < 40200
inline uint32_t __builtin_bswap32(uint32_t val) {
	return ((((val) & 0xff000000u) >> 24) |
			(((val) & 0x00ff0000u) >>  8) |
			(((val) & 0x0000ff00u) <<  8) |
			(((val) & 0x000000ffu) << 24));
}
inline uint64_t __builtin_bswap64(uint64_t val) {
	return ((((val) & 0xff00000000000000ul) >> 56) |
			(((val) & 0x00ff000000000000ul) >> 40) |
			(((val) & 0x0000ff0000000000ul) >> 24) |
			(((val) & 0x000000ff00000000ul) >> 8 ) |
			(((val) & 0x00000000ff000000ul) << 8 ) |
			(((val) & 0x0000000000ff0000ul) << 24) |
			(((val) & 0x000000000000ff00ul) << 40) |
			(((val) & 0x00000000000000fful) << 56));
}
#endif

void convert_endian(LogFileFormat* log) {
	union { double d; long l; } t;
#define BSWAP_INT(v) v = __builtin_bswap32(v)
#define BSWAP_LONG(v) v = __builtin_bswap64(v)
#define BSWAP_DOUBLE(v) \
	t.d = v;\
	BSWAP_LONG(t.l);\
	v = t.d

	BSWAP_INT(log->scale);
	BSWAP_INT(log->edge_factor);
	BSWAP_INT(log->mpi_size);
	BSWAP_INT(log->num_runs);
	BSWAP_DOUBLE(log->generation_time);
	BSWAP_DOUBLE(log->construction_time);
	BSWAP_DOUBLE(log->redistribution_time);
	for(int i = 0; i < 64; ++i) {
		BSWAP_DOUBLE(log->times[i].bfs_time);
		BSWAP_DOUBLE(log->times[i].validate_time);
		BSWAP_LONG(log->times[i].edge_counts);
	}

#undef BSWAP_INT
#undef BSWAP_LONG
#undef BSWAP_DOUBLE
}

void read_log_file(const char* logfilename)
{
	FILE* fp = fopen(logfilename, "rb");
	if(fp == NULL) {
		fprintf(stderr, "File not found.\n");
		return;
	}
	LogFileFormat log;
	fread(&log, sizeof(log), 1, fp);
	fclose(fp);

	if(log.num_runs < 0 || log.num_runs > 64) {
		convert_endian(&log);
	}

	double bfs_times[64], validate_times[64], edge_counts[64];
	for (int i = 0; i < log.num_runs; ++i) {
	//	fprintf(stdout, "Running BFS %d\n", i);
		fprintf(stdout, "Time for BFS %d is %f\n", i, log.times[i].bfs_time);
	//	fprintf(stdout, "Validating BFS %d\n", i);
		fprintf(stdout, "Validate time for BFS %d is %f\n", i, log.times[i].validate_time);
		fprintf(stdout, "TEPS for BFS %d is %g\n", i, log.times[i].edge_counts / log.times[i].bfs_time);

		bfs_times[i] = log.times[i].bfs_time;
		validate_times[i] = log.times[i].validate_time;
		edge_counts[i] = log.times[i].edge_counts;
	}
	fprintf(stdout, "=========== %sResult ============\n", log.num_runs == 64 ? "" : "Current ");
	fprintf(stdout, "SCALE:                          %d\n", log.scale);
	fprintf(stdout, "edgefactor:                     %d\n", log.edge_factor);
	fprintf(stdout, "graph_generation:               %.12g\n", log.generation_time);
	fprintf(stdout, "num_mpi_processes:              %.d\n", log.mpi_size);
	fprintf(stdout, "construction_time:              %.12g\n", log.construction_time);
	fprintf(stdout, "redistribution_time:            %.12g\n", log.redistribution_time);
	print_bfs_result(log.num_runs, bfs_times, validate_times, edge_counts, true);
}

int main(int argc, char** argv)
{
	if(argc <= 1) {
		fprintf(stderr, "Usage: %s logfile\n", argv[0]);
		return 1;
	}
	read_log_file(argv[1]);

	return 0;
}


