/*
 * main.cc
 *
 *  Created on: Dec 9, 2011
 *      Author: koji
 */

// C includes
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

// C++ includes
#include <string>

#include "parameters.h"
#include "utils_core.h"
#include "primitives.hpp"
#include "utils.hpp"
#include "../generator/graph_generator.hpp"
#include "graph_constructor.hpp"
#include "validate.hpp"
#include "benchmark_helper.hpp"
#include "bfs.hpp"
#include "bfs_cpu.hpp"
#if CUDA_ENABLED
#include "bfs_gpu.hpp"
#endif

void graph500_bfs(int SCALE, int edgefactor)
{
	using namespace PRM;
	SET_AFFINITY;

	double bfs_times[64], validate_times[64], edge_counts[64];
	LogFileFormat log = {0};
	int root_start = read_log_file(&log, SCALE, edgefactor, bfs_times, validate_times, edge_counts);
	if(mpi.isMaster() && root_start != 0)
		print_with_prefix("Resume from %d th run", root_start);

	EdgeListStorage<UnweightedPackedEdge, 8*1024*1024> edge_list(
//	EdgeListStorage<UnweightedPackedEdge, 512*1024> edge_list(
			(int64_t(1) << SCALE) * edgefactor / mpi.size_2d, getenv("TMPFILE"));

	BfsOnCPU::printInformation();

	if(mpi.isMaster()) print_with_prefix("Graph generation");
	double generation_time = MPI_Wtime();
	generate_graph_spec2010(&edge_list, SCALE, edgefactor);
	generation_time = MPI_Wtime() - generation_time;

	if(mpi.isMaster()) print_with_prefix("Graph construction");
	// Create BFS instance and the *COMMUNICATION THREAD*.
	BfsOnCPU* benchmark = new BfsOnCPU();
	double construction_time = MPI_Wtime();
	benchmark->construct(&edge_list);
	construction_time = MPI_Wtime() - construction_time;

	if(mpi.isMaster()) print_with_prefix("Redistributing edge list...");
	double redistribution_time = MPI_Wtime();
	redistribute_edge_2d(&edge_list);
	redistribution_time = MPI_Wtime() - redistribution_time;

	int64_t bfs_roots[NUM_BFS_ROOTS];
	int num_bfs_roots = NUM_BFS_ROOTS;
	find_roots(benchmark->graph_, bfs_roots, num_bfs_roots);
	const int64_t max_used_vertex = find_max_used_vertex(benchmark->graph_);
	const int64_t nlocalverts = benchmark->graph_.pred_size();

	int64_t *pred = static_cast<int64_t*>(
		cache_aligned_xmalloc(nlocalverts*sizeof(pred[0])));

#if INIT_PRED_ONCE	// Only Spec2010 needs this initialization
#pragma omp parallel for
	for(int64_t i = 0; i < nlocalverts; ++i) {
		pred[i] = -1;
	}
#endif

	bool result_ok = true;

	if(root_start == 0)
		init_log(SCALE, edgefactor, generation_time, construction_time, redistribution_time, &log);

	benchmark->prepare_bfs();
// narashi
		double time_left = PRE_EXEC_TIME;
        for(int c = root_start; time_left > 0.0; ++c) {
                if(mpi.isMaster())  print_with_prefix("========== Pre Running BFS %d ==========", c);
                MPI_Barrier(mpi.comm_2d);
                double bfs_time = MPI_Wtime();
                benchmark->run_bfs(bfs_roots[c % num_bfs_roots], pred);
                bfs_time = MPI_Wtime() - bfs_time;
                if(mpi.isMaster()) {
                        print_with_prefix("Time for BFS %d is %f", c, bfs_time);
                        time_left -= bfs_time;
                }
               MPI_Bcast(&time_left, 1, MPI_DOUBLE, 0, mpi.comm_2d);
        }
/////////////////////
	for(int i = root_start; i < num_bfs_roots; ++i) {
	//for(int i = 0; i < num_bfs_roots; ++i) {
		VERVOSE(print_max_memory_usage());

		if(mpi.isMaster())  print_with_prefix("========== Running BFS %d ==========", i);
#if ENABLE_FUJI_PROF
		fapp_start("bfs", i, 1);
#endif
		MPI_Barrier(mpi.comm_2d);
		PROF(profiling::g_pis.reset());
		bfs_times[i] = MPI_Wtime();
		benchmark->run_bfs(bfs_roots[i], pred);
		bfs_times[i] = MPI_Wtime() - bfs_times[i];
#if ENABLE_FUJI_PROF
		fapp_stop("bfs", i, 1);
#endif
		PROF(profiling::g_pis.printResult());
		if(mpi.isMaster()) {
			print_with_prefix("Time for BFS %d is %f", i, bfs_times[i]);
			print_with_prefix("Validating BFS %d", i);
		}

		benchmark->get_pred(pred);

		validate_times[i] = MPI_Wtime();
		int64_t edge_visit_count = 0;
#if VALIDATION_LEVEL >= 2
		result_ok = validate_bfs_result(
					&edge_list, max_used_vertex + 1, nlocalverts, bfs_roots[i], pred, &edge_visit_count);
#elif VALIDATION_LEVEL == 1
		if(i == 0) {
			result_ok = validate_bfs_result(
						&edge_list, max_used_vertex + 1, nlocalverts, bfs_roots[i], pred, &edge_visit_count);
			pf_nedge[SCALE] = edge_visit_count;
		}
		else {
			edge_visit_count = pf_nedge[SCALE];
		}
#else
		edge_visit_count = pf_nedge[SCALE];
#endif
		validate_times[i] = MPI_Wtime() - validate_times[i];
		edge_counts[i] = (double)edge_visit_count;

		if(mpi.isMaster()) {
			print_with_prefix("Validate time for BFS %d is %f", i, validate_times[i]);
			print_with_prefix("Number of traversed edges is %"PRId64"", edge_visit_count);
			print_with_prefix("TEPS for BFS %d is %g", i, edge_visit_count / bfs_times[i]);
		}

		if(result_ok == false) {
			break;
		}

		update_log_file(&log, bfs_times[i], validate_times[i], edge_visit_count);
	}
	benchmark->end_bfs();

	if(mpi.isMaster()) {
	  fprintf(stdout, "============= Result ==============\n");
	  fprintf(stdout, "SCALE:                          %d\n", SCALE);
	  fprintf(stdout, "edgefactor:                     %d\n", edgefactor);
	  fprintf(stdout, "NBFS:                           %d\n", num_bfs_roots);
	  fprintf(stdout, "graph_generation:               %g\n", generation_time);
	  fprintf(stdout, "num_mpi_processes:              %d\n", mpi.size_2d);
	  fprintf(stdout, "construction_time:              %g\n", construction_time);
	  fprintf(stdout, "redistribution_time:            %g\n", redistribution_time);
	  print_bfs_result(num_bfs_roots, bfs_times, validate_times, edge_counts, result_ok);
	}

	delete benchmark;

	free(pred);
}
#if 0
void test02(int SCALE, int edgefactor)
{
	EdgeListStorage<UnweightedPackedEdge, 8*1024*1024> edge_list(
			(INT64_C(1) << SCALE) * edgefactor / mpi.size, getenv("TMPFILE"));
	RmatGraphGenerator<UnweightedPackedEdge> graph_generator(
//	RandomGraphGenerator<UnweightedPackedEdge> graph_generator(
				SCALE, edgefactor, 2, 3, InitialEdgeType::NONE);
	Graph2DCSR<Pack40bit, uint32_t> graph;

	double generation_time = MPI_Wtime();
	generate_graph(&edge_list, &graph_generator);
	generation_time = MPI_Wtime() - generation_time;

	double construction_time = MPI_Wtime();
	construct_graph(&edge_list, true, graph);
	construction_time = MPI_Wtime() - construction_time;

	if(mpi.isMaster()) {
		print_with_prefix("TEST02");
		fprintf(stdout, "SCALE:                          %d\n", SCALE);
		fprintf(stdout, "edgefactor:                     %d\n", edgefactor);
		fprintf(stdout, "graph_generation:               %g\n", generation_time);
		fprintf(stdout, "num_mpi_processes:              %d\n", mpi.size);
		fprintf(stdout, "construction_time:              %g\n", construction_time);
	}
}
#endif

int main(int argc, char** argv)
{
	// Parse arguments.
	int SCALE = 16;
	int edgefactor = 16; // nedges / nvertices, i.e., 2*avg. degree
	if (argc >= 2) SCALE = atoi(argv[1]);
	if (argc >= 3) edgefactor = atoi(argv[2]);
	if (argc <= 1 || argc >= 4 || SCALE == 0 || edgefactor == 0) {
		fprintf(IMD_OUT, "Usage: %s SCALE edgefactor\n"
				"SCALE = log_2(# vertices) [integer, required]\n"
				"edgefactor = (# edges) / (# vertices) = .5 * (average vertex degree) [integer, defaults to 16]\n"
				"(Random number seed are in main.c)\n",
				argv[0]);
		return 0;
	}

	setup_globals(argc, argv, SCALE, edgefactor);

	graph500_bfs(SCALE, edgefactor);

	cleanup_globals();
	return 0;
}


