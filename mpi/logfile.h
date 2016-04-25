/*
 * logfile.h
 *
 *  Created on: 2012/10/27
 *      Author: ueno
 */

#ifndef LOGFILE_H_
#define LOGFILE_H_


static int compare_doubles(const void* a, const void* b) {
  double aa = *(const double*)a;
  double bb = *(const double*)b;
  return (aa < bb) ? -1 : (aa == bb) ? 0 : 1;
}

enum {s_minimum, s_firstquartile, s_median, s_thirdquartile, s_maximum, s_mean, s_std, s_LAST};
static void get_statistics(const double x[], int n, double r[s_LAST]) {
  double temp;
  int i;
  /* Compute mean. */
  temp = 0;
  for (i = 0; i < n; ++i) temp += x[i];
  temp /= n;
  r[s_mean] = temp;
  /* Compute std. dev. */
  temp = 0;
  for (i = 0; i < n; ++i) temp += (x[i] - r[s_mean]) * (x[i] - r[s_mean]);
  temp /= n - 1;
  r[s_std] = sqrt(temp);
  /* Sort x. */
  double* xx = (double*)malloc(n * sizeof(double));
  memcpy(xx, x, n * sizeof(double));
  qsort(xx, n, sizeof(double), compare_doubles);
  /* Get order statistics. */
  r[s_minimum] = xx[0];
  r[s_firstquartile] = (xx[(n - 1) / 4] + xx[n / 4]) * .5;
  r[s_median] = (xx[(n - 1) / 2] + xx[n / 2]) * .5;
  r[s_thirdquartile] = (xx[n - 1 - (n - 1) / 4] + xx[n - 1 - n / 4]) * .5;
  r[s_maximum] = xx[n - 1];
  /* Clean up. */
  free(xx);
}

void print_bfs_result(
	int num_bfs_roots,
	double* bfs_times,
	double* validate_times,
	double* edge_counts,
	bool result_ok)
{
	if (!result_ok) {
	  fprintf(stdout, "No results printed for invalid run.\n");
	} else {
	  int i;
	  double stats[s_LAST];
	  get_statistics(bfs_times, num_bfs_roots, stats);
	  fprintf(stdout, "min_time:                       %.12g\n", stats[s_minimum]);
	  fprintf(stdout, "firstquartile_time:             %.12g\n", stats[s_firstquartile]);
	  fprintf(stdout, "median_time:                    %.12g\n", stats[s_median]);
	  fprintf(stdout, "thirdquartile_time:             %.12g\n", stats[s_thirdquartile]);
	  fprintf(stdout, "max_time:                       %.12g\n", stats[s_maximum]);
	  fprintf(stdout, "mean_time:                      %.12g\n", stats[s_mean]);
	  fprintf(stdout, "stddev_time:                    %.12g\n", stats[s_std]);
	  get_statistics(edge_counts, num_bfs_roots, stats);
	  fprintf(stdout, "min_nedge:                      %.11g\n", stats[s_minimum]);
	  fprintf(stdout, "firstquartile_nedge:            %.11g\n", stats[s_firstquartile]);
	  fprintf(stdout, "median_nedge:                   %.11g\n", stats[s_median]);
	  fprintf(stdout, "thirdquartile_nedge:            %.11g\n", stats[s_thirdquartile]);
	  fprintf(stdout, "max_nedge:                      %.11g\n", stats[s_maximum]);
	  fprintf(stdout, "mean_nedge:                     %.11g\n", stats[s_mean]);
	  fprintf(stdout, "stddev_nedge:                   %.11g\n", stats[s_std]);
	  double* secs_per_edge = (double*)malloc(num_bfs_roots * sizeof(double));
	  for (i = 0; i < num_bfs_roots; ++i) secs_per_edge[i] = bfs_times[i] / edge_counts[i];
	  get_statistics(secs_per_edge, num_bfs_roots, stats);
	  fprintf(stdout, "min_TEPS:                       %.12g\n", 1. / stats[s_maximum]);
	  fprintf(stdout, "firstquartile_TEPS:             %.12g\n", 1. / stats[s_thirdquartile]);
	  fprintf(stdout, "median_TEPS:                    %.12g\n", 1. / stats[s_median]);
	  fprintf(stdout, "thirdquartile_TEPS:             %.12g\n", 1. / stats[s_firstquartile]);
	  fprintf(stdout, "max_TEPS:                       %.12g\n", 1. / stats[s_minimum]);
	  fprintf(stdout, "harmonic_mean_TEPS:             %.12g\n", 1. / stats[s_mean]);
	  /* Formula from:
	   * Title: The Standard Errors of the Geometric and Harmonic Means and
	   *        Their Application to Index Numbers
	   * Author(s): Nilan Norris
	   * Source: The Annals of Mathematical Statistics, Vol. 11, No. 4 (Dec., 1940), pp. 445-448
	   * Publisher(s): Institute of Mathematical Statistics
	   * Stable URL: http://www.jstor.org/stable/2235723
	   * (same source as in specification). */
	  fprintf(stdout, "harmonic_stddev_TEPS:           %.12g\n", stats[s_std] / (stats[s_mean] * stats[s_mean] * sqrt(num_bfs_roots - 1)));
	  free(secs_per_edge); secs_per_edge = NULL;
	  get_statistics(validate_times, num_bfs_roots, stats);
	  fprintf(stdout, "min_validate:                   %.12g\n", stats[s_minimum]);
	  fprintf(stdout, "firstquartile_validate:         %.12g\n", stats[s_firstquartile]);
	  fprintf(stdout, "median_validate:                %.12g\n", stats[s_median]);
	  fprintf(stdout, "thirdquartile_validate:         %.12g\n", stats[s_thirdquartile]);
	  fprintf(stdout, "max_validate:                   %.12g\n", stats[s_maximum]);
	  fprintf(stdout, "mean_validate:                  %.12g\n", stats[s_mean]);
	  fprintf(stdout, "stddev_validate:                %.12g\n", stats[s_std]);
#if 0
	  for (i = 0; i < num_bfs_roots; ++i) {
		fprintf(stdout, "Run %3d:                        %g s, validation %g s\n", i + 1, bfs_times[i], validate_times[i]);
	  }
#endif
	}
}

struct LogFileTime {
	double bfs_time;
	double validate_time;
	int64_t edge_counts;
};

struct LogFileFormat {
	int scale;
	int edge_factor;
	int mpi_size;
	int num_runs;
	double generation_time;
	double construction_time;
	double redistribution_time;
	LogFileTime times[64];
};

#endif /* LOGFILE_H_ */
