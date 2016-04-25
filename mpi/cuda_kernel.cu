
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

#define _FERMI_ (__CUDA_ARCH__ >= 200)

#define WARP_IDX (blockIdx.x * blockDim.y + threadIdx.y)

#include "parameters.h"

#undef CUDA_CHECK_PRINT_RANK

__device__ MPI_INFO_ON_GPU mpig;

#include "gpu.hpp"
#include "bfs_kernel.hpp"









