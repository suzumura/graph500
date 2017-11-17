 Please cite the following paper if you use our code. 
 
  Koji Ueno, Toyotaro Suzumura, Naoya Maruyama, Katsuki Fujisawa, Satoshi Matsuoka:
Efficient Breadth-First Search on Massively Parallel and Distributed-Memory Machines. Data Science and Engineering 2(1): 22-35 (2017)

  Koji Ueno, Toyotaro Suzumura, Naoya Maruyama, Katsuki Fujisawa, Satoshi Matsuoka (2016-12-01). "Extreme scale breadth-first search on supercomputers". 2016 IEEE International Conference on Big Data (Big Data): 1040–1047. doi:10.1109/BigData.2016.7840705.

The license of the code is Apache License, Version 2.0. 

Here is how you build and run our code. All the detailed technical information is described in the above our paper. 

## Build

### Prerequisites

Libraries/Frameworks shown below are needed to be installed.

* libnuma
* C/C++ compiler supports OpenMP (e.g. gcc/g++)
* GNU make
* MPI library & runtime
    * OpenMPI
    * MPICH
    * MVAPICH

### Build instruction

1. Change directory to root of this repository.
2. Run this commands.

```sh
cd mpi
make cpu
```

3. Now you can run benchmarking with command like this.

```sh
# OpenMPI >= 1.8
mpirun -np 1 -bind-to none -output-filename ./log/lP1T8S16VR0BNONE -x OMP_NUM_THREADS=8 ./runnable 16
# OpenMPI <= 1.6.5
mpirun -np 1 -bind-to-none -output-filename ./log/lP1T8S16VR0BNONE -x OMP_NUM_THREADS=8 ./runnable 16
# MPICH / MVAPICH
mpirun -n 1 -outfile-pattern ./log/lP1T8S16VR0BNONE -genv OMP_NUM_THREADS 8 ./mpi/runnable 16
```


## Available options

### `Specifying number of threads and number of scales`
```sh
# OpenMPI >= 1.8
mpirun -np 1 -bind-to none -output-filename ./log/lP1T8S16VR0BNONE -x OMP_NUM_THREADS=<nthreads> ./runnable <nscale>
# OpenMPI <= 1.6.5
mpirun -np 1 -bind-to-none -output-filename ./log/lP1T8S16VR0BNONE -x OMP_NUM_THREADS=<nthreads> ./runnable <nscale>
# MPICH / MVAPICH
mpirun -n 1 -outfile-pattern ./log/lP1T8S16VR0BNONE -genv OMP_NUM_THREADS <nthreads> ./mpi/runnable <nscale>
```

* `nthreads` : number of threads
* `nscale` : number of scale

### make options
```sh
make [VERBOSE=<bool>] [VERTEX_REORDERING=<0|1|2>] [REAL_BENCHMARK=<bool>] cpu
```

* `VERBOSE` : toggle verbose output. true = enable, false = disenable.
* `VERTEX_REORDERING` : specify vertex reordering mode. 0 = do nothing (default), 1 = only reduce isolated vertices, 2 = sort by degree and reduce isolated vertices.
* `REAL_BENCHMARK` : change BFS iteration times. true = 64 times, false = 16 times (for testing).


## Benchmarking support script

In this repository, we provides support script written in python. This script includes these features:

* spawn rebuilding if nesessary
* generate logging file name automatically
* generate and spawn mpirun command (supports OpenMPI(>= 1.8, <= 1.6.5)/MPICH/MVAPICH)
* increasing-scale mode (iterate benchmarking with increasing scale unless `runnable` returns non-zero exit status)

For usage, please run this command:

```sh
# change directory to root of this repository and ...
./run-benchmark.py -h
# or ...
python run-benchmark.py -h
```

### Examples

```sh
# scale 20, thread x 8, real benchmarking, MPI runtime : OpenMPI (>= 1.8)
./run-benchmark.py -s 20 -t 8 --logfile-dest ./log201606XX -m OpenMPI
# Notes : rebuilding `runnable` binary is executed in this script, so manual rebuilding is not needed.

# scale 20, thread x 8, test mode (BFS x 16), vertex reordering mode : 1, MPI runtime : MPICH
./run-benchmark.py -s 20 -t 8 --logfile-dest ./log201606XXtest --test --vertex-reordering 0 -m MPICH
```

### Tips

* If you need to rebuild forcibly, remove file `prev_build_options`.




## Appendix : Tested environment

* OS : Ubuntu 14.04 LTS (64bit)
* Linux kernel version : 3.13.0-88-generic
* libnuma, mpich are installed via apt
* CPU : Intel Core i7-2640
* gcc version : 4.8.4 (Ubuntu 4.8.4-2ubuntu1~14.04.3)
* tested MPI runtime : MPICH 3.0.4
