# For MPICH
#mpirun -v -n 1 -outfile-pattern l-P1T1S18 -genv OMP_NUM_THREADS 1 ./mpi/runnable 18
#mpirun -v -n 1 -outfile-pattern l-P1T2S18 -genv OMP_NUM_THREADS 2 ./mpi/runnable 18
#mpirun -v -n 1 -outfile-pattern l-P1T4S18 -genv OMP_NUM_THREADS 4 ./mpi/runnable 18
mpirun -v -n 1 -outfile-pattern l-P1T12S18 -genv OMP_NUM_THREADS 12 ./mpi/runnable 22
#mpirun -v -n 1 -outfile-pattern l-P1T24S18 -genv OMP_NUM_THREADS 24 ./mpi/runnable 18
