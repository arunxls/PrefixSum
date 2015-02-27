#
# makefile for summation demo programs
#
# prompt> make
#
CC     = g++     # the c compiler to use
MPICC  = mpic++  # the MPI cc compiler
CFLAGS = -O3     # optimize code
DFLAGS =         # common defines

default:all

all: sum_openmp sum_mpi sum_mpi2

#
# OpenMP summation demo program
#

sum_openmp:sum_openmp.cpp
	$(CC) $(CFLAGS) $(DFLAGS) -fopenmp  -o $@  $@.cpp

#
# MPI summation demo programs
#

sum_mpi:sum_mpi.cpp
	$(MPICC) $(CFLAGS) $(DFLAGS) -o $@ $@.cpp

sum_mpi2:sum_mpi2.cpp
	$(MPICC) $(CFLAGS) $(DFLAGS) -o $@ $@.cpp

#
# clean up
#
clean:
	rm sum_openmp sum_mpi sum_mpi2 > /dev/null 2>&1
