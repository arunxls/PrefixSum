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

all: 
	icpc -O3 -openmp -lrt -o prefix_sum.x prefix_sum.cpp

clean:
	rm prefix_sum.x prefix_openmp_*.o*> /dev/null 2>&1

launch:
	qsub prefix_sum_4_4.job
	qsub prefix_sum_8_4.job
	qsub prefix_sum_12_4.job
	qsub prefix_sum_1_8.job
	qsub prefix_sum_2_8.job
	qsub prefix_sum_4_8.job
	qsub prefix_sum_8_8.job
	qsub prefix_sum_12_8.job
	qsub prefix_sum_2_16.job
	qsub prefix_sum_4_16.job
	qsub prefix_sum_8_16.job
	qsub prefix_sum_12_16.job
	qsub prefix_sum_1_20.job
	qsub prefix_sum_2_20.job
	qsub prefix_sum_4_20.job
	qsub prefix_sum_8_20.job
	qsub prefix_sum_12_20.job
