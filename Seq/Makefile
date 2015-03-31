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
	icpc -O3 -lrt -o prefix_sum.x prefix_sum.cpp

clean:
	rm prefix_sum.x prefix_seq_*.o*> /dev/null 2>&1

launch:
	qsub prefix_sum_1.job
	qsub prefix_sum_2.job
	qsub prefix_sum_4.job
	qsub prefix_sum_8.job
	qsub prefix_sum_16.job
	qsub prefix_sum_20.job
