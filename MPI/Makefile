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
	mpiicpc prefix_sum.cpp -o prefix_sum.x

clean:
	rm prefix_sum.x prefix_sum_mpi_*.o*> /dev/null 2>&1
