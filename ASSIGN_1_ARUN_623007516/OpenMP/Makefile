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
	icpc -O2 -openmp -lrt -o prefix_sum.x prefix_sum.cpp

clean:
	rm prefix_sum.x psum_openmp_*.o*> /dev/null 2>&1

