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
	mpiicpc -O3 prefix_sum.cpp -o prefix_sum.x

clean:
	rm prefix_sum.x prefix_mpi_*> /dev/null 2>&1

launch:
	# qsub prefix_sum_1_1.job
	# qsub prefix_sum_2_1.job
	# qsub prefix_sum_4_1.job
	# qsub prefix_sum_8_1.job
	qsub prefix_sum_12_1.job
	qsub prefix_sum_16_1.job
	qsub prefix_sum_32_1.job
	qsub prefix_sum_64_1.job
	qsub prefix_sum_128_1.job
	# qsub prefix_sum_1_2.job
	# qsub prefix_sum_2_2.job
	# qsub prefix_sum_4_2.job
	# qsub prefix_sum_8_2.job
	qsub prefix_sum_12_2.job
	qsub prefix_sum_16_2.job
	qsub prefix_sum_32_2.job
	qsub prefix_sum_64_2.job
	qsub prefix_sum_128_2.job
	# qsub prefix_sum_1_4.job
	# qsub prefix_sum_2_4.job
	# qsub prefix_sum_4_4.job
	# qsub prefix_sum_8_4.job
	qsub prefix_sum_12_4.job
	qsub prefix_sum_16_4.job
	qsub prefix_sum_32_4.job
	qsub prefix_sum_64_4.job
	qsub prefix_sum_128_4.job
	# qsub prefix_sum_1_8.job
	# qsub prefix_sum_2_8.job
	# qsub prefix_sum_4_8.job
	# qsub prefix_sum_8_8.job
	qsub prefix_sum_12_8.job
	qsub prefix_sum_16_8.job
	qsub prefix_sum_32_8.job
	qsub prefix_sum_64_8.job
	qsub prefix_sum_128_8.job
	# qsub prefix_sum_1_16.job
	# qsub prefix_sum_2_16.job
	# qsub prefix_sum_4_16.job
	# qsub prefix_sum_8_16.job
	qsub prefix_sum_12_16.job
	qsub prefix_sum_16_16.job
	qsub prefix_sum_32_16.job
	qsub prefix_sum_64_16.job
	qsub prefix_sum_128_16.job
