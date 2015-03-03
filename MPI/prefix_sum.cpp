/*
 *  sum_mpi.c - Demonstrates parallelism via random fill and sum routines.
 *              This program uses MPI.
 */

/*---------------------------------------------------------
 *  Parallel Summation 
 *
 *  1. each processor generates numints random integers (in parallel)
 *  2. each processor calculates the prefix-sums for his numints random integers (in parallel)
 *  3. Each processor (other than processor 0) sends its result to processor 0
 *  4. Processor 0 prints out the input for each processor in order
 *  5. Processor 0 prints out the output for each processor in order
 *  6. Time for processor-wise sums
 *  6.1  All processors call MPI_Scan method, with their n/p section sum
 *  6.2  All processors calculate the difference between their calculated sum and that returned
         from MPI_Scan.
 *  6.3  This difference is then added to each of their elements, which returns the true
         prefix sum.
 *  6.4  Each processor (other than processor 0) sends its result to processor 0
 *  6.5  Processor 0 prints out the output for each processor in order
 *
 *  NOTE: steps 2-3 are repeated as many times as requested (numiterations)
 *---------------------------------------------------------*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <iostream>
#include <math.h>

using namespace std;

/*==============================================================
 * p_generate_random_ints (processor-wise generation of random ints)
 *==============================================================*/
void p_generate_random_ints(int* memory, int n) {
    int my_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    srand(my_id + time(NULL));                  /* Seed rand functions */
    for (int i = 0; i <  n; ++i) {
        memory[i] = rand() % 10;
        memory[i] = (my_id*n) + i + 1;
    }
}

/*==============================================================
 * p_summation (processor-wise summation of ints by calculating
 * prefix-sums for each n/p section)
 *==============================================================*/
int p_summation(int* memory, int n) {
    for (int i = 1; i < n; ++i) {
        memory[i] += memory[i-1];
    }
    
    return memory[n-1];
}

/*==============================================================
 * get_elapsed (get timing statistics)
 *==============================================================*/
long get_elapsed(struct timeval* start, struct timeval* end) {

    struct timeval elapsed;
    /* calculate elapsed time */
    if(start->tv_usec > end->tv_usec) {
        end->tv_usec += 1000000;
        end->tv_sec--;
    }

    elapsed.tv_usec = end->tv_usec - start->tv_usec;
    elapsed.tv_sec  = end->tv_sec  - start->tv_sec;

    return (elapsed.tv_sec*1000000 + elapsed.tv_usec);
}

/*==============================================================
 * print_data (prints memory array to stdout in order)
 *==============================================================*/

void print_data(char* desc, int* memory, int n, int comm_size, int ntotal) {
    int* buffer = (int *) malloc(sizeof(int) * n);
    MPI_Status status;
    int rank;
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        int index = 0;
        cout << "\n==============BEGIN "<< desc <<"==============================\n";
        for (int i = 0; i < n; i++) {
            std::cout << " " << memory[i];
            index++;
        }
        for (int i = 1; i < comm_size; i++) {
            MPI_Recv(buffer, n, MPI_INT, i, 123, MPI_COMM_WORLD, &status);
            for (int k = 0; k < n && index < ntotal; k++) {
                std::cout << " " << buffer[k];
                index++;
            }
        }
        cout << "\n==============END "<< desc <<"================================\n";
    } else {
        MPI_Send(memory, n, MPI_INT, 0, 123, MPI_COMM_WORLD);
    }

    free(buffer);
    buffer = NULL;
    MPI_Barrier(MPI_COMM_WORLD);
}

/*==============================================================
 *  Main Program (Parallel Summation)
 *==============================================================*/
int main ( int argc, char **argv) {
    int nprocs, numints, numiterations, ntotal; /* command line args */
    int my_id, iteration;
    int* mymemory;        /* pointer to processes memory              */
    int* buffer;

    struct timeval gen_start, gen_end; /* gettimeofday stuff */
    struct timeval start, end;         /* gettimeofday stuff */
    struct timezone tzp;

    MPI_Status status;              /* Status variable for MPI operations */

    /*---------------------------------------------------------
    * Initializing the MPI environment
    * "nprocs" copies of this program will be initiated by MPI.
    * All the variables will be private, only the program owner could see its own variables
    * If there must be a inter-procesor communication, the communication must be
    * done explicitly using MPI communication functions.
    *---------------------------------------------------------*/
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id); /* Getting the ID for this process */

    /*---------------------------------------------------------
    *  Read Command Line
    *  - check usage and parse args
    *---------------------------------------------------------*/
    if(argc < 3) {
        if(my_id == 0)
            printf("Usage: %s [numints] [numiterations]\n\n", argv[0]);
            MPI_Finalize();
            exit(1);
    }

    ntotal        = atoi(argv[1]);
    numiterations = atoi(argv[2]);

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* Get number of processors */

    if(my_id == 0)
    printf("\nExecuting %s: nprocs=%d, numints=%d, numiterations=%d\n",
            argv[0], nprocs, ntotal, numiterations);

    /*---------------------------------------------------------
    *  Initialization
    *  - allocate memory for work area structures and work area
    *---------------------------------------------------------*/

    numints  = (int)ceil(((float) ntotal/(float) nprocs));

    mymemory = (int *) calloc(numints, sizeof(int) * numints);
    buffer   = (int *) malloc(sizeof(int));

    if(mymemory == NULL || buffer == NULL) {
        printf("Processor %d - unable to malloc()\n", my_id);
        MPI_Finalize();
        exit(1);
    }

    /* repeat for numiterations times */
    for (iteration = 0; iteration < numiterations; iteration++) {
        
        /* get starting time */
        gettimeofday(&gen_start, &tzp);
        p_generate_random_ints(mymemory, numints);  /* random parallel fill */
        gettimeofday(&gen_end, &tzp);
        
        //Print out the input data
        print_data("INPUT", mymemory, numints, nprocs, ntotal);

        if(my_id == 0) {
            printf("\nInput generation time = %ld (usec)\n", get_elapsed(&gen_start, &gen_end));
        }

        /* Global barrier */
        MPI_Barrier(MPI_COMM_WORLD); 

        gettimeofday(&gen_start, &tzp);
        //Compute the prefix section for each n/p section.
        for (int i = 1; i < numints; ++i) {
            mymemory[i] += mymemory[i-1];
        }

        int sum = mymemory[numints - 1]; /* sum of each individual processor */
        int iters = pow(2,ceil(log(nprocs)/log(2)));

        for(int i = 0; i < iters - 1 ; i++) {
            //Send only to processors in range
            if(my_id + pow(2, i) < nprocs) {
                MPI_Send(&sum, 1, MPI_INT, my_id + pow(2, i), 111, MPI_COMM_WORLD);
            } 

            //Receive only from processors in range
            if(my_id - pow(2,i) >= 0) {
                MPI_Recv(buffer, 1, MPI_INT, my_id - pow(2, i), 111, MPI_COMM_WORLD, &status);
                sum = sum + *buffer;
            }
        }

        //Now that we have the prefix sums, find the diff and
        //add to the remaining elements to get a true prefix sum
        int diff = sum - mymemory[numints - 1];
        for(int i = 0; i < numints; i++) {
            mymemory[i] = mymemory[i] + diff;
        }
        gettimeofday(&gen_end, &tzp);
        print_data("OUTPUT", mymemory, numints, nprocs, ntotal);
        if(my_id == 0) {
            printf("\nOutput generation time = %ld (usec)\n", get_elapsed(&gen_start, &gen_end));
        }
    }

    //Print out the output data

    /* free memory */
    free(mymemory);
    MPI_Finalize();

    return 0;
}
