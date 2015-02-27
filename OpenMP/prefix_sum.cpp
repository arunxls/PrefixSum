#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include <sstream>
#include <iostream>

class Prefix {
public:

};

void print_elapsed(char* desc, struct timeval* start, struct timeval* end, int niters) {

    struct timeval elapsed;
    /* calculate elapsed time */
    if(start->tv_usec > end->tv_usec) {
        end->tv_usec += 1000000;
        end->tv_sec--;
    }

    elapsed.tv_usec = end->tv_usec - start->tv_usec;
    elapsed.tv_sec  = end->tv_sec  - start->tv_sec;

    printf("\n %s total elapsed time = %ld (usec)", desc, (elapsed.tv_sec*1000000 + elapsed.tv_usec) / niters);
}


/*==============================================================
 *  Main Program (Parallel Summation)
 *==============================================================*/
int main(int argc, char *argv[]) {

    int numints               = 0;
    int numiterations         = 0;
    int numthreads            = 1;

    int* data                 = NULL;
    long long* partial_sums   = NULL;

    long long total_sum       = 0;

    struct timeval start, end;   /* gettimeofday stuff */
    struct timezone tzp;

    if( argc < 3) {
        printf("Usage: %s [numthreads] [numints] [numiterations]\n\n", argv[0]);
        exit(1);
    }

    double param = 5.5;
    double result = log (param);
    // printf ("log(%f) = %f\n", param, result );

    numthreads    = atoi(argv[1]);
    numints       = atoi(argv[2]);
    numiterations = atoi(argv[3]);

    //Set the number of threads
    omp_set_num_threads(numthreads);

    printf("\nExecuting %s: nthreads=%d, numints=%d, numiterations=%d\n",
            argv[0], omp_get_max_threads(), numints, numiterations);

    /* Allocate shared memory, enough for each thread to have numints*/
    data = (int *) malloc(sizeof(int) * numints * omp_get_max_threads());

    /* Allocate shared memory for partial_sums */
    partial_sums = (long long*) malloc(sizeof(long long) * omp_get_max_threads());

    /*****************************************************
    * Generate the random ints in parallel              *
    *****************************************************/
    #pragma omp parallel shared(numints,data)
    {
        int tid;

        /* get the current thread ID in the parallel region */
        tid = omp_get_thread_num();

        srand(tid + time(NULL));    /* Seed rand functions */

        int i;
        for(i = tid * numints; i < (tid +1) * numints; ++i) {
            // data[i] = rand()%10;
            data[i] = i + 1;
        }
    }

    std::cout << "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";

    for(int i = 0; i < (numthreads) * numints; ++i) {
        std::ostringstream oss;
        oss << " " << data[i];
        std::cout << oss.str();
    }

    std::cout << "\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";


    /*****************************************************
    * Generate the sum of the ints in parallel          *
    * NOTE: Repeated for numiterations                  *
    *****************************************************/
    gettimeofday(&start, &tzp);

    int iteration;
    for(iteration=0; iteration < numiterations; ++iteration) {

        #pragma omp parallel shared(numints,data,partial_sums,total_sum)
        {
            int tid;

            /* get the current thread ID in the parallel region */
            tid = omp_get_thread_num();

            /* Compute the local partial sum */
            long long partial_sum = 0;

            int start_id = tid * numints;
            int end_id = (tid + 1) * numints;

            int i;
            for(i = start_id + 1; i < end_id; ++i) {
                data[i] += data[i-1];
            }

            /* Write the partial result to share memory */
            partial_sums[tid] = data[end_id - 1];
        }

        /* Compute the sum of the partial sums */
        total_sum = 0;
        int max_threads = omp_get_max_threads();

        int i;
        for(i = 0; i < max_threads ; ++i) {
            total_sum += partial_sums[i];
        }
    }

    // std::cout << "=======================================================\n";
    // for(int i = 0; i < numints * omp_get_max_threads(); ++i) {
    //   std::ostringstream oss;
    //   oss << " " << data[i];
    //   std::cout << oss.str();
    // }
    // std::cout << "\n=======================================================\n\n";

    // std::cout << "\nStarting Sweep-up" <<std::endl;

    for(int h = 0; h < floor(log(numthreads)/log(2) + 0.5); h++) {
        omp_set_num_threads(numthreads/(int) pow(2, h));

        #pragma omp parallel for shared(numints,data,partial_sums,total_sum)
        for(int i = 0; i < (numthreads/(int) pow(2, h+1)); i++) {
            int a = (((int) pow(2, h+1)) * (i + 1)) -1;
            int b = a - (int) pow(2,h);
            partial_sums[a] = partial_sums[a] + partial_sums[b];
        }

          // std::cout << "\n=======================================================\n";
          // for(int i = 0; i < numthreads; ++i) {
          //   std::ostringstream oss;
          //   oss << " " << partial_sums[i];
          //   std::cout << oss.str();
          // }
          // std::cout << "\n=======================================================\n\n";

    }

    // std::cout << "\nStarting Sweep-down" <<std::endl;
    int max = partial_sums[numthreads-1];
    partial_sums[numthreads-1] = 0;

    for(int h = floor(log(numthreads)/log(2) + 0.5) - 1; h > -1; h--) {
        omp_set_num_threads(numthreads/(int) pow(2, h));

        #pragma omp parallel for shared(numints,data,partial_sums,total_sum)
        for(int i = 0; i < (numthreads/(int) pow(2, h+1)); i++) {
            int a = (((int) pow(2, h+1)) * (i + 1)) -1;
            int b = a - (int) pow(2,h);
            int temp = partial_sums[a];
            partial_sums[a] = partial_sums[a] + partial_sums[b];
            partial_sums[b] = temp;
        }

      // std::cout << "\n=======================================================\n";
      // for(int i = 0; i < numthreads; ++i) {
      //   std::ostringstream oss;
      //   oss << " " << partial_sums[i];
      //   std::cout << oss.str();
      // }
      // std::cout << "\n=======================================================\n\n";

    }
    omp_set_num_threads(numthreads);
    #pragma omp parallel shared(numints,data,partial_sums,total_sum)
    {
        int tid;

        /* get the current thread ID in the parallel region */
        tid = omp_get_thread_num();

        /* Compute the local partial sum */
        long long partial_sum = 0;

        int start_id = tid * numints;
        int end_id = (tid + 1) * numints;

        long long diff = partial_sums[tid] - data[end_id-1];
        for(int i = start_id; i < end_id; ++i) {
          data[i] += diff;
        }
    }

    gettimeofday(&end,&tzp);

    /*****************************************************
    * Output timing results                             *
    *****************************************************/

    print_elapsed("Summation", &start, &end, numiterations);

    std::cout << "\n=======================================================\n";
    for(int i = 1; i < numthreads; ++i) {
        std::ostringstream oss;
        oss << " " << partial_sums[i];
        std::cout << oss.str();
    }
    std::cout << " " << max;
    std::cout << "\n=======================================================\n\n";

    free(data);
    free(partial_sums);

    return(0);
}
