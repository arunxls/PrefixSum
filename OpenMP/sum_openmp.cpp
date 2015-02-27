/*
 *  sum_openmp.cpp - Demonstrates parallelism via random fill and sum routines
 *                   This program uses OpenMP.
 */

/*---------------------------------------------------------
 *  Parallel Summation 
 *
 *  1. Each thread generates numints random integers (in parallel OpenMP region)
 *  2. Each thread sums his numints random integers (in parallel OpenMP region)
 *  3  One thread sums the partial results.
 *
 *  NOTE: steps 2-3 are repeated as many times as requested (numiterations)
 *---------------------------------------------------------*/


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <omp.h>
#include <vector>
#include <algorithm>

using namespace std;

/*==============================================================
 * print_elapsed (prints timing statistics)
 *==============================================================*/
void print_elapsed(const char* desc, struct timeval* start, struct timeval* end, int niters) {

  struct timeval elapsed;
  /* calculate elapsed time */
  if(start->tv_usec > end->tv_usec) {

    end->tv_usec += 1000000;
    end->tv_sec--;
  }
  elapsed.tv_usec = end->tv_usec - start->tv_usec;
  elapsed.tv_sec  = end->tv_sec  - start->tv_sec;

  printf("\n %s total elapsed time = %ld (usec)",
    desc, (elapsed.tv_sec*1000000 + elapsed.tv_usec) / niters);
}

// Functor to sum the numbers
template<typename T>
struct sum_functor {

  // Constructor
  sum_functor() : m_sum(0) {
  }

  void operator() (int& num) {
    m_sum +=num;
  }

  T get_sum() const {
    return m_sum;
  }

  protected:

  T m_sum;
};

/*==============================================================
 *  Main Program (Parallel Summation)
 *==============================================================*/
int main(int argc, char *argv[]) {

  int numints = 0;
  int numiterations = 0;

  vector<int> data;
  vector<long> partial_sums;

  long total_sum = 0;

  struct timeval start, end;   /* gettimeofday stuff */
  struct timezone tzp;

  if( argc < 3) {
    printf("Usage: %s [numints] [numiterations]\n\n", argv[0]);
    exit(1);
  }

  numints       = atoi(argv[1]);
  numiterations = atoi(argv[2]);

  printf("\nExecuting %s: nthreads=%d, numints=%d, numiterations=%d\n",
            argv[0], omp_get_max_threads(), numints, numiterations);

  /* Allocate shared memory, enough for each thread to have numints*/
  data.reserve(numints * omp_get_max_threads());

  /* Allocate shared memory for partial_sums */
  partial_sums.resize(omp_get_max_threads());

  /*****************************************************
   * Generate the random ints in parallel              *
   *****************************************************/
  #pragma omp parallel shared(numints,data) 
  {
    int tid;

    /* get the current thread ID in the parallel region */
    tid = omp_get_thread_num();

    srand(tid + time(NULL));    /* Seed rand functions */

    vector<int>::iterator it_cur = data.begin();
    std::advance(it_cur, tid * numints);

    vector<int>::iterator it_end = it_cur;
    std::advance(it_end, numints);

    for(; it_cur != it_end ; ++it_cur) {

      *it_cur = rand();
    }
  }

  /*****************************************************
   * Generate the sum of the ints in parallel          *
   * NOTE: Repeated for numiterations                  *
   *****************************************************/
  gettimeofday(&start, &tzp);

  for(int iteration=0; iteration < numiterations; ++iteration) {

    #pragma omp parallel shared(numints,data,partial_sums,total_sum)
    {
      int tid;

      /* get the current thread ID in the parallel region */
      tid = omp_get_thread_num();

      /* Compute the local partial sum */
      long partial_sum = 0;

      vector<int>::iterator it_cur = data.begin();
      std::advance(it_cur, tid * numints);

      vector<int>::iterator it_end = it_cur;
      std::advance(it_end, numints);

      sum_functor<long> result = std::for_each(it_cur, it_end, sum_functor<long>());

      /* Write the partial result to share memory */
      partial_sums[tid] = result.get_sum();
    }

    /* Compute the sum of the partial sums */
    total_sum = 0;
    int max_threads = omp_get_max_threads();
    for(int i = 0; i < max_threads ; ++i) {

      total_sum += partial_sums[i];
    }
  }

  gettimeofday(&end,&tzp);

  /*****************************************************
   * Output timing results                             *
   *****************************************************/

  print_elapsed("Summation", &start, &end, numiterations);
  printf("\n Total sum = %6ld\n", total_sum);

  return(0);
}
