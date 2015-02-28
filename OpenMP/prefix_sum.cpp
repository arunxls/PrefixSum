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

using namespace std;

class Prefix {
public:
    Prefix(int proc, int numints) {
        this->proc = proc;
        this->numints = numints;

        /* Allocate shared memory, enough for each thread to have numints*/
        this->data = (int *) malloc(sizeof(int) * numints * proc);

        /* Allocate shared memory for partial_sums */
        this->partial_sums = (long long*) malloc(sizeof(long long) * proc);
    }

    ~Prefix() {
        free(data);
        free(partial_sums);
    }

    void generate_input(int mod) {
        #pragma omp parallel
        {
            /* get the current thread ID in the parallel region */
            int tid = omp_get_thread_num();
            srand(tid + time(NULL));    /* Seed rand functions */

            for(int i = tid * numints; i < (tid +1) * numints; ++i) {
                // data[i] = rand()%mod;
                data[i] = i + 1;
            }
        }
    }

    void print(ostringstream &o) {
        o.str("");
        o.clear();
    
        for(int i = 0; i < (proc) * numints; ++i) {
            o << " " << data[i];
        }
    }

/*****************************************************
    * Generate the sum of the ints in parallel          *
    * NOTE: Repeated for numiterations                  *
    *****************************************************/
    void calculate_prefix() {

        //Calculate n/p prefix sums
        #pragma omp parallel num_threads(proc)
        {
            /* get the current thread ID in the parallel region */
            int tid = omp_get_thread_num();

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

        // std::cout << "=======================================================\n";
        // for(int i = 0; i < numints * proc; ++i) {
        //   std::ostringstream oss;
        //   oss << " " << data[i];
        //   std::cout << oss.str();
        // }
        // std::cout << "\n=======================================================\n\n";

        // std::cout << "\nStarting Sweep-up" <<std::endl;

        for(int h = 0; h < floor(log(proc)/log(2) + 0.5); h++) {
            #pragma omp parallel for num_threads(proc/(int) pow(2, h))
            for(int i = 0; i < (proc/(int) pow(2, h+1)); i++) {
                int a = (((int) pow(2, h+1)) * (i + 1)) -1;
                int b = a - (int) pow(2,h);
                partial_sums[a] = partial_sums[a] + partial_sums[b];
            }

          // std::cout << "\n=======================================================\n";
          // for(int i = 0; i < proc; ++i) {
          //   std::ostringstream oss;
          //   oss << " " << partial_sums[i];
          //   std::cout << oss.str();
          // }
          // std::cout << "\n=======================================================\n\n";

        }

        // std::cout << "\nStarting Sweep-down" <<std::endl;
        int max = partial_sums[proc-1];
        partial_sums[proc-1] = 0;

        for(int h = floor(log(proc)/log(2) + 0.5) - 1; h > -1; h--) {
            #pragma omp parallel for num_threads(proc/(int) pow(2, h))
            for(int i = 0; i < (proc/(int) pow(2, h+1)); i++) {
                int a = (((int) pow(2, h+1)) * (i + 1)) -1;
                int b = a - (int) pow(2,h);
                int temp = partial_sums[a];
                partial_sums[a] = partial_sums[a] + partial_sums[b];
                partial_sums[b] = temp;
            }

          // std::cout << "\n=======================================================\n";
          // for(int i = 0; i < proc; ++i) {
          //   std::ostringstream oss;
          //   oss << " " << partial_sums[i];
          //   std::cout << oss.str();
          // }
          // std::cout << "\n=======================================================\n\n";

        }

        #pragma omp parallel num_threads(proc)
        {

            /* get the current thread ID in the parallel region */
            int tid = omp_get_thread_num();

            /* Compute the local partial sum */
            long long partial_sum = 0;

            int start_id = tid * numints;
            int end_id = (tid + 1) * numints;

            long long diff = partial_sums[tid] - data[end_id-1];
            for(int i = start_id; i < end_id; ++i) {
              data[i] += diff;
            }
        }

        /* Allocate shared memory, enough for each thread to have numints*/
        int* temp = (int *) malloc(sizeof(int) * numints * proc);
        temp[proc-1] = max;

        #pragma omp parallel num_threads(proc-1)
        {

            /* get the current thread ID in the parallel region */
            int tid = omp_get_thread_num();
            temp[tid] = data[tid + 1];
        }

        free(data);
        this->data = temp;
    }

private:
    int proc;
    int *data;
    long long *partial_sums;
    int numints;
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

    printf("\n%s total elapsed time = %ld (usec)\n", desc, (elapsed.tv_sec*1000000 + elapsed.tv_usec) / niters);
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

    numthreads    = atoi(argv[1]);
    numints       = atoi(argv[2]);
    numiterations = atoi(argv[3]);

    //Set the number of threads
    omp_set_num_threads(numthreads);

    printf("\nExecuting %s: nthreads=%d, numints=%d, numiterations=%d\n",
            argv[0], omp_get_max_threads(), numints, numiterations);

    

    /*****************************************************
    * Generate the random ints in parallel              *
    *****************************************************/

    Prefix* p = new Prefix(numthreads, numints);
    std::ostringstream oss;

    p->generate_input(10);
    
    std::cout << "\n=======================================================\n";
    p->print(oss);
    cout << oss.str();
    std::cout << "\n=======================================================\n";

    gettimeofday(&start, &tzp);
    p->calculate_prefix();
    gettimeofday(&end,&tzp);


    
    std::cout << "\n=======================================================\n";
    p->print(oss);
    cout << oss.str();
    std::cout << "\n=======================================================\n";
    

    /*****************************************************
    * Output timing results                             *
    *****************************************************/

    print_elapsed("Summation", &start, &end, numiterations);

    // std::cout << "\n=======================================================\n";
    // for(int i = 1; i < numthreads; ++i) {
    //     std::ostringstream oss;
    //     oss << " " << partial_sums[i];
    //     std::cout << oss.str();
    // }
    // std::cout << " " << max;
    // std::cout << "\n=======================================================\n\n";

    delete(p);

    return(0);
}
