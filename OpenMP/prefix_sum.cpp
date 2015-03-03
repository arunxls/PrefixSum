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
#include <assert.h>

using namespace std;

class Prefix {
public:
    Prefix(int proc, int numints) {
        this->proc = proc;
        this->numints = numints;
        this->s_proc = pow(2,ceil(log(proc)/log(2)));

        /* Allocate shared memory, enough for each thread to have numints*/
        this->data  = (int *) malloc(sizeof(int) * numints * proc);

        /* Allocate shared memory for partial_sums */
        this->partial_sums = (long long*) calloc(proc, sizeof(long long) * proc);
    }

    ~Prefix() {
        free(data);
        free(partial_sums);
    }

    /*****************************************************
    * Generate the random ints in parallel              *
    *****************************************************/
    void generate_input(int mod) {
        #pragma omp parallel num_threads(proc)
        {
            /* get the current thread ID in the parallel region */
            int tid = omp_get_thread_num();
            srand(tid + time(NULL));    /* Seed rand functions */

            for(int i = tid * numints; i < (tid +1) * numints; ++i) {
                //data[i] = rand()%10000;
                data[i] = i + 1;
            }
        }
    }

    /*****************************************************
    * Print the entries of *data                         *
    *****************************************************/
    void print(ostringstream &o) {
        o.str("");
        o.clear();
    
        for(int i = 0; i < (proc) * numints; ++i) {
            o << " " << data[i];
        }
    }

    /*****************************************************
    * Perform the Prefix-Sum Balanced Tree Algorithm     *
    *****************************************************/
    void calculate_prefix() {
        //Calculate n/p prefix sums
        #pragma omp parallel num_threads(proc)
        {
            int tid               = omp_get_thread_num();
            long long partial_sum = 0;
            int start_id          = tid * numints;
            int end_id            = (tid + 1) * numints;

            for(int i = start_id + 1; i < end_id; ++i) {
                data[i] += data[i-1];
            }

            /* Write the partial result to share memory */
            partial_sums[tid] = data[end_id - 1];
        }

        //Starting Sweep-up operation
        for(int h = 0; h < floor(log(s_proc)/log(2) + 0.5); h++) {
            #pragma omp parallel for num_threads(s_proc/(int) pow(2, h))
            for(int i = 0; i < (s_proc/(int) pow(2, h+1)); i++) {
                int a           = (((int) pow(2, h+1)) * (i + 1)) -1;
                int b           = a - (int) pow(2,h);
                partial_sums[a] = partial_sums[a] + partial_sums[b];
            }
        }

        //Starting Sweep-down operation
        int max = partial_sums[s_proc-1];
        partial_sums[s_proc-1] = 0;

        for(int h = floor(log(s_proc)/log(2) + 0.5) - 1; h > -1; h--) {
            #pragma omp parallel for num_threads(s_proc/(int) pow(2, h))
            for(int i = 0; i < (s_proc/(int) pow(2, h+1)); i++) {
                int a           = (((int) pow(2, h+1)) * (i + 1)) -1;
                int b           = a - (int) pow(2,h);
                int temp        = partial_sums[a];

                partial_sums[a] = partial_sums[a] + partial_sums[b];
                partial_sums[b] = temp;
            }
        }

        //Allocate a temp memory to pop off the first value, which is a '0'
        long long* temp = (long long*) malloc(sizeof(long long) * proc);
        temp[proc-1] = max;

        #pragma omp parallel for num_threads(proc - 1)
        for(int i = 0; i < proc - 1; i++) {
            temp[i] = partial_sums[i + 1];
        }

        free(partial_sums);
        partial_sums = NULL;
        this->partial_sums = temp;

        //Now calculate the prefix sum for elements insde each n/p section
        #pragma omp parallel num_threads(proc)
        {
            int tid        = omp_get_thread_num();
            int start_id   = tid * numints;
            int end_id     = (tid + 1) * numints;
            long long diff = partial_sums[tid] - data[end_id-1];

            for(int i = start_id; i < end_id; ++i) {
              data[i] += diff;
            }
        }
    }

private:
    int proc;
    int numints;
    int s_proc;

    int *data;
    long long *partial_sums;

};

long print_elapsed(struct timeval* start, struct timeval* end) {

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

    assert(numthreads > 0);

    //Set the number of threads
    omp_set_num_threads(numthreads);

    printf("\nExecuting %s: nthreads=%d, numints=%d, numiterations=%d\n",
            argv[0], omp_get_max_threads(), numints, numiterations);

    

    long total_time = 0.0;
    for(int i = 0; i < numiterations; i++) {
        ostringstream oss;
        Prefix* p = new Prefix(numthreads, numints);
        gettimeofday(&start, &tzp);
        p->generate_input(10);
        gettimeofday(&end, &tzp);

        cout << "\n==============BEGIN INPUT===============================\n";
        p->print(oss);
        cout << oss.str();
        cout << "\n==============END INPUT=================================\n"; 

        printf("\nInput generation time = %ld (usec)\n", print_elapsed(&start, &end));

        gettimeofday(&start, &tzp);
        p->calculate_prefix();
        gettimeofday(&end,&tzp);

         cout << "\n==============BEGIN OUTPUT==============================\n";
         p->print(oss);
         cout << oss.str();
         cout << "\n==============END OUTPUT================================\n"; 
        delete(p);
        total_time += print_elapsed(&start, &end);
        printf("\nOutput generation time = %ld (usec)\n", print_elapsed(&start, &end));
    }
    
    /*****************************************************
    * Output timing results                             *
    *****************************************************/

    printf("\nSummation total elapsed time = %ld (usec)\n", total_time / numiterations);

    return(0);
}
