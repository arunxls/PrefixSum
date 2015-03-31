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
#include <limits>

using namespace std;

int get_elapsed(struct timeval* start, struct timeval* end) {

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

class Prefix {
public:
    Prefix(int proc, long long int numints, long long int ntotal) {
        this->proc         = proc;
        this->numints      = numints;
        this->ntotal       = ntotal;
        this->s_proc       = pow(2,ceil(log(proc)/log(2)));
        
        /* Allocate shared memory, enough for each thread to have numints*/
        this->data         = (long long int *) malloc(sizeof(long long int) * numints * proc);
        /* Allocate shared memory for partial_sums */
        this->partial_sums = (long long int *) calloc(proc, sizeof(long long int) * proc);

        if(this->data == NULL || this->partial_sums == NULL) {
            cerr << "unable to malloc()\n";
            exit(1);
        }
    }

    ~Prefix() {
        free(data);
        free(partial_sums);
    }

    /*****************************************************
    * Generate the random ints in parallel              *
    *****************************************************/
    void generate_input(long long int mod) {
        #pragma omp parallel num_threads(proc)
        {
            /* get the current thread ID in the parallel region */
            int tid = omp_get_thread_num();
            srand(tid + time(NULL));    /* Seed rand functions */
            mod = std::numeric_limits<long long int>::max() / (ntotal + 1);

            for(long long int i = tid * numints; i < (tid +1) * numints; ++i) {
                data[i] = rand()%mod;
                // data[i] = i + 1;
            }
        }
    }

    /*****************************************************
    * Print the entries of *data                         *
    *****************************************************/
    void print(ostringstream &o) {
        o.str("");
        o.clear();
    
        long long int index = 0;
        for(long long int i = 0; i < (proc) * numints && index < ntotal; ++i) {
            o << " " << data[i];
            index++;
        }
    }

    /*****************************************************
    * Perform the Prefix-Sum Balanced Tree Algorithm     *
    *****************************************************/
    long long int calculate_prefix() {
        //Calculate n/p prefix sums
        #pragma omp parallel num_threads(proc)
        {
            int tid                = omp_get_thread_num();
            if(tid == 0) {
                gettimeofday(&start, &tzp);
            }
            long long int start_id = tid * numints;
            long long int end_id   = (tid + 1) * numints;

            for(long long int i = start_id + 1; i < end_id; ++i) {
                data[i] += data[i-1];
            }

            /* Write the partial result to share memory */
            partial_sums[tid] = data[end_id - 1];
        }
        // gettimeofday(&end, &tzp);
        // cout << "1 - " << get_elapsed(&start, &end) << "\n";

        if(proc == 1) {
            gettimeofday(&end, &tzp);
            return get_elapsed(&start, &end);
        }

        //Starting Sweep-up operation
        // gettimeofday(&start, &tzp);
        for(int h = 0; h < floor(log(s_proc)/log(2) + 0.5); h++) {
            #pragma omp parallel for num_threads(s_proc/(int) pow(2, h))
            for(int i = 0; i < (s_proc/(int) pow(2, h+1)); i++) {
                int a           = (((int) pow(2, h+1)) * (i + 1)) -1;
                int b           = a - (int) pow(2,h);
                partial_sums[a] = partial_sums[a] + partial_sums[b];
            }
        }
        // gettimeofday(&end, &tzp);
        // cout << "2 - " << get_elapsed(&start, &end) << "\n";

        //Starting Sweep-down operation
        long long int max = partial_sums[s_proc-1];
        partial_sums[s_proc-1] = 0;

        gettimeofday(&start, &tzp);
        for(int h = floor(log(s_proc)/log(2) + 0.5) - 1; h > -1; h--) {
            #pragma omp parallel for num_threads(s_proc/(int) pow(2, h))
            for(int i = 0; i < (s_proc/(int) pow(2, h+1)); i++) {
                int a           = (((int) pow(2, h+1)) * (i + 1)) -1;
                int b           = a - (int) pow(2,h);
                long long int temp        = partial_sums[a];

                partial_sums[a] = partial_sums[a] + partial_sums[b];
                partial_sums[b] = temp;
            }
        }
        // gettimeofday(&end, &tzp);
        // cout << "3 - " << get_elapsed(&start, &end) << "\n";

        //Now calculate the prefix sum for elements inside each n/p section
        // gettimeofday(&start, &tzp);
        gettimeofday(&end, &tzp);
        long long int total_time = get_elapsed(&start, &end);
        #pragma omp parallel num_threads(proc)
        {
            int tid                = omp_get_thread_num();
            if(tid == 0) {
                gettimeofday(&start, &tzp);
            }          
            long long int start_id = tid * numints;
            long long int end_id   = (tid + 1) * numints;

            for(long long int i = start_id; i < end_id; ++i) {
              data[i] += partial_sums[tid];
            }
        }
        gettimeofday(&end, &tzp);
        total_time += get_elapsed(&start, &end);
        return total_time;
    }

private:
    int proc;
    long long numints;
    int s_proc;
    long long int ntotal;

    long long int *data;
    long long int *partial_sums;

    struct timeval start, end;   /* gettimeofday stuff */
    struct timezone tzp;

};




/*==============================================================
 *  Main Program (Parallel Summation)
 *==============================================================*/
int main(int argc, char *argv[]) {
    long long int numints = 0;
    long long int ntotal  = 0;
    int numiterations     = 0;
    int numthreads        = 1;
    int nmult             = 0;
    
    long long total_sum   = 0;

    struct timeval start, end;   /* gettimeofday stuff */
    struct timezone tzp;

    if( argc < 5) {
        printf("Usage: %s [numthreads] [ntotal] [nmult] [numiterations]\n\n", argv[0]);
        exit(1);
    }

    numthreads    = atoi(argv[1]);
    ntotal        = atol(argv[2]);
    nmult         = atoi(argv[3]);
    numiterations = atoi(argv[4]);

    ntotal = ntotal * nmult;

    // numints  = (long long int)ceil(((long long) ntotal/(long long) numthreads));
    numints = 1 + ((ntotal - 1) / numthreads);

    assert(numthreads > 0);

    //Set the number of threads
    omp_set_num_threads(numthreads);

    // printf("\nExecuting %s: nthreads=%d, ntotal=%d, numiterations=%d\n",
    //         argv[0], omp_get_max_threads(), ntotal, numiterations);

    int total_time = 0;
    for(int i = 0; i < numiterations; i++) {
        ostringstream oss;
        Prefix* p = new Prefix(numthreads, numints, ntotal);
        gettimeofday(&start, &tzp);
        p->generate_input(10);

        gettimeofday(&end, &tzp);

        // cout << "\n==============BEGIN INPUT===============================\n";
        // p->print(oss);
        // cout << oss.str();
        // cout << "\n==============END INPUT=================================\n"; 

        // cout << "\nInput generation time = " << get_elapsed(&start, &end) << " (usec)\n";

        // gettimeofday(&start, &tzp);
        total_time += p->calculate_prefix();
        // gettimeofday(&end,&tzp);

        // cout << "\n==============BEGIN OUTPUT==============================\n";
        // p->print(oss);
        // cout << oss.str();
        // cout << "\n==============END OUTPUT================================\n"; 
        delete(p);
        // total_time += get_elapsed(&start, &end);
        // cout << "\nOutput generation time = " << get_elapsed(&start, &end) << " (usec)\n";
    }
    
    /*****************************************************
    * Output timing results                             *
    *****************************************************/

    // cout << "Average output generation time = " << (float) total_time/numiterations << " (usec)\n"; 
    cout << total_time << "\n";

    return(0);
}
