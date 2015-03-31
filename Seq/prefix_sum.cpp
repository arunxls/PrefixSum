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

void generate_input(long long int* data, long long int total) {
    srand(time(NULL));    /* Seed rand functions */
    long long int mod = std::numeric_limits<long long int>::max() / (total + 1);

    for(long long int i = 0; i < total; ++i) {
        data[i] = rand()%mod;
        // data[i] = i;
    }
}

/*==============================================================
 *  Main Program (Parallel Summation)
 *==============================================================*/
int main(int argc, char *argv[]) {

    long long int* data  = NULL;
    int total_time       = 0;
    long long int ntotal = 0;
    int nmult            = 0;

    struct timeval start, end;   /* gettimeofday stuff */
    struct timezone tzp;

    if( argc < 3) {
        printf("Usage: %s [ntotal] [nmult]\n\n", argv[0]);
        exit(1);
    }

    ntotal = atol(argv[1]);
    nmult  = atoi(argv[2]);
    ntotal = ntotal * nmult;

    // cout << sizeof(long long int) << "\n";
    // exit(1);

    long long int full_size = ntotal * sizeof(long long int);
    data = (long long int *) malloc(full_size);
    if(data == NULL) {
        printf("unable to malloc()");
        exit(1);
    }

    generate_input(data, ntotal);
    
    // for(int i = 0; i < ntotal; ++i) {
    //     cout << data[i] << "\n";
    // }

    // cout << "==============\n";

    gettimeofday(&start, &tzp);

    for(long long int i = 1; i < ntotal; ++i) {
        data[i] = data[i-1] + data[i];
    }

    gettimeofday(&end, &tzp);

    // for(int i = 0; i < ntotal; ++i) {
    //     cout << data[i] << "\n";
    // }
    
    total_time = get_elapsed(&start, &end);
    
    /*****************************************************
    * Output timing results                             *
    *****************************************************/

    // cout << "Average output generation time = " << total_time << " (usec)\n"; 
    cout << total_time <<"\n"; 

    return(0);
}
