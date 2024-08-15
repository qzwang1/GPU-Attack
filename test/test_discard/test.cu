#include "stdio.h"
#include "stdlib.h"
#include "fcntl.h"
#include "errno.h"
#include "inttypes.h"
#include "sys/mman.h"
#include "cuda_runtime.h"


#define PATT_LEN_MAX  128


extern int errno;

__global__
void mem_init (uint32_t *a, float *b, bool value)
{
    float m = 1.5f;
    float n = 1.5f;
    //uint64_t max = 1048576ULL; 
    uint64_t max = 32ULL;
    uint32_t value_init;
    if(value == 1) value_init = 1111;
    else value_init = 0;
    for(uint64_t x = 0; x < max; x++)
    {
        a[x] = value_init;
        //printf("%d\n",a[x]);
        //a[x] = 0;
    }
    __syncthreads();
    
}

__device__
void discardaddr (uint32_t* a, float* b)
{
    uint32_t t1 = 0;
    uint32_t t2 = 0;
    b[0]=0;
    uint64_t thread_id = blockDim.x*blockIdx.x+threadIdx.x;
    uint64_t start_idx = thread_id;
    uint64_t index = 0;

    t1 = clock();

            index = start_idx;
            a[index] = 12345;
            a[index*10] = 78912;
            __syncthreads();
            
            //asm volatile("discard.global.L2 [%0],128;"
            //:: "l"(&(a[index]))
              //  );
    t2 = clock();
    __syncthreads();
    b[0] = 0;
    b[1] = t2-t1;
    b[2] = a[index];

}

__global__
void discard(uint32_t*a, float*b) {
    discardaddr(a, b);
    printf("discard: %f\n", b[2]);
}

int main()
{

    
    setvbuf(stdout, NULL, _IOLBF, 0);
    int a = 1;
    printf("%d\n", a);


    cudaError_t status;

    uint64_t PATT_LEN;
    uint32_t *da;
    uint64_t *dc, *dd, *de;
    float* db;

    //uint64_t max = 1048576ULL*4ULL;
    uint64_t max = 32ULL;
    status = cudaMalloc((void**)&da, max*sizeof(uint32_t));
    if(status != cudaSuccess)
        printf("ERROR!!!\n");
    //db is for recording the timing and data after discard
    status = cudaMallocManaged(&db, 3*PATT_LEN_MAX*sizeof(float));
    if(status != cudaSuccess)
        printf("ERROR!!!\n");

    printf("Done cudaMalloc, start initializing VRAM\n");
    //initialize the memory to all 1s 
    //mem_init<<<1, 1>>>(da, db, 1);
    //cudaDeviceSynchronize();
    //printf("done initialize\n");

    status = cudaGetLastError();
    if(status != cudaSuccess)
        printf("iniii%s\n", cudaGetErrorString(status));
    
    discard<<<1, 1>>>(da, db);
    cudaDeviceSynchronize();
    status = cudaGetLastError();
    if(status != cudaSuccess)
        printf("discard %s\n", cudaGetErrorString(status));
    else
    {
        printf("%f\n", db[2]);
    }

    

    return 0;
}

