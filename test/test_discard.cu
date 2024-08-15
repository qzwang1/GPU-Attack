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
    uint64_t max = 1048576ULL; 
    uint32_t value_init;
    if(value == 1) value_init = (1<<32)-1;
    else value_init = 0;
    for(uint64_t x = 0; x < max; x++)
    {
        a[x] = value_init;
        //a[x] = 0;
    }
    __syncthreads();
    
}

__global__
void discard (uint32_t* a, float* b)
{
    float m = 15;
    float n = 15;


    uint32_t t1 = 0;
    uint32_t t2 = 0;
    b[0]=0;
    uint64_t thread_id = blockDim.x*blockIdx.x+threadIdx.x;
    uint64_t start_idx = thread_id;
    uint64_t index = 0;


    t1 = clock();

            index = start_idx;
            a[index] = 0x123456;
            a[index*10] = 0xdeadbeef;
            m = a[index]+a[index+1]; //seems like ldcg is not necessary
            n = n + m;
            __syncthreads();

            //asm volatile("discard.global.L2 [%0],128;"
            //:: "l"(&(a[index]))
            //    );
    t2 = clock();
    __syncthreads();
    b[0] = n;
    b[1] = t2-t1;
    b[2] = a[index];

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

    uint64_t max = 1048576ULL*4ULL;
    status = cudaMalloc((void**)&da, max);
    if(status != cudaSuccess)
        printf("ERROR!!!\n");
    //db is for recording the timing and data after discard
    status = cudaMallocManaged(&db, 3*PATT_LEN_MAX*sizeof(float));
    if(status != cudaSuccess)
        printf("ERROR!!!\n");

    printf("Done cudaMalloc, start initializing VRAM\n");
    //initialize the memory to all 1s 
    mem_init<<<1, 1>>>(da, db, 1);
    cudaDeviceSynchronize();
    printf("done initialize\n");

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

