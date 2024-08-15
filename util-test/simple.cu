#include "cuda_runtime.h"
#include "stdio.h"
#include "stdlib.h"
#include "inttypes.h"

#define CONFLICT_SET_SIZE (8 * 5120)
#define MEMORY_SIZE (10 * 1024 * 1024) // 10MB
#define EVICTION_THRESHOLD 800 // Threshold for eviction detection

__global__
void init_mem(uint8_t *mem) {
    int numBlocks = MEMORY_SIZE / 128;
    for(int idx = 0; idx < numBlocks; idx++) {
        // Each 128-byte block begins with a uint64_t pointing to the next block
        uint64_t *ptr = (uint64_t*)(mem + idx * 128);
        *ptr = idx * 128 + 128; // Point to the start of the next block
    }
}


__global__
void check_eviction(uint64_t basePtr, uint8_t* mem, uint64_t numOfElements, uint64_t* SharedTimeBuff) {
    unsigned int start_time, end_time;
    int volatile dummy = 0;
    uint64_t *otherptr;

    start_time = clock();
    uint64_t nxtIdx = __ldcg((uint64_t*)basePtr);
    dummy += nxtIdx;
    end_time = clock();
    __threadfence();
    SharedTimeBuff[0] = end_time - start_time;

    dummy += 1;

    
    /**
        when i == numOfElements - 1
        nxtIdx == content(mem[128*(numOfElements -1)]) so otherptr is mem + numOfElements - 1
     */

    for(int i = 0; i < numOfElements; i++) {
        otherptr = (uint64_t*)(mem + nxtIdx);
        nxtIdx = __ldcg(otherptr);
        dummy += nxtIdx;
        __threadfence();
    }

    dummy += 2;

    start_time = clock();
    nxtIdx = __ldcg((uint64_t*)basePtr);
    dummy += nxtIdx;
    end_time = clock();
    __threadfence();
    SharedTimeBuff[1] = end_time - start_time;

    dummy += 3;

    if(dummy == -1) {
        printf("This should never happen\n");
    }
}



__global__
void remove_addr(uint8_t*mem, uint64_t idx) {
    uint64_t *ptr = (uint64_t*)(mem + idx * 128);
    *(ptr+128) = 0;
    while(1) {
        int i = 2;
        if(*(ptr + i * 128) != 0) {
            *ptr = idx * 128 + i * 128;
            break;
        }
        i++;
    }
}


void find_evictionset(uint8_t*d_memory, uint64_t*d_sharedTimeBuff) {
    uint64_t basePtr = reinterpret_cast<uint64_t>(d_memory);
    int numOfElements = 0;
    uint64_t h_sharedTimeBuff[2];
    uint64_t threshold = 800;
    do{
        check_eviction<<<1,1>>>(basePtr, d_memory, numOfElements, d_sharedTimeBuff);
        numOfElements++;

        cudaMemcpy(h_sharedTimeBuff, d_sharedTimeBuff, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    }while((uint64_t)h_sharedTimeBuff[1] < threshold);
    remove_addr<<<1,1>>>(d_memory, numOfElements-3);
    //check_eviction<<<1,1>>>(basePtr, d_memory, numOfElements-1, d_sharedTimeBuff);
    cudaDeviceSynchronize();
    cudaMemcpy(h_sharedTimeBuff, d_sharedTimeBuff, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    printf("Initial Access Time: %" PRIu64 " cycles\n", h_sharedTimeBuff[0]);
    printf("Repeated Access Time: %" PRIu64 " cycles\n", h_sharedTimeBuff[1]);
    printf("idx:%d\n",numOfElements - 2);
}

int main() {
    setvbuf(stdout, NULL, _IOLBF, 0);

    cudaError_t status;

 
    uint8_t *d_memory;
    uint64_t *d_sharedTimeBuff;


    status = cudaMalloc((void**)&d_memory, MEMORY_SIZE);
    if(status != cudaSuccess) {
        printf("Allocate memory error\n");
        return -1;
    }

    cudaMalloc(&d_sharedTimeBuff, 2 * sizeof(uint64_t));




    init_mem<<<1,1>>>(d_memory);
    cudaDeviceSynchronize();

    for(int i = 0; i < 16; i++) {
        find_evictionset(d_memory, d_sharedTimeBuff);
    }


    // Output the times to observe eviction effects

    cudaFree(d_memory);
    cudaFree(d_sharedTimeBuff);

    return 0;
}
