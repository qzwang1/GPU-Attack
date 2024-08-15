#include "stdio.h"
#include "stdlib.h"
#include "fcntl.h"
#include "errno.h"
#include "inttypes.h"
#include "sys/mman.h"
#include "cuda_runtime.h"

#define CHUNK_SIZE   (10 * 1024 * 1024)
#define CONFLICT_SET_SIZE  25000
#define EVICTION_SET_SIZE 23

__device__ void wait_delay(uint64_t delay) {
    uint64_t start;
    uint64_t diff;
  
    start = clock64();
    do {
        diff = clock64() - start;
    } while (diff < delay);
}

__device__ bool check_eviction(uint64_t* target, uint64_t* S, int size) {
    uint64_t index = 0;
    target[index] = 123456;
    target[index * 10] = 789123;
    __syncthreads();
    int volatile dummy = target[index];
    for (int i = 0; i < size; i++) {
        uint64_t* addr = (uint64_t*)S[i];
        //dummy += *addr;
        *addr = 0;
    }
    __syncthreads();
    asm volatile("discard.global.L2 [%0], 128;" :: "l"(target));
    __syncthreads();
    //wait_delay(1000L);

    uint64_t value = *target;
    // if(value == 789123) {
    // asm volatile("discard.global.L2 [%0], 128;" :: "l"((uint64_t*)S[size-1]));
    // __syncthreads(); 
    // *target = 0;
    // return true;
    // }
    // return false;
    return value == 789123;
}

__device__
void discard_single(uint64_t*addr) {
    uint64_t index = 0;
    addr[index] = 123456;
    asm volatile("discard.global.L2 [%0], 128;"::"l"(addr));
    __syncthreads();
    wait_delay(1000000L);
}

__global__ void build_evictionsets(uint64_t* mem, uint64_t* conflict_set, int* conflict_set_size, uint64_t* eviction_set) {
    uint64_t* target = mem;
    int index = 1;
    int eviction_set_size = 0;
    while (*conflict_set_size < CONFLICT_SET_SIZE) {
        uint64_t* addr = (uint64_t*)((char*)mem + index * 128);
        conflict_set[*conflict_set_size] = (uint64_t)addr;
        (*conflict_set_size)++;
        index++;

        if (check_eviction(target, conflict_set, *conflict_set_size)) {
            eviction_set[eviction_set_size++] = (uint64_t)addr;
            //discard_single((uint64_t*)addr);
            break;
        }
    }

    if(check_eviction(target, conflict_set, *conflict_set_size)) {
        printf("check eviciton works\n");
    }
    
    // while (eviction_set_size < EVICTION_SET_SIZE && index < CHUNK_SIZE / 128) {
    //     uint64_t* addr = (uint64_t*)((char*)mem + index * 128);
    //     conflict_set[(*conflict_set_size)-1] = (uint64_t)addr;

    //     index++;
    //     if (check_eviction(target, conflict_set, *conflict_set_size)) {
    //         eviction_set[eviction_set_size++] = (uint64_t)addr;
    //         discard_single((uint64_t*)addr);
    //     }
    // }
}

__global__ void init_mem(uint64_t* mem) {
    // Using a single thread to iterate through memory
    for (int index = 0; index < CHUNK_SIZE / 128; ++index) {
        uint64_t* addr = (uint64_t*)((char*)mem + index * 128);
        *addr = index;
    }
}


__global__
void check_conflict(uint64_t* target, uint64_t* conflict_set, int* conflict_set_size, uint64_t*time) {
    uint64_t t1 = 0;
    uint64_t t2 = 0;

    uint64_t volatile temp = 0;
    t1 = clock();
    temp = *target;
    t2 = clock();
    __syncthreads();

    time[0] = t2 - t1;

    for(int i = 0; i < *conflict_set_size; i++) {
        temp += *((uint64_t*)conflict_set[i]);
    }
    __syncthreads();

    t1 = clock();
    temp += *target;
    t2 = clock();
    __syncthreads();

    time[1] = t2 - t1;
    time[2] = temp;
}



__global__ void check_is_eviction(uint64_t* eviction_set, uint64_t* time) {
    uint64_t t1 = 0;
    uint64_t t2 = 0;

    uint64_t volatile temp = 0;
    t1 = clock();
    temp = *((uint64_t*)eviction_set[0]);
    t2 = clock();
    __syncthreads();

    time[0] = t2 - t1;

    for(int i = 1; i < EVICTION_SET_SIZE; i++) {
        temp += *((uint64_t*)eviction_set[i]);
    }
    __syncthreads();

    t1 = clock();
    temp += *((uint64_t*)eviction_set[0]);
    t2 = clock();
    __syncthreads();

    time[1] = t2 - t1;
    time[2] = temp;
}


int main() {
    uint64_t *mem;
    uint64_t *conflict_set;
    uint64_t *eviction_set;
    int *conflict_set_size;
    
    // Allocate GPU memory
    cudaMalloc(&mem, CHUNK_SIZE);
    cudaMalloc(&conflict_set, CONFLICT_SET_SIZE * sizeof(uint64_t));
    cudaMalloc(&conflict_set_size, sizeof(int));
    cudaMalloc(&eviction_set, EVICTION_SET_SIZE * sizeof(uint64_t));
    
    // Initialize memory
    init_mem<<<1, 1>>>(mem);
    cudaDeviceSynchronize();
    
    // Launch kernel to build eviction sets
    build_evictionsets<<<1, 1>>>(mem, conflict_set, conflict_set_size, eviction_set);
    cudaDeviceSynchronize();
    
    // Retrieve and print the size of the conflict set
    int host_conflict_set_size;
    cudaMemcpy(&host_conflict_set_size, conflict_set_size, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Conflict set size: %d\n", host_conflict_set_size);

    // Retrieve and print the eviction set
    uint64_t host_eviction_set[EVICTION_SET_SIZE];
    cudaMemcpy(host_eviction_set, eviction_set, EVICTION_SET_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    printf("Eviction Set:\n");
    for (int i = 0; i < EVICTION_SET_SIZE; i++) {
        printf("Address %d: %p\n", i, (void*)host_eviction_set[i]);
    }

    // Allocate memory for timing results
    uint64_t *conflict_time;
    cudaMalloc(&conflict_time, 3 * sizeof(uint64_t));

    // Launch kernel to check if the eviction set causes eviction
    check_conflict<<<1,1>>>(&mem[0], conflict_set, conflict_set_size, conflict_time);
    cudaDeviceSynchronize();


    // Retrieve and print the timing results
    uint64_t host_tim[3];
    cudaMemcpy(host_tim, conflict_time, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    printf("Initial access time: %llu\n", host_tim[0]);
    printf("Eviction access time: %llu\n", host_tim[1]);
    printf("Accumulated temp value: %llu\n", host_tim[2]);



    // Allocate memory for timing results
    uint64_t *time;
    cudaMalloc(&time, 3 * sizeof(uint64_t));

    // Launch kernel to check if the eviction set causes eviction
    check_is_eviction<<<1, 1>>>(eviction_set, time);
    cudaDeviceSynchronize();

    // Retrieve and print the timing results
    uint64_t host_time[3];
    cudaMemcpy(host_time, time, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    printf("Initial access time: %llu\n", host_time[0]);
    printf("Eviction access time: %llu\n", host_time[1]);
    printf("Accumulated temp value: %llu\n", host_time[2]);

    // Free GPU memory
    cudaFree(mem);
    cudaFree(conflict_set);
    cudaFree(conflict_set_size);
    cudaFree(eviction_set);
    cudaFree(time);

    return 0;
}
