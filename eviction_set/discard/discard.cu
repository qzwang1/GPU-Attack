#include "stdio.h"
#include "stdlib.h"
#include "fcntl.h"
#include "errno.h"
#include "inttypes.h"
#include "sys/mman.h"
#include "cuda_runtime.h"
#include <algorithm>
#include <iostream>

#define MEM_SIZE 5*1024*1024       
#define LINES_SIZE MEM_SIZE/128


//--------------------------------------------------------------------------------------------------------
/**
    传入gpu side的lines， 这个函数作用是在host端malloc一个lines然后赋值从0-LINES_SIZE，然后randomize，然后copy到gpu
    side的lines
 */


void build_lines(uint64_t* lines) {

    uint64_t* h_lines = (uint64_t*)malloc(LINES_SIZE*sizeof(uint64_t));
    for(int i = 0; i < LINES_SIZE; i++) {
        h_lines[i] = i;
    }
    std::random_shuffle(h_lines, h_lines+LINES_SIZE);
    cudaMemcpy(lines, h_lines, LINES_SIZE*sizeof(uint64_t), cudaMemcpyHostToDevice);
    free(h_lines);
}
//--------------------------------------------------------------------------------------------------------




//--------------------------------------------------------------------------------------------------------
__device__ 
bool check_eviction(uint64_t* target, uint64_t* S, int size, int exclude_index) {

    *target = 789123;
    __syncthreads();
    int volatile dummy = *target;
    for (int i = 0; i < size; i++) {
        //uint64_t* addr = (uint64_t*)S[i];
       // dummy += *addr;
        //*addr = 0;
        if(i != exclude_index)
        *(uint64_t*)S[i] = 0;
    }
    __syncthreads();
    asm volatile("discard.global.L2 [%0], 128;" :: "l"(target));
    __syncthreads();

    uint64_t value = *target;
    return value == 789123;
}

__global__
void build_conflictset(uint64_t* mem, uint64_t* lines, uint64_t* conflictset, int* conflictset_size, uint64_t* candidate, int* candidate_size) {
    *conflictset_size = 0;
    for(int i = 0; i < LINES_SIZE; i++) {
        uint64_t index = lines[i];
        uint64_t* addr = (uint64_t*)((char*)mem + index*128);
        if(!check_eviction(addr, conflictset, *conflictset_size, -1)) {
            conflictset[(*conflictset_size)++] = (uint64_t)addr;
        } else {
            candidate[(*candidate_size)++] = (uint64_t)addr;
        }
    }
}

__device__
bool is_in_conflictset(uint64_t addr, uint64_t*conflictset, int conflictset_size) {
    for(int i = 0; i < conflictset_size; i++) {
        if(addr == conflictset[i])
            return true;
    }
    return false;
}

//先找一个
__global__ 
void build_candidate(uint64_t*mem, uint64_t* candidate, uint64_t* conflictset, int* conflictset_size) {
    int idx = 0;
    for(int i = 0; i < LINES_SIZE; i++) {
        uint64_t addr = (uint64_t)((char*)mem + i * 128);
        if(!is_in_conflictset(addr, conflictset, *conflictset_size)) {
            candidate[idx++] = addr; 
        }
    }
}

__global__
void find_evictionsets(uint64_t* candidate, uint64_t* conflictset, int* conflictset_size, uint64_t* evictionset, int* evicitonset_size) {
    uint64_t* addr = (uint64_t*)(*candidate);
    if(check_eviction(addr, conflictset, *conflictset_size, -1)) {
        for(int i = 0; i < *conflictset_size; i++) {
            if(!check_eviction(addr, conflictset, *conflictset_size, i)) {
                evictionset[(*evicitonset_size)++] = conflictset[i];
                if((*evicitonset_size) > 16)
                    break;
            }
        }
    } 
}


//--------------------------------------------------------------------------------------------------------

__global__ 
void init_mem(uint64_t* mem) {
    // Using a single thread to iterate through memory
    for (int index = 0; index < MEM_SIZE / 128; ++index) {
        uint64_t* addr = (uint64_t*)((char*)mem + index * 128);
        *addr = index;
    }
}


//--------------------------------------------------------------------------------------------------------
/**
host: lines 0-81920
host: randomize lines
gpu:  build_conflict(lines, conflict_set, conflict_set_size)
host: build_candidate(lines, conflict_set, conflict_set_size)
host: candidate[0] -> candidate[size]:call find_eviction_set
gpu:   find_eviction_set(candidate[i], conflict_set, conflict_set_size, eviction_et, eviction_set_size)
host: 更新conflict_set,更新candidate，
 */
void build_evictionsets() {
    uint64_t* lines;
    cudaMalloc(&lines, LINES_SIZE * sizeof(uint64_t));
    build_lines(lines);


    uint64_t* mem;
    cudaMalloc(&mem, MEM_SIZE);
    init_mem<<<1,1>>>(mem);
    cudaDeviceSynchronize();

    uint64_t* conflictset;
    uint64_t* candidate;
    int* conflictset_size;
    int* candidate_size;
    cudaMalloc(&conflictset, LINES_SIZE * sizeof(uint64_t));
    cudaMalloc(&candidate, LINES_SIZE * sizeof(uint64_t));
    cudaMalloc(&conflictset_size, sizeof(int));
    cudaMalloc(&candidate_size, sizeof(int));
    build_conflictset<<<1,1>>>(mem, lines, conflictset, conflictset_size, candidate, candidate_size);
    cudaDeviceSynchronize();
    int h_size[2];
    cudaMemcpy(&h_size[1], conflictset_size, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "conflict set size:" << h_size[1] << std::endl;
    cudaMemcpy(&h_size[2], candidate_size, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "candidate set size:" << h_size[2] << std::endl;
    

    uint64_t* evictionset;
    int* evictionset_size;
    cudaMalloc(&evictionset, 16*sizeof(uint64_t));
    cudaMalloc(&evictionset_size, sizeof(int));
    int siz = 0;
    for(int i = 0; i < h_size[2]; i++) {
        find_evictionsets<<<1,1>>>((candidate+i), conflictset, conflictset_size, evictionset, evictionset_size);
        cudaDeviceSynchronize();

        cudaMemcpy(&siz, evictionset_size, sizeof(int), cudaMemcpyDeviceToHost);
        if(siz > 0)
        break;
    }



    uint64_t *h_set = (uint64_t*)malloc(1000*sizeof(uint64_t));
    cudaMemcpy(h_set, evictionset, 16*sizeof(uint64_t), cudaMemcpyDeviceToHost);

    std::cout<< siz<< std::endl;


    cudaFree(lines);
    cudaFree(mem);
    cudaFree(conflictset);
    cudaFree(conflictset_size);
}


int main() {
    build_evictionsets();
}