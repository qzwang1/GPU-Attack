#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define MEMORY_CHUNK_SIZE (10 * 1024 * 1024)  // 10MB
#define L2_CACHE_LINE_SIZE 128  // 128B

int main() {
    int *memoryChunk;
    int *d_memoryChunk;


    // 分配设备内存
    cudaError_t err = cudaMalloc(&d_memoryChunk, MEMORY_CHUNK_SIZE);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory: " << cudaGetErrorString(err) << std::endl;
        free(memoryChunk);
        return -1;
    }

    // 将设备内存地址存储在向量中
    std::vector<int *> addresses;
    for (size_t offset = 0; offset < MEMORY_CHUNK_SIZE; offset += L2_CACHE_LINE_SIZE) {
        addresses.push_back(reinterpret_cast<int *>(d_memoryChunk + offset / sizeof(int)));
    }


    // 释放设备内存
    cudaFree(d_memoryChunk);

    return 0;
}
