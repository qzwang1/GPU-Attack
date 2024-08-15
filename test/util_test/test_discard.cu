#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

const int DATA_BLOCK_SIZE = 128;
const int NUM_THREADS_PER_BLOCK = 1024;
const int NUM_BLOCKS = 128;  // 增加线程块的数量以增加分配到不同 SM 的可能性

__global__ void readKernel(int *d_array, int *results, int idx) {
    if (threadIdx.x == 0) {
        results[idx] = d_array[0];  // 读取数据块的值
    }
}

__global__ void writeKernel(int *d_array, int value) {
    if (threadIdx.x == 0) {
        d_array[0] = value;  // 将指定值写入数据块
    }
}

__global__ void discardKernel(int *d_array) {
    if (threadIdx.x == 0) {
        // 执行 discard 操作
        asm volatile("discard.global.L2 [%0], 128;" ::"l"(d_array) : "memory");
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Device name: " << prop.name << std::endl;
    std::cout << "Total SMs: " << prop.multiProcessorCount << std::endl;

    int *d_array;
    int *d_results;
    int h_results[6];  // 存储每个步骤的结果

    // 分配设备内存
    cudaMalloc((void**)&d_array, DATA_BLOCK_SIZE * sizeof(int));
    cudaMalloc((void**)&d_results, 6 * sizeof(int));

    // 初始化数据块为 0
    cudaMemset(d_array, 0, DATA_BLOCK_SIZE * sizeof(int));

    // 创建两个流
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // 在 stream1 和 stream2 上执行初始读取操作
    readKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, stream1>>>(d_array, d_results, 0);
    readKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, stream2>>>(d_array, d_results, 1);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // 在 stream1 和 stream2 上执行写入操作
    writeKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, stream1>>>(d_array, 1);
    writeKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, stream2>>>(d_array, 2);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // 在 stream1 和 stream2 上执行写入后的读取操作
    readKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, stream1>>>(d_array, d_results, 2);
    readKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, stream2>>>(d_array, d_results, 3);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // 在 stream1 上执行 discard 操作
    discardKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, stream1>>>(d_array);
    cudaStreamSynchronize(stream1);

    // 在 stream1 和 stream2 上执行最终读取操作
    readKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, stream1>>>(d_array, d_results, 4);
    readKernel<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK, 0, stream2>>>(d_array, d_results, 5);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // 将结果拷贝回主机
    cudaMemcpy(h_results, d_results, 6 * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Initial value read by stream 1: " << h_results[0] << std::endl;
    std::cout << "Initial value read by stream 2: " << h_results[1] << std::endl;
    std::cout << "Value read by stream 1 after write: " << h_results[2] << std::endl;
    std::cout << "Value read by stream 2 after write: " << h_results[3] << std::endl;
    std::cout << "Final value read by stream 1 after discard: " << h_results[4] << std::endl;
    std::cout << "Final value read by stream 2 after discard: " << h_results[5] << std::endl;

    // 释放内存
    cudaFree(d_array);
    cudaFree(d_results);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return 0;
}
