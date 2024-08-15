#include <cuda_runtime.h>
#include <iostream>

// 第一次调用内核函数，将数据加载到L1缓存
__global__ void loadToL1(int *d_array, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        volatile int temp = __ldca(&d_array[idx]); // 加载数据到L1缓存
    }
}

// 第二次调用内核函数，测量从L1缓存读取的时间
__global__ void readFromL1(int *d_array, int size, int *output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        int start = clock(); // 获取开始时间
        volatile int temp = d_array[idx]; // 从L1缓存读取数据
        int end = clock(); // 获取结束时间
        output[idx] = end - start; // 计算并存储时间差
    }
}

int main() {
    int size = 1; // 使用一个线程读取一个数据
    int *h_array = new int[size];
    int *h_output = new int[size];

    // 初始化主机数据
    for (int i = 0; i < size; i++) {
        h_array[i] = i;
    }

    int *d_array;
    int *d_output;

    // 分配设备内存
    cudaMalloc((void**)&d_array, size * sizeof(int));
    cudaMalloc((void**)&d_output, size * sizeof(int));

    // 将数据从主机传输到设备
    cudaMemcpy(d_array, h_array, size * sizeof(int), cudaMemcpyHostToDevice);

    // 第一次调用内核函数，将数据加载到L1缓存
    loadToL1<<<1, size>>>(d_array, size);
    cudaDeviceSynchronize(); // 确保前一个核函数完成

    // 第二次调用内核函数，测量从L1缓存读取的时间
    readFromL1<<<1, size>>>(d_array, size, d_output);
    cudaDeviceSynchronize(); // 确保核函数完成

    // 将结果从设备传回主机
    cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);

    // 输出结果
    std::cout << "Time to read from L1 cache: " << h_output[0] << " clock cycles" << std::endl;

    // 释放内存
    cudaFree(d_array);
    cudaFree(d_output);
    delete[] h_array;
    delete[] h_output;

    return 0;
}

