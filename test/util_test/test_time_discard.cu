#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>

__global__ void readMemory(int *d_array, int size, int *output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        // Initial read
        int start1 = clock(); 
        volatile int temp1 = d_array[idx]; 
        int end1 = clock();
        output[0] = end1 - start1;

        // Second read
        int start2 = clock(); 
        volatile int temp2 = d_array[idx]; 
        int end2 = clock();
        output[1] = end2 - start2;

    //     // Discard the cache line
    //     uintptr_t addr = (uintptr_t)(&d_array[idx]);
    //     asm volatile("discard.global.L2 [%0], 128;" : : "l"(addr) : "memory");
    //     // Busy wait to ensure discard completes
    //    for (volatile int i = 0; i < 1; ++i);//就算只循环1次也使结果发生很大变化

    //     // Third read
    //     int start3 = clock(); 
    //     volatile int temp3 = d_array[idx]; 
    //     int end3 = clock();
    //     output[2] = end3 - start3;

        output[0] += temp1;
        output[1] += 0;
        output[2] += 0;
    }
}

int main() {
    int size = 1; 
    int *h_array = new int[size];
    int *h_output = new int[3]; // Array to hold three output times

    for (int i = 0; i < size; i++) {
        h_array[i] = i;
    }

    int *d_array;
    int *d_output;

    cudaMalloc((void**)&d_array, size * sizeof(int));
    cudaMalloc((void**)&d_output, 3 * sizeof(int)); // Allocate space for three times

    cudaMemcpy(d_array, h_array, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to read memory three times
    readMemory<<<1, 1>>>(d_array, size, d_output);

    cudaMemcpy(h_output, d_output, 3 * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Time to read memory first time: " << h_output[0] << " clock cycles" << std::endl;
    std::cout << "Time to read memory second time: " << h_output[1] << " clock cycles" << std::endl;
    std::cout << "Time to read memory after discard: " << h_output[2] << " clock cycles" << std::endl;

    cudaFree(d_array);
    cudaFree(d_output);
    delete[] h_array;
    delete[] h_output;

    return 0;
}
