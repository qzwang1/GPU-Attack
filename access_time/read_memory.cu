#include <cuda_runtime.h>
#include <iostream>


__global__ void readMemory(int *d_array, int size, int *output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        int start = clock(); 
        volatile  int temp = d_array[idx]; 
        int end = clock();
        output[idx] = end - start + temp; 
    }
}

int main() {
    int size = 1; 
    int *h_array = new int[size];
    int *h_output = new int[size];

    
    for (int i = 0; i < size; i++) {
        h_array[i] = i;
    }

    int *d_array;
    int *d_output;

  
    cudaMalloc((void**)&d_array, size * sizeof(int));
    cudaMalloc((void**)&d_output, size * sizeof(int));

    
    cudaMemcpy(d_array, h_array, size * sizeof(int), cudaMemcpyHostToDevice);

    
    readMemory<<<1, 1>>>(d_array, size, d_output);

    
    cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);

   
    std::cout << "Time to read memory: " << h_output[0] - h_array[0] << " clock cycles" << std::endl;

    // 释放内存
    cudaFree(d_array);
    cudaFree(d_output);
    delete[] h_array;
    delete[] h_output;

    return 0;
}

