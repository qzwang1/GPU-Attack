#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

__device__ void readMemory(int *d_array, int size, int *output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        int start = clock(); 
        
        // 使用 ld.global 指令从设备内存读取数据
        int temp;
        asm volatile(
            "ld.global.cg.b32 %0, [%1];"   // 读取数据，使用 cache 行为
            : "=r"(temp)                    // 输出寄存器
            : "l"(&d_array[idx])            // 输入地址
            : "memory"
        );

        int end = clock();
        output[idx] = end - start + temp; 
    }
}

__device__ void discardMemory(int *d_array) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    uintptr_t addr = (uintptr_t)(&d_array[idx]);
    asm volatile("discard.global.L2 [%0], 128;" : : "l"((void*)addr) : "memory");
    //for(int volatile i = 0; i < 10; i++);
}

__global__ void testKernel(int *d_array, int size, int *output) {
    __shared__ int local_output[1];
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Initialize output for this thread
    if (idx == 0) {
        local_output[0] = 0;
    }

    __syncthreads(); // Ensure initialization is complete

    // First read memory
    if (idx == 0) {
        readMemory(d_array, size, output);
    }

    __syncthreads(); // Ensure the read operation is completed

    // Second read memory
    if (idx == 0) {
        readMemory(d_array, size, output);
    }

    __syncthreads(); // Ensure the second read operation is completed

    // Perform discard operation
    discardMemory(d_array);

    __syncthreads(); // Ensure discard operation is completed

    // Third read memory
    if (idx == 0) {
        readMemory(d_array, size, output);
    }

    __syncthreads(); // Ensure the third read operation is completed

    // Write results to output
    if (idx == 0) {
        // Here you can perform additional operations if necessary
        local_output[0] = output[0]; // Example operation to ensure data is used
        output[0] = local_output[0];
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

    assert(((uintptr_t)d_array % 128) == 0);
    cudaMemcpy(d_array, h_array, size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    testKernel<<<1, 1>>>(d_array, size, d_output);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "First time to read memory: " << h_output[0] << " clock cycles" << std::endl;
    std::cout << "Second time to read memory: " << h_output[0] << " clock cycles" << std::endl;
    std::cout << "Third time to read memory: " << h_output[0] << " clock cycles" << std::endl;

    // Free memory
    cudaFree(d_array);
    cudaFree(d_output);
    delete[] h_array;
    delete[] h_output;

    return 0;
}
