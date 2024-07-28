#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <cstdint>

__global__ void thread1(int *d_array, int *result1) {
    // Initial read
    int initial_value;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(initial_value) : "l"(d_array));
    result1[0] = initial_value;

    // Write to address using PTX volatile write
    int value = 1;
    asm volatile("st.global.wb.u32 [%0], %1;" : : "l"(d_array), "r"(value) : "memory");

    // Discard the cache line
    asm volatile("discard.global.L2 [%0], 128;" ::"l"(d_array) : "memory");

    // Busy wait to ensure discard completes
    for (volatile int i = 0; i < 100000; ++i);

    // Read after discard
    int discarded_value;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(discarded_value) : "l"(d_array));
    result1[1] = discarded_value;
}

__global__ void thread2(int *d_array, int *result2) {
    // Read after discard
    int final_value;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(final_value) : "l"(d_array));
    result2[0] = final_value;
}

int main() {
    int *d_array;
    int *d_result1;
    int *d_result2;
    int h_result1[2];
    int h_result2;

    // Allocate memory and ensure alignment
    cudaMalloc((void**)&d_array, 128 * sizeof(int));
    cudaMalloc((void**)&d_result1, 2 * sizeof(int));
    cudaMalloc((void**)&d_result2, sizeof(int));

    uintptr_t ptr_value = (uintptr_t)d_array;
    if (ptr_value % 128 != 0) {
        ptr_value += 128 - (ptr_value % 128);
    }
    d_array = (int*)ptr_value;

    // Initialize the memory
    cudaMemset(d_array, 0, 128 * sizeof(int));

    // Launch thread1 to read, write and discard
    thread1<<<1, 1>>>(d_array, d_result1);
    cudaDeviceSynchronize();

    // Launch thread2 to read the value after discard
    thread2<<<1, 1>>>(d_array, d_result2);
    cudaDeviceSynchronize();

    // Copy the results back to the host
    cudaMemcpy(h_result1, d_result1, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_result2, d_result2, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Initial value read by thread1: " << h_result1[0] << std::endl;
    std::cout << "Value read by thread1 after discard: " << h_result1[1] << std::endl;
    std::cout << "Value read by thread2 after discard: " << h_result2 << std::endl;

    cudaFree(d_array);
    cudaFree(d_result1);
    cudaFree(d_result2);

    return 0;
}
