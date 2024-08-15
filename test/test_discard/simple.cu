#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

#define DATA_BLOCK_SIZE 128  // 128B
#define OFFSET 0             

__device__ void writeValue(char* ptr, int value) {

    asm volatile("st.weak.global.cg.b32 [%0], %1;" : : "l"(ptr), "r"(value) : "memory");
}

__device__
void discardCache (uint32_t* ptr)
{
    float m = 15;
    float n = 15;


    uint32_t t1 = 0;
    uint32_t t2 = 0;
    uint64_t thread_id = blockDim.x*blockIdx.x+threadIdx.x;
    uint64_t start_idx = thread_id;
    uint64_t index = 0;


    t1 = clock();

            index = start_idx;
            ptr[index] = 0x123456;
            ptr[index*10] = 0xdeadbeef;
            m = ptr[index]+ptr[index+1]; //seems like ldcg is not necessary
            n = n + m;
            __syncthreads();

           asm volatile("discard.global.L2 [%0],128;"
            :: "l"(&(ptr[index]))
                );
    t2 = clock();
    __syncthreads();

}

__device__ void performComputation(int* v_data) {

    for (int i = 1; i < 10; ++i) {
        v_data[i] = v_data[i - 1] * 2 + i;
    }
    v_data[0] = v_data[9]; 
}

__global__ void testKernel(uint32_t *data, int *v_data, int *output_value) {

    char* base_ptr = (char*)data;
    char* offset_ptr = base_ptr + OFFSET;


    //int value;
    //asm volatile("ld.global.L1::no_allocate.b32 %0, [%1];" : "=r"(value) : "l"(offset_ptr));
    //v_data[0] += value;


   // writeValue(offset_ptr, 0xdeadbeef);


    discardCache((uint32_t*)offset_ptr);
    //discardCache(offset_ptr-128);
    //discardCache(offset_ptr+128);



    //asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(value) : "l"(offset_ptr));
    *output_value = ((int*)offset_ptr)[0];
}

int main() {
    uint32_t *d_data;
    int *h_data = new int[1]; 
    int *vv_data = new int[10];
    int *v_data;
    int *d_output_value;
    int h_output_value;

    cudaMalloc(&v_data, sizeof(int) * 10); 
    cudaMalloc(&d_output_value, sizeof(int));
    uint64_t max = 1048576ULL*4ULL;
    cudaError_t err = cudaMalloc((void**)&d_data, max);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }


    assert(reinterpret_cast<uintptr_t>(d_data) % 128 == 0);

  
    cudaMemset(d_data, 0, DATA_BLOCK_SIZE);
    cudaMemset(v_data, 0, sizeof(int) * 10); 


    testKernel<<<1, 1>>>(d_data, v_data, d_output_value);
    cudaDeviceSynchronize();


    cudaMemcpy(h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(vv_data, v_data, sizeof(int) * 10, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_output_value, d_output_value, sizeof(int), cudaMemcpyDeviceToHost);


    std::cout << "Final value after discard: " << std::hex << h_output_value << std::endl;
    std::cout << "Computation result: " << std::hex << vv_data[0] << std::endl;


    cudaFree(d_data);
    cudaFree(v_data);
    cudaFree(d_output_value);
    delete[] h_data;
    delete[] vv_data;

    return 0;
}
