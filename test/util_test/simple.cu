#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

#define DATA_BLOCK_SIZE 128  // 128B
#define OFFSET 64            // 将数据写入数据块的中间部分

__global__ void testKernel(int *data, int *v_data) {
    if (threadIdx.x == 0) {
        // 设置偏移指针，指向数据块的中间部分
        char* base_ptr = (char*)data;
        char* offset_ptr = base_ptr + OFFSET;

        // 第一次读取，初始值应该是 0
        int value;
        asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(value) : "l"(offset_ptr));

        // 写入新值 0xdeadbeef
        asm volatile("st.global.cg.b32 [%0], %1;" : : "l"(offset_ptr), "r"(0xdeadbeef) : "memory");

        // 添加一些操作以确保写入操作不会被优化掉
        int dummy = 0;
        asm volatile(
            "{\n"
            "    .reg .u32 r0, r1, r2, r3;\n"
            "    mov.u32 r0, %1;\n"  // r0 = 0xdeadbeef
            "    mov.u32 r1, %2;\n"  // r1 = 0
            "    add.u32 r2, r0, r1;\n" // r2 = r0 + r1
            "    sub.u32 r3, r2, r1;\n" // r3 = r2 - r1
            "    mul.u32 r3, r3, r1;\n" // r3 = r3 * r1
            "    mov.u32 %0, r3;\n"    // 将计算结果存储到 dummy 中
            "}\n"
            : "=r"(dummy)
            : "r"(0xdeadbeef), "r"(0)
            : "r0", "r1", "r2", "r3", "memory");

        //*v_data = dummy;

        // 确保 discard 操作不会被优化掉
        asm volatile("discard.global.L2 [%0], 128; membar.gl;" : : "l"(offset_ptr) : "memory");
        __syncwarp(); // 确保 warp 内的所有线程同步

        // 第三次读取，查看 discard 后的值
        asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(value) : "l"(offset_ptr));

        // 将最终的值写入设备内存的开头部分
        data[0] = value;
    }
}

int main() {
    int *d_data;
    int *h_data = new int[1];  // 为了存储最终结果
    int *vv_data = new int[1];
    int *v_data;
    cudaMalloc(&v_data, sizeof(int));
    // 分配 128B CUDA 内存，确保其 128B 对齐
    cudaError_t err = cudaMalloc(&d_data, DATA_BLOCK_SIZE);
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // 确保分配的内存地址是 128B 对齐的
    assert(reinterpret_cast<uintptr_t>(d_data) % 128 == 0);

    // 初始化 CUDA 内存为 0
    cudaMemset(d_data, 0, DATA_BLOCK_SIZE);
    cudaMemset(v_data, 0, sizeof(int));
    // 启动内核函数进行测试
    testKernel<<<1, 1>>>(d_data, v_data);
    cudaDeviceSynchronize();

    // 复制结果回主机
    cudaMemcpy(h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(vv_data, v_data, sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    std::cout << "Final value after discard: " << std::hex << h_data[0] << std::endl;
    std::cout << "Value from v_data: " << *vv_data << std::endl;

    // 释放 CUDA 内存
    cudaFree(d_data);
    cudaFree(v_data);
    delete[] h_data;
    delete[] vv_data;

    return 0;
}
