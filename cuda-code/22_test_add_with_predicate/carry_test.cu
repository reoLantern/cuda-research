#include <cuda_runtime.h>
#include <stdio.h>

// 使用 extern "C" 防止 C++ 命名修饰，方便在 PTX/SASS 中查找函数名
extern "C" __global__ void add64_kernel(const unsigned long long* a,
                                        const unsigned long long* b,
                                        unsigned long long* c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    unsigned long long val_a = a[idx];
    unsigned long long val_b = b[idx];
    
    unsigned long long result = val_a + val_b;
    
    c[idx] = result;
}

int main() {
    // 简单的 Host 代码用于分配显存，确保 kernel 可以被编译
    unsigned long long *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, 1024);
    cudaMalloc(&d_b, 1024);
    cudaMalloc(&d_c, 1024);

    add64_kernel<<<1, 32>>>(d_a, d_b, d_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}