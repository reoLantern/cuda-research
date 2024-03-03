#include <iostream>

__global__ void testKernel(int *result)
{
    unsigned int x;
    asm volatile("mov.u32 %0, %%clock;"
                 : "=r"(x));

    // 在这里插入你要记录运行周期的代码
    // 这里只是简单的将结果写入全局内存，以便在主机代码中读取

    result[threadIdx.x] = x;
}

int main()
{
    const int N = 32;
    int *result;
    cudaMallocManaged(&result, N * sizeof(int));

    testKernel<<<1, N>>>(result);
    cudaDeviceSynchronize();

    for (int i = 0; i < N; i++)
    {
        std::cout << "Thread " << i << " cycles: " << result[i] << std::endl;
    }

    cudaFree(result);

    return 0;
}
