// #define __CUDA_ARCH__ 860

#include <cstdio>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

__global__ void hello_world()
{
    int i = threadIdx.x;
    printf("hello world from thread %d, blockidx.x=%d, blockdim.x=%d\n", i, blockIdx.x, blockDim.x);
}

__global__ void kernel(int a, int b, int *c)
{
    *c = a + b;
}

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
        printf("未发现可用的GPU设备\n");
        return 0;
    }
    printf("可用的GPU数量：%d\n", deviceCount);
    cudaSetDevice(0);

    hello_world<<<1, 10>>>();
    cudaDeviceSynchronize();
    printf("Execution ends\n");
}
