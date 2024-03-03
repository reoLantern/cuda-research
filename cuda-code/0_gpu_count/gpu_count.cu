#include <stdio.h>
#include <cuda_runtime.h>

int main()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("未发现可用的GPU设备\n");
        return 0;
    }

    printf("可用的GPU数量：%d\n", deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        printf("---- GPU 设备信息 ----\n");
        printf("设备索引号: %d\n", i);
        printf("设备名称: %s\n", deviceProp.name);
        printf("设备计算能力: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("多处理器数量: %d\n", deviceProp.multiProcessorCount);
        printf("每个多处理器的最大线程块数量: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("每个线程块的最大线程数量: %d\n", deviceProp.maxThreadsPerBlock);
        printf("设备全局内存大小: %.2f GB\n", static_cast<float>(deviceProp.totalGlobalMem) / (1024 * 1024 * 1024));
        printf("\n");
    }

    int selectedDevice = 0;  // 选择要使用的GPU索引号

    cudaSetDevice(selectedDevice);

    // 这里可以执行您的CUDA代码

    return 0;
}
