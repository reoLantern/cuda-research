#include <mma.h>
#include <iostream>
#include <iomanip>

using namespace nvcuda;

#define WMMA_M 8
#define WMMA_N 8
#define WMMA_K 32

#define WARP_SIZE 32

#define M 8
#define N 16
#define K 32

int signExt4bitToInt(int num)
{ // 对符号位进行扩展
    return (num & 0x08) ? (num | 0xFFFFFFF0) : num;
}

__global__ void imma8832NaiveKernel(const uint32_t *__restrict__ A, const uint32_t *__restrict__ B,
                                    uint32_t *__restrict__ C)
{
    __shared__ uint32_t shmem_A[WMMA_M][WMMA_K / 8];
    __shared__ uint32_t shmem_B[2][WMMA_N][WMMA_K / 8];
    __shared__ uint32_t shmem_C[2][WMMA_M][WMMA_N];

    const size_t warpId = threadIdx.x / WARP_SIZE;
    const size_t laneId = threadIdx.x % WARP_SIZE;

    if (warpId == 0 && laneId < 8) // 仅需要8个线程来初始化shmem
    {
        *((int4 *)(&shmem_A[laneId][0])) = *((int4 *)(&A[laneId * WMMA_K / 8]));
    }
    if (laneId < 8)
    {
        *((int4 *)(&shmem_B[warpId][laneId][0])) = *((int4 *)(&B[warpId * K * WMMA_N / 8 + laneId * WMMA_K / 8]));
    }
    __syncthreads();

    uint32_t RA;
    uint32_t RB;
    uint32_t RC[2] = {0, 0};

    uint32_t shmem_A_lane_addr = __cvta_generic_to_shared(&shmem_A[laneId % 8][0]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                 : "=r"(RA)
                 : "r"(shmem_A_lane_addr));

    uint32_t shmem_B_lane_addr = __cvta_generic_to_shared(&shmem_B[warpId][laneId % 8][0]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                 : "=r"(RB)
                 : "r"(shmem_B_lane_addr));

    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0, %1}, {%2}, {%3}, {%4, %5};\n"
                 : "=r"(RC[0]), "=r"(RC[1])
                 : "r"(RA), "r"(RB), "r"(RC[0]), "r"(RC[1]));

    __syncthreads();

    *((uint32_t *)(&shmem_C[warpId][laneId / 4][(laneId % 4) * 2])) = RC[0];
    *((uint32_t *)(&shmem_C[warpId][laneId / 4][(laneId % 4) * 2] + 1)) = RC[1];

    __syncthreads();

    if (laneId < 16)
    {
        *((int4 *)(&C[(laneId / 2) * 16 + (laneId % 2) * 4 + warpId * 8])) =
            *((int4 *)(&shmem_C[warpId][laneId / 2][(laneId % 2) * 4]));
    }

    __syncthreads();
}

int main()
{

    uint32_t *d_a, *h_a, *d_b, *h_b; // every 8 u4 stored in 1 uint
    uint32_t *d_c, *h_c;
    h_c = new uint32_t[M * N];
    h_b = new uint32_t[K * N / 8];
    h_a = new uint32_t[M * K / 8];
    cudaMalloc(reinterpret_cast<void **>(&d_a), M * K / 8 * sizeof(uint32_t));
    cudaMalloc(reinterpret_cast<void **>(&d_b), K * N / 8 * sizeof(uint32_t));
    cudaMalloc(reinterpret_cast<void **>(&d_c), M * N * sizeof(int));

    for (int i = 0; i < M * K / 8; i++)
    {
        uint32_t value = 0;
        for (int k = 0; k < 8; k++)
        {
            value |= (((k + i) & 0x0F) << (k * 4));
        }
        h_a[i] = value; // A为row-major，每行32个int4，存放在4个int32中
    }
    for (int i = 0; i < K * N / 8; i++)
    {
        uint32_t value = 0;
        for (int k = 0; k < 8; k++)
        {
            value |= ((i / 2 & 0x0F) << (k * 4));
        }
        h_b[i] = value; // B为column-major，每行32个int4，存放在4个int32中
    }
    std::cout << "A=" << std::hex;
    for (int i = 0; i < M * K / 8; i++)
        std::cout << std::setw(8) << std::setfill('0') << h_a[i] << ",";
    std::cout << "\nB=";
    for (int i = 0; i < K * N / 8; i++)
        std::cout << std::setw(8) << std::setfill('0') << h_b[i] << ",";
    std::cout << std::dec << "\n";
    cudaMemcpy(d_a, h_a, M * K / 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N / 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    imma8832NaiveKernel<<<1, 64>>>(d_a, d_b, d_c);

    // cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, M * N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::cout << "immaC=" << std::hex;
    for (int i = 0; i < M * N; i++)
        std::cout << h_c[i] << ",";
    std::cout << std::endl;

    /*** reference ***/
    int *ref_C = new int[M * N];
    for (int i = 0; i < M; ++i) // 计算参考结果
    {
        for (int j = 0; j < N; ++j)
        {
            int result = 0;
            for (int k = 0; k < K; ++k)
            {
                // 获取矩阵A和B中对应的整数索引和位偏移
                int indexA = (i * K + k) / 8;
                int offsetA = k % 8;
                int indexB = (j * K + k) / 8;
                int offsetB = k % 8;
                // 从矩阵A和B中获取对应的整数值
                int valueA = h_a[indexA];
                int valueB = h_b[indexB];
                // 获取矩阵A和B中对应的4位元素值
                int elementA = signExt4bitToInt((valueA >> (offsetA * 4)) & 0x0F);
                int elementB = signExt4bitToInt((valueB >> (offsetB * 4)) & 0x0F);
                // 进行乘法运算并累加结果
                result += (elementA * elementB);
            }
            ref_C[i * N + j] = result;
        }
    }
    std::cout << "ref_C=" << std::hex;
    for (int i = 0; i < M * N; i++)
        std::cout << ref_C[i] << ",";
    std::cout << std::endl;
}
