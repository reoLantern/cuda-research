#include <mma.h>
#include <iostream>

using namespace nvcuda;

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_SIZE 32
#define DIV_CEIL(x, y) (((x) + (y)-1) / (y))

// 本示例只考虑简单情形，默认kernel输入的M、N、K与MMA的size相同

__global__ void mma16816NaiveKernel(const half *__restrict__ A, const half *__restrict__ B, half *__restrict__ C,
                                    size_t M, size_t N, size_t K)
{
    __shared__ half shmem_A[MMA_M][MMA_K];
    __shared__ half shmem_B[MMA_N][MMA_K];
    __shared__ half shmem_C[MMA_M][MMA_N];

    const size_t laneId = threadIdx.x % WARP_SIZE;

    uint32_t RA[4];
    uint32_t RB[8];
    uint32_t RC[4] = {0, 0, 0, 0};

    *((int4 *)(&shmem_A[laneId / 2][0]) + laneId % 2) =
        *((int4 *)(&A[(laneId / 2) * K]) + laneId % 2);

    if (laneId < MMA_N * 2) // laneId<16，只用16个线程存B
    {
        *((int4 *)(&shmem_B[laneId / 2][0]) + laneId % 2) =
            *((int4 *)(&B[(laneId / 2) * K]) + laneId % 2);
    }

    __syncthreads();

    uint32_t shmem_A_lane_addr = __cvta_generic_to_shared(&shmem_A[laneId % 16][(laneId / 16) * 8]);
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(RA[0]), "=r"(RA[1]), "=r"(RA[2]), "=r"(RA[3])
                 : "r"(shmem_A_lane_addr));

    // 这里需要注意嵌入式的PTX汇编中，shared memory的指针需要特殊处理一下。因为用
    // &shmem[...]这样得到的是generic的指针（8字节），直接该8字节值作为shared memory
    // 的地址可能会超出shared memory的地址范围，所以需要使__cvta_generic_to_shared()
    // 或者将该8字节值与上0xFFFFFF，使该指针指向shared memory的地址空间。详情参见
    // https://forums.developer.nvidia.com/t/problem-about-ptx-instruction-cp-async-ca-shared-global/224219

    uint32_t shmem_B_lane_addr = __cvta_generic_to_shared(&shmem_B[laneId % 8][((laneId / 8) % 2) * 8]);
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(RB[0]), "=r"(RB[1])
                 : "r"(shmem_B_lane_addr));

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
        : "=r"(RC[0]), "=r"(RC[1])
        : "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]), "r"(RB[0]), "r"(RB[1]), "r"(RC[0]), "r"(RC[1]));
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%5, %6};\n"
        : "=r"(RC[0]), "=r"(RC[1])
        : "r"(RA[0]), "r"(RA[1]), "r"(RB[0]), "r"(RC[0]), "r"(RC[1]));

    __syncthreads();

    *((uint32_t *)(&shmem_C[laneId / 4][0]) + laneId % 4) = RC[0];
    *((uint32_t *)(&shmem_C[laneId / 4 + 8][0]) + laneId % 4) = RC[1];

    __syncthreads();

    if (laneId < MMA_M)
    {
        *((int4 *)(&C[(laneId)*N])) = *((int4 *)(&shmem_C[laneId][0]));
    }

    __syncthreads();
}

int main()
{
    return 0;
}