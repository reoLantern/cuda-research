#include <mma.h>
#include <iostream>
#include <iomanip>

using namespace nvcuda;

#define WMMA_M 8
#define WMMA_N 8
#define WMMA_K 32

#define WARP_SIZE 32

__global__ void imma8832NaiveKernel(const uint32_t *__restrict__ A, const uint32_t *__restrict__ B,
                                    uint32_t *__restrict__ C)
{

    __shared__ uint32_t shmem_A[WMMA_M][WMMA_K / 8];
    __shared__ uint32_t shmem_B[WMMA_N][WMMA_K / 8];
    __shared__ uint32_t shmem_C[WMMA_M][WMMA_N];

    const uint32_t laneId = threadIdx.x % WARP_SIZE;

    if (laneId < 8) // 仅需要8个线程来初始化shmem
    {
        *((int4 *)(&shmem_A[laneId][0])) = *((int4 *)(&A[laneId * WMMA_K / 8]));
        *((int4 *)(&shmem_B[laneId][0])) = *((int4 *)(&B[laneId * WMMA_K / 8]));
    }

    // __syncthreads(); // 若只调用1个warp，不需要syncthreads

    uint32_t RA;
    uint32_t RB;
    uint32_t RC[2] = {0, 0};
    uint32_t start, stop;

    uint32_t shmem_A_lane_addr = __cvta_generic_to_shared(&shmem_A[laneId % 8][0]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                 : "=r"(RA)
                 : "r"(shmem_A_lane_addr));

    uint32_t shmem_B_lane_addr = __cvta_generic_to_shared(&shmem_B[laneId % 8][0]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                 : "=r"(RB)
                 : "r"(shmem_B_lane_addr));

    asm volatile("mov.u32 %0, %%clock;"
                 : "=r"(start)
                 :
                 : "memory");

    asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0, %1}, {%2}, {%3}, {%4, %5};\n"
                 : "=r"(RC[0]), "=r"(RC[1])
                 : "r"(RA), "r"(RB), "r"(RC[0]), "r"(RC[1]));

    asm volatile("mov.u32 %0, %%clock;"
                 : "=r"(stop)
                 :
                 : "memory");

    printf("Thread%u imma starts at cycle %u, ends at cycle %u, takes %u cycles.\n", laneId, start, stop, stop - start);

    // __syncthreads(); // 若只调用1个warp，不需要syncthreads

    *((uint32_t *)(&shmem_C[laneId / 4][(laneId % 4) * 2])) = RC[0];
    *((uint32_t *)(&shmem_C[laneId / 4][(laneId % 4) * 2] + 1)) = RC[1];

    // __syncthreads(); // 若只调用1个warp，不需要syncthreads

    if (laneId < 16)
    {
        *((int4 *)(&C[laneId * 4])) = *((int4 *)(&shmem_C[laneId / 2][(laneId % 2) * 4]));
    }

    // __syncthreads(); // 若只调用1个warp，不需要syncthreads
}

int main()
{

    uint32_t *d_a, *h_a, *d_b, *h_b; // every 8 u4 stored in 1 uint
    uint32_t *d_c, *h_c;
    h_c = new uint32_t[WMMA_M * WMMA_N];
    h_b = new uint32_t[WMMA_K * WMMA_N / 8];
    h_a = new uint32_t[WMMA_M * WMMA_K / 8];
    cudaMalloc(reinterpret_cast<void **>(&d_a), WMMA_M * WMMA_K / 8 * sizeof(uint32_t));
    cudaMalloc(reinterpret_cast<void **>(&d_b), WMMA_K * WMMA_N / 8 * sizeof(uint32_t));
    cudaMalloc(reinterpret_cast<void **>(&d_c), WMMA_M * WMMA_N * sizeof(int));

    for (int i = 0; i < WMMA_M * WMMA_K / 8; i++)
    {
        uint32_t value = 0;
        for (int k = 0; k < 8; k++)
        {
            value |= (((k + i) & 0x0F) << (k * 4));
        }
        h_a[i] = value; // A为row-major，每行32个int4，存放在4个int32中
    }
    for (int i = 0; i < WMMA_M * WMMA_K / 8; i++)
    {
        uint32_t value = 0;
        for (int k = 0; k < 8; k++)
        {
            value |= ((i / 2 & 0x0F) << (k * 4));
        }
        h_b[i] = value; // B为column-major，每行32个int4，存放在4个int32中
    }
    std::cout << "A=" << std::hex;
    for (int i = 0; i < WMMA_M * WMMA_K / 8; i++)
        std::cout << std::setw(8) << std::setfill('0') << h_a[i] << ",";
    std::cout << "\nB=";
    for (int i = 0; i < WMMA_M * WMMA_K / 8; i++)
        std::cout << std::setw(8) << std::setfill('0') << h_b[i] << ",";
    std::cout << std::dec << "\n";
    cudaMemcpy(d_a, h_a, WMMA_M * WMMA_K / 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, WMMA_K * WMMA_N / 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    imma8832NaiveKernel<<<1, 32>>>(d_a, d_b, d_c);

    // cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, WMMA_M * WMMA_N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::cout << "immaC=" << std::hex;
    for (int i = 0; i < WMMA_M * WMMA_N; i++)
        std::cout << h_c[i] << ",";
    std::cout << std::endl;
}
