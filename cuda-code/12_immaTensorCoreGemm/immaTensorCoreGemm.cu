#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>

// Externally configurable parameters.

#ifndef CPU_DEBUG
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define CPU_DEBUG 0
#endif

const int iter = 32;     // 测时间迭代次数
const int numkernel = 4; // 运行的kernel种类

// GPU configuration.

#define WARP_SIZE 32

// MMA matrix tile dimensions.

#define M 8
#define N 8
#define K 32

// GEMM configuration.
#if CPU_DEBUG
#define M_TILES 4
#define N_TILES 4
#define K_TILES 4
#else
#define M_TILES 512
#define N_TILES 512
#define K_TILES 512
#endif

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define SEXT_4TO32(num) ((num & 0x08) ? (num | 0xFFFFFFF0) : num)

using namespace nvcuda;

__host__ void init_host_matrices(uint32_t *a, uint32_t *b, int *c)
{
    uint32_t value;
    for (int i = 0; i < M_GLOBAL; i++)
    {
        for (int j = 0; j < K_GLOBAL / 8; j++)
        {
            value = 0;
            for (int k = 0; k < 8; k++)
            {
                value |= ((rand() & 0x0F) << (k * 4));
            }
            a[i * K_GLOBAL / 8 + j] = value;
        }
    }

    for (int i = 0; i < N_GLOBAL; i++)
    {
        for (int j = 0; j < K_GLOBAL / 8; j++)
        {
            value = 0;
            for (int k = 0; k < 8; k++)
            {
                value |= ((rand() & 0x0F) << (k * 4));
            }
            b[i * K_GLOBAL / 8 + j] = value;
        }
    }

    for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++)
    {
        c[t] = (rand() % 3);
    }
}

__host__ void matMultiplyOnHost(const uint32_t *A, const uint32_t *B, int *C,
                                int numARows, int numAColumns,
                                int numBRows, int numBColumns, int numCRows,
                                int numCColumns)
{
    for (int i = 0; i < numCRows; i++)
    {
        // printf("host computing C[%d][]...\n", i);
        for (int j = 0; j < numCColumns; j++)
        {
            int temp = 0;
            int debugtmp = 0;
            for (int k = 0; k < numAColumns; k++)
            {

                // 获取矩阵A和B中对应的整数索引和位偏移
                int indexA = (i * numAColumns + k) / 8;
                int offsetA = (i * numAColumns + k) % 8;
                int indexB = (j * numBRows + k) / 8;
                int offsetB = (j * numBRows + k) % 8;
                // 从矩阵A和B中获取对应的整数值
                int valueA = A[indexA];
                int valueB = B[indexB];

                // 获取矩阵A和B中对应的4位元素值
                int elementA = SEXT_4TO32((valueA >> (offsetA * 4)) & 0x0F);
                int elementB = SEXT_4TO32((valueB >> (offsetB * 4)) & 0x0F);

                // 进行乘法运算并累加结果
                temp += (elementA * elementB);
                debugtmp += (elementA * elementB);
            }

            C[i * numCColumns + j] = temp + C[i * numCColumns + j];
        }
    }
}
__global__ void compute_gemm_imma(const uint8_t *A, const uint8_t *B,
                                  const int *C, int *D, int alpha, int beta)
{
}

__global__ void simple_wmma_gemm_imma8832(const uint32_t *a, const uint32_t *b,
                                          const int *c, int *d, int m_ld, int n_ld,
                                          int k_ld)
{
    const int WMMA_M = 8;
    const int WMMA_N = 8;
    const int WMMA_K = 32;

    // Leading dimensions. Packed with no transpositions.
    int lda = k_ld;
    int ldb = k_ld;
    int ldc = n_ld;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::experimental::precision::s4,
                   wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::experimental::precision::s4,
                   wmma::col_major>
        b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> acc_frag;

    int cCol = warpN * WMMA_N;
    int cRow = warpM * WMMA_M;
    if (cRow < m_ld && cCol < n_ld)
    {
        wmma::load_matrix_sync(acc_frag, c + cCol + cRow * ldc, ldc,
                               wmma::mem_row_major);
    }

    // Loop over k
    for (int i = 0; i < k_ld; i += WMMA_K)
    {
        int aCol = i;
        int aRow = warpM * WMMA_M;

        int bCol = i;
        int bRow = warpN * WMMA_N;

        // Bounds checking
        if (aRow < m_ld && aCol < k_ld && bRow < n_ld && bCol < k_ld)
        {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aCol / 8 + aRow * (lda / 8), lda);
            wmma::load_matrix_sync(b_frag, b + bCol / 8 + bRow * (ldb / 8), ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    wmma::store_matrix_sync(d + cCol + cRow * ldc, acc_frag, ldc, wmma::mem_row_major);
}
__host__ float do_simple_wmma_gemm_imma8832(const uint32_t *A, const uint32_t *B,
                                            const int *C, int *D,
                                            const int *result_host,
                                            int m_ld, int n_ld, int k_ld)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

#if CPU_DEBUG
    int *result_hD = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);
#endif
    const int WMMA_M = 8;
    const int WMMA_N = 8;

    cudaEventRecord(start);

    dim3 gridDim;
    dim3 blockDim;
    // blockDim.x must be a multiple of warpSize
    // 128x4 means we have 16 warps and a block computes a 32x32 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) /
                (WMMA_M * blockDim.x / 32);
    gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    printf("Computing... using simple_wmma_gemm_imma8832 kernel\n");
    simple_wmma_gemm_imma8832<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, N_GLOBAL, K_GLOBAL);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

#if CPU_DEBUG
    cudaMemcpy(result_hD, D, sizeof(int) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);

    printf("Verifying correctness of the computations...\n");

    for (int i = 0; i < N_GLOBAL * M_GLOBAL; i++)
    {
        if (abs(result_hD[i] - result_host[i]) > 0)
        {
            printf("mismatch i=%d result_hD=%d result_host=%d\n", i, result_hD[i],
                   result_host[i]);
        }
    }
    free(result_hD);
#endif

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("wmma8832 Time: %f ms\n", milliseconds);
    printf("wmma8832 TOPS: %.2f\n", (((double)M_GLOBAL * N_GLOBAL * K_GLOBAL * 2) / (milliseconds / 1000.)) / 1e12);
    // cudaDeviceSynchronize();
    return milliseconds;
}

__global__ void simple_mma_gemm_imma8832(const uint32_t *a, const uint32_t *b,
                                         const int *c, int *d, int m_ld, int n_ld,
                                         int k_ld)
{
    const int WMMA_M = 8;
    const int WMMA_N = 8;
    const int WMMA_K = 32;

    // Leading dimensions. Packed with no transpositions.
    int lda = k_ld;
    int ldb = k_ld;
    int ldc = n_ld;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    const uint32_t laneId = threadIdx.x % WARP_SIZE;

    // Declare the fragments
    uint32_t RA, RB;
    uint32_t RC[2];

    int cCol = warpN * WMMA_N;
    int cRow = warpM * WMMA_M;
    const int *src_gmem_ptr = NULL;
    if (cRow < m_ld && cCol < n_ld)
    {
        src_gmem_ptr = (c + cCol + cRow * ldc + laneId / 4 * ldc + (laneId % 4) * 2);
        RC[0] = *(src_gmem_ptr);
        RC[1] = *(src_gmem_ptr + 1);
    }

    // Loop over k
    for (int i = 0; i < k_ld; i += WMMA_K)
    {
        int aCol = i;
        int aRow = warpM * WMMA_M;

        int bCol = i;
        int bRow = warpN * WMMA_N;

        // Bounds checking
        if (aRow < m_ld && aCol < k_ld && bRow < n_ld && bCol < k_ld)
        {
            // Load the inputs
            RA = *(a + aCol / 8 + aRow * (lda / 8) + (laneId / 4) * (lda / 8) + laneId % 4);
            RB = *(b + bCol / 8 + bRow * (ldb / 8) + (laneId / 4) * (lda / 8) + laneId % 4);

            // Perform the matrix multiplication
            asm volatile("mma.sync.aligned.m8n8k32.row.col.s32.s4.s4.s32 {%0, %1}, {%2}, {%3}, {%4, %5};\n"
                         : "=r"(RC[0]), "=r"(RC[1])
                         : "r"(RA), "r"(RB), "r"(RC[0]), "r"(RC[1]));
        }
    }

    int *dst_gmem_ptr = (d + cCol + cRow * ldc + laneId / 4 * ldc + (laneId % 4) * 2);
    *(dst_gmem_ptr) = RC[0];
    *(dst_gmem_ptr + 1) = RC[1];
}
__host__ float do_simple_mma_gemm_imma8832(const uint32_t *A, const uint32_t *B,
                                           const int *C, int *D,
                                           const int *result_host,
                                           int m_ld, int n_ld, int k_ld)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

#if CPU_DEBUG
    int *result_hD = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);
#endif
    const int WMMA_M = 8;
    const int WMMA_N = 8;

    cudaEventRecord(start);

    dim3 gridDim;
    dim3 blockDim;
    // blockDim.x must be a multiple of warpSize
    // 128x4 means we have 16 warps and a block computes a 32x32 output tile
    blockDim.x = 128;
    blockDim.y = 4;

    gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) /
                (WMMA_M * blockDim.x / 32);
    gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    printf("Computing... using simple_mma_gemm_imma8832 kernel\n");
    simple_mma_gemm_imma8832<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, N_GLOBAL, K_GLOBAL);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

#if CPU_DEBUG
    cudaMemcpy(result_hD, D, sizeof(int) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);

    printf("Verifying correctness of the computations...\n");

    for (int i = 0; i < N_GLOBAL * M_GLOBAL; i++)
    {
        if (abs(result_hD[i] - result_host[i]) > 0)
        {
            printf("mismatch i=%d result_hD=%d result_host=%d\n", i, result_hD[i],
                   result_host[i]);
        }
    }
    free(result_hD);
#endif

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("mma8832 Time: %f ms\n", milliseconds);
    printf("mma8832 TOPS: %.2f\n", (((double)M_GLOBAL * N_GLOBAL * K_GLOBAL * 2) / (milliseconds / 1000.)) / 1e12);
    // cudaDeviceSynchronize();
    return milliseconds;
}

__global__ void simple_mma_gemm_imma16832(const uint32_t *a, const uint32_t *b,
                                          const int *c, int *d, int m_ld, int n_ld,
                                          int k_ld)
{
    const int WMMA_M = 16;
    const int WMMA_N = 8;
    const int WMMA_K = 32;

    // Leading dimensions. Packed with no transpositions.
    int lda = k_ld;
    int ldb = k_ld;
    int ldc = n_ld;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    const uint32_t laneId = threadIdx.x % WARP_SIZE;

    // Declare the fragments
    uint32_t RA[2], RB;
    uint32_t RC[4];

    int cCol = warpN * WMMA_N;
    int cRow = warpM * WMMA_M;
    const int *src_gmem_ptr = NULL;
    const uint32_t *src_gmem_ptr_a = NULL;
    if (cRow < m_ld && cCol < n_ld)
    {
        src_gmem_ptr = (c + cCol + cRow * ldc + laneId / 4 * ldc + (laneId % 4) * 2);
        RC[0] = *(src_gmem_ptr);
        RC[1] = *(src_gmem_ptr + 1);
        RC[2] = *(src_gmem_ptr + 8 * ldc);
        RC[3] = *(src_gmem_ptr + 8 * ldc + 1);
    }

    // Loop over k
    for (int i = 0; i < k_ld; i += WMMA_K)
    {
        int aCol = i;
        int aRow = warpM * WMMA_M;

        int bCol = i;
        int bRow = warpN * WMMA_N;

        // Bounds checking
        if (aRow < m_ld && aCol < k_ld && bRow < n_ld && bCol < k_ld)
        {
            // Load the inputs
            src_gmem_ptr_a = (a + aCol / 8 + aRow * (lda / 8) + (laneId / 4) * (lda / 8) + laneId % 4);
            RA[0] = *(src_gmem_ptr_a);
            RA[1] = *(src_gmem_ptr_a + 8 * (lda / 8));
            RB = *(b + bCol / 8 + bRow * (ldb / 8) + (laneId / 4) * (lda / 8) + laneId % 4);

            // Perform the matrix multiplication
            asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                         : "=r"(RC[0]), "=r"(RC[1]), "=r"(RC[2]), "=r"(RC[3])
                         : "r"(RA[0]), "r"(RA[1]), "r"(RB) "r"(RC[0]), "r"(RC[1]), "r"(RC[2]), "r"(RC[3]));
        }
    }

    int *dst_gmem_ptr = (d + cCol + cRow * ldc + laneId / 4 * ldc + (laneId % 4) * 2);
    *(dst_gmem_ptr) = RC[0];
    *(dst_gmem_ptr + 1) = RC[1];
    *(dst_gmem_ptr + 8 * ldc) = RC[2];
    *(dst_gmem_ptr + 8 * ldc + 1) = RC[3];
}
__host__ float do_simple_mma_gemm_imma16832(const uint32_t *A, const uint32_t *B,
                                            const int *C, int *D,
                                            const int *result_host,
                                            int m_ld, int n_ld, int k_ld)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

#if CPU_DEBUG
    int *result_hD = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);
#endif
    const int WMMA_M = 16;
    const int WMMA_N = 8;

    cudaEventRecord(start);

    dim3 gridDim;
    dim3 blockDim;
    // blockDim.x must be a multiple of warpSize
    // 128x4 means we have 16 warps and a block computes a 32x32 output tile
    blockDim.x = 64;
    blockDim.y = 4;

    gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) /
                (WMMA_M * blockDim.x / 32);
    gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    printf("Computing... using simple_mma_gemm_imma16832 kernel\n");
    simple_mma_gemm_imma16832<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, N_GLOBAL, K_GLOBAL);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

#if CPU_DEBUG
    cudaMemcpy(result_hD, D, sizeof(int) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);

    printf("Verifying correctness of the computations...\n");

    for (int i = 0; i < N_GLOBAL * M_GLOBAL; i++)
    {
        if (abs(result_hD[i] - result_host[i]) > 0)
        {
            printf("mismatch i=%d result_hD=%d result_host=%d\n", i, result_hD[i],
                   result_host[i]);
        }
    }
    free(result_hD);
#endif

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("mma16832 Time: %f ms\n", milliseconds);
    printf("mma16832 TOPS: %.2f\n", (((double)M_GLOBAL * N_GLOBAL * K_GLOBAL * 2) / (milliseconds / 1000.)) / 1e12);
    // cudaDeviceSynchronize();
    return milliseconds;
}

__global__ void simple_mma_gemm_imma16864(const uint32_t *a, const uint32_t *b,
                                          const int *c, int *d, int m_ld, int n_ld,
                                          int k_ld)
{
    const int WMMA_M = 16;
    const int WMMA_N = 8;
    const int WMMA_K = 64;

    // Leading dimensions. Packed with no transpositions.
    int lda = k_ld;
    int ldb = k_ld;
    int ldc = n_ld;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    const uint32_t laneId = threadIdx.x % WARP_SIZE;

    // Declare the fragments
    uint32_t RA[4], RB[2];
    uint32_t RC[4];

    int cCol = warpN * WMMA_N;
    int cRow = warpM * WMMA_M;
    const int *src_gmem_ptr = NULL;
    const uint32_t *src_gmem_ptr_a = NULL;
    if (cRow < m_ld && cCol < n_ld)
    {
        src_gmem_ptr = (c + cCol + cRow * ldc + laneId / 4 * ldc + (laneId % 4) * 2);
        RC[0] = *(src_gmem_ptr);
        RC[1] = *(src_gmem_ptr + 1);
        RC[2] = *(src_gmem_ptr + 8 * ldc);
        RC[3] = *(src_gmem_ptr + 8 * ldc + 1);
    }

    // Loop over k
    for (int i = 0; i < k_ld; i += WMMA_K)
    {
        int aCol = i;
        int aRow = warpM * WMMA_M;

        int bCol = i;
        int bRow = warpN * WMMA_N;

        // Bounds checking
        if (aRow < m_ld && aCol < k_ld && bRow < n_ld && bCol < k_ld)
        {
            // Load the inputs
            src_gmem_ptr_a = (a + aCol / 8 + aRow * (lda / 8) + (laneId / 4) * (lda / 8) + laneId % 4);
            RA[0] = *(src_gmem_ptr_a);
            RA[1] = *(src_gmem_ptr_a + 8 * (lda / 8));
            RA[2] = *(src_gmem_ptr_a + 4);
            RA[3] = *(src_gmem_ptr_a + 8 * (lda / 8) + 4);
            src_gmem_ptr_a = (b + bCol / 8 + bRow * (ldb / 8) + (laneId / 4) * (lda / 8) + laneId % 4);
            RB[0] = *(src_gmem_ptr_a);
            RB[1] = *(src_gmem_ptr_a + 4);

            // Perform the matrix multiplication
            asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                         : "=r"(RC[0]), "=r"(RC[1]), "=r"(RC[2]), "=r"(RC[3])
                         : "r"(RA[0]), "r"(RA[1]), "r"(RA[2]), "r"(RA[3]), "r"(RB[0]), "r"(RB[1]),
                           "r"(RC[0]), "r"(RC[1]), "r"(RC[2]), "r"(RC[3]));
        }
    }

    int *dst_gmem_ptr = (d + cCol + cRow * ldc + laneId / 4 * ldc + (laneId % 4) * 2);
    *(dst_gmem_ptr) = RC[0];
    *(dst_gmem_ptr + 1) = RC[1];
    *(dst_gmem_ptr + 8 * ldc) = RC[2];
    *(dst_gmem_ptr + 1 + 8 * ldc) = RC[3];
}
__host__ float do_simple_mma_gemm_imma16864(const uint32_t *A, const uint32_t *B,
                                            const int *C, int *D,
                                            const int *result_host,
                                            int m_ld, int n_ld, int k_ld)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

#if CPU_DEBUG
    int *result_hD = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);
#endif
    const int WMMA_M = 16;
    const int WMMA_N = 8;

    cudaEventRecord(start);

    dim3 gridDim;
    dim3 blockDim;
    // blockDim.x must be a multiple of warpSize
    // 128x4 means we have 16 warps and a block computes a 32x32 output tile
    blockDim.x = 64;
    blockDim.y = 4;

    gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) /
                (WMMA_M * blockDim.x / 32);
    gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

    printf("Computing... using simple_mma_gemm_imma16864 kernel\n");
    simple_mma_gemm_imma16864<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, N_GLOBAL, K_GLOBAL);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

#if CPU_DEBUG
    cudaMemcpy(result_hD, D, sizeof(int) * M_GLOBAL * N_GLOBAL, cudaMemcpyDeviceToHost);

    printf("Verifying correctness of the computations...\n");

    for (int i = 0; i < N_GLOBAL * M_GLOBAL; i++)
    {
        if (abs(result_hD[i] - result_host[i]) > 0)
        {
            printf("mismatch i=%d result_hD=%d result_host=%d\n", i, result_hD[i],
                   result_host[i]);
        }
    }
    free(result_hD);
#endif

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("mma16864 Time: %f ms\n", milliseconds);
    printf("mma16864 TOPS: %.2f\n", (((double)M_GLOBAL * N_GLOBAL * K_GLOBAL * 2) / (milliseconds / 1000.)) / 1e12);
    // cudaDeviceSynchronize();
    return milliseconds;
}

void printArrayTableAndAverage(float arr[numkernel][iter])
{
    // Printing the table
    std::cout << "Array Table:\n";
    for (int i = 0; i < numkernel; ++i)
    {
        float sum = 0.0;
        for (int j = 0; j < iter; ++j)
        {
            std::cout << arr[i][j] << "\t";
            sum += arr[i][j];
        }
        float average = sum / iter;
        std::cout << " | Average: " << average << std::endl;
    }
}

int main(int argc, char **argv)
{
    printf("Initializing...\n");
    printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
    printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
    printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

    uint32_t *A, *A_h, *B, *B_h; // every 8 u4 stored in 1 uint32_t
    int *C, *C_h, *D;
    int *result_host = NULL;
    A_h = (uint32_t *)malloc(sizeof(uint32_t) * M_GLOBAL * K_GLOBAL / 8); // row major
    B_h = (uint32_t *)malloc(sizeof(uint32_t) * N_GLOBAL * K_GLOBAL / 8); // column major
    C_h = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);
    cudaMalloc(reinterpret_cast<void **>(&A), sizeof(uint32_t) * M_GLOBAL * K_GLOBAL / 8);
    cudaMalloc(reinterpret_cast<void **>(&B), sizeof(uint32_t) * K_GLOBAL * N_GLOBAL / 8);
    cudaMalloc(reinterpret_cast<void **>(&C), sizeof(int) * M_GLOBAL * N_GLOBAL);
    cudaMalloc(reinterpret_cast<void **>(&D), sizeof(int) * M_GLOBAL * N_GLOBAL);

    assert(((unsigned long long)A) % 128 == 0);
    assert(((unsigned long long)B) % 128 == 0);
    assert(((unsigned long long)C) % 128 == 0);
    assert(((unsigned long long)D) % 128 == 0);

    init_host_matrices(A_h, B_h, C_h);

    cudaMemcpy(A, A_h, sizeof(uint32_t) * M_GLOBAL * K_GLOBAL / 8, cudaMemcpyHostToDevice);
    cudaMemcpy(B, B_h, sizeof(uint32_t) * K_GLOBAL * N_GLOBAL / 8, cudaMemcpyHostToDevice);
    cudaMemcpy(C, C_h, sizeof(int) * M_GLOBAL * N_GLOBAL, cudaMemcpyHostToDevice);
    cudaMemcpy(D, 0, sizeof(int) * M_GLOBAL * N_GLOBAL, cudaMemcpyHostToDevice);

    printf("Preparing data for GPU...\n");

    assert(((unsigned long long)A) % 128 == 0);
    assert(((unsigned long long)B) % 128 == 0);
    assert(((unsigned long long)C) % 128 == 0);
    assert(((unsigned long long)D) % 128 == 0);

#if CPU_DEBUG
    result_host = (int *)malloc(sizeof(int) * M_GLOBAL * N_GLOBAL);
    memcpy(result_host, C_h, sizeof(int) * M_GLOBAL * N_GLOBAL);
    printf("CPU_DEBUG: Start host compute\n");
    matMultiplyOnHost(A_h, B_h, result_host, M_GLOBAL, K_GLOBAL,
                      K_GLOBAL, N_GLOBAL, M_GLOBAL, N_GLOBAL);
    printf("CPU_DEBUG: Finish host compute\n");
#endif

    float time_record[numkernel][iter];

    for (int i = 0; i < iter; i++)
    {
        time_record[0][i] = do_simple_wmma_gemm_imma8832(A, B, C, D, result_host, M_GLOBAL, N_GLOBAL, K_GLOBAL);
        time_record[1][i] = do_simple_mma_gemm_imma8832(A, B, C, D, result_host, M_GLOBAL, N_GLOBAL, K_GLOBAL);
        time_record[2][i] = do_simple_mma_gemm_imma16832(A, B, C, D, result_host, M_GLOBAL, N_GLOBAL, K_GLOBAL);
        time_record[3][i] = do_simple_mma_gemm_imma16864(A, B, C, D, result_host, M_GLOBAL, N_GLOBAL, K_GLOBAL);
    }
    printArrayTableAndAverage(time_record);

    free(A_h);
    free(B_h);
    free(C_h);
#if CPU_DEBUG
    free(result_host);
#endif
    cudaFree(reinterpret_cast<void *>(A));
    cudaFree(reinterpret_cast<void *>(B));
    cudaFree(reinterpret_cast<void *>(C));
    cudaFree(reinterpret_cast<void *>(D));

    return EXIT_SUCCESS;
}
