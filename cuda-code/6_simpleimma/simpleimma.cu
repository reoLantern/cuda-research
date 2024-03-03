#include <mma.h>
#include <iostream>
#include <iomanip>

using namespace nvcuda;

#define WMMA_M 8
#define WMMA_N 8
#define WMMA_K 32

__global__ void wmma_ker(uint32_t *a, uint32_t *b, int *c, uint *start, uint *stop)
{
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::experimental::precision::u4, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::experimental::precision::u4, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, int> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0);

    int lda = WMMA_K;
    int ldb = WMMA_K;
    int ldc = WMMA_N;
    // Load the inputs
    wmma::load_matrix_sync(a_frag, a, lda);
    wmma::load_matrix_sync(b_frag, b, ldb);

    uint x, y;
    asm volatile("mov.u32 %0, %%clock;"
                 : "=r"(x));

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    asm volatile("mov.u32 %0, %%clock;"
                 : "=r"(y));

    start[threadIdx.x] = x;
    stop[threadIdx.x] = y;

    // Store the output
    wmma::store_matrix_sync(c, c_frag, ldc, wmma::mem_row_major);
}

int main()
{

    uint32_t *d_a, *h_a, *d_b, *h_b; // every 8 u4 stored in 1 uint
    int *d_c, *h_c;
    h_c = new int[WMMA_M * WMMA_N];
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

    uint *d_start, *d_stop;
    cudaMallocManaged(reinterpret_cast<void **>(&d_start), 32 * sizeof(int));
    cudaMallocManaged(reinterpret_cast<void **>(&d_stop), 32 * sizeof(int));

    wmma_ker<<<1, 32>>>(d_a, d_b, d_c, d_start, d_stop);

    cudaMemcpy(h_c, d_c, WMMA_M * WMMA_N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Thread" << 0 << " mma_sync starts at " << d_start[0] << ", ends at " << d_stop[0] << "\n";

    std::cout << "immaC=" << std::hex;
    for (int i = 0; i < WMMA_M * WMMA_N; i++)
        std::cout << h_c[i] << ",";
    std::cout << std::endl;

    /*** reference ***/
    int *ref_C = new int[WMMA_M * WMMA_N];
    for (int i = 0; i < WMMA_M; ++i) // 计算参考结果
    {
        for (int j = 0; j < WMMA_N; ++j)
        {
            int result = 0;
            for (int k = 0; k < WMMA_K; ++k)
            {
                // 获取矩阵A和B中对应的整数索引和位偏移
                int indexA = (i * WMMA_K + k) / 8;
                int offsetA = (i * WMMA_K + k) % 8;
                int indexB = (j * WMMA_K + k) / 8;
                int offsetB = (j * WMMA_K + k) % 8;
                // 从矩阵A和B中获取对应的整数值
                int valueA = h_a[indexA];
                int valueB = h_b[indexB];
                // 获取矩阵A和B中对应的4位元素值
                int elementA = (valueA >> (offsetA * 4)) & 0x0F;
                int elementB = (valueB >> (offsetB * 4)) & 0x0F;
                // 进行乘法运算并累加结果
                result += (elementA * elementB);
            }
            ref_C[i * WMMA_N + j] = result;
        }
    }
    std::cout << "ref_C=" << std::hex;
    for (int i = 0; i < WMMA_M * WMMA_N; i++)
        std::cout << ref_C[i] << ",";
    std::cout << std::endl;
}
