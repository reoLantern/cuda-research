#include <mma.h>
#include <iostream>
#include <iomanip>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 8
#define WMMA_K 8

#define WARP_SIZE 32

__global__ void f64mma1688NaiveKernel(const double *__restrict__ A, const double *__restrict__ B,
                                      double *__restrict__ C)
{

    const uint32_t laneId = threadIdx.x % WARP_SIZE;

    double RA[4];
    double RB[2];
    double RC[4] = {0, 0, 0, 0};

    RA[0] = A[laneId / 4 * WMMA_K + laneId % 4];
    RA[1] = A[8 * WMMA_K + laneId / 4 * WMMA_K + laneId % 4];
    RA[2] = A[laneId / 4 * WMMA_K + laneId % 4 + 4];
    RA[3] = A[8 * WMMA_K + laneId / 4 * WMMA_K + laneId % 4 + 4];

    RB[0] = B[laneId / 4 * WMMA_K + laneId % 4];
    RB[1] = B[8 * WMMA_K + laneId / 4 * WMMA_K + laneId % 4];


    asm volatile("mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=d"(RC[0]), "=d"(RC[1]), "=d"(RC[2]), "=d"(RC[3])
                 : "d"(RA[0]), "d"(RA[1]), "d"(RA[2]), "d"(RA[3]), "d"(RB[0]), "d"(RB[1]), "d"(RC[0]), "d"(RC[1]), "d"(RC[2]), "d"(RC[3]));

    C[laneId / 4 * WMMA_N + laneId % 4 * 2] = RC[0];
    C[laneId / 4 * WMMA_N + laneId % 4 * 2 + 1] = RC[1];
    C[8 * WMMA_N + laneId / 4 * WMMA_N + laneId % 4 * 2] = RC[2];
    C[8 * WMMA_N + laneId / 4 * WMMA_N + laneId % 4 * 2 + 1] = RC[3];
}

int main()
{

    double *d_a, *h_a, *d_b, *h_b; // every 8 u4 stored in 1 uint
    double *d_c, *h_c;
    h_c = new double[WMMA_M * WMMA_N];
    h_b = new double[WMMA_K * WMMA_N];
    h_a = new double[WMMA_M * WMMA_K];
    cudaMalloc(reinterpret_cast<void **>(&d_a), WMMA_M * WMMA_K * sizeof(double));
    cudaMalloc(reinterpret_cast<void **>(&d_b), WMMA_K * WMMA_N * sizeof(double));
    cudaMalloc(reinterpret_cast<void **>(&d_c), WMMA_M * WMMA_N * sizeof(double));

    for (int i = 0; i < WMMA_M * WMMA_K; i++)
    {
        double value = (double)i;
        h_a[i] = value;
    }
    for (int i = 0; i < WMMA_N * WMMA_K; i++)
    {
        double value = (double)i / 4;
        h_b[i] = value;
    }
    std::cout << "A=";
    for (int i = 0; i < WMMA_M * WMMA_K; i++)
        std::cout << h_a[i] << ",";
    std::cout << "\nB=";
    for (int i = 0; i < WMMA_M * WMMA_K; i++)
        std::cout << h_b[i] << ",";
    std::cout << "\n";
    cudaMemcpy(d_a, h_a, WMMA_M * WMMA_K * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, WMMA_K * WMMA_N * sizeof(double), cudaMemcpyHostToDevice);

    f64mma1688NaiveKernel<<<1, 32>>>(d_a, d_b, d_c);

    // cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, WMMA_M * WMMA_N * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "mma_C=";
    for (int i = 0; i < WMMA_M * WMMA_N; i++)
        std::cout << h_c[i] << ",";
    std::cout << std::endl;

    /*** reference ***/
    double *ref_C = new double[WMMA_M * WMMA_N];
    for (int i = 0; i < WMMA_M; ++i) // 计算参考结果
    {
        for (int j = 0; j < WMMA_N; ++j)
        {
            double result = 0;
            for (int k = 0; k < WMMA_K; ++k)
            {
                double elementA = h_a[i * WMMA_K + k];
                double elementB = h_b[j * WMMA_K + k];
                result += (elementA * elementB);
            }
            ref_C[i * WMMA_N + j] = result;
        }
    }
    std::cout << "ref_C=";
    for (int i = 0; i < WMMA_M * WMMA_N; i++)
        std::cout << ref_C[i] << ",";
    std::cout << std::endl;
}
