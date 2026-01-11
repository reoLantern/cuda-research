// brev.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   BREV R9, R2 ;                          /* 0x0000000200097301 */
        // /* 0x004e280000000000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;           /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int brev_b32(unsigned int a) {
    unsigned int out;
    asm volatile("brev.b32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9b00 */
        // /*00a0*/                   IMAD.WIDE R6, R9, 0x8, R6 ;               /* 0x0000000809067825 */
        // /* 0x008fe200078e0206 */
        // /*00b0*/                   BREV R5, R2 ;                             /* 0x0000000200057301 */
        // /* 0x004ff00000000000 */
        // /*00c0*/                   BREV R4, R3 ;                             /* 0x0000000300047301 */
        // /* 0x000e240000000000 */
        // /*00d0*/                   STG.E.64 desc[UR4][R6.64], R4 ;           /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ unsigned long long brev_b64(unsigned long long a) {
    unsigned long long out;
    asm volatile("brev.b64 %0, %1;" : "=l"(out) : "l"(a));
    return out;
}

extern "C" __global__ void brev_kernel(
    const unsigned int* __restrict__ in_a32,
    const unsigned long long* __restrict__ in_a64,
    unsigned int* __restrict__ out_b32,
    unsigned long long* __restrict__ out_b64
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    unsigned int a32 = in_a32[tid];
    unsigned long long a64 = in_a64[tid];

    out_b32[tid] = brev_b32(a32);
    out_b64[tid] = brev_b64(a64);
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    constexpr int N = 256;

    unsigned int *in_a32;
    unsigned long long *in_a64;
    unsigned int *out_b32;
    unsigned long long *out_b64;

    ck(cudaMallocManaged(&in_a32, N * sizeof(unsigned int)), "cudaMallocManaged in_a32");
    ck(cudaMallocManaged(&in_a64, N * sizeof(unsigned long long)), "cudaMallocManaged in_a64");
    ck(cudaMallocManaged(&out_b32, N * sizeof(unsigned int)), "cudaMallocManaged out_b32");
    ck(cudaMallocManaged(&out_b64, N * sizeof(unsigned long long)), "cudaMallocManaged out_b64");

    for (int i = 0; i < N; ++i) {
        in_a32[i] = (unsigned int)(0x12345678u + i);
        in_a64[i] = (unsigned long long)(0x0123456789abcdefull + i);
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    brev_kernel<<<grid, block>>>(in_a32, in_a64, out_b32, out_b64);

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("brev_b32=%u brev_b64=%llu\n", out_b32[0], out_b64[0]);

    cudaFree(in_a32);
    cudaFree(in_a64);
    cudaFree(out_b32);
    cudaFree(out_b64);
    return 0;
}
