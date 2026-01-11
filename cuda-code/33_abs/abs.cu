// abs.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   IABS R0, R2 ;                          /* 0x0000000200007213 */
        // /* 0x004fc80000000000 */
        // /*00c0*/                   MOV R7, R0 ;                           /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;           /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int abs_s32(int a) {
    int out;
    asm volatile("abs.s32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.CONSTANT R2, desc[UR4][R2.64] ;           /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9b00 */
        // /*00a0*/                   IMAD.WIDE R4, R9, 0x8, R4 ;                        /* 0x0000000809047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   IADD3 R7, P1, PT, RZ, -R2.reuse, RZ ;              /* 0x80000002ff077210 */
        // /* 0x084fe40007f3e0ff */
        // /*00c0*/                   ISETP.LT.AND P0, PT, R3, RZ, PT ;                  /* 0x000000ff0300720c */
        // /* 0x000fe40003f01270 */
        // /*00d0*/                   IADD3.X R11, PT, PT, RZ, ~R3.reuse, RZ, P1, !PT ;  /* 0x80000003ff0b7210 */
        // /* 0x080fe40000ffe4ff */
        // /*00e0*/                   SEL R6, R7, R2, P0 ;                               /* 0x0000000207067207 */
        // /* 0x000fe40000000000 */
        // /*00f0*/                   SEL R7, R11, R3, P0 ;                              /* 0x000000030b077207 */
        // /* 0x000fca0000000000 */
        // /*0100*/                   STG.E.64 desc[UR4][R4.64], R6 ;                    /* 0x0000000604007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ long long abs_s64(long long a) {
    long long out;
    asm volatile("abs.s64 %0, %1;" : "=l"(out) : "l"(a));
    return out;
}

extern "C" __global__ void abs_kernel(
    const int* __restrict__ in_a32,
    const long long* __restrict__ in_a64,
    int* __restrict__ out_s32,
    long long* __restrict__ out_s64
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    int a32 = in_a32[tid];
    long long a64 = in_a64[tid];

    out_s32[tid] = abs_s32(a32);
    out_s64[tid] = abs_s64(a64);
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    constexpr int N = 256;

    int *in_a32;
    long long *in_a64;
    int *out_s32;
    long long *out_s64;

    ck(cudaMallocManaged(&in_a32, N * sizeof(int)), "cudaMallocManaged in_a32");
    ck(cudaMallocManaged(&in_a64, N * sizeof(long long)), "cudaMallocManaged in_a64");
    ck(cudaMallocManaged(&out_s32, N * sizeof(int)), "cudaMallocManaged out_s32");
    ck(cudaMallocManaged(&out_s64, N * sizeof(long long)), "cudaMallocManaged out_s64");

    for (int i = 0; i < N; ++i) {
        in_a32[i] = (i & 1) ? -(i * 3 + 1) : (i * 3 + 1);
        in_a64[i] = (i & 1) ? -(long long)(i * 5 + 7) : (long long)(i * 5 + 7);
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    abs_kernel<<<grid, block>>>(in_a32, in_a64, out_s32, out_s64);

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("abs_s32=%d abs_s64=%lld\n", out_s32[0], out_s64[0]);

    cudaFree(in_a32);
    cudaFree(in_a64);
    cudaFree(out_s32);
    cudaFree(out_s64);
    return 0;
}
