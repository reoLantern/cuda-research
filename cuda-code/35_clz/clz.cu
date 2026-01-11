// clz.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FLO.U32 R0, R2 ;                       /* 0x0000000200007300 */
        // /* 0x004e2400000e0000 */
        // /*00c0*/                   IADD3 R7, PT, PT, -R0, 0x1f, RZ ;      /* 0x0000001f00077810 */
        // /* 0x001fca0007ffe1ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;           /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int clz_b32(unsigned int a) {
    unsigned int out;
    asm volatile("clz.b32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;               /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   ISETP.NE.U32.AND P0, PT, R3, RZ, PT ;     /* 0x000000ff0300720c */
        // /* 0x004fc80003f05070 */
        // /*00c0*/                   SEL R0, R2, R3, !P0 ;                     /* 0x0000000302007207 */
        // /* 0x000fcc0004000000 */
        // /*00d0*/                   FLO.U32 R0, R0 ;                          /* 0x0000000000007300 */
        // /* 0x000e2400000e0000 */
        // /*00e0*/                   IADD3 R6, PT, PT, -R0.reuse, 0x1f, RZ ;   /* 0x0000001f00067810 */
        // /* 0x041fe40007ffe1ff */
        // /*00f0*/                   IADD3 R9, PT, PT, -R0, 0x3f, RZ ;         /* 0x0000003f00097810 */
        // /* 0x000fe40007ffe1ff */
        // /*0100*/               @P0 IADD3 R9, PT, PT, R6, RZ, RZ ;            /* 0x000000ff06090210 */
        // /* 0x000fca0007ffe0ff */
        // /*0110*/                   STG.E desc[UR4][R4.64], R9 ;              /* 0x0000000904007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int clz_b64(unsigned long long a) {
    unsigned int out;
    asm volatile("clz.b64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

extern "C" __global__ void clz_kernel(
    const unsigned int* __restrict__ in_a32,
    const unsigned long long* __restrict__ in_a64,
    unsigned int* __restrict__ out_b32,
    unsigned int* __restrict__ out_b64
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    unsigned int a32 = in_a32[tid];
    unsigned long long a64 = in_a64[tid];

    out_b32[tid] = clz_b32(a32);
    out_b64[tid] = clz_b64(a64);
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
    unsigned int *out_b64;

    ck(cudaMallocManaged(&in_a32, N * sizeof(unsigned int)), "cudaMallocManaged in_a32");
    ck(cudaMallocManaged(&in_a64, N * sizeof(unsigned long long)), "cudaMallocManaged in_a64");
    ck(cudaMallocManaged(&out_b32, N * sizeof(unsigned int)), "cudaMallocManaged out_b32");
    ck(cudaMallocManaged(&out_b64, N * sizeof(unsigned int)), "cudaMallocManaged out_b64");

    for (int i = 0; i < N; ++i) {
        in_a32[i] = (unsigned int)(1u << (i % 31));
        in_a64[i] = (unsigned long long)(1ull << (i % 63));
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    clz_kernel<<<grid, block>>>(in_a32, in_a64, out_b32, out_b64);

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("clz_b32=%u clz_b64=%u\n", out_b32[0], out_b64[0]);

    cudaFree(in_a32);
    cudaFree(in_a64);
    cudaFree(out_b32);
    cudaFree(out_b64);
    return 0;
}
