// szext.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

        // /*00a0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;            /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;  /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;            /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   SGXT R9, R2, R5 ;                      /* 0x000000050209721a */
        // /* 0x004fca0000000200 */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;           /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int szext_clamp_s32(int a, unsigned int b) {
    int out;
    asm volatile("szext.clamp.s32 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;            /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;  /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;            /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   SGXT.W R9, R2, R5 ;                    /* 0x000000050209721a */
        // /* 0x004fca0000000a00 */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;           /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int szext_wrap_s32(int a, unsigned int b) {
    int out;
    asm volatile("szext.wrap.s32 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;            /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;  /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;            /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   SGXT.U32 R9, R2, R5 ;                  /* 0x000000050209721a */
        // /* 0x004fca0000000000 */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;           /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int szext_clamp_u32(unsigned int a, unsigned int b) {
    unsigned int out;
    asm volatile("szext.clamp.u32 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;            /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;  /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;            /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   SGXT.W.U32 R9, R2, R5 ;                /* 0x000000050209721a */
        // /* 0x004fca0000000800 */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;           /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int szext_wrap_u32(unsigned int a, unsigned int b) {
    unsigned int out;
    asm volatile("szext.wrap.u32 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
}

extern "C" __global__ void szext_kernel(
    const int* __restrict__ in_s32,
    const unsigned int* __restrict__ in_u32,
    const unsigned int* __restrict__ in_bits,
    int* __restrict__ out_s32_clamp,
    int* __restrict__ out_s32_wrap,
    unsigned int* __restrict__ out_u32_clamp,
    unsigned int* __restrict__ out_u32_wrap
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    int s32 = in_s32[tid];
    unsigned int u32 = in_u32[tid];
    unsigned int bits = in_bits[tid];

    out_s32_clamp[tid] = szext_clamp_s32(s32, bits);
    out_s32_wrap[tid] = szext_wrap_s32(s32, bits);
    out_u32_clamp[tid] = szext_clamp_u32(u32, bits);
    out_u32_wrap[tid] = szext_wrap_u32(u32, bits);
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    constexpr int N = 256;

    int *in_s32;
    unsigned int *in_u32;
    unsigned int *in_bits;
    int *out_s32_clamp;
    int *out_s32_wrap;
    unsigned int *out_u32_clamp;
    unsigned int *out_u32_wrap;

    ck(cudaMallocManaged(&in_s32, N * sizeof(int)), "cudaMallocManaged in_s32");
    ck(cudaMallocManaged(&in_u32, N * sizeof(unsigned int)), "cudaMallocManaged in_u32");
    ck(cudaMallocManaged(&in_bits, N * sizeof(unsigned int)), "cudaMallocManaged in_bits");

    ck(cudaMallocManaged(&out_s32_clamp, N * sizeof(int)), "cudaMallocManaged out_s32_clamp");
    ck(cudaMallocManaged(&out_s32_wrap, N * sizeof(int)), "cudaMallocManaged out_s32_wrap");
    ck(cudaMallocManaged(&out_u32_clamp, N * sizeof(unsigned int)), "cudaMallocManaged out_u32_clamp");
    ck(cudaMallocManaged(&out_u32_wrap, N * sizeof(unsigned int)), "cudaMallocManaged out_u32_wrap");

    for (int i = 0; i < N; ++i) {
        in_s32[i] = (i & 1) ? -(i * 3 + 1) : (i * 3 + 1);
        in_u32[i] = (unsigned int)(0xabcdef01u + i);
        in_bits[i] = (unsigned int)(i % 40);
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    szext_kernel<<<grid, block>>>(
        in_s32, in_u32, in_bits,
        out_s32_clamp, out_s32_wrap,
        out_u32_clamp, out_u32_wrap
    );

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("szext_s32_clamp=%d szext_u32_clamp=%u\n", out_s32_clamp[0], out_u32_clamp[0]);

    cudaFree(in_s32);
    cudaFree(in_u32);
    cudaFree(in_bits);
    cudaFree(out_s32_clamp);
    cudaFree(out_s32_wrap);
    cudaFree(out_u32_clamp);
    cudaFree(out_u32_wrap);
    return 0;
}
