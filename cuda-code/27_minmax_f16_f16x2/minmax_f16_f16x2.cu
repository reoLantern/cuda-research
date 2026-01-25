// minmax_f16_f16x2.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

//         /*00a0*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
//                                                                               /* 0x002ea2000c1e9500 */
//         /*00b0*/                   IMAD.WIDE R4, R9, 0x2, R4 ;                /* 0x0000000209047825 */
//                                                                               /* 0x008fcc00078e0204 */
//         /*00c0*/                   LDG.E.U16.CONSTANT R5, desc[UR4][R4.64] ;  /* 0x0000000404057981 */
//                                                                               /* 0x000ea2000c1e9500 */
//         /*00d0*/                   IMAD.WIDE R6, R9, 0x2, R6 ;                /* 0x0000000209067825 */
//                                                                               /* 0x001fe200078e0206 */
//         /*00e0*/                   HMNMX2 R5, R2.H0_H0, R5.H0_H0, PT ;        /* 0x2000000502057240 */
//                                                                               /* 0x004fca0003800800 */
//         /*00f0*/                   STG.E.U16 desc[UR4][R6.64], R5 ;           /* 0x0000000506007986 */
__device__ __forceinline__ __half min_f16(__half a, __half b) {
    __half_raw ar = static_cast<__half_raw>(a);
    __half_raw br = static_cast<__half_raw>(b);
    __half_raw out;
    asm volatile("min.f16 %0, %1, %2;" : "=h"(out.x) : "h"(ar.x), "h"(br.x));
    return __half(out);
}

//         /*00a0*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
//                                                                               /* 0x002ea2000c1e9500 */
//         /*00b0*/                   IMAD.WIDE R4, R9, 0x2, R4 ;                /* 0x0000000209047825 */
//                                                                               /* 0x008fcc00078e0204 */
//         /*00c0*/                   LDG.E.U16.CONSTANT R5, desc[UR4][R4.64] ;  /* 0x0000000404057981 */
//                                                                               /* 0x000ea2000c1e9500 */
//         /*00d0*/                   IMAD.WIDE R6, R9, 0x2, R6 ;                /* 0x0000000209067825 */
//                                                                               /* 0x001fe200078e0206 */
//         /*00e0*/                   HMNMX2 R5, R2.H0_H0, R5.H0_H0, !PT ;       /* 0x2000000502057240 */
//                                                                               /* 0x004fca0007800800 */
//         /*00f0*/                   STG.E.U16 desc[UR4][R6.64], R5 ;           /* 0x0000000506007986 */
__device__ __forceinline__ __half max_f16(__half a, __half b) {
    __half_raw ar = static_cast<__half_raw>(a);
    __half_raw br = static_cast<__half_raw>(b);
    __half_raw out;
    asm volatile("max.f16 %0, %1, %2;" : "=h"(out.x) : "h"(ar.x), "h"(br.x));
    return __half(out);
}

//         /*00a0*/                   LDG.E R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
//                                                                  /* 0x002ea2000c1e1900 */
//         /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;   /* 0x0000000409047825 */
//                                                                  /* 0x008fcc00078e0204 */
//         /*00c0*/                   LDG.E R5, desc[UR4][R4.64] ;  /* 0x0000000404057981 */
//                                                                  /* 0x000ea2000c1e1900 */
//         /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;   /* 0x0000000409067825 */
//                                                                  /* 0x001fe200078e0206 */
//         /*00e0*/                   HMNMX2 R9, R2, R5, PT ;       /* 0x0000000502097240 */
//                                                                  /* 0x004fca0003800000 */
//         /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;  /* 0x0000000906007986 */
__device__ __forceinline__ uint32_t min_f16x2(__half2 a, __half2 b) {
    __half2_raw ar = static_cast<__half2_raw>(a);
    __half2_raw br = static_cast<__half2_raw>(b);
    uint32_t a_bits = (uint32_t)ar.x | ((uint32_t)ar.y << 16);
    uint32_t b_bits = (uint32_t)br.x | ((uint32_t)br.y << 16);
    uint32_t out_bits;
    asm volatile("min.f16x2 %0, %1, %2;" : "=r"(out_bits) : "r"(a_bits), "r"(b_bits));
    return out_bits;
}

//         /*00a0*/                   LDG.E R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
//                                                                  /* 0x002ea2000c1e1900 */
//         /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;   /* 0x0000000409047825 */
//                                                                  /* 0x008fcc00078e0204 */
//         /*00c0*/                   LDG.E R5, desc[UR4][R4.64] ;  /* 0x0000000404057981 */
//                                                                  /* 0x000ea2000c1e1900 */
//         /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;   /* 0x0000000409067825 */
//                                                                  /* 0x001fe200078e0206 */
//         /*00e0*/                   HMNMX2 R9, R2, R5, !PT ;      /* 0x0000000502097240 */
//                                                                  /* 0x004fca0007800000 */
//         /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;  /* 0x0000000906007986 */
__device__ __forceinline__ uint32_t max_f16x2(__half2 a, __half2 b) {
    __half2_raw ar = static_cast<__half2_raw>(a);
    __half2_raw br = static_cast<__half2_raw>(b);
    uint32_t a_bits = (uint32_t)ar.x | ((uint32_t)ar.y << 16);
    uint32_t b_bits = (uint32_t)br.x | ((uint32_t)br.y << 16);
    uint32_t out_bits;
    asm volatile("max.f16x2 %0, %1, %2;" : "=r"(out_bits) : "r"(a_bits), "r"(b_bits));
    return out_bits;
}

//         /*00a0*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;    /* 0x0000000402027981 */
//                                                                                 /* 0x002ea2000c1e9500 */
//         /*00b0*/                   IMAD.WIDE R4, R9, 0x2, R4 ;                  /* 0x0000000209047825 */
//                                                                                 /* 0x008fcc00078e0204 */
//         /*00c0*/                   LDG.E.U16.CONSTANT R5, desc[UR4][R4.64] ;    /* 0x0000000404057981 */
//                                                                                 /* 0x000ea2000c1e9500 */
//         /*00d0*/                   IMAD.WIDE R6, R9, 0x2, R6 ;                  /* 0x0000000209067825 */
//                                                                                 /* 0x001fe200078e0206 */
//         /*00e0*/                   HMNMX2.BF16_V2 R5, R2.H0_H0, R5.H0_H0, PT ;  /* 0x2000000502057240 */
//                                                                                 /* 0x004fca0003a00800 */
//         /*00f0*/                   STG.E.U16 desc[UR4][R6.64], R5 ;             /* 0x0000000506007986 */
__device__ __forceinline__ __nv_bfloat16 min_bf16(__nv_bfloat16 a, __nv_bfloat16 b) {
    __nv_bfloat16_raw ar = static_cast<__nv_bfloat16_raw>(a);
    __nv_bfloat16_raw br = static_cast<__nv_bfloat16_raw>(b);
    __nv_bfloat16_raw out;
    asm volatile("min.bf16 %0, %1, %2;" : "=h"(out.x) : "h"(ar.x), "h"(br.x));
    return __nv_bfloat16(out);
}

//         /*00a0*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;     /* 0x0000000402027981 */
//                                                                                  /* 0x002ea2000c1e9500 */
//         /*00b0*/                   IMAD.WIDE R4, R9, 0x2, R4 ;                   /* 0x0000000209047825 */
//                                                                                  /* 0x008fcc00078e0204 */
//         /*00c0*/                   LDG.E.U16.CONSTANT R5, desc[UR4][R4.64] ;     /* 0x0000000404057981 */
//                                                                                  /* 0x000ea2000c1e9500 */
//         /*00d0*/                   IMAD.WIDE R6, R9, 0x2, R6 ;                   /* 0x0000000209067825 */
//                                                                                  /* 0x001fe200078e0206 */
//         /*00e0*/                   HMNMX2.BF16_V2 R5, R2.H0_H0, R5.H0_H0, !PT ;  /* 0x2000000502057240 */
//                                                                                  /* 0x004fca0007a00800 */
//         /*00f0*/                   STG.E.U16 desc[UR4][R6.64], R5 ;              /* 0x0000000506007986 */
__device__ __forceinline__ __nv_bfloat16 max_bf16(__nv_bfloat16 a, __nv_bfloat16 b) {
    __nv_bfloat16_raw ar = static_cast<__nv_bfloat16_raw>(a);
    __nv_bfloat16_raw br = static_cast<__nv_bfloat16_raw>(b);
    __nv_bfloat16_raw out;
    asm volatile("max.bf16 %0, %1, %2;" : "=h"(out.x) : "h"(ar.x), "h"(br.x));
    return __nv_bfloat16(out);
}

//         /*00a0*/                   LDG.E R2, desc[UR4][R2.64] ;     /* 0x0000000402027981 */
//                                                                     /* 0x002ea2000c1e1900 */
//         /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;      /* 0x0000000409047825 */
//                                                                     /* 0x008fcc00078e0204 */
//         /*00c0*/                   LDG.E R5, desc[UR4][R4.64] ;     /* 0x0000000404057981 */
//                                                                     /* 0x000ea2000c1e1900 */
//         /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;      /* 0x0000000409067825 */
//                                                                     /* 0x001fe200078e0206 */
//         /*00e0*/                   HMNMX2.BF16_V2 R9, R2, R5, PT ;  /* 0x0000000502097240 */
//                                                                     /* 0x004fca0003a00000 */
//         /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;     /* 0x0000000906007986 */
__device__ __forceinline__ uint32_t min_bf16x2(__nv_bfloat162 a, __nv_bfloat162 b) {
    __nv_bfloat162_raw ar = static_cast<__nv_bfloat162_raw>(a);
    __nv_bfloat162_raw br = static_cast<__nv_bfloat162_raw>(b);
    uint32_t a_bits = (uint32_t)ar.x | ((uint32_t)ar.y << 16);
    uint32_t b_bits = (uint32_t)br.x | ((uint32_t)br.y << 16);
    uint32_t out_bits;
    asm volatile("min.bf16x2 %0, %1, %2;" : "=r"(out_bits) : "r"(a_bits), "r"(b_bits));
    return out_bits;
}

//         /*00a0*/                   LDG.E R2, desc[UR4][R2.64] ;      /* 0x0000000402027981 */
//                                                                      /* 0x002ea2000c1e1900 */
//         /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;       /* 0x0000000409047825 */
//                                                                      /* 0x008fcc00078e0204 */
//         /*00c0*/                   LDG.E R5, desc[UR4][R4.64] ;      /* 0x0000000404057981 */
//                                                                      /* 0x000ea2000c1e1900 */
//         /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;       /* 0x0000000409067825 */
//                                                                      /* 0x001fe200078e0206 */
//         /*00e0*/                   HMNMX2.BF16_V2 R9, R2, R5, !PT ;  /* 0x0000000502097240 */
//                                                                      /* 0x004fca0007a00000 */
//         /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;      /* 0x0000000906007986 */
__device__ __forceinline__ uint32_t max_bf16x2(__nv_bfloat162 a, __nv_bfloat162 b) {
    __nv_bfloat162_raw ar = static_cast<__nv_bfloat162_raw>(a);
    __nv_bfloat162_raw br = static_cast<__nv_bfloat162_raw>(b);
    uint32_t a_bits = (uint32_t)ar.x | ((uint32_t)ar.y << 16);
    uint32_t b_bits = (uint32_t)br.x | ((uint32_t)br.y << 16);
    uint32_t out_bits;
    asm volatile("max.bf16x2 %0, %1, %2;" : "=r"(out_bits) : "r"(a_bits), "r"(b_bits));
    return out_bits;
}

extern "C" __global__ void minmax_f16_bf16_kernel(
    const __half* __restrict__ in_f16_a,
    const __half* __restrict__ in_f16_b,
    const __half2* __restrict__ in_f16x2_a,
    const __half2* __restrict__ in_f16x2_b,
    const __nv_bfloat16* __restrict__ in_bf16_a,
    const __nv_bfloat16* __restrict__ in_bf16_b,
    const __nv_bfloat162* __restrict__ in_bf16x2_a,
    const __nv_bfloat162* __restrict__ in_bf16x2_b,
    __half* __restrict__ out_min_f16,
    __half* __restrict__ out_max_f16,
    uint32_t* __restrict__ out_min_f16x2,
    uint32_t* __restrict__ out_max_f16x2,
    __nv_bfloat16* __restrict__ out_min_bf16,
    __nv_bfloat16* __restrict__ out_max_bf16,
    uint32_t* __restrict__ out_min_bf16x2,
    uint32_t* __restrict__ out_max_bf16x2
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    __half a_h = in_f16_a[tid];
    __half b_h = in_f16_b[tid];
    __half2 a_h2 = in_f16x2_a[tid];
    __half2 b_h2 = in_f16x2_b[tid];
    __nv_bfloat16 a_bf = in_bf16_a[tid];
    __nv_bfloat16 b_bf = in_bf16_b[tid];
    __nv_bfloat162 a_bf2 = in_bf16x2_a[tid];
    __nv_bfloat162 b_bf2 = in_bf16x2_b[tid];

    out_min_f16[tid] = min_f16(a_h, b_h);
    // out_max_f16[tid] = max_f16(a_h, b_h);
    // out_min_f16x2[tid] = min_f16x2(a_h2, b_h2);
    // out_max_f16x2[tid] = max_f16x2(a_h2, b_h2);
    // out_min_bf16[tid] = min_bf16(a_bf, b_bf);
    // out_max_bf16[tid] = max_bf16(a_bf, b_bf);
    // out_min_bf16x2[tid] = min_bf16x2(a_bf2, b_bf2);
    // out_max_bf16x2[tid] = max_bf16x2(a_bf2, b_bf2);
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    constexpr int N = 256;

    __half *in_f16_a, *in_f16_b, *out_min_f16, *out_max_f16;
    __half2 *in_f16x2_a, *in_f16x2_b;
    uint32_t *out_min_f16x2, *out_max_f16x2;
    __nv_bfloat16 *in_bf16_a, *in_bf16_b, *out_min_bf16, *out_max_bf16;
    __nv_bfloat162 *in_bf16x2_a, *in_bf16x2_b;
    uint32_t *out_min_bf16x2, *out_max_bf16x2;

    ck(cudaMallocManaged(&in_f16_a, N * sizeof(__half)), "cudaMallocManaged in_f16_a");
    ck(cudaMallocManaged(&in_f16_b, N * sizeof(__half)), "cudaMallocManaged in_f16_b");
    ck(cudaMallocManaged(&in_f16x2_a, N * sizeof(__half2)), "cudaMallocManaged in_f16x2_a");
    ck(cudaMallocManaged(&in_f16x2_b, N * sizeof(__half2)), "cudaMallocManaged in_f16x2_b");
    ck(cudaMallocManaged(&in_bf16_a, N * sizeof(__nv_bfloat16)), "cudaMallocManaged in_bf16_a");
    ck(cudaMallocManaged(&in_bf16_b, N * sizeof(__nv_bfloat16)), "cudaMallocManaged in_bf16_b");
    ck(cudaMallocManaged(&in_bf16x2_a, N * sizeof(__nv_bfloat162)), "cudaMallocManaged in_bf16x2_a");
    ck(cudaMallocManaged(&in_bf16x2_b, N * sizeof(__nv_bfloat162)), "cudaMallocManaged in_bf16x2_b");

    ck(cudaMallocManaged(&out_min_f16, N * sizeof(__half)), "cudaMallocManaged out_min_f16");
    ck(cudaMallocManaged(&out_max_f16, N * sizeof(__half)), "cudaMallocManaged out_max_f16");
    ck(cudaMallocManaged(&out_min_f16x2, N * sizeof(uint32_t)), "cudaMallocManaged out_min_f16x2");
    ck(cudaMallocManaged(&out_max_f16x2, N * sizeof(uint32_t)), "cudaMallocManaged out_max_f16x2");
    ck(cudaMallocManaged(&out_min_bf16, N * sizeof(__nv_bfloat16)), "cudaMallocManaged out_min_bf16");
    ck(cudaMallocManaged(&out_max_bf16, N * sizeof(__nv_bfloat16)), "cudaMallocManaged out_max_bf16");
    ck(cudaMallocManaged(&out_min_bf16x2, N * sizeof(uint32_t)), "cudaMallocManaged out_min_bf16x2");
    ck(cudaMallocManaged(&out_max_bf16x2, N * sizeof(uint32_t)), "cudaMallocManaged out_max_bf16x2");

    for (int i = 0; i < N; ++i) {
        float fa = (float)(i) * 0.25f + 1.0f;
        float fb = (float)(i) * -0.50f + 2.0f;

        in_f16_a[i] = __float2half(fa);
        in_f16_b[i] = __float2half(fb);
        in_f16x2_a[i] = __floats2half2_rn(fa, -fa);
        in_f16x2_b[i] = __floats2half2_rn(fb, -fb);

        in_bf16_a[i] = __float2bfloat16(fa);
        in_bf16_b[i] = __float2bfloat16(fb);
        in_bf16x2_a[i] = __floats2bfloat162_rn(fa, -fa);
        in_bf16x2_b[i] = __floats2bfloat162_rn(fb, -fb);
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    minmax_f16_bf16_kernel<<<grid, block>>>(
        in_f16_a, in_f16_b,
        in_f16x2_a, in_f16x2_b,
        in_bf16_a, in_bf16_b,
        in_bf16x2_a, in_bf16x2_b,
        out_min_f16, out_max_f16,
        out_min_f16x2, out_max_f16x2,
        out_min_bf16, out_max_bf16,
        out_min_bf16x2, out_max_bf16x2
    );

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf(
        "min_f16=%f max_f16=%f min_bf16=%f max_bf16=%f\n",
        __half2float(out_min_f16[0]),
        __half2float(out_max_f16[0]),
        __bfloat162float(out_min_bf16[0]),
        __bfloat162float(out_max_bf16[0])
    );

    cudaFree(in_f16_a);
    cudaFree(in_f16_b);
    cudaFree(in_f16x2_a);
    cudaFree(in_f16x2_b);
    cudaFree(in_bf16_a);
    cudaFree(in_bf16_b);
    cudaFree(in_bf16x2_a);
    cudaFree(in_bf16x2_b);

    cudaFree(out_min_f16);
    cudaFree(out_max_f16);
    cudaFree(out_min_f16x2);
    cudaFree(out_max_f16x2);
    cudaFree(out_min_bf16);
    cudaFree(out_max_bf16);
    cudaFree(out_min_bf16x2);
    cudaFree(out_max_bf16x2);
    return 0;
}
