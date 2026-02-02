// ld_weak_cop_cache_prefetch_vec.cu
//
// Covers:
// ld{.weak}{.ss}{.cop}{.level::cache_hint}{.level::prefetch_size}{.vec}.type d, [a]{.unified}{, cache-policy};

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

        // /*0090*/                   LDG.E R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1e1900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;   /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;  /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
        // /*0090*/                   LDG.E R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1e1900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;   /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;  /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ uint32_t ld_weak_global_u32(const uint32_t* p) {
    uint32_t d;
    asm volatile("ld.weak.global.u32 %0, [%1];" : "=r"(d) : "l"(p));
    return d;
}

        // /*0090*/                   LDG.E.STRONG.SM R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1eb900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;             /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;            /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
        // /*0090*/                   LDG.E.STRONG.SM R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1eb900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;             /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;            /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ uint32_t ld_global_ca_u32(const uint32_t* p) {
    uint32_t d;
    asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(d) : "l"(p));
    return d;
}

        // /*0090*/                   LDG.E.STRONG.GPU R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1ef900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;             /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
        // /*0090*/                   LDG.E.STRONG.GPU R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1ef900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;             /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ uint32_t ld_global_cg_u32(const uint32_t* p) {
    uint32_t d;
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(d) : "l"(p));
    return d;
}

        // /*0090*/                   LDG.E.EF R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c0e1900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;      /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;     /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
        // /*0090*/                   LDG.E.EF R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c0e1900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;      /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;     /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ uint32_t ld_global_cs_u32(const uint32_t* p) {
    uint32_t d;
    asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(d) : "l"(p));
    return d;
}

        // /*0090*/                   LDG.E.LU R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c3e1900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;      /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;     /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
        // /*0090*/                   LDG.E.LU R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c3e1900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;      /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;     /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ uint32_t ld_global_lu_u32(const uint32_t* p) {
    uint32_t d;
    asm volatile("ld.global.lu.u32 %0, [%1];" : "=r"(d) : "l"(p));
    return d;
}

        // /*0090*/                   LDG.E.STRONG.SYS R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;             /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
        // /*0090*/                   LDG.E.STRONG.SYS R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;             /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ uint32_t ld_global_cv_u32(const uint32_t* p) {
    uint32_t d;
    asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(d) : "l"(p));
    return d;
}

        // /*0090*/                   LDG.E.LTC64B R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1e1910 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;          /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;         /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
        // /*0090*/                   LDG.E.LTC64B R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1e1910 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;          /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;         /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ uint32_t ld_global_L2_64B_u32(const uint32_t* p) {
    uint32_t d;
    asm volatile("ld.global.L2::64B.u32 %0, [%1];" : "=r"(d) : "l"(p));
    return d;
}

        // /*0090*/                   LDG.E R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1e1900 */
        // /*00a0*/                   LDCU.64 UR4, c[0x0][0x358] ;  /* 0x00006b00ff0477ac */
        // /* 0x000ea20008000a00 */
        // /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;   /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R3 ;  /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
        // /*0090*/                   LDG.E R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1e1900 */
        // /*00a0*/                   LDCU.64 UR4, c[0x0][0x358] ;  /* 0x00006b00ff0477ac */
        // /* 0x000ea20008000a00 */
        // /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;   /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R3 ;  /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ uint32_t ld_global_cache_hint_policy_u32(const uint32_t* p, uint64_t policy) {
    uint32_t d;
    asm volatile("ld.global.L2::cache_hint.u32 %0, [%1], %2;" : "=r"(d) : "l"(p), "l"(policy));
    return d;
}

        // /*00b0*/                   LDG.E.64 R4, desc[UR4][R4.64] ;               /* 0x0000000404047981 */
        // /* 0x001ea2000c1e1b00 */
        // /*00c0*/                   IADD3 R2, P0, PT, R2, UR10, RZ ;              /* 0x0000000a02027c10 */
        // /* 0x000fc8000ff1e0ff */
        // /*00d0*/                   IADD3.X R3, PT, PT, R3, UR11, RZ, P0, !PT ;   /* 0x0000000b03037c10 */
        // /* 0x000fe400087fe4ff */
        // /*00e0*/                   LOP3.LUT R7, R4, R5, RZ, 0x3c, !PT ;          /* 0x0000000504077212 */
        // /* 0x004fca00078e3cff */
        // /*00f0*/                   STG.E desc[UR4][R2.64], R7 ;                  /* 0x0000000702007986 */
        // /* 0x000fe2000c101904 */
        // /*00b0*/                   LDG.E.64 R4, desc[UR4][R4.64] ;               /* 0x0000000404047981 */
        // /* 0x001ea2000c1e1b00 */
        // /*00c0*/                   IADD3 R2, P0, PT, R2, UR10, RZ ;              /* 0x0000000a02027c10 */
        // /* 0x000fc8000ff1e0ff */
        // /*00d0*/                   IADD3.X R3, PT, PT, R3, UR11, RZ, P0, !PT ;   /* 0x0000000b03037c10 */
        // /* 0x000fe400087fe4ff */
        // /*00e0*/                   LOP3.LUT R7, R4, R5, RZ, 0x3c, !PT ;          /* 0x0000000504077212 */
        // /* 0x004fca00078e3cff */
        // /*00f0*/                   STG.E desc[UR4][R2.64], R7 ;                  /* 0x0000000702007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t ld_global_v2_u32(const uint32_t* p) {
    uint32_t d0, d1;
    asm volatile("ld.global.v2.u32 {%0, %1}, [%2];" : "=r"(d0), "=r"(d1) : "l"(p));
    return d0 ^ d1;
}

        // /*00b0*/                   LDG.E.128 R4, desc[UR4][R4.64] ;              /* 0x0000000404047981 */
        // /* 0x001ea2000c1e1d00 */
        // /*00c0*/                   IADD3 R2, P0, PT, R2, UR10, RZ ;              /* 0x0000000a02027c10 */
        // /* 0x000fc8000ff1e0ff */
        // /*00d0*/                   IADD3.X R3, PT, PT, R3, UR11, RZ, P0, !PT ;   /* 0x0000000b03037c10 */
        // /* 0x000fe400087fe4ff */
        // /*00e0*/                   LOP3.LUT R6, R6, R4, R5, 0x96, !PT ;          /* 0x0000000406067212 */
        // /* 0x004fc800078e9605 */
        // /*00f0*/                   LOP3.LUT R7, R6, R7, RZ, 0x3c, !PT ;          /* 0x0000000706077212 */
        // /* 0x000fca00078e3cff */
        // /*0100*/                   STG.E desc[UR4][R2.64], R7 ;                  /* 0x0000000702007986 */
        // /* 0x000fe2000c101904 */
        // /*00b0*/                   LDG.E.128 R4, desc[UR4][R4.64] ;              /* 0x0000000404047981 */
        // /* 0x001ea2000c1e1d00 */
        // /*00c0*/                   IADD3 R2, P0, PT, R2, UR10, RZ ;              /* 0x0000000a02027c10 */
        // /* 0x000fc8000ff1e0ff */
        // /*00d0*/                   IADD3.X R3, PT, PT, R3, UR11, RZ, P0, !PT ;   /* 0x0000000b03037c10 */
        // /* 0x000fe400087fe4ff */
        // /*00e0*/                   LOP3.LUT R6, R6, R4, R5, 0x96, !PT ;          /* 0x0000000406067212 */
        // /* 0x004fc800078e9605 */
        // /*00f0*/                   LOP3.LUT R7, R6, R7, RZ, 0x3c, !PT ;          /* 0x0000000706077212 */
        // /* 0x000fca00078e3cff */
        // /*0100*/                   STG.E desc[UR4][R2.64], R7 ;                  /* 0x0000000702007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t ld_global_v4_u32(const uint32_t* p) {
    uint32_t d0, d1, d2, d3;
    asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "l"(p));
    return d0 ^ d1 ^ d2 ^ d3;
}

        // /*00b0*/                   LDG.E.ENL2.256 R4, R8, desc[UR4][R4.64] ;     /* 0xfe0000040408797e */
        // /* 0x001ea20008121904 */
        // /*00c0*/                   IADD3 R2, P0, PT, R2, UR10, RZ ;              /* 0x0000000a02027c10 */
        // /* 0x000fc8000ff1e0ff */
        // /*00d0*/                   IADD3.X R3, PT, PT, R3, UR11, RZ, P0, !PT ;   /* 0x0000000b03037c10 */
        // /* 0x000fe400087fe4ff */
        // /*00e0*/                   LOP3.LUT R8, R10, R8, R9, 0x96, !PT ;         /* 0x000000080a087212 */
        // /* 0x004fc800078e9609 */
        // /*00f0*/                   LOP3.LUT R8, R4, R8, R11, 0x96, !PT ;         /* 0x0000000804087212 */
        // /* 0x000fc800078e960b */
        // /*0100*/                   LOP3.LUT R6, R6, R8, R5, 0x96, !PT ;          /* 0x0000000806067212 */
        // /* 0x000fc800078e9605 */
        // /*0110*/                   LOP3.LUT R7, R6, R7, RZ, 0x3c, !PT ;          /* 0x0000000706077212 */
        // /* 0x000fca00078e3cff */
        // /*0120*/                   STG.E desc[UR4][R2.64], R7 ;                  /* 0x0000000702007986 */
        // /* 0x000fe2000c101904 */
        // /*00b0*/                   LDG.E.ENL2.256 R4, R8, desc[UR4][R4.64] ;     /* 0xfe0000040408797e */
        // /* 0x001ea20008121904 */
        // /*00c0*/                   IADD3 R2, P0, PT, R2, UR10, RZ ;              /* 0x0000000a02027c10 */
        // /* 0x000fc8000ff1e0ff */
        // /*00d0*/                   IADD3.X R3, PT, PT, R3, UR11, RZ, P0, !PT ;   /* 0x0000000b03037c10 */
        // /* 0x000fe400087fe4ff */
        // /*00e0*/                   LOP3.LUT R8, R10, R8, R9, 0x96, !PT ;         /* 0x000000080a087212 */
        // /* 0x004fc800078e9609 */
        // /*00f0*/                   LOP3.LUT R8, R4, R8, R11, 0x96, !PT ;         /* 0x0000000804087212 */
        // /* 0x000fc800078e960b */
        // /*0100*/                   LOP3.LUT R6, R6, R8, R5, 0x96, !PT ;          /* 0x0000000806067212 */
        // /* 0x000fc800078e9605 */
        // /*0110*/                   LOP3.LUT R7, R6, R7, RZ, 0x3c, !PT ;          /* 0x0000000706077212 */
        // /* 0x000fca00078e3cff */
        // /*0120*/                   STG.E desc[UR4][R2.64], R7 ;                  /* 0x0000000702007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t ld_global_v8_u32(const uint32_t* p) {
    uint32_t d0, d1, d2, d3, d4, d5, d6, d7;
    asm volatile("ld.global.v8.u32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3),
                   "=r"(d4), "=r"(d5), "=r"(d6), "=r"(d7)
                 : "l"(p));
    return d0 ^ d1 ^ d2 ^ d3 ^ d4 ^ d5 ^ d6 ^ d7;
}

extern "C" __global__ void ld_weak_cop_cache_prefetch_vec_kernel(
    const uint32_t* __restrict__ in,
    uint32_t* __restrict__ out,
    uint64_t policy) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t* p = in + tid;

    const uint32_t* p2 = in + (tid & ~1);
    const uint32_t* p4 = in + (tid & ~3);
    const uint32_t* p8 = in + (tid & ~7);

    uint32_t acc = 0;
    acc ^= ld_weak_global_u32(p);
    acc ^= ld_global_ca_u32(p);
    acc ^= ld_global_cg_u32(p);
    acc ^= ld_global_cs_u32(p);
    acc ^= ld_global_lu_u32(p);
    acc ^= ld_global_cv_u32(p);
    acc ^= ld_global_L2_64B_u32(p);
    acc ^= ld_global_cache_hint_policy_u32(p, policy);
    acc ^= ld_global_v2_u32(p2);
    acc ^= ld_global_v4_u32(p4);
    acc ^= ld_global_v8_u32(p8);

    out[tid] = acc;
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    constexpr int N = 256;

    uint32_t* in;
    uint32_t* out;

    ck(cudaMallocManaged(&in, N * sizeof(uint32_t)), "cudaMallocManaged in");
    ck(cudaMallocManaged(&out, N * sizeof(uint32_t)), "cudaMallocManaged out");

    for (int i = 0; i < N; ++i) {
        in[i] = (uint32_t)((i * 13 + 7) ^ 0x5a5ac33cu);
        out[i] = 0u;
    }

    uint64_t policy = 0ull;

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    ld_weak_cop_cache_prefetch_vec_kernel<<<grid, block>>>(in, out, policy);

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("acc=%u\n", (unsigned)out[0]);

    cudaFree(in);
    cudaFree(out);
    return 0;
}
