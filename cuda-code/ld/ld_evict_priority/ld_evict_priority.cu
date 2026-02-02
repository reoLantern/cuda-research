// ld_evict_priority.cu
//
// Covers:
// ld{.weak}{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type d, [a]{.unified}{, cache-policy};

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

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
__device__ __forceinline__ uint32_t ld_global_L1_evict_first_u32(const uint32_t* p) {
    uint32_t d;
    asm volatile("ld.global.L1::evict_first.u32 %0, [%1];" : "=r"(d) : "l"(p));
    return d;
}

        // /*0090*/                   LDG.E.EL R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c2e1900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;      /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;     /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
        // /*0090*/                   LDG.E.EL R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c2e1900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;      /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;     /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ uint32_t ld_global_L1_evict_last_u32(const uint32_t* p) {
    uint32_t d;
    asm volatile("ld.global.L1::evict_last.u32 %0, [%1];" : "=r"(d) : "l"(p));
    return d;
}

        // /*0090*/                   LDG.E.NA R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c5e1900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;      /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;     /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
        // /*0090*/                   LDG.E.NA R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c5e1900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;      /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;     /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ uint32_t ld_global_L1_no_allocate_u32(const uint32_t* p) {
    uint32_t d;
    asm volatile("ld.global.L1::no_allocate.u32 %0, [%1];" : "=r"(d) : "l"(p));
    return d;
}

        // /*00b0*/                   LDG.E.EF.ELL2.256 R4, R8, desc[UR4][R4.64] ;  /* 0xfe0000040408797e */
        // /* 0x001ea20008041904 */
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
        // /*00b0*/                   LDG.E.EF.ELL2.256 R4, R8, desc[UR4][R4.64] ;  /* 0xfe0000040408797e */
        // /* 0x001ea20008041904 */
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
__device__ __forceinline__ uint32_t ld_global_L1_evict_first_L2_evict_last_v8_u32(const uint32_t* p) {
    uint32_t d0, d1, d2, d3, d4, d5, d6, d7;
    asm volatile("ld.global.L1::evict_first.L2::evict_last.v8.u32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3),
                   "=r"(d4), "=r"(d5), "=r"(d6), "=r"(d7)
                 : "l"(p));
    return d0 ^ d1 ^ d2 ^ d3 ^ d4 ^ d5 ^ d6 ^ d7;
}

extern "C" __global__ void ld_evict_priority_kernel(
    const uint32_t* __restrict__ in,
    uint32_t* __restrict__ out) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t* p = in + tid;
    const uint32_t* p8 = in + (tid & ~7);

    uint32_t acc = 0;
    acc ^= ld_global_L1_evict_first_u32(p);
    acc ^= ld_global_L1_evict_last_u32(p);
    acc ^= ld_global_L1_no_allocate_u32(p);
    acc ^= ld_global_L1_evict_first_L2_evict_last_v8_u32(p8);

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
        in[i] = (uint32_t)((i * 17 + 1) ^ 0x3c3ca5a5u);
        out[i] = 0u;
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    ld_evict_priority_kernel<<<grid, block>>>(in, out);

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("acc=%u\n", (unsigned)out[0]);

    cudaFree(in);
    cudaFree(out);
    return 0;
}
