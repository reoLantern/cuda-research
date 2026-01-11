// mad.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

        // /*00b0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   IMAD.WIDE R4, R11, 0x4, R4 ;           /* 0x000000040b047825 */
        // /* 0x008fcc00078e0204 */
        // /*00d0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;  /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00e0*/                   IMAD.WIDE R6, R11, 0x4, R6 ;           /* 0x000000040b067825 */
        // /* 0x010fcc00078e0206 */
        // /*00f0*/                   LDG.E.CONSTANT R6, desc[UR4][R6.64] ;  /* 0x0000000406067981 */
        // /* 0x000ea2000c1e9900 */
        // /*0100*/                   SHF.L.U32 R11, R11, 0x1, RZ ;          /* 0x000000010b0b7819 */
        // /* 0x000fca00000006ff */
        // /*0110*/                   IMAD.WIDE R8, R11, 0x4, R8 ;           /* 0x000000040b087825 */
        // /* 0x001fc800078e0208 */
        // /*0120*/                   IMAD R11, R2, R5, R6 ;                 /* 0x00000005020b7224 */
        // /* 0x004fca00078e0206 */
        // /*0130*/                   STG.E desc[UR4][R8.64], R11 ;          /* 0x0000000b08007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int mad_lo_s32(int a, int b, int c) {
    int out;
    asm volatile("mad.lo.s32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(c));
    return out;
}

        // /*00b0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   IMAD.WIDE R6, R11, 0x4, R6 ;           /* 0x000000040b067825 */
        // /* 0x008fcc00078e0206 */
        // /*00d0*/                   LDG.E.CONSTANT R7, desc[UR4][R6.64] ;  /* 0x0000000406077981 */
        // /* 0x0002a2000c1e9900 */
        // /*00e0*/                   IMAD.WIDE R4, R11, 0x4, R4 ;           /* 0x000000040b047825 */
        // /* 0x010fcc00078e0204 */
        // /*00f0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;  /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*0100*/                   HFMA2 R6, -RZ, RZ, 0, 0 ;              /* 0x00000000ff067431 */
        // /* 0x002fe200000001ff */
        // /*0110*/                   SHF.L.U32 R11, R11, 0x1, RZ ;          /* 0x000000010b0b7819 */
        // /* 0x000fca00000006ff */
        // /*0120*/                   IMAD.WIDE R8, R11, 0x4, R8 ;           /* 0x000000040b087825 */
        // /* 0x001fc800078e0208 */
        // /*0130*/                   IMAD.HI R11, R3, R4, R6 ;              /* 0x00000004030b7227 */
        // /* 0x004fca00078e0206 */
        // /*0140*/                   STG.E desc[UR4][R8.64+0x4], R11 ;      /* 0x0000040b08007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int mad_hi_s32(int a, int b, int c) {
    int out;
    asm volatile("mad.hi.s32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(c));
    return out;
}

        // /*00b0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                                       /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   IMAD.WIDE R4, R11, 0x4, R4 ;                                                /* 0x000000040b047825 */
        // /* 0x008fcc00078e0204 */
        // /*00d0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;                                       /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00e0*/                   IMAD.WIDE R6, R11, 0x4, R6 ;                                                /* 0x000000040b067825 */
        // /* 0x010fcc00078e0206 */
        // /*00f0*/                   LDG.E.CONSTANT R6, desc[UR4][R6.64] ;                                       /* 0x0000000406067981 */
        // /* 0x000ee2000c1e9900 */
        // /*0100*/                   IMAD.WIDE R8, R11, 0x4, R8 ;                                                /* 0x000000040b087825 */
        // /* 0x001fc800078e0208 */
        // /*0110*/                   IMAD.HI R13, R2, R5, RZ ;                                                   /* 0x00000005020d7227 */
        // /* 0x004fca00078e02ff */
        // /*0120*/                   IADD3 R0, PT, PT, R13, R6, RZ ;                                             /* 0x000000060d007210 */
        // /* 0x008fc80007ffe0ff */
        // /*0130*/                   PLOP3.LUT P0, PT, R13.reuse.SIGN, R6.reuse.SIGN, R0.reuse.SIGN, 0x2, 0x0 ;  /* 0x000000060d00721f */
        // /* 0x1c0fe40000700200 */
        // /*0140*/                   PLOP3.LUT P1, PT, R13.SIGN, R6.SIGN, R0.SIGN, 0x40, 0x0 ;                   /* 0x000000060d00721f */
        // /* 0x000fe40000724000 */
        // /*0150*/                   SEL R0, R0, 0x7fffffff, !P0 ;                                               /* 0x7fffffff00007807 */
        // /* 0x000fc80004000000 */
        // /*0160*/                   SEL R11, R0, 0x80000000, !P1 ;                                              /* 0x80000000000b7807 */
        // /* 0x000fca0004800000 */
        // /*0170*/                   STG.E desc[UR4][R8.64], R11 ;                                               /* 0x0000000b08007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int mad_hi_sat_s32(int a, int b, int c) {
    int out;
    asm volatile("mad.hi.sat.s32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(c));
    return out;
}

        // /*00b0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;       /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   IMAD.WIDE R4, R15, 0x4, R4 ;                /* 0x000000040f047825 */
        // /* 0x008fcc00078e0204 */
        // /*00d0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;       /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00e0*/                   IMAD.WIDE R6, R15, 0x8, R6 ;                /* 0x000000080f067825 */
        // /* 0x010fcc00078e0206 */
        // /*00f0*/                   LDG.E.64.CONSTANT R6, desc[UR4][R6.64] ;    /* 0x0000000406067981 */
        // /* 0x000ee2000c1e9b00 */
        // /*0100*/                   IMAD.WIDE R8, R2, R5, RZ ;                  /* 0x0000000502087225 */
        // /* 0x004fc600078e02ff */
        // /*0110*/                   IADD3 R10, P0, PT, R6, R8, RZ ;             /* 0x00000008060a7210 */
        // /* 0x008fc80007f1e0ff */
        // /*0120*/                   IADD3.X R11, PT, PT, R7, R9, RZ, P0, !PT ;  /* 0x00000009070b7210 */
        // /* 0x000fe200007fe4ff */
        // /*0130*/                   IMAD.WIDE R8, R15, 0x8, R12 ;               /* 0x000000080f087825 */
        // /* 0x001fca00078e020c */
        // /*0140*/                   STG.E.64 desc[UR4][R8.64], R10 ;            /* 0x0000000a08007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ long long mad_wide_s32(int a, int b, long long c) {
    long long out;
    asm volatile("mad.wide.s32 %0, %1, %2, %3;" : "=l"(out) : "r"(a), "r"(b), "l"(c));
    return out;
}

        // /*00b0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   IMAD.WIDE R4, R11, 0x4, R4 ;           /* 0x000000040b047825 */
        // /* 0x008fcc00078e0204 */
        // /*00d0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;  /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00e0*/                   IMAD.WIDE R6, R11, 0x4, R6 ;           /* 0x000000040b067825 */
        // /* 0x010fcc00078e0206 */
        // /*00f0*/                   LDG.E.CONSTANT R6, desc[UR4][R6.64] ;  /* 0x0000000406067981 */
        // /* 0x000ea2000c1e9900 */
        // /*0100*/                   SHF.L.U32 R11, R11, 0x1, RZ ;          /* 0x000000010b0b7819 */
        // /* 0x000fca00000006ff */
        // /*0110*/                   IMAD.WIDE R8, R11, 0x4, R8 ;           /* 0x000000040b087825 */
        // /* 0x001fc800078e0208 */
        // /*0120*/                   IMAD R11, R2, R5, R6 ;                 /* 0x00000005020b7224 */
        // /* 0x004fca00078e0206 */
        // /*0130*/                   STG.E desc[UR4][R8.64], R11 ;          /* 0x0000000b08007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int mad_lo_u32(unsigned int a, unsigned int b, unsigned int c) {
    unsigned int out;
    asm volatile("mad.lo.u32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(c));
    return out;
}

        // /*00b0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   IMAD.WIDE R6, R11, 0x4, R6 ;           /* 0x000000040b067825 */
        // /* 0x008fcc00078e0206 */
        // /*00d0*/                   LDG.E.CONSTANT R7, desc[UR4][R6.64] ;  /* 0x0000000406077981 */
        // /* 0x0002a2000c1e9900 */
        // /*00e0*/                   IMAD.WIDE R4, R11, 0x4, R4 ;           /* 0x000000040b047825 */
        // /* 0x010fcc00078e0204 */
        // /*00f0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;  /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*0100*/                   HFMA2 R6, -RZ, RZ, 0, 0 ;              /* 0x00000000ff067431 */
        // /* 0x002fe200000001ff */
        // /*0110*/                   SHF.L.U32 R11, R11, 0x1, RZ ;          /* 0x000000010b0b7819 */
        // /* 0x000fca00000006ff */
        // /*0120*/                   IMAD.WIDE R8, R11, 0x4, R8 ;           /* 0x000000040b087825 */
        // /* 0x001fc800078e0208 */
        // /*0130*/                   IMAD.HI.U32 R11, R3, R4, R6 ;          /* 0x00000004030b7227 */
        // /* 0x004fca00078e0006 */
        // /*0140*/                   STG.E desc[UR4][R8.64+0x4], R11 ;      /* 0x0000040b08007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int mad_hi_u32(unsigned int a, unsigned int b, unsigned int c) {
    unsigned int out;
    asm volatile("mad.hi.u32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(c));
    return out;
}

        // /*00b0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;       /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   IMAD.WIDE R4, R15, 0x4, R4 ;                /* 0x000000040f047825 */
        // /* 0x008fcc00078e0204 */
        // /*00d0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;       /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00e0*/                   IMAD.WIDE R6, R15, 0x8, R6 ;                /* 0x000000080f067825 */
        // /* 0x010fcc00078e0206 */
        // /*00f0*/                   LDG.E.64.CONSTANT R6, desc[UR4][R6.64] ;    /* 0x0000000406067981 */
        // /* 0x000ee2000c1e9b00 */
        // /*0100*/                   IMAD.WIDE.U32 R8, R2, R5, RZ ;              /* 0x0000000502087225 */
        // /* 0x004fc600078e00ff */
        // /*0110*/                   IADD3 R10, P0, PT, R6, R8, RZ ;             /* 0x00000008060a7210 */
        // /* 0x008fc80007f1e0ff */
        // /*0120*/                   IADD3.X R11, PT, PT, R7, R9, RZ, P0, !PT ;  /* 0x00000009070b7210 */
        // /* 0x000fe200007fe4ff */
        // /*0130*/                   IMAD.WIDE R8, R15, 0x8, R12 ;               /* 0x000000080f087825 */
        // /* 0x001fca00078e020c */
        // /*0140*/                   STG.E.64 desc[UR4][R8.64], R10 ;            /* 0x0000000a08007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ unsigned long long mad_wide_u32(unsigned int a, unsigned int b, unsigned long long c) {
    unsigned long long out;
    asm volatile("mad.wide.u32 %0, %1, %2, %3;" : "=l"(out) : "r"(a), "r"(b), "l"(c));
    return out;
}

extern "C" __global__ void mad_kernel(
    const int* __restrict__ in_a32,
    const int* __restrict__ in_b32,
    const int* __restrict__ in_c32,
    const long long* __restrict__ in_c64,
    int* __restrict__ out_s32,
    int* __restrict__ out_s32_sat,
    unsigned int* __restrict__ out_u32,
    long long* __restrict__ out_wide_s32,
    unsigned long long* __restrict__ out_wide_u32
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    int a32 = in_a32[tid];
    int b32 = in_b32[tid];
    int c32 = in_c32[tid];
    long long c64 = in_c64[tid];

    unsigned int ua32 = (unsigned int)a32;
    unsigned int ub32 = (unsigned int)b32;
    unsigned int uc32 = (unsigned int)c32;
    unsigned long long uc64 = (unsigned long long)c64;

    int o32 = tid * 2;
    out_s32[o32 + 0] = mad_lo_s32(a32, b32, c32);
    out_s32[o32 + 1] = mad_hi_s32(a32, b32, c32);
    out_s32_sat[tid] = mad_hi_sat_s32(a32, b32, c32);

    out_u32[o32 + 0] = mad_lo_u32(ua32, ub32, uc32);
    out_u32[o32 + 1] = mad_hi_u32(ua32, ub32, uc32);

    out_wide_s32[tid] = mad_wide_s32(a32, b32, c64);
    out_wide_u32[tid] = mad_wide_u32(ua32, ub32, uc64);
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    constexpr int N = 256;

    int *in_a32, *in_b32, *in_c32;
    long long *in_c64;
    int *out_s32;
    int *out_s32_sat;
    unsigned int *out_u32;
    long long *out_wide_s32;
    unsigned long long *out_wide_u32;

    ck(cudaMallocManaged(&in_a32, N * sizeof(int)), "cudaMallocManaged in_a32");
    ck(cudaMallocManaged(&in_b32, N * sizeof(int)), "cudaMallocManaged in_b32");
    ck(cudaMallocManaged(&in_c32, N * sizeof(int)), "cudaMallocManaged in_c32");
    ck(cudaMallocManaged(&in_c64, N * sizeof(long long)), "cudaMallocManaged in_c64");

    ck(cudaMallocManaged(&out_s32, N * 2 * sizeof(int)), "cudaMallocManaged out_s32");
    ck(cudaMallocManaged(&out_s32_sat, N * sizeof(int)), "cudaMallocManaged out_s32_sat");
    ck(cudaMallocManaged(&out_u32, N * 2 * sizeof(unsigned int)), "cudaMallocManaged out_u32");
    ck(cudaMallocManaged(&out_wide_s32, N * sizeof(long long)), "cudaMallocManaged out_wide_s32");
    ck(cudaMallocManaged(&out_wide_u32, N * sizeof(unsigned long long)), "cudaMallocManaged out_wide_u32");

    for (int i = 0; i < N; ++i) {
        in_a32[i] = i * 3 + 1;
        in_b32[i] = i * 5 + 7;
        in_c32[i] = i * 2 + 9;
        in_c64[i] = (long long)i * 0x100000001LL + 0x1111;
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    mad_kernel<<<grid, block>>>(
        in_a32, in_b32, in_c32, in_c64,
        out_s32, out_s32_sat, out_u32,
        out_wide_s32, out_wide_u32
    );

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("s32_lo=%d u32_lo=%u sat=%d wide=%lld\n",
        out_s32[0], out_u32[0], out_s32_sat[0], out_wide_s32[0]);

    cudaFree(in_a32);
    cudaFree(in_b32);
    cudaFree(in_c32);
    cudaFree(in_c64);
    cudaFree(out_s32);
    cudaFree(out_s32_sat);
    cudaFree(out_u32);
    cudaFree(out_wide_s32);
    cudaFree(out_wide_u32);
    return 0;
}
