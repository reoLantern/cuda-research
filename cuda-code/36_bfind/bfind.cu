// bfind.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FLO.U32 R9, R2 ;                       /* 0x0000000200097300 */
        // /* 0x004e2800000e0000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;           /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int bfind_u32(unsigned int a) {
    unsigned int out;
    asm volatile("bfind.u32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FLO R9, R2 ;                           /* 0x0000000200097300 */
        // /* 0x004e2800000e0200 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;           /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int bfind_s32(int a) {
    unsigned int out;
    asm volatile("bfind.s32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FLO.U32.SH R9, R2 ;                    /* 0x0000000200097300 */
        // /* 0x004e2800000e0400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;           /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int bfind_shift_u32(unsigned int a) {
    unsigned int out;
    asm volatile("bfind.shiftamt.u32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FLO.SH R9, R2 ;                        /* 0x0000000200097300 */
        // /* 0x004e2800000e0600 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;           /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int bfind_shift_s32(int a) {
    unsigned int out;
    asm volatile("bfind.shiftamt.s32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9b00 */
        // /*00a0*/                   BSSY.RECONVERGENT B0, 0x120 ;             /* 0x0000007000007945 */
        // /* 0x000fe20003800200 */
        // /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;               /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00c0*/                   FLO.U32 R0, R3 ;                          /* 0x0000000300007300 */
        // /* 0x004e2400000e0000 */
        // /*00d0*/                   ISETP.NE.U32.AND P0, PT, R0, -0x1, PT ;   /* 0xffffffff0000780c */
        // /* 0x001fda0003f05070 */
        // /*00e0*/               @P0 IADD3 R7, PT, PT, R0, 0x20, RZ ;          /* 0x0000002000070810 */
        // /* 0x000fe20007ffe0ff */
        // /*00f0*/               @P0 BRA 0x110 ;                               /* 0x0000000000040947 */
        // /* 0x000fec0003800000 */
        // /*0100*/                   FLO.U32 R7, R2 ;                          /* 0x0000000200077300 */
        // /* 0x00006400000e0000 */
        // /*0110*/                   BSYNC.RECONVERGENT B0 ;                   /* 0x0000000000007941 */
        // /* 0x000fea0003800200 */
        // /*0120*/                   STG.E desc[UR4][R4.64], R7 ;              /* 0x0000000704007986 */
        // /* 0x002fe2000c101904 */
__device__ __forceinline__ unsigned int bfind_u64(unsigned long long a) {
    unsigned int out;
    asm volatile("bfind.u64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0080*/                   LDG.E.64.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9b00 */
        // /*0090*/                   BSSY.RECONVERGENT B0, 0x160 ;             /* 0x000000c000007945 */
        // /* 0x000fe20003800200 */
        // /*00a0*/                   FLO R0, R3 ;                              /* 0x0000000300007300 */
        // /* 0x004e2400000e0200 */
        // /*00b0*/                   ISETP.NE.U32.AND P0, PT, R0, -0x1, PT ;   /* 0xffffffff0000780c */
        // /* 0x001fda0003f05070 */
        // /*00c0*/               @P0 IADD3 R7, PT, PT, R0, 0x20, RZ ;          /* 0x0000002000070810 */
        // /* 0x000fe20007ffe0ff */
        // /*00d0*/               @P0 BRA 0x150 ;                               /* 0x00000000001c0947 */
        // /* 0x000fec0003800000 */
        // /*00e0*/                   ISETP.GE.AND P0, PT, R3, RZ, PT ;         /* 0x000000ff0300720c */
        // /* 0x000fe20003f06270 */
        // /*00f0*/                   BSSY.RECONVERGENT B1, 0x150 ;             /* 0x0000005000017945 */
        // /* 0x000fd80003800200 */
        // /*0100*/              @!P0 BRA 0x130 ;                               /* 0x0000000000088947 */
        // /* 0x000fea0003800000 */
        // /*0110*/                   FLO.U32 R7, R2 ;                          /* 0x0000000200077300 */
        // /* 0x00006200000e0000 */
        // /*0120*/                   BRA 0x140 ;                               /* 0x0000000000047947 */
        // /* 0x000fea0003800000 */
        // /*0130*/                   FLO.U32 R7, ~R2 ;                         /* 0x8000000200077300 */
        // /* 0x0004e400000e0000 */
        // /*0140*/                   BSYNC.RECONVERGENT B1 ;                   /* 0x0000000000017941 */
        // /* 0x000fea0003800200 */
        // /*0150*/                   BSYNC.RECONVERGENT B0 ;                   /* 0x0000000000007941 */
        // /* 0x000fea0003800200 */
        // /*0160*/                   LDC.64 R2, c[0x0][0x3b8] ;                /* 0x0000ee00ff027b82 */
        // /* 0x005e240000000a00 */
        // /*0170*/                   IMAD.WIDE R2, R5, 0x4, R2 ;               /* 0x0000000405027825 */
        // /* 0x001fca00078e0202 */
        // /*0180*/                   STG.E desc[UR4][R2.64], R7 ;              /* 0x0000000702007986 */
        // /* 0x00afe2000c101904 */
__device__ __forceinline__ unsigned int bfind_s64(long long a) {
    unsigned int out;
    asm volatile("bfind.s64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.CONSTANT R2, desc[UR4][R2.64] ;            /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                         /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FLO.U32.SH R0, R3 ;                                 /* 0x0000000300007300 */
        // /* 0x004e3000000e0400 */
        // /*00c0*/                   FLO.U32.SH R6, R2 ;                                 /* 0x0000000200067300 */
        // /* 0x000e6200000e0400 */
        // /*00d0*/                   ISETP.NE.U32.AND P1, PT, R0, -0x1, PT ;             /* 0xffffffff0000780c */
        // /* 0x001fda0003f25070 */
        // /*00e0*/              @!P1 ISETP.NE.U32.AND P0, PT, R6.reuse, -0x1, PT ;       /* 0xffffffff0600980c */
        // /* 0x042fe40003f05070 */
        // /*00f0*/              @!P1 IADD3 R6, PT, PT, R6, 0x20, RZ ;                    /* 0x0000002006069810 */
        // /* 0x000fc80007ffe0ff */
        // /*0100*/              @!P1 SEL R9, R6, 0xffffffff, P0 ;                        /* 0xffffffff06099807 */
        // /* 0x000fe40000000000 */
        // /*0110*/               @P1 MOV R9, R0 ;                                        /* 0x0000000000091202 */
        // /* 0x000fca0000000f00 */
        // /*0120*/                   STG.E desc[UR4][R4.64], R9 ;                        /* 0x0000000904007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int bfind_shift_u64(unsigned long long a) {
    unsigned int out;
    asm volatile("bfind.shiftamt.u64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0080*/                   LDG.E.64.CONSTANT R2, desc[UR4][R2.64] ;            /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9b00 */
        // /*0090*/                   BSSY.RECONVERGENT B0, 0x110 ;                       /* 0x0000007000007945 */
        // /* 0x000fe20003800200 */
        // /*00a0*/                   ISETP.GE.AND P0, PT, R3, RZ, PT ;                   /* 0x000000ff0300720c */
        // /* 0x004fe20003f06270 */
        // /*00b0*/                   FLO.SH R0, R3 ;                                     /* 0x0000000300007300 */
        // /* 0x00005800000e0600 */
        // /*00c0*/              @!P0 BRA 0xf0 ;                                          /* 0x0000000000088947 */
        // /* 0x000fea0003800000 */
        // /*00d0*/                   FLO.U32.SH R4, R2 ;                                 /* 0x0000000200047300 */
        // /* 0x0008a200000e0400 */
        // /*00e0*/                   BRA 0x100 ;                                         /* 0x0000000000047947 */
        // /* 0x000fea0003800000 */
        // /*00f0*/                   FLO.U32.SH R4, ~R2 ;                                /* 0x8000000200047300 */
        // /* 0x0004e400000e0400 */
        // /*0100*/                   BSYNC.RECONVERGENT B0 ;                             /* 0x0000000000007941 */
        // /* 0x000fea0003800200 */
        // /*0110*/                   LDC.64 R2, c[0x0][0x3c8] ;                          /* 0x0000f200ff027b82 */
        // /* 0x015e220000000a00 */
        // /*0120*/                   ISETP.NE.U32.AND P1, PT, R0, -0x1, PT ;             /* 0xffffffff0000780c */
        // /* 0x002fda0003f25070 */
        // /*0130*/              @!P1 ISETP.NE.U32.AND P0, PT, R4.reuse, -0x1, PT ;       /* 0xffffffff0400980c */
        // /* 0x048fe20003f05070 */
        // /*0140*/              @!P1 VIADD R4, R4, 0x20 ;                                /* 0x0000002004049836 */
        // /* 0x000fca0000000000 */
        // /*0150*/              @!P1 SEL R7, R4, 0xffffffff, P0 ;                        /* 0xffffffff04079807 */
        // /* 0x000fe40000000000 */
        // /*0160*/               @P1 MOV R7, R0 ;                                        /* 0x0000000000071202 */
        // /* 0x000fe20000000f00 */
        // /*0170*/                   IMAD.WIDE R2, R5, 0x4, R2 ;                         /* 0x0000000405027825 */
        // /* 0x001fca00078e0202 */
        // /*0180*/                   STG.E desc[UR4][R2.64], R7 ;                        /* 0x0000000702007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int bfind_shift_s64(long long a) {
    unsigned int out;
    asm volatile("bfind.shiftamt.s64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

extern "C" __global__ void bfind_kernel(
    const int* __restrict__ in_a32,
    const long long* __restrict__ in_a64,
    unsigned int* __restrict__ out_u32,
    unsigned int* __restrict__ out_s32,
    unsigned int* __restrict__ out_shift_u32,
    unsigned int* __restrict__ out_shift_s32,
    unsigned int* __restrict__ out_u64,
    unsigned int* __restrict__ out_s64,
    unsigned int* __restrict__ out_shift_u64,
    unsigned int* __restrict__ out_shift_s64
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    int a32 = in_a32[tid];
    long long a64 = in_a64[tid];

    unsigned int ua32 = (unsigned int)a32;
    unsigned long long ua64 = (unsigned long long)a64;

    // out_u32[tid] = bfind_u32(ua32);
    // out_s32[tid] = bfind_s32(a32);
    // out_shift_u32[tid] = bfind_shift_u32(ua32);
    // out_shift_s32[tid] = bfind_shift_s32(a32);

    out_u64[tid] = bfind_u64(ua64);
    // out_s64[tid] = bfind_s64(a64);
    // out_shift_u64[tid] = bfind_shift_u64(ua64);
    // out_shift_s64[tid] = bfind_shift_s64(a64);
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
    unsigned int *out_u32;
    unsigned int *out_s32;
    unsigned int *out_shift_u32;
    unsigned int *out_shift_s32;
    unsigned int *out_u64;
    unsigned int *out_s64;
    unsigned int *out_shift_u64;
    unsigned int *out_shift_s64;

    ck(cudaMallocManaged(&in_a32, N * sizeof(int)), "cudaMallocManaged in_a32");
    ck(cudaMallocManaged(&in_a64, N * sizeof(long long)), "cudaMallocManaged in_a64");

    ck(cudaMallocManaged(&out_u32, N * sizeof(unsigned int)), "cudaMallocManaged out_u32");
    ck(cudaMallocManaged(&out_s32, N * sizeof(unsigned int)), "cudaMallocManaged out_s32");
    ck(cudaMallocManaged(&out_shift_u32, N * sizeof(unsigned int)), "cudaMallocManaged out_shift_u32");
    ck(cudaMallocManaged(&out_shift_s32, N * sizeof(unsigned int)), "cudaMallocManaged out_shift_s32");
    ck(cudaMallocManaged(&out_u64, N * sizeof(unsigned int)), "cudaMallocManaged out_u64");
    ck(cudaMallocManaged(&out_s64, N * sizeof(unsigned int)), "cudaMallocManaged out_s64");
    ck(cudaMallocManaged(&out_shift_u64, N * sizeof(unsigned int)), "cudaMallocManaged out_shift_u64");
    ck(cudaMallocManaged(&out_shift_s64, N * sizeof(unsigned int)), "cudaMallocManaged out_shift_s64");

    for (int i = 0; i < N; ++i) {
        in_a32[i] = (i & 1) ? (1 << (i % 30)) : -(1 << (i % 30));
        in_a64[i] = (i & 1) ? (1ll << (i % 62)) : -(1ll << (i % 62));
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    bfind_kernel<<<grid, block>>>(
        in_a32, in_a64,
        out_u32, out_s32,
        out_shift_u32, out_shift_s32,
        out_u64, out_s64,
        out_shift_u64, out_shift_s64
    );

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("bfind_u32=%u bfind_s32=%u bfind_u64=%u\n", out_u32[0], out_s32[0], out_u64[0]);

    cudaFree(in_a32);
    cudaFree(in_a64);
    cudaFree(out_u32);
    cudaFree(out_s32);
    cudaFree(out_shift_u32);
    cudaFree(out_shift_s32);
    cudaFree(out_u64);
    cudaFree(out_s64);
    cudaFree(out_shift_u64);
    cudaFree(out_shift_s64);
    return 0;
}
