// div.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;        /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R0, 0x4, R4 ;                  /* 0x0000000400047825 */
        // /* 0x001fca00078e0204 */
        // /*00b0*/                   LDG.E.CONSTANT R9, desc[UR4][R4.64] ;        /* 0x0000000404097981 */
        // /* 0x0000e2000c1e9900 */
        // /*00c0*/                   IABS R10, R2.reuse ;                         /* 0x00000002000a7213 */
        // /* 0x084fe40000000000 */
        // /*00d0*/                   IABS R5, R2 ;                                /* 0x0000000200057213 */
        // /* 0x001fe40000000000 */
        // /*00e0*/                   I2F.RP R8, R10 ;                             /* 0x0000000a00087306 */
        // /* 0x000e220000209400 */
        // /*00f0*/                   IABS R4, R9 ;                                /* 0x0000000900047213 */
        // /* 0x008fce0000000000 */
        // /*0100*/                   MUFU.RCP R8, R8 ;                            /* 0x0000000800087308 */
        // /* 0x001e220000001000 */
        // /*0110*/                   LOP3.LUT R9, R9, R2, RZ, 0x3c, !PT ;         /* 0x0000000209097212 */
        // /* 0x000fc800078e3cff */
        // /*0120*/                   ISETP.GE.AND P1, PT, R9, RZ, PT ;            /* 0x000000ff0900720c */
        // /* 0x000fe20003f26270 */
        // /*0130*/                   VIADD R6, R8, 0xffffffe ;                    /* 0x0ffffffe08067836 */
        // /* 0x001fc80000000000 */
        // /*0140*/                   F2I.FTZ.U32.TRUNC.NTZ R7, R6 ;               /* 0x0000000600077305 */
        // /* 0x000064000021f000 */
        // /*0150*/                   HFMA2 R6, -RZ, RZ, 0, 0 ;                    /* 0x00000000ff067431 */
        // /* 0x001fe400000001ff */
        // /*0160*/                   IMAD.MOV R3, RZ, RZ, -R7 ;                   /* 0x000000ffff037224 */
        // /* 0x002fc800078e0a07 */
        // /*0170*/                   IMAD R3, R3, R10, RZ ;                       /* 0x0000000a03037224 */
        // /* 0x000fc800078e02ff */
        // /*0180*/                   IMAD.HI.U32 R7, R7, R3, R6 ;                 /* 0x0000000307077227 */
        // /* 0x000fc800078e0006 */
        // /*0190*/                   IMAD.MOV R3, RZ, RZ, -R5 ;                   /* 0x000000ffff037224 */
        // /* 0x000fe400078e0a05 */
        // /*01a0*/                   IMAD.HI.U32 R7, R7, R4, RZ ;                 /* 0x0000000407077227 */
        // /* 0x000fc800078e00ff */
        // /*01b0*/                   IMAD R3, R7, R3, R4 ;                        /* 0x0000000307037224 */
        // /* 0x000fe400078e0204 */
        // /*01c0*/                   LDC.64 R4, c[0x0][0x3a0] ;                   /* 0x0000e800ff047b82 */
        // /* 0x000e260000000a00 */
        // /*01d0*/                   ISETP.GT.U32.AND P2, PT, R10, R3, PT ;       /* 0x000000030a00720c */
        // /* 0x000fda0003f44070 */
        // /*01e0*/              @!P2 IADD3 R3, PT, PT, R3, -R10.reuse, RZ ;       /* 0x8000000a0303a210 */
        // /* 0x080fe20007ffe0ff */
        // /*01f0*/              @!P2 VIADD R7, R7, 0x1 ;                          /* 0x000000010707a836 */
        // /* 0x000fe20000000000 */
        // /*0200*/                   ISETP.NE.AND P2, PT, R2, RZ, PT ;            /* 0x000000ff0200720c */
        // /* 0x000fe40003f45270 */
        // /*0210*/                   ISETP.GE.U32.AND P0, PT, R3, R10, PT ;       /* 0x0000000a0300720c */
        // /* 0x000fe20003f06070 */
        // /*0220*/                   IMAD.WIDE R4, R0, 0x4, R4 ;                  /* 0x0000000400047825 */
        // /* 0x001fd800078e0204 */
        // /*0230*/               @P0 IADD3 R7, PT, PT, R7, 0x1, RZ ;              /* 0x0000000107070810 */
        // /* 0x000fc80007ffe0ff */
        // /*0240*/              @!P1 IADD3 R7, PT, PT, -R7, RZ, RZ ;              /* 0x000000ff07079210 */
        // /* 0x000fe40007ffe1ff */
        // /*0250*/              @!P2 LOP3.LUT R7, RZ, R2, RZ, 0x33, !PT ;         /* 0x00000002ff07a212 */
        // /* 0x000fca00078e33ff */
        // /*0260*/                   STG.E desc[UR4][R4.64], R7 ;                 /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int div_s32(int a, int b) {
    int out;
    asm volatile("div.s32 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;      /* 0x0000000404057981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R2, R9, 0x4, R2 ;                /* 0x0000000409027825 */
        // /* 0x008fcc00078e0202 */
        // /*00b0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;      /* 0x0000000402027981 */
        // /* 0x000ee2000c1e9900 */
        // /*00c0*/                   I2F.U32.RP R0, R5 ;                        /* 0x0000000500007306 */
        // /* 0x004e220000209000 */
        // /*00d0*/                   IADD3 R11, PT, PT, RZ, -R5, RZ ;           /* 0x80000005ff0b7210 */
        // /* 0x000fe40007ffe0ff */
        // /*00e0*/                   ISETP.NE.U32.AND P2, PT, R5, RZ, PT ;      /* 0x000000ff0500720c */
        // /* 0x000fca0003f45070 */
        // /*00f0*/                   MUFU.RCP R0, R0 ;                          /* 0x0000000000007308 */
        // /* 0x001e240000001000 */
        // /*0100*/                   IADD3 R6, PT, PT, R0, 0xffffffe, RZ ;      /* 0x0ffffffe00067810 */
        // /* 0x001fcc0007ffe0ff */
        // /*0110*/                   F2I.FTZ.U32.TRUNC.NTZ R7, R6 ;             /* 0x0000000600077305 */
        // /* 0x000064000021f000 */
        // /*0120*/                   HFMA2 R6, -RZ, RZ, 0, 0 ;                  /* 0x00000000ff067431 */
        // /* 0x001fe400000001ff */
        // /*0130*/                   IMAD R11, R11, R7, RZ ;                    /* 0x000000070b0b7224 */
        // /* 0x002fc800078e02ff */
        // /*0140*/                   IMAD.HI.U32 R7, R7, R11, R6 ;              /* 0x0000000b07077227 */
        // /* 0x000fcc00078e0006 */
        // /*0150*/                   IMAD.HI.U32 R7, R7, R2, RZ ;               /* 0x0000000207077227 */
        // /* 0x008fca00078e00ff */
        // /*0160*/                   IADD3 R4, PT, PT, -R7, RZ, RZ ;            /* 0x000000ff07047210 */
        // /* 0x000fca0007ffe1ff */
        // /*0170*/                   IMAD R4, R5, R4, R2 ;                      /* 0x0000000405047224 */
        // /* 0x000fe400078e0202 */
        // /*0180*/                   LDC.64 R2, c[0x0][0x3a8] ;                 /* 0x0000ea00ff027b82 */
        // /* 0x000e260000000a00 */
        // /*0190*/                   ISETP.GE.U32.AND P0, PT, R4, R5, PT ;      /* 0x000000050400720c */
        // /* 0x000fda0003f06070 */
        // /*01a0*/               @P0 IADD3 R4, PT, PT, -R5, R4, RZ ;            /* 0x0000000405040210 */
        // /* 0x000fe40007ffe1ff */
        // /*01b0*/               @P0 IADD3 R7, PT, PT, R7, 0x1, RZ ;            /* 0x0000000107070810 */
        // /* 0x000fe40007ffe0ff */
        // /*01c0*/                   ISETP.GE.U32.AND P1, PT, R4, R5, PT ;      /* 0x000000050400720c */
        // /* 0x000fe20003f26070 */
        // /*01d0*/                   IMAD.WIDE R2, R9, 0x4, R2 ;                /* 0x0000000409027825 */
        // /* 0x001fd800078e0202 */
        // /*01e0*/               @P1 IADD3 R7, PT, PT, R7, 0x1, RZ ;            /* 0x0000000107071810 */
        // /* 0x000fe40007ffe0ff */
        // /*01f0*/              @!P2 LOP3.LUT R7, RZ, R5, RZ, 0x33, !PT ;       /* 0x00000005ff07a212 */
        // /* 0x000fca00078e33ff */
        // /*0200*/                   STG.E desc[UR4][R2.64], R7 ;               /* 0x0000000702007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int div_u32(unsigned int a, unsigned int b) {
    unsigned int out;
    asm volatile("div.u32 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
}

        // /*0090*/                   LDG.E.64.CONSTANT R2, desc[UR4][R2.64] ;                   /* 0x0000000402027981 */
        // /* 0x002f62000c1e9b00 */
        // /*00a0*/                   IMAD.WIDE R4, R0, 0x8, R4 ;                                /* 0x0000000800047825 */
        // /* 0x008fcc00078e0204 */
        // /*00b0*/                   LDG.E.64.CONSTANT R4, desc[UR4][R4.64] ;                   /* 0x0000000404047981 */
        // /* 0x000f62000c1e9b00 */
        // /*00c0*/                   MOV R8, 0xe0 ;                                             /* 0x000000e000087802 */
        // /* 0x000fce0000000f00 */
        // /*00d0*/                   CALL.REL.NOINC 0x120 ;                                     /* 0x0000000000107944 */
        // /* 0x020fea0003c00000 */
        // /*00e0*/                   LDC.64 R4, c[0x0][0x3b0] ;                                 /* 0x0000ec00ff047b82 */
        // /* 0x000e240000000a00 */
        // /*00f0*/                   IMAD.WIDE R4, R0, 0x8, R4 ;                                /* 0x0000000800047825 */
        // /* 0x001fca00078e0204 */
        // /*0100*/                   STG.E.64 desc[UR4][R4.64], R2 ;                            /* 0x0000000204007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ long long div_s64(long long a, long long b) {
    long long out;
    asm volatile("div.s64 %0, %1, %2;" : "=l"(out) : "l"(a), "l"(b));
    return out;
}

        // /*0090*/                   LDG.E.64.CONSTANT R2, desc[UR4][R2.64] ;         /* 0x0000000402027981 */
        // /* 0x002f62000c1e9b00 */
        // /*00a0*/                   IMAD.WIDE R4, R0, 0x8, R4 ;                      /* 0x0000000800047825 */
        // /* 0x008fcc00078e0204 */
        // /*00b0*/                   LDG.E.64.CONSTANT R4, desc[UR4][R4.64] ;         /* 0x0000000404047981 */
        // /* 0x000f62000c1e9b00 */
        // /*00c0*/                   MOV R6, 0xe0 ;                                   /* 0x000000e000067802 */
        // /* 0x000fce0000000f00 */
        // /*00d0*/                   CALL.REL.NOINC 0x120 ;                           /* 0x0000000000107944 */
        // /* 0x020fea0003c00000 */
        // /*00e0*/                   LDC.64 R4, c[0x0][0x3b8] ;                       /* 0x0000ee00ff047b82 */
        // /* 0x000e240000000a00 */
        // /*00f0*/                   IMAD.WIDE R4, R0, 0x8, R4 ;                      /* 0x0000000800047825 */
        // /* 0x001fca00078e0204 */
        // /*0100*/                   STG.E.64 desc[UR4][R4.64], R2 ;                  /* 0x0000000204007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ unsigned long long div_u64(unsigned long long a, unsigned long long b) {
    unsigned long long out;
    asm volatile("div.u64 %0, %1, %2;" : "=l"(out) : "l"(a), "l"(b));
    return out;
}

extern "C" __global__ void div_kernel(
    const int* __restrict__ in_a32,
    const int* __restrict__ in_b32,
    const long long* __restrict__ in_a64,
    const long long* __restrict__ in_b64,
    int* __restrict__ out_s32,
    unsigned int* __restrict__ out_u32,
    long long* __restrict__ out_s64,
    unsigned long long* __restrict__ out_u64
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    int a32 = in_a32[tid];
    int b32 = in_b32[tid];
    long long a64 = in_a64[tid];
    long long b64 = in_b64[tid];

    unsigned int ua32 = (unsigned int)a32;
    unsigned int ub32 = (unsigned int)b32;
    unsigned long long ua64 = (unsigned long long)a64;
    unsigned long long ub64 = (unsigned long long)b64;

    out_s32[tid] = div_s32(a32, b32);
    out_u32[tid] = div_u32(ua32, ub32);
    out_s64[tid] = div_s64(a64, b64);
    out_u64[tid] = div_u64(ua64, ub64);
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    constexpr int N = 256;

    int *in_a32, *in_b32;
    long long *in_a64, *in_b64;
    int *out_s32;
    unsigned int *out_u32;
    long long *out_s64;
    unsigned long long *out_u64;

    ck(cudaMallocManaged(&in_a32, N * sizeof(int)), "cudaMallocManaged in_a32");
    ck(cudaMallocManaged(&in_b32, N * sizeof(int)), "cudaMallocManaged in_b32");
    ck(cudaMallocManaged(&in_a64, N * sizeof(long long)), "cudaMallocManaged in_a64");
    ck(cudaMallocManaged(&in_b64, N * sizeof(long long)), "cudaMallocManaged in_b64");

    ck(cudaMallocManaged(&out_s32, N * sizeof(int)), "cudaMallocManaged out_s32");
    ck(cudaMallocManaged(&out_u32, N * sizeof(unsigned int)), "cudaMallocManaged out_u32");
    ck(cudaMallocManaged(&out_s64, N * sizeof(long long)), "cudaMallocManaged out_s64");
    ck(cudaMallocManaged(&out_u64, N * sizeof(unsigned long long)), "cudaMallocManaged out_u64");

    for (int i = 0; i < N; ++i) {
        in_a32[i] = i * 3 + 1000;
        in_b32[i] = (i % 127) + 1;
        in_a64[i] = (long long)i * 0x100000001LL + 0x1234;
        in_b64[i] = (long long)(i % 127) + 1;
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    div_kernel<<<grid, block>>>(
        in_a32, in_b32,
        in_a64, in_b64,
        out_s32, out_u32, out_s64, out_u64
    );

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("div_s32=%d div_u32=%u div_s64=%lld\n", out_s32[0], out_u32[0], out_s64[0]);

    cudaFree(in_a32);
    cudaFree(in_b32);
    cudaFree(in_a64);
    cudaFree(in_b64);
    cudaFree(out_s32);
    cudaFree(out_u32);
    cudaFree(out_s64);
    cudaFree(out_u64);
    return 0;
}
