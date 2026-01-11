// rem.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;      /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R0, 0x4, R4 ;                /* 0x0000000400047825 */
        // /* 0x001fca00078e0204 */
        // /*00b0*/                   LDG.E.CONSTANT R8, desc[UR4][R4.64] ;      /* 0x0000000404087981 */
        // /* 0x0000e2000c1e9900 */
        // /*00c0*/                   IABS R10, R2.reuse ;                       /* 0x00000002000a7213 */
        // /* 0x084fe40000000000 */
        // /*00d0*/                   IABS R5, R2 ;                              /* 0x0000000200057213 */
        // /* 0x001fe40000000000 */
        // /*00e0*/                   I2F.RP R9, R10 ;                           /* 0x0000000a00097306 */
        // /* 0x000e220000209400 */
        // /*00f0*/                   IABS R4, R8 ;                              /* 0x0000000800047213 */
        // /* 0x008fce0000000000 */
        // /*0100*/                   MUFU.RCP R9, R9 ;                          /* 0x0000000900097308 */
        // /* 0x001e220000001000 */
        // /*0110*/                   ISETP.GE.AND P2, PT, R8, RZ, PT ;          /* 0x000000ff0800720c */
        // /* 0x000fe40003f46270 */
        // /*0120*/                   IADD3 R6, PT, PT, R9, 0xffffffe, RZ ;      /* 0x0ffffffe09067810 */
        // /* 0x001fca0007ffe0ff */
        // /*0130*/                   F2I.FTZ.U32.TRUNC.NTZ R7, R6 ;             /* 0x0000000600077305 */
        // /* 0x000064000021f000 */
        // /*0140*/                   IMAD.MOV.U32 R6, RZ, RZ, RZ ;              /* 0x000000ffff067224 */
        // /* 0x001fe200078e00ff */
        // /*0150*/                   IADD3 R3, PT, PT, RZ, -R7, RZ ;            /* 0x80000007ff037210 */
        // /* 0x002fca0007ffe0ff */
        // /*0160*/                   IMAD R3, R3, R10, RZ ;                     /* 0x0000000a03037224 */
        // /* 0x000fc800078e02ff */
        // /*0170*/                   IMAD.HI.U32 R7, R7, R3, R6 ;               /* 0x0000000307077227 */
        // /* 0x000fe200078e0006 */
        // /*0180*/                   IADD3 R3, PT, PT, RZ, -R5, RZ ;            /* 0x80000005ff037210 */
        // /* 0x000fca0007ffe0ff */
        // /*0190*/                   IMAD.HI.U32 R7, R7, R4, RZ ;               /* 0x0000000407077227 */
        // /* 0x000fc800078e00ff */
        // /*01a0*/                   IMAD R7, R7, R3, R4 ;                      /* 0x0000000307077224 */
        // /* 0x000fe400078e0204 */
        // /*01b0*/                   LDC.64 R4, c[0x0][0x3a0] ;                 /* 0x0000e800ff047b82 */
        // /* 0x000e260000000a00 */
        // /*01c0*/                   ISETP.GT.U32.AND P0, PT, R10, R7, PT ;     /* 0x000000070a00720c */
        // /* 0x000fda0003f04070 */
        // /*01d0*/              @!P0 IMAD.IADD R7, R7, 0x1, -R10 ;              /* 0x0000000107078824 */
        // /* 0x000fe200078e0a0a */
        // /*01e0*/                   ISETP.NE.AND P0, PT, R2, RZ, PT ;          /* 0x000000ff0200720c */
        // /* 0x000fc80003f05270 */
        // /*01f0*/                   ISETP.GT.U32.AND P1, PT, R10, R7, PT ;     /* 0x000000070a00720c */
        // /* 0x000fe20003f24070 */
        // /*0200*/                   IMAD.WIDE R4, R0, 0x4, R4 ;                /* 0x0000000400047825 */
        // /* 0x001fd800078e0204 */
        // /*0210*/              @!P1 IADD3 R7, PT, PT, R7, -R10, RZ ;           /* 0x8000000a07079210 */
        // /* 0x000fca0007ffe0ff */
        // /*0220*/              @!P2 IMAD.MOV R7, RZ, RZ, -R7 ;                 /* 0x000000ffff07a224 */
        // /* 0x000fe200078e0a07 */
        // /*0230*/              @!P0 LOP3.LUT R7, RZ, R2, RZ, 0x33, !PT ;       /* 0x00000002ff078212 */
        // /* 0x000fca00078e33ff */
        // /*0240*/                   STG.E desc[UR4][R4.64], R7 ;               /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int rem_s32(int a, int b) {
    int out;
    asm volatile("rem.s32 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
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
        // /*00e0*/                   ISETP.NE.U32.AND P1, PT, R5, RZ, PT ;      /* 0x000000ff0500720c */
        // /* 0x000fca0003f25070 */
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
        // /*0160*/                   IADD3 R7, PT, PT, -R7, RZ, RZ ;            /* 0x000000ff07077210 */
        // /* 0x000fca0007ffe1ff */
        // /*0170*/                   IMAD R4, R5, R7, R2 ;                      /* 0x0000000705047224 */
        // /* 0x000fe400078e0202 */
        // /*0180*/                   LDC.64 R2, c[0x0][0x3a8] ;                 /* 0x0000ea00ff027b82 */
        // /* 0x000e260000000a00 */
        // /*0190*/                   ISETP.GE.U32.AND P0, PT, R4, R5, PT ;      /* 0x000000050400720c */
        // /* 0x000fda0003f06070 */
        // /*01a0*/               @P0 IADD3 R4, PT, PT, -R5, R4, RZ ;            /* 0x0000000405040210 */
        // /* 0x000fc80007ffe1ff */
        // /*01b0*/                   ISETP.GE.U32.AND P0, PT, R4, R5, PT ;      /* 0x000000050400720c */
        // /* 0x000fe20003f06070 */
        // /*01c0*/                   IMAD.WIDE R2, R9, 0x4, R2 ;                /* 0x0000000409027825 */
        // /* 0x001fd800078e0202 */
        // /*01d0*/               @P0 IADD3 R4, PT, PT, -R5, R4, RZ ;            /* 0x0000000405040210 */
        // /* 0x000fe40007ffe1ff */
        // /*01e0*/              @!P1 LOP3.LUT R4, RZ, R5, RZ, 0x33, !PT ;       /* 0x00000005ff049212 */
        // /* 0x000fca00078e33ff */
        // /*01f0*/                   STG.E desc[UR4][R2.64], R4 ;               /* 0x0000000402007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int rem_u32(unsigned int a, unsigned int b) {
    unsigned int out;
    asm volatile("rem.u32 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
}

__device__ __forceinline__ long long rem_s64(long long a, long long b) {
    long long out;
    asm volatile("rem.s64 %0, %1, %2;" : "=l"(out) : "l"(a), "l"(b));
    return out;
}

        // /*0120*/                   I2F.U64.RP R7, R4 ;                               /* 0x0000000400077312 */
        //                                                                              /* 0x000e300000309000 */
        // /*0130*/                   MUFU.RCP R7, R7 ;                                 /* 0x0000000700077308 */
        //                                                                              /* 0x001e240000001000 */
        // /*0140*/                   IADD3 R8, PT, PT, R7, 0x1ffffffe, RZ ;            /* 0x1ffffffe07087810 */
        //                                                                              /* 0x001fcc0007ffe0ff */
        // /*0150*/                   F2I.U64.TRUNC R8, R8 ;                            /* 0x0000000800087311 */
        //                                                                              /* 0x000e24000020d800 */
        // /*0160*/                   IMAD.WIDE.U32 R10, R8, R4, RZ ;                   /* 0x00000004080a7225 */
        //                                                                              /* 0x001fc800078e00ff */
        // /*0170*/                   IMAD R11, R5, R8, R11 ;                           /* 0x00000008050b7224 */
        //                                                                              /* 0x000fe200078e020b */
        // /*0180*/                   IADD3 R13, P0, PT, RZ, -R10, RZ ;                 /* 0x8000000aff0d7210 */
        //                                                                              /* 0x000fc60007f1e0ff */
        // /*0190*/                   IMAD R11, R4, R9, R11 ;                           /* 0x00000009040b7224 */
        //                                                                              /* 0x000fe400078e020b */
        // /*01a0*/                   IMAD.HI.U32 R10, R8, R13, RZ ;                    /* 0x0000000d080a7227 */
        //                                                                              /* 0x000fc600078e00ff */
        // /*01b0*/                   IADD3.X R15, PT, PT, RZ, ~R11, RZ, P0, !PT ;      /* 0x8000000bff0f7210 */
        //                                                                              /* 0x000fe200007fe4ff */
        // /*01c0*/                   IMAD.MOV.U32 R11, RZ, RZ, R8 ;                    /* 0x000000ffff0b7224 */
        //                                                                              /* 0x000fc800078e0008 */
        // /*01d0*/                   IMAD.WIDE.U32 R10, P0, R8, R15, R10 ;             /* 0x0000000f080a7225 */
        //                                                                              /* 0x000fc8000780000a */
        // /*01e0*/                   IMAD R17, R9.reuse, R15, RZ ;                     /* 0x0000000f09117224 */
        //                                                                              /* 0x040fe400078e02ff */
        // /*01f0*/                   IMAD.HI.U32 R10, P1, R9, R13, R10 ;               /* 0x0000000d090a7227 */
        //                                                                              /* 0x000fc8000782000a */
        // /*0200*/                   IMAD.HI.U32 R7, R9, R15, RZ ;                     /* 0x0000000f09077227 */
        //                                                                              /* 0x000fe200078e00ff */
        // /*0210*/                   IADD3 R11, P2, PT, R17, R10, RZ ;                 /* 0x0000000a110b7210 */
        //                                                                              /* 0x000fc80007f5e0ff */
        // /*0220*/                   IADD3.X R7, PT, PT, R7, R9, RZ, P0, !PT ;         /* 0x0000000907077210 */
        //                                                                              /* 0x000fe200007fe4ff */
        // /*0230*/                   IMAD.WIDE.U32 R8, R11, R4, RZ ;                   /* 0x000000040b087225 */
        //                                                                              /* 0x000fc600078e00ff */
        // /*0240*/                   IADD3.X R7, PT, PT, RZ, RZ, R7, P2, P1 ;          /* 0x000000ffff077210 */
        //                                                                              /* 0x000fe200017e2407 */
        // /*0250*/                   IMAD R9, R5, R11, R9 ;                            /* 0x0000000b05097224 */
        //                                                                              /* 0x000fe200078e0209 */
        // /*0260*/                   IADD3 R13, P0, PT, RZ, -R8, RZ ;                  /* 0x80000008ff0d7210 */
        //                                                                              /* 0x000fc60007f1e0ff */
        // /*0270*/                   IMAD R9, R4, R7, R9 ;                             /* 0x0000000704097224 */
        //                                                                              /* 0x000fe400078e0209 */
        // /*0280*/                   IMAD.HI.U32 R10, R11, R13, RZ ;                   /* 0x0000000d0b0a7227 */
        //                                                                              /* 0x000fc800078e00ff */
        // /*0290*/                   IMAD.X R8, RZ, RZ, ~R9, P0 ;                      /* 0x000000ffff087224 */
        //                                                                              /* 0x000fe400000e0e09 */
        // /*02a0*/                   HFMA2 R9, -RZ, RZ, 0, 0 ;                         /* 0x00000000ff097431 */
        //                                                                              /* 0x000fe400000001ff */
        // /*02b0*/                   IMAD.WIDE.U32 R10, P0, R11, R8, R10 ;             /* 0x000000080b0a7225 */
        //                                                                              /* 0x000fc8000780000a */
        // /*02c0*/                   IMAD R12, R7.reuse, R8, RZ ;                      /* 0x00000008070c7224 */
        //                                                                              /* 0x040fe400078e02ff */
        // /*02d0*/                   IMAD.HI.U32 R11, P1, R7, R13, R10 ;               /* 0x0000000d070b7227 */
        //                                                                              /* 0x000fc8000782000a */
        // /*02e0*/                   IMAD.HI.U32 R8, R7, R8, RZ ;                      /* 0x0000000807087227 */
        //                                                                              /* 0x000fe200078e00ff */
        // /*02f0*/                   IADD3 R11, P2, PT, R12, R11, RZ ;                 /* 0x0000000b0c0b7210 */
        //                                                                              /* 0x000fc80007f5e0ff */
        // /*0300*/                   IADD3.X R7, PT, PT, R8, R7, RZ, P0, !PT ;         /* 0x0000000708077210 */
        //                                                                              /* 0x000fe200007fe4ff */
        // /*0310*/                   IMAD.HI.U32 R8, R11, R2, RZ ;                     /* 0x000000020b087227 */
        //                                                                              /* 0x000fc600078e00ff */
        // /*0320*/                   IADD3.X R7, PT, PT, RZ, RZ, R7, P2, P1 ;          /* 0x000000ffff077210 */
        //                                                                              /* 0x000fe200017e2407 */
        // /*0330*/                   IMAD.WIDE.U32 R8, R3, R11, R8 ;                   /* 0x0000000b03087225 */
        //                                                                              /* 0x000fc800078e0008 */
        // /*0340*/                   IMAD R11, R3, R7.reuse, RZ ;                      /* 0x00000007030b7224 */
        //                                                                              /* 0x080fe400078e02ff */
        // /*0350*/                   IMAD.HI.U32 R8, P0, R2, R7, R8 ;                  /* 0x0000000702087227 */
        //                                                                              /* 0x000fc80007800008 */
        // /*0360*/                   IMAD.HI.U32 R7, R3, R7, RZ ;                      /* 0x0000000703077227 */
        //                                                                              /* 0x000fe200078e00ff */
        // /*0370*/                   IADD3 R11, P1, PT, R11, R8, RZ ;                  /* 0x000000080b0b7210 */
        //                                                                              /* 0x000fc60007f3e0ff */
        // /*0380*/                   IMAD.X R7, RZ, RZ, R7, P0 ;                       /* 0x000000ffff077224 */
        //                                                                              /* 0x000fe400000e0607 */
        // /*0390*/                   IMAD.WIDE.U32 R8, R11, R4, RZ ;                   /* 0x000000040b087225 */
        //                                                                              /* 0x000fc600078e00ff */
        // /*03a0*/                   IADD3.X R7, PT, PT, RZ, R7, RZ, P1, !PT ;         /* 0x00000007ff077210 */
        //                                                                              /* 0x000fe20000ffe4ff */
        // /*03b0*/                   IMAD R11, R5, R11, R9 ;                           /* 0x0000000b050b7224 */
        //                                                                              /* 0x000fe200078e0209 */
        // /*03c0*/                   IADD3 R13, P1, PT, R2, -R8, RZ ;                  /* 0x80000008020d7210 */
        //                                                                              /* 0x000fc60007f3e0ff */
        // /*03d0*/                   IMAD R7, R4, R7, R11 ;                            /* 0x0000000704077224 */
        //                                                                              /* 0x000fe200078e020b */
        // /*03e0*/                   ISETP.GE.U32.AND P0, PT, R13, R4, PT ;            /* 0x000000040d00720c */
        //                                                                              /* 0x000fc80003f06070 */
        // /*03f0*/                   IADD3.X R3, PT, PT, R3, ~R7, RZ, P1, !PT ;        /* 0x8000000703037210 */
        //                                                                              /* 0x000fe40000ffe4ff */
        // /*0400*/                   IADD3 R7, P1, PT, -R4, R13, RZ ;                  /* 0x0000000d04077210 */
        //                                                                              /* 0x000fe40007f3e1ff */
        // /*0410*/                   ISETP.GE.U32.AND.EX P0, PT, R3, R5, PT, P0 ;      /* 0x000000050300720c */
        //                                                                              /* 0x000fc60003f06100 */
        // /*0420*/                   IMAD.X R9, R3, 0x1, ~R5, P1 ;                     /* 0x0000000103097824 */
        //                                                                              /* 0x000fe200008e0e05 */
        // /*0430*/                   SEL R7, R7, R13, P0 ;                             /* 0x0000000d07077207 */
        //                                                                              /* 0x000fe40000000000 */
        // /*0440*/                   ISETP.NE.U32.AND P1, PT, R4.reuse, RZ, PT ;       /* 0x000000ff0400720c */
        //                                                                              /* 0x040fe40003f25070 */
        // /*0450*/                   SEL R9, R9, R3, P0 ;                              /* 0x0000000309097207 */
        //                                                                              /* 0x000fe40000000000 */
        // /*0460*/                   ISETP.GE.U32.AND P0, PT, R7, R4, PT ;             /* 0x000000040700720c */
        //                                                                              /* 0x000fe40003f06070 */
        // /*0470*/                   IADD3 R2, P2, PT, -R4, R7, RZ ;                   /* 0x0000000704027210 */
        //                                                                              /* 0x000fe40007f5e1ff */
        // /*0480*/                   ISETP.GE.U32.AND.EX P0, PT, R9, R5, PT, P0 ;      /* 0x000000050900720c */
        //                                                                              /* 0x000fc40003f06100 */
        // /*0490*/                   IADD3.X R3, PT, PT, ~R5.reuse, R9, RZ, P2, !PT ;  /* 0x0000000905037210 */
        //                                                                              /* 0x040fe400017fe5ff */
        // /*04a0*/                   SEL R2, R2, R7, P0 ;                              /* 0x0000000702027207 */
        //                                                                              /* 0x000fe40000000000 */
        // /*04b0*/                   MOV R7, 0x0 ;                                     /* 0x0000000000077802 */
        //                                                                              /* 0x000fe40000000f00 */
        // /*04c0*/                   ISETP.NE.AND.EX P1, PT, R5, RZ, PT, P1 ;          /* 0x000000ff0500720c */
        //                                                                              /* 0x000fe40003f25310 */
        // /*04d0*/                   SEL R3, R3, R9, P0 ;                              /* 0x0000000903037207 */
        //                                                                              /* 0x000fe40000000000 */
        // /*04e0*/                   SEL R2, R2, 0xffffffff, P1 ;                      /* 0xffffffff02027807 */
        //                                                                              /* 0x000fc40000800000 */
        // /*04f0*/                   SEL R3, R3, 0xffffffff, P1 ;                      /* 0xffffffff03037807 */
        //                                                                              /* 0x000fe20000800000 */
__device__ __forceinline__ unsigned long long rem_u64(unsigned long long a, unsigned long long b) {
    unsigned long long out;
    asm volatile("rem.u64 %0, %1, %2;" : "=l"(out) : "l"(a), "l"(b));
    return out;
}

extern "C" __global__ void rem_kernel(
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

    // out_s32[tid] = rem_s32(a32, b32);
    // out_u32[tid] = rem_u32(ua32, ub32);
    // out_s64[tid] = rem_s64(a64, b64);
    out_u64[tid] = rem_u64(ua64, ub64);
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
    rem_kernel<<<grid, block>>>(
        in_a32, in_b32,
        in_a64, in_b64,
        out_s32, out_u32, out_s64, out_u64
    );

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("rem_s32=%d rem_u32=%u rem_s64=%lld\n", out_s32[0], out_u32[0], out_s64[0]);

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
