// sad.cu
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
        // /*0100*/                   IMAD.WIDE R8, R11, 0x4, R8 ;           /* 0x000000040b087825 */
        // /* 0x001fe200078e0208 */
        // /*0110*/                   VABSDIFF R11, R2, R5, R6 ;             /* 0x00000005020b7214 */
        // /* 0x004fca00000e0206 */
        // /*0120*/                   STG.E desc[UR4][R8.64], R11 ;          /* 0x0000000b08007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int sad_s32(int a, int b, int c) {
    int out;
    asm volatile("sad.s32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(c));
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
        // /*0100*/                   IMAD.WIDE R8, R11, 0x4, R8 ;           /* 0x000000040b087825 */
        // /* 0x001fe200078e0208 */
        // /*0110*/                   VABSDIFF.U32 R11, R2, R5, R6 ;         /* 0x00000005020b7214 */
        // /* 0x004fca00000e0006 */
        // /*0120*/                   STG.E desc[UR4][R8.64], R11 ;          /* 0x0000000b08007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int sad_u32(unsigned int a, unsigned int b, unsigned int c) {
    unsigned int out;
    asm volatile("sad.u32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(c));
    return out;
}

        // /*00b0*/                   LDG.E.64.CONSTANT R2, desc[UR4][R2.64] ;              /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9b00 */
        // /*00c0*/                   IMAD.WIDE R4, R11, 0x8, R4 ;                          /* 0x000000080b047825 */
        // /* 0x008fcc00078e0204 */
        // /*00d0*/                   LDG.E.64.CONSTANT R4, desc[UR4][R4.64] ;              /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9b00 */
        // /*00e0*/                   IMAD.WIDE R6, R11, 0x8, R6 ;                          /* 0x000000080b067825 */
        // /* 0x010fcc00078e0206 */
        // /*00f0*/                   LDG.E.64.CONSTANT R6, desc[UR4][R6.64] ;              /* 0x0000000406067981 */
        // /* 0x000ee2000c1e9b00 */
        // /*0100*/                   ISETP.GE.U32.AND P0, PT, R2.reuse, R4.reuse, PT ;     /* 0x000000040200720c */
        // /* 0x0c4fe40003f06070 */
        // /*0110*/                   IADD3 R0, P1, PT, R2.reuse, -R4.reuse, RZ ;           /* 0x8000000402007210 */
        // /* 0x0c0fe40007f3e0ff */
        // /*0120*/                   IADD3 R13, P2, PT, -R2, R4, RZ ;                      /* 0x00000004020d7210 */
        // /* 0x000fe40007f5e1ff */
        // /*0130*/                   ISETP.GE.AND.EX P0, PT, R3.reuse, R5.reuse, PT, P0 ;  /* 0x000000050300720c */
        // /* 0x0c0fe20003f06300 */
        // /*0140*/                   IMAD.X R10, R3.reuse, 0x1, ~R5, P1 ;                  /* 0x00000001030a7824 */
        // /* 0x040fe200008e0e05 */
        // /*0150*/                   IADD3.X R15, PT, PT, ~R3, R5, RZ, P2, !PT ;           /* 0x00000005030f7210 */
        // /* 0x000fe400017fe5ff */
        // /*0160*/                   SEL R13, R0, R13, P0 ;                                /* 0x0000000d000d7207 */
        // /* 0x000fc40000000000 */
        // /*0170*/                   SEL R15, R10, R15, P0 ;                               /* 0x0000000f0a0f7207 */
        // /* 0x000fe40000000000 */
        // /*0180*/                   IADD3 R4, P0, PT, R6, R13, RZ ;                       /* 0x0000000d06047210 */
        // /* 0x008fe20007f1e0ff */
        // /*0190*/                   IMAD.WIDE R2, R11, 0x8, R8 ;                          /* 0x000000080b027825 */
        // /* 0x001fc800078e0208 */
        // /*01a0*/                   IMAD.X R5, R7, 0x1, R15, P0 ;                         /* 0x0000000107057824 */
        // /* 0x000fca00000e060f */
        // /*01b0*/                   STG.E.64 desc[UR4][R2.64], R4 ;                       /* 0x0000000402007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ long long sad_s64(long long a, long long b, long long c) {
    long long out;
    asm volatile("sad.s64 %0, %1, %2, %3;" : "=l"(out) : "l"(a), "l"(b), "l"(c));
    return out;
}

        // /*00b0*/                   LDG.E.64.CONSTANT R2, desc[UR4][R2.64] ;                  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9b00 */
        // /*00c0*/                   IMAD.WIDE R4, R11, 0x8, R4 ;                              /* 0x000000080b047825 */
        // /* 0x008fcc00078e0204 */
        // /*00d0*/                   LDG.E.64.CONSTANT R4, desc[UR4][R4.64] ;                  /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9b00 */
        // /*00e0*/                   IMAD.WIDE R6, R11, 0x8, R6 ;                              /* 0x000000080b067825 */
        // /* 0x010fcc00078e0206 */
        // /*00f0*/                   LDG.E.64.CONSTANT R6, desc[UR4][R6.64] ;                  /* 0x0000000406067981 */
        // /* 0x000ee2000c1e9b00 */
        // /*0100*/                   ISETP.GE.U32.AND P0, PT, R2.reuse, R4.reuse, PT ;         /* 0x000000040200720c */
        // /* 0x0c4fe40003f06070 */
        // /*0110*/                   IADD3 R0, P1, PT, R2.reuse, -R4.reuse, RZ ;               /* 0x8000000402007210 */
        // /* 0x0c0fe40007f3e0ff */
        // /*0120*/                   IADD3 R13, P2, PT, -R2, R4, RZ ;                          /* 0x00000004020d7210 */
        // /* 0x000fe40007f5e1ff */
        // /*0130*/                   ISETP.GE.U32.AND.EX P0, PT, R3.reuse, R5.reuse, PT, P0 ;  /* 0x000000050300720c */
        // /* 0x0c0fe20003f06100 */
        // /*0140*/                   IMAD.X R10, R3.reuse, 0x1, ~R5, P1 ;                      /* 0x00000001030a7824 */
        // /* 0x040fe200008e0e05 */
        // /*0150*/                   IADD3.X R15, PT, PT, ~R3, R5, RZ, P2, !PT ;               /* 0x00000005030f7210 */
        // /* 0x000fe400017fe5ff */
        // /*0160*/                   SEL R13, R0, R13, P0 ;                                    /* 0x0000000d000d7207 */
        // /* 0x000fc40000000000 */
        // /*0170*/                   SEL R15, R10, R15, P0 ;                                   /* 0x0000000f0a0f7207 */
        // /* 0x000fe40000000000 */
        // /*0180*/                   IADD3 R4, P0, PT, R6, R13, RZ ;                           /* 0x0000000d06047210 */
        // /* 0x008fe20007f1e0ff */
        // /*0190*/                   IMAD.WIDE R2, R11, 0x8, R8 ;                              /* 0x000000080b027825 */
        // /* 0x001fc800078e0208 */
        // /*01a0*/                   IMAD.X R5, R7, 0x1, R15, P0 ;                             /* 0x0000000107057824 */
        // /* 0x000fca00000e060f */
        // /*01b0*/                   STG.E.64 desc[UR4][R2.64], R4 ;                           /* 0x0000000402007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ unsigned long long sad_u64(unsigned long long a, unsigned long long b, unsigned long long c) {
    unsigned long long out;
    asm volatile("sad.u64 %0, %1, %2, %3;" : "=l"(out) : "l"(a), "l"(b), "l"(c));
    return out;
}

extern "C" __global__ void sad_kernel(
    const int* __restrict__ in_a32,
    const int* __restrict__ in_b32,
    const int* __restrict__ in_c32,
    const long long* __restrict__ in_a64,
    const long long* __restrict__ in_b64,
    const long long* __restrict__ in_c64,
    int* __restrict__ out_s32,
    unsigned int* __restrict__ out_u32,
    long long* __restrict__ out_s64,
    unsigned long long* __restrict__ out_u64
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    int a32 = in_a32[tid];
    int b32 = in_b32[tid];
    int c32 = in_c32[tid];

    long long a64 = in_a64[tid];
    long long b64 = in_b64[tid];
    long long c64 = in_c64[tid];

    unsigned int ua32 = (unsigned int)a32;
    unsigned int ub32 = (unsigned int)b32;
    unsigned int uc32 = (unsigned int)c32;
    unsigned long long ua64 = (unsigned long long)a64;
    unsigned long long ub64 = (unsigned long long)b64;
    unsigned long long uc64 = (unsigned long long)c64;

    out_s32[tid] = sad_s32(a32, b32, c32);
    out_u32[tid] = sad_u32(ua32, ub32, uc32);
    out_s64[tid] = sad_s64(a64, b64, c64);
    out_u64[tid] = sad_u64(ua64, ub64, uc64);
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
    long long *in_a64, *in_b64, *in_c64;
    int *out_s32;
    unsigned int *out_u32;
    long long *out_s64;
    unsigned long long *out_u64;

    ck(cudaMallocManaged(&in_a32, N * sizeof(int)), "cudaMallocManaged in_a32");
    ck(cudaMallocManaged(&in_b32, N * sizeof(int)), "cudaMallocManaged in_b32");
    ck(cudaMallocManaged(&in_c32, N * sizeof(int)), "cudaMallocManaged in_c32");

    ck(cudaMallocManaged(&in_a64, N * sizeof(long long)), "cudaMallocManaged in_a64");
    ck(cudaMallocManaged(&in_b64, N * sizeof(long long)), "cudaMallocManaged in_b64");
    ck(cudaMallocManaged(&in_c64, N * sizeof(long long)), "cudaMallocManaged in_c64");

    ck(cudaMallocManaged(&out_s32, N * sizeof(int)), "cudaMallocManaged out_s32");
    ck(cudaMallocManaged(&out_u32, N * sizeof(unsigned int)), "cudaMallocManaged out_u32");
    ck(cudaMallocManaged(&out_s64, N * sizeof(long long)), "cudaMallocManaged out_s64");
    ck(cudaMallocManaged(&out_u64, N * sizeof(unsigned long long)), "cudaMallocManaged out_u64");

    for (int i = 0; i < N; ++i) {
        in_a32[i] = i * 3 + 1;
        in_b32[i] = i * 5 + 7;
        in_c32[i] = i * 2 + 11;
        in_a64[i] = (long long)i * 0x100000001LL + 0x1234;
        in_b64[i] = (long long)i * 0x100000007LL + 0x55;
        in_c64[i] = (long long)i * 0x100000003LL + 0x77;
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    sad_kernel<<<grid, block>>>(
        in_a32, in_b32, in_c32,
        in_a64, in_b64, in_c64,
        out_s32, out_u32, out_s64, out_u64
    );

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("sad_s32=%d sad_u32=%u sad_s64=%lld\n", out_s32[0], out_u32[0], out_s64[0]);

    cudaFree(in_a32);
    cudaFree(in_b32);
    cudaFree(in_c32);
    cudaFree(in_a64);
    cudaFree(in_b64);
    cudaFree(in_c64);
    cudaFree(out_s32);
    cudaFree(out_u32);
    cudaFree(out_s64);
    cudaFree(out_u64);
    return 0;
}
