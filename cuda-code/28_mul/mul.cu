// mul.cu
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
        // /*00d0*/                   SHF.L.U32 R9, R9, 0x1, RZ ;            /* 0x0000000109097819 */
        // /* 0x000fca00000006ff */
        // /*00e0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;            /* 0x0000000409067825 */
        // /* 0x001fc800078e0206 */
        // /*00f0*/                   IMAD R9, R2, R5, RZ ;                  /* 0x0000000502097224 */
        // /* 0x004fca00078e02ff */
        // /*0100*/                   STG.E desc[UR4][R6.64], R9 ;           /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int mul_lo_s32(int a, int b) {
    int out;
    asm volatile("mul.lo.s32 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;            /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;  /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   SHF.L.U32 R9, R9, 0x1, RZ ;            /* 0x0000000109097819 */
        // /* 0x000fca00000006ff */
        // /*00e0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;            /* 0x0000000409067825 */
        // /* 0x001fc800078e0206 */
        // /*00f0*/                   IMAD.HI R9, R2, R5, RZ ;               /* 0x0000000502097227 */
        // /* 0x004fca00078e02ff */
        // /*0100*/                   STG.E desc[UR4][R6.64+0x4], R9 ;       /* 0x0000040906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int mul_hi_s32(int a, int b) {
    int out;
    asm volatile("mul.hi.s32 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;            /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;  /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   SHF.L.U32 R9, R9, 0x1, RZ ;            /* 0x0000000109097819 */
        // /* 0x000fca00000006ff */
        // /*00e0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;            /* 0x0000000409067825 */
        // /* 0x001fc800078e0206 */
        // /*00f0*/                   IMAD R9, R2, R5, RZ ;                  /* 0x0000000502097224 */
        // /* 0x004fca00078e02ff */
        // /*0100*/                   STG.E desc[UR4][R6.64], R9 ;           /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int mul_lo_u32(unsigned int a, unsigned int b) {
    unsigned int out;
    asm volatile("mul.lo.u32 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;            /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;  /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   SHF.L.U32 R9, R9, 0x1, RZ ;            /* 0x0000000109097819 */
        // /* 0x000fca00000006ff */
        // /*00e0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;            /* 0x0000000409067825 */
        // /* 0x001fc800078e0206 */
        // /*00f0*/                   IMAD.HI.U32 R9, R2, R5, RZ ;           /* 0x0000000502097227 */
        // /* 0x004fca00078e00ff */
        // /*0100*/                   STG.E desc[UR4][R6.64+0x4], R9 ;       /* 0x0000040906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int mul_hi_u32(unsigned int a, unsigned int b) {
    unsigned int out;
    asm volatile("mul.hi.u32 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;  /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R8, R7, 0x8, R8 ;            /* 0x0000000807087825 */
        // /* 0x001fc800078e0208 */
        // /*00e0*/                   IMAD.WIDE R6, R2, R5, RZ ;             /* 0x0000000502067225 */
        // /* 0x004fca00078e02ff */
        // /*00f0*/                   STG.E.64 desc[UR4][R8.64], R6 ;        /* 0x0000000608007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ long long mul_wide_s32(int a, int b) {
    long long out;
    asm volatile("mul.wide.s32 %0, %1, %2;" : "=l"(out) : "r"(a), "r"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;  /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R8, R7, 0x8, R8 ;            /* 0x0000000807087825 */
        // /* 0x001fc800078e0208 */
        // /*00e0*/                   IMAD.WIDE.U32 R6, R2, R5, RZ ;         /* 0x0000000502067225 */
        // /* 0x004fca00078e00ff */
        // /*00f0*/                   STG.E.64 desc[UR4][R8.64], R6 ;        /* 0x0000000608007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ unsigned long long mul_wide_u32(unsigned int a, unsigned int b) {
    unsigned long long out;
    asm volatile("mul.wide.u32 %0, %1, %2;" : "=l"(out) : "r"(a), "r"(b));
    return out;
}

        // /*00a0*/                   LDG.E.64.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9b00 */
        // /*00b0*/                   IMAD.WIDE R4, R11, 0x8, R4 ;              /* 0x000000080b047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.64.CONSTANT R4, desc[UR4][R4.64] ;  /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9b00 */
        // /*00d0*/                   SHF.L.U32 R11, R11, 0x1, RZ ;             /* 0x000000010b0b7819 */
        // /* 0x000fca00000006ff */
        // /*00e0*/                   IMAD.WIDE R6, R11, 0x8, R6 ;              /* 0x000000080b067825 */
        // /* 0x001fc800078e0206 */
        // /*00f0*/                   IMAD R13, R3, R4.reuse, RZ ;              /* 0x00000004030d7224 */
        // /* 0x084fe400078e02ff */
        // /*0100*/                   IMAD.WIDE.U32 R8, R2, R4, RZ ;            /* 0x0000000402087225 */
        // /* 0x000fc800078e00ff */
        // /*0110*/                   IMAD R13, R2, R5, R13 ;                   /* 0x00000005020d7224 */
        // /* 0x000fca00078e020d */
        // /*0120*/                   IADD3 R9, PT, PT, R9, R13, RZ ;           /* 0x0000000d09097210 */
        // /* 0x000fca0007ffe0ff */
        // /*0130*/                   STG.E.64 desc[UR4][R6.64], R8 ;           /* 0x0000000806007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ long long mul_lo_s64(long long a, long long b) {
    long long out;
    asm volatile("mul.lo.s64 %0, %1, %2;" : "=l"(out) : "l"(a), "l"(b));
    return out;
}

        // /*0090*/                   LDG.E.64.CONSTANT R2, desc[UR4][R2.64] ;     /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9b00 */
        // /*00a0*/                   IMAD.WIDE R6, R0, 0x8, R6 ;                  /* 0x0000000800067825 */
        // /* 0x008fcc00078e0206 */
        // /*00b0*/                   LDG.E.64.CONSTANT R6, desc[UR4][R6.64] ;     /* 0x0000000406067981 */
        // /* 0x000ea4000c1e9b00 */
        // /*00c0*/                   IMAD.WIDE.U32 R4, R3, R6, RZ ;               /* 0x0000000603047225 */
        // /* 0x004fc800078e00ff */
        // /*00d0*/                   IMAD.WIDE.U32 R8, P0, R2, R7, R4 ;           /* 0x0000000702087225 */
        // /* 0x000fc80007800004 */
        // /*00e0*/                   IMAD.WIDE.U32 R4, R2, R6, RZ ;               /* 0x0000000602047225 */
        // /* 0x000fe200078e00ff */
        // /*00f0*/                   MOV R10, R9 ;                                /* 0x00000009000a7202 */
        // /* 0x000fc60000000f00 */
        // /*0100*/                   IMAD.X R11, RZ, RZ, RZ, P0 ;                 /* 0x000000ffff0b7224 */
        // /* 0x000fe200000e06ff */
        // /*0110*/                   IADD3 RZ, P0, PT, R5, R8, RZ ;               /* 0x0000000805ff7210 */
        // /* 0x000fe40007f1e0ff */
        // /*0120*/                   LDC.64 R4, c[0x0][0x3b0] ;                   /* 0x0000ec00ff047b82 */
        // /* 0x000e260000000a00 */
        // /*0130*/                   IMAD.WIDE.U32.X R8, R3, R7, R10, P0 ;        /* 0x0000000703087225 */
        // /* 0x000fe200000e040a */
        // /*0140*/                   ISETP.LT.AND P1, PT, R7, RZ, PT ;            /* 0x000000ff0700720c */
        // /* 0x000fe40003f21270 */
        // /*0150*/                   IADD3 R11, P0, PT, -R2, R8, RZ ;             /* 0x00000008020b7210 */
        // /* 0x000fc80007f1e1ff */
        // /*0160*/                   SEL R11, R11, R8, P1 ;                       /* 0x000000080b0b7207 */
        // /* 0x000fe20000800000 */
        // /*0170*/                   IMAD.X R13, R9, 0x1, ~R3, P0 ;               /* 0x00000001090d7824 */
        // /* 0x000fc600000e0e03 */
        // /*0180*/                   IADD3 R2, P0, PT, -R6, R11, RZ ;             /* 0x0000000b06027210 */
        // /* 0x000fe40007f1e1ff */
        // /*0190*/                   SEL R13, R13, R9, P1 ;                       /* 0x000000090d0d7207 */
        // /* 0x000fe40000800000 */
        // /*01a0*/                   ISETP.LT.AND P1, PT, R3, RZ, PT ;            /* 0x000000ff0300720c */
        // /* 0x000fe40003f21270 */
        // /*01b0*/                   IADD3.X R6, PT, PT, ~R7, R13, RZ, P0, !PT ;  /* 0x0000000d07067210 */
        // /* 0x000fe200007fe5ff */
        // /*01c0*/                   IMAD.SHL.U32 R7, R0, 0x2, RZ ;               /* 0x0000000200077824 */
        // /* 0x000fe200078e00ff */
        // /*01d0*/                   SEL R2, R2, R11, P1 ;                        /* 0x0000000b02027207 */
        // /* 0x000fe40000800000 */
        // /*01e0*/                   SEL R3, R6, R13, P1 ;                        /* 0x0000000d06037207 */
        // /* 0x000fe20000800000 */
        // /*01f0*/                   IMAD.WIDE R4, R7, 0x8, R4 ;                  /* 0x0000000807047825 */
        // /* 0x001fca00078e0204 */
        // /*0200*/                   STG.E.64 desc[UR4][R4.64+0x8], R2 ;          /* 0x0000080204007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ long long mul_hi_s64(long long a, long long b) {
    long long out;
    asm volatile("mul.hi.s64 %0, %1, %2;" : "=l"(out) : "l"(a), "l"(b));
    return out;
}

        // /*00a0*/                   LDG.E.64.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9b00 */
        // /*00b0*/                   IMAD.WIDE R4, R11, 0x8, R4 ;              /* 0x000000080b047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.64.CONSTANT R4, desc[UR4][R4.64] ;  /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9b00 */
        // /*00d0*/                   SHF.L.U32 R11, R11, 0x1, RZ ;             /* 0x000000010b0b7819 */
        // /* 0x000fca00000006ff */
        // /*00e0*/                   IMAD.WIDE R6, R11, 0x8, R6 ;              /* 0x000000080b067825 */
        // /* 0x001fc800078e0206 */
        // /*00f0*/                   IMAD R13, R3, R4.reuse, RZ ;              /* 0x00000004030d7224 */
        // /* 0x084fe400078e02ff */
        // /*0100*/                   IMAD.WIDE.U32 R8, R2, R4, RZ ;            /* 0x0000000402087225 */
        // /* 0x000fc800078e00ff */
        // /*0110*/                   IMAD R13, R2, R5, R13 ;                   /* 0x00000005020d7224 */
        // /* 0x000fca00078e020d */
        // /*0120*/                   IADD3 R9, PT, PT, R9, R13, RZ ;           /* 0x0000000d09097210 */
        // /* 0x000fca0007ffe0ff */
        // /*0130*/                   STG.E.64 desc[UR4][R6.64], R8 ;           /* 0x0000000806007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ unsigned long long mul_lo_u64(unsigned long long a, unsigned long long b) {
    unsigned long long out;
    asm volatile("mul.lo.u64 %0, %1, %2;" : "=l"(out) : "l"(a), "l"(b));
    return out;
}

        // /*00a0*/                   LDG.E.64.CONSTANT R2, desc[UR4][R2.64] ;    /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9b00 */
        // /*00b0*/                   IMAD.WIDE R4, R11, 0x8, R4 ;                /* 0x000000080b047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.64.CONSTANT R4, desc[UR4][R4.64] ;    /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9b00 */
        // /*00d0*/                   SHF.L.U32 R15, R11, 0x1, RZ ;               /* 0x000000010b0f7819 */
        // /* 0x000fe200000006ff */
        // /*00e0*/                   IMAD.WIDE.U32 R6, R3, R4, RZ ;              /* 0x0000000403067225 */
        // /* 0x004fc800078e00ff */
        // /*00f0*/                   IMAD.WIDE.U32 R8, P0, R2, R5, R6 ;          /* 0x0000000502087225 */
        // /* 0x000fc80007800006 */
        // /*0100*/                   IMAD.WIDE.U32 R6, R2, R4, RZ ;              /* 0x0000000402067225 */
        // /* 0x000fe200078e00ff */
        // /*0110*/                   IADD3.X R11, PT, PT, RZ, RZ, RZ, P0, !PT ;  /* 0x000000ffff0b7210 */
        // /* 0x000fe400007fe4ff */
        // /*0120*/                   MOV R10, R9 ;                               /* 0x00000009000a7202 */
        // /* 0x000fe40000000f00 */
        // /*0130*/                   IADD3 RZ, P0, PT, R7, R8, RZ ;              /* 0x0000000807ff7210 */
        // /* 0x000fe20007f1e0ff */
        // /*0140*/                   IMAD.WIDE R6, R15, 0x8, R12 ;               /* 0x000000080f067825 */
        // /* 0x001fc800078e020c */
        // /*0150*/                   IMAD.WIDE.U32.X R2, R3, R5, R10, P0 ;       /* 0x0000000503027225 */
        // /* 0x000fca00000e040a */
        // /*0160*/                   STG.E.64 desc[UR4][R6.64+0x8], R2 ;         /* 0x0000080206007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ unsigned long long mul_hi_u64(unsigned long long a, unsigned long long b) {
    unsigned long long out;
    asm volatile("mul.hi.u64 %0, %1, %2;" : "=l"(out) : "l"(a), "l"(b));
    return out;
}

extern "C" __global__ void mul_kernel(
    const int* __restrict__ in_a32,
    const int* __restrict__ in_b32,
    const long long* __restrict__ in_a64,
    const long long* __restrict__ in_b64,
    int* __restrict__ out_s32,
    unsigned int* __restrict__ out_u32,
    long long* __restrict__ out_s64,
    unsigned long long* __restrict__ out_u64,
    long long* __restrict__ out_wide_s32,
    unsigned long long* __restrict__ out_wide_u32
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

    int o32 = tid * 2;
    out_s32[o32 + 0] = mul_lo_s32(a32, b32);
    out_s32[o32 + 1] = mul_hi_s32(a32, b32);

    out_u32[o32 + 0] = mul_lo_u32(ua32, ub32);
    out_u32[o32 + 1] = mul_hi_u32(ua32, ub32);

    int o64 = tid * 2;
    out_s64[o64 + 0] = mul_lo_s64(a64, b64);
    out_s64[o64 + 1] = mul_hi_s64(a64, b64);

    out_u64[o64 + 0] = mul_lo_u64(ua64, ub64);
    out_u64[o64 + 1] = mul_hi_u64(ua64, ub64);

    out_wide_s32[tid] = mul_wide_s32(a32, b32);
    out_wide_u32[tid] = mul_wide_u32(ua32, ub32);
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
    long long *out_wide_s32;
    unsigned long long *out_wide_u32;

    ck(cudaMallocManaged(&in_a32, N * sizeof(int)), "cudaMallocManaged in_a32");
    ck(cudaMallocManaged(&in_b32, N * sizeof(int)), "cudaMallocManaged in_b32");
    ck(cudaMallocManaged(&in_a64, N * sizeof(long long)), "cudaMallocManaged in_a64");
    ck(cudaMallocManaged(&in_b64, N * sizeof(long long)), "cudaMallocManaged in_b64");

    ck(cudaMallocManaged(&out_s32, N * 2 * sizeof(int)), "cudaMallocManaged out_s32");
    ck(cudaMallocManaged(&out_u32, N * 2 * sizeof(unsigned int)), "cudaMallocManaged out_u32");
    ck(cudaMallocManaged(&out_s64, N * 2 * sizeof(long long)), "cudaMallocManaged out_s64");
    ck(cudaMallocManaged(&out_u64, N * 2 * sizeof(unsigned long long)), "cudaMallocManaged out_u64");
    ck(cudaMallocManaged(&out_wide_s32, N * sizeof(long long)), "cudaMallocManaged out_wide_s32");
    ck(cudaMallocManaged(&out_wide_u32, N * sizeof(unsigned long long)), "cudaMallocManaged out_wide_u32");

    for (int i = 0; i < N; ++i) {
        in_a32[i] = i * 3 + 1;
        in_b32[i] = i * 5 + 7;
        in_a64[i] = (long long)i * 0x100000001LL + 0x1234;
        in_b64[i] = (long long)i * 0x100000007LL + 0x55;
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    mul_kernel<<<grid, block>>>(
        in_a32, in_b32, in_a64, in_b64,
        out_s32, out_u32, out_s64, out_u64,
        out_wide_s32, out_wide_u32
    );

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf(
        "s32_lo=%d u32_lo=%u s64_lo=%lld u64_lo=%llu wide_s32=%lld\n",
        out_s32[0], out_u32[0], out_s64[0],
        (unsigned long long)out_u64[0], out_wide_s32[0]
    );

    cudaFree(in_a32);
    cudaFree(in_b32);
    cudaFree(in_a64);
    cudaFree(in_b64);
    cudaFree(out_s32);
    cudaFree(out_u32);
    cudaFree(out_s64);
    cudaFree(out_u64);
    cudaFree(out_wide_s32);
    cudaFree(out_wide_u32);
    return 0;
}
