// bfe.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

        // /*00b0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;  /* 0x0000000404047981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   IMAD.WIDE R6, R11, 0x4, R6 ;           /* 0x000000040b067825 */
        // /* 0x008fcc00078e0206 */
        // /*00d0*/                   LDG.E.CONSTANT R7, desc[UR4][R6.64] ;  /* 0x0000000406077981 */
        // /* 0x000ea2000c1e9900 */
        // /*00e0*/                   IMAD.WIDE R2, R11, 0x4, R2 ;           /* 0x000000040b027825 */
        // /* 0x010fcc00078e0202 */
        // /*00f0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x000ee2000c1e9900 */
        // /*0100*/                   IMAD.WIDE R8, R11, 0x4, R8 ;           /* 0x000000040b087825 */
        // /* 0x001fe200078e0208 */
        // /*0110*/                   PRMT R0, R7, 0x7604, R4 ;              /* 0x0000760407007816 */
        // /* 0x004fc80000000004 */
        // /*0120*/                   PRMT R13, RZ, 0x4, R0.reuse ;          /* 0x00000004ff0d7816 */
        // /* 0x100fe40000000000 */
        // /*0130*/                   PRMT R0, RZ, 0x5, R0 ;                 /* 0x00000005ff007816 */
        // /* 0x000fe40000000000 */
        // /*0140*/                   SHF.R.U32.HI R13, RZ, R13, R2 ;        /* 0x0000000dff0d7219 */
        // /* 0x008fc80000011602 */
        // /*0150*/                   SGXT.U32 R11, R13, R0 ;                /* 0x000000000d0b721a */
        // /* 0x000fca0000000000 */
        // /*0160*/                   STG.E desc[UR4][R8.64], R11 ;          /* 0x0000000b08007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int bfe_u32(unsigned int a, unsigned int pos, unsigned int len) {
    unsigned int out;
    asm volatile("bfe.u32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(pos), "r"(len));
    return out;
}

        // /*00b0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;  /* 0x0000000404047981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   IMAD.WIDE R6, R11, 0x4, R6 ;           /* 0x000000040b067825 */
        // /* 0x008fcc00078e0206 */
        // /*00d0*/                   LDG.E.CONSTANT R7, desc[UR4][R6.64] ;  /* 0x0000000406077981 */
        // /* 0x000ea2000c1e9900 */
        // /*00e0*/                   IMAD.WIDE R2, R11, 0x4, R2 ;           /* 0x000000040b027825 */
        // /* 0x010fcc00078e0202 */
        // /*00f0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x000ee2000c1e9900 */
        // /*0100*/                   IMAD.WIDE R8, R11, 0x4, R8 ;           /* 0x000000040b087825 */
        // /* 0x001fe200078e0208 */
        // /*0110*/                   PRMT R0, R7, 0x7604, R4 ;              /* 0x0000760407007816 */
        // /* 0x004fc80000000004 */
        // /*0120*/                   PRMT R13, RZ, 0x4, R0.reuse ;          /* 0x00000004ff0d7816 */
        // /* 0x100fe40000000000 */
        // /*0130*/                   PRMT R0, RZ, 0x5, R0 ;                 /* 0x00000005ff007816 */
        // /* 0x000fe40000000000 */
        // /*0140*/                   SHF.R.S32.HI R13, RZ, R13, R2 ;        /* 0x0000000dff0d7219 */
        // /* 0x008fc80000011402 */
        // /*0150*/                   SGXT R11, R13, R0 ;                    /* 0x000000000d0b721a */
        // /* 0x000fca0000000200 */
        // /*0160*/                   STG.E desc[UR4][R8.64], R11 ;          /* 0x0000000b08007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int bfe_s32(int a, unsigned int pos, unsigned int len) {
    int out;
    asm volatile("bfe.s32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(pos), "r"(len));
    return out;
}

        // /*00b0*/                   LDG.E.CONSTANT R6, desc[UR4][R6.64] ;         /* 0x0000000406067981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   IMAD.WIDE R4, R11, 0x4, R4 ;                  /* 0x000000040b047825 */
        // /* 0x008fcc00078e0204 */
        // /*00d0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;         /* 0x0000000404057981 */
        // /* 0x000ee2000c1e9900 */
        // /*00e0*/                   IMAD.WIDE R2, R11, 0x8, R2 ;                  /* 0x000000080b027825 */
        // /* 0x010fcc00078e0202 */
        // /*00f0*/                   LDG.E.64.CONSTANT R2, desc[UR4][R2.64] ;      /* 0x0000000402027981 */
        // /* 0x000ee2000c1e9b00 */
        // /*0100*/                   IMAD.MOV.U32 R15, RZ, RZ, 0x1 ;               /* 0x00000001ff0f7424 */
        // /* 0x000fca00078e00ff */
        // /*0110*/                   SHF.L.U32 R13, R15.reuse, R6.reuse, RZ ;      /* 0x000000060f0d7219 */
        // /* 0x0c4fe400000006ff */
        // /*0120*/                   SHF.L.U64.HI R12, R15, R6, RZ ;               /* 0x000000060f0c7219 */
        // /* 0x000fe400000102ff */
        // /*0130*/                   IADD3 R13, P0, PT, R13, -0x1, RZ ;            /* 0xffffffff0d0d7810 */
        // /* 0x000fc80007f1e0ff */
        // /*0140*/                   IADD3.X R7, PT, PT, R12, -0x1, RZ, P0, !PT ;  /* 0xffffffff0c077810 */
        // /* 0x000fe400007fe4ff */
        // /*0150*/                   SHF.R.U64 R0, R2, R5.reuse, R3.reuse ;        /* 0x0000000502007219 */
        // /* 0x188fe40000001203 */
        // /*0160*/                   SHF.R.U32.HI R10, RZ, R5, R3 ;                /* 0x00000005ff0a7219 */
        // /* 0x000fe20000011603 */
        // /*0170*/                   IMAD.WIDE R2, R11, 0x8, R8 ;                  /* 0x000000080b027825 */
        // /* 0x001fe200078e0208 */
        // /*0180*/                   LOP3.LUT R6, R13, R0, RZ, 0xc0, !PT ;         /* 0x000000000d067212 */
        // /* 0x000fe400078ec0ff */
        // /*0190*/                   LOP3.LUT R7, R7, R10, RZ, 0xc0, !PT ;         /* 0x0000000a07077212 */
        // /* 0x000fca00078ec0ff */
        // /*01a0*/                   STG.E.64 desc[UR4][R2.64], R6 ;               /* 0x0000000602007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ unsigned long long bfe_u64(unsigned long long a, unsigned int pos, unsigned int len) {
    unsigned long long out;
    asm volatile("bfe.u64 %0, %1, %2, %3;" : "=l"(out) : "l"(a), "r"(pos), "r"(len));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R6, desc[UR4][R6.64] ;          /* 0x0000000406067981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R0, 0x4, R4 ;                    /* 0x0000000400047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;          /* 0x0000000404057981 */
        // /* 0x000ee2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R2, R0, 0x8, R2 ;                    /* 0x0000000800027825 */
        // /* 0x010fcc00078e0202 */
        // /*00e0*/                   LDG.E.64.CONSTANT R2, desc[UR4][R2.64] ;       /* 0x0000000402027981 */
        // /* 0x000ee2000c1e9b00 */
        // /*00f0*/                   IMAD.MOV.U32 R9, RZ, RZ, 0x1 ;                 /* 0x00000001ff097424 */
        // /* 0x000fca00078e00ff */
        // /*0100*/                   SHF.L.U32 R10, R9.reuse, R6.reuse, RZ ;        /* 0x00000006090a7219 */
        // /* 0x0c4fe400000006ff */
        // /*0110*/                   SHF.L.U64.HI R12, R9, R6, RZ ;                 /* 0x00000006090c7219 */
        // /* 0x000fe400000102ff */
        // /*0120*/                   IADD3 R10, P0, PT, R10, -0x1, RZ ;             /* 0xffffffff0a0a7810 */
        // /* 0x000fe20007f1e0ff */
        // /*0130*/                   VIADD R6, R6, 0xffffffff ;                     /* 0xffffffff06067836 */
        // /* 0x000fc60000000000 */
        // /*0140*/                   IADD3.X R12, PT, PT, R12, -0x1, RZ, P0, !PT ;  /* 0xffffffff0c0c7810 */
        // /* 0x000fe400007fe4ff */
        // /*0150*/                   SHF.R.S64 R8, R2, R5.reuse, R3.reuse ;         /* 0x0000000502087219 */
        // /* 0x188fe40000001003 */
        // /*0160*/                   SHF.R.S32.HI R9, RZ, R5, R3 ;                  /* 0x00000005ff097219 */
        // /* 0x000fe40000011403 */
        // /*0170*/                   LOP3.LUT R7, R10, R8, RZ, 0xc0, !PT ;          /* 0x000000080a077212 */
        // /* 0x000fe200078ec0ff */
        // /*0180*/                   LDC.64 R2, c[0x0][0x3c8] ;                     /* 0x0000f200ff027b82 */
        // /* 0x000e220000000a00 */
        // /*0190*/                   LOP3.LUT R5, R12, R9, RZ, 0xc0, !PT ;          /* 0x000000090c057212 */
        // /* 0x000fc800078ec0ff */
        // /*01a0*/                   SHF.R.U64 R4, R7, R6.reuse, R5.reuse ;         /* 0x0000000607047219 */
        // /* 0x180fe40000001205 */
        // /*01b0*/                   SHF.R.U32.HI R6, RZ, R6, R5 ;                  /* 0x00000006ff067219 */
        // /* 0x000fe40000011605 */
        // /*01c0*/                   ISETP.NE.U32.AND P0, PT, R4, RZ, PT ;          /* 0x000000ff0400720c */
        // /* 0x000fc80003f05070 */
        // /*01d0*/                   ISETP.NE.U32.AND.EX P0, PT, R6, RZ, PT, P0 ;   /* 0x000000ff0600720c */
        // /* 0x000fda0003f05100 */
        // /*01e0*/               @P0 LOP3.LUT R7, R7, R10, RZ, 0xf3, !PT ;          /* 0x0000000a07070212 */
        // /* 0x000fe200078ef3ff */
        // /*01f0*/                   IMAD.WIDE R2, R0, 0x8, R2 ;                    /* 0x0000000800027825 */
        // /* 0x001fe200078e0202 */
        // /*0200*/               @P0 LOP3.LUT R5, R5, R12, RZ, 0xf3, !PT ;          /* 0x0000000c05050212 */
        // /* 0x000fc600078ef3ff */
        // /*0210*/                   IMAD.MOV.U32 R4, RZ, RZ, R7 ;                  /* 0x000000ffff047224 */
        // /* 0x000fca00078e0007 */
        // /*0220*/                   STG.E.64 desc[UR4][R2.64], R4 ;                /* 0x0000000402007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ long long bfe_s64(long long a, unsigned int pos, unsigned int len) {
    long long out;
    asm volatile("bfe.s64 %0, %1, %2, %3;" : "=l"(out) : "l"(a), "r"(pos), "r"(len));
    return out;
}

extern "C" __global__ void bfe_kernel(
    const unsigned int* __restrict__ in_a32,
    const int* __restrict__ in_s32,
    const unsigned long long* __restrict__ in_a64,
    const long long* __restrict__ in_s64,
    const unsigned int* __restrict__ in_pos,
    const unsigned int* __restrict__ in_len,
    unsigned int* __restrict__ out_u32,
    int* __restrict__ out_s32,
    unsigned long long* __restrict__ out_u64,
    long long* __restrict__ out_s64
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    unsigned int a32 = in_a32[tid];
    int s32 = in_s32[tid];
    unsigned long long a64 = in_a64[tid];
    long long s64 = in_s64[tid];
    unsigned int pos = in_pos[tid];
    unsigned int len = in_len[tid];

    out_u32[tid] = bfe_u32(a32, pos, len);
    out_s32[tid] = bfe_s32(s32, pos, len);
    out_u64[tid] = bfe_u64(a64, pos, len);
    out_s64[tid] = bfe_s64(s64, pos, len);
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    constexpr int N = 256;

    unsigned int *in_a32, *in_pos, *in_len;
    int *in_s32;
    unsigned long long *in_a64;
    long long *in_s64;
    unsigned int *out_u32;
    int *out_s32;
    unsigned long long *out_u64;
    long long *out_s64;

    ck(cudaMallocManaged(&in_a32, N * sizeof(unsigned int)), "cudaMallocManaged in_a32");
    ck(cudaMallocManaged(&in_s32, N * sizeof(int)), "cudaMallocManaged in_s32");
    ck(cudaMallocManaged(&in_a64, N * sizeof(unsigned long long)), "cudaMallocManaged in_a64");
    ck(cudaMallocManaged(&in_s64, N * sizeof(long long)), "cudaMallocManaged in_s64");
    ck(cudaMallocManaged(&in_pos, N * sizeof(unsigned int)), "cudaMallocManaged in_pos");
    ck(cudaMallocManaged(&in_len, N * sizeof(unsigned int)), "cudaMallocManaged in_len");

    ck(cudaMallocManaged(&out_u32, N * sizeof(unsigned int)), "cudaMallocManaged out_u32");
    ck(cudaMallocManaged(&out_s32, N * sizeof(int)), "cudaMallocManaged out_s32");
    ck(cudaMallocManaged(&out_u64, N * sizeof(unsigned long long)), "cudaMallocManaged out_u64");
    ck(cudaMallocManaged(&out_s64, N * sizeof(long long)), "cudaMallocManaged out_s64");

    for (int i = 0; i < N; ++i) {
        in_a32[i] = 0x12345678u + (unsigned int)i;
        in_s32[i] = (int)in_a32[i];
        in_a64[i] = 0x0123456789abcdefull + (unsigned long long)i;
        in_s64[i] = (long long)in_a64[i];
        in_pos[i] = (unsigned int)(i % 16);
        in_len[i] = (unsigned int)((i % 8) + 1);
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    bfe_kernel<<<grid, block>>>(
        in_a32, in_s32, in_a64, in_s64,
        in_pos, in_len,
        out_u32, out_s32, out_u64, out_s64
    );

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("bfe_u32=%u bfe_s32=%d bfe_u64=%llu\n", out_u32[0], out_s32[0], out_u64[0]);

    cudaFree(in_a32);
    cudaFree(in_s32);
    cudaFree(in_a64);
    cudaFree(in_s64);
    cudaFree(in_pos);
    cudaFree(in_len);
    cudaFree(out_u32);
    cudaFree(out_s32);
    cudaFree(out_u64);
    cudaFree(out_s64);
    return 0;
}
