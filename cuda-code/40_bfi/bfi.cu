// bfi.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

        // /*00b0*/                   LDG.E.CONSTANT R6, desc[UR4][R6.64] ;   /* 0x0000000406067981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   IMAD.WIDE R8, R13.reuse, 0x4, R8 ;      /* 0x000000040d087825 */
        // /* 0x048fe200078e0208 */
        // /*00d0*/                   LDC.64 R10, c[0x0][0x3b0] ;             /* 0x0000ec00ff0a7b82 */
        // /* 0x000e6a0000000a00 */
        // /*00e0*/                   LDG.E.CONSTANT R9, desc[UR4][R8.64] ;   /* 0x0000000408097981 */
        // /* 0x000ea2000c1e9900 */
        // /*00f0*/                   IMAD.WIDE R2, R13, 0x4, R2 ;            /* 0x000000040d027825 */
        // /* 0x010fcc00078e0202 */
        // /*0100*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;   /* 0x0000000402027981 */
        // /* 0x000ee2000c1e9900 */
        // /*0110*/                   IMAD.WIDE R4, R13, 0x4, R4 ;            /* 0x000000040d047825 */
        // /* 0x001fcc00078e0204 */
        // /*0120*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;   /* 0x0000000404047981 */
        // /* 0x000f22000c1e9900 */
        // /*0130*/                   PRMT R0, R9, 0x7604, R6 ;               /* 0x0000760409007816 */
        // /* 0x004fc80000000006 */
        // /*0140*/                   PRMT R15, RZ, 0x4, R0.reuse ;           /* 0x00000004ff0f7816 */
        // /* 0x100fe40000000000 */
        // /*0150*/                   PRMT R0, RZ, 0x5, R0 ;                  /* 0x00000005ff007816 */
        // /* 0x000fc80000000000 */
        // /*0160*/                   BMSK R17, R15, R0 ;                     /* 0x000000000f11721b */
        // /* 0x000fe40000000000 */
        // /*0170*/                   SHF.L.U32 R15, R2, R15, RZ ;            /* 0x0000000f020f7219 */
        // /* 0x008fe200000006ff */
        // /*0180*/                   IMAD.WIDE R6, R13, 0x4, R10 ;           /* 0x000000040d067825 */
        // /* 0x002fc600078e020a */
        // /*0190*/                   LOP3.LUT R9, R15, R17, R4, 0xe2, !PT ;  /* 0x000000110f097212 */
        // /* 0x010fca00078ee204 */
        // /*01a0*/                   STG.E desc[UR4][R6.64], R9 ;            /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int bfi_b32(unsigned int a, unsigned int b, unsigned int pos, unsigned int len) {
    unsigned int out;
    asm volatile("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(out) : "r"(a), "r"(b), "r"(pos), "r"(len));
    return out;
}

        // /*00b0*/                   LDG.E.CONSTANT R10, desc[UR4][R10.64] ;        /* 0x000000040a0a7981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   IMAD.WIDE R6, R0.reuse, 0x8, R6 ;              /* 0x0000000800067825 */
        // /* 0x048fe200078e0206 */
        // /*00d0*/                   LDC.64 R2, c[0x0][0x3b8] ;                     /* 0x0000ee00ff027b82 */
        // /* 0x000e6a0000000a00 */
        // /*00e0*/                   LDG.E.64.CONSTANT R6, desc[UR4][R6.64] ;       /* 0x0000000406067981 */
        // /* 0x000ee2000c1e9b00 */
        // /*00f0*/                   IMAD.WIDE R8, R0, 0x4, R8 ;                    /* 0x0000000400087825 */
        // /* 0x010fcc00078e0208 */
        // /*0100*/                   LDG.E.CONSTANT R8, desc[UR4][R8.64] ;          /* 0x0000000408087981 */
        // /* 0x000f22000c1e9900 */
        // /*0110*/                   IMAD.WIDE R4, R0, 0x8, R4 ;                    /* 0x0000000800047825 */
        // /* 0x001fcc00078e0204 */
        // /*0120*/                   LDG.E.64.CONSTANT R4, desc[UR4][R4.64] ;       /* 0x0000000404047981 */
        // /* 0x000f62000c1e9b00 */
        // /*0130*/                   IMAD.MOV.U32 R15, RZ, RZ, 0x1 ;                /* 0x00000001ff0f7424 */
        // /* 0x000fe400078e00ff */
        // /*0140*/                   IMAD.WIDE R2, R0, 0x8, R2 ;                    /* 0x0000000800027825 */
        // /* 0x002fc600078e0202 */
        // /*0150*/                   SHF.L.U32 R13, R15.reuse, R10.reuse, RZ ;      /* 0x0000000a0f0d7219 */
        // /* 0x0c4fe400000006ff */
        // /*0160*/                   SHF.L.U64.HI R15, R15, R10, RZ ;               /* 0x0000000a0f0f7219 */
        // /* 0x000fe400000102ff */
        // /*0170*/                   IADD3 R13, P0, PT, R13, -0x1, RZ ;             /* 0xffffffff0d0d7810 */
        // /* 0x000fc80007f1e0ff */
        // /*0180*/                   IADD3.X R15, PT, PT, R15, -0x1, RZ, P0, !PT ;  /* 0xffffffff0f0f7810 */
        // /* 0x000fe400007fe4ff */
        // /*0190*/                   LOP3.LUT R11, R13, R6, RZ, 0xc0, !PT ;         /* 0x000000060d0b7212 */
        // /* 0x008fe400078ec0ff */
        // /*01a0*/                   LOP3.LUT R10, R15, R7, RZ, 0xc0, !PT ;         /* 0x000000070f0a7212 */
        // /* 0x000fe400078ec0ff */
        // /*01b0*/                   SHF.L.U64.HI R9, R13.reuse, R8.reuse, R15 ;    /* 0x000000080d097219 */
        // /* 0x0d0fe4000001020f */
        // /*01c0*/                   SHF.L.U32 R7, R13, R8.reuse, RZ ;              /* 0x000000080d077219 */
        // /* 0x080fe400000006ff */
        // /*01d0*/                   SHF.L.U32 R6, R11, R8, RZ ;                    /* 0x000000080b067219 */
        // /* 0x000fc400000006ff */
        // /*01e0*/                   SHF.L.U64.HI R8, R11, R8, R10 ;                /* 0x000000080b087219 */
        // /* 0x000fe4000001020a */
        // /*01f0*/                   LOP3.LUT R4, R6, R4, R7, 0xf4, !PT ;           /* 0x0000000406047212 */
        // /* 0x020fe400078ef407 */
        // /*0200*/                   LOP3.LUT R5, R8, R5, R9, 0xf4, !PT ;           /* 0x0000000508057212 */
        // /* 0x000fca00078ef409 */
        // /*0210*/                   STG.E.64 desc[UR4][R2.64], R4 ;                /* 0x0000000402007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ unsigned long long bfi_b64(unsigned long long a, unsigned long long b, unsigned int pos, unsigned int len) {
    unsigned long long out;
    asm volatile("bfi.b64 %0, %1, %2, %3, %4;" : "=l"(out) : "l"(a), "l"(b), "r"(pos), "r"(len));
    return out;
}

extern "C" __global__ void bfi_kernel(
    const unsigned int* __restrict__ in_a32,
    const unsigned int* __restrict__ in_b32,
    const unsigned long long* __restrict__ in_a64,
    const unsigned long long* __restrict__ in_b64,
    const unsigned int* __restrict__ in_pos,
    const unsigned int* __restrict__ in_len,
    unsigned int* __restrict__ out_b32,
    unsigned long long* __restrict__ out_b64
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    unsigned int a32 = in_a32[tid];
    unsigned int b32 = in_b32[tid];
    unsigned long long a64 = in_a64[tid];
    unsigned long long b64 = in_b64[tid];
    unsigned int pos = in_pos[tid];
    unsigned int len = in_len[tid];

    out_b32[tid] = bfi_b32(a32, b32, pos, len);
    // out_b64[tid] = bfi_b64(a64, b64, pos, len);
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    constexpr int N = 256;

    unsigned int *in_a32, *in_b32, *in_pos, *in_len;
    unsigned long long *in_a64, *in_b64;
    unsigned int *out_b32;
    unsigned long long *out_b64;

    ck(cudaMallocManaged(&in_a32, N * sizeof(unsigned int)), "cudaMallocManaged in_a32");
    ck(cudaMallocManaged(&in_b32, N * sizeof(unsigned int)), "cudaMallocManaged in_b32");
    ck(cudaMallocManaged(&in_a64, N * sizeof(unsigned long long)), "cudaMallocManaged in_a64");
    ck(cudaMallocManaged(&in_b64, N * sizeof(unsigned long long)), "cudaMallocManaged in_b64");
    ck(cudaMallocManaged(&in_pos, N * sizeof(unsigned int)), "cudaMallocManaged in_pos");
    ck(cudaMallocManaged(&in_len, N * sizeof(unsigned int)), "cudaMallocManaged in_len");

    ck(cudaMallocManaged(&out_b32, N * sizeof(unsigned int)), "cudaMallocManaged out_b32");
    ck(cudaMallocManaged(&out_b64, N * sizeof(unsigned long long)), "cudaMallocManaged out_b64");

    for (int i = 0; i < N; ++i) {
        in_a32[i] = 0x12345678u + (unsigned int)i;
        in_b32[i] = 0xaaaaaaaaU ^ (unsigned int)i;
        in_a64[i] = 0x0123456789abcdefull + (unsigned long long)i;
        in_b64[i] = 0xf0f0f0f0f0f0f0f0ull ^ (unsigned long long)i;
        in_pos[i] = (unsigned int)(i % 16);
        in_len[i] = (unsigned int)((i % 8) + 1);
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    bfi_kernel<<<grid, block>>>(
        in_a32, in_b32, in_a64, in_b64,
        in_pos, in_len,
        out_b32, out_b64
    );

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("bfi_b32=%u bfi_b64=%llu\n", out_b32[0], out_b64[0]);

    cudaFree(in_a32);
    cudaFree(in_b32);
    cudaFree(in_a64);
    cudaFree(in_b64);
    cudaFree(in_pos);
    cudaFree(in_len);
    cudaFree(out_b32);
    cudaFree(out_b64);
    return 0;
}
