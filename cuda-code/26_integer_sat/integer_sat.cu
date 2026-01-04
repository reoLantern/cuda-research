// integer_sat.cu
#include <climits>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

        // /*00c0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;  /* 0x0000000404057981 */
        // /* 0x000ee2000c1e9900 */
        // /*00d0*/                   SHF.L.U32 R9, R9, 0x2, RZ ;            /* 0x0000000209097819 */
        // /* 0x000fca00000006ff */
        // /*00e0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;            /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00f0*/                   IADD3 R9, R2, R5, RZ ;                 /* 0x0000000502097210 */
        // /* 0x008fca0007ffe0ff */
        // /*0100*/                   STG.E desc[UR4][R6.64], R9 ;           /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
        // /*0110*/                   EXIT ;                                 /* 0x000000000000794d */
        // /* 0x000fea0003800000 */
__device__ __forceinline__ int add_s32(int a, int b) {
    int out;
    asm volatile("add.s32 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
}

        // /*00c0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;                                      /* 0x0000000404057981 */
        // /* 0x000ee2000c1e9900 */
        // /*00d0*/                   SHF.L.U32 R9, R9, 0x2, RZ ;                                                /* 0x0000000209097819 */
        // /* 0x000fca00000006ff */
        // /*00e0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;                                                /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00f0*/                   IADD3 R0, R2, R5, RZ ;                                                     /* 0x0000000502007210 */
        // /* 0x008fc80007ffe0ff */
        // /*0100*/                   PLOP3.LUT P0, PT, R2.reuse.SIGN, R5.reuse.SIGN, R0.reuse.SIGN, 0x2, 0x0 ;  /* 0x000000050200721f */
        // /* 0x1c0fe40000700200 */
        // /*0110*/                   PLOP3.LUT P1, PT, R2.SIGN, R5.SIGN, R0.SIGN, 0x40, 0x0 ;                   /* 0x000000050200721f */
        // /* 0x000fe40000724000 */
        // /*0120*/                   SEL R0, R0, 0x7fffffff, !P0 ;                                              /* 0x7fffffff00007807 */
        // /* 0x000fc80004000000 */
        // /*0130*/                   SEL R9, R0, 0x80000000, !P1 ;                                              /* 0x8000000000097807 */
        // /* 0x000fca0004800000 */
        // /*0140*/                   STG.E desc[UR4][R6.64+0x4], R9 ;                                           /* 0x0000040906007986 */
        // /* 0x000fe2000c101904 */
        // /*0150*/                   EXIT ;                                                                     /* 0x000000000000794d */
        // /* 0x000fea0003800000 */
__device__ __forceinline__ int add_s32_sat(int a, int b) {
    int out;
    asm volatile("add.sat.s32 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
}

        // /*00c0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;  /* 0x0000000404057981 */
        // /* 0x000ee2000c1e9900 */
        // /*00d0*/                   SHF.L.U32 R9, R9, 0x2, RZ ;            /* 0x0000000209097819 */
        // /* 0x000fca00000006ff */
        // /*00e0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;            /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00f0*/                   IADD3 R9, R2, -R5, RZ ;                /* 0x8000000502097210 */
        // /* 0x008fca0007ffe0ff */
        // /*0100*/                   STG.E desc[UR4][R6.64+0x8], R9 ;       /* 0x0000080906007986 */
        // /* 0x000fe2000c101904 */
        // /*0110*/                   EXIT ;                                 /* 0x000000000000794d */
        // /* 0x000fea0003800000 */
__device__ __forceinline__ int sub_s32(int a, int b) {
    int out;
    asm volatile("sub.s32 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
}

        // /*00c0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;                                      /* 0x0000000404057981 */
        // /* 0x000ee2000c1e9900 */
        // /*00d0*/                   SHF.L.U32 R9, R9, 0x2, RZ ;                                                /* 0x0000000209097819 */
        // /* 0x000fca00000006ff */
        // /*00e0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;                                                /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00f0*/                   IADD3 R0, R2, -R5, RZ ;                                                    /* 0x8000000502007210 */
        // /* 0x008fc80007ffe0ff */
        // /*0100*/                   PLOP3.LUT P0, PT, R2.reuse.SIGN, R5.reuse.SIGN, R0.reuse.SIGN, 0x8, 0x0 ;  /* 0x000000050200721f */
        // /* 0x1c0fe40000700800 */
        // /*0110*/                   PLOP3.LUT P1, PT, R2.SIGN, R5.SIGN, R0.SIGN, 0x10, 0x0 ;                   /* 0x000000050200721f */
        // /* 0x000fe40000721000 */
        // /*0120*/                   SEL R0, R0, 0x7fffffff, !P0 ;                                              /* 0x7fffffff00007807 */
        // /* 0x000fc80004000000 */
        // /*0130*/                   SEL R9, R0, 0x80000000, !P1 ;                                              /* 0x8000000000097807 */
        // /* 0x000fca0004800000 */
        // /*0140*/                   STG.E desc[UR4][R6.64+0xc], R9 ;                                           /* 0x00000c0906007986 */
        // /* 0x000fe2000c101904 */
        // /*0150*/                   EXIT ;                                                                     /* 0x000000000000794d */
        // /* 0x000fea0003800000 */
__device__ __forceinline__ int sub_s32_sat(int a, int b) {
    int out;
    asm volatile("sub.sat.s32 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
}

extern "C" __global__ void integer_sat_kernel(
    const int* __restrict__ in_a,
    const int* __restrict__ in_b,
    int* __restrict__ out
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    int a = in_a[tid];
    int b = in_b[tid];

    int o = tid * 4;
    out[o + 0] = add_s32(a, b);
    out[o + 1] = add_s32_sat(a, b);
    out[o + 2] = sub_s32(a, b);
    out[o + 3] = sub_s32_sat(a, b);
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    constexpr int N = 256;
    constexpr int OUTN = N * 4;
    constexpr int kEdgeN = 8;

    int *in_a, *in_b, *out;

    ck(cudaMallocManaged(&in_a, N * sizeof(int)), "cudaMallocManaged in_a");
    ck(cudaMallocManaged(&in_b, N * sizeof(int)), "cudaMallocManaged in_b");
    ck(cudaMallocManaged(&out, OUTN * sizeof(int)), "cudaMallocManaged out");

    const int a_vals[kEdgeN] = {
        INT_MAX, INT_MAX, INT_MIN, INT_MIN,
        100, -100, 0x70000000, -0x70000000
    };
    const int b_vals[kEdgeN] = {
        1, 100, -1, -100,
        INT_MAX, INT_MIN, 0x10000000, -0x20000000
    };

    for (int i = 0; i < N; ++i) {
        if (i < kEdgeN) {
            in_a[i] = a_vals[i];
            in_b[i] = b_vals[i];
        } else {
            in_a[i] = i * 3 + 1;
            in_b[i] = i * 5 + 7;
        }
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    integer_sat_kernel<<<grid, block>>>(in_a, in_b, out);

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    for (int i = 0; i < kEdgeN; ++i) {
        int o = i * 4;
        std::printf(
            "i=%d a=%d b=%d add=%d add_sat=%d sub=%d sub_sat=%d\n",
            i, in_a[i], in_b[i], out[o + 0], out[o + 1], out[o + 2], out[o + 3]
        );
    }

    cudaFree(in_a);
    cudaFree(in_b);
    cudaFree(out);
    return 0;
}
