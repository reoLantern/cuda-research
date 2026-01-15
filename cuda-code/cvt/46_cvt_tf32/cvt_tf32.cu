// cvt_tf32.cu

// cvt.rna{.satfinite}.tf32.f32               d, a;
// cvt.frnd2{.satfinite}{.relu}.tf32.f32      d, a;

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

        // /*0090*/                   LDG.E.CONSTANT R0, desc[UR4][R2.64] ;         /* 0x0000000402007981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   HFMA2 R9, -RZ, RZ, 0, 0.00048828125 ;         /* 0x00001000ff097431 */
        // /* 0x000fe400000001ff */
        // /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                   /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00c0*/                   FSETP.GEU.AND P0, PT, |R0|, +INF , PT ;       /* 0x7f8000000000780b */
        // /* 0x004fda0003f0e200 */
        // /*00d0*/              @!P0 IMAD.IADD.U32 R0, R0, 0x1, R9 ;               /* 0x0000000100008824 */
        // /* 0x000fca00078e0009 */
        // /*00e0*/                   LOP3.LUT R7, R0, 0xffffe000, RZ, 0xc0, !PT ;  /* 0xffffe00000077812 */
        // /* 0x000fca00078ec0ff */
        // /*00f0*/                   STG.E desc[UR4][R4.64], R7 ;                  /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
// lowered 成“位级实现”：把 f32 当 bit-pattern，加舍入偏置（相当于 +0x1000），
// 再用掩码清掉低 13 位（0xffffe000），并用 FSETP/IADD3 做 INF/NaN 或 satfinite 相关处理
__device__ __forceinline__ uint32_t cvt_rna_tf32_f32(float a) {
    uint32_t out;
    asm volatile("cvt.rna.tf32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R0, desc[UR4][R2.64] ;         /* 0x0000000402007981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                   /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00c0*/                   FSETP.GEU.AND P0, PT, |R0|, +INF , PT ;       /* 0x7f8000000000780b */
        // /* 0x004fda0003f0e200 */
        // /*00d0*/              @!P0 IMAD.IADD.U32 R0, R0, 0x1, R9 ;               /* 0x0000000100008824 */
        // /* 0x000fca00078e0009 */
        // /*00e0*/                   LOP3.LUT R9, R0, 0xffffe000, RZ, 0xc0, !PT ;  /* 0xffffe00000097812 */
        // /* 0x000fc800078ec0ff */
        // /*00f0*/                   FSETP.GEU.AND P0, PT, |R9|, +INF , PT ;       /* 0x7f8000000900780b */
        // /* 0x000fda0003f0e200 */
        // /*0100*/               @P0 IADD3 R9, PT, PT, R9, -0x2000, RZ ;           /* 0xffffe00009090810 */
        // /* 0x000fca0007ffe0ff */
        // /*0110*/                   STG.E desc[UR4][R4.64], R9 ;                  /* 0x0000000904007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rna_satfinite_tf32_f32(float a) {
    uint32_t out;
    asm volatile("cvt.rna.satfinite.tf32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R9, desc[UR4][R2.64] ;  /* 0x0000000402097981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.TF32.F32.PACK_B R9, R9 ;          /* 0x00000009ff09723e */
        // /* 0x004fca00024050ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;           /* 0x0000000904007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_tf32_f32(float a) {
    uint32_t out;
    asm volatile("cvt.rn.tf32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R9, desc[UR4][R2.64] ;  /* 0x0000000402097981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.TF32.F32.PACK_B.RZ R9, R9 ;       /* 0x00000009ff09723e */
        // /* 0x004fca000241d0ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;           /* 0x0000000904007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rz_tf32_f32(float a) {
    uint32_t out;
    asm volatile("cvt.rz.tf32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R9, desc[UR4][R2.64] ;    /* 0x0000000402097981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.SATFINITE.TF32.F32.PACK_B R9, R9 ;  /* 0x00000009ff09723e */
        // /* 0x004fca00024070ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_satfinite_tf32_f32(float a) {
    uint32_t out;
    asm volatile("cvt.rn.satfinite.tf32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R9, desc[UR4][R2.64] ;       /* 0x0000000402097981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.SATFINITE.TF32.F32.PACK_B.RZ R9, R9 ;  /* 0x00000009ff09723e */
        // /* 0x004fca000241f0ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                /* 0x0000000904007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rz_satfinite_tf32_f32(float a) {
    uint32_t out;
    asm volatile("cvt.rz.satfinite.tf32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R9, desc[UR4][R2.64] ;         /* 0x0000000402097981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                   /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.SATFINITE.RELU.TF32.F32.PACK_B R9, R9 ;  /* 0x00000009ff09723e */
        // /* 0x004fca00024078ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                  /* 0x0000000904007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_satfinite_relu_tf32_f32(float a) {
    uint32_t out;
    asm volatile("cvt.rn.satfinite.relu.tf32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R9, desc[UR4][R2.64] ;            /* 0x0000000402097981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                      /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.SATFINITE.RELU.TF32.F32.PACK_B.RZ R9, R9 ;  /* 0x00000009ff09723e */
        // /* 0x004fca000241f8ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                     /* 0x0000000904007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rz_satfinite_relu_tf32_f32(float a) {
    uint32_t out;
    asm volatile("cvt.rz.satfinite.relu.tf32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R9, desc[UR4][R2.64] ;  /* 0x0000000402097981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.RELU.TF32.F32.PACK_B R9, R9 ;     /* 0x00000009ff09723e */
        // /* 0x004fca00024058ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;           /* 0x0000000904007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_relu_tf32_f32(float a) {
    uint32_t out;
    asm volatile("cvt.rn.relu.tf32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R9, desc[UR4][R2.64] ;  /* 0x0000000402097981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.RELU.TF32.F32.PACK_B.RZ R9, R9 ;  /* 0x00000009ff09723e */
        // /* 0x004fca000241d8ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;           /* 0x0000000904007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rz_relu_tf32_f32(float a) {
    uint32_t out;
    asm volatile("cvt.rz.relu.tf32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

extern "C" __global__ void cvt_tf32_kernel(
    const float* __restrict__ in_f32,
    uint32_t* __restrict__ out_rna,
    uint32_t* __restrict__ out_rna_satfinite,
    uint32_t* __restrict__ out_rn,
    uint32_t* __restrict__ out_rz,
    uint32_t* __restrict__ out_rn_satfinite,
    uint32_t* __restrict__ out_rz_satfinite,
    uint32_t* __restrict__ out_rn_satfinite_relu,
    uint32_t* __restrict__ out_rz_satfinite_relu,
    uint32_t* __restrict__ out_rn_relu,
    uint32_t* __restrict__ out_rz_relu
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    float a = in_f32[tid];

    out_rna[tid] = cvt_rna_tf32_f32(a);
    // out_rna_satfinite[tid] = cvt_rna_satfinite_tf32_f32(a);
    // out_rn[tid] = cvt_rn_tf32_f32(a);
    // out_rz[tid] = cvt_rz_tf32_f32(a);
    // out_rn_satfinite[tid] = cvt_rn_satfinite_tf32_f32(a);
    // out_rz_satfinite[tid] = cvt_rz_satfinite_tf32_f32(a);
    // out_rn_satfinite_relu[tid] = cvt_rn_satfinite_relu_tf32_f32(a);
    // out_rz_satfinite_relu[tid] = cvt_rz_satfinite_relu_tf32_f32(a);
    // out_rn_relu[tid] = cvt_rn_relu_tf32_f32(a);
    // out_rz_relu[tid] = cvt_rz_relu_tf32_f32(a);
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    constexpr int N = 256;

    float* in_f32;
    uint32_t* out_rna;
    uint32_t* out_rna_satfinite;
    uint32_t* out_rn;
    uint32_t* out_rz;
    uint32_t* out_rn_satfinite;
    uint32_t* out_rz_satfinite;
    uint32_t* out_rn_satfinite_relu;
    uint32_t* out_rz_satfinite_relu;
    uint32_t* out_rn_relu;
    uint32_t* out_rz_relu;

    ck(cudaMallocManaged(&in_f32, N * sizeof(float)), "cudaMallocManaged in_f32");
    ck(cudaMallocManaged(&out_rna, N * sizeof(uint32_t)), "cudaMallocManaged out_rna");
    ck(cudaMallocManaged(&out_rna_satfinite, N * sizeof(uint32_t)), "cudaMallocManaged out_rna_satfinite");
    ck(cudaMallocManaged(&out_rn, N * sizeof(uint32_t)), "cudaMallocManaged out_rn");
    ck(cudaMallocManaged(&out_rz, N * sizeof(uint32_t)), "cudaMallocManaged out_rz");
    ck(cudaMallocManaged(&out_rn_satfinite, N * sizeof(uint32_t)), "cudaMallocManaged out_rn_satfinite");
    ck(cudaMallocManaged(&out_rz_satfinite, N * sizeof(uint32_t)), "cudaMallocManaged out_rz_satfinite");
    ck(cudaMallocManaged(&out_rn_satfinite_relu, N * sizeof(uint32_t)), "cudaMallocManaged out_rn_satfinite_relu");
    ck(cudaMallocManaged(&out_rz_satfinite_relu, N * sizeof(uint32_t)), "cudaMallocManaged out_rz_satfinite_relu");
    ck(cudaMallocManaged(&out_rn_relu, N * sizeof(uint32_t)), "cudaMallocManaged out_rn_relu");
    ck(cudaMallocManaged(&out_rz_relu, N * sizeof(uint32_t)), "cudaMallocManaged out_rz_relu");

    for (int i = 0; i < N; ++i) {
        float base = (float)(i * 0.5f + 0.25f);
        in_f32[i] = base;
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    cvt_tf32_kernel<<<grid, block>>>(
        in_f32,
        out_rna,
        out_rna_satfinite,
        out_rn,
        out_rz,
        out_rn_satfinite,
        out_rz_satfinite,
        out_rn_satfinite_relu,
        out_rz_satfinite_relu,
        out_rn_relu,
        out_rz_relu
    );

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("tf32=0x%08x\n", out_rna[0]);

    cudaFree(in_f32);
    cudaFree(out_rna);
    cudaFree(out_rna_satfinite);
    cudaFree(out_rn);
    cudaFree(out_rz);
    cudaFree(out_rn_satfinite);
    cudaFree(out_rz_satfinite);
    cudaFree(out_rn_satfinite_relu);
    cudaFree(out_rz_satfinite_relu);
    cudaFree(out_rn_relu);
    cudaFree(out_rz_relu);
    return 0;
}
