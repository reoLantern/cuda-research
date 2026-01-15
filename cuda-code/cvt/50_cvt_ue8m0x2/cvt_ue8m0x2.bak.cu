// cvt_ue8m0x2.cu
//
// cvt.frnd3{.satfinite}.ue8m0x2.f32          d, a, b;
// cvt.frnd3{.satfinite}.ue8m0x2.bf16x2       d, a;
// cvt.rn.bf16x2.ue8m0x2                      d, a;

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

//         /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;                      /* 0x0000000402037981 */
//                                                                                               /* 0x002ea2000c1e9900 */
//         /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;                                /* 0x0000000409047825 */
//                                                                                               /* 0x008fcc00078e0204 */
//         /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;                      /* 0x0000000404047981 */
//                                                                                               /* 0x000ea2000c1e9900 */
//         /*00d0*/                   IMAD.WIDE R6, R9, 0x2, R6 ;                                /* 0x0000000209067825 */
//                                                                                               /* 0x001fe200078e0206 */
//         /*00e0*/                   F2FP.SATFINITE.E8.F32.PACK_AB_MERGE_C.RZ R9, R3, R4, RZ ;  /* 0x000000040309723e */
//                                                                                               /* 0x004fca0004c1e0ff */
//         /*00f0*/                   STG.E.U16 desc[UR4][R6.64], R9 ;                           /* 0x0000000906007986 */
//                                                                                               /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_sat_ue8m0x2_f32(float a, float b) {
    uint16_t out;
    asm volatile("cvt.rz.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(out) : "f"(a), "f"(b));
    return out;
}

//         /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                    /* 0x0000000402027981 */
//                                                                                             /* 0x002ea2000c1e9900 */
//         /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                              /* 0x0000000207047825 */
//                                                                                             /* 0x008fe200078e0204 */
//         /*00b0*/                   F2FP.SATFINITE.E8.BF16.UNPACK_B_MERGE_C.RP R7, R2, RZ ;  /* 0x00000002ff07723e */
//                                                                                             /* 0x004fca0004c922ff */
//         /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;                         /* 0x0000000704007986 */
//                                                                                             /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rp_sat_ue8m0x2_bf16x2(uint32_t a) {
    uint16_t out;
    asm volatile("cvt.rp.satfinite.ue8m0x2.bf16x2 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

//         /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
//                                                                               /* 0x002ea2000c1e9500 */
//         /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
//                                                                               /* 0x008fe200078e0204 */
//         /*00b0*/                   F2FP.BF16.E8.UNPACK_B R7, R2 ;             /* 0x00000002ff07723e */
//                                                                               /* 0x004fca00020816ff */
//         /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;               /* 0x0000000704007986 */
//                                                                               /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_bf16x2_ue8m0x2(uint16_t a) {
    uint32_t out;
    asm volatile("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

extern "C" __global__ void cvt_ue8m0x2_kernel(
    const float* __restrict__ in_a,
    const float* __restrict__ in_b,
    const uint32_t* __restrict__ in_bf16x2,
    const uint16_t* __restrict__ in_ue8m0x2,
    uint16_t* __restrict__ out_ue8m0x2_f32,
    uint16_t* __restrict__ out_ue8m0x2_bf16x2,
    uint32_t* __restrict__ out_bf16x2_ue8m0x2
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    float a = in_a[tid];
    float b = in_b[tid];
    uint32_t bf16x2 = in_bf16x2[tid];
    uint16_t ue8m0x2 = in_ue8m0x2[tid];

    out_ue8m0x2_f32[tid] = cvt_rz_sat_ue8m0x2_f32(a, b);
    out_ue8m0x2_bf16x2[tid] = cvt_rp_sat_ue8m0x2_bf16x2(bf16x2);
    out_bf16x2_ue8m0x2[tid] = cvt_rn_bf16x2_ue8m0x2(ue8m0x2);
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    constexpr int N = 256;

    float* in_a;
    float* in_b;
    uint32_t* in_bf16x2;
    uint16_t* in_ue8m0x2;

    uint16_t* out_ue8m0x2_f32;
    uint16_t* out_ue8m0x2_bf16x2;
    uint32_t* out_bf16x2_ue8m0x2;

    ck(cudaMallocManaged(&in_a, N * sizeof(float)), "cudaMallocManaged in_a");
    ck(cudaMallocManaged(&in_b, N * sizeof(float)), "cudaMallocManaged in_b");
    ck(cudaMallocManaged(&in_bf16x2, N * sizeof(uint32_t)), "cudaMallocManaged in_bf16x2");
    ck(cudaMallocManaged(&in_ue8m0x2, N * sizeof(uint16_t)), "cudaMallocManaged in_ue8m0x2");

    ck(cudaMallocManaged(&out_ue8m0x2_f32, N * sizeof(uint16_t)), "cudaMallocManaged out_ue8m0x2_f32");
    ck(cudaMallocManaged(&out_ue8m0x2_bf16x2, N * sizeof(uint16_t)),
        "cudaMallocManaged out_ue8m0x2_bf16x2");
    ck(cudaMallocManaged(&out_bf16x2_ue8m0x2, N * sizeof(uint32_t)),
        "cudaMallocManaged out_bf16x2_ue8m0x2");

    for (int i = 0; i < N; ++i) {
        float base = (float)(i * 0.5f + 0.25f);
        in_a[i] = base;
        in_b[i] = -base * 0.75f;
        in_bf16x2[i] = 0x3f803f80u + (uint32_t)(i & 0xffu);
        in_ue8m0x2[i] = (uint16_t)(0x7f00u + (uint16_t)(i & 0xffu));
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    cvt_ue8m0x2_kernel<<<grid, block>>>(
        in_a,
        in_b,
        in_bf16x2,
        in_ue8m0x2,
        out_ue8m0x2_f32,
        out_ue8m0x2_bf16x2,
        out_bf16x2_ue8m0x2
    );

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("ue8m0x2=0x%04x bf16x2=0x%08x\n",
        (unsigned int)out_ue8m0x2_f32[0],
        (unsigned int)out_bf16x2_ue8m0x2[0]);

    cudaFree(in_a);
    cudaFree(in_b);
    cudaFree(in_bf16x2);
    cudaFree(in_ue8m0x2);
    cudaFree(out_ue8m0x2_f32);
    cudaFree(out_ue8m0x2_bf16x2);
    cudaFree(out_bf16x2_ue8m0x2);
    return 0;
}
