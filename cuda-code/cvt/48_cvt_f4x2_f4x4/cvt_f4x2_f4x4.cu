// cvt_f4x2_f4x4.cu

// cvt.rn.satfinite{.relu}.f4x2type.f32        d, a, b;
// cvt.rn.satfinite{.relu}.f4x2type.fp16x2type d, a;
// cvt.rn{.relu}.f16x2.f4x2type                d, a;
// cvt.rs{.relu}.satfinite.f4x4type.f32        d, {a, b, e, f}, rbits;

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

//         /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;                     /* 0x0000000402037981 */
//                                                                                              /* 0x002ea2000c1e9900 */
//         /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;                               /* 0x0000000409047825 */
//                                                                                              /* 0x008fcc00078e0204 */
//         /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;                     /* 0x0000000404047981 */
//                                                                                              /* 0x000ea2000c1e9900 */
//         /*00d0*/                   IMAD.WIDE R6, R9, 0x2, R6 ;                               /* 0x0000000209067825 */
//                                                                                              /* 0x001fe200078e0206 */
//         /*00e0*/                   F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C R0, R3, R4, RZ ;  /* 0x000000040300723e */
//                                                                                              /* 0x004fc800042070ff */
//         /*00f0*/                   LOP3.LUT R9, R0, 0xff, RZ, 0xc0, !PT ;                    /* 0x000000ff00097812 */
//                                                                                              /* 0x000fca00078ec0ff */
//         /*0100*/                   STG.E.U16 desc[UR4][R6.64], R9 ;                          /* 0x0000000906007986 */
//                                                                                              /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_satfinite_e2m1x2_f32(float a, float b) {
    uint16_t out;
    asm volatile("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %1, %2; cvt.u16.u8 %0, t; }"
                 : "=h"(out) : "f"(a), "f"(b));
    return out;
}

//         /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;                          /* 0x0000000402037981 */
//                                                                                                   /* 0x002ea2000c1e9900 */
//         /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;                                    /* 0x0000000409047825 */
//                                                                                                   /* 0x008fcc00078e0204 */
//         /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;                          /* 0x0000000404047981 */
//                                                                                                   /* 0x000ea2000c1e9900 */
//         /*00d0*/                   IMAD.WIDE R6, R9, 0x2, R6 ;                                    /* 0x0000000209067825 */
//                                                                                                   /* 0x001fe200078e0206 */
//         /*00e0*/                   F2FP.SATFINITE.RELU.E2M1.F32.PACK_AB_MERGE_C R0, R3, R4, RZ ;  /* 0x000000040300723e */
//                                                                                                   /* 0x004fc800042078ff */
//         /*00f0*/                   LOP3.LUT R9, R0, 0xff, RZ, 0xc0, !PT ;                         /* 0x000000ff00097812 */
//                                                                                                   /* 0x000fca00078ec0ff */
//         /*0100*/                   STG.E.U16 desc[UR4][R6.64], R9 ;                               /* 0x0000000906007986 */
//                                                                                                   /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_relu_satfinite_e2m1x2_f32(float a, float b) {
    uint16_t out;
    asm volatile("{ .reg .b8 t; cvt.rn.relu.satfinite.e2m1x2.f32 t, %1, %2; cvt.u16.u8 %0, t; }"
                 : "=h"(out) : "f"(a), "f"(b));
    return out;
}

//         /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                  /* 0x0000000402027981 */
//                                                                                           /* 0x002ea2000c1e9900 */
//         /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                            /* 0x0000000207047825 */
//                                                                                           /* 0x008fe200078e0204 */
//         /*00b0*/                   F2FP.SATFINITE.E2M1.F16.UNPACK_B_MERGE_C R0, R2, RZ ;  /* 0x00000002ff00723e */
//                                                                                           /* 0x004fc800042032ff */
//         /*00c0*/                   LOP3.LUT R7, R0, 0xff, RZ, 0xc0, !PT ;                 /* 0x000000ff00077812 */
//                                                                                           /* 0x000fca00078ec0ff */
//         /*00d0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;                       /* 0x0000000704007986 */
//                                                                                           /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_satfinite_e2m1x2_f16x2(uint32_t a) {
    uint16_t out;
    asm volatile("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f16x2 t, %1; cvt.u16.u8 %0, t; }"
                 : "=h"(out) : "r"(a));
    return out;
}

//         /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                       /* 0x0000000402027981 */
//                                                                                                /* 0x002ea2000c1e9900 */
//         /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                                 /* 0x0000000207047825 */
//                                                                                                /* 0x008fe200078e0204 */
//         /*00b0*/                   F2FP.SATFINITE.RELU.E2M1.F16.UNPACK_B_MERGE_C R0, R2, RZ ;  /* 0x00000002ff00723e */
//                                                                                                /* 0x004fc80004203aff */
//         /*00c0*/                   LOP3.LUT R7, R0, 0xff, RZ, 0xc0, !PT ;                      /* 0x000000ff00077812 */
//                                                                                                /* 0x000fca00078ec0ff */
//         /*00d0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;                            /* 0x0000000704007986 */
//                                                                                                /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_relu_satfinite_e2m1x2_f16x2(uint32_t a) {
    uint16_t out;
    asm volatile("{ .reg .b8 t; cvt.rn.relu.satfinite.e2m1x2.f16x2 t, %1; cvt.u16.u8 %0, t; }"
                 : "=h"(out) : "r"(a));
    return out;
}

//         /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                   /* 0x0000000402027981 */
//                                                                                            /* 0x002ea2000c1e9900 */
//         /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                             /* 0x0000000207047825 */
//                                                                                            /* 0x008fe200078e0204 */
//         /*00b0*/                   F2FP.SATFINITE.E2M1.BF16.UNPACK_B_MERGE_C R0, R2, RZ ;  /* 0x00000002ff00723e */
//                                                                                            /* 0x004fc800042832ff */
//         /*00c0*/                   LOP3.LUT R7, R0, 0xff, RZ, 0xc0, !PT ;                  /* 0x000000ff00077812 */
//                                                                                            /* 0x000fca00078ec0ff */
//         /*00d0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;                        /* 0x0000000704007986 */
//                                                                                            /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_satfinite_e2m1x2_bf16x2(uint32_t a) {
    uint16_t out;
    asm volatile("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.bf16x2 t, %1; cvt.u16.u8 %0, t; }"
                 : "=h"(out) : "r"(a));
    return out;
}

//         /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                        /* 0x0000000402027981 */
//                                                                                                 /* 0x002ea2000c1e9900 */
//         /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                                  /* 0x0000000207047825 */
//                                                                                                 /* 0x008fe200078e0204 */
//         /*00b0*/                   F2FP.SATFINITE.RELU.E2M1.BF16.UNPACK_B_MERGE_C R0, R2, RZ ;  /* 0x00000002ff00723e */
//                                                                                                 /* 0x004fc80004283aff */
//         /*00c0*/                   LOP3.LUT R7, R0, 0xff, RZ, 0xc0, !PT ;                       /* 0x000000ff00077812 */
//                                                                                                 /* 0x000fca00078ec0ff */
//         /*00d0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;                             /* 0x0000000704007986 */
//                                                                                                 /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_relu_satfinite_e2m1x2_bf16x2(uint32_t a) {
    uint16_t out;
    asm volatile("{ .reg .b8 t; cvt.rn.relu.satfinite.e2m1x2.bf16x2 t, %1; cvt.u16.u8 %0, t; }"
                 : "=h"(out) : "r"(a));
    return out;
}

//         /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
//                                                                               /* 0x002ea2000c1e9500 */
//         /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
//                                                                               /* 0x008fe200078e0204 */
//         /*00b0*/                   F2FP.F16.E2M1.UNPACK_B R7, R2 ;            /* 0x00000002ff07723e */
//                                                                               /* 0x004fca00020402ff */
//         /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;               /* 0x0000000704007986 */
//                                                                               /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_f16x2_e2m1x2(uint16_t a) {
    uint32_t out;
    asm volatile("{ .reg .b8 t; cvt.u8.u16 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }"
                 : "=r"(out) : "h"(a));
    return out;
}

//         /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
//                                                                               /* 0x002ea2000c1e9500 */
//         /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
//                                                                               /* 0x008fe200078e0204 */
//         /*00b0*/                   F2FP.RELU.F16.E2M1.UNPACK_B R7, R2 ;       /* 0x00000002ff07723e */
//                                                                               /* 0x004fca0002040aff */
//         /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;               /* 0x0000000704007986 */
//                                                                               /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_relu_f16x2_e2m1x2(uint16_t a) {
    uint32_t out;
    asm volatile("{ .reg .b8 t; cvt.u8.u16 t, %1; cvt.rn.relu.f16x2.e2m1x2 %0, t; }"
                 : "=r"(out) : "h"(a));
    return out;
}

//         /*00b0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                         /* 0x0000000402027981 */
//                                                                                                  /* 0x002ea2000c1e9900 */
//         /*00c0*/                   LDC.64 R8, c[0x0][0x398] ;                                    /* 0x0000e600ff087b82 */
//                                                                                                  /* 0x000e620000000a00 */
//         /*00d0*/                   IMAD.WIDE R4, R15, 0x4, R4 ;                                  /* 0x000000040f047825 */
//                                                                                                  /* 0x008fcc00078e0204 */
//         /*00e0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;                         /* 0x0000000404057981 */
//                                                                                                  /* 0x000ea2000c1e9900 */
//         /*00f0*/                   IMAD.WIDE R10, R15.reuse, 0x4, R10 ;                          /* 0x000000040f0a7825 */
//                                                                                                  /* 0x050fe200078e020a */
//         /*0100*/                   LDC.64 R12, c[0x0][0x400] ;                                   /* 0x00010000ff0c7b82 */
//                                                                                                  /* 0x000eea0000000a00 */
//         /*0110*/                   LDG.E.CONSTANT R10, desc[UR4][R10.64] ;                       /* 0x000000040a0a7981 */
//                                                                                                  /* 0x000ea2000c1e9900 */
//         /*0120*/                   IMAD.WIDE R6, R15, 0x4, R6 ;                                  /* 0x000000040f067825 */
//                                                                                                  /* 0x001fcc00078e0206 */
//         /*0130*/                   LDG.E.CONSTANT R7, desc[UR4][R6.64] ;                         /* 0x0000000406077981 */
//                                                                                                  /* 0x000f22000c1e9900 */
//         /*0140*/                   IMAD.WIDE R8, R15, 0x4, R8 ;                                  /* 0x000000040f087825 */
//                                                                                                  /* 0x002fcc00078e0208 */
//         /*0150*/                   LDG.E.CONSTANT R8, desc[UR4][R8.64] ;                         /* 0x0000000408087981 */
//                                                                                                  /* 0x000f22000c1e9900 */
//         /*0160*/                   IMAD.WIDE R12, R15, 0x2, R12 ;                                /* 0x000000020f0c7825 */
//                                                                                                  /* 0x008fe200078e020c */
//         /*0170*/                   F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C.RS R0, R2, R5, R10 ;  /* 0x000000050200723e */
//                                                                                                  /* 0x004fc8000422700a */
//         /*0180*/                   F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C.RS R15, R7, R8, R0 ;  /* 0x00000008070f723e */
//                                                                                                  /* 0x010fca0004227000 */
//         /*0190*/                   STG.E.U16 desc[UR4][R12.64], R15 ;                            /* 0x0000000f0c007986 */
//                                                                                                  /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rs_satfinite_e2m1x4_f32(float a, float b, float e, float f, uint32_t rbits) {
    uint16_t out;
    asm volatile("cvt.rs.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;"
                 : "=h"(out) : "f"(a), "f"(b), "f"(e), "f"(f), "r"(rbits));
    return out;
}

//         /*00b0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                              /* 0x0000000402027981 */
//                                                                                                       /* 0x002ea2000c1e9900 */
//         /*00c0*/                   LDC.64 R8, c[0x0][0x398] ;                                         /* 0x0000e600ff087b82 */
//                                                                                                       /* 0x000e620000000a00 */
//         /*00d0*/                   IMAD.WIDE R4, R15, 0x4, R4 ;                                       /* 0x000000040f047825 */
//                                                                                                       /* 0x008fcc00078e0204 */
//         /*00e0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;                              /* 0x0000000404057981 */
//                                                                                                       /* 0x000ea2000c1e9900 */
//         /*00f0*/                   IMAD.WIDE R10, R15.reuse, 0x4, R10 ;                               /* 0x000000040f0a7825 */
//                                                                                                       /* 0x050fe200078e020a */
//         /*0100*/                   LDC.64 R12, c[0x0][0x408] ;                                        /* 0x00010200ff0c7b82 */
//                                                                                                       /* 0x000eea0000000a00 */
//         /*0110*/                   LDG.E.CONSTANT R10, desc[UR4][R10.64] ;                            /* 0x000000040a0a7981 */
//                                                                                                       /* 0x000ea2000c1e9900 */
//         /*0120*/                   IMAD.WIDE R6, R15, 0x4, R6 ;                                       /* 0x000000040f067825 */
//                                                                                                       /* 0x001fcc00078e0206 */
//         /*0130*/                   LDG.E.CONSTANT R7, desc[UR4][R6.64] ;                              /* 0x0000000406077981 */
//                                                                                                       /* 0x000f22000c1e9900 */
//         /*0140*/                   IMAD.WIDE R8, R15, 0x4, R8 ;                                       /* 0x000000040f087825 */
//                                                                                                       /* 0x002fcc00078e0208 */
//         /*0150*/                   LDG.E.CONSTANT R8, desc[UR4][R8.64] ;                              /* 0x0000000408087981 */
//                                                                                                       /* 0x000f22000c1e9900 */
//         /*0160*/                   IMAD.WIDE R12, R15, 0x2, R12 ;                                     /* 0x000000020f0c7825 */
//                                                                                                       /* 0x008fe200078e020c */
//         /*0170*/                   F2FP.SATFINITE.RELU.E2M1.F32.PACK_AB_MERGE_C.RS R0, R2, R5, R10 ;  /* 0x000000050200723e */
//                                                                                                       /* 0x004fc8000422780a */
//         /*0180*/                   F2FP.SATFINITE.RELU.E2M1.F32.PACK_AB_MERGE_C.RS R15, R7, R8, R0 ;  /* 0x00000008070f723e */
//                                                                                                       /* 0x010fca0004227800 */
//         /*0190*/                   STG.E.U16 desc[UR4][R12.64], R15 ;                                 /* 0x0000000f0c007986 */
//                                                                                                       /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rs_relu_satfinite_e2m1x4_f32(float a, float b, float e, float f, uint32_t rbits) {
    uint16_t out;
    asm volatile("cvt.rs.relu.satfinite.e2m1x4.f32 %0, {%1, %2, %3, %4}, %5;"
                 : "=h"(out) : "f"(a), "f"(b), "f"(e), "f"(f), "r"(rbits));
    return out;
}

extern "C" __global__ void cvt_f4x2_f4x4_kernel(
    const float* __restrict__ in_a,
    const float* __restrict__ in_b,
    const float* __restrict__ in_e,
    const float* __restrict__ in_f,
    const uint32_t* __restrict__ in_f16x2,
    const uint32_t* __restrict__ in_bf16x2,
    const uint16_t* __restrict__ in_e2m1x2,
    const uint32_t* __restrict__ in_rbits,
    uint16_t* __restrict__ out_e2m1x2_f32,
    uint16_t* __restrict__ out_e2m1x2_f32_relu,
    uint16_t* __restrict__ out_e2m1x2_f16x2,
    uint16_t* __restrict__ out_e2m1x2_f16x2_relu,
    uint16_t* __restrict__ out_e2m1x2_bf16x2,
    uint16_t* __restrict__ out_e2m1x2_bf16x2_relu,
    uint32_t* __restrict__ out_f16x2_e2m1x2,
    uint32_t* __restrict__ out_f16x2_e2m1x2_relu,
    uint16_t* __restrict__ out_e2m1x4_f32,
    uint16_t* __restrict__ out_e2m1x4_f32_relu
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    float a = in_a[tid];
    float b = in_b[tid];
    float e = in_e[tid];
    float f = in_f[tid];
    uint32_t f16x2 = in_f16x2[tid];
    uint32_t bf16x2 = in_bf16x2[tid];
    uint16_t e2m1x2 = in_e2m1x2[tid];
    uint32_t rbits = in_rbits[tid];

    out_e2m1x2_f32[tid] = cvt_rn_satfinite_e2m1x2_f32(a, b);
    out_e2m1x2_f32_relu[tid] = cvt_rn_relu_satfinite_e2m1x2_f32(a, b);
    out_e2m1x2_f16x2[tid] = cvt_rn_satfinite_e2m1x2_f16x2(f16x2);
    out_e2m1x2_f16x2_relu[tid] = cvt_rn_relu_satfinite_e2m1x2_f16x2(f16x2);
    out_e2m1x2_bf16x2[tid] = cvt_rn_satfinite_e2m1x2_bf16x2(bf16x2);
    out_e2m1x2_bf16x2_relu[tid] = cvt_rn_relu_satfinite_e2m1x2_bf16x2(bf16x2);
    out_f16x2_e2m1x2[tid] = cvt_rn_f16x2_e2m1x2(e2m1x2);
    out_f16x2_e2m1x2_relu[tid] = cvt_rn_relu_f16x2_e2m1x2(e2m1x2);
    out_e2m1x4_f32[tid] = cvt_rs_satfinite_e2m1x4_f32(a, b, e, f, rbits);
    out_e2m1x4_f32_relu[tid] = cvt_rs_relu_satfinite_e2m1x4_f32(a, b, e, f, rbits);
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
    float* in_e;
    float* in_f;
    uint32_t* in_f16x2;
    uint32_t* in_bf16x2;
    uint16_t* in_e2m1x2;
    uint32_t* in_rbits;

    uint16_t* out_e2m1x2_f32;
    uint16_t* out_e2m1x2_f32_relu;
    uint16_t* out_e2m1x2_f16x2;
    uint16_t* out_e2m1x2_f16x2_relu;
    uint16_t* out_e2m1x2_bf16x2;
    uint16_t* out_e2m1x2_bf16x2_relu;
    uint32_t* out_f16x2_e2m1x2;
    uint32_t* out_f16x2_e2m1x2_relu;
    uint16_t* out_e2m1x4_f32;
    uint16_t* out_e2m1x4_f32_relu;

    ck(cudaMallocManaged(&in_a, N * sizeof(float)), "cudaMallocManaged in_a");
    ck(cudaMallocManaged(&in_b, N * sizeof(float)), "cudaMallocManaged in_b");
    ck(cudaMallocManaged(&in_e, N * sizeof(float)), "cudaMallocManaged in_e");
    ck(cudaMallocManaged(&in_f, N * sizeof(float)), "cudaMallocManaged in_f");
    ck(cudaMallocManaged(&in_f16x2, N * sizeof(uint32_t)), "cudaMallocManaged in_f16x2");
    ck(cudaMallocManaged(&in_bf16x2, N * sizeof(uint32_t)), "cudaMallocManaged in_bf16x2");
    ck(cudaMallocManaged(&in_e2m1x2, N * sizeof(uint16_t)), "cudaMallocManaged in_e2m1x2");
    ck(cudaMallocManaged(&in_rbits, N * sizeof(uint32_t)), "cudaMallocManaged in_rbits");

    ck(cudaMallocManaged(&out_e2m1x2_f32, N * sizeof(uint16_t)), "cudaMallocManaged out_e2m1x2_f32");
    ck(cudaMallocManaged(&out_e2m1x2_f32_relu, N * sizeof(uint16_t)),
        "cudaMallocManaged out_e2m1x2_f32_relu");
    ck(cudaMallocManaged(&out_e2m1x2_f16x2, N * sizeof(uint16_t)), "cudaMallocManaged out_e2m1x2_f16x2");
    ck(cudaMallocManaged(&out_e2m1x2_f16x2_relu, N * sizeof(uint16_t)),
        "cudaMallocManaged out_e2m1x2_f16x2_relu");
    ck(cudaMallocManaged(&out_e2m1x2_bf16x2, N * sizeof(uint16_t)), "cudaMallocManaged out_e2m1x2_bf16x2");
    ck(cudaMallocManaged(&out_e2m1x2_bf16x2_relu, N * sizeof(uint16_t)),
        "cudaMallocManaged out_e2m1x2_bf16x2_relu");
    ck(cudaMallocManaged(&out_f16x2_e2m1x2, N * sizeof(uint32_t)), "cudaMallocManaged out_f16x2_e2m1x2");
    ck(cudaMallocManaged(&out_f16x2_e2m1x2_relu, N * sizeof(uint32_t)),
        "cudaMallocManaged out_f16x2_e2m1x2_relu");
    ck(cudaMallocManaged(&out_e2m1x4_f32, N * sizeof(uint16_t)), "cudaMallocManaged out_e2m1x4_f32");
    ck(cudaMallocManaged(&out_e2m1x4_f32_relu, N * sizeof(uint16_t)),
        "cudaMallocManaged out_e2m1x4_f32_relu");

    for (int i = 0; i < N; ++i) {
        float base = (float)(i * 0.5f + 0.25f);
        in_a[i] = base;
        in_b[i] = -base * 0.75f;
        in_e[i] = base + 1.0f;
        in_f[i] = -base + 0.5f;
        in_f16x2[i] = 0x3c003c00u + (uint32_t)(i & 0xffu);
        in_bf16x2[i] = 0x3f803f80u + (uint32_t)(i & 0xffu);
        in_e2m1x2[i] = (uint16_t)(0x0f00u + (uint16_t)(i & 0xffu));
        in_rbits[i] = 0x12340000u ^ (uint32_t)i;
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    cvt_f4x2_f4x4_kernel<<<grid, block>>>(
        in_a,
        in_b,
        in_e,
        in_f,
        in_f16x2,
        in_bf16x2,
        in_e2m1x2,
        in_rbits,
        out_e2m1x2_f32,
        out_e2m1x2_f32_relu,
        out_e2m1x2_f16x2,
        out_e2m1x2_f16x2_relu,
        out_e2m1x2_bf16x2,
        out_e2m1x2_bf16x2_relu,
        out_f16x2_e2m1x2,
        out_f16x2_e2m1x2_relu,
        out_e2m1x4_f32,
        out_e2m1x4_f32_relu
    );

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("e2m1x2=0x%04x e2m1x4=0x%04x\n",
        (unsigned int)out_e2m1x2_f32[0],
        (unsigned int)out_e2m1x4_f32[0]);

    cudaFree(in_a);
    cudaFree(in_b);
    cudaFree(in_e);
    cudaFree(in_f);
    cudaFree(in_f16x2);
    cudaFree(in_bf16x2);
    cudaFree(in_e2m1x2);
    cudaFree(in_rbits);
    cudaFree(out_e2m1x2_f32);
    cudaFree(out_e2m1x2_f32_relu);
    cudaFree(out_e2m1x2_f16x2);
    cudaFree(out_e2m1x2_f16x2_relu);
    cudaFree(out_e2m1x2_bf16x2);
    cudaFree(out_e2m1x2_bf16x2_relu);
    cudaFree(out_f16x2_e2m1x2);
    cudaFree(out_f16x2_e2m1x2_relu);
    cudaFree(out_e2m1x4_f32);
    cudaFree(out_e2m1x4_f32_relu);
    return 0;
}
