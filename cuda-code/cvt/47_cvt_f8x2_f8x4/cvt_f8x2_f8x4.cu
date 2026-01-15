// cvt_f8x2_f8x4.cu

// cvt.rn.satfinite{.relu}.f8x2type.f32       d, a, b;
// cvt.rn.satfinite{.relu}.f8x2type.fp16x2    d, a;
// cvt.rn.{.relu}.f16x2.f8x2type              d, a;
// cvt.rs{.relu}.satfinite.f8x4type.f32       d, {a, b, e, f}, rbits;

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

        // /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;                     /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;                               /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;                     /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x2, R6 ;                               /* 0x0000000209067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C R9, R3, R4, RZ ;  /* 0x000000040309723e */
        // /* 0x004fca00048070ff */
        // /*00f0*/                   STG.E.U16 desc[UR4][R6.64], R9 ;                          /* 0x0000000906007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_satfinite_e4m3x2_f32(float a, float b) {
    uint16_t out;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(out) : "f"(a), "f"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;                          /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;                                    /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;                          /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x2, R6 ;                                    /* 0x0000000209067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   F2FP.SATFINITE.RELU.E4M3.F32.PACK_AB_MERGE_C R9, R3, R4, RZ ;  /* 0x000000040309723e */
        // /* 0x004fca00048078ff */
        // /*00f0*/                   STG.E.U16 desc[UR4][R6.64], R9 ;                               /* 0x0000000906007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_relu_satfinite_e4m3x2_f32(float a, float b) {
    uint16_t out;
    asm volatile("cvt.rn.relu.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(out) : "f"(a), "f"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;                     /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;                               /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;                     /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x2, R6 ;                               /* 0x0000000209067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   F2FP.SATFINITE.E5M2.F32.PACK_AB_MERGE_C R9, R3, R4, RZ ;  /* 0x000000040309723e */
        // /* 0x004fca00048060ff */
        // /*00f0*/                   STG.E.U16 desc[UR4][R6.64], R9 ;                          /* 0x0000000906007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_satfinite_e5m2x2_f32(float a, float b) {
    uint16_t out;
    asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(out) : "f"(a), "f"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;                          /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;                                    /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;                          /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x2, R6 ;                                    /* 0x0000000209067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   F2FP.SATFINITE.RELU.E5M2.F32.PACK_AB_MERGE_C R9, R3, R4, RZ ;  /* 0x000000040309723e */
        // /* 0x004fca00048068ff */
        // /*00f0*/                   STG.E.U16 desc[UR4][R6.64], R9 ;                               /* 0x0000000906007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_relu_satfinite_e5m2x2_f32(float a, float b) {
    uint16_t out;
    asm volatile("cvt.rn.relu.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(out) : "f"(a), "f"(b));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                            /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.SATFINITE.E4M3.F16.UNPACK_B_MERGE_C R7, R2, RZ ;  /* 0x00000002ff07723e */
        // /* 0x004fca00048032ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;                       /* 0x0000000704007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_satfinite_e4m3x2_f16x2(uint32_t a) {
    uint16_t out;
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                       /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                                 /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.SATFINITE.RELU.E4M3.F16.UNPACK_B_MERGE_C R7, R2, RZ ;  /* 0x00000002ff07723e */
        // /* 0x004fca0004803aff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;                            /* 0x0000000704007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_relu_satfinite_e4m3x2_f16x2(uint32_t a) {
    uint16_t out;
    asm volatile("cvt.rn.relu.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                            /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.SATFINITE.E5M2.F16.UNPACK_B_MERGE_C R7, R2, RZ ;  /* 0x00000002ff07723e */
        // /* 0x004fca00048022ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;                       /* 0x0000000704007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_satfinite_e5m2x2_f16x2(uint32_t a) {
    uint16_t out;
    asm volatile("cvt.rn.satfinite.e5m2x2.f16x2 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                       /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                                 /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.SATFINITE.RELU.E5M2.F16.UNPACK_B_MERGE_C R7, R2, RZ ;  /* 0x00000002ff07723e */
        // /* 0x004fca0004802aff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;                            /* 0x0000000704007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_relu_satfinite_e5m2x2_f16x2(uint32_t a) {
    uint16_t out;
    asm volatile("cvt.rn.relu.satfinite.e5m2x2.f16x2 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                   /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                             /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.SATFINITE.E4M3.BF16.UNPACK_B_MERGE_C R7, R2, RZ ;  /* 0x00000002ff07723e */
        // /* 0x004fca00048832ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;                        /* 0x0000000704007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_satfinite_e4m3x2_bf16x2(uint32_t a) {
    uint16_t out;
    asm volatile("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                        /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.SATFINITE.RELU.E4M3.BF16.UNPACK_B_MERGE_C R7, R2, RZ ;  /* 0x00000002ff07723e */
        // /* 0x004fca0004883aff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;                             /* 0x0000000704007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_relu_satfinite_e4m3x2_bf16x2(uint32_t a) {
    uint16_t out;
    asm volatile("cvt.rn.relu.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                   /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                             /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.SATFINITE.E5M2.BF16.UNPACK_B_MERGE_C R7, R2, RZ ;  /* 0x00000002ff07723e */
        // /* 0x004fca00048822ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;                        /* 0x0000000704007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_satfinite_e5m2x2_bf16x2(uint32_t a) {
    uint16_t out;
    asm volatile("cvt.rn.satfinite.e5m2x2.bf16x2 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                        /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.SATFINITE.RELU.E5M2.BF16.UNPACK_B_MERGE_C R7, R2, RZ ;  /* 0x00000002ff07723e */
        // /* 0x004fca0004882aff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;                             /* 0x0000000704007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_relu_satfinite_e5m2x2_bf16x2(uint32_t a) {
    uint16_t out;
    asm volatile("cvt.rn.relu.satfinite.e5m2x2.bf16x2 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.F16.E4M3.UNPACK_B R7, R2 ;            /* 0x00000002ff07723e */
        // /* 0x004fca00020006ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;               /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_f16x2_e4m3x2(uint16_t a) {
    uint32_t out;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.RELU.F16.E4M3.UNPACK_B R7, R2 ;       /* 0x00000002ff07723e */
        // /* 0x004fca0002000eff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;               /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_relu_f16x2_e4m3x2(uint16_t a) {
    uint32_t out;
    asm volatile("cvt.rn.relu.f16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.F16.E5M2.UNPACK_B R7, R2 ;            /* 0x00000002ff07723e */
        // /* 0x004fca00020004ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;               /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_f16x2_e5m2x2(uint16_t a) {
    uint32_t out;
    asm volatile("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.RELU.F16.E5M2.UNPACK_B R7, R2 ;       /* 0x00000002ff07723e */
        // /* 0x004fca0002000cff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;               /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_relu_f16x2_e5m2x2(uint16_t a) {
    uint32_t out;
    asm volatile("cvt.rn.relu.f16x2.e5m2x2 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*00b0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                         /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   LDC.64 R8, c[0x0][0x398] ;                                    /* 0x0000e600ff087b82 */
        // /* 0x000e620000000a00 */
        // /*00d0*/                   IMAD.WIDE R4, R15, 0x4, R4 ;                                  /* 0x000000040f047825 */
        // /* 0x008fcc00078e0204 */
        // /*00e0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;                         /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00f0*/                   IMAD.WIDE R10, R15.reuse, 0x4, R10 ;                          /* 0x000000040f0a7825 */
        // /* 0x050fe200078e020a */
        // /*0100*/                   LDC.64 R12, c[0x0][0x448] ;                                   /* 0x00011200ff0c7b82 */
        // /* 0x000eea0000000a00 */
        // /*0110*/                   LDG.E.CONSTANT R10, desc[UR4][R10.64] ;                       /* 0x000000040a0a7981 */
        // /* 0x000ea2000c1e9900 */
        // /*0120*/                   IMAD.WIDE R6, R15, 0x4, R6 ;                                  /* 0x000000040f067825 */
        // /* 0x001fcc00078e0206 */
        // /*0130*/                   LDG.E.CONSTANT R7, desc[UR4][R6.64] ;                         /* 0x0000000406077981 */
        // /* 0x000f22000c1e9900 */
        // /*0140*/                   IMAD.WIDE R8, R15, 0x4, R8 ;                                  /* 0x000000040f087825 */
        // /* 0x002fcc00078e0208 */
        // /*0150*/                   LDG.E.CONSTANT R8, desc[UR4][R8.64] ;                         /* 0x0000000408087981 */
        // /* 0x000f22000c1e9900 */
        // /*0160*/                   IMAD.WIDE R12, R15, 0x4, R12 ;                                /* 0x000000040f0c7825 */
        // /* 0x008fe200078e020c */
        // /*0170*/                   F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C.RS R0, R2, R5, R10 ;  /* 0x000000050200723e */
        // /* 0x004fc8000482700a */
        // /*0180*/                   F2FP.SATFINITE.E4M3.F32.PACK_AB_MERGE_C.RS R15, R7, R8, R0 ;  /* 0x00000008070f723e */
        // /* 0x010fca0004827000 */
        // /*0190*/                   STG.E desc[UR4][R12.64], R15 ;                                /* 0x0000000f0c007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rs_satfinite_e4m3x4_f32(float a, float b, float e, float f, uint32_t rbits) {
    uint32_t out;
    asm volatile("cvt.rs.satfinite.e4m3x4.f32 %0, {%1, %2, %3, %4}, %5;"
                 : "=r"(out) : "f"(a), "f"(b), "f"(e), "f"(f), "r"(rbits));
    return out;
}

        // /*00b0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                              /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   LDC.64 R8, c[0x0][0x398] ;                                         /* 0x0000e600ff087b82 */
        // /* 0x000e620000000a00 */
        // /*00d0*/                   IMAD.WIDE R4, R15, 0x4, R4 ;                                       /* 0x000000040f047825 */
        // /* 0x008fcc00078e0204 */
        // /*00e0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;                              /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00f0*/                   IMAD.WIDE R10, R15.reuse, 0x4, R10 ;                               /* 0x000000040f0a7825 */
        // /* 0x050fe200078e020a */
        // /*0100*/                   LDC.64 R12, c[0x0][0x450] ;                                        /* 0x00011400ff0c7b82 */
        // /* 0x000eea0000000a00 */
        // /*0110*/                   LDG.E.CONSTANT R10, desc[UR4][R10.64] ;                            /* 0x000000040a0a7981 */
        // /* 0x000ea2000c1e9900 */
        // /*0120*/                   IMAD.WIDE R6, R15, 0x4, R6 ;                                       /* 0x000000040f067825 */
        // /* 0x001fcc00078e0206 */
        // /*0130*/                   LDG.E.CONSTANT R7, desc[UR4][R6.64] ;                              /* 0x0000000406077981 */
        // /* 0x000f22000c1e9900 */
        // /*0140*/                   IMAD.WIDE R8, R15, 0x4, R8 ;                                       /* 0x000000040f087825 */
        // /* 0x002fcc00078e0208 */
        // /*0150*/                   LDG.E.CONSTANT R8, desc[UR4][R8.64] ;                              /* 0x0000000408087981 */
        // /* 0x000f22000c1e9900 */
        // /*0160*/                   IMAD.WIDE R12, R15, 0x4, R12 ;                                     /* 0x000000040f0c7825 */
        // /* 0x008fe200078e020c */
        // /*0170*/                   F2FP.SATFINITE.RELU.E4M3.F32.PACK_AB_MERGE_C.RS R0, R2, R5, R10 ;  /* 0x000000050200723e */
        // /* 0x004fc8000482780a */
        // /*0180*/                   F2FP.SATFINITE.RELU.E4M3.F32.PACK_AB_MERGE_C.RS R15, R7, R8, R0 ;  /* 0x00000008070f723e */
        // /* 0x010fca0004827800 */
        // /*0190*/                   STG.E desc[UR4][R12.64], R15 ;                                     /* 0x0000000f0c007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rs_relu_satfinite_e4m3x4_f32(float a, float b, float e, float f, uint32_t rbits) {
    uint32_t out;
    asm volatile("cvt.rs.relu.satfinite.e4m3x4.f32 %0, {%1, %2, %3, %4}, %5;"
                 : "=r"(out) : "f"(a), "f"(b), "f"(e), "f"(f), "r"(rbits));
    return out;
}

        // /*00b0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                         /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   LDC.64 R8, c[0x0][0x398] ;                                    /* 0x0000e600ff087b82 */
        // /* 0x000e620000000a00 */
        // /*00d0*/                   IMAD.WIDE R4, R15, 0x4, R4 ;                                  /* 0x000000040f047825 */
        // /* 0x008fcc00078e0204 */
        // /*00e0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;                         /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00f0*/                   IMAD.WIDE R10, R15.reuse, 0x4, R10 ;                          /* 0x000000040f0a7825 */
        // /* 0x050fe200078e020a */
        // /*0100*/                   LDC.64 R12, c[0x0][0x458] ;                                   /* 0x00011600ff0c7b82 */
        // /* 0x000eea0000000a00 */
        // /*0110*/                   LDG.E.CONSTANT R10, desc[UR4][R10.64] ;                       /* 0x000000040a0a7981 */
        // /* 0x000ea2000c1e9900 */
        // /*0120*/                   IMAD.WIDE R6, R15, 0x4, R6 ;                                  /* 0x000000040f067825 */
        // /* 0x001fcc00078e0206 */
        // /*0130*/                   LDG.E.CONSTANT R7, desc[UR4][R6.64] ;                         /* 0x0000000406077981 */
        // /* 0x000f22000c1e9900 */
        // /*0140*/                   IMAD.WIDE R8, R15, 0x4, R8 ;                                  /* 0x000000040f087825 */
        // /* 0x002fcc00078e0208 */
        // /*0150*/                   LDG.E.CONSTANT R8, desc[UR4][R8.64] ;                         /* 0x0000000408087981 */
        // /* 0x000f22000c1e9900 */
        // /*0160*/                   IMAD.WIDE R12, R15, 0x4, R12 ;                                /* 0x000000040f0c7825 */
        // /* 0x008fe200078e020c */
        // /*0170*/                   F2FP.SATFINITE.E5M2.F32.PACK_AB_MERGE_C.RS R0, R2, R5, R10 ;  /* 0x000000050200723e */
        // /* 0x004fc8000482600a */
        // /*0180*/                   F2FP.SATFINITE.E5M2.F32.PACK_AB_MERGE_C.RS R15, R7, R8, R0 ;  /* 0x00000008070f723e */
        // /* 0x010fca0004826000 */
        // /*0190*/                   STG.E desc[UR4][R12.64], R15 ;                                /* 0x0000000f0c007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rs_satfinite_e5m2x4_f32(float a, float b, float e, float f, uint32_t rbits) {
    uint32_t out;
    asm volatile("cvt.rs.satfinite.e5m2x4.f32 %0, {%1, %2, %3, %4}, %5;"
                 : "=r"(out) : "f"(a), "f"(b), "f"(e), "f"(f), "r"(rbits));
    return out;
}

        // /*00b0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                              /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   LDC.64 R8, c[0x0][0x398] ;                                         /* 0x0000e600ff087b82 */
        // /* 0x000e620000000a00 */
        // /*00d0*/                   IMAD.WIDE R4, R15, 0x4, R4 ;                                       /* 0x000000040f047825 */
        // /* 0x008fcc00078e0204 */
        // /*00e0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;                              /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00f0*/                   IMAD.WIDE R10, R15.reuse, 0x4, R10 ;                               /* 0x000000040f0a7825 */
        // /* 0x050fe200078e020a */
        // /*0100*/                   LDC.64 R12, c[0x0][0x460] ;                                        /* 0x00011800ff0c7b82 */
        // /* 0x000eea0000000a00 */
        // /*0110*/                   LDG.E.CONSTANT R10, desc[UR4][R10.64] ;                            /* 0x000000040a0a7981 */
        // /* 0x000ea2000c1e9900 */
        // /*0120*/                   IMAD.WIDE R6, R15, 0x4, R6 ;                                       /* 0x000000040f067825 */
        // /* 0x001fcc00078e0206 */
        // /*0130*/                   LDG.E.CONSTANT R7, desc[UR4][R6.64] ;                              /* 0x0000000406077981 */
        // /* 0x000f22000c1e9900 */
        // /*0140*/                   IMAD.WIDE R8, R15, 0x4, R8 ;                                       /* 0x000000040f087825 */
        // /* 0x002fcc00078e0208 */
        // /*0150*/                   LDG.E.CONSTANT R8, desc[UR4][R8.64] ;                              /* 0x0000000408087981 */
        // /* 0x000f22000c1e9900 */
        // /*0160*/                   IMAD.WIDE R12, R15, 0x4, R12 ;                                     /* 0x000000040f0c7825 */
        // /* 0x008fe200078e020c */
        // /*0170*/                   F2FP.SATFINITE.RELU.E5M2.F32.PACK_AB_MERGE_C.RS R0, R2, R5, R10 ;  /* 0x000000050200723e */
        // /* 0x004fc8000482680a */
        // /*0180*/                   F2FP.SATFINITE.RELU.E5M2.F32.PACK_AB_MERGE_C.RS R15, R7, R8, R0 ;  /* 0x00000008070f723e */
        // /* 0x010fca0004826800 */
        // /*0190*/                   STG.E desc[UR4][R12.64], R15 ;                                     /* 0x0000000f0c007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rs_relu_satfinite_e5m2x4_f32(float a, float b, float e, float f, uint32_t rbits) {
    uint32_t out;
    asm volatile("cvt.rs.relu.satfinite.e5m2x4.f32 %0, {%1, %2, %3, %4}, %5;"
                 : "=r"(out) : "f"(a), "f"(b), "f"(e), "f"(f), "r"(rbits));
    return out;
}

extern "C" __global__ void cvt_f8x2_f8x4_kernel(
    const float* __restrict__ in_a,
    const float* __restrict__ in_b,
    const float* __restrict__ in_e,
    const float* __restrict__ in_f,
    const uint32_t* __restrict__ in_f16x2,
    const uint32_t* __restrict__ in_bf16x2,
    const uint16_t* __restrict__ in_e4m3x2,
    const uint16_t* __restrict__ in_e5m2x2,
    const uint32_t* __restrict__ in_rbits,
    uint16_t* __restrict__ out_e4m3x2_f32,
    uint16_t* __restrict__ out_e4m3x2_f32_relu,
    uint16_t* __restrict__ out_e5m2x2_f32,
    uint16_t* __restrict__ out_e5m2x2_f32_relu,
    uint16_t* __restrict__ out_e4m3x2_f16x2,
    uint16_t* __restrict__ out_e4m3x2_f16x2_relu,
    uint16_t* __restrict__ out_e5m2x2_f16x2,
    uint16_t* __restrict__ out_e5m2x2_f16x2_relu,
    uint16_t* __restrict__ out_e4m3x2_bf16x2,
    uint16_t* __restrict__ out_e4m3x2_bf16x2_relu,
    uint16_t* __restrict__ out_e5m2x2_bf16x2,
    uint16_t* __restrict__ out_e5m2x2_bf16x2_relu,
    uint32_t* __restrict__ out_f16x2_e4m3x2,
    uint32_t* __restrict__ out_f16x2_e4m3x2_relu,
    uint32_t* __restrict__ out_f16x2_e5m2x2,
    uint32_t* __restrict__ out_f16x2_e5m2x2_relu,
    uint32_t* __restrict__ out_e4m3x4_f32,
    uint32_t* __restrict__ out_e4m3x4_f32_relu,
    uint32_t* __restrict__ out_e5m2x4_f32,
    uint32_t* __restrict__ out_e5m2x4_f32_relu
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    float a = in_a[tid];
    float b = in_b[tid];
    float e = in_e[tid];
    float f = in_f[tid];
    uint32_t f16x2 = in_f16x2[tid];
    uint32_t bf16x2 = in_bf16x2[tid];
    uint16_t e4m3x2 = in_e4m3x2[tid];
    uint16_t e5m2x2 = in_e5m2x2[tid];
    uint32_t rbits = in_rbits[tid];

    out_e4m3x2_f32[tid] = cvt_rn_satfinite_e4m3x2_f32(a, b);
    out_e4m3x2_f32_relu[tid] = cvt_rn_relu_satfinite_e4m3x2_f32(a, b);
    out_e5m2x2_f32[tid] = cvt_rn_satfinite_e5m2x2_f32(a, b);
    out_e5m2x2_f32_relu[tid] = cvt_rn_relu_satfinite_e5m2x2_f32(a, b);

    out_e4m3x2_f16x2[tid] = cvt_rn_satfinite_e4m3x2_f16x2(f16x2);
    out_e4m3x2_f16x2_relu[tid] = cvt_rn_relu_satfinite_e4m3x2_f16x2(f16x2);
    out_e5m2x2_f16x2[tid] = cvt_rn_satfinite_e5m2x2_f16x2(f16x2);
    out_e5m2x2_f16x2_relu[tid] = cvt_rn_relu_satfinite_e5m2x2_f16x2(f16x2);

    out_e4m3x2_bf16x2[tid] = cvt_rn_satfinite_e4m3x2_bf16x2(bf16x2);
    out_e4m3x2_bf16x2_relu[tid] = cvt_rn_relu_satfinite_e4m3x2_bf16x2(bf16x2);
    out_e5m2x2_bf16x2[tid] = cvt_rn_satfinite_e5m2x2_bf16x2(bf16x2);
    out_e5m2x2_bf16x2_relu[tid] = cvt_rn_relu_satfinite_e5m2x2_bf16x2(bf16x2);

    out_f16x2_e4m3x2[tid] = cvt_rn_f16x2_e4m3x2(e4m3x2);
    out_f16x2_e4m3x2_relu[tid] = cvt_rn_relu_f16x2_e4m3x2(e4m3x2);
    out_f16x2_e5m2x2[tid] = cvt_rn_f16x2_e5m2x2(e5m2x2);
    out_f16x2_e5m2x2_relu[tid] = cvt_rn_relu_f16x2_e5m2x2(e5m2x2);

    out_e4m3x4_f32[tid] = cvt_rs_satfinite_e4m3x4_f32(a, b, e, f, rbits);
    out_e4m3x4_f32_relu[tid] = cvt_rs_relu_satfinite_e4m3x4_f32(a, b, e, f, rbits);
    out_e5m2x4_f32[tid] = cvt_rs_satfinite_e5m2x4_f32(a, b, e, f, rbits);
    out_e5m2x4_f32_relu[tid] = cvt_rs_relu_satfinite_e5m2x4_f32(a, b, e, f, rbits);
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
    uint16_t* in_e4m3x2;
    uint16_t* in_e5m2x2;
    uint32_t* in_rbits;

    uint16_t* out_e4m3x2_f32;
    uint16_t* out_e4m3x2_f32_relu;
    uint16_t* out_e5m2x2_f32;
    uint16_t* out_e5m2x2_f32_relu;
    uint16_t* out_e4m3x2_f16x2;
    uint16_t* out_e4m3x2_f16x2_relu;
    uint16_t* out_e5m2x2_f16x2;
    uint16_t* out_e5m2x2_f16x2_relu;
    uint16_t* out_e4m3x2_bf16x2;
    uint16_t* out_e4m3x2_bf16x2_relu;
    uint16_t* out_e5m2x2_bf16x2;
    uint16_t* out_e5m2x2_bf16x2_relu;
    uint32_t* out_f16x2_e4m3x2;
    uint32_t* out_f16x2_e4m3x2_relu;
    uint32_t* out_f16x2_e5m2x2;
    uint32_t* out_f16x2_e5m2x2_relu;
    uint32_t* out_e4m3x4_f32;
    uint32_t* out_e4m3x4_f32_relu;
    uint32_t* out_e5m2x4_f32;
    uint32_t* out_e5m2x4_f32_relu;

    ck(cudaMallocManaged(&in_a, N * sizeof(float)), "cudaMallocManaged in_a");
    ck(cudaMallocManaged(&in_b, N * sizeof(float)), "cudaMallocManaged in_b");
    ck(cudaMallocManaged(&in_e, N * sizeof(float)), "cudaMallocManaged in_e");
    ck(cudaMallocManaged(&in_f, N * sizeof(float)), "cudaMallocManaged in_f");
    ck(cudaMallocManaged(&in_f16x2, N * sizeof(uint32_t)), "cudaMallocManaged in_f16x2");
    ck(cudaMallocManaged(&in_bf16x2, N * sizeof(uint32_t)), "cudaMallocManaged in_bf16x2");
    ck(cudaMallocManaged(&in_e4m3x2, N * sizeof(uint16_t)), "cudaMallocManaged in_e4m3x2");
    ck(cudaMallocManaged(&in_e5m2x2, N * sizeof(uint16_t)), "cudaMallocManaged in_e5m2x2");
    ck(cudaMallocManaged(&in_rbits, N * sizeof(uint32_t)), "cudaMallocManaged in_rbits");

    ck(cudaMallocManaged(&out_e4m3x2_f32, N * sizeof(uint16_t)), "cudaMallocManaged out_e4m3x2_f32");
    ck(cudaMallocManaged(&out_e4m3x2_f32_relu, N * sizeof(uint16_t)), "cudaMallocManaged out_e4m3x2_f32_relu");
    ck(cudaMallocManaged(&out_e5m2x2_f32, N * sizeof(uint16_t)), "cudaMallocManaged out_e5m2x2_f32");
    ck(cudaMallocManaged(&out_e5m2x2_f32_relu, N * sizeof(uint16_t)), "cudaMallocManaged out_e5m2x2_f32_relu");
    ck(cudaMallocManaged(&out_e4m3x2_f16x2, N * sizeof(uint16_t)), "cudaMallocManaged out_e4m3x2_f16x2");
    ck(cudaMallocManaged(&out_e4m3x2_f16x2_relu, N * sizeof(uint16_t)), "cudaMallocManaged out_e4m3x2_f16x2_relu");
    ck(cudaMallocManaged(&out_e5m2x2_f16x2, N * sizeof(uint16_t)), "cudaMallocManaged out_e5m2x2_f16x2");
    ck(cudaMallocManaged(&out_e5m2x2_f16x2_relu, N * sizeof(uint16_t)), "cudaMallocManaged out_e5m2x2_f16x2_relu");
    ck(cudaMallocManaged(&out_e4m3x2_bf16x2, N * sizeof(uint16_t)), "cudaMallocManaged out_e4m3x2_bf16x2");
    ck(cudaMallocManaged(&out_e4m3x2_bf16x2_relu, N * sizeof(uint16_t)),
        "cudaMallocManaged out_e4m3x2_bf16x2_relu");
    ck(cudaMallocManaged(&out_e5m2x2_bf16x2, N * sizeof(uint16_t)), "cudaMallocManaged out_e5m2x2_bf16x2");
    ck(cudaMallocManaged(&out_e5m2x2_bf16x2_relu, N * sizeof(uint16_t)),
        "cudaMallocManaged out_e5m2x2_bf16x2_relu");
    ck(cudaMallocManaged(&out_f16x2_e4m3x2, N * sizeof(uint32_t)), "cudaMallocManaged out_f16x2_e4m3x2");
    ck(cudaMallocManaged(&out_f16x2_e4m3x2_relu, N * sizeof(uint32_t)),
        "cudaMallocManaged out_f16x2_e4m3x2_relu");
    ck(cudaMallocManaged(&out_f16x2_e5m2x2, N * sizeof(uint32_t)), "cudaMallocManaged out_f16x2_e5m2x2");
    ck(cudaMallocManaged(&out_f16x2_e5m2x2_relu, N * sizeof(uint32_t)),
        "cudaMallocManaged out_f16x2_e5m2x2_relu");
    ck(cudaMallocManaged(&out_e4m3x4_f32, N * sizeof(uint32_t)), "cudaMallocManaged out_e4m3x4_f32");
    ck(cudaMallocManaged(&out_e4m3x4_f32_relu, N * sizeof(uint32_t)),
        "cudaMallocManaged out_e4m3x4_f32_relu");
    ck(cudaMallocManaged(&out_e5m2x4_f32, N * sizeof(uint32_t)), "cudaMallocManaged out_e5m2x4_f32");
    ck(cudaMallocManaged(&out_e5m2x4_f32_relu, N * sizeof(uint32_t)),
        "cudaMallocManaged out_e5m2x4_f32_relu");

    for (int i = 0; i < N; ++i) {
        float base = (float)(i * 0.5f + 0.25f);
        in_a[i] = base;
        in_b[i] = -base * 0.75f;
        in_e[i] = base + 1.0f;
        in_f[i] = -base + 0.5f;
        in_f16x2[i] = 0x3c003c00u + (uint32_t)(i & 0xffu);
        in_bf16x2[i] = 0x3f803f80u + (uint32_t)(i & 0xffu);
        in_e4m3x2[i] = (uint16_t)(0x5a00u + (uint16_t)(i & 0xffu));
        in_e5m2x2[i] = (uint16_t)(0x6b00u + (uint16_t)(i & 0xffu));
        in_rbits[i] = 0x12340000u ^ (uint32_t)i;
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    cvt_f8x2_f8x4_kernel<<<grid, block>>>(
        in_a,
        in_b,
        in_e,
        in_f,
        in_f16x2,
        in_bf16x2,
        in_e4m3x2,
        in_e5m2x2,
        in_rbits,
        out_e4m3x2_f32,
        out_e4m3x2_f32_relu,
        out_e5m2x2_f32,
        out_e5m2x2_f32_relu,
        out_e4m3x2_f16x2,
        out_e4m3x2_f16x2_relu,
        out_e5m2x2_f16x2,
        out_e5m2x2_f16x2_relu,
        out_e4m3x2_bf16x2,
        out_e4m3x2_bf16x2_relu,
        out_e5m2x2_bf16x2,
        out_e5m2x2_bf16x2_relu,
        out_f16x2_e4m3x2,
        out_f16x2_e4m3x2_relu,
        out_f16x2_e5m2x2,
        out_f16x2_e5m2x2_relu,
        out_e4m3x4_f32,
        out_e4m3x4_f32_relu,
        out_e5m2x4_f32,
        out_e5m2x4_f32_relu
    );

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("e4m3x2=0x%04x e5m2x2=0x%04x e4m3x4=0x%08x e5m2x4=0x%08x\n",
        (unsigned int)out_e4m3x2_f32[0],
        (unsigned int)out_e5m2x2_f32[0],
        out_e4m3x4_f32[0],
        out_e5m2x4_f32[0]);

    cudaFree(in_a);
    cudaFree(in_b);
    cudaFree(in_e);
    cudaFree(in_f);
    cudaFree(in_f16x2);
    cudaFree(in_bf16x2);
    cudaFree(in_e4m3x2);
    cudaFree(in_e5m2x2);
    cudaFree(in_rbits);
    cudaFree(out_e4m3x2_f32);
    cudaFree(out_e4m3x2_f32_relu);
    cudaFree(out_e5m2x2_f32);
    cudaFree(out_e5m2x2_f32_relu);
    cudaFree(out_e4m3x2_f16x2);
    cudaFree(out_e4m3x2_f16x2_relu);
    cudaFree(out_e5m2x2_f16x2);
    cudaFree(out_e5m2x2_f16x2_relu);
    cudaFree(out_e4m3x2_bf16x2);
    cudaFree(out_e4m3x2_bf16x2_relu);
    cudaFree(out_e5m2x2_bf16x2);
    cudaFree(out_e5m2x2_bf16x2_relu);
    cudaFree(out_f16x2_e4m3x2);
    cudaFree(out_f16x2_e4m3x2_relu);
    cudaFree(out_f16x2_e5m2x2);
    cudaFree(out_f16x2_e5m2x2_relu);
    cudaFree(out_e4m3x4_f32);
    cudaFree(out_e4m3x4_f32_relu);
    cudaFree(out_e5m2x4_f32);
    cudaFree(out_e5m2x4_f32_relu);
    return 0;
}
