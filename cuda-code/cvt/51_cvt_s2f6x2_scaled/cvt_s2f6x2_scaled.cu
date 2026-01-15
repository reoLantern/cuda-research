// cvt_s2f6x2_scaled.cu
//
// cvt.rn.satfinite{.relu}{.scaled::n2::ue8m0}.s2f6x2.f32      d, a, b{, scale-factor};
// cvt.rn.satfinite{.relu}{.scaled::n2::ue8m0}.s2f6x2.bf16x2   d, a{, scale-factor};
// cvt.rn{.satfinite}{.relu}{.scaled::n2::ue8m0}.bf16x2.s2f6x2 d, a{, scale-factor};

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

//         /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;                                             /* 0x0000000402037981 */
//                                                                                                                      /* 0x002ea2000c1e9900 */
//         /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;                                                       /* 0x0000000409047825 */
//                                                                                                                      /* 0x008fcc00078e0204 */
//         /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;                                             /* 0x0000000404047981 */
//                                                                                                                      /* 0x000ea2000c1e9900 */
//         /*00d0*/                   IMAD.WIDE R6, R9, 0x2, R6 ;                                                       /* 0x0000000209067825 */
//                                                                                                                      /* 0x001fe200078e0206 */
//         /*00e0*/                   F2FP.SATFINITE.S2_6.F32.PACK_AB_MERGE_C R9, R3, R4, 3.38953138925153547590e+38 ;  /* 0x7f7f00000309743e */
//                                                                                                                      /* 0x004fca0004a05004 */
//         /*00f0*/                   STG.E.U16 desc[UR4][R6.64], R9 ;                                                  /* 0x0000000906007986 */
//                                                                                                                      /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_sat_s2f6x2_f32(float a, float b) {
    uint16_t out;
    asm volatile("cvt.rn.satfinite.s2f6x2.f32 %0, %1, %2;" : "=h"(out) : "f"(a), "f"(b));
    return out;
}

//         /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;                                                  /* 0x0000000402037981 */
//                                                                                                                           /* 0x002ea2000c1e9900 */
//         /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;                                                            /* 0x0000000409047825 */
//                                                                                                                           /* 0x008fcc00078e0204 */
//         /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;                                                  /* 0x0000000404047981 */
//                                                                                                                           /* 0x000ea2000c1e9900 */
//         /*00d0*/                   IMAD.WIDE R6, R9, 0x2, R6 ;                                                            /* 0x0000000209067825 */
//                                                                                                                           /* 0x001fe200078e0206 */
//         /*00e0*/                   F2FP.SATFINITE.RELU.S2_6.F32.PACK_AB_MERGE_C R9, R3, R4, 3.38953138925153547590e+38 ;  /* 0x7f7f00000309743e */
//                                                                                                                           /* 0x004fca0004a05804 */
//         /*00f0*/                   STG.E.U16 desc[UR4][R6.64], R9 ;                                                       /* 0x0000000906007986 */
//                                                                                                                           /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_relu_sat_s2f6x2_f32(float a, float b) {
    uint16_t out;
    asm volatile("cvt.rn.relu.satfinite.s2f6x2.f32 %0, %1, %2;" : "=h"(out) : "f"(a), "f"(b));
    return out;
}

//         /*00b0*/                   LDG.E.U16.CONSTANT R6, desc[UR4][R6.64] ;                  /* 0x0000000406067981 */
//                                                                                               /* 0x002ea2000c1e9500 */
//         /*00c0*/                   IMAD.WIDE R2, R11, 0x4, R2 ;                               /* 0x000000040b027825 */
//                                                                                               /* 0x008fcc00078e0202 */
//         /*00d0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;                      /* 0x0000000402037981 */
//                                                                                               /* 0x000ee2000c1e9900 */
//         /*00e0*/                   IMAD.WIDE R4, R11, 0x4, R4 ;                               /* 0x000000040b047825 */
//                                                                                               /* 0x010fcc00078e0204 */
//         /*00f0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;                      /* 0x0000000404047981 */
//                                                                                               /* 0x000ee2000c1e9900 */
//         /*0100*/                   IMAD.WIDE R8, R11, 0x2, R8 ;                               /* 0x000000020b087825 */
//                                                                                               /* 0x001fe200078e0208 */
//         /*0110*/                   SHF.L.U32 R0, R6, 0x10, RZ ;                               /* 0x0000001006007819 */
//                                                                                               /* 0x004fc800000006ff */
//         /*0120*/                   F2FP.SATFINITE.S2_6.F32.PACK_AB_MERGE_C R11, R3, R4, R0 ;  /* 0x00000004030b723e */
//                                                                                               /* 0x008fca0004a05000 */
//         /*0130*/                   STG.E.U16 desc[UR4][R8.64], R11 ;                          /* 0x0000000b08007986 */
//                                                                                               /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_sat_scaled_s2f6x2_f32(float a, float b, uint16_t scale) {
    uint16_t out;
    asm volatile("cvt.rn.satfinite.scaled::n2::ue8m0.s2f6x2.f32 %0, %1, %2, %3;"
                 : "=h"(out) : "f"(a), "f"(b), "h"(scale));
    return out;
}

//         /*00b0*/                   LDG.E.U16.CONSTANT R6, desc[UR4][R6.64] ;                       /* 0x0000000406067981 */
//                                                                                                   /* 0x002ea2000c1e9500 */
//         /*00c0*/                   IMAD.WIDE R2, R11, 0x4, R2 ;                                    /* 0x000000040b027825 */
//                                                                                                   /* 0x008fcc00078e0202 */
//         /*00d0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;                           /* 0x0000000402037981 */
//                                                                                                   /* 0x000ee2000c1e9900 */
//         /*00e0*/                   IMAD.WIDE R4, R11, 0x4, R4 ;                                    /* 0x000000040b047825 */
//                                                                                                   /* 0x010fcc00078e0204 */
//         /*00f0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;                           /* 0x0000000404047981 */
//                                                                                                   /* 0x000ee2000c1e9900 */
//         /*0100*/                   IMAD.WIDE R8, R11, 0x2, R8 ;                                    /* 0x000000020b087825 */
//                                                                                                   /* 0x001fe200078e0208 */
//         /*0110*/                   SHF.L.U32 R0, R6, 0x10, RZ ;                                    /* 0x0000001006007819 */
//                                                                                                   /* 0x004fc800000006ff */
//         /*0120*/                   F2FP.SATFINITE.RELU.S2_6.F32.PACK_AB_MERGE_C R11, R3, R4, R0 ;  /* 0x00000004030b723e */
//                                                                                                   /* 0x008fca0004a05800 */
//         /*0130*/                   STG.E.U16 desc[UR4][R8.64], R11 ;                               /* 0x0000000b08007986 */
//                                                                                                   /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_relu_sat_scaled_s2f6x2_f32(float a, float b, uint16_t scale) {
    uint16_t out;
    asm volatile("cvt.rn.relu.satfinite.scaled::n2::ue8m0.s2f6x2.f32 %0, %1, %2, %3;"
                 : "=h"(out) : "f"(a), "f"(b), "h"(scale));
    return out;
}

//         /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                                           /* 0x0000000402027981 */
//                                                                                                                    /* 0x002ea2000c1e9900 */
//         /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                                                     /* 0x0000000207047825 */
//                                                                                                                    /* 0x008fe200078e0204 */
//         /*00b0*/                   F2FP.SATFINITE.S2_6.BF16.UNPACK_B_MERGE_C R7, R2, 3.38953138925153547590e+38 ;  /* 0x7f7f0000ff07743e */
//                                                                                                                    /* 0x004fca0004a81202 */
//         /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;                                                /* 0x0000000704007986 */
//                                                                                                                    /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_sat_s2f6x2_bf16x2(uint32_t a) {
    uint16_t out;
    asm volatile("cvt.rn.satfinite.s2f6x2.bf16x2 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

//         /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                                                /* 0x0000000402027981 */
//                                                                                                                        /* 0x002ea2000c1e9900 */
//         /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                                                          /* 0x0000000207047825 */
//                                                                                                                        /* 0x008fe200078e0204 */
//         /*00b0*/                   F2FP.SATFINITE.RELU.S2_6.BF16.UNPACK_B_MERGE_C R7, R2, 3.38953138925153547590e+38 ;  /* 0x7f7f0000ff07743e */
//                                                                                                                        /* 0x004fca0004a81a02 */
//         /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;                                                     /* 0x0000000704007986 */
//                                                                                                                        /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_relu_sat_s2f6x2_bf16x2(uint32_t a) {
    uint16_t out;
    asm volatile("cvt.rn.relu.satfinite.s2f6x2.bf16x2 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

//         /*00a0*/                   LDG.E.U16.CONSTANT R4, desc[UR4][R4.64] ;                /* 0x0000000404047981 */
//                                                                                             /* 0x002ea2000c1e9500 */
//         /*00b0*/                   IMAD.WIDE R2, R9, 0x4, R2 ;                              /* 0x0000000409027825 */
//                                                                                             /* 0x008fcc00078e0202 */
//         /*00c0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                    /* 0x0000000402027981 */
//                                                                                             /* 0x000ee2000c1e9900 */
//         /*00d0*/                   IMAD.WIDE R6, R9, 0x2, R6 ;                              /* 0x0000000209067825 */
//                                                                                             /* 0x001fe200078e0206 */
//         /*00e0*/                   SHF.L.U32 R11, R4, 0x10, RZ ;                            /* 0x00000010040b7819 */
//                                                                                             /* 0x004fc800000006ff */
//         /*00f0*/                   F2FP.SATFINITE.S2_6.BF16.UNPACK_B_MERGE_C R9, R2, R11 ;  /* 0x00000002ff09723e */
//                                                                                             /* 0x008fca0004a8120b */
//         /*0100*/                   STG.E.U16 desc[UR4][R6.64], R9 ;                         /* 0x0000000906007986 */
//                                                                                             /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_sat_scaled_s2f6x2_bf16x2(uint32_t a, uint16_t scale) {
    uint16_t out;
    asm volatile("cvt.rn.satfinite.scaled::n2::ue8m0.s2f6x2.bf16x2 %0, %1, %2;"
                 : "=h"(out) : "r"(a), "h"(scale));
    return out;
}

//         /*00a0*/                   LDG.E.U16.CONSTANT R4, desc[UR4][R4.64] ;                /* 0x0000000404047981 */
//                                                                                             /* 0x002ea2000c1e9500 */
//         /*00b0*/                   IMAD.WIDE R2, R9, 0x4, R2 ;                              /* 0x0000000409027825 */
//                                                                                             /* 0x008fcc00078e0202 */
//         /*00c0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                    /* 0x0000000402027981 */
//                                                                                             /* 0x000ee2000c1e9900 */
//         /*00d0*/                   IMAD.WIDE R6, R9, 0x2, R6 ;                              /* 0x0000000209067825 */
//                                                                                             /* 0x001fe200078e0206 */
//         /*00e0*/                   SHF.L.U32 R11, R4, 0x10, RZ ;                            /* 0x00000010040b7819 */
//                                                                                             /* 0x004fc800000006ff */
//         /*00f0*/                   F2FP.SATFINITE.RELU.S2_6.BF16.UNPACK_B_MERGE_C R9, R2, R11 ;  /* 0x00000002ff09723e */
//                                                                                             /* 0x008fca0004a81a0b */
//         /*0100*/                   STG.E.U16 desc[UR4][R6.64], R9 ;                         /* 0x0000000906007986 */
//                                                                                             /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_relu_sat_scaled_s2f6x2_bf16x2(uint32_t a, uint16_t scale) {
    uint16_t out;
    asm volatile("cvt.rn.relu.satfinite.scaled::n2::ue8m0.s2f6x2.bf16x2 %0, %1, %2;"
                 : "=h"(out) : "r"(a), "h"(scale));
    return out;
}

//         /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;                       /* 0x0000000402027981 */
//                                                                                                    /* 0x002ea2000c1e9500 */
//         /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                                     /* 0x0000000407047825 */
//                                                                                                    /* 0x008fe200078e0204 */
//         /*00b0*/                   F2FP.BF16.S2_6.UNPACK_B R7, R2, 4.5736980577097704378e-41.H0 ;  /* 0x00007f7f00077e3e */
//                                                                                                    /* 0x004fca0002081402 */
//         /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                                    /* 0x0000000704007986 */
//                                                                                                    /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_bf16x2_s2f6x2(uint16_t a) {
    uint32_t out;
    asm volatile("cvt.rn.bf16x2.s2f6x2 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

//         /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;                       /* 0x0000000402027981 */
//                                                                                                    /* 0x002ea2000c1e9500 */
//         /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                                     /* 0x0000000407047825 */
//                                                                                                    /* 0x008fe200078e0204 */
//         /*00b0*/                   F2FP.RELU.BF16.S2_6.UNPACK_B R7, R2, 4.5736980577097704378e-41.H0 ;  /* 0x00007f7f00077e3e */
//                                                                                                    /* 0x004fca0002081c02 */
//         /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                                    /* 0x0000000704007986 */
//                                                                                                    /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_relu_bf16x2_s2f6x2(uint16_t a) {
    uint32_t out;
    asm volatile("cvt.rn.relu.bf16x2.s2f6x2 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

//         /*00a0*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;     /* 0x0000000402027981 */
//                                                                                  /* 0x002ea2000c1e9500 */
//         /*00b0*/                   IMAD.WIDE R4, R9, 0x2, R4 ;                   /* 0x0000000209047825 */
//                                                                                  /* 0x008fcc00078e0204 */
//         /*00c0*/                   LDG.E.U16.CONSTANT R5, desc[UR4][R4.64] ;     /* 0x0000000404057981 */
//                                                                                  /* 0x000ea2000c1e9500 */
//         /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;                   /* 0x0000000409067825 */
//                                                                                  /* 0x001fe200078e0206 */
//         /*00e0*/                   F2FP.BF16.S2_6.UNPACK_B R9, R2, R5.H0 ;       /* 0x000000020009763e */
//                                                                                  /* 0x004fca0002081405 */
//         /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;                  /* 0x0000000906007986 */
//                                                                                  /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_scaled_bf16x2_s2f6x2(uint16_t a, uint16_t scale) {
    uint32_t out;
    asm volatile("cvt.rn.scaled::n2::ue8m0.bf16x2.s2f6x2 %0, %1, %2;" : "=r"(out) : "h"(a), "h"(scale));
    return out;
}

//         /*00a0*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;     /* 0x0000000402027981 */
//                                                                                  /* 0x002ea2000c1e9500 */
//         /*00b0*/                   IMAD.WIDE R4, R9, 0x2, R4 ;                   /* 0x0000000209047825 */
//                                                                                  /* 0x008fcc00078e0204 */
//         /*00c0*/                   LDG.E.U16.CONSTANT R5, desc[UR4][R4.64] ;     /* 0x0000000404057981 */
//                                                                                  /* 0x000ea2000c1e9500 */
//         /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;                   /* 0x0000000409067825 */
//                                                                                  /* 0x001fe200078e0206 */
//         /*00e0*/                   F2FP.RELU.BF16.S2_6.UNPACK_B R9, R2, R5.H0 ;  /* 0x000000020009763e */
//                                                                                  /* 0x004fca0002081c05 */
//         /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;                  /* 0x0000000906007986 */
//                                                                                  /* 0x000fe2000c101904 */
//         /*00a0*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;     /* 0x0000000402027981 */
//                                                                                  /* 0x002ea2000c1e9500 */
//         /*00b0*/                   IMAD.WIDE R4, R9, 0x2, R4 ;                   /* 0x0000000209047825 */
//                                                                                  /* 0x008fcc00078e0204 */
//         /*00c0*/                   LDG.E.U16.CONSTANT R5, desc[UR4][R4.64] ;     /* 0x0000000404057981 */
//                                                                                  /* 0x000ea2000c1e9500 */
//         /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;                   /* 0x0000000409067825 */
//                                                                                  /* 0x001fe200078e0206 */
//         /*00e0*/                   F2FP.RELU.BF16.S2_6.UNPACK_B R9, R2, R5.H0 ;  /* 0x000000020009763e */
//                                                                                  /* 0x004fca0002081c05 */
//         /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;                  /* 0x0000000906007986 */
//                                                                                  /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_relu_scaled_bf16x2_s2f6x2(uint16_t a, uint16_t scale) {
    uint32_t out;
    asm volatile("cvt.rn.relu.scaled::n2::ue8m0.bf16x2.s2f6x2 %0, %1, %2;"
                 : "=r"(out) : "h"(a), "h"(scale));
    return out;
}

//         /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;                                 /* 0x0000000402027981 */
//                                                                                                             /* 0x002ea2000c1e9500 */
//         /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                                               /* 0x0000000407047825 */
//                                                                                                             /* 0x008fe200078e0204 */
//         /*00b0*/                   F2FP.SATFINITE.BF16.S2_6.UNPACK_B R7, R2, 4.5736980577097704378e-41.H0 ;  /* 0x00007f7f00077e3e */
//                                                                                                             /* 0x004fca0002083402 */
//         /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                                              /* 0x0000000704007986 */
//                                                                                                             /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_sat_bf16x2_s2f6x2(uint16_t a) {
    uint32_t out;
    asm volatile("cvt.rn.satfinite.bf16x2.s2f6x2 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

//         /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;                                 /* 0x0000000402027981 */
//                                                                                                             /* 0x002ea2000c1e9500 */
//         /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                                               /* 0x0000000407047825 */
//                                                                                                             /* 0x008fe200078e0204 */
//         /*00b0*/                   F2FP.SATFINITE.RELU.BF16.S2_6.UNPACK_B R7, R2, 4.5736980577097704378e-41.H0 ;  /* 0x00007f7f00077e3e */
//                                                                                                             /* 0x004fca0002083c02 */
//         /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                                              /* 0x0000000704007986 */
//                                                                                                             /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_relu_sat_bf16x2_s2f6x2(uint16_t a) {
    uint32_t out;
    asm volatile("cvt.rn.relu.satfinite.bf16x2.s2f6x2 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

//         /*00a0*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;     /* 0x0000000402027981 */
//                                                                                  /* 0x002ea2000c1e9500 */
//         /*00b0*/                   IMAD.WIDE R4, R9, 0x2, R4 ;                   /* 0x0000000209047825 */
//                                                                                  /* 0x008fcc00078e0204 */
//         /*00c0*/                   LDG.E.U16.CONSTANT R5, desc[UR4][R4.64] ;     /* 0x0000000404057981 */
//                                                                                  /* 0x000ea2000c1e9500 */
//         /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;                   /* 0x0000000409067825 */
//                                                                                  /* 0x001fe200078e0206 */
//         /*00e0*/                   F2FP.SATFINITE.BF16.S2_6.UNPACK_B R9, R2, R5.H0 ;  /* 0x000000020009763e */
//                                                                                  /* 0x004fca0002083405 */
//         /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;                  /* 0x0000000906007986 */
//                                                                                  /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_sat_scaled_bf16x2_s2f6x2(uint16_t a, uint16_t scale) {
    uint32_t out;
    asm volatile("cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2 %0, %1, %2;" : "=r"(out) : "h"(a), "h"(scale));
    return out;
}

//         /*00a0*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;     /* 0x0000000402027981 */
//                                                                                  /* 0x002ea2000c1e9500 */
//         /*00b0*/                   IMAD.WIDE R4, R9, 0x2, R4 ;                   /* 0x0000000209047825 */
//                                                                                  /* 0x008fcc00078e0204 */
//         /*00c0*/                   LDG.E.U16.CONSTANT R5, desc[UR4][R4.64] ;     /* 0x0000000404057981 */
//                                                                                  /* 0x000ea2000c1e9500 */
//         /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;                   /* 0x0000000409067825 */
//                                                                                  /* 0x001fe200078e0206 */
//         /*00e0*/                   F2FP.SATFINITE.RELU.BF16.S2_6.UNPACK_B R9, R2, R5.H0 ;  /* 0x000000020009763e */
//                                                                                  /* 0x004fca0002083c05 */
//         /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;                  /* 0x0000000906007986 */
//                                                                                  /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_relu_sat_scaled_bf16x2_s2f6x2(uint16_t a, uint16_t scale) {
    uint32_t out;
    asm volatile("cvt.rn.relu.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2 %0, %1, %2;" : "=r"(out) : "h"(a), "h"(scale));
    return out;
}

extern "C" __global__ void cvt_s2f6x2_scaled_kernel(
    const float* __restrict__ in_a,
    const float* __restrict__ in_b,
    const uint32_t* __restrict__ in_bf16x2,
    const uint16_t* __restrict__ in_s2f6x2,
    const uint16_t* __restrict__ in_scale,
    uint16_t* __restrict__ out_s2f6x2_f32,
    uint16_t* __restrict__ out_s2f6x2_f32_relu,
    uint16_t* __restrict__ out_s2f6x2_f32_scaled,
    uint16_t* __restrict__ out_s2f6x2_f32_relu_scaled,
    uint16_t* __restrict__ out_s2f6x2_bf16x2,
    uint16_t* __restrict__ out_s2f6x2_bf16x2_relu,
    uint16_t* __restrict__ out_s2f6x2_bf16x2_scaled,
    uint16_t* __restrict__ out_s2f6x2_bf16x2_relu_scaled,
    uint32_t* __restrict__ out_bf16x2_s2f6x2,
    uint32_t* __restrict__ out_bf16x2_s2f6x2_relu,
    uint32_t* __restrict__ out_bf16x2_s2f6x2_scaled,
    uint32_t* __restrict__ out_bf16x2_s2f6x2_relu_scaled,
    uint32_t* __restrict__ out_bf16x2_s2f6x2_sat,
    uint32_t* __restrict__ out_bf16x2_s2f6x2_relu_sat,
    uint32_t* __restrict__ out_bf16x2_s2f6x2_sat_scaled,
    uint32_t* __restrict__ out_bf16x2_s2f6x2_relu_sat_scaled
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    float a = in_a[tid];
    float b = in_b[tid];
    uint32_t bf16x2 = in_bf16x2[tid];
    uint16_t s2f6x2 = in_s2f6x2[tid];
    uint16_t scale = in_scale[tid];

    out_s2f6x2_f32[tid] = cvt_rn_sat_s2f6x2_f32(a, b);
    out_s2f6x2_f32_relu[tid] = cvt_rn_relu_sat_s2f6x2_f32(a, b);
    out_s2f6x2_f32_scaled[tid] = cvt_rn_sat_scaled_s2f6x2_f32(a, b, scale);
    out_s2f6x2_f32_relu_scaled[tid] = cvt_rn_relu_sat_scaled_s2f6x2_f32(a, b, scale);

    out_s2f6x2_bf16x2[tid] = cvt_rn_sat_s2f6x2_bf16x2(bf16x2);
    out_s2f6x2_bf16x2_relu[tid] = cvt_rn_relu_sat_s2f6x2_bf16x2(bf16x2);
    out_s2f6x2_bf16x2_scaled[tid] = cvt_rn_sat_scaled_s2f6x2_bf16x2(bf16x2, scale);
    out_s2f6x2_bf16x2_relu_scaled[tid] = cvt_rn_relu_sat_scaled_s2f6x2_bf16x2(bf16x2, scale);

    out_bf16x2_s2f6x2[tid] = cvt_rn_bf16x2_s2f6x2(s2f6x2);
    out_bf16x2_s2f6x2_relu[tid] = cvt_rn_relu_bf16x2_s2f6x2(s2f6x2);
    out_bf16x2_s2f6x2_scaled[tid] = cvt_rn_scaled_bf16x2_s2f6x2(s2f6x2, scale);
    out_bf16x2_s2f6x2_relu_scaled[tid] = cvt_rn_relu_scaled_bf16x2_s2f6x2(s2f6x2, scale);
    out_bf16x2_s2f6x2_sat[tid] = cvt_rn_sat_bf16x2_s2f6x2(s2f6x2);
    out_bf16x2_s2f6x2_relu_sat[tid] = cvt_rn_relu_sat_bf16x2_s2f6x2(s2f6x2);
    out_bf16x2_s2f6x2_sat_scaled[tid] = cvt_rn_sat_scaled_bf16x2_s2f6x2(s2f6x2, scale);
    out_bf16x2_s2f6x2_relu_sat_scaled[tid] = cvt_rn_relu_sat_scaled_bf16x2_s2f6x2(s2f6x2, scale);
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
    uint16_t* in_s2f6x2;
    uint16_t* in_scale;

    uint16_t* out_s2f6x2_f32;
    uint16_t* out_s2f6x2_f32_relu;
    uint16_t* out_s2f6x2_f32_scaled;
    uint16_t* out_s2f6x2_f32_relu_scaled;
    uint16_t* out_s2f6x2_bf16x2;
    uint16_t* out_s2f6x2_bf16x2_relu;
    uint16_t* out_s2f6x2_bf16x2_scaled;
    uint16_t* out_s2f6x2_bf16x2_relu_scaled;
    uint32_t* out_bf16x2_s2f6x2;
    uint32_t* out_bf16x2_s2f6x2_relu;
    uint32_t* out_bf16x2_s2f6x2_scaled;
    uint32_t* out_bf16x2_s2f6x2_relu_scaled;
    uint32_t* out_bf16x2_s2f6x2_sat;
    uint32_t* out_bf16x2_s2f6x2_relu_sat;
    uint32_t* out_bf16x2_s2f6x2_sat_scaled;
    uint32_t* out_bf16x2_s2f6x2_relu_sat_scaled;

    ck(cudaMallocManaged(&in_a, N * sizeof(float)), "cudaMallocManaged in_a");
    ck(cudaMallocManaged(&in_b, N * sizeof(float)), "cudaMallocManaged in_b");
    ck(cudaMallocManaged(&in_bf16x2, N * sizeof(uint32_t)), "cudaMallocManaged in_bf16x2");
    ck(cudaMallocManaged(&in_s2f6x2, N * sizeof(uint16_t)), "cudaMallocManaged in_s2f6x2");
    ck(cudaMallocManaged(&in_scale, N * sizeof(uint16_t)), "cudaMallocManaged in_scale");

    ck(cudaMallocManaged(&out_s2f6x2_f32, N * sizeof(uint16_t)), "cudaMallocManaged out_s2f6x2_f32");
    ck(cudaMallocManaged(&out_s2f6x2_f32_relu, N * sizeof(uint16_t)),
        "cudaMallocManaged out_s2f6x2_f32_relu");
    ck(cudaMallocManaged(&out_s2f6x2_f32_scaled, N * sizeof(uint16_t)),
        "cudaMallocManaged out_s2f6x2_f32_scaled");
    ck(cudaMallocManaged(&out_s2f6x2_f32_relu_scaled, N * sizeof(uint16_t)),
        "cudaMallocManaged out_s2f6x2_f32_relu_scaled");
    ck(cudaMallocManaged(&out_s2f6x2_bf16x2, N * sizeof(uint16_t)),
        "cudaMallocManaged out_s2f6x2_bf16x2");
    ck(cudaMallocManaged(&out_s2f6x2_bf16x2_relu, N * sizeof(uint16_t)),
        "cudaMallocManaged out_s2f6x2_bf16x2_relu");
    ck(cudaMallocManaged(&out_s2f6x2_bf16x2_scaled, N * sizeof(uint16_t)),
        "cudaMallocManaged out_s2f6x2_bf16x2_scaled");
    ck(cudaMallocManaged(&out_s2f6x2_bf16x2_relu_scaled, N * sizeof(uint16_t)),
        "cudaMallocManaged out_s2f6x2_bf16x2_relu_scaled");

    ck(cudaMallocManaged(&out_bf16x2_s2f6x2, N * sizeof(uint32_t)),
        "cudaMallocManaged out_bf16x2_s2f6x2");
    ck(cudaMallocManaged(&out_bf16x2_s2f6x2_relu, N * sizeof(uint32_t)),
        "cudaMallocManaged out_bf16x2_s2f6x2_relu");
    ck(cudaMallocManaged(&out_bf16x2_s2f6x2_scaled, N * sizeof(uint32_t)),
        "cudaMallocManaged out_bf16x2_s2f6x2_scaled");
    ck(cudaMallocManaged(&out_bf16x2_s2f6x2_relu_scaled, N * sizeof(uint32_t)),
        "cudaMallocManaged out_bf16x2_s2f6x2_relu_scaled");
    ck(cudaMallocManaged(&out_bf16x2_s2f6x2_sat, N * sizeof(uint32_t)),
        "cudaMallocManaged out_bf16x2_s2f6x2_sat");
    ck(cudaMallocManaged(&out_bf16x2_s2f6x2_relu_sat, N * sizeof(uint32_t)),
        "cudaMallocManaged out_bf16x2_s2f6x2_relu_sat");
    ck(cudaMallocManaged(&out_bf16x2_s2f6x2_sat_scaled, N * sizeof(uint32_t)),
        "cudaMallocManaged out_bf16x2_s2f6x2_sat_scaled");
    ck(cudaMallocManaged(&out_bf16x2_s2f6x2_relu_sat_scaled, N * sizeof(uint32_t)),
        "cudaMallocManaged out_bf16x2_s2f6x2_relu_sat_scaled");

    for (int i = 0; i < N; ++i) {
        float base = (float)(i * 0.5f + 0.25f);
        in_a[i] = base;
        in_b[i] = -base * 0.75f;
        in_bf16x2[i] = 0x3f803f80u + (uint32_t)(i & 0xffu);
        in_s2f6x2[i] = (uint16_t)(0x7f00u + (uint16_t)(i & 0xffu));
        in_scale[i] = (uint16_t)(0x7f7fu ^ (uint16_t)i);
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    cvt_s2f6x2_scaled_kernel<<<grid, block>>>(
        in_a,
        in_b,
        in_bf16x2,
        in_s2f6x2,
        in_scale,
        out_s2f6x2_f32,
        out_s2f6x2_f32_relu,
        out_s2f6x2_f32_scaled,
        out_s2f6x2_f32_relu_scaled,
        out_s2f6x2_bf16x2,
        out_s2f6x2_bf16x2_relu,
        out_s2f6x2_bf16x2_scaled,
        out_s2f6x2_bf16x2_relu_scaled,
        out_bf16x2_s2f6x2,
        out_bf16x2_s2f6x2_relu,
        out_bf16x2_s2f6x2_scaled,
        out_bf16x2_s2f6x2_relu_scaled,
        out_bf16x2_s2f6x2_sat,
        out_bf16x2_s2f6x2_relu_sat,
        out_bf16x2_s2f6x2_sat_scaled,
        out_bf16x2_s2f6x2_relu_sat_scaled
    );

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("s2f6x2=0x%04x bf16x2=0x%08x\n",
        (unsigned int)out_s2f6x2_f32[0],
        (unsigned int)out_bf16x2_s2f6x2[0]);

    cudaFree(in_a);
    cudaFree(in_b);
    cudaFree(in_bf16x2);
    cudaFree(in_s2f6x2);
    cudaFree(in_scale);
    cudaFree(out_s2f6x2_f32);
    cudaFree(out_s2f6x2_f32_relu);
    cudaFree(out_s2f6x2_f32_scaled);
    cudaFree(out_s2f6x2_f32_relu_scaled);
    cudaFree(out_s2f6x2_bf16x2);
    cudaFree(out_s2f6x2_bf16x2_relu);
    cudaFree(out_s2f6x2_bf16x2_scaled);
    cudaFree(out_s2f6x2_bf16x2_relu_scaled);
    cudaFree(out_bf16x2_s2f6x2);
    cudaFree(out_bf16x2_s2f6x2_relu);
    cudaFree(out_bf16x2_s2f6x2_scaled);
    cudaFree(out_bf16x2_s2f6x2_relu_scaled);
    cudaFree(out_bf16x2_s2f6x2_sat);
    cudaFree(out_bf16x2_s2f6x2_relu_sat);
    cudaFree(out_bf16x2_s2f6x2_sat_scaled);
    cudaFree(out_bf16x2_s2f6x2_relu_sat_scaled);
    return 0;
}
