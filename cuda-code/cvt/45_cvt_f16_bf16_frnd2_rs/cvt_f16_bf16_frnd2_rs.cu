// cvt_f16_bf16_frnd2_rs.cu

// cvt.frnd2{.relu}{.satfinite}.f16.f32       d, a;
// cvt.frnd2{.relu}{.satfinite}.f16x2.f32     d, a, b;
// cvt.rs{.relu}{.satfinite}.f16x2.f32        d, a, b, rbits;

// cvt.frnd2{.relu}{.satfinite}.bf16.f32      d, a;
// cvt.frnd2{.relu}{.satfinite}.bf16x2.f32    d, a, b;
// cvt.rs{.relu}{.satfinite}.bf16x2.f32       d, a, b, rbits;

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;            /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.F16.F32.PACK_AB R0, RZ, R2 ;      /* 0x00000002ff00723e */
        // /* 0x004fca00000000ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;       /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_f16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;            /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.F16.F32.PACK_AB.RZ R0, RZ, R2 ;   /* 0x00000002ff00723e */
        // /* 0x004fca00000180ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;       /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_f16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rz.f16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;   /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;             /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.RELU.F16.F32.MERGE_C R0, R2, RZ ;  /* 0x00000002ff00723e */
        // /* 0x004fca00000048ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;        /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_relu_f16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rn.relu.f16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;      /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.RELU.F16.F32.MERGE_C.RZ R0, R2, RZ ;  /* 0x00000002ff00723e */
        // /* 0x004fca000001c8ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;           /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_relu_f16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rz.relu.f16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;        /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.SATFINITE.F16.F32.MERGE_C R0, R2, RZ ;  /* 0x00000002ff00723e */
        // /* 0x004fca00000060ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;             /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_satfinite_f16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rn.satfinite.f16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;           /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                     /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.SATFINITE.F16.F32.MERGE_C.RZ R0, R2, RZ ;  /* 0x00000002ff00723e */
        // /* 0x004fca000001e0ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;                /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_satfinite_f16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rz.satfinite.f16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;             /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                       /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.SATFINITE.RELU.F16.F32.MERGE_C R0, R2, RZ ;  /* 0x00000002ff00723e */
        // /* 0x004fca00000068ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;                  /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_relu_satfinite_f16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rn.relu.satfinite.f16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                          /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.SATFINITE.RELU.F16.F32.MERGE_C.RZ R0, R2, RZ ;  /* 0x00000002ff00723e */
        // /* 0x004fca000001e8ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;                     /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_relu_satfinite_f16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rz.relu.satfinite.f16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;            /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;  /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;            /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   F2FP.F16.F32.PACK_AB R9, R3, R4 ;      /* 0x000000040309723e */
        // /* 0x004fca00000000ff */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;           /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_f16x2_f32(float a, float b) {
    uint32_t out;
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(a), "f"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;            /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;  /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;            /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   F2FP.F16.F32.PACK_AB.RZ R9, R3, R4 ;   /* 0x000000040309723e */
        // /* 0x004fca00000180ff */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;           /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rz_f16x2_f32(float a, float b) {
    uint32_t out;
    asm volatile("cvt.rz.f16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(a), "f"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;   /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;             /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;   /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;             /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   F2FP.RELU.F16.F32.PACK_AB R9, R3, R4 ;  /* 0x000000040309723e */
        // /* 0x004fca00000008ff */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;            /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_relu_f16x2_f32(float a, float b) {
    uint32_t out;
    asm volatile("cvt.rn.relu.f16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(a), "f"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;      /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;                /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;      /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;                /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   F2FP.RELU.F16.F32.PACK_AB.RZ R9, R3, R4 ;  /* 0x000000040309723e */
        // /* 0x004fca00000188ff */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;               /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rz_relu_f16x2_f32(float a, float b) {
    uint32_t out;
    asm volatile("cvt.rz.relu.f16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(a), "f"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;        /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;                  /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;        /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;                  /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   F2FP.SATFINITE.F16.F32.PACK_AB R9, R3, R4 ;  /* 0x000000040309723e */
        // /* 0x004fca00000020ff */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;                 /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_satfinite_f16x2_f32(float a, float b) {
    uint32_t out;
    asm volatile("cvt.rn.satfinite.f16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(a), "f"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;           /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;                     /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;           /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;                     /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   F2FP.SATFINITE.F16.F32.PACK_AB.RZ R9, R3, R4 ;  /* 0x000000040309723e */
        // /* 0x004fca000001a0ff */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;                    /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rz_satfinite_f16x2_f32(float a, float b) {
    uint32_t out;
    asm volatile("cvt.rz.satfinite.f16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(a), "f"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;             /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;                       /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;             /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;                       /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   F2FP.SATFINITE.RELU.F16.F32.PACK_AB R9, R3, R4 ;  /* 0x000000040309723e */
        // /* 0x004fca00000028ff */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;                      /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_relu_satfinite_f16x2_f32(float a, float b) {
    uint32_t out;
    asm volatile("cvt.rn.relu.satfinite.f16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(a), "f"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;                /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;                          /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;                /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;                          /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   F2FP.SATFINITE.RELU.F16.F32.PACK_AB.RZ R9, R3, R4 ;  /* 0x000000040309723e */
        // /* 0x004fca000001a8ff */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;                         /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rz_relu_satfinite_f16x2_f32(float a, float b) {
    uint32_t out;
    asm volatile("cvt.rz.relu.satfinite.f16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(a), "f"(b));
    return out;
}

        // /*00b0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;      /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   IMAD.WIDE R4, R11, 0x4, R4 ;               /* 0x000000040b047825 */
        // /* 0x008fcc00078e0204 */
        // /*00d0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;      /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00e0*/                   IMAD.WIDE R6, R11, 0x4, R6 ;               /* 0x000000040b067825 */
        // /* 0x010fcc00078e0206 */
        // /*00f0*/                   LDG.E.CONSTANT R6, desc[UR4][R6.64] ;      /* 0x0000000406067981 */
        // /* 0x000ea2000c1e9900 */
        // /*0100*/                   IMAD.WIDE R8, R11, 0x4, R8 ;               /* 0x000000040b087825 */
        // /* 0x001fe200078e0208 */
        // /*0110*/                   F2FP.F16.F32.PACK_AB.RS R10, R2, R5, R6 ;  /* 0x00000005020a723e */
        // /* 0x004fca0000020006 */
        // /*0120*/                   STG.E desc[UR4][R8.64], R10 ;              /* 0x0000000a08007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rs_f16x2_f32(float a, float b, uint32_t rbits) {
    uint32_t out;
    asm volatile("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(out) : "f"(a), "f"(b), "r"(rbits));
    return out;
}

        // /*00b0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;           /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   IMAD.WIDE R4, R11, 0x4, R4 ;                    /* 0x000000040b047825 */
        // /* 0x008fcc00078e0204 */
        // /*00d0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;           /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00e0*/                   IMAD.WIDE R6, R11, 0x4, R6 ;                    /* 0x000000040b067825 */
        // /* 0x010fcc00078e0206 */
        // /*00f0*/                   LDG.E.CONSTANT R6, desc[UR4][R6.64] ;           /* 0x0000000406067981 */
        // /* 0x000ea2000c1e9900 */
        // /*0100*/                   IMAD.WIDE R8, R11, 0x4, R8 ;                    /* 0x000000040b087825 */
        // /* 0x001fe200078e0208 */
        // /*0110*/                   F2FP.RELU.F16.F32.PACK_AB.RS R10, R2, R5, R6 ;  /* 0x00000005020a723e */
        // /* 0x004fca0000020806 */
        // /*0120*/                   STG.E desc[UR4][R8.64], R10 ;                   /* 0x0000000a08007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rs_relu_f16x2_f32(float a, float b, uint32_t rbits) {
    uint32_t out;
    asm volatile("cvt.rs.relu.f16x2.f32 %0, %1, %2, %3;" : "=r"(out) : "f"(a), "f"(b), "r"(rbits));
    return out;
}

        // /*00b0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   IMAD.WIDE R4, R11, 0x4, R4 ;                         /* 0x000000040b047825 */
        // /* 0x008fcc00078e0204 */
        // /*00d0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;                /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00e0*/                   IMAD.WIDE R6, R11, 0x4, R6 ;                         /* 0x000000040b067825 */
        // /* 0x010fcc00078e0206 */
        // /*00f0*/                   LDG.E.CONSTANT R6, desc[UR4][R6.64] ;                /* 0x0000000406067981 */
        // /* 0x000ea2000c1e9900 */
        // /*0100*/                   IMAD.WIDE R8, R11, 0x4, R8 ;                         /* 0x000000040b087825 */
        // /* 0x001fe200078e0208 */
        // /*0110*/                   F2FP.SATFINITE.F16.F32.PACK_AB.RS R10, R2, R5, R6 ;  /* 0x00000005020a723e */
        // /* 0x004fca0000022006 */
        // /*0120*/                   STG.E desc[UR4][R8.64], R10 ;                        /* 0x0000000a08007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rs_satfinite_f16x2_f32(float a, float b, uint32_t rbits) {
    uint32_t out;
    asm volatile("cvt.rs.satfinite.f16x2.f32 %0, %1, %2, %3;" : "=r"(out) : "f"(a), "f"(b), "r"(rbits));
    return out;
}

        // /*00b0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                     /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   IMAD.WIDE R4, R11, 0x4, R4 ;                              /* 0x000000040b047825 */
        // /* 0x008fcc00078e0204 */
        // /*00d0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;                     /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00e0*/                   IMAD.WIDE R6, R11, 0x4, R6 ;                              /* 0x000000040b067825 */
        // /* 0x010fcc00078e0206 */
        // /*00f0*/                   LDG.E.CONSTANT R6, desc[UR4][R6.64] ;                     /* 0x0000000406067981 */
        // /* 0x000ea2000c1e9900 */
        // /*0100*/                   IMAD.WIDE R8, R11, 0x4, R8 ;                              /* 0x000000040b087825 */
        // /* 0x001fe200078e0208 */
        // /*0110*/                   F2FP.SATFINITE.RELU.F16.F32.PACK_AB.RS R10, R2, R5, R6 ;  /* 0x00000005020a723e */
        // /* 0x004fca0000022806 */
        // /*0120*/                   STG.E desc[UR4][R8.64], R10 ;                             /* 0x0000000a08007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rs_relu_satfinite_f16x2_f32(float a, float b, uint32_t rbits) {
    uint32_t out;
    asm volatile("cvt.rs.relu.satfinite.f16x2.f32 %0, %1, %2, %3;" : "=r"(out) : "f"(a), "f"(b), "r"(rbits));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;            /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.BF16.F32.PACK_AB R0, RZ, R2 ;     /* 0x00000002ff00723e */
        // /* 0x004fca00000010ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;       /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_bf16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;            /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.BF16.F32.PACK_AB.RZ R0, RZ, R2 ;  /* 0x00000002ff00723e */
        // /* 0x004fca00000190ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;       /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_bf16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rz.bf16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;    /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.RELU.BF16.F32.MERGE_C R0, R2, RZ ;  /* 0x00000002ff00723e */
        // /* 0x004fca00000058ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;         /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_relu_bf16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rn.relu.bf16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;       /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                 /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.RELU.BF16.F32.MERGE_C.RZ R0, R2, RZ ;  /* 0x00000002ff00723e */
        // /* 0x004fca000001d8ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;            /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_relu_bf16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rz.relu.bf16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;         /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                   /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.SATFINITE.BF16.F32.MERGE_C R0, R2, RZ ;  /* 0x00000002ff00723e */
        // /* 0x004fca00000070ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;              /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_satfinite_bf16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rn.satfinite.bf16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;            /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                      /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.SATFINITE.BF16.F32.MERGE_C.RZ R0, R2, RZ ;  /* 0x00000002ff00723e */
        // /* 0x004fca000001f0ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;                 /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_satfinite_bf16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rz.satfinite.bf16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;              /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                        /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.SATFINITE.RELU.BF16.F32.MERGE_C R0, R2, RZ ;  /* 0x00000002ff00723e */
        // /* 0x004fca00000078ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;                   /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_relu_satfinite_bf16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rn.relu.satfinite.bf16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                 /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                           /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.SATFINITE.RELU.BF16.F32.MERGE_C.RZ R0, R2, RZ ;  /* 0x00000002ff00723e */
        // /* 0x004fca000001f8ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;                      /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_relu_satfinite_bf16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rz.relu.satfinite.bf16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;            /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;  /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;            /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   F2FP.BF16.F32.PACK_AB R9, R3, R4 ;     /* 0x000000040309723e */
        // /* 0x004fca00000010ff */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;           /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_bf16x2_f32(float a, float b) {
    uint32_t out;
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(a), "f"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;            /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;  /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;            /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   F2FP.BF16.F32.PACK_AB.RZ R9, R3, R4 ;  /* 0x000000040309723e */
        // /* 0x004fca00000190ff */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;           /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rz_bf16x2_f32(float a, float b) {
    uint32_t out;
    asm volatile("cvt.rz.bf16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(a), "f"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;    /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;              /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;    /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;              /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   F2FP.RELU.BF16.F32.PACK_AB R9, R3, R4 ;  /* 0x000000040309723e */
        // /* 0x004fca00000018ff */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;             /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_relu_bf16x2_f32(float a, float b) {
    uint32_t out;
    asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(a), "f"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;       /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;                 /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;       /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;                 /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   F2FP.RELU.BF16.F32.PACK_AB.RZ R9, R3, R4 ;  /* 0x000000040309723e */
        // /* 0x004fca00000198ff */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;                /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rz_relu_bf16x2_f32(float a, float b) {
    uint32_t out;
    asm volatile("cvt.rz.relu.bf16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(a), "f"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;         /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;                   /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;         /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;                   /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   F2FP.SATFINITE.BF16.F32.PACK_AB R9, R3, R4 ;  /* 0x000000040309723e */
        // /* 0x004fca00000030ff */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;                  /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_satfinite_bf16x2_f32(float a, float b) {
    uint32_t out;
    asm volatile("cvt.rn.satfinite.bf16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(a), "f"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;            /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;                      /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;            /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;                      /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   F2FP.SATFINITE.BF16.F32.PACK_AB.RZ R9, R3, R4 ;  /* 0x000000040309723e */
        // /* 0x004fca000001b0ff */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;                     /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rz_satfinite_bf16x2_f32(float a, float b) {
    uint32_t out;
    asm volatile("cvt.rz.satfinite.bf16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(a), "f"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;              /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;                        /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;              /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;                        /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   F2FP.SATFINITE.RELU.BF16.F32.PACK_AB R9, R3, R4 ;  /* 0x000000040309723e */
        // /* 0x004fca00000038ff */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;                       /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rn_relu_satfinite_bf16x2_f32(float a, float b) {
    uint32_t out;
    asm volatile("cvt.rn.relu.satfinite.bf16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(a), "f"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;                 /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;                           /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R4, desc[UR4][R4.64] ;                 /* 0x0000000404047981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;                           /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   F2FP.SATFINITE.RELU.BF16.F32.PACK_AB.RZ R9, R3, R4 ;  /* 0x000000040309723e */
        // /* 0x004fca000001b8ff */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;                          /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rz_relu_satfinite_bf16x2_f32(float a, float b) {
    uint32_t out;
    asm volatile("cvt.rz.relu.satfinite.bf16x2.f32 %0, %1, %2;" : "=r"(out) : "f"(a), "f"(b));
    return out;
}

        // /*00b0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;       /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   IMAD.WIDE R4, R11, 0x4, R4 ;                /* 0x000000040b047825 */
        // /* 0x008fcc00078e0204 */
        // /*00d0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;       /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00e0*/                   IMAD.WIDE R6, R11, 0x4, R6 ;                /* 0x000000040b067825 */
        // /* 0x010fcc00078e0206 */
        // /*00f0*/                   LDG.E.CONSTANT R6, desc[UR4][R6.64] ;       /* 0x0000000406067981 */
        // /* 0x000ea2000c1e9900 */
        // /*0100*/                   IMAD.WIDE R8, R11, 0x4, R8 ;                /* 0x000000040b087825 */
        // /* 0x001fe200078e0208 */
        // /*0110*/                   F2FP.BF16.F32.PACK_AB.RS R10, R2, R5, R6 ;  /* 0x00000005020a723e */
        // /* 0x004fca0000021006 */
        // /*0120*/                   STG.E desc[UR4][R8.64], R10 ;               /* 0x0000000a08007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rs_bf16x2_f32(float a, float b, uint32_t rbits) {
    uint32_t out;
    asm volatile("cvt.rs.bf16x2.f32 %0, %1, %2, %3;" : "=r"(out) : "f"(a), "f"(b), "r"(rbits));
    return out;
}

        // /*00b0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;            /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   IMAD.WIDE R4, R11, 0x4, R4 ;                     /* 0x000000040b047825 */
        // /* 0x008fcc00078e0204 */
        // /*00d0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;            /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00e0*/                   IMAD.WIDE R6, R11, 0x4, R6 ;                     /* 0x000000040b067825 */
        // /* 0x010fcc00078e0206 */
        // /*00f0*/                   LDG.E.CONSTANT R6, desc[UR4][R6.64] ;            /* 0x0000000406067981 */
        // /* 0x000ea2000c1e9900 */
        // /*0100*/                   IMAD.WIDE R8, R11, 0x4, R8 ;                     /* 0x000000040b087825 */
        // /* 0x001fe200078e0208 */
        // /*0110*/                   F2FP.RELU.BF16.F32.PACK_AB.RS R10, R2, R5, R6 ;  /* 0x00000005020a723e */
        // /* 0x004fca0000021806 */
        // /*0120*/                   STG.E desc[UR4][R8.64], R10 ;                    /* 0x0000000a08007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rs_relu_bf16x2_f32(float a, float b, uint32_t rbits) {
    uint32_t out;
    asm volatile("cvt.rs.relu.bf16x2.f32 %0, %1, %2, %3;" : "=r"(out) : "f"(a), "f"(b), "r"(rbits));
    return out;
}

        // /*00b0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                 /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   IMAD.WIDE R4, R11, 0x4, R4 ;                          /* 0x000000040b047825 */
        // /* 0x008fcc00078e0204 */
        // /*00d0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;                 /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00e0*/                   IMAD.WIDE R6, R11, 0x4, R6 ;                          /* 0x000000040b067825 */
        // /* 0x010fcc00078e0206 */
        // /*00f0*/                   LDG.E.CONSTANT R6, desc[UR4][R6.64] ;                 /* 0x0000000406067981 */
        // /* 0x000ea2000c1e9900 */
        // /*0100*/                   IMAD.WIDE R8, R11, 0x4, R8 ;                          /* 0x000000040b087825 */
        // /* 0x001fe200078e0208 */
        // /*0110*/                   F2FP.SATFINITE.BF16.F32.PACK_AB.RS R10, R2, R5, R6 ;  /* 0x00000005020a723e */
        // /* 0x004fca0000023006 */
        // /*0120*/                   STG.E desc[UR4][R8.64], R10 ;                         /* 0x0000000a08007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rs_satfinite_bf16x2_f32(float a, float b, uint32_t rbits) {
    uint32_t out;
    asm volatile("cvt.rs.satfinite.bf16x2.f32 %0, %1, %2, %3;" : "=r"(out) : "f"(a), "f"(b), "r"(rbits));
    return out;
}

        // /*00b0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;                      /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00c0*/                   IMAD.WIDE R4, R11, 0x4, R4 ;                               /* 0x000000040b047825 */
        // /* 0x008fcc00078e0204 */
        // /*00d0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;                      /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00e0*/                   IMAD.WIDE R6, R11, 0x4, R6 ;                               /* 0x000000040b067825 */
        // /* 0x010fcc00078e0206 */
        // /*00f0*/                   LDG.E.CONSTANT R6, desc[UR4][R6.64] ;                      /* 0x0000000406067981 */
        // /* 0x000ea2000c1e9900 */
        // /*0100*/                   IMAD.WIDE R8, R11, 0x4, R8 ;                               /* 0x000000040b087825 */
        // /* 0x001fe200078e0208 */
        // /*0110*/                   F2FP.SATFINITE.RELU.BF16.F32.PACK_AB.RS R10, R2, R5, R6 ;  /* 0x00000005020a723e */
        // /* 0x004fca0000023806 */
        // /*0120*/                   STG.E desc[UR4][R8.64], R10 ;                              /* 0x0000000a08007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t cvt_rs_relu_satfinite_bf16x2_f32(float a, float b, uint32_t rbits) {
    uint32_t out;
    asm volatile("cvt.rs.relu.satfinite.bf16x2.f32 %0, %1, %2, %3;" : "=r"(out) : "f"(a), "f"(b), "r"(rbits));
    return out;
}

extern "C" __global__ void cvt_f16_bf16_frnd2_rs_kernel(
    const float* __restrict__ in_a,
    const float* __restrict__ in_b,
    const uint32_t* __restrict__ in_rbits,
    uint16_t* __restrict__ out_f16_rn,
    uint16_t* __restrict__ out_f16_rz,
    uint16_t* __restrict__ out_f16_rn_relu,
    uint16_t* __restrict__ out_f16_rz_relu,
    uint16_t* __restrict__ out_f16_rn_satfinite,
    uint16_t* __restrict__ out_f16_rz_satfinite,
    uint16_t* __restrict__ out_f16_rn_relu_satfinite,
    uint16_t* __restrict__ out_f16_rz_relu_satfinite,
    uint32_t* __restrict__ out_f16x2_rn,
    uint32_t* __restrict__ out_f16x2_rz,
    uint32_t* __restrict__ out_f16x2_rn_relu,
    uint32_t* __restrict__ out_f16x2_rz_relu,
    uint32_t* __restrict__ out_f16x2_rn_satfinite,
    uint32_t* __restrict__ out_f16x2_rz_satfinite,
    uint32_t* __restrict__ out_f16x2_rn_relu_satfinite,
    uint32_t* __restrict__ out_f16x2_rz_relu_satfinite,
    uint32_t* __restrict__ out_f16x2_rs,
    uint32_t* __restrict__ out_f16x2_rs_relu,
    uint32_t* __restrict__ out_f16x2_rs_satfinite,
    uint32_t* __restrict__ out_f16x2_rs_relu_satfinite,
    uint16_t* __restrict__ out_bf16_rn,
    uint16_t* __restrict__ out_bf16_rz,
    uint16_t* __restrict__ out_bf16_rn_relu,
    uint16_t* __restrict__ out_bf16_rz_relu,
    uint16_t* __restrict__ out_bf16_rn_satfinite,
    uint16_t* __restrict__ out_bf16_rz_satfinite,
    uint16_t* __restrict__ out_bf16_rn_relu_satfinite,
    uint16_t* __restrict__ out_bf16_rz_relu_satfinite,
    uint32_t* __restrict__ out_bf16x2_rn,
    uint32_t* __restrict__ out_bf16x2_rz,
    uint32_t* __restrict__ out_bf16x2_rn_relu,
    uint32_t* __restrict__ out_bf16x2_rz_relu,
    uint32_t* __restrict__ out_bf16x2_rn_satfinite,
    uint32_t* __restrict__ out_bf16x2_rz_satfinite,
    uint32_t* __restrict__ out_bf16x2_rn_relu_satfinite,
    uint32_t* __restrict__ out_bf16x2_rz_relu_satfinite,
    uint32_t* __restrict__ out_bf16x2_rs,
    uint32_t* __restrict__ out_bf16x2_rs_relu,
    uint32_t* __restrict__ out_bf16x2_rs_satfinite,
    uint32_t* __restrict__ out_bf16x2_rs_relu_satfinite
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    float a = in_a[tid];
    float b = in_b[tid];
    uint32_t rbits = in_rbits[tid];

    out_f16_rn[tid] = cvt_rn_f16_f32(a);
    out_f16_rz[tid] = cvt_rz_f16_f32(a);
    out_f16_rn_relu[tid] = cvt_rn_relu_f16_f32(a);
    out_f16_rz_relu[tid] = cvt_rz_relu_f16_f32(a);
    out_f16_rn_satfinite[tid] = cvt_rn_satfinite_f16_f32(a);
    out_f16_rz_satfinite[tid] = cvt_rz_satfinite_f16_f32(a);
    out_f16_rn_relu_satfinite[tid] = cvt_rn_relu_satfinite_f16_f32(a);
    out_f16_rz_relu_satfinite[tid] = cvt_rz_relu_satfinite_f16_f32(a);

    out_f16x2_rn[tid] = cvt_rn_f16x2_f32(a, b);
    out_f16x2_rz[tid] = cvt_rz_f16x2_f32(a, b);
    out_f16x2_rn_relu[tid] = cvt_rn_relu_f16x2_f32(a, b);
    out_f16x2_rz_relu[tid] = cvt_rz_relu_f16x2_f32(a, b);
    out_f16x2_rn_satfinite[tid] = cvt_rn_satfinite_f16x2_f32(a, b);
    out_f16x2_rz_satfinite[tid] = cvt_rz_satfinite_f16x2_f32(a, b);
    out_f16x2_rn_relu_satfinite[tid] = cvt_rn_relu_satfinite_f16x2_f32(a, b);
    out_f16x2_rz_relu_satfinite[tid] = cvt_rz_relu_satfinite_f16x2_f32(a, b);
    out_f16x2_rs[tid] = cvt_rs_f16x2_f32(a, b, rbits);
    out_f16x2_rs_relu[tid] = cvt_rs_relu_f16x2_f32(a, b, rbits);
    out_f16x2_rs_satfinite[tid] = cvt_rs_satfinite_f16x2_f32(a, b, rbits);
    out_f16x2_rs_relu_satfinite[tid] = cvt_rs_relu_satfinite_f16x2_f32(a, b, rbits);

    out_bf16_rn[tid] = cvt_rn_bf16_f32(a);
    out_bf16_rz[tid] = cvt_rz_bf16_f32(a);
    out_bf16_rn_relu[tid] = cvt_rn_relu_bf16_f32(a);
    out_bf16_rz_relu[tid] = cvt_rz_relu_bf16_f32(a);
    out_bf16_rn_satfinite[tid] = cvt_rn_satfinite_bf16_f32(a);
    out_bf16_rz_satfinite[tid] = cvt_rz_satfinite_bf16_f32(a);
    out_bf16_rn_relu_satfinite[tid] = cvt_rn_relu_satfinite_bf16_f32(a);
    out_bf16_rz_relu_satfinite[tid] = cvt_rz_relu_satfinite_bf16_f32(a);

    out_bf16x2_rn[tid] = cvt_rn_bf16x2_f32(a, b);
    out_bf16x2_rz[tid] = cvt_rz_bf16x2_f32(a, b);
    out_bf16x2_rn_relu[tid] = cvt_rn_relu_bf16x2_f32(a, b);
    out_bf16x2_rz_relu[tid] = cvt_rz_relu_bf16x2_f32(a, b);
    out_bf16x2_rn_satfinite[tid] = cvt_rn_satfinite_bf16x2_f32(a, b);
    out_bf16x2_rz_satfinite[tid] = cvt_rz_satfinite_bf16x2_f32(a, b);
    out_bf16x2_rn_relu_satfinite[tid] = cvt_rn_relu_satfinite_bf16x2_f32(a, b);
    out_bf16x2_rz_relu_satfinite[tid] = cvt_rz_relu_satfinite_bf16x2_f32(a, b);
    out_bf16x2_rs[tid] = cvt_rs_bf16x2_f32(a, b, rbits);
    out_bf16x2_rs_relu[tid] = cvt_rs_relu_bf16x2_f32(a, b, rbits);
    out_bf16x2_rs_satfinite[tid] = cvt_rs_satfinite_bf16x2_f32(a, b, rbits);
    out_bf16x2_rs_relu_satfinite[tid] = cvt_rs_relu_satfinite_bf16x2_f32(a, b, rbits);
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
    uint32_t* in_rbits;

    uint16_t* out_f16_rn;
    uint16_t* out_f16_rz;
    uint16_t* out_f16_rn_relu;
    uint16_t* out_f16_rz_relu;
    uint16_t* out_f16_rn_satfinite;
    uint16_t* out_f16_rz_satfinite;
    uint16_t* out_f16_rn_relu_satfinite;
    uint16_t* out_f16_rz_relu_satfinite;
    uint32_t* out_f16x2_rn;
    uint32_t* out_f16x2_rz;
    uint32_t* out_f16x2_rn_relu;
    uint32_t* out_f16x2_rz_relu;
    uint32_t* out_f16x2_rn_satfinite;
    uint32_t* out_f16x2_rz_satfinite;
    uint32_t* out_f16x2_rn_relu_satfinite;
    uint32_t* out_f16x2_rz_relu_satfinite;
    uint32_t* out_f16x2_rs;
    uint32_t* out_f16x2_rs_relu;
    uint32_t* out_f16x2_rs_satfinite;
    uint32_t* out_f16x2_rs_relu_satfinite;

    uint16_t* out_bf16_rn;
    uint16_t* out_bf16_rz;
    uint16_t* out_bf16_rn_relu;
    uint16_t* out_bf16_rz_relu;
    uint16_t* out_bf16_rn_satfinite;
    uint16_t* out_bf16_rz_satfinite;
    uint16_t* out_bf16_rn_relu_satfinite;
    uint16_t* out_bf16_rz_relu_satfinite;
    uint32_t* out_bf16x2_rn;
    uint32_t* out_bf16x2_rz;
    uint32_t* out_bf16x2_rn_relu;
    uint32_t* out_bf16x2_rz_relu;
    uint32_t* out_bf16x2_rn_satfinite;
    uint32_t* out_bf16x2_rz_satfinite;
    uint32_t* out_bf16x2_rn_relu_satfinite;
    uint32_t* out_bf16x2_rz_relu_satfinite;
    uint32_t* out_bf16x2_rs;
    uint32_t* out_bf16x2_rs_relu;
    uint32_t* out_bf16x2_rs_satfinite;
    uint32_t* out_bf16x2_rs_relu_satfinite;

    ck(cudaMallocManaged(&in_a, N * sizeof(float)), "cudaMallocManaged in_a");
    ck(cudaMallocManaged(&in_b, N * sizeof(float)), "cudaMallocManaged in_b");
    ck(cudaMallocManaged(&in_rbits, N * sizeof(uint32_t)), "cudaMallocManaged in_rbits");

    ck(cudaMallocManaged(&out_f16_rn, N * sizeof(uint16_t)), "cudaMallocManaged out_f16_rn");
    ck(cudaMallocManaged(&out_f16_rz, N * sizeof(uint16_t)), "cudaMallocManaged out_f16_rz");
    ck(cudaMallocManaged(&out_f16_rn_relu, N * sizeof(uint16_t)), "cudaMallocManaged out_f16_rn_relu");
    ck(cudaMallocManaged(&out_f16_rz_relu, N * sizeof(uint16_t)), "cudaMallocManaged out_f16_rz_relu");
    ck(cudaMallocManaged(&out_f16_rn_satfinite, N * sizeof(uint16_t)), "cudaMallocManaged out_f16_rn_satfinite");
    ck(cudaMallocManaged(&out_f16_rz_satfinite, N * sizeof(uint16_t)), "cudaMallocManaged out_f16_rz_satfinite");
    ck(cudaMallocManaged(&out_f16_rn_relu_satfinite, N * sizeof(uint16_t)),
        "cudaMallocManaged out_f16_rn_relu_satfinite");
    ck(cudaMallocManaged(&out_f16_rz_relu_satfinite, N * sizeof(uint16_t)),
        "cudaMallocManaged out_f16_rz_relu_satfinite");
    ck(cudaMallocManaged(&out_f16x2_rn, N * sizeof(uint32_t)), "cudaMallocManaged out_f16x2_rn");
    ck(cudaMallocManaged(&out_f16x2_rz, N * sizeof(uint32_t)), "cudaMallocManaged out_f16x2_rz");
    ck(cudaMallocManaged(&out_f16x2_rn_relu, N * sizeof(uint32_t)), "cudaMallocManaged out_f16x2_rn_relu");
    ck(cudaMallocManaged(&out_f16x2_rz_relu, N * sizeof(uint32_t)), "cudaMallocManaged out_f16x2_rz_relu");
    ck(cudaMallocManaged(&out_f16x2_rn_satfinite, N * sizeof(uint32_t)),
        "cudaMallocManaged out_f16x2_rn_satfinite");
    ck(cudaMallocManaged(&out_f16x2_rz_satfinite, N * sizeof(uint32_t)),
        "cudaMallocManaged out_f16x2_rz_satfinite");
    ck(cudaMallocManaged(&out_f16x2_rn_relu_satfinite, N * sizeof(uint32_t)),
        "cudaMallocManaged out_f16x2_rn_relu_satfinite");
    ck(cudaMallocManaged(&out_f16x2_rz_relu_satfinite, N * sizeof(uint32_t)),
        "cudaMallocManaged out_f16x2_rz_relu_satfinite");
    ck(cudaMallocManaged(&out_f16x2_rs, N * sizeof(uint32_t)), "cudaMallocManaged out_f16x2_rs");
    ck(cudaMallocManaged(&out_f16x2_rs_relu, N * sizeof(uint32_t)), "cudaMallocManaged out_f16x2_rs_relu");
    ck(cudaMallocManaged(&out_f16x2_rs_satfinite, N * sizeof(uint32_t)),
        "cudaMallocManaged out_f16x2_rs_satfinite");
    ck(cudaMallocManaged(&out_f16x2_rs_relu_satfinite, N * sizeof(uint32_t)),
        "cudaMallocManaged out_f16x2_rs_relu_satfinite");

    ck(cudaMallocManaged(&out_bf16_rn, N * sizeof(uint16_t)), "cudaMallocManaged out_bf16_rn");
    ck(cudaMallocManaged(&out_bf16_rz, N * sizeof(uint16_t)), "cudaMallocManaged out_bf16_rz");
    ck(cudaMallocManaged(&out_bf16_rn_relu, N * sizeof(uint16_t)), "cudaMallocManaged out_bf16_rn_relu");
    ck(cudaMallocManaged(&out_bf16_rz_relu, N * sizeof(uint16_t)), "cudaMallocManaged out_bf16_rz_relu");
    ck(cudaMallocManaged(&out_bf16_rn_satfinite, N * sizeof(uint16_t)),
        "cudaMallocManaged out_bf16_rn_satfinite");
    ck(cudaMallocManaged(&out_bf16_rz_satfinite, N * sizeof(uint16_t)),
        "cudaMallocManaged out_bf16_rz_satfinite");
    ck(cudaMallocManaged(&out_bf16_rn_relu_satfinite, N * sizeof(uint16_t)),
        "cudaMallocManaged out_bf16_rn_relu_satfinite");
    ck(cudaMallocManaged(&out_bf16_rz_relu_satfinite, N * sizeof(uint16_t)),
        "cudaMallocManaged out_bf16_rz_relu_satfinite");
    ck(cudaMallocManaged(&out_bf16x2_rn, N * sizeof(uint32_t)), "cudaMallocManaged out_bf16x2_rn");
    ck(cudaMallocManaged(&out_bf16x2_rz, N * sizeof(uint32_t)), "cudaMallocManaged out_bf16x2_rz");
    ck(cudaMallocManaged(&out_bf16x2_rn_relu, N * sizeof(uint32_t)), "cudaMallocManaged out_bf16x2_rn_relu");
    ck(cudaMallocManaged(&out_bf16x2_rz_relu, N * sizeof(uint32_t)), "cudaMallocManaged out_bf16x2_rz_relu");
    ck(cudaMallocManaged(&out_bf16x2_rn_satfinite, N * sizeof(uint32_t)),
        "cudaMallocManaged out_bf16x2_rn_satfinite");
    ck(cudaMallocManaged(&out_bf16x2_rz_satfinite, N * sizeof(uint32_t)),
        "cudaMallocManaged out_bf16x2_rz_satfinite");
    ck(cudaMallocManaged(&out_bf16x2_rn_relu_satfinite, N * sizeof(uint32_t)),
        "cudaMallocManaged out_bf16x2_rn_relu_satfinite");
    ck(cudaMallocManaged(&out_bf16x2_rz_relu_satfinite, N * sizeof(uint32_t)),
        "cudaMallocManaged out_bf16x2_rz_relu_satfinite");
    ck(cudaMallocManaged(&out_bf16x2_rs, N * sizeof(uint32_t)), "cudaMallocManaged out_bf16x2_rs");
    ck(cudaMallocManaged(&out_bf16x2_rs_relu, N * sizeof(uint32_t)), "cudaMallocManaged out_bf16x2_rs_relu");
    ck(cudaMallocManaged(&out_bf16x2_rs_satfinite, N * sizeof(uint32_t)),
        "cudaMallocManaged out_bf16x2_rs_satfinite");
    ck(cudaMallocManaged(&out_bf16x2_rs_relu_satfinite, N * sizeof(uint32_t)),
        "cudaMallocManaged out_bf16x2_rs_relu_satfinite");

    for (int i = 0; i < N; ++i) {
        float base = (float)(i * 0.5f + 0.25f);
        in_a[i] = base;
        in_b[i] = -base * 0.75f;
        in_rbits[i] = 0x12340000u ^ (uint32_t)i;
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    cvt_f16_bf16_frnd2_rs_kernel<<<grid, block>>>(
        in_a,
        in_b,
        in_rbits,
        out_f16_rn,
        out_f16_rz,
        out_f16_rn_relu,
        out_f16_rz_relu,
        out_f16_rn_satfinite,
        out_f16_rz_satfinite,
        out_f16_rn_relu_satfinite,
        out_f16_rz_relu_satfinite,
        out_f16x2_rn,
        out_f16x2_rz,
        out_f16x2_rn_relu,
        out_f16x2_rz_relu,
        out_f16x2_rn_satfinite,
        out_f16x2_rz_satfinite,
        out_f16x2_rn_relu_satfinite,
        out_f16x2_rz_relu_satfinite,
        out_f16x2_rs,
        out_f16x2_rs_relu,
        out_f16x2_rs_satfinite,
        out_f16x2_rs_relu_satfinite,
        out_bf16_rn,
        out_bf16_rz,
        out_bf16_rn_relu,
        out_bf16_rz_relu,
        out_bf16_rn_satfinite,
        out_bf16_rz_satfinite,
        out_bf16_rn_relu_satfinite,
        out_bf16_rz_relu_satfinite,
        out_bf16x2_rn,
        out_bf16x2_rz,
        out_bf16x2_rn_relu,
        out_bf16x2_rz_relu,
        out_bf16x2_rn_satfinite,
        out_bf16x2_rz_satfinite,
        out_bf16x2_rn_relu_satfinite,
        out_bf16x2_rz_relu_satfinite,
        out_bf16x2_rs,
        out_bf16x2_rs_relu,
        out_bf16x2_rs_satfinite,
        out_bf16x2_rs_relu_satfinite
    );

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("f16=0x%04x bf16=0x%04x f16x2=0x%08x bf16x2=0x%08x\n",
        (unsigned int)out_f16_rn[0],
        (unsigned int)out_bf16_rn[0],
        out_f16x2_rn[0],
        out_bf16x2_rn[0]);

    cudaFree(in_a);
    cudaFree(in_b);
    cudaFree(in_rbits);
    cudaFree(out_f16_rn);
    cudaFree(out_f16_rz);
    cudaFree(out_f16_rn_relu);
    cudaFree(out_f16_rz_relu);
    cudaFree(out_f16_rn_satfinite);
    cudaFree(out_f16_rz_satfinite);
    cudaFree(out_f16_rn_relu_satfinite);
    cudaFree(out_f16_rz_relu_satfinite);
    cudaFree(out_f16x2_rn);
    cudaFree(out_f16x2_rz);
    cudaFree(out_f16x2_rn_relu);
    cudaFree(out_f16x2_rz_relu);
    cudaFree(out_f16x2_rn_satfinite);
    cudaFree(out_f16x2_rz_satfinite);
    cudaFree(out_f16x2_rn_relu_satfinite);
    cudaFree(out_f16x2_rz_relu_satfinite);
    cudaFree(out_f16x2_rs);
    cudaFree(out_f16x2_rs_relu);
    cudaFree(out_f16x2_rs_satfinite);
    cudaFree(out_f16x2_rs_relu_satfinite);
    cudaFree(out_bf16_rn);
    cudaFree(out_bf16_rz);
    cudaFree(out_bf16_rn_relu);
    cudaFree(out_bf16_rz_relu);
    cudaFree(out_bf16_rn_satfinite);
    cudaFree(out_bf16_rz_satfinite);
    cudaFree(out_bf16_rn_relu_satfinite);
    cudaFree(out_bf16_rz_relu_satfinite);
    cudaFree(out_bf16x2_rn);
    cudaFree(out_bf16x2_rz);
    cudaFree(out_bf16x2_rn_relu);
    cudaFree(out_bf16x2_rz_relu);
    cudaFree(out_bf16x2_rn_satfinite);
    cudaFree(out_bf16x2_rz_satfinite);
    cudaFree(out_bf16x2_rn_relu_satfinite);
    cudaFree(out_bf16x2_rz_relu_satfinite);
    cudaFree(out_bf16x2_rs);
    cudaFree(out_bf16x2_rs_relu);
    cudaFree(out_bf16x2_rs_satfinite);
    cudaFree(out_bf16x2_rs_relu_satfinite);
    return 0;
}
