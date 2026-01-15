// cvt_basic_xxx.cu

// cvt{.irnd}{.ftz}{.sat}.dtype.atype         d, a;  // integer rounding
// cvt{.frnd}{.ftz}{.sat}.dtype.atype         d, a;  // fp rounding

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#ifndef CVT_TRY_ILLEGAL
#define CVT_TRY_ILLEGAL 0
#endif

#ifndef CVT_TRY_ILLEGAL_SAT
#define CVT_TRY_ILLEGAL_SAT 0
#endif

#if CVT_TRY_ILLEGAL
// Experiment: does `.rna` work for "regular" cvt (non-tf32) forms?
//
// Build example (manual, because Makefile doesn't pass -D):
//   nvcc -DCVT_TRY_ILLEGAL=1 -DCVT_TRY_RNA_CASE=1 -O3 -lineinfo -gencode arch=compute_100a,code=sm_100a cvt_basic_xxx.cu -o /tmp/a.out
//
// Switch `CVT_TRY_RNA_CASE` to test different operands.
#ifndef CVT_TRY_RNA_CASE
#define CVT_TRY_RNA_CASE 1
#endif

__device__ __forceinline__ float cvt_rna_f32_s32(int a) {
    float out;
    asm volatile("cvt.rna.f32.s32 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

__device__ __forceinline__ float cvt_rna_f32_u32(unsigned int a) {
    float out;
    asm volatile("cvt.rna.f32.u32 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

__device__ __forceinline__ float cvt_rna_f32_f64(double a) {
    float out;
    asm volatile("cvt.rna.f32.f64 %0, %1;" : "=f"(out) : "d"(a));
    return out;
}

__device__ __forceinline__ uint16_t cvt_rna_f16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rna.f16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

__device__ __forceinline__ uint16_t cvt_rna_bf16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rna.bf16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

__device__ __forceinline__ float cvt_rna_f32_bf16(uint16_t a) {
    float out;
    asm volatile("cvt.rna.f32.bf16 %0, %1;" : "=f"(out) : "h"(a));
    return out;
}
#endif

// float-to-float with integer rounding (.irnd): round to an integer value, store as fp.
        // /*0080*/                   IMAD.WIDE R2, R7, 0x4, R2 ;            /* 0x0000000407027825 */
        // /* 0x004fcc00078e0202 */
        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND R9, R2 ;                          /* 0x0000000200097307 */
        // /* 0x004e280000201000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;           /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rni_f32_f32(float a) {
    float out;
    asm volatile("cvt.rni.f32.f32 %0, %1;" : "=f"(out) : "f"(a));
    return out;
}

        // /*0080*/                   IMAD.WIDE R2, R7, 0x4, R2 ;            /* 0x0000000407027825 */
        // /* 0x004fcc00078e0202 */
        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.TRUNC R9, R2 ;                    /* 0x0000000200097307 */
        // /* 0x004e28000020d000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;           /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rzi_f32_f32(float a) {
    float out;
    asm volatile("cvt.rzi.f32.f32 %0, %1;" : "=f"(out) : "f"(a));
    return out;
}

        // /*0080*/                   IMAD.WIDE R2, R9, 0x8, R2 ;               /* 0x0000000809027825 */
        // /* 0x004fcc00078e0202 */
        // /*0090*/                   LDG.E.64.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9b00 */
        // /*00a0*/                   IMAD.WIDE R6, R9, 0x8, R6 ;               /* 0x0000000809067825 */
        // /* 0x008fe200078e0206 */
        // /*00b0*/                   FRND.F64 R4, R2 ;                         /* 0x0000000200047313 */
        // /* 0x004e280000301800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;           /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rni_f64_f64(double a) {
    double out;
    asm volatile("cvt.rni.f64.f64 %0, %1;" : "=d"(out) : "d"(a));
    return out;
}

        // /*0080*/                   IMAD.WIDE R2, R9, 0x8, R2 ;               /* 0x0000000809027825 */
        // /* 0x004fcc00078e0202 */
        // /*0090*/                   LDG.E.64.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9b00 */
        // /*00a0*/                   IMAD.WIDE R6, R9, 0x8, R6 ;               /* 0x0000000809067825 */
        // /* 0x008fe200078e0206 */
        // /*00b0*/                   FRND.F64.TRUNC R4, R2 ;                   /* 0x0000000200047313 */
        // /* 0x004e28000030d800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;           /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rzi_f64_f64(double a) {
    double out;
    asm volatile("cvt.rzi.f64.f64 %0, %1;" : "=d"(out) : "d"(a));
    return out;
}

        // /*0080*/                   IMAD.WIDE R2, R7, 0x2, R2 ;                /* 0x0000000207027825 */
        // /* 0x004fcc00078e0202 */
        // /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.F16 R9, R2 ;                          /* 0x0000000200097307 */
        // /* 0x004e280000100800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;           /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rni_f16_f16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rni.f16.f16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0080*/                   IMAD.WIDE R2, R7, 0x2, R2 ;                /* 0x0000000207027825 */
        // /* 0x004fcc00078e0202 */
        // /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.F16.TRUNC R9, R2 ;                    /* 0x0000000200097307 */
        // /* 0x004e28000010c800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;           /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rzi_f16_f16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rzi.f16.f16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0080*/                   IMAD.WIDE R2, R7, 0x2, R2 ;                /* 0x0000000207027825 */
        // /* 0x004fcc00078e0202 */
        // /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.BF16 R9, R2 ;                         /* 0x0000000200097307 */
        // /* 0x004e280000402000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;           /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rni_bf16_bf16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rni.bf16.bf16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0080*/                   IMAD.WIDE R2, R7, 0x2, R2 ;                /* 0x0000000207027825 */
        // /* 0x004fcc00078e0202 */
        // /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.BF16.TRUNC R9, R2 ;                   /* 0x0000000200097307 */
        // /* 0x004e28000040e000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;           /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rzi_bf16_bf16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rzi.bf16.bf16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.BF16.NTZ R9, R2 ;                      /* 0x0000000200097305 */
        // /* 0x004e280000403100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;               /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rni_s32_bf16(uint16_t a) {
    int out;
    asm volatile("cvt.rni.s32.bf16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U32.BF16.TRUNC.NTZ R9, R2 ;            /* 0x0000000200097305 */
        // /* 0x004e28000040f000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;               /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rzi_u32_bf16(uint16_t a) {
    unsigned int out;
    asm volatile("cvt.rzi.u32.bf16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S16.BF16.NTZ R9, R2 ;                  /* 0x0000000200097305 */
        // /* 0x004e280000402900 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;           /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ int16_t cvt_rni_s16_bf16(uint16_t a) {
    int16_t out;
    asm volatile("cvt.rni.s16.bf16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U16.BF16.TRUNC.NTZ R9, R2 ;            /* 0x0000000200097305 */
        // /* 0x004e28000040e800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;           /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rzi_u16_bf16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rzi.u16.bf16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   SHF.L.U32 R7, R2, 0x10, RZ ;               /* 0x0000001002077819 */
        // /* 0x004fca00000006ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;               /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rn_f32_bf16(uint16_t a) {
    float out;
    asm volatile("cvt.rn.f32.bf16 %0, %1;" : "=f"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   SHF.L.U32 R7, R2, 0x10, RZ ;               /* 0x0000001002077819 */
        // /* 0x004fca00000006ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;               /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rz_f32_bf16(uint16_t a) {
    float out;
    asm volatile("cvt.rz.f32.bf16 %0, %1;" : "=f"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9500 */
        // /*00a0*/                   IMAD.WIDE R6, R9, 0x8, R6 ;                /* 0x0000000809067825 */
        // /* 0x008fe200078e0206 */
        // /*00b0*/                   F2F.F64.BF16 R4, R2 ;                      /* 0x0000000200047310 */
        // /* 0x004e280000401800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;            /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rn_f64_bf16(uint16_t a) {
    double out;
    asm volatile("cvt.rn.f64.bf16 %0, %1;" : "=d"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;            /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.BF16 R9, R2 ;                      /* 0x0000000200097306 */
        // /* 0x004e280000202400 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;       /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_bf16_s32(int a) {
    uint16_t out;
    asm volatile("cvt.rn.bf16.s32 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;            /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.BF16.U32.RZ R9, R2 ;               /* 0x0000000200097306 */
        // /* 0x004e28000020e000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;       /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_bf16_u32(unsigned int a) {
    uint16_t out;
    asm volatile("cvt.rz.bf16.u32 %0, %1;" : "=h"(out) : "r"(a));
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
    asm volatile("cvt.rz.relu.bf16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}
    // asm volatile("cvt.rz.relu.bf16.f32 %0, %1;" : "=h"(out) : "f"(a));
    // F2FP.RELU.BF16.F32.MERGE_C.RZ R0, R2, RZ


        // /*0090*/                   LDG.E.64.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;               /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.BF16.F64 R9, R2 ;                     /* 0x0000000200097310 */
        // /* 0x004e280000302000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;          /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_bf16_f64(double a) {
    uint16_t out;
    asm volatile("cvt.rn.bf16.f64 %0, %1;" : "=h"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.BF16.F16 R9, R2 ;                      /* 0x0000000200097304 */
        // /* 0x004e280000102000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;           /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_bf16_f16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rn.bf16.f16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.F16.BF16 R9, R2 ;                      /* 0x0000000200097304 */
        // /* 0x004e280000400800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;           /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_f16_bf16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rn.f16.bf16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

//         /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
//                                                                               /* 0x002ea2000c1e9500 */
//         /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
//                                                                               /* 0x008fe200078e0204 */
//         /*00b0*/                   F2I.S8.BF16.NTZ R9, R2 ;                   /* 0x0000000200097305 */
//                                                                               /* 0x004e280000402100 */
//         /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;               /* 0x0000000904007986 */
//                                                                               /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rni_s8_bf16(uint16_t a) {
    int out;
    asm volatile("{ .reg .s8 t; cvt.rni.s8.bf16 t, %1; cvt.s32.s8 %0, t; }"
                 : "=r"(out) : "h"(a));
    return out;
}

//         /*0090*/                   LDG.E.U16.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
//                                                                               /* 0x002ea2000c1e9500 */
//         /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
//                                                                               /* 0x008fe200078e0204 */
//         /*00b0*/                   F2I.U8.BF16.TRUNC.NTZ R9, R2 ;             /* 0x0000000200097305 */
//                                                                               /* 0x004e28000040e000 */
//         /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;               /* 0x0000000904007986 */
//                                                                               /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rzi_u8_bf16(uint16_t a) {
    unsigned int out;
    asm volatile("{ .reg .u8 t; cvt.rzi.u8.bf16 t, %1; cvt.u32.u8 %0, t; }"
                 : "=r"(out) : "h"(a));
    return out;
}

#if CVT_TRY_ILLEGAL && CVT_TRY_ILLEGAL_SAT
__device__ __forceinline__ int cvt_rni_sat_s8_bf16(uint16_t a) {
    int out;
    asm volatile("{ .reg .s8 t; cvt.rni.sat.s8.bf16 t, %1; cvt.s32.s8 %0, t; }"
                 : "=r"(out) : "h"(a));
    return out;
}

__device__ __forceinline__ unsigned int cvt_rzi_sat_u8_bf16(uint16_t a) {
    unsigned int out;
    asm volatile("{ .reg .u8 t; cvt.rzi.sat.u8.bf16 t, %1; cvt.u32.u8 %0, t; }"
                 : "=r"(out) : "h"(a));
    return out;
}

#endif

extern "C" __global__ void cvt_basic_xxx_kernel(
    const uint16_t* __restrict__ in_bf16,
    const uint16_t* __restrict__ in_f16,
    const float* __restrict__ in_f32,
    const double* __restrict__ in_f64,
    const int* __restrict__ in_s32,
    const unsigned int* __restrict__ in_u32,
    int* __restrict__ out_s32_from_bf16_rni,
    unsigned int* __restrict__ out_u32_from_bf16_rzi,
    int16_t* __restrict__ out_s16_from_bf16_rni,
    uint16_t* __restrict__ out_u16_from_bf16_rzi,
    float* __restrict__ out_f32_from_bf16_rn,
    float* __restrict__ out_f32_from_bf16_rz,
    double* __restrict__ out_f64_from_bf16_rn,
    uint16_t* __restrict__ out_bf16_from_s32_rn,
    uint16_t* __restrict__ out_bf16_from_u32_rz,
    uint16_t* __restrict__ out_bf16_from_f32_rn,
    uint16_t* __restrict__ out_bf16_from_f32_rz,
    uint16_t* __restrict__ out_bf16_from_f64_rn,
    uint16_t* __restrict__ out_bf16_from_f16_rn,
    uint16_t* __restrict__ out_f16_from_bf16_rn,
    int* __restrict__ out_s8_from_bf16_rni,
    unsigned int* __restrict__ out_u8_from_bf16_rzi,
    float* __restrict__ out_f32_from_f32_rni,
    float* __restrict__ out_f32_from_f32_rzi,
    double* __restrict__ out_f64_from_f64_rni,
    double* __restrict__ out_f64_from_f64_rzi,
    uint16_t* __restrict__ out_f16_from_f16_rni,
    uint16_t* __restrict__ out_f16_from_f16_rzi,
    uint16_t* __restrict__ out_bf16_from_bf16_rni,
    uint16_t* __restrict__ out_bf16_from_bf16_rzi
#if CVT_TRY_ILLEGAL && CVT_TRY_ILLEGAL_SAT
    , int* __restrict__ out_s8_from_bf16_rni_sat
    , unsigned int* __restrict__ out_u8_from_bf16_rzi_sat
#endif
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    uint16_t bf16 = in_bf16[tid];
    uint16_t f16 = in_f16[tid];
    float f32 = in_f32[tid];
    double f64 = in_f64[tid];
    int s32 = in_s32[tid];
    unsigned int u32 = in_u32[tid];

    out_s32_from_bf16_rni[tid] = cvt_rni_s32_bf16(bf16);
    out_u32_from_bf16_rzi[tid] = cvt_rzi_u32_bf16(bf16);
    out_s16_from_bf16_rni[tid] = cvt_rni_s16_bf16(bf16);
    out_u16_from_bf16_rzi[tid] = cvt_rzi_u16_bf16(bf16);
    out_f32_from_bf16_rn[tid] = cvt_rn_f32_bf16(bf16);
    out_f32_from_bf16_rz[tid] = cvt_rz_f32_bf16(bf16);
    out_f64_from_bf16_rn[tid] = cvt_rn_f64_bf16(bf16);
    out_bf16_from_s32_rn[tid] = cvt_rn_bf16_s32(s32);
    out_bf16_from_u32_rz[tid] = cvt_rz_bf16_u32(u32);
    out_bf16_from_f32_rn[tid] = cvt_rn_bf16_f32(f32);
    out_bf16_from_f32_rz[tid] = cvt_rz_bf16_f32(f32);
    out_bf16_from_f64_rn[tid] = cvt_rn_bf16_f64(f64);
    out_bf16_from_f16_rn[tid] = cvt_rn_bf16_f16(f16);
    out_f16_from_bf16_rn[tid] = cvt_rn_f16_bf16(bf16);
    out_s8_from_bf16_rni[tid] = cvt_rni_s8_bf16(bf16);
    out_u8_from_bf16_rzi[tid] = cvt_rzi_u8_bf16(bf16);
    out_f32_from_f32_rni[tid] = cvt_rni_f32_f32(f32);
    out_f32_from_f32_rzi[tid] = cvt_rzi_f32_f32(f32);
    out_f64_from_f64_rni[tid] = cvt_rni_f64_f64(f64);
    out_f64_from_f64_rzi[tid] = cvt_rzi_f64_f64(f64);
    out_f16_from_f16_rni[tid] = cvt_rni_f16_f16(f16);
    out_f16_from_f16_rzi[tid] = cvt_rzi_f16_f16(f16);
    out_bf16_from_bf16_rni[tid] = cvt_rni_bf16_bf16(bf16);
    out_bf16_from_bf16_rzi[tid] = cvt_rzi_bf16_bf16(bf16);

#if CVT_TRY_ILLEGAL
#if CVT_TRY_RNA_CASE == 1
    out_f32_from_f32_rni[tid] = cvt_rna_f32_s32(s32);
#elif CVT_TRY_RNA_CASE == 2
    out_f32_from_f32_rni[tid] = cvt_rna_f32_u32(u32);
#elif CVT_TRY_RNA_CASE == 3
    out_f32_from_f32_rni[tid] = cvt_rna_f32_f64(f64);
#elif CVT_TRY_RNA_CASE == 4
    out_f16_from_f16_rni[tid] = cvt_rna_f16_f32(f32);
#elif CVT_TRY_RNA_CASE == 5
    out_bf16_from_bf16_rni[tid] = cvt_rna_bf16_f32(f32);
#elif CVT_TRY_RNA_CASE == 6
    out_f32_from_f32_rni[tid] = cvt_rna_f32_bf16(bf16);
#endif
#endif
#if CVT_TRY_ILLEGAL && CVT_TRY_ILLEGAL_SAT
    out_s8_from_bf16_rni_sat[tid] = cvt_rni_sat_s8_bf16(bf16);
    out_u8_from_bf16_rzi_sat[tid] = cvt_rzi_sat_u8_bf16(bf16);
#endif
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    constexpr int N = 256;

    uint16_t* in_bf16;
    uint16_t* in_f16;
    float* in_f32;
    double* in_f64;
    int* in_s32;
    unsigned int* in_u32;

    int* out_s32_from_bf16_rni;
    unsigned int* out_u32_from_bf16_rzi;
    int16_t* out_s16_from_bf16_rni;
    uint16_t* out_u16_from_bf16_rzi;
    float* out_f32_from_bf16_rn;
    float* out_f32_from_bf16_rz;
    double* out_f64_from_bf16_rn;
    uint16_t* out_bf16_from_s32_rn;
    uint16_t* out_bf16_from_u32_rz;
    uint16_t* out_bf16_from_f32_rn;
    uint16_t* out_bf16_from_f32_rz;
    uint16_t* out_bf16_from_f64_rn;
    uint16_t* out_bf16_from_f16_rn;
    uint16_t* out_f16_from_bf16_rn;
    int* out_s8_from_bf16_rni;
    unsigned int* out_u8_from_bf16_rzi;
    float* out_f32_from_f32_rni;
    float* out_f32_from_f32_rzi;
    double* out_f64_from_f64_rni;
    double* out_f64_from_f64_rzi;
    uint16_t* out_f16_from_f16_rni;
    uint16_t* out_f16_from_f16_rzi;
    uint16_t* out_bf16_from_bf16_rni;
    uint16_t* out_bf16_from_bf16_rzi;
#if CVT_TRY_ILLEGAL && CVT_TRY_ILLEGAL_SAT
    int* out_s8_from_bf16_rni_sat;
    unsigned int* out_u8_from_bf16_rzi_sat;
#endif

    ck(cudaMallocManaged(&in_bf16, N * sizeof(uint16_t)), "cudaMallocManaged in_bf16");
    ck(cudaMallocManaged(&in_f16, N * sizeof(uint16_t)), "cudaMallocManaged in_f16");
    ck(cudaMallocManaged(&in_f32, N * sizeof(float)), "cudaMallocManaged in_f32");
    ck(cudaMallocManaged(&in_f64, N * sizeof(double)), "cudaMallocManaged in_f64");
    ck(cudaMallocManaged(&in_s32, N * sizeof(int)), "cudaMallocManaged in_s32");
    ck(cudaMallocManaged(&in_u32, N * sizeof(unsigned int)), "cudaMallocManaged in_u32");

    ck(cudaMallocManaged(&out_s32_from_bf16_rni, N * sizeof(int)), "cudaMallocManaged out_s32_from_bf16_rni");
    ck(cudaMallocManaged(&out_u32_from_bf16_rzi, N * sizeof(unsigned int)), "cudaMallocManaged out_u32_from_bf16_rzi");
    ck(cudaMallocManaged(&out_s16_from_bf16_rni, N * sizeof(int16_t)), "cudaMallocManaged out_s16_from_bf16_rni");
    ck(cudaMallocManaged(&out_u16_from_bf16_rzi, N * sizeof(uint16_t)), "cudaMallocManaged out_u16_from_bf16_rzi");
    ck(cudaMallocManaged(&out_f32_from_bf16_rn, N * sizeof(float)), "cudaMallocManaged out_f32_from_bf16_rn");
    ck(cudaMallocManaged(&out_f32_from_bf16_rz, N * sizeof(float)), "cudaMallocManaged out_f32_from_bf16_rz");
    ck(cudaMallocManaged(&out_f64_from_bf16_rn, N * sizeof(double)), "cudaMallocManaged out_f64_from_bf16_rn");
    ck(cudaMallocManaged(&out_bf16_from_s32_rn, N * sizeof(uint16_t)), "cudaMallocManaged out_bf16_from_s32_rn");
    ck(cudaMallocManaged(&out_bf16_from_u32_rz, N * sizeof(uint16_t)), "cudaMallocManaged out_bf16_from_u32_rz");
    ck(cudaMallocManaged(&out_bf16_from_f32_rn, N * sizeof(uint16_t)), "cudaMallocManaged out_bf16_from_f32_rn");
    ck(cudaMallocManaged(&out_bf16_from_f32_rz, N * sizeof(uint16_t)), "cudaMallocManaged out_bf16_from_f32_rz");
    ck(cudaMallocManaged(&out_bf16_from_f64_rn, N * sizeof(uint16_t)), "cudaMallocManaged out_bf16_from_f64_rn");
    ck(cudaMallocManaged(&out_bf16_from_f16_rn, N * sizeof(uint16_t)), "cudaMallocManaged out_bf16_from_f16_rn");
    ck(cudaMallocManaged(&out_f16_from_bf16_rn, N * sizeof(uint16_t)), "cudaMallocManaged out_f16_from_bf16_rn");
    ck(cudaMallocManaged(&out_s8_from_bf16_rni, N * sizeof(int)), "cudaMallocManaged out_s8_from_bf16_rni");
    ck(cudaMallocManaged(&out_u8_from_bf16_rzi, N * sizeof(unsigned int)),
        "cudaMallocManaged out_u8_from_bf16_rzi");
    ck(cudaMallocManaged(&out_f32_from_f32_rni, N * sizeof(float)), "cudaMallocManaged out_f32_from_f32_rni");
    ck(cudaMallocManaged(&out_f32_from_f32_rzi, N * sizeof(float)), "cudaMallocManaged out_f32_from_f32_rzi");
    ck(cudaMallocManaged(&out_f64_from_f64_rni, N * sizeof(double)), "cudaMallocManaged out_f64_from_f64_rni");
    ck(cudaMallocManaged(&out_f64_from_f64_rzi, N * sizeof(double)), "cudaMallocManaged out_f64_from_f64_rzi");
    ck(cudaMallocManaged(&out_f16_from_f16_rni, N * sizeof(uint16_t)), "cudaMallocManaged out_f16_from_f16_rni");
    ck(cudaMallocManaged(&out_f16_from_f16_rzi, N * sizeof(uint16_t)), "cudaMallocManaged out_f16_from_f16_rzi");
    ck(cudaMallocManaged(&out_bf16_from_bf16_rni, N * sizeof(uint16_t)), "cudaMallocManaged out_bf16_from_bf16_rni");
    ck(cudaMallocManaged(&out_bf16_from_bf16_rzi, N * sizeof(uint16_t)), "cudaMallocManaged out_bf16_from_bf16_rzi");
#if CVT_TRY_ILLEGAL && CVT_TRY_ILLEGAL_SAT
    ck(cudaMallocManaged(&out_s8_from_bf16_rni_sat, N * sizeof(int)), "cudaMallocManaged out_u32_from_bf16_rn");
    ck(cudaMallocManaged(&out_u8_from_bf16_rzi_sat, N * sizeof(unsigned int)), "cudaMallocManaged out_f32_from_bf16_rni");
#endif

    for (int i = 0; i < N; ++i) {
        float base = (float)(i * 0.25f + 0.5f);
        in_bf16[i] = (uint16_t)(0x3f80u + (uint16_t)(i & 0x7fu));
        in_f16[i] = (uint16_t)(0x3c00u + (uint16_t)(i & 0x7fu));
        in_f32[i] = base;
        in_f64[i] = (double)base * 1.25;
        in_s32[i] = (int)(i * 3 - 123);
        in_u32[i] = (unsigned int)(i * 7 + 5);
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    cvt_basic_xxx_kernel<<<grid, block>>>(
        in_bf16,
        in_f16,
        in_f32,
        in_f64,
        in_s32,
        in_u32,
        out_s32_from_bf16_rni,
        out_u32_from_bf16_rzi,
        out_s16_from_bf16_rni,
        out_u16_from_bf16_rzi,
        out_f32_from_bf16_rn,
        out_f32_from_bf16_rz,
        out_f64_from_bf16_rn,
        out_bf16_from_s32_rn,
        out_bf16_from_u32_rz,
        out_bf16_from_f32_rn,
        out_bf16_from_f32_rz,
        out_bf16_from_f64_rn,
        out_bf16_from_f16_rn,
        out_f16_from_bf16_rn,
        out_s8_from_bf16_rni,
        out_u8_from_bf16_rzi,
        out_f32_from_f32_rni,
        out_f32_from_f32_rzi,
        out_f64_from_f64_rni,
        out_f64_from_f64_rzi,
        out_f16_from_f16_rni,
        out_f16_from_f16_rzi,
        out_bf16_from_bf16_rni,
        out_bf16_from_bf16_rzi
#if CVT_TRY_ILLEGAL && CVT_TRY_ILLEGAL_SAT
        , out_s8_from_bf16_rni_sat
        , out_u8_from_bf16_rzi_sat
#endif
    );

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("s32=%d u32=%u f32=%.2f f64=%.2f bf16=0x%04x f16=0x%04x\n",
        out_s32_from_bf16_rni[0],
        out_u32_from_bf16_rzi[0],
        out_f32_from_bf16_rn[0],
        out_f64_from_bf16_rn[0],
        (unsigned int)out_bf16_from_f32_rn[0],
        (unsigned int)out_f16_from_bf16_rn[0]);
    std::printf("irnd: f32_in=%.2f rni=%.2f rzi=%.2f | f64_in=%.2f rni=%.2f rzi=%.2f\n",
        in_f32[0],
        out_f32_from_f32_rni[0],
        out_f32_from_f32_rzi[0],
        in_f64[0],
        out_f64_from_f64_rni[0],
        out_f64_from_f64_rzi[0]);
    std::printf("irnd: f16_in=0x%04x rni=0x%04x rzi=0x%04x | bf16_in=0x%04x rni=0x%04x rzi=0x%04x\n",
        (unsigned int)in_f16[0],
        (unsigned int)out_f16_from_f16_rni[0],
        (unsigned int)out_f16_from_f16_rzi[0],
        (unsigned int)in_bf16[0],
        (unsigned int)out_bf16_from_bf16_rni[0],
        (unsigned int)out_bf16_from_bf16_rzi[0]);

    cudaFree(in_bf16);
    cudaFree(in_f16);
    cudaFree(in_f32);
    cudaFree(in_f64);
    cudaFree(in_s32);
    cudaFree(in_u32);
    cudaFree(out_s32_from_bf16_rni);
    cudaFree(out_u32_from_bf16_rzi);
    cudaFree(out_s16_from_bf16_rni);
    cudaFree(out_u16_from_bf16_rzi);
    cudaFree(out_f32_from_bf16_rn);
    cudaFree(out_f32_from_bf16_rz);
    cudaFree(out_f64_from_bf16_rn);
    cudaFree(out_bf16_from_s32_rn);
    cudaFree(out_bf16_from_u32_rz);
    cudaFree(out_bf16_from_f32_rn);
    cudaFree(out_bf16_from_f32_rz);
    cudaFree(out_bf16_from_f64_rn);
    cudaFree(out_bf16_from_f16_rn);
    cudaFree(out_f16_from_bf16_rn);
    cudaFree(out_s8_from_bf16_rni);
    cudaFree(out_u8_from_bf16_rzi);
    cudaFree(out_f32_from_f32_rni);
    cudaFree(out_f32_from_f32_rzi);
    cudaFree(out_f64_from_f64_rni);
    cudaFree(out_f64_from_f64_rzi);
    cudaFree(out_f16_from_f16_rni);
    cudaFree(out_f16_from_f16_rzi);
    cudaFree(out_bf16_from_bf16_rni);
    cudaFree(out_bf16_from_bf16_rzi);
#if CVT_TRY_ILLEGAL && CVT_TRY_ILLEGAL_SAT
    cudaFree(out_s8_from_bf16_rni_sat);
    cudaFree(out_u8_from_bf16_rzi_sat);
#endif
    return 0;
}
