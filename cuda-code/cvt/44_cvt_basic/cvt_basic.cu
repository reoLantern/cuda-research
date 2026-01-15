// cvt_basic.cu
//
// PTX cvt (generic forms):
//   cvt{.irnd}{.ftz}{.sat}.dtype.atype   d, a;  // integer rounding
//   cvt{.frnd}{.ftz}{.sat}.dtype.atype   d, a;  // fp rounding
//
// Types covered here (PTX):
//   .dtype/.atype = { .u8, .u16, .u32, .u64, .s8, .s16, .s32, .s64, .bf16, .f16, .f32, .f64 }
//
// Notes:
// - The SASS blocks above each wrapper are from an isolate kernel that only performs that conversion,
//   compiled for sm_103a.
// - Special cvt forms (frnd2/rs/tf32/f8/f4/f6/ue8m0/s2f6x2) are covered in other folders.

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// ---- wrappers ----

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.BF16.FLOOR R9, R2 ;                     /* 0x0000000200097307 */
        // /* 0x004e280000406000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rmi_bf16_bf16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rmi.bf16.bf16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.BF16 R9, R2 ;                           /* 0x0000000200097307 */
        // /* 0x004e280000402000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rni_bf16_bf16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rni.bf16.bf16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.BF16.CEIL R9, R2 ;                      /* 0x0000000200097307 */
        // /* 0x004e28000040a000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rpi_bf16_bf16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rpi.bf16.bf16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.BF16.TRUNC R9, R2 ;                     /* 0x0000000200097307 */
        // /* 0x004e28000040e000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rzi_bf16_bf16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rzi.bf16.bf16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.BF16.F16.RM R9, R2 ;                     /* 0x0000000200097304 */
        // /* 0x004e280000106000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rm_bf16_f16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rm.bf16.f16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.BF16.F16 R9, R2 ;                        /* 0x0000000200097304 */
        // /* 0x004e280000102000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_bf16_f16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rn.bf16.f16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.BF16.F16.RP R9, R2 ;                     /* 0x0000000200097304 */
        // /* 0x004e28000010a000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rp_bf16_f16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rp.bf16.f16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.BF16.F16.RZ R9, R2 ;                     /* 0x0000000200097304 */
        // /* 0x004e28000010e000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_bf16_f16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rz.bf16.f16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.BF16.F32.RM R9, R2 ;                 /* 0x0000000200097304 */
        // /* 0x004e280000206000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rm_bf16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rm.bf16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.BF16.F32.PACK_AB R0, RZ, R2 ;       /* 0x00000002ff00723e */
        // /* 0x004fca00000010ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;         /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_bf16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.FTZ.BF16.F32 R9, R2 ;                /* 0x0000000200097304 */
        // /* 0x004e280000212000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_ftz_bf16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rn.ftz.bf16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.BF16.F32.RP R9, R2 ;                 /* 0x0000000200097304 */
        // /* 0x004e28000020a000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rp_bf16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rp.bf16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.BF16.F32.PACK_AB.RZ R0, RZ, R2 ;    /* 0x00000002ff00723e */
        // /* 0x004fca00000190ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;         /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_bf16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rz.bf16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.FTZ.BF16.F32.RZ R9, R2 ;             /* 0x0000000200097304 */
        // /* 0x004e28000021e000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_ftz_bf16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rz.ftz.bf16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                 /* 0x0000000207047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   F2F.BF16.F64.RM R7, R2 ;                    /* 0x0000000200077310 */
        // /* 0x004e260000306000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;            /* 0x0000000704007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rm_bf16_f64(double a) {
    uint16_t out;
    asm volatile("cvt.rm.bf16.f64 %0, %1;" : "=h"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                 /* 0x0000000207047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   F2F.BF16.F64 R7, R2 ;                       /* 0x0000000200077310 */
        // /* 0x004e260000302000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;            /* 0x0000000704007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_bf16_f64(double a) {
    uint16_t out;
    asm volatile("cvt.rn.bf16.f64 %0, %1;" : "=h"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                 /* 0x0000000207047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   F2F.BF16.F64.RP R7, R2 ;                    /* 0x0000000200077310 */
        // /* 0x004e26000030a000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;            /* 0x0000000704007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rp_bf16_f64(double a) {
    uint16_t out;
    asm volatile("cvt.rp.bf16.f64 %0, %1;" : "=h"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                 /* 0x0000000207047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   F2F.BF16.F64.RZ R7, R2 ;                    /* 0x0000000200077310 */
        // /* 0x004e26000030e000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;            /* 0x0000000704007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_bf16_f64(double a) {
    uint16_t out;
    asm volatile("cvt.rz.bf16.f64 %0, %1;" : "=h"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.BF16 R9, R2 ;                        /* 0x0000000200097306 */
        // /* 0x004e280000202400 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_bf16_s32(int a) {
    uint16_t out;
    asm volatile("cvt.rn.bf16.s32 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.BF16.RZ R9, R2 ;                     /* 0x0000000200097306 */
        // /* 0x004e28000020e400 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_bf16_s32(int a) {
    uint16_t out;
    asm volatile("cvt.rz.bf16.s32 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                 /* 0x0000000207047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   I2F.BF16.S64 R7, R2 ;                       /* 0x0000000200077312 */
        // /* 0x004e260000302400 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;            /* 0x0000000704007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_bf16_s64(long long a) {
    uint16_t out;
    asm volatile("cvt.rn.bf16.s64 %0, %1;" : "=h"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                 /* 0x0000000207047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   I2F.BF16.S64.RZ R7, R2 ;                    /* 0x0000000200077312 */
        // /* 0x004e26000030e400 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;            /* 0x0000000704007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_bf16_s64(long long a) {
    uint16_t out;
    asm volatile("cvt.rz.bf16.s64 %0, %1;" : "=h"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.BF16.U32 R9, R2 ;                    /* 0x0000000200097306 */
        // /* 0x004e280000202000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_bf16_u32(unsigned int a) {
    uint16_t out;
    asm volatile("cvt.rn.bf16.u32 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.BF16.U32.RZ R9, R2 ;                 /* 0x0000000200097306 */
        // /* 0x004e28000020e000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_bf16_u32(unsigned int a) {
    uint16_t out;
    asm volatile("cvt.rz.bf16.u32 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                 /* 0x0000000207047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   I2F.BF16.U64 R7, R2 ;                       /* 0x0000000200077312 */
        // /* 0x004e260000302000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;            /* 0x0000000704007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_bf16_u64(unsigned long long a) {
    uint16_t out;
    asm volatile("cvt.rn.bf16.u64 %0, %1;" : "=h"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                 /* 0x0000000207047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   I2F.BF16.U64.RZ R7, R2 ;                    /* 0x0000000200077312 */
        // /* 0x004e26000030e000 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;            /* 0x0000000704007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_bf16_u64(unsigned long long a) {
    uint16_t out;
    asm volatile("cvt.rz.bf16.u64 %0, %1;" : "=h"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.F16.BF16.RM R9, R2 ;                     /* 0x0000000200097304 */
        // /* 0x004e280000404800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rm_f16_bf16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rm.f16.bf16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.F16.BF16 R9, R2 ;                        /* 0x0000000200097304 */
        // /* 0x004e280000400800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_f16_bf16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rn.f16.bf16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.F16.BF16.RP R9, R2 ;                     /* 0x0000000200097304 */
        // /* 0x004e280000408800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rp_f16_bf16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rp.f16.bf16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.F16.BF16.RZ R9, R2 ;                     /* 0x0000000200097304 */
        // /* 0x004e28000040c800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_f16_bf16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rz.f16.bf16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.F16.FLOOR R9, R2 ;                      /* 0x0000000200097307 */
        // /* 0x004e280000104800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rmi_f16_f16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rmi.f16.f16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.F16.FLOOR R0, R2 ;                      /* 0x0000000200007307 */
        // /* 0x004e260000104800 */
        // /*00c0*/                   HADD2.SAT R0, R0.H0_H0, -RZ.H0_H0 ;          /* 0xa00000ff00007230 */
        // /* 0x001fca0000002800 */
        // /*00d0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;             /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rmi_sat_f16_f16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rmi.sat.f16.f16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.F16 R9, R2 ;                            /* 0x0000000200097307 */
        // /* 0x004e280000100800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rni_f16_f16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rni.f16.f16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.F16 R0, R2 ;                            /* 0x0000000200007307 */
        // /* 0x004e260000100800 */
        // /*00c0*/                   HADD2.SAT R0, R0.H0_H0, -RZ.H0_H0 ;          /* 0xa00000ff00007230 */
        // /* 0x001fca0000002800 */
        // /*00d0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;             /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rni_sat_f16_f16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rni.sat.f16.f16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.F16.CEIL R9, R2 ;                       /* 0x0000000200097307 */
        // /* 0x004e280000108800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rpi_f16_f16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rpi.f16.f16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.F16.CEIL R0, R2 ;                       /* 0x0000000200007307 */
        // /* 0x004e260000108800 */
        // /*00c0*/                   HADD2.SAT R0, R0.H0_H0, -RZ.H0_H0 ;          /* 0xa00000ff00007230 */
        // /* 0x001fca0000002800 */
        // /*00d0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;             /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rpi_sat_f16_f16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rpi.sat.f16.f16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.F16.TRUNC R9, R2 ;                      /* 0x0000000200097307 */
        // /* 0x004e28000010c800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rzi_f16_f16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rzi.f16.f16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                  /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.F16.TRUNC R0, R2 ;                      /* 0x0000000200007307 */
        // /* 0x004e26000010c800 */
        // /*00c0*/                   HADD2.SAT R0, R0.H0_H0, -RZ.H0_H0 ;          /* 0xa00000ff00007230 */
        // /* 0x001fca0000002800 */
        // /*00d0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;             /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rzi_sat_f16_f16(uint16_t a) {
    uint16_t out;
    asm volatile("cvt.rzi.sat.f16.f16 %0, %1;" : "=h"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.F16.F32.RM R9, R2 ;                  /* 0x0000000200097304 */
        // /* 0x004e280000204800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rm_f16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rm.f16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.F16.F32.PACK_AB R0, RZ, R2 ;        /* 0x00000002ff00723e */
        // /* 0x004fca00000000ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;         /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_f16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.F16.F32.PACK_AB R0, RZ, R2 ;        /* 0x00000002ff00723e */
        // /* 0x004fca00000000ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;         /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_ftz_f16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rn.ftz.f16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.F16.F32.PACK_AB R0, RZ, R2 ;        /* 0x00000002ff00723e */
        // /* 0x004fca00000000ff */
        // /*00c0*/                   HADD2.SAT R0, R0.H0_H0, -RZ.H0_H0 ;      /* 0xa00000ff00007230 */
        // /* 0x000fca0000002800 */
        // /*00d0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;         /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_sat_f16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rn.sat.f16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.F16.F32.RP R9, R2 ;                  /* 0x0000000200097304 */
        // /* 0x004e280000208800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rp_f16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rp.f16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.F16.F32.PACK_AB.RZ R0, RZ, R2 ;     /* 0x00000002ff00723e */
        // /* 0x004fca00000180ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;         /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_f16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rz.f16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.F16.F32.PACK_AB.RZ R0, RZ, R2 ;     /* 0x00000002ff00723e */
        // /* 0x004fca00000180ff */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;         /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_ftz_f16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rz.ftz.f16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2FP.F16.F32.PACK_AB.RZ R0, RZ, R2 ;     /* 0x00000002ff00723e */
        // /* 0x004fca00000180ff */
        // /*00c0*/                   HADD2.SAT R0, R0.H0_H0, -RZ.H0_H0 ;      /* 0xa00000ff00007230 */
        // /* 0x000fca0000002800 */
        // /*00d0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;         /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_sat_f16_f32(float a) {
    uint16_t out;
    asm volatile("cvt.rz.sat.f16.f32 %0, %1;" : "=h"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                 /* 0x0000000207047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   F2F.F16.F64.RM R7, R2 ;                     /* 0x0000000200077310 */
        // /* 0x004e260000304800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;            /* 0x0000000704007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rm_f16_f64(double a) {
    uint16_t out;
    asm volatile("cvt.rm.f16.f64 %0, %1;" : "=h"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                 /* 0x0000000207047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   F2F.F16.F64 R7, R2 ;                        /* 0x0000000200077310 */
        // /* 0x004e260000300800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;            /* 0x0000000704007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_f16_f64(double a) {
    uint16_t out;
    asm volatile("cvt.rn.f16.f64 %0, %1;" : "=h"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                 /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.F16.F64 R0, R2 ;                        /* 0x0000000200007310 */
        // /* 0x004e260000300800 */
        // /*00c0*/                   HADD2.SAT R0, R0.H0_H0, -RZ.H0_H0 ;         /* 0xa00000ff00007230 */
        // /* 0x001fca0000002800 */
        // /*00d0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;            /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_sat_f16_f64(double a) {
    uint16_t out;
    asm volatile("cvt.rn.sat.f16.f64 %0, %1;" : "=h"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                 /* 0x0000000207047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   F2F.F16.F64.RP R7, R2 ;                     /* 0x0000000200077310 */
        // /* 0x004e260000308800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;            /* 0x0000000704007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rp_f16_f64(double a) {
    uint16_t out;
    asm volatile("cvt.rp.f16.f64 %0, %1;" : "=h"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                 /* 0x0000000207047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   F2F.F16.F64.RZ R7, R2 ;                     /* 0x0000000200077310 */
        // /* 0x004e26000030c800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;            /* 0x0000000704007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_f16_f64(double a) {
    uint16_t out;
    asm volatile("cvt.rz.f16.f64 %0, %1;" : "=h"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                 /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.F16.F64.RZ R0, R2 ;                     /* 0x0000000200007310 */
        // /* 0x004e26000030c800 */
        // /*00c0*/                   HADD2.SAT R0, R0.H0_H0, -RZ.H0_H0 ;         /* 0xa00000ff00007230 */
        // /* 0x001fca0000002800 */
        // /*00d0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;            /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_sat_f16_f64(double a) {
    uint16_t out;
    asm volatile("cvt.rz.sat.f16.f64 %0, %1;" : "=h"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.F16.S16 R9, R2 ;                     /* 0x0000000200097306 */
        // /* 0x004e280000100c00 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_f16_s16(int a) {
    uint16_t out;
    asm volatile("cvt.rn.f16.s16 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.F16.S16.RZ R9, R2 ;                  /* 0x0000000200097306 */
        // /* 0x004e28000010cc00 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_f16_s16(int a) {
    uint16_t out;
    asm volatile("cvt.rz.f16.s16 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.F16 R9, R2 ;                         /* 0x0000000200097306 */
        // /* 0x004e280000200c00 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_f16_s32(int a) {
    uint16_t out;
    asm volatile("cvt.rn.f16.s32 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.F16 R0, R2 ;                         /* 0x0000000200007306 */
        // /* 0x004e260000200c00 */
        // /*00c0*/                   HADD2.SAT R0, R0.H0_H0, -RZ.H0_H0 ;      /* 0xa00000ff00007230 */
        // /* 0x001fca0000002800 */
        // /*00d0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;         /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_sat_f16_s32(int a) {
    uint16_t out;
    asm volatile("cvt.rn.sat.f16.s32 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.F16.RZ R9, R2 ;                      /* 0x0000000200097306 */
        // /* 0x004e28000020cc00 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_f16_s32(int a) {
    uint16_t out;
    asm volatile("cvt.rz.f16.s32 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                 /* 0x0000000207047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   I2F.F16.S64 R7, R2 ;                        /* 0x0000000200077312 */
        // /* 0x004e260000300c00 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;            /* 0x0000000704007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_f16_s64(long long a) {
    uint16_t out;
    asm volatile("cvt.rn.f16.s64 %0, %1;" : "=h"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                 /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.F16.S64 R0, R2 ;                        /* 0x0000000200007312 */
        // /* 0x004e260000300c00 */
        // /*00c0*/                   HADD2.SAT R0, R0.H0_H0, -RZ.H0_H0 ;         /* 0xa00000ff00007230 */
        // /* 0x001fca0000002800 */
        // /*00d0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;            /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_sat_f16_s64(long long a) {
    uint16_t out;
    asm volatile("cvt.rn.sat.f16.s64 %0, %1;" : "=h"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                 /* 0x0000000207047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   I2F.F16.S64.RZ R7, R2 ;                     /* 0x0000000200077312 */
        // /* 0x004e26000030cc00 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;            /* 0x0000000704007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_f16_s64(long long a) {
    uint16_t out;
    asm volatile("cvt.rz.f16.s64 %0, %1;" : "=h"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.F16.S8 R9, R2 ;                      /* 0x0000000200097306 */
        // /* 0x004e280000000c00 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_f16_s8(int a) {
    uint16_t out;
    asm volatile("cvt.rn.f16.s8 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.F16.S8.RZ R9, R2 ;                   /* 0x0000000200097306 */
        // /* 0x004e28000000cc00 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_f16_s8(int a) {
    uint16_t out;
    asm volatile("cvt.rz.f16.s8 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.F16.U16 R9, R2 ;                     /* 0x0000000200097306 */
        // /* 0x004e280000100800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_f16_u16(unsigned int a) {
    uint16_t out;
    asm volatile("cvt.rn.f16.u16 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.F16.U16.RZ R9, R2 ;                  /* 0x0000000200097306 */
        // /* 0x004e28000010c800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_f16_u16(unsigned int a) {
    uint16_t out;
    asm volatile("cvt.rz.f16.u16 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.F16.U32 R9, R2 ;                     /* 0x0000000200097306 */
        // /* 0x004e280000200800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_f16_u32(unsigned int a) {
    uint16_t out;
    asm volatile("cvt.rn.f16.u32 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.F16.U32 R0, R2 ;                     /* 0x0000000200007306 */
        // /* 0x004e260000200800 */
        // /*00c0*/                   HADD2.SAT R0, R0.H0_H0, -RZ.H0_H0 ;      /* 0xa00000ff00007230 */
        // /* 0x001fca0000002800 */
        // /*00d0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;         /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_sat_f16_u32(unsigned int a) {
    uint16_t out;
    asm volatile("cvt.rn.sat.f16.u32 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.F16.U32.RZ R9, R2 ;                  /* 0x0000000200097306 */
        // /* 0x004e28000020c800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_f16_u32(unsigned int a) {
    uint16_t out;
    asm volatile("cvt.rz.f16.u32 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                 /* 0x0000000207047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   I2F.F16.U64 R7, R2 ;                        /* 0x0000000200077312 */
        // /* 0x004e260000300800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;            /* 0x0000000704007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_f16_u64(unsigned long long a) {
    uint16_t out;
    asm volatile("cvt.rn.f16.u64 %0, %1;" : "=h"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                 /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.F16.U64 R0, R2 ;                        /* 0x0000000200007312 */
        // /* 0x004e260000300800 */
        // /*00c0*/                   HADD2.SAT R0, R0.H0_H0, -RZ.H0_H0 ;         /* 0xa00000ff00007230 */
        // /* 0x001fca0000002800 */
        // /*00d0*/                   STG.E.U16 desc[UR4][R4.64], R0 ;            /* 0x0000000004007986 */
        // /* 0x000fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_sat_f16_u64(unsigned long long a) {
    uint16_t out;
    asm volatile("cvt.rn.sat.f16.u64 %0, %1;" : "=h"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;                 /* 0x0000000207047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   I2F.F16.U64.RZ R7, R2 ;                     /* 0x0000000200077312 */
        // /* 0x004e26000030c800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R7 ;            /* 0x0000000704007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_f16_u64(unsigned long long a) {
    uint16_t out;
    asm volatile("cvt.rz.f16.u64 %0, %1;" : "=h"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.F16.U8 R9, R2 ;                      /* 0x0000000200097306 */
        // /* 0x004e280000000800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rn_f16_u8(unsigned int a) {
    uint16_t out;
    asm volatile("cvt.rn.f16.u8 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x2, R4 ;              /* 0x0000000207047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.F16.U8.RZ R9, R2 ;                   /* 0x0000000200097306 */
        // /* 0x004e28000000c800 */
        // /*00c0*/                   STG.E.U16 desc[UR4][R4.64], R9 ;         /* 0x0000000904007986 */
        // /* 0x001fe2000c101504 */
__device__ __forceinline__ uint16_t cvt_rz_f16_u8(unsigned int a) {
    uint16_t out;
    asm volatile("cvt.rz.f16.u8 %0, %1;" : "=h"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   SHF.L.U32 R7, R2, 0x10, RZ ;                 /* 0x0000001002077819 */
        // /* 0x004fca00000006ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                 /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rm_f32_bf16(uint16_t a) {
    float out;
    asm volatile("cvt.rm.f32.bf16 %0, %1;" : "=f"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   SHF.L.U32 R7, R2, 0x10, RZ ;                 /* 0x0000001002077819 */
        // /* 0x004fca00000006ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                 /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rn_f32_bf16(uint16_t a) {
    float out;
    asm volatile("cvt.rn.f32.bf16 %0, %1;" : "=f"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.FTZ.F32.BF16 R9, R2 ;                    /* 0x0000000200097304 */
        // /* 0x004e280000411000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rn_ftz_f32_bf16(uint16_t a) {
    float out;
    asm volatile("cvt.rn.ftz.f32.bf16 %0, %1;" : "=f"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   SHF.L.U32 R7, R2, 0x10, RZ ;                 /* 0x0000001002077819 */
        // /* 0x004fca00000006ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                 /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rp_f32_bf16(uint16_t a) {
    float out;
    asm volatile("cvt.rp.f32.bf16 %0, %1;" : "=f"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   SHF.L.U32 R7, R2, 0x10, RZ ;                 /* 0x0000001002077819 */
        // /* 0x004fca00000006ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                 /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rz_f32_bf16(uint16_t a) {
    float out;
    asm volatile("cvt.rz.f32.bf16 %0, %1;" : "=f"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.FTZ.F32.BF16.RZ R9, R2 ;                 /* 0x0000000200097304 */
        // /* 0x004e28000041d000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rz_ftz_f32_bf16(uint16_t a) {
    float out;
    asm volatile("cvt.rz.ftz.f32.bf16 %0, %1;" : "=f"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.FLOOR R9, R2 ;                      /* 0x0000000200097307 */
        // /* 0x004e280000205000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rmi_f32_f32(float a) {
    float out;
    asm volatile("cvt.rmi.f32.f32 %0, %1;" : "=f"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.FTZ.FLOOR R9, R2 ;                  /* 0x0000000200097307 */
        // /* 0x004e280000215000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rmi_ftz_f32_f32(float a) {
    float out;
    asm volatile("cvt.rmi.ftz.f32.f32 %0, %1;" : "=f"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.FLOOR R0, R2 ;                      /* 0x0000000200007307 */
        // /* 0x004e260000205000 */
        // /*00c0*/                   FADD.SAT R7, RZ, R0 ;                    /* 0x00000000ff077221 */
        // /* 0x001fca0000002000 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rmi_sat_f32_f32(float a) {
    float out;
    asm volatile("cvt.rmi.sat.f32.f32 %0, %1;" : "=f"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND R9, R2 ;                            /* 0x0000000200097307 */
        // /* 0x004e280000201000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rni_f32_f32(float a) {
    float out;
    asm volatile("cvt.rni.f32.f32 %0, %1;" : "=f"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.FTZ R9, R2 ;                        /* 0x0000000200097307 */
        // /* 0x004e280000211000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rni_ftz_f32_f32(float a) {
    float out;
    asm volatile("cvt.rni.ftz.f32.f32 %0, %1;" : "=f"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND R0, R2 ;                            /* 0x0000000200007307 */
        // /* 0x004e260000201000 */
        // /*00c0*/                   FADD.SAT R7, RZ, R0 ;                    /* 0x00000000ff077221 */
        // /* 0x001fca0000002000 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rni_sat_f32_f32(float a) {
    float out;
    asm volatile("cvt.rni.sat.f32.f32 %0, %1;" : "=f"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.CEIL R9, R2 ;                       /* 0x0000000200097307 */
        // /* 0x004e280000209000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rpi_f32_f32(float a) {
    float out;
    asm volatile("cvt.rpi.f32.f32 %0, %1;" : "=f"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.FTZ.CEIL R9, R2 ;                   /* 0x0000000200097307 */
        // /* 0x004e280000219000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rpi_ftz_f32_f32(float a) {
    float out;
    asm volatile("cvt.rpi.ftz.f32.f32 %0, %1;" : "=f"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.CEIL R0, R2 ;                       /* 0x0000000200007307 */
        // /* 0x004e260000209000 */
        // /*00c0*/                   FADD.SAT R7, RZ, R0 ;                    /* 0x00000000ff077221 */
        // /* 0x001fca0000002000 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rpi_sat_f32_f32(float a) {
    float out;
    asm volatile("cvt.rpi.sat.f32.f32 %0, %1;" : "=f"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.TRUNC R9, R2 ;                      /* 0x0000000200097307 */
        // /* 0x004e28000020d000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rzi_f32_f32(float a) {
    float out;
    asm volatile("cvt.rzi.f32.f32 %0, %1;" : "=f"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.FTZ.TRUNC R9, R2 ;                  /* 0x0000000200097307 */
        // /* 0x004e28000021d000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rzi_ftz_f32_f32(float a) {
    float out;
    asm volatile("cvt.rzi.ftz.f32.f32 %0, %1;" : "=f"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   FRND.TRUNC R0, R2 ;                      /* 0x0000000200007307 */
        // /* 0x004e26000020d000 */
        // /*00c0*/                   FADD.SAT R7, RZ, R0 ;                    /* 0x00000000ff077221 */
        // /* 0x001fca0000002000 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rzi_sat_f32_f32(float a) {
    float out;
    asm volatile("cvt.rzi.sat.f32.f32 %0, %1;" : "=f"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   F2F.F32.F64.RM R7, R2 ;                     /* 0x0000000200077310 */
        // /* 0x004e260000305000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rm_f32_f64(double a) {
    float out;
    asm volatile("cvt.rm.f32.f64 %0, %1;" : "=f"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   F2F.F32.F64 R7, R2 ;                        /* 0x0000000200077310 */
        // /* 0x004e260000301000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rn_f32_f64(double a) {
    float out;
    asm volatile("cvt.rn.f32.f64 %0, %1;" : "=f"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;    /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   UMOV UR6, 0xf0000000 ;                        /* 0xf000000000067882 */
        // /* 0x000fe20000000000 */
        // /*00b0*/                   UMOV UR7, 0x380fffff ;                        /* 0x380fffff00077882 */
        // /* 0x000fe20000000000 */
        // /*00c0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                   /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00d0*/                   DSETP.GEU.AND P0, PT, |R2|, UR6, PT ;         /* 0x0000000602007e2a */
        // /* 0x004e1e000bf0e200 */
        // /*00e0*/                   NOP ;                                         /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*00f0*/                   NOP ;                                         /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*0100*/                   NOP ;                                         /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*0110*/                   NOP ;                                         /* 0x0000000000007918 */
        // /* 0x000fc80000000000 */
        // /*0120*/                   F2F.F32.F64 R9, R2 ;                          /* 0x0000000200097310 */
        // /* 0x000e240000301000 */
        // /*0130*/              @!P0 FMUL R9, R9, 1.175494350822287508e-38 ;       /* 0x0080000009098820 */
        // /* 0x001fca0000400000 */
        // /*0140*/                   STG.E desc[UR4][R4.64], R9 ;                  /* 0x0000000904007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rn_ftz_f32_f64(double a) {
    float out;
    asm volatile("cvt.rn.ftz.f32.f64 %0, %1;" : "=f"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.F32.F64 R0, R2 ;                        /* 0x0000000200007310 */
        // /* 0x004e260000301000 */
        // /*00c0*/                   FADD.SAT R7, RZ, R0 ;                       /* 0x00000000ff077221 */
        // /* 0x001fca0000002000 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rn_sat_f32_f64(double a) {
    float out;
    asm volatile("cvt.rn.sat.f32.f64 %0, %1;" : "=f"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   F2F.F32.F64.RP R7, R2 ;                     /* 0x0000000200077310 */
        // /* 0x004e260000309000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rp_f32_f64(double a) {
    float out;
    asm volatile("cvt.rp.f32.f64 %0, %1;" : "=f"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   F2F.F32.F64.RZ R7, R2 ;                     /* 0x0000000200077310 */
        // /* 0x004e26000030d000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rz_f32_f64(double a) {
    float out;
    asm volatile("cvt.rz.f32.f64 %0, %1;" : "=f"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;                  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.F32.F64.RZ R9, R2 ;                                     /* 0x0000000200097310 */
        // /* 0x004e24000030d000 */
        // /*00c0*/                   FSETP.GEU.AND P0, PT, |R9|, 1.175494350822287508e-38, PT ;  /* 0x008000000900780b */
        // /* 0x001fda0003f0e200 */
        // /*00d0*/              @!P0 FMUL R9, R9, 1.175494350822287508e-38 ;                     /* 0x0080000009098820 */
        // /* 0x000fca0000400000 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R9 ;                                /* 0x0000000904007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rz_ftz_f32_f64(double a) {
    float out;
    asm volatile("cvt.rz.ftz.f32.f64 %0, %1;" : "=f"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2F.F32.F64.RZ R0, R2 ;                     /* 0x0000000200007310 */
        // /* 0x004e26000030d000 */
        // /*00c0*/                   FADD.SAT R7, RZ, R0 ;                       /* 0x00000000ff077221 */
        // /* 0x001fca0000002000 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rz_sat_f32_f64(double a) {
    float out;
    asm volatile("cvt.rz.sat.f32.f64 %0, %1;" : "=f"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.S16.RM R9, R2 ;                      /* 0x0000000200097306 */
        // /* 0x004e280000105400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rm_f32_s16(int a) {
    float out;
    asm volatile("cvt.rm.f32.s16 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.S16 R9, R2 ;                         /* 0x0000000200097306 */
        // /* 0x004e280000101400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rn_f32_s16(int a) {
    float out;
    asm volatile("cvt.rn.f32.s16 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.S16 R9, R2 ;                         /* 0x0000000200097306 */
        // /* 0x004e280000101400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rn_ftz_f32_s16(int a) {
    float out;
    asm volatile("cvt.rn.ftz.f32.s16 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.S16.RP R9, R2 ;                      /* 0x0000000200097306 */
        // /* 0x004e280000109400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rp_f32_s16(int a) {
    float out;
    asm volatile("cvt.rp.f32.s16 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.S16.RZ R9, R2 ;                      /* 0x0000000200097306 */
        // /* 0x004e28000010d400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rz_f32_s16(int a) {
    float out;
    asm volatile("cvt.rz.f32.s16 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.S16.RZ R9, R2 ;                      /* 0x0000000200097306 */
        // /* 0x004e28000010d400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rz_ftz_f32_s16(int a) {
    float out;
    asm volatile("cvt.rz.ftz.f32.s16 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.RM R9, R2 ;                          /* 0x0000000200097306 */
        // /* 0x004e280000205400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rm_f32_s32(int a) {
    float out;
    asm volatile("cvt.rm.f32.s32 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2FP.F32.S32 R7, R2 ;                    /* 0x0000000200077245 */
        // /* 0x004fca0000201400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rn_f32_s32(int a) {
    float out;
    asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2FP.F32.S32 R7, R2 ;                    /* 0x0000000200077245 */
        // /* 0x004fca0000201400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rn_ftz_f32_s32(int a) {
    float out;
    asm volatile("cvt.rn.ftz.f32.s32 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2FP.F32.S32 R0, R2 ;                    /* 0x0000000200007245 */
        // /* 0x004fca0000201400 */
        // /*00c0*/                   FADD.SAT R7, RZ, R0 ;                    /* 0x00000000ff077221 */
        // /* 0x000fca0000002000 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rn_ftz_sat_f32_s32(int a) {
    float out;
    asm volatile("cvt.rn.ftz.sat.f32.s32 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2FP.F32.S32 R0, R2 ;                    /* 0x0000000200007245 */
        // /* 0x004fca0000201400 */
        // /*00c0*/                   FADD.SAT R7, RZ, R0 ;                    /* 0x00000000ff077221 */
        // /* 0x000fca0000002000 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rn_sat_f32_s32(int a) {
    float out;
    asm volatile("cvt.rn.sat.f32.s32 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.RP R9, R2 ;                          /* 0x0000000200097306 */
        // /* 0x004e280000209400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rp_f32_s32(int a) {
    float out;
    asm volatile("cvt.rp.f32.s32 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2FP.F32.S32.RZ R7, R2 ;                 /* 0x0000000200077245 */
        // /* 0x004fca000020d400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rz_f32_s32(int a) {
    float out;
    asm volatile("cvt.rz.f32.s32 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2FP.F32.S32.RZ R7, R2 ;                 /* 0x0000000200077245 */
        // /* 0x004fca000020d400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rz_ftz_f32_s32(int a) {
    float out;
    asm volatile("cvt.rz.ftz.f32.s32 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   I2F.S64.RM R7, R2 ;                         /* 0x0000000200077312 */
        // /* 0x004e260000305400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rm_f32_s64(long long a) {
    float out;
    asm volatile("cvt.rm.f32.s64 %0, %1;" : "=f"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   I2F.S64 R7, R2 ;                            /* 0x0000000200077312 */
        // /* 0x004e260000301400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rn_f32_s64(long long a) {
    float out;
    asm volatile("cvt.rn.f32.s64 %0, %1;" : "=f"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   I2F.S64 R7, R2 ;                            /* 0x0000000200077312 */
        // /* 0x004e260000301400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rn_ftz_f32_s64(long long a) {
    float out;
    asm volatile("cvt.rn.ftz.f32.s64 %0, %1;" : "=f"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.S64 R0, R2 ;                            /* 0x0000000200007312 */
        // /* 0x004e260000301400 */
        // /*00c0*/                   FADD.SAT R7, RZ, R0 ;                       /* 0x00000000ff077221 */
        // /* 0x001fca0000002000 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rn_ftz_sat_f32_s64(long long a) {
    float out;
    asm volatile("cvt.rn.ftz.sat.f32.s64 %0, %1;" : "=f"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.S64 R0, R2 ;                            /* 0x0000000200007312 */
        // /* 0x004e260000301400 */
        // /*00c0*/                   FADD.SAT R7, RZ, R0 ;                       /* 0x00000000ff077221 */
        // /* 0x001fca0000002000 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rn_sat_f32_s64(long long a) {
    float out;
    asm volatile("cvt.rn.sat.f32.s64 %0, %1;" : "=f"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   I2F.S64.RP R7, R2 ;                         /* 0x0000000200077312 */
        // /* 0x004e260000309400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rp_f32_s64(long long a) {
    float out;
    asm volatile("cvt.rp.f32.s64 %0, %1;" : "=f"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   I2F.S64.RZ R7, R2 ;                         /* 0x0000000200077312 */
        // /* 0x004e26000030d400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rz_f32_s64(long long a) {
    float out;
    asm volatile("cvt.rz.f32.s64 %0, %1;" : "=f"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   I2F.S64.RZ R7, R2 ;                         /* 0x0000000200077312 */
        // /* 0x004e26000030d400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rz_ftz_f32_s64(long long a) {
    float out;
    asm volatile("cvt.rz.ftz.f32.s64 %0, %1;" : "=f"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.S8.RM R9, R2 ;                       /* 0x0000000200097306 */
        // /* 0x004e280000005400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rm_f32_s8(int a) {
    float out;
    asm volatile("cvt.rm.f32.s8 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.S8 R9, R2 ;                          /* 0x0000000200097306 */
        // /* 0x004e280000001400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rn_f32_s8(int a) {
    float out;
    asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.S8 R9, R2 ;                          /* 0x0000000200097306 */
        // /* 0x004e280000001400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rn_ftz_f32_s8(int a) {
    float out;
    asm volatile("cvt.rn.ftz.f32.s8 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.S8.RP R9, R2 ;                       /* 0x0000000200097306 */
        // /* 0x004e280000009400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rp_f32_s8(int a) {
    float out;
    asm volatile("cvt.rp.f32.s8 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.S8.RZ R9, R2 ;                       /* 0x0000000200097306 */
        // /* 0x004e28000000d400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rz_f32_s8(int a) {
    float out;
    asm volatile("cvt.rz.f32.s8 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.S8.RZ R9, R2 ;                       /* 0x0000000200097306 */
        // /* 0x004e28000000d400 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rz_ftz_f32_s8(int a) {
    float out;
    asm volatile("cvt.rz.ftz.f32.s8 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.U16.RM R9, R2 ;                      /* 0x0000000200097306 */
        // /* 0x004e280000105000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rm_f32_u16(unsigned int a) {
    float out;
    asm volatile("cvt.rm.f32.u16 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.U16 R9, R2 ;                         /* 0x0000000200097306 */
        // /* 0x004e280000101000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rn_f32_u16(unsigned int a) {
    float out;
    asm volatile("cvt.rn.f32.u16 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.U16 R9, R2 ;                         /* 0x0000000200097306 */
        // /* 0x004e280000101000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rn_ftz_f32_u16(unsigned int a) {
    float out;
    asm volatile("cvt.rn.ftz.f32.u16 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.U16.RP R9, R2 ;                      /* 0x0000000200097306 */
        // /* 0x004e280000109000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rp_f32_u16(unsigned int a) {
    float out;
    asm volatile("cvt.rp.f32.u16 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.U16.RZ R9, R2 ;                      /* 0x0000000200097306 */
        // /* 0x004e28000010d000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rz_f32_u16(unsigned int a) {
    float out;
    asm volatile("cvt.rz.f32.u16 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.U16.RZ R9, R2 ;                      /* 0x0000000200097306 */
        // /* 0x004e28000010d000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rz_ftz_f32_u16(unsigned int a) {
    float out;
    asm volatile("cvt.rz.ftz.f32.u16 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.U32.RM R9, R2 ;                      /* 0x0000000200097306 */
        // /* 0x004e280000205000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rm_f32_u32(unsigned int a) {
    float out;
    asm volatile("cvt.rm.f32.u32 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2FP.F32.U32 R7, R2 ;                    /* 0x0000000200077245 */
        // /* 0x004fca0000201000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rn_f32_u32(unsigned int a) {
    float out;
    asm volatile("cvt.rn.f32.u32 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2FP.F32.U32 R7, R2 ;                    /* 0x0000000200077245 */
        // /* 0x004fca0000201000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rn_ftz_f32_u32(unsigned int a) {
    float out;
    asm volatile("cvt.rn.ftz.f32.u32 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2FP.F32.U32 R0, R2 ;                    /* 0x0000000200007245 */
        // /* 0x004fca0000201000 */
        // /*00c0*/                   FADD.SAT R7, RZ, R0 ;                    /* 0x00000000ff077221 */
        // /* 0x000fca0000002000 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rn_ftz_sat_f32_u32(unsigned int a) {
    float out;
    asm volatile("cvt.rn.ftz.sat.f32.u32 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2FP.F32.U32 R0, R2 ;                    /* 0x0000000200007245 */
        // /* 0x004fca0000201000 */
        // /*00c0*/                   FADD.SAT R7, RZ, R0 ;                    /* 0x00000000ff077221 */
        // /* 0x000fca0000002000 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rn_sat_f32_u32(unsigned int a) {
    float out;
    asm volatile("cvt.rn.sat.f32.u32 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.U32.RP R9, R2 ;                      /* 0x0000000200097306 */
        // /* 0x004e280000209000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rp_f32_u32(unsigned int a) {
    float out;
    asm volatile("cvt.rp.f32.u32 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2FP.F32.U32.RZ R7, R2 ;                 /* 0x0000000200077245 */
        // /* 0x004fca000020d000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rz_f32_u32(unsigned int a) {
    float out;
    asm volatile("cvt.rz.f32.u32 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2FP.F32.U32.RZ R7, R2 ;                 /* 0x0000000200077245 */
        // /* 0x004fca000020d000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rz_ftz_f32_u32(unsigned int a) {
    float out;
    asm volatile("cvt.rz.ftz.f32.u32 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   I2F.U64.RM R7, R2 ;                         /* 0x0000000200077312 */
        // /* 0x004e260000305000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rm_f32_u64(unsigned long long a) {
    float out;
    asm volatile("cvt.rm.f32.u64 %0, %1;" : "=f"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   I2F.U64 R7, R2 ;                            /* 0x0000000200077312 */
        // /* 0x004e260000301000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rn_f32_u64(unsigned long long a) {
    float out;
    asm volatile("cvt.rn.f32.u64 %0, %1;" : "=f"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   I2F.U64 R7, R2 ;                            /* 0x0000000200077312 */
        // /* 0x004e260000301000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rn_ftz_f32_u64(unsigned long long a) {
    float out;
    asm volatile("cvt.rn.ftz.f32.u64 %0, %1;" : "=f"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.U64 R0, R2 ;                            /* 0x0000000200007312 */
        // /* 0x004e260000301000 */
        // /*00c0*/                   FADD.SAT R7, RZ, R0 ;                       /* 0x00000000ff077221 */
        // /* 0x001fca0000002000 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rn_ftz_sat_f32_u64(unsigned long long a) {
    float out;
    asm volatile("cvt.rn.ftz.sat.f32.u64 %0, %1;" : "=f"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.U64 R0, R2 ;                            /* 0x0000000200007312 */
        // /* 0x004e260000301000 */
        // /*00c0*/                   FADD.SAT R7, RZ, R0 ;                       /* 0x00000000ff077221 */
        // /* 0x001fca0000002000 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float cvt_rn_sat_f32_u64(unsigned long long a) {
    float out;
    asm volatile("cvt.rn.sat.f32.u64 %0, %1;" : "=f"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   I2F.U64.RP R7, R2 ;                         /* 0x0000000200077312 */
        // /* 0x004e260000309000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rp_f32_u64(unsigned long long a) {
    float out;
    asm volatile("cvt.rp.f32.u64 %0, %1;" : "=f"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   I2F.U64.RZ R7, R2 ;                         /* 0x0000000200077312 */
        // /* 0x004e26000030d000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rz_f32_u64(unsigned long long a) {
    float out;
    asm volatile("cvt.rz.f32.u64 %0, %1;" : "=f"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   I2F.U64.RZ R7, R2 ;                         /* 0x0000000200077312 */
        // /* 0x004e26000030d000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rz_ftz_f32_u64(unsigned long long a) {
    float out;
    asm volatile("cvt.rz.ftz.f32.u64 %0, %1;" : "=f"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.U8.RM R9, R2 ;                       /* 0x0000000200097306 */
        // /* 0x004e280000005000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rm_f32_u8(unsigned int a) {
    float out;
    asm volatile("cvt.rm.f32.u8 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.U8 R9, R2 ;                          /* 0x0000000200097306 */
        // /* 0x004e280000001000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rn_f32_u8(unsigned int a) {
    float out;
    asm volatile("cvt.rn.f32.u8 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.U8 R9, R2 ;                          /* 0x0000000200097306 */
        // /* 0x004e280000001000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rn_ftz_f32_u8(unsigned int a) {
    float out;
    asm volatile("cvt.rn.ftz.f32.u8 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.U8.RP R9, R2 ;                       /* 0x0000000200097306 */
        // /* 0x004e280000009000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rp_f32_u8(unsigned int a) {
    float out;
    asm volatile("cvt.rp.f32.u8 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.U8.RZ R9, R2 ;                       /* 0x0000000200097306 */
        // /* 0x004e28000000d000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rz_f32_u8(unsigned int a) {
    float out;
    asm volatile("cvt.rz.f32.u8 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2F.U8.RZ R9, R2 ;                       /* 0x0000000200097306 */
        // /* 0x004e28000000d000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ float cvt_rz_ftz_f32_u8(unsigned int a) {
    float out;
    asm volatile("cvt.rz.ftz.f32.u8 %0, %1;" : "=f"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                  /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2F.F64.BF16.RM R4, R2 ;                     /* 0x0000000200047310 */
        // /* 0x004e260000405800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;              /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rm_f64_bf16(uint16_t a) {
    double out;
    asm volatile("cvt.rm.f64.bf16 %0, %1;" : "=d"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                  /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2F.F64.BF16 R4, R2 ;                        /* 0x0000000200047310 */
        // /* 0x004e260000401800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;              /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rn_f64_bf16(uint16_t a) {
    double out;
    asm volatile("cvt.rn.f64.bf16 %0, %1;" : "=d"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                  /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2F.F64.BF16.RP R4, R2 ;                     /* 0x0000000200047310 */
        // /* 0x004e260000409800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;              /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rp_f64_bf16(uint16_t a) {
    double out;
    asm volatile("cvt.rp.f64.bf16 %0, %1;" : "=d"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                  /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2F.F64.BF16.RZ R4, R2 ;                     /* 0x0000000200047310 */
        // /* 0x004e26000040d800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;              /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rz_f64_bf16(uint16_t a) {
    double out;
    asm volatile("cvt.rz.f64.bf16 %0, %1;" : "=d"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                 /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   FRND.F64.FLOOR R4, R2 ;                     /* 0x0000000200047313 */
        // /* 0x004e260000305800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;             /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rmi_f64_f64(double a) {
    double out;
    asm volatile("cvt.rmi.f64.f64 %0, %1;" : "=d"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;        /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.MOV.U32 R0, RZ, RZ, RZ ;                     /* 0x000000ffff007224 */
        // /* 0x000fe400078e00ff */
        // /*00b0*/                   HFMA2 R8, -RZ, RZ, 4.76837158203125e-07, 0 ;      /* 0x00080000ff087431 */
        // /* 0x000fe200000001ff */
        // /*00c0*/                   FRND.F64.FLOOR R4, R2 ;                           /* 0x0000000200047313 */
        // /* 0x004e240000305800 */
        // /*00d0*/                   IMAD.MOV.U32 R13, RZ, RZ, R5 ;                    /* 0x000000ffff0d7224 */
        // /* 0x001fde00078e0005 */
        // /*00e0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*00f0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*0100*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*0110*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fc40000000000 */
        // /*0120*/                   DSETP.MAX.AND P0, P1, RZ, R4, PT ;                /* 0x00000004ff00722a */
        // /* 0x000062000390f000 */
        // /*0130*/                   MOV R11, R4 ;                                     /* 0x00000004000b7202 */
        // /* 0x000fe20000000f00 */
        // /*0140*/                   IMAD.MOV.U32 R4, RZ, RZ, RZ ;                     /* 0x000000ffff047224 */
        // /* 0x001fe200078e00ff */
        // /*0150*/                   FSEL R15, R0, R13, P0 ;                           /* 0x0000000d000f7208 */
        // /* 0x002fe40000000000 */
        // /*0160*/               @P1 LOP3.LUT R15, R13, 0x80000, RZ, 0xfc, !PT ;       /* 0x000800000d0f1812 */
        // /* 0x000fe400078efcff */
        // /*0170*/                   SEL R4, R4, R11, P0 ;                             /* 0x0000000b04047207 */
        // /* 0x000fc60000000000 */
        // /*0180*/                   IMAD.MOV.U32 R5, RZ, RZ, R15 ;                    /* 0x000000ffff057224 */
        // /* 0x000fca00078e000f */
        // /*0190*/                   MOV R0, R5 ;                                      /* 0x0000000500007202 */
        // /* 0x000fde0000000f00 */
        // /*01a0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*01b0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*01c0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fc80000000000 */
        // /*01d0*/                   DSETP.MIN.AND P0, P1, R4, 1, PT ;                 /* 0x3ff000000400742a */
        // /* 0x000e240003900000 */
        // /*01e0*/                   FSEL R3, R0, 1.875, P0 ;                          /* 0x3ff0000000037808 */
        // /* 0x001fe40000000000 */
        // /*01f0*/                   SEL R2, R4, RZ, P0 ;                              /* 0x000000ff04027207 */
        // /* 0x000fe20000000000 */
        // /*0200*/                   IMAD.WIDE R4, R9, 0x8, R6 ;                       /* 0x0000000809047825 */
        // /* 0x008fe200078e0206 */
        // /*0210*/               @P1 LOP3.LUT R3, R8, 0x3ff00000, RZ, 0xfc, !PT ;      /* 0x3ff0000008031812 */
        // /* 0x000fca00078efcff */
        // /*0220*/                   STG.E.64 desc[UR4][R4.64], R2 ;                   /* 0x0000000204007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ double cvt_rmi_sat_f64_f64(double a) {
    double out;
    asm volatile("cvt.rmi.sat.f64.f64 %0, %1;" : "=d"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                 /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   FRND.F64 R4, R2 ;                           /* 0x0000000200047313 */
        // /* 0x004e260000301800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;             /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rni_f64_f64(double a) {
    double out;
    asm volatile("cvt.rni.f64.f64 %0, %1;" : "=d"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;        /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.MOV.U32 R0, RZ, RZ, RZ ;                     /* 0x000000ffff007224 */
        // /* 0x000fe400078e00ff */
        // /*00b0*/                   HFMA2 R8, -RZ, RZ, 4.76837158203125e-07, 0 ;      /* 0x00080000ff087431 */
        // /* 0x000fe200000001ff */
        // /*00c0*/                   FRND.F64 R4, R2 ;                                 /* 0x0000000200047313 */
        // /* 0x004e240000301800 */
        // /*00d0*/                   IMAD.MOV.U32 R13, RZ, RZ, R5 ;                    /* 0x000000ffff0d7224 */
        // /* 0x001fde00078e0005 */
        // /*00e0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*00f0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*0100*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*0110*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fc40000000000 */
        // /*0120*/                   DSETP.MAX.AND P0, P1, RZ, R4, PT ;                /* 0x00000004ff00722a */
        // /* 0x000062000390f000 */
        // /*0130*/                   MOV R11, R4 ;                                     /* 0x00000004000b7202 */
        // /* 0x000fe20000000f00 */
        // /*0140*/                   IMAD.MOV.U32 R4, RZ, RZ, RZ ;                     /* 0x000000ffff047224 */
        // /* 0x001fe200078e00ff */
        // /*0150*/                   FSEL R15, R0, R13, P0 ;                           /* 0x0000000d000f7208 */
        // /* 0x002fe40000000000 */
        // /*0160*/               @P1 LOP3.LUT R15, R13, 0x80000, RZ, 0xfc, !PT ;       /* 0x000800000d0f1812 */
        // /* 0x000fe400078efcff */
        // /*0170*/                   SEL R4, R4, R11, P0 ;                             /* 0x0000000b04047207 */
        // /* 0x000fc60000000000 */
        // /*0180*/                   IMAD.MOV.U32 R5, RZ, RZ, R15 ;                    /* 0x000000ffff057224 */
        // /* 0x000fca00078e000f */
        // /*0190*/                   MOV R0, R5 ;                                      /* 0x0000000500007202 */
        // /* 0x000fde0000000f00 */
        // /*01a0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*01b0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*01c0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fc80000000000 */
        // /*01d0*/                   DSETP.MIN.AND P0, P1, R4, 1, PT ;                 /* 0x3ff000000400742a */
        // /* 0x000e240003900000 */
        // /*01e0*/                   FSEL R3, R0, 1.875, P0 ;                          /* 0x3ff0000000037808 */
        // /* 0x001fe40000000000 */
        // /*01f0*/                   SEL R2, R4, RZ, P0 ;                              /* 0x000000ff04027207 */
        // /* 0x000fe20000000000 */
        // /*0200*/                   IMAD.WIDE R4, R9, 0x8, R6 ;                       /* 0x0000000809047825 */
        // /* 0x008fe200078e0206 */
        // /*0210*/               @P1 LOP3.LUT R3, R8, 0x3ff00000, RZ, 0xfc, !PT ;      /* 0x3ff0000008031812 */
        // /* 0x000fca00078efcff */
        // /*0220*/                   STG.E.64 desc[UR4][R4.64], R2 ;                   /* 0x0000000204007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ double cvt_rni_sat_f64_f64(double a) {
    double out;
    asm volatile("cvt.rni.sat.f64.f64 %0, %1;" : "=d"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                 /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   FRND.F64.CEIL R4, R2 ;                      /* 0x0000000200047313 */
        // /* 0x004e260000309800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;             /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rpi_f64_f64(double a) {
    double out;
    asm volatile("cvt.rpi.f64.f64 %0, %1;" : "=d"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;        /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.MOV.U32 R0, RZ, RZ, RZ ;                     /* 0x000000ffff007224 */
        // /* 0x000fe400078e00ff */
        // /*00b0*/                   HFMA2 R8, -RZ, RZ, 4.76837158203125e-07, 0 ;      /* 0x00080000ff087431 */
        // /* 0x000fe200000001ff */
        // /*00c0*/                   FRND.F64.CEIL R4, R2 ;                            /* 0x0000000200047313 */
        // /* 0x004e240000309800 */
        // /*00d0*/                   IMAD.MOV.U32 R13, RZ, RZ, R5 ;                    /* 0x000000ffff0d7224 */
        // /* 0x001fde00078e0005 */
        // /*00e0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*00f0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*0100*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*0110*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fc40000000000 */
        // /*0120*/                   DSETP.MAX.AND P0, P1, RZ, R4, PT ;                /* 0x00000004ff00722a */
        // /* 0x000062000390f000 */
        // /*0130*/                   MOV R11, R4 ;                                     /* 0x00000004000b7202 */
        // /* 0x000fe20000000f00 */
        // /*0140*/                   IMAD.MOV.U32 R4, RZ, RZ, RZ ;                     /* 0x000000ffff047224 */
        // /* 0x001fe200078e00ff */
        // /*0150*/                   FSEL R15, R0, R13, P0 ;                           /* 0x0000000d000f7208 */
        // /* 0x002fe40000000000 */
        // /*0160*/               @P1 LOP3.LUT R15, R13, 0x80000, RZ, 0xfc, !PT ;       /* 0x000800000d0f1812 */
        // /* 0x000fe400078efcff */
        // /*0170*/                   SEL R4, R4, R11, P0 ;                             /* 0x0000000b04047207 */
        // /* 0x000fc60000000000 */
        // /*0180*/                   IMAD.MOV.U32 R5, RZ, RZ, R15 ;                    /* 0x000000ffff057224 */
        // /* 0x000fca00078e000f */
        // /*0190*/                   MOV R0, R5 ;                                      /* 0x0000000500007202 */
        // /* 0x000fde0000000f00 */
        // /*01a0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*01b0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*01c0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fc80000000000 */
        // /*01d0*/                   DSETP.MIN.AND P0, P1, R4, 1, PT ;                 /* 0x3ff000000400742a */
        // /* 0x000e240003900000 */
        // /*01e0*/                   FSEL R3, R0, 1.875, P0 ;                          /* 0x3ff0000000037808 */
        // /* 0x001fe40000000000 */
        // /*01f0*/                   SEL R2, R4, RZ, P0 ;                              /* 0x000000ff04027207 */
        // /* 0x000fe20000000000 */
        // /*0200*/                   IMAD.WIDE R4, R9, 0x8, R6 ;                       /* 0x0000000809047825 */
        // /* 0x008fe200078e0206 */
        // /*0210*/               @P1 LOP3.LUT R3, R8, 0x3ff00000, RZ, 0xfc, !PT ;      /* 0x3ff0000008031812 */
        // /* 0x000fca00078efcff */
        // /*0220*/                   STG.E.64 desc[UR4][R4.64], R2 ;                   /* 0x0000000204007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ double cvt_rpi_sat_f64_f64(double a) {
    double out;
    asm volatile("cvt.rpi.sat.f64.f64 %0, %1;" : "=d"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                 /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   FRND.F64.TRUNC R4, R2 ;                     /* 0x0000000200047313 */
        // /* 0x004e26000030d800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;             /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rzi_f64_f64(double a) {
    double out;
    asm volatile("cvt.rzi.f64.f64 %0, %1;" : "=d"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;        /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.MOV.U32 R0, RZ, RZ, RZ ;                     /* 0x000000ffff007224 */
        // /* 0x000fe400078e00ff */
        // /*00b0*/                   HFMA2 R8, -RZ, RZ, 4.76837158203125e-07, 0 ;      /* 0x00080000ff087431 */
        // /* 0x000fe200000001ff */
        // /*00c0*/                   FRND.F64.TRUNC R4, R2 ;                           /* 0x0000000200047313 */
        // /* 0x004e24000030d800 */
        // /*00d0*/                   IMAD.MOV.U32 R13, RZ, RZ, R5 ;                    /* 0x000000ffff0d7224 */
        // /* 0x001fde00078e0005 */
        // /*00e0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*00f0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*0100*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*0110*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fc40000000000 */
        // /*0120*/                   DSETP.MAX.AND P0, P1, RZ, R4, PT ;                /* 0x00000004ff00722a */
        // /* 0x000062000390f000 */
        // /*0130*/                   MOV R11, R4 ;                                     /* 0x00000004000b7202 */
        // /* 0x000fe20000000f00 */
        // /*0140*/                   IMAD.MOV.U32 R4, RZ, RZ, RZ ;                     /* 0x000000ffff047224 */
        // /* 0x001fe200078e00ff */
        // /*0150*/                   FSEL R15, R0, R13, P0 ;                           /* 0x0000000d000f7208 */
        // /* 0x002fe40000000000 */
        // /*0160*/               @P1 LOP3.LUT R15, R13, 0x80000, RZ, 0xfc, !PT ;       /* 0x000800000d0f1812 */
        // /* 0x000fe400078efcff */
        // /*0170*/                   SEL R4, R4, R11, P0 ;                             /* 0x0000000b04047207 */
        // /* 0x000fc60000000000 */
        // /*0180*/                   IMAD.MOV.U32 R5, RZ, RZ, R15 ;                    /* 0x000000ffff057224 */
        // /* 0x000fca00078e000f */
        // /*0190*/                   MOV R0, R5 ;                                      /* 0x0000000500007202 */
        // /* 0x000fde0000000f00 */
        // /*01a0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*01b0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*01c0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fc80000000000 */
        // /*01d0*/                   DSETP.MIN.AND P0, P1, R4, 1, PT ;                 /* 0x3ff000000400742a */
        // /* 0x000e240003900000 */
        // /*01e0*/                   FSEL R3, R0, 1.875, P0 ;                          /* 0x3ff0000000037808 */
        // /* 0x001fe40000000000 */
        // /*01f0*/                   SEL R2, R4, RZ, P0 ;                              /* 0x000000ff04027207 */
        // /* 0x000fe20000000000 */
        // /*0200*/                   IMAD.WIDE R4, R9, 0x8, R6 ;                       /* 0x0000000809047825 */
        // /* 0x008fe200078e0206 */
        // /*0210*/               @P1 LOP3.LUT R3, R8, 0x3ff00000, RZ, 0xfc, !PT ;      /* 0x3ff0000008031812 */
        // /* 0x000fca00078efcff */
        // /*0220*/                   STG.E.64 desc[UR4][R4.64], R2 ;                   /* 0x0000000204007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ double cvt_rzi_sat_f64_f64(double a) {
    double out;
    asm volatile("cvt.rzi.sat.f64.f64 %0, %1;" : "=d"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.S16.RM R4, R2 ;                  /* 0x0000000200047312 */
        // /* 0x004e260000105c00 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rm_f64_s16(int a) {
    double out;
    asm volatile("cvt.rm.f64.s16 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.S16 R4, R2 ;                     /* 0x0000000200047312 */
        // /* 0x004e260000101c00 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rn_f64_s16(int a) {
    double out;
    asm volatile("cvt.rn.f64.s16 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.S16.RP R4, R2 ;                  /* 0x0000000200047312 */
        // /* 0x004e260000109c00 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rp_f64_s16(int a) {
    double out;
    asm volatile("cvt.rp.f64.s16 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.S16.RZ R4, R2 ;                  /* 0x0000000200047312 */
        // /* 0x004e26000010dc00 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rz_f64_s16(int a) {
    double out;
    asm volatile("cvt.rz.f64.s16 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.RM R4, R2 ;                      /* 0x0000000200047312 */
        // /* 0x004e260000205c00 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rm_f64_s32(int a) {
    double out;
    asm volatile("cvt.rm.f64.s32 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64 R4, R2 ;                         /* 0x0000000200047312 */
        // /* 0x004e260000201c00 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rn_f64_s32(int a) {
    double out;
    asm volatile("cvt.rn.f64.s32 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;           /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.MOV.U32 R0, RZ, RZ, RZ ;                     /* 0x000000ffff007224 */
        // /* 0x000fe400078e00ff */
        // /*00b0*/                   HFMA2 R8, -RZ, RZ, 4.76837158203125e-07, 0 ;      /* 0x00080000ff087431 */
        // /* 0x000fe200000001ff */
        // /*00c0*/                   I2F.F64 R4, R2 ;                                  /* 0x0000000200047312 */
        // /* 0x004e240000201c00 */
        // /*00d0*/                   IMAD.MOV.U32 R13, RZ, RZ, R5 ;                    /* 0x000000ffff0d7224 */
        // /* 0x001fde00078e0005 */
        // /*00e0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*00f0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*0100*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*0110*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fc40000000000 */
        // /*0120*/                   DSETP.MAX.AND P0, P1, RZ, R4, PT ;                /* 0x00000004ff00722a */
        // /* 0x000062000390f000 */
        // /*0130*/                   MOV R11, R4 ;                                     /* 0x00000004000b7202 */
        // /* 0x000fe20000000f00 */
        // /*0140*/                   IMAD.MOV.U32 R4, RZ, RZ, RZ ;                     /* 0x000000ffff047224 */
        // /* 0x001fe200078e00ff */
        // /*0150*/                   FSEL R15, R0, R13, P0 ;                           /* 0x0000000d000f7208 */
        // /* 0x002fe40000000000 */
        // /*0160*/               @P1 LOP3.LUT R15, R13, 0x80000, RZ, 0xfc, !PT ;       /* 0x000800000d0f1812 */
        // /* 0x000fe400078efcff */
        // /*0170*/                   SEL R4, R4, R11, P0 ;                             /* 0x0000000b04047207 */
        // /* 0x000fc60000000000 */
        // /*0180*/                   IMAD.MOV.U32 R5, RZ, RZ, R15 ;                    /* 0x000000ffff057224 */
        // /* 0x000fca00078e000f */
        // /*0190*/                   MOV R0, R5 ;                                      /* 0x0000000500007202 */
        // /* 0x000fde0000000f00 */
        // /*01a0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*01b0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*01c0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fc80000000000 */
        // /*01d0*/                   DSETP.MIN.AND P0, P1, R4, 1, PT ;                 /* 0x3ff000000400742a */
        // /* 0x000e240003900000 */
        // /*01e0*/                   FSEL R3, R0, 1.875, P0 ;                          /* 0x3ff0000000037808 */
        // /* 0x001fe40000000000 */
        // /*01f0*/                   SEL R2, R4, RZ, P0 ;                              /* 0x000000ff04027207 */
        // /* 0x000fe20000000000 */
        // /*0200*/                   IMAD.WIDE R4, R9, 0x8, R6 ;                       /* 0x0000000809047825 */
        // /* 0x008fe200078e0206 */
        // /*0210*/               @P1 LOP3.LUT R3, R8, 0x3ff00000, RZ, 0xfc, !PT ;      /* 0x3ff0000008031812 */
        // /* 0x000fca00078efcff */
        // /*0220*/                   STG.E.64 desc[UR4][R4.64], R2 ;                   /* 0x0000000204007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ double cvt_rn_sat_f64_s32(int a) {
    double out;
    asm volatile("cvt.rn.sat.f64.s32 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.RP R4, R2 ;                      /* 0x0000000200047312 */
        // /* 0x004e260000209c00 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rp_f64_s32(int a) {
    double out;
    asm volatile("cvt.rp.f64.s32 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.RZ R4, R2 ;                      /* 0x0000000200047312 */
        // /* 0x004e26000020dc00 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rz_f64_s32(int a) {
    double out;
    asm volatile("cvt.rz.f64.s32 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                 /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.S64.RM R4, R2 ;                     /* 0x0000000200047312 */
        // /* 0x004e260000305c00 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;             /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rm_f64_s64(long long a) {
    double out;
    asm volatile("cvt.rm.f64.s64 %0, %1;" : "=d"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                 /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.S64 R4, R2 ;                        /* 0x0000000200047312 */
        // /* 0x004e260000301c00 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;             /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rn_f64_s64(long long a) {
    double out;
    asm volatile("cvt.rn.f64.s64 %0, %1;" : "=d"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;        /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.MOV.U32 R0, RZ, RZ, RZ ;                     /* 0x000000ffff007224 */
        // /* 0x000fe400078e00ff */
        // /*00b0*/                   HFMA2 R8, -RZ, RZ, 4.76837158203125e-07, 0 ;      /* 0x00080000ff087431 */
        // /* 0x000fe200000001ff */
        // /*00c0*/                   I2F.F64.S64 R4, R2 ;                              /* 0x0000000200047312 */
        // /* 0x004e240000301c00 */
        // /*00d0*/                   IMAD.MOV.U32 R13, RZ, RZ, R5 ;                    /* 0x000000ffff0d7224 */
        // /* 0x001fde00078e0005 */
        // /*00e0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*00f0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*0100*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*0110*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fc40000000000 */
        // /*0120*/                   DSETP.MAX.AND P0, P1, RZ, R4, PT ;                /* 0x00000004ff00722a */
        // /* 0x000062000390f000 */
        // /*0130*/                   MOV R11, R4 ;                                     /* 0x00000004000b7202 */
        // /* 0x000fe20000000f00 */
        // /*0140*/                   IMAD.MOV.U32 R4, RZ, RZ, RZ ;                     /* 0x000000ffff047224 */
        // /* 0x001fe200078e00ff */
        // /*0150*/                   FSEL R15, R0, R13, P0 ;                           /* 0x0000000d000f7208 */
        // /* 0x002fe40000000000 */
        // /*0160*/               @P1 LOP3.LUT R15, R13, 0x80000, RZ, 0xfc, !PT ;       /* 0x000800000d0f1812 */
        // /* 0x000fe400078efcff */
        // /*0170*/                   SEL R4, R4, R11, P0 ;                             /* 0x0000000b04047207 */
        // /* 0x000fc60000000000 */
        // /*0180*/                   IMAD.MOV.U32 R5, RZ, RZ, R15 ;                    /* 0x000000ffff057224 */
        // /* 0x000fca00078e000f */
        // /*0190*/                   MOV R0, R5 ;                                      /* 0x0000000500007202 */
        // /* 0x000fde0000000f00 */
        // /*01a0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*01b0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*01c0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fc80000000000 */
        // /*01d0*/                   DSETP.MIN.AND P0, P1, R4, 1, PT ;                 /* 0x3ff000000400742a */
        // /* 0x000e240003900000 */
        // /*01e0*/                   FSEL R3, R0, 1.875, P0 ;                          /* 0x3ff0000000037808 */
        // /* 0x001fe40000000000 */
        // /*01f0*/                   SEL R2, R4, RZ, P0 ;                              /* 0x000000ff04027207 */
        // /* 0x000fe20000000000 */
        // /*0200*/                   IMAD.WIDE R4, R9, 0x8, R6 ;                       /* 0x0000000809047825 */
        // /* 0x008fe200078e0206 */
        // /*0210*/               @P1 LOP3.LUT R3, R8, 0x3ff00000, RZ, 0xfc, !PT ;      /* 0x3ff0000008031812 */
        // /* 0x000fca00078efcff */
        // /*0220*/                   STG.E.64 desc[UR4][R4.64], R2 ;                   /* 0x0000000204007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ double cvt_rn_sat_f64_s64(long long a) {
    double out;
    asm volatile("cvt.rn.sat.f64.s64 %0, %1;" : "=d"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                 /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.S64.RP R4, R2 ;                     /* 0x0000000200047312 */
        // /* 0x004e260000309c00 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;             /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rp_f64_s64(long long a) {
    double out;
    asm volatile("cvt.rp.f64.s64 %0, %1;" : "=d"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                 /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.S64.RZ R4, R2 ;                     /* 0x0000000200047312 */
        // /* 0x004e26000030dc00 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;             /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rz_f64_s64(long long a) {
    double out;
    asm volatile("cvt.rz.f64.s64 %0, %1;" : "=d"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.S8.RM R4, R2 ;                   /* 0x0000000200047312 */
        // /* 0x004e260000005c00 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rm_f64_s8(int a) {
    double out;
    asm volatile("cvt.rm.f64.s8 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.S8 R4, R2 ;                      /* 0x0000000200047312 */
        // /* 0x004e260000001c00 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rn_f64_s8(int a) {
    double out;
    asm volatile("cvt.rn.f64.s8 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.S8.RP R4, R2 ;                   /* 0x0000000200047312 */
        // /* 0x004e260000009c00 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rp_f64_s8(int a) {
    double out;
    asm volatile("cvt.rp.f64.s8 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.S8.RZ R4, R2 ;                   /* 0x0000000200047312 */
        // /* 0x004e26000000dc00 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rz_f64_s8(int a) {
    double out;
    asm volatile("cvt.rz.f64.s8 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.U16.RM R4, R2 ;                  /* 0x0000000200047312 */
        // /* 0x004e260000105800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rm_f64_u16(unsigned int a) {
    double out;
    asm volatile("cvt.rm.f64.u16 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.U16 R4, R2 ;                     /* 0x0000000200047312 */
        // /* 0x004e260000101800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rn_f64_u16(unsigned int a) {
    double out;
    asm volatile("cvt.rn.f64.u16 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.U16.RP R4, R2 ;                  /* 0x0000000200047312 */
        // /* 0x004e260000109800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rp_f64_u16(unsigned int a) {
    double out;
    asm volatile("cvt.rp.f64.u16 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.U16.RZ R4, R2 ;                  /* 0x0000000200047312 */
        // /* 0x004e26000010d800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rz_f64_u16(unsigned int a) {
    double out;
    asm volatile("cvt.rz.f64.u16 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.U32.RM R4, R2 ;                  /* 0x0000000200047312 */
        // /* 0x004e260000205800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rm_f64_u32(unsigned int a) {
    double out;
    asm volatile("cvt.rm.f64.u32 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.U32 R4, R2 ;                     /* 0x0000000200047312 */
        // /* 0x004e260000201800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rn_f64_u32(unsigned int a) {
    double out;
    asm volatile("cvt.rn.f64.u32 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;           /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.MOV.U32 R0, RZ, RZ, RZ ;                     /* 0x000000ffff007224 */
        // /* 0x000fe400078e00ff */
        // /*00b0*/                   HFMA2 R8, -RZ, RZ, 4.76837158203125e-07, 0 ;      /* 0x00080000ff087431 */
        // /* 0x000fe200000001ff */
        // /*00c0*/                   I2F.F64.U32 R4, R2 ;                              /* 0x0000000200047312 */
        // /* 0x004e240000201800 */
        // /*00d0*/                   IMAD.MOV.U32 R13, RZ, RZ, R5 ;                    /* 0x000000ffff0d7224 */
        // /* 0x001fde00078e0005 */
        // /*00e0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*00f0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*0100*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*0110*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fc40000000000 */
        // /*0120*/                   DSETP.MAX.AND P0, P1, RZ, R4, PT ;                /* 0x00000004ff00722a */
        // /* 0x000062000390f000 */
        // /*0130*/                   MOV R11, R4 ;                                     /* 0x00000004000b7202 */
        // /* 0x000fe20000000f00 */
        // /*0140*/                   IMAD.MOV.U32 R4, RZ, RZ, RZ ;                     /* 0x000000ffff047224 */
        // /* 0x001fe200078e00ff */
        // /*0150*/                   FSEL R15, R0, R13, P0 ;                           /* 0x0000000d000f7208 */
        // /* 0x002fe40000000000 */
        // /*0160*/               @P1 LOP3.LUT R15, R13, 0x80000, RZ, 0xfc, !PT ;       /* 0x000800000d0f1812 */
        // /* 0x000fe400078efcff */
        // /*0170*/                   SEL R4, R4, R11, P0 ;                             /* 0x0000000b04047207 */
        // /* 0x000fc60000000000 */
        // /*0180*/                   IMAD.MOV.U32 R5, RZ, RZ, R15 ;                    /* 0x000000ffff057224 */
        // /* 0x000fca00078e000f */
        // /*0190*/                   MOV R0, R5 ;                                      /* 0x0000000500007202 */
        // /* 0x000fde0000000f00 */
        // /*01a0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*01b0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*01c0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fc80000000000 */
        // /*01d0*/                   DSETP.MIN.AND P0, P1, R4, 1, PT ;                 /* 0x3ff000000400742a */
        // /* 0x000e240003900000 */
        // /*01e0*/                   FSEL R3, R0, 1.875, P0 ;                          /* 0x3ff0000000037808 */
        // /* 0x001fe40000000000 */
        // /*01f0*/                   SEL R2, R4, RZ, P0 ;                              /* 0x000000ff04027207 */
        // /* 0x000fe20000000000 */
        // /*0200*/                   IMAD.WIDE R4, R9, 0x8, R6 ;                       /* 0x0000000809047825 */
        // /* 0x008fe200078e0206 */
        // /*0210*/               @P1 LOP3.LUT R3, R8, 0x3ff00000, RZ, 0xfc, !PT ;      /* 0x3ff0000008031812 */
        // /* 0x000fca00078efcff */
        // /*0220*/                   STG.E.64 desc[UR4][R4.64], R2 ;                   /* 0x0000000204007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ double cvt_rn_sat_f64_u32(unsigned int a) {
    double out;
    asm volatile("cvt.rn.sat.f64.u32 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.U32.RP R4, R2 ;                  /* 0x0000000200047312 */
        // /* 0x004e260000209800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rp_f64_u32(unsigned int a) {
    double out;
    asm volatile("cvt.rp.f64.u32 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.U32.RZ R4, R2 ;                  /* 0x0000000200047312 */
        // /* 0x004e26000020d800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rz_f64_u32(unsigned int a) {
    double out;
    asm volatile("cvt.rz.f64.u32 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                 /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.U64.RM R4, R2 ;                     /* 0x0000000200047312 */
        // /* 0x004e260000305800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;             /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rm_f64_u64(unsigned long long a) {
    double out;
    asm volatile("cvt.rm.f64.u64 %0, %1;" : "=d"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                 /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.U64 R4, R2 ;                        /* 0x0000000200047312 */
        // /* 0x004e260000301800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;             /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rn_f64_u64(unsigned long long a) {
    double out;
    asm volatile("cvt.rn.f64.u64 %0, %1;" : "=d"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;        /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.MOV.U32 R0, RZ, RZ, RZ ;                     /* 0x000000ffff007224 */
        // /* 0x000fe400078e00ff */
        // /*00b0*/                   HFMA2 R8, -RZ, RZ, 4.76837158203125e-07, 0 ;      /* 0x00080000ff087431 */
        // /* 0x000fe200000001ff */
        // /*00c0*/                   I2F.F64.U64 R4, R2 ;                              /* 0x0000000200047312 */
        // /* 0x004e240000301800 */
        // /*00d0*/                   IMAD.MOV.U32 R13, RZ, RZ, R5 ;                    /* 0x000000ffff0d7224 */
        // /* 0x001fde00078e0005 */
        // /*00e0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*00f0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*0100*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*0110*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fc40000000000 */
        // /*0120*/                   DSETP.MAX.AND P0, P1, RZ, R4, PT ;                /* 0x00000004ff00722a */
        // /* 0x000062000390f000 */
        // /*0130*/                   MOV R11, R4 ;                                     /* 0x00000004000b7202 */
        // /* 0x000fe20000000f00 */
        // /*0140*/                   IMAD.MOV.U32 R4, RZ, RZ, RZ ;                     /* 0x000000ffff047224 */
        // /* 0x001fe200078e00ff */
        // /*0150*/                   FSEL R15, R0, R13, P0 ;                           /* 0x0000000d000f7208 */
        // /* 0x002fe40000000000 */
        // /*0160*/               @P1 LOP3.LUT R15, R13, 0x80000, RZ, 0xfc, !PT ;       /* 0x000800000d0f1812 */
        // /* 0x000fe400078efcff */
        // /*0170*/                   SEL R4, R4, R11, P0 ;                             /* 0x0000000b04047207 */
        // /* 0x000fc60000000000 */
        // /*0180*/                   IMAD.MOV.U32 R5, RZ, RZ, R15 ;                    /* 0x000000ffff057224 */
        // /* 0x000fca00078e000f */
        // /*0190*/                   MOV R0, R5 ;                                      /* 0x0000000500007202 */
        // /* 0x000fde0000000f00 */
        // /*01a0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*01b0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fde0000000000 */
        // /*01c0*/                   NOP ;                                             /* 0x0000000000007918 */
        // /* 0x000fc80000000000 */
        // /*01d0*/                   DSETP.MIN.AND P0, P1, R4, 1, PT ;                 /* 0x3ff000000400742a */
        // /* 0x000e240003900000 */
        // /*01e0*/                   FSEL R3, R0, 1.875, P0 ;                          /* 0x3ff0000000037808 */
        // /* 0x001fe40000000000 */
        // /*01f0*/                   SEL R2, R4, RZ, P0 ;                              /* 0x000000ff04027207 */
        // /* 0x000fe20000000000 */
        // /*0200*/                   IMAD.WIDE R4, R9, 0x8, R6 ;                       /* 0x0000000809047825 */
        // /* 0x008fe200078e0206 */
        // /*0210*/               @P1 LOP3.LUT R3, R8, 0x3ff00000, RZ, 0xfc, !PT ;      /* 0x3ff0000008031812 */
        // /* 0x000fca00078efcff */
        // /*0220*/                   STG.E.64 desc[UR4][R4.64], R2 ;                   /* 0x0000000204007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ double cvt_rn_sat_f64_u64(unsigned long long a) {
    double out;
    asm volatile("cvt.rn.sat.f64.u64 %0, %1;" : "=d"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                 /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.U64.RP R4, R2 ;                     /* 0x0000000200047312 */
        // /* 0x004e260000309800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;             /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rp_f64_u64(unsigned long long a) {
    double out;
    asm volatile("cvt.rp.f64.u64 %0, %1;" : "=d"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                 /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.U64.RZ R4, R2 ;                     /* 0x0000000200047312 */
        // /* 0x004e26000030d800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;             /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rz_f64_u64(unsigned long long a) {
    double out;
    asm volatile("cvt.rz.f64.u64 %0, %1;" : "=d"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.U8.RM R4, R2 ;                   /* 0x0000000200047312 */
        // /* 0x004e260000005800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rm_f64_u8(unsigned int a) {
    double out;
    asm volatile("cvt.rm.f64.u8 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.U8 R4, R2 ;                      /* 0x0000000200047312 */
        // /* 0x004e260000001800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rn_f64_u8(unsigned int a) {
    double out;
    asm volatile("cvt.rn.f64.u8 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.U8.RP R4, R2 ;                   /* 0x0000000200047312 */
        // /* 0x004e260000009800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rp_f64_u8(unsigned int a) {
    double out;
    asm volatile("cvt.rp.f64.u8 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   I2F.F64.U8.RZ R4, R2 ;                   /* 0x0000000200047312 */
        // /* 0x004e26000000d800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ double cvt_rz_f64_u8(unsigned int a) {
    double out;
    asm volatile("cvt.rz.f64.u8 %0, %1;" : "=d"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S16.F16.FLOOR.NTZ R9, R2 ;               /* 0x0000000200097305 */
        // /* 0x004e280000106900 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rmi_s16_f16(uint16_t a) {
    int out;
    asm volatile("cvt.rmi.s16.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S16.F16.NTZ R9, R2 ;                     /* 0x0000000200097305 */
        // /* 0x004e280000102900 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rni_s16_f16(uint16_t a) {
    int out;
    asm volatile("cvt.rni.s16.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S16.F16.CEIL.NTZ R9, R2 ;                /* 0x0000000200097305 */
        // /* 0x004e28000010a900 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rpi_s16_f16(uint16_t a) {
    int out;
    asm volatile("cvt.rpi.s16.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S16.F16.TRUNC.NTZ R9, R2 ;               /* 0x0000000200097305 */
        // /* 0x004e28000010e900 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rzi_s16_f16(uint16_t a) {
    int out;
    asm volatile("cvt.rzi.s16.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S16.FLOOR.NTZ R9, R2 ;               /* 0x0000000200097305 */
        // /* 0x004e280000206900 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rmi_s16_f32(float a) {
    int out;
    asm volatile("cvt.rmi.s16.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.FTZ.S16.NTZ R0, R2 ;                 /* 0x0000000200007305 */
        // /* 0x004e240000212900 */
        // /*00c0*/                   PRMT R0, R0, 0x9910, RZ ;                /* 0x0000991000007816 */
        // /* 0x001fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_rni_ftz_s16_f32(float a) {
    int out;
    asm volatile("cvt.rni.ftz.s16.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.FTZ.S16.NTZ R0, R2 ;                 /* 0x0000000200007305 */
        // /* 0x004e240000212900 */
        // /*00c0*/                   PRMT R0, R0, 0x9910, RZ ;                /* 0x0000991000007816 */
        // /* 0x001fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_rni_ftz_sat_s16_f32(float a) {
    int out;
    asm volatile("cvt.rni.ftz.sat.s16.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S16.NTZ R9, R2 ;                     /* 0x0000000200097305 */
        // /* 0x004e280000202900 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rni_s16_f32(float a) {
    int out;
    asm volatile("cvt.rni.s16.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S16.NTZ R9, R2 ;                     /* 0x0000000200097305 */
        // /* 0x004e280000202900 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rni_sat_s16_f32(float a) {
    int out;
    asm volatile("cvt.rni.sat.s16.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S16.CEIL.NTZ R9, R2 ;                /* 0x0000000200097305 */
        // /* 0x004e28000020a900 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rpi_s16_f32(float a) {
    int out;
    asm volatile("cvt.rpi.s16.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.FTZ.S16.TRUNC.NTZ R0, R2 ;           /* 0x0000000200007305 */
        // /* 0x004e24000021e900 */
        // /*00c0*/                   PRMT R0, R0, 0x9910, RZ ;                /* 0x0000991000007816 */
        // /* 0x001fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_rzi_ftz_s16_f32(float a) {
    int out;
    asm volatile("cvt.rzi.ftz.s16.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.FTZ.S16.TRUNC.NTZ R0, R2 ;           /* 0x0000000200007305 */
        // /* 0x004e24000021e900 */
        // /*00c0*/                   PRMT R0, R0, 0x9910, RZ ;                /* 0x0000991000007816 */
        // /* 0x001fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_rzi_ftz_sat_s16_f32(float a) {
    int out;
    asm volatile("cvt.rzi.ftz.sat.s16.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S16.TRUNC.NTZ R9, R2 ;               /* 0x0000000200097305 */
        // /* 0x004e28000020e900 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rzi_s16_f32(float a) {
    int out;
    asm volatile("cvt.rzi.s16.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S16.TRUNC.NTZ R9, R2 ;               /* 0x0000000200097305 */
        // /* 0x004e28000020e900 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rzi_sat_s16_f32(float a) {
    int out;
    asm volatile("cvt.rzi.sat.s16.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S16.F64.FLOOR R0, R2 ;                  /* 0x0000000200007311 */
        // /* 0x004e240000304900 */
        // /*00c0*/                   PRMT R0, R0, 0x9910, RZ ;                   /* 0x0000991000007816 */
        // /* 0x001fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                                /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_rmi_s16_f64(double a) {
    int out;
    asm volatile("cvt.rmi.s16.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S16.F64 R0, R2 ;                        /* 0x0000000200007311 */
        // /* 0x004e240000300900 */
        // /*00c0*/                   PRMT R0, R0, 0x9910, RZ ;                   /* 0x0000991000007816 */
        // /* 0x001fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                                /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_rni_s16_f64(double a) {
    int out;
    asm volatile("cvt.rni.s16.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S16.F64 R0, R2 ;                        /* 0x0000000200007311 */
        // /* 0x004e240000300900 */
        // /*00c0*/                   PRMT R0, R0, 0x9910, RZ ;                   /* 0x0000991000007816 */
        // /* 0x001fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                                /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_rni_sat_s16_f64(double a) {
    int out;
    asm volatile("cvt.rni.sat.s16.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S16.F64.CEIL R0, R2 ;                   /* 0x0000000200007311 */
        // /* 0x004e240000308900 */
        // /*00c0*/                   PRMT R0, R0, 0x9910, RZ ;                   /* 0x0000991000007816 */
        // /* 0x001fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                                /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_rpi_s16_f64(double a) {
    int out;
    asm volatile("cvt.rpi.s16.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S16.F64.TRUNC R0, R2 ;                  /* 0x0000000200007311 */
        // /* 0x004e24000030c900 */
        // /*00c0*/                   PRMT R0, R0, 0x9910, RZ ;                   /* 0x0000991000007816 */
        // /* 0x001fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                                /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_rzi_s16_f64(double a) {
    int out;
    asm volatile("cvt.rzi.s16.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S16.F64.TRUNC R0, R2 ;                  /* 0x0000000200007311 */
        // /* 0x004e24000030c900 */
        // /*00c0*/                   PRMT R0, R0, 0x9910, RZ ;                   /* 0x0000991000007816 */
        // /* 0x001fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                                /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_rzi_sat_s16_f64(double a) {
    int out;
    asm volatile("cvt.rzi.sat.s16.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x9910, RZ ;                /* 0x0000991002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_s16_s32(int a) {
    int out;
    asm volatile("cvt.s16.s32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2I.S16.S32.SAT R0, R2 ;                 /* 0x0000000200007238 */
        // /* 0x004fc80000003000 */
        // /*00c0*/                   PRMT R0, R0, 0x9910, RZ ;                /* 0x0000991000007816 */
        // /* 0x000fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_sat_s16_s32(int a) {
    int out;
    asm volatile("cvt.sat.s16.s32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x9910, RZ ;                    /* 0x0000991002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   MOV R7, R0 ;                                 /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                 /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_s16_s64(long long a) {
    int out;
    asm volatile("cvt.s16.s64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.MOV.U32 R6, RZ, RZ, 0x7fffffff ;       /* 0x7fffffffff067424 */
        // /* 0x000fe200078e00ff */
        // /*00b0*/                   SHF.R.S32.HI R9, RZ, 0x1f, R2 ;             /* 0x0000001fff097819 */
        // /* 0x004fe40000011402 */
        // /*00c0*/                   ISETP.GE.AND P0, PT, R3, RZ, PT ;           /* 0x000000ff0300720c */
        // /* 0x000fe40003f06270 */
        // /*00d0*/                   LOP3.LUT R0, R9, R3, RZ, 0x3c, !PT ;        /* 0x0000000309007212 */
        // /* 0x000fe400078e3cff */
        // /*00e0*/                   SEL R9, R6, 0x80000000, P0 ;                /* 0x8000000006097807 */
        // /* 0x000fe40000000000 */
        // /*00f0*/                   ISETP.EQ.AND P0, PT, R0, RZ, PT ;           /* 0x000000ff0000720c */
        // /* 0x000fc80003f02270 */
        // /*0100*/                   SEL R0, R2, R9, P0 ;                        /* 0x0000000902007207 */
        // /* 0x000fe20000000000 */
        // /*0110*/                   IMAD.WIDE R2, R7, 0x4, R4 ;                 /* 0x0000000407027825 */
        // /* 0x008fc600078e0204 */
        // /*0120*/                   I2I.S16.S32.SAT R0, R0 ;                    /* 0x0000000000007238 */
        // /* 0x000fc80000003000 */
        // /*0130*/                   PRMT R0, R0, 0x9910, RZ ;                   /* 0x0000991000007816 */
        // /* 0x000fca00000000ff */
        // /*0140*/                   IMAD.MOV.U32 R5, RZ, RZ, R0 ;               /* 0x000000ffff057224 */
        // /* 0x000fca00078e0000 */
        // /*0150*/                   STG.E desc[UR4][R2.64], R5 ;                /* 0x0000000502007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_sat_s16_s64(long long a) {
    int out;
    asm volatile("cvt.sat.s16.s64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x8880, RZ ;                /* 0x0000888002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_s16_s8(int a) {
    int out;
    asm volatile("cvt.s16.s8 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x9910, RZ ;                /* 0x0000991002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_s16_u16(unsigned int a) {
    int out;
    asm volatile("cvt.s16.u16 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x7710, RZ ;                /* 0x0000771002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   VIMNMX.U32 R0, PT, PT, R0, 0x7fff, PT ;  /* 0x00007fff00007848 */
        // /* 0x000fc80003fe0000 */
        // /*00d0*/                   PRMT R0, R0, 0x9910, RZ ;                /* 0x0000991000007816 */
        // /* 0x000fc800000000ff */
        // /*00e0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00f0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_sat_s16_u16(unsigned int a) {
    int out;
    asm volatile("cvt.sat.s16.u16 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x9910, RZ ;                /* 0x0000991002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_s16_u32(unsigned int a) {
    int out;
    asm volatile("cvt.s16.u32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   VIMNMX.U32 R0, PT, PT, R2, 0x7fff, PT ;  /* 0x00007fff02007848 */
        // /* 0x004fc80003fe0000 */
        // /*00c0*/                   PRMT R0, R0, 0x9910, RZ ;                /* 0x0000991000007816 */
        // /* 0x000fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_sat_s16_u32(unsigned int a) {
    int out;
    asm volatile("cvt.sat.s16.u32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x9910, RZ ;                    /* 0x0000991002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   MOV R7, R0 ;                                 /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                 /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_s16_u64(unsigned long long a) {
    int out;
    asm volatile("cvt.s16.u64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   ISETP.NE.U32.AND P0, PT, R3, RZ, PT ;       /* 0x000000ff0300720c */
        // /* 0x004fc80003f05070 */
        // /*00c0*/                   SEL R9, RZ, 0xffffffff, !P0 ;               /* 0xffffffffff097807 */
        // /* 0x000fc80004000000 */
        // /*00d0*/                   LOP3.LUT R0, R9, R2, RZ, 0xfc, !PT ;        /* 0x0000000209007212 */
        // /* 0x000fc800078efcff */
        // /*00e0*/                   VIMNMX.U32 R0, PT, PT, R0, 0x7fff, PT ;     /* 0x00007fff00007848 */
        // /* 0x000fc80003fe0000 */
        // /*00f0*/                   PRMT R0, R0, 0x9910, RZ ;                   /* 0x0000991000007816 */
        // /* 0x000fca00000000ff */
        // /*0100*/                   IMAD.MOV.U32 R7, RZ, RZ, R0 ;               /* 0x000000ffff077224 */
        // /* 0x000fca00078e0000 */
        // /*0110*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_sat_s16_u64(unsigned long long a) {
    int out;
    asm volatile("cvt.sat.s16.u64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   LOP3.LUT R7, R2, 0xff, RZ, 0xc0, !PT ;   /* 0x000000ff02077812 */
        // /* 0x004fca00078ec0ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_s16_u8(unsigned int a) {
    int out;
    asm volatile("cvt.s16.u8 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.BF16.FLOOR.NTZ R9, R2 ;                  /* 0x0000000200097305 */
        // /* 0x004e280000407100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rmi_s32_bf16(uint16_t a) {
    int out;
    asm volatile("cvt.rmi.s32.bf16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.BF16.NTZ R9, R2 ;                        /* 0x0000000200097305 */
        // /* 0x004e280000403100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rni_s32_bf16(uint16_t a) {
    int out;
    asm volatile("cvt.rni.s32.bf16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.BF16.CEIL.NTZ R9, R2 ;                   /* 0x0000000200097305 */
        // /* 0x004e28000040b100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rpi_s32_bf16(uint16_t a) {
    int out;
    asm volatile("cvt.rpi.s32.bf16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.BF16.TRUNC.NTZ R9, R2 ;                  /* 0x0000000200097305 */
        // /* 0x004e28000040f100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rzi_s32_bf16(uint16_t a) {
    int out;
    asm volatile("cvt.rzi.s32.bf16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.F16.FLOOR.NTZ R9, R2 ;                   /* 0x0000000200097305 */
        // /* 0x004e280000107100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rmi_s32_f16(uint16_t a) {
    int out;
    asm volatile("cvt.rmi.s32.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.F16.NTZ R9, R2 ;                         /* 0x0000000200097305 */
        // /* 0x004e280000103100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rni_s32_f16(uint16_t a) {
    int out;
    asm volatile("cvt.rni.s32.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.F16.CEIL.NTZ R9, R2 ;                    /* 0x0000000200097305 */
        // /* 0x004e28000010b100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rpi_s32_f16(uint16_t a) {
    int out;
    asm volatile("cvt.rpi.s32.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.F16.TRUNC.NTZ R9, R2 ;                   /* 0x0000000200097305 */
        // /* 0x004e28000010f100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rzi_s32_f16(uint16_t a) {
    int out;
    asm volatile("cvt.rzi.s32.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.FLOOR.NTZ R9, R2 ;                   /* 0x0000000200097305 */
        // /* 0x004e280000207100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rmi_s32_f32(float a) {
    int out;
    asm volatile("cvt.rmi.s32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.FTZ.NTZ R9, R2 ;                     /* 0x0000000200097305 */
        // /* 0x004e280000213100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rni_ftz_s32_f32(float a) {
    int out;
    asm volatile("cvt.rni.ftz.s32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.NTZ R9, R2 ;                         /* 0x0000000200097305 */
        // /* 0x004e280000203100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rni_s32_f32(float a) {
    int out;
    asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.CEIL.NTZ R9, R2 ;                    /* 0x0000000200097305 */
        // /* 0x004e28000020b100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rpi_s32_f32(float a) {
    int out;
    asm volatile("cvt.rpi.s32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.FTZ.TRUNC.NTZ R9, R2 ;               /* 0x0000000200097305 */
        // /* 0x004e28000021f100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rzi_ftz_s32_f32(float a) {
    int out;
    asm volatile("cvt.rzi.ftz.s32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.TRUNC.NTZ R9, R2 ;                   /* 0x0000000200097305 */
        // /* 0x004e28000020f100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rzi_s32_f32(float a) {
    int out;
    asm volatile("cvt.rzi.s32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   F2I.F64.FLOOR R7, R2 ;                      /* 0x0000000200077311 */
        // /* 0x004e260000305100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rmi_s32_f64(double a) {
    int out;
    asm volatile("cvt.rmi.s32.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   F2I.F64 R7, R2 ;                            /* 0x0000000200077311 */
        // /* 0x004e260000301100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rni_s32_f64(double a) {
    int out;
    asm volatile("cvt.rni.s32.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   F2I.F64.CEIL R7, R2 ;                       /* 0x0000000200077311 */
        // /* 0x004e260000309100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rpi_s32_f64(double a) {
    int out;
    asm volatile("cvt.rpi.s32.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   F2I.F64.TRUNC R7, R2 ;                      /* 0x0000000200077311 */
        // /* 0x004e26000030d100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rzi_s32_f64(double a) {
    int out;
    asm volatile("cvt.rzi.s32.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x9910, RZ ;                /* 0x0000991002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_s32_s16(int a) {
    int out;
    asm volatile("cvt.s32.s16 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;             /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ int cvt_s32_s64(long long a) {
    int out;
    asm volatile("cvt.s32.s64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.MOV.U32 R6, RZ, RZ, 0x7fffffff ;       /* 0x7fffffffff067424 */
        // /* 0x000fe400078e00ff */
        // /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00c0*/                   SHF.R.S32.HI R9, RZ, 0x1f, R2 ;             /* 0x0000001fff097819 */
        // /* 0x004fe40000011402 */
        // /*00d0*/                   ISETP.GE.AND P0, PT, R3, RZ, PT ;           /* 0x000000ff0300720c */
        // /* 0x000fe40003f06270 */
        // /*00e0*/                   LOP3.LUT R0, R9, R3, RZ, 0x3c, !PT ;        /* 0x0000000309007212 */
        // /* 0x000fe400078e3cff */
        // /*00f0*/                   SEL R9, R6, 0x80000000, P0 ;                /* 0x8000000006097807 */
        // /* 0x000fc40000000000 */
        // /*0100*/                   ISETP.EQ.AND P0, PT, R0, RZ, PT ;           /* 0x000000ff0000720c */
        // /* 0x000fc80003f02270 */
        // /*0110*/                   SEL R9, R2, R9, P0 ;                        /* 0x0000000902097207 */
        // /* 0x000fca0000000000 */
        // /*0120*/                   STG.E desc[UR4][R4.64], R9 ;                /* 0x0000000904007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_sat_s32_s64(long long a) {
    int out;
    asm volatile("cvt.sat.s32.s64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x8880, RZ ;                /* 0x0000888002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_s32_s8(int a) {
    int out;
    asm volatile("cvt.s32.s8 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;   /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;               /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   LOP3.LUT R7, R2, 0xffff, RZ, 0xc0, !PT ;  /* 0x0000ffff02077812 */
        // /* 0x004fca00078ec0ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;              /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_s32_u16(unsigned int a) {
    int out;
    asm volatile("cvt.s32.u16 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;             /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ int cvt_s32_u32(unsigned int a) {
    int out;
    asm volatile("cvt.s32.u32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;         /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                     /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   ISETP.GE.U32.AND P0, PT, R2, -0x80000000, PT ;  /* 0x800000000200780c */
        // /* 0x004fc80003f06070 */
        // /*00c0*/                   SEL R7, R2, 0x7fffffff, !P0 ;                   /* 0x7fffffff02077807 */
        // /* 0x000fca0004000000 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                    /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_sat_s32_u32(unsigned int a) {
    int out;
    asm volatile("cvt.sat.s32.u32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;             /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ int cvt_s32_u64(unsigned long long a) {
    int out;
    asm volatile("cvt.s32.u64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;      /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                     /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   ISETP.NE.U32.AND P0, PT, R3, RZ, PT ;           /* 0x000000ff0300720c */
        // /* 0x004fc80003f05070 */
        // /*00c0*/                   SEL R9, RZ, 0xffffffff, !P0 ;                   /* 0xffffffffff097807 */
        // /* 0x000fc80004000000 */
        // /*00d0*/                   LOP3.LUT R9, R9, R2, RZ, 0xfc, !PT ;            /* 0x0000000209097212 */
        // /* 0x000fc800078efcff */
        // /*00e0*/                   ISETP.GE.U32.AND P0, PT, R9, -0x80000000, PT ;  /* 0x800000000900780c */
        // /* 0x000fc80003f06070 */
        // /*00f0*/                   SEL R7, R9, 0x7fffffff, !P0 ;                   /* 0x7fffffff09077807 */
        // /* 0x000fca0004000000 */
        // /*0100*/                   STG.E desc[UR4][R4.64], R7 ;                    /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_sat_s32_u64(unsigned long long a) {
    int out;
    asm volatile("cvt.sat.s32.u64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   LOP3.LUT R7, R2, 0xff, RZ, 0xc0, !PT ;   /* 0x000000ff02077812 */
        // /* 0x004fca00078ec0ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_s32_u8(unsigned int a) {
    int out;
    asm volatile("cvt.s32.u8 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                  /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.S64.BF16.FLOOR R4, R2 ;                  /* 0x0000000200047311 */
        // /* 0x004e260000405900 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;              /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ long long cvt_rmi_s64_bf16(uint16_t a) {
    long long out;
    asm volatile("cvt.rmi.s64.bf16 %0, %1;" : "=l"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                  /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.S64.BF16 R4, R2 ;                        /* 0x0000000200047311 */
        // /* 0x004e260000401900 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;              /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ long long cvt_rni_s64_bf16(uint16_t a) {
    long long out;
    asm volatile("cvt.rni.s64.bf16 %0, %1;" : "=l"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                  /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.S64.BF16.CEIL R4, R2 ;                   /* 0x0000000200047311 */
        // /* 0x004e260000409900 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;              /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ long long cvt_rpi_s64_bf16(uint16_t a) {
    long long out;
    asm volatile("cvt.rpi.s64.bf16 %0, %1;" : "=l"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                  /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.S64.BF16.TRUNC R4, R2 ;                  /* 0x0000000200047311 */
        // /* 0x004e26000040d900 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;              /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ long long cvt_rzi_s64_bf16(uint16_t a) {
    long long out;
    asm volatile("cvt.rzi.s64.bf16 %0, %1;" : "=l"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                  /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.S64.F16.FLOOR R4, R2 ;                   /* 0x0000000200047311 */
        // /* 0x004e260000105900 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;              /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ long long cvt_rmi_s64_f16(uint16_t a) {
    long long out;
    asm volatile("cvt.rmi.s64.f16 %0, %1;" : "=l"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                  /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.S64.F16 R4, R2 ;                         /* 0x0000000200047311 */
        // /* 0x004e260000101900 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;              /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ long long cvt_rni_s64_f16(uint16_t a) {
    long long out;
    asm volatile("cvt.rni.s64.f16 %0, %1;" : "=l"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                  /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.S64.F16.CEIL R4, R2 ;                    /* 0x0000000200047311 */
        // /* 0x004e260000109900 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;              /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ long long cvt_rpi_s64_f16(uint16_t a) {
    long long out;
    asm volatile("cvt.rpi.s64.f16 %0, %1;" : "=l"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                  /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.S64.F16.TRUNC R4, R2 ;                   /* 0x0000000200047311 */
        // /* 0x004e26000010d900 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;              /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ long long cvt_rzi_s64_f16(uint16_t a) {
    long long out;
    asm volatile("cvt.rzi.s64.f16 %0, %1;" : "=l"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.S64.FLOOR R4, R2 ;                   /* 0x0000000200047311 */
        // /* 0x004e260000205900 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ long long cvt_rmi_s64_f32(float a) {
    long long out;
    asm volatile("cvt.rmi.s64.f32 %0, %1;" : "=l"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.S64 R4, R2 ;                         /* 0x0000000200047311 */
        // /* 0x004e260000201900 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ long long cvt_rni_ftz_s64_f32(float a) {
    long long out;
    asm volatile("cvt.rni.ftz.s64.f32 %0, %1;" : "=l"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.S64 R4, R2 ;                         /* 0x0000000200047311 */
        // /* 0x004e260000201900 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ long long cvt_rni_s64_f32(float a) {
    long long out;
    asm volatile("cvt.rni.s64.f32 %0, %1;" : "=l"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.S64.CEIL R4, R2 ;                    /* 0x0000000200047311 */
        // /* 0x004e260000209900 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ long long cvt_rpi_s64_f32(float a) {
    long long out;
    asm volatile("cvt.rpi.s64.f32 %0, %1;" : "=l"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.S64.TRUNC R4, R2 ;                   /* 0x0000000200047311 */
        // /* 0x004e26000020d900 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ long long cvt_rzi_ftz_s64_f32(float a) {
    long long out;
    asm volatile("cvt.rzi.ftz.s64.f32 %0, %1;" : "=l"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.S64.TRUNC R4, R2 ;                   /* 0x0000000200047311 */
        // /* 0x004e26000020d900 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ long long cvt_rzi_s64_f32(float a) {
    long long out;
    asm volatile("cvt.rzi.s64.f32 %0, %1;" : "=l"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                 /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.S64.F64.FLOOR R4, R2 ;                  /* 0x0000000200047311 */
        // /* 0x004e260000305900 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;             /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ long long cvt_rmi_s64_f64(double a) {
    long long out;
    asm volatile("cvt.rmi.s64.f64 %0, %1;" : "=l"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                 /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.S64.F64 R4, R2 ;                        /* 0x0000000200047311 */
        // /* 0x004e260000301900 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;             /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ long long cvt_rni_s64_f64(double a) {
    long long out;
    asm volatile("cvt.rni.s64.f64 %0, %1;" : "=l"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                 /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.S64.F64.CEIL R4, R2 ;                   /* 0x0000000200047311 */
        // /* 0x004e260000309900 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;             /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ long long cvt_rpi_s64_f64(double a) {
    long long out;
    asm volatile("cvt.rpi.s64.f64 %0, %1;" : "=l"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                 /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.S64.F64.TRUNC R4, R2 ;                  /* 0x0000000200047311 */
        // /* 0x004e26000030d900 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;             /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ long long cvt_rzi_s64_f64(double a) {
    long long out;
    asm volatile("cvt.rzi.s64.f64 %0, %1;" : "=l"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R9, 0x8, R6 ;              /* 0x0000000809067825 */
        // /* 0x008fe200078e0206 */
        // /*00b0*/                   PRMT R4, R2, 0x9910, RZ ;                /* 0x0000991002047816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   SHF.R.S32.HI R5, RZ, 0x1f, R4 ;          /* 0x0000001fff057819 */
        // /* 0x000fca0000011404 */
        // /*00d0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ long long cvt_s64_s16(int a) {
    long long out;
    asm volatile("cvt.s64.s16 %0, %1;" : "=l"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R4, desc[UR4][R2.64] ;  /* 0x0000000402047981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R9, 0x8, R6 ;              /* 0x0000000809067825 */
        // /* 0x008fe200078e0206 */
        // /*00b0*/                   SHF.R.S32.HI R5, RZ, 0x1f, R4 ;          /* 0x0000001fff057819 */
        // /* 0x004fca0000011404 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ long long cvt_s64_s32(int a) {
    long long out;
    asm volatile("cvt.s64.s32 %0, %1;" : "=l"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R9, 0x8, R6 ;              /* 0x0000000809067825 */
        // /* 0x008fe200078e0206 */
        // /*00b0*/                   PRMT R4, R2, 0x8880, RZ ;                /* 0x0000888002047816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   SHF.R.S32.HI R5, RZ, 0x1f, R4 ;          /* 0x0000001fff057819 */
        // /* 0x000fca0000011404 */
        // /*00d0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ long long cvt_s64_s8(int a) {
    long long out;
    asm volatile("cvt.s64.s8 %0, %1;" : "=l"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;   /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;               /* 0x0000000805067825 */
        // /* 0x008fc800078e0206 */
        // /*00b0*/                   HFMA2 R5, -RZ, RZ, 0, 0 ;                 /* 0x00000000ff057431 */
        // /* 0x000fe200000001ff */
        // /*00c0*/                   LOP3.LUT R4, R2, 0xffff, RZ, 0xc0, !PT ;  /* 0x0000ffff02047812 */
        // /* 0x004fca00078ec0ff */
        // /*00d0*/                   STG.E.64 desc[UR4][R6.64], R4 ;           /* 0x0000000406007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ long long cvt_s64_u16(unsigned int a) {
    long long out;
    asm volatile("cvt.s64.u16 %0, %1;" : "=l"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x0020a2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x8, R4 ;              /* 0x0000000807047825 */
        // /* 0x008fc800078e0204 */
        // /*00b0*/                   HFMA2 R3, -RZ, RZ, 0, 0 ;                /* 0x00000000ff037431 */
        // /* 0x001fca00000001ff */
        // /*00c0*/                   STG.E.64 desc[UR4][R4.64], R2 ;          /* 0x0000000204007986 */
        // /* 0x004fe2000c101b04 */
__device__ __forceinline__ long long cvt_s64_u32(unsigned int a) {
    long long out;
    asm volatile("cvt.s64.u32 %0, %1;" : "=l"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x8, R4 ;                 /* 0x0000000807047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E.64 desc[UR4][R4.64], R2 ;             /* 0x0000000204007986 */
        // /* 0x004fe2000c101b04 */
__device__ __forceinline__ long long cvt_s64_u64(unsigned long long a) {
    long long out;
    asm volatile("cvt.s64.u64 %0, %1;" : "=l"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;            /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R9, 0x8, R6 ;                           /* 0x0000000809067825 */
        // /* 0x008fe200078e0206 */
        // /*00b0*/                   ISETP.LT.AND P0, PT, R3.reuse, RZ, PT ;               /* 0x000000ff0300720c */
        // /* 0x044fe40003f01270 */
        // /*00c0*/                   ISETP.GE.U32.AND P1, PT, R3.reuse, -0x80000000, PT ;  /* 0x800000000300780c */
        // /* 0x040fe40003f26070 */
        // /*00d0*/                   SEL R11, RZ, 0xffffffff, !P0 ;                        /* 0xffffffffff0b7807 */
        // /* 0x000fe40004000000 */
        // /*00e0*/                   SEL R5, R3, 0x7fffffff, !P1 ;                         /* 0x7fffffff03057807 */
        // /* 0x000fe40004800000 */
        // /*00f0*/                   LOP3.LUT R4, R11, R2, RZ, 0xfc, !PT ;                 /* 0x000000020b047212 */
        // /* 0x000fca00078efcff */
        // /*0100*/                   STG.E.64 desc[UR4][R6.64], R4 ;                       /* 0x0000000406007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ long long cvt_sat_s64_u64(unsigned long long a) {
    long long out;
    asm volatile("cvt.sat.s64.u64 %0, %1;" : "=l"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fc800078e0206 */
        // /*00b0*/                   HFMA2 R5, -RZ, RZ, 0, 0 ;                /* 0x00000000ff057431 */
        // /* 0x000fe200000001ff */
        // /*00c0*/                   LOP3.LUT R4, R2, 0xff, RZ, 0xc0, !PT ;   /* 0x000000ff02047812 */
        // /* 0x004fca00078ec0ff */
        // /*00d0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ long long cvt_s64_u8(unsigned int a) {
    long long out;
    asm volatile("cvt.s64.u8 %0, %1;" : "=l"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S8.F16.FLOOR.NTZ R9, R2 ;                /* 0x0000000200097305 */
        // /* 0x004e280000106100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rmi_s8_f16(uint16_t a) {
    int out;
    asm volatile("cvt.rmi.s8.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S8.F16.NTZ R9, R2 ;                      /* 0x0000000200097305 */
        // /* 0x004e280000102100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rni_s8_f16(uint16_t a) {
    int out;
    asm volatile("cvt.rni.s8.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S8.F16.CEIL.NTZ R9, R2 ;                 /* 0x0000000200097305 */
        // /* 0x004e28000010a100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rpi_s8_f16(uint16_t a) {
    int out;
    asm volatile("cvt.rpi.s8.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S8.F16.TRUNC.NTZ R9, R2 ;                /* 0x0000000200097305 */
        // /* 0x004e28000010e100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rzi_s8_f16(uint16_t a) {
    int out;
    asm volatile("cvt.rzi.s8.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S8.FLOOR.NTZ R9, R2 ;                /* 0x0000000200097305 */
        // /* 0x004e280000206100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rmi_s8_f32(float a) {
    int out;
    asm volatile("cvt.rmi.s8.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.FTZ.S8.NTZ R0, R2 ;                  /* 0x0000000200007305 */
        // /* 0x004e240000212100 */
        // /*00c0*/                   PRMT R0, R0, 0x8880, RZ ;                /* 0x0000888000007816 */
        // /* 0x001fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_rni_ftz_s8_f32(float a) {
    int out;
    asm volatile("cvt.rni.ftz.s8.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.FTZ.S8.NTZ R0, R2 ;                  /* 0x0000000200007305 */
        // /* 0x004e240000212100 */
        // /*00c0*/                   PRMT R0, R0, 0x8880, RZ ;                /* 0x0000888000007816 */
        // /* 0x001fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_rni_ftz_sat_s8_f32(float a) {
    int out;
    asm volatile("cvt.rni.ftz.sat.s8.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S8.NTZ R9, R2 ;                      /* 0x0000000200097305 */
        // /* 0x004e280000202100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rni_s8_f32(float a) {
    int out;
    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S8.NTZ R9, R2 ;                      /* 0x0000000200097305 */
        // /* 0x004e280000202100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rni_sat_s8_f32(float a) {
    int out;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S8.CEIL.NTZ R9, R2 ;                 /* 0x0000000200097305 */
        // /* 0x004e28000020a100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rpi_s8_f32(float a) {
    int out;
    asm volatile("cvt.rpi.s8.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.FTZ.S8.TRUNC.NTZ R0, R2 ;            /* 0x0000000200007305 */
        // /* 0x004e24000021e100 */
        // /*00c0*/                   PRMT R0, R0, 0x8880, RZ ;                /* 0x0000888000007816 */
        // /* 0x001fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_rzi_ftz_s8_f32(float a) {
    int out;
    asm volatile("cvt.rzi.ftz.s8.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.FTZ.S8.TRUNC.NTZ R0, R2 ;            /* 0x0000000200007305 */
        // /* 0x004e24000021e100 */
        // /*00c0*/                   PRMT R0, R0, 0x8880, RZ ;                /* 0x0000888000007816 */
        // /* 0x001fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_rzi_ftz_sat_s8_f32(float a) {
    int out;
    asm volatile("cvt.rzi.ftz.sat.s8.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S8.TRUNC.NTZ R9, R2 ;                /* 0x0000000200097305 */
        // /* 0x004e28000020e100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rzi_s8_f32(float a) {
    int out;
    asm volatile("cvt.rzi.s8.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S8.TRUNC.NTZ R9, R2 ;                /* 0x0000000200097305 */
        // /* 0x004e28000020e100 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ int cvt_rzi_sat_s8_f32(float a) {
    int out;
    asm volatile("cvt.rzi.sat.s8.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S8.F64.FLOOR R0, R2 ;                   /* 0x0000000200007311 */
        // /* 0x004e240000304100 */
        // /*00c0*/                   PRMT R0, R0, 0x8880, RZ ;                   /* 0x0000888000007816 */
        // /* 0x001fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                                /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_rmi_s8_f64(double a) {
    int out;
    asm volatile("cvt.rmi.s8.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S8.F64 R0, R2 ;                         /* 0x0000000200007311 */
        // /* 0x004e240000300100 */
        // /*00c0*/                   PRMT R0, R0, 0x8880, RZ ;                   /* 0x0000888000007816 */
        // /* 0x001fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                                /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_rni_s8_f64(double a) {
    int out;
    asm volatile("cvt.rni.s8.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S8.F64 R0, R2 ;                         /* 0x0000000200007311 */
        // /* 0x004e240000300100 */
        // /*00c0*/                   PRMT R0, R0, 0x8880, RZ ;                   /* 0x0000888000007816 */
        // /* 0x001fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                                /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_rni_sat_s8_f64(double a) {
    int out;
    asm volatile("cvt.rni.sat.s8.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S8.F64.CEIL R0, R2 ;                    /* 0x0000000200007311 */
        // /* 0x004e240000308100 */
        // /*00c0*/                   PRMT R0, R0, 0x8880, RZ ;                   /* 0x0000888000007816 */
        // /* 0x001fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                                /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_rpi_s8_f64(double a) {
    int out;
    asm volatile("cvt.rpi.s8.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S8.F64.TRUNC R0, R2 ;                   /* 0x0000000200007311 */
        // /* 0x004e24000030c100 */
        // /*00c0*/                   PRMT R0, R0, 0x8880, RZ ;                   /* 0x0000888000007816 */
        // /* 0x001fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                                /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_rzi_s8_f64(double a) {
    int out;
    asm volatile("cvt.rzi.s8.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.S8.F64.TRUNC R0, R2 ;                   /* 0x0000000200007311 */
        // /* 0x004e24000030c100 */
        // /*00c0*/                   PRMT R0, R0, 0x8880, RZ ;                   /* 0x0000888000007816 */
        // /* 0x001fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                                /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_rzi_sat_s8_f64(double a) {
    int out;
    asm volatile("cvt.rzi.sat.s8.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x8880, RZ ;                /* 0x0000888002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_s8_s16(int a) {
    int out;
    asm volatile("cvt.s8.s16 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;     /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x9910, RZ ;                   /* 0x0000991002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   ISETP.LT.AND P0, PT, R0.reuse, -0x80, PT ;  /* 0xffffff800000780c */
        // /* 0x040fe40003f01270 */
        // /*00d0*/                   ISETP.GT.AND P1, PT, R0.reuse, 0x7f, PT ;   /* 0x0000007f0000780c */
        // /* 0x040fe40003f24270 */
        // /*00e0*/                   SEL R0, R0, 0xffffff80, !P0 ;               /* 0xffffff8000007807 */
        // /* 0x000fc80004000000 */
        // /*00f0*/                   SEL R0, R0, 0x7f, !P1 ;                     /* 0x0000007f00007807 */
        // /* 0x000fc80004800000 */
        // /*0100*/                   PRMT R0, R0, 0x8880, RZ ;                   /* 0x0000888000007816 */
        // /* 0x000fca00000000ff */
        // /*0110*/                   IMAD.MOV.U32 R7, RZ, RZ, R0 ;               /* 0x000000ffff077224 */
        // /* 0x000fca00078e0000 */
        // /*0120*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_sat_s8_s16(int a) {
    int out;
    asm volatile("cvt.sat.s8.s16 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x8880, RZ ;                /* 0x0000888002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_s8_s32(int a) {
    int out;
    asm volatile("cvt.s8.s32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2I.S8.S32.SAT R0, R2 ;                  /* 0x0000000200007238 */
        // /* 0x004fc80000001000 */
        // /*00c0*/                   PRMT R0, R0, 0x8880, RZ ;                /* 0x0000888000007816 */
        // /* 0x000fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_sat_s8_s32(int a) {
    int out;
    asm volatile("cvt.sat.s8.s32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x8880, RZ ;                    /* 0x0000888002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   MOV R7, R0 ;                                 /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                 /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_s8_s64(long long a) {
    int out;
    asm volatile("cvt.s8.s64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.MOV.U32 R6, RZ, RZ, 0x7fffffff ;       /* 0x7fffffffff067424 */
        // /* 0x000fe200078e00ff */
        // /*00b0*/                   SHF.R.S32.HI R9, RZ, 0x1f, R2 ;             /* 0x0000001fff097819 */
        // /* 0x004fe40000011402 */
        // /*00c0*/                   ISETP.GE.AND P0, PT, R3, RZ, PT ;           /* 0x000000ff0300720c */
        // /* 0x000fe40003f06270 */
        // /*00d0*/                   LOP3.LUT R0, R9, R3, RZ, 0x3c, !PT ;        /* 0x0000000309007212 */
        // /* 0x000fe400078e3cff */
        // /*00e0*/                   SEL R9, R6, 0x80000000, P0 ;                /* 0x8000000006097807 */
        // /* 0x000fe40000000000 */
        // /*00f0*/                   ISETP.EQ.AND P0, PT, R0, RZ, PT ;           /* 0x000000ff0000720c */
        // /* 0x000fc80003f02270 */
        // /*0100*/                   SEL R0, R2, R9, P0 ;                        /* 0x0000000902007207 */
        // /* 0x000fe20000000000 */
        // /*0110*/                   IMAD.WIDE R2, R7, 0x4, R4 ;                 /* 0x0000000407027825 */
        // /* 0x008fc600078e0204 */
        // /*0120*/                   I2I.S8.S32.SAT R0, R0 ;                     /* 0x0000000000007238 */
        // /* 0x000fc80000001000 */
        // /*0130*/                   PRMT R0, R0, 0x8880, RZ ;                   /* 0x0000888000007816 */
        // /* 0x000fca00000000ff */
        // /*0140*/                   IMAD.MOV.U32 R5, RZ, RZ, R0 ;               /* 0x000000ffff057224 */
        // /* 0x000fca00078e0000 */
        // /*0150*/                   STG.E desc[UR4][R2.64], R5 ;                /* 0x0000000502007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_sat_s8_s64(long long a) {
    int out;
    asm volatile("cvt.sat.s8.s64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x8880, RZ ;                /* 0x0000888002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_s8_u16(unsigned int a) {
    int out;
    asm volatile("cvt.s8.u16 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x7710, RZ ;                /* 0x0000771002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   VIMNMX.U32 R0, PT, PT, R0, 0x7f, PT ;    /* 0x0000007f00007848 */
        // /* 0x000fc80003fe0000 */
        // /*00d0*/                   PRMT R0, R0, 0x8880, RZ ;                /* 0x0000888000007816 */
        // /* 0x000fc800000000ff */
        // /*00e0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00f0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_sat_s8_u16(unsigned int a) {
    int out;
    asm volatile("cvt.sat.s8.u16 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x8880, RZ ;                /* 0x0000888002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_s8_u32(unsigned int a) {
    int out;
    asm volatile("cvt.s8.u32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   VIMNMX.U32 R0, PT, PT, R2, 0x7f, PT ;    /* 0x0000007f02007848 */
        // /* 0x004fc80003fe0000 */
        // /*00c0*/                   PRMT R0, R0, 0x8880, RZ ;                /* 0x0000888000007816 */
        // /* 0x000fc800000000ff */
        // /*00d0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_sat_s8_u32(unsigned int a) {
    int out;
    asm volatile("cvt.sat.s8.u32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x8880, RZ ;                    /* 0x0000888002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   MOV R7, R0 ;                                 /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                 /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_s8_u64(unsigned long long a) {
    int out;
    asm volatile("cvt.s8.u64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   ISETP.NE.U32.AND P0, PT, R3, RZ, PT ;       /* 0x000000ff0300720c */
        // /* 0x004fc80003f05070 */
        // /*00c0*/                   SEL R9, RZ, 0xffffffff, !P0 ;               /* 0xffffffffff097807 */
        // /* 0x000fc80004000000 */
        // /*00d0*/                   LOP3.LUT R0, R9, R2, RZ, 0xfc, !PT ;        /* 0x0000000209007212 */
        // /* 0x000fc800078efcff */
        // /*00e0*/                   VIMNMX.U32 R0, PT, PT, R0, 0x7f, PT ;       /* 0x0000007f00007848 */
        // /* 0x000fc80003fe0000 */
        // /*00f0*/                   PRMT R0, R0, 0x8880, RZ ;                   /* 0x0000888000007816 */
        // /* 0x000fca00000000ff */
        // /*0100*/                   IMAD.MOV.U32 R7, RZ, RZ, R0 ;               /* 0x000000ffff077224 */
        // /* 0x000fca00078e0000 */
        // /*0110*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_sat_s8_u64(unsigned long long a) {
    int out;
    asm volatile("cvt.sat.s8.u64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x8880, RZ ;                /* 0x0000888002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_s8_u8(unsigned int a) {
    int out;
    asm volatile("cvt.s8.u8 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x7770, RZ ;                /* 0x0000777002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   VIMNMX.U32 R0, PT, PT, R0, 0x7f, PT ;    /* 0x0000007f00007848 */
        // /* 0x000fc80003fe0000 */
        // /*00d0*/                   PRMT R0, R0, 0x8880, RZ ;                /* 0x0000888000007816 */
        // /* 0x000fc800000000ff */
        // /*00e0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00f0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int cvt_sat_s8_u8(unsigned int a) {
    int out;
    asm volatile("cvt.sat.s8.u8 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U16.F16.FLOOR.NTZ R9, R2 ;               /* 0x0000000200097305 */
        // /* 0x004e280000106800 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rmi_u16_f16(uint16_t a) {
    unsigned int out;
    asm volatile("cvt.rmi.u16.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U16.F16.NTZ R9, R2 ;                     /* 0x0000000200097305 */
        // /* 0x004e280000102800 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rni_u16_f16(uint16_t a) {
    unsigned int out;
    asm volatile("cvt.rni.u16.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U16.F16.CEIL.NTZ R9, R2 ;                /* 0x0000000200097305 */
        // /* 0x004e28000010a800 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rpi_u16_f16(uint16_t a) {
    unsigned int out;
    asm volatile("cvt.rpi.u16.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U16.F16.TRUNC.NTZ R9, R2 ;               /* 0x0000000200097305 */
        // /* 0x004e28000010e800 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rzi_u16_f16(uint16_t a) {
    unsigned int out;
    asm volatile("cvt.rzi.u16.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U16.FLOOR.NTZ R9, R2 ;               /* 0x0000000200097305 */
        // /* 0x004e280000206800 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rmi_u16_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rmi.u16.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;   /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;               /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.FTZ.U16.NTZ R0, R2 ;                  /* 0x0000000200007305 */
        // /* 0x004e240000212800 */
        // /*00c0*/                   LOP3.LUT R7, R0, 0xffff, RZ, 0xc0, !PT ;  /* 0x0000ffff00077812 */
        // /* 0x001fca00078ec0ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;              /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rni_ftz_sat_u16_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rni.ftz.sat.u16.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;   /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;               /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.FTZ.U16.NTZ R0, R2 ;                  /* 0x0000000200007305 */
        // /* 0x004e240000212800 */
        // /*00c0*/                   LOP3.LUT R7, R0, 0xffff, RZ, 0xc0, !PT ;  /* 0x0000ffff00077812 */
        // /* 0x001fca00078ec0ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;              /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rni_ftz_u16_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rni.ftz.u16.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U16.NTZ R9, R2 ;                     /* 0x0000000200097305 */
        // /* 0x004e280000202800 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rni_sat_u16_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rni.sat.u16.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U16.NTZ R9, R2 ;                     /* 0x0000000200097305 */
        // /* 0x004e280000202800 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rni_u16_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rni.u16.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U16.CEIL.NTZ R9, R2 ;                /* 0x0000000200097305 */
        // /* 0x004e28000020a800 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rpi_u16_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rpi.u16.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;   /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;               /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.FTZ.U16.TRUNC.NTZ R0, R2 ;            /* 0x0000000200007305 */
        // /* 0x004e24000021e800 */
        // /*00c0*/                   LOP3.LUT R7, R0, 0xffff, RZ, 0xc0, !PT ;  /* 0x0000ffff00077812 */
        // /* 0x001fca00078ec0ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;              /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rzi_ftz_sat_u16_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rzi.ftz.sat.u16.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;   /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;               /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.FTZ.U16.TRUNC.NTZ R0, R2 ;            /* 0x0000000200007305 */
        // /* 0x004e24000021e800 */
        // /*00c0*/                   LOP3.LUT R7, R0, 0xffff, RZ, 0xc0, !PT ;  /* 0x0000ffff00077812 */
        // /* 0x001fca00078ec0ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;              /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rzi_ftz_u16_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rzi.ftz.u16.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U16.TRUNC.NTZ R9, R2 ;               /* 0x0000000200097305 */
        // /* 0x004e28000020e800 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rzi_sat_u16_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rzi.sat.u16.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U16.TRUNC.NTZ R9, R2 ;               /* 0x0000000200097305 */
        // /* 0x004e28000020e800 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rzi_u16_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rzi.u16.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U16.F64.FLOOR R0, R2 ;                  /* 0x0000000200007311 */
        // /* 0x004e240000304800 */
        // /*00c0*/                   LOP3.LUT R7, R0, 0xffff, RZ, 0xc0, !PT ;    /* 0x0000ffff00077812 */
        // /* 0x001fca00078ec0ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rmi_u16_f64(double a) {
    unsigned int out;
    asm volatile("cvt.rmi.u16.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U16.F64 R0, R2 ;                        /* 0x0000000200007311 */
        // /* 0x004e240000300800 */
        // /*00c0*/                   LOP3.LUT R7, R0, 0xffff, RZ, 0xc0, !PT ;    /* 0x0000ffff00077812 */
        // /* 0x001fca00078ec0ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rni_sat_u16_f64(double a) {
    unsigned int out;
    asm volatile("cvt.rni.sat.u16.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U16.F64 R0, R2 ;                        /* 0x0000000200007311 */
        // /* 0x004e240000300800 */
        // /*00c0*/                   LOP3.LUT R7, R0, 0xffff, RZ, 0xc0, !PT ;    /* 0x0000ffff00077812 */
        // /* 0x001fca00078ec0ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rni_u16_f64(double a) {
    unsigned int out;
    asm volatile("cvt.rni.u16.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U16.F64.CEIL R0, R2 ;                   /* 0x0000000200007311 */
        // /* 0x004e240000308800 */
        // /*00c0*/                   LOP3.LUT R7, R0, 0xffff, RZ, 0xc0, !PT ;    /* 0x0000ffff00077812 */
        // /* 0x001fca00078ec0ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rpi_u16_f64(double a) {
    unsigned int out;
    asm volatile("cvt.rpi.u16.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U16.F64.TRUNC R0, R2 ;                  /* 0x0000000200007311 */
        // /* 0x004e24000030c800 */
        // /*00c0*/                   LOP3.LUT R7, R0, 0xffff, RZ, 0xc0, !PT ;    /* 0x0000ffff00077812 */
        // /* 0x001fca00078ec0ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rzi_sat_u16_f64(double a) {
    unsigned int out;
    asm volatile("cvt.rzi.sat.u16.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U16.F64.TRUNC R0, R2 ;                  /* 0x0000000200007311 */
        // /* 0x004e24000030c800 */
        // /*00c0*/                   LOP3.LUT R7, R0, 0xffff, RZ, 0xc0, !PT ;    /* 0x0000ffff00077812 */
        // /* 0x001fca00078ec0ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rzi_u16_f64(double a) {
    unsigned int out;
    asm volatile("cvt.rzi.u16.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;      /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x9910, RZ ;                    /* 0x0000991002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   ISETP.LT.AND P0, PT, R0.reuse, RZ, PT ;      /* 0x000000ff0000720c */
        // /* 0x040fe40003f01270 */
        // /*00d0*/                   ISETP.GT.AND P1, PT, R0.reuse, 0xffff, PT ;  /* 0x0000ffff0000780c */
        // /* 0x040fe40003f24270 */
        // /*00e0*/                   SEL R0, R0, RZ, !P0 ;                        /* 0x000000ff00007207 */
        // /* 0x000fc80004000000 */
        // /*00f0*/                   SEL R7, R0, 0xffff, !P1 ;                    /* 0x0000ffff00077807 */
        // /* 0x000fca0004800000 */
        // /*0100*/                   STG.E desc[UR4][R4.64], R7 ;                 /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_sat_u16_s16(int a) {
    unsigned int out;
    asm volatile("cvt.sat.u16.s16 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;   /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;               /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   LOP3.LUT R7, R2, 0xffff, RZ, 0xc0, !PT ;  /* 0x0000ffff02077812 */
        // /* 0x004fca00078ec0ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;              /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_u16_s16(int a) {
    unsigned int out;
    asm volatile("cvt.u16.s16 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2I.U16.S32.SAT R7, R2 ;                 /* 0x0000000200077238 */
        // /* 0x004fca0000002000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_sat_u16_s32(int a) {
    unsigned int out;
    asm volatile("cvt.sat.u16.s32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;   /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;               /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   LOP3.LUT R7, R2, 0xffff, RZ, 0xc0, !PT ;  /* 0x0000ffff02077812 */
        // /* 0x004fca00078ec0ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;              /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_u16_s32(int a) {
    unsigned int out;
    asm volatile("cvt.u16.s32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;   /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   ISETP.NE.U32.AND P0, PT, R3.reuse, RZ, PT ;  /* 0x000000ff0300720c */
        // /* 0x044fe40003f05070 */
        // /*00c0*/                   ISETP.GE.AND P1, PT, R3, RZ, PT ;            /* 0x000000ff0300720c */
        // /* 0x000fe40003f26270 */
        // /*00d0*/                   SEL R9, RZ, 0xffffffff, !P0 ;                /* 0xffffffffff097807 */
        // /* 0x000fe40004000000 */
        // /*00e0*/                   SEL R0, RZ, 0xffffffff, !P1 ;                /* 0xffffffffff007807 */
        // /* 0x000fc80004800000 */
        // /*00f0*/                   LOP3.LUT R0, R0, R9, R2, 0xe0, !PT ;         /* 0x0000000900007212 */
        // /* 0x000fc800078ee002 */
        // /*0100*/                   VIMNMX.U32 R7, PT, PT, R0, 0xffff, PT ;      /* 0x0000ffff00077848 */
        // /* 0x000fca0003fe0000 */
        // /*0110*/                   STG.E desc[UR4][R4.64], R7 ;                 /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_sat_u16_s64(long long a) {
    unsigned int out;
    asm volatile("cvt.sat.u16.s64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   LOP3.LUT R7, R2, 0xffff, RZ, 0xc0, !PT ;     /* 0x0000ffff02077812 */
        // /* 0x004fca00078ec0ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                 /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_u16_s64(long long a) {
    unsigned int out;
    asm volatile("cvt.u16.s64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x8880, RZ ;                /* 0x0000888002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   PRMT R7, R0, 0x7710, RZ ;                /* 0x0000771000077816 */
        // /* 0x000fca00000000ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_u16_s8(int a) {
    unsigned int out;
    asm volatile("cvt.u16.s8 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   VIMNMX.U32 R7, PT, PT, R2, 0xffff, PT ;  /* 0x0000ffff02077848 */
        // /* 0x004fca0003fe0000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_sat_u16_u32(unsigned int a) {
    unsigned int out;
    asm volatile("cvt.sat.u16.u32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;   /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;               /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   LOP3.LUT R7, R2, 0xffff, RZ, 0xc0, !PT ;  /* 0x0000ffff02077812 */
        // /* 0x004fca00078ec0ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;              /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_u16_u32(unsigned int a) {
    unsigned int out;
    asm volatile("cvt.u16.u32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   ISETP.NE.U32.AND P0, PT, R3, RZ, PT ;       /* 0x000000ff0300720c */
        // /* 0x004fc80003f05070 */
        // /*00c0*/                   SEL R9, RZ, 0xffffffff, !P0 ;               /* 0xffffffffff097807 */
        // /* 0x000fc80004000000 */
        // /*00d0*/                   LOP3.LUT R9, R9, R2, RZ, 0xfc, !PT ;        /* 0x0000000209097212 */
        // /* 0x000fc800078efcff */
        // /*00e0*/                   VIMNMX.U32 R7, PT, PT, R9, 0xffff, PT ;     /* 0x0000ffff09077848 */
        // /* 0x000fca0003fe0000 */
        // /*00f0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_sat_u16_u64(unsigned long long a) {
    unsigned int out;
    asm volatile("cvt.sat.u16.u64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   LOP3.LUT R7, R2, 0xffff, RZ, 0xc0, !PT ;     /* 0x0000ffff02077812 */
        // /* 0x004fca00078ec0ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                 /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_u16_u64(unsigned long long a) {
    unsigned int out;
    asm volatile("cvt.u16.u64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   LOP3.LUT R7, R2, 0xff, RZ, 0xc0, !PT ;   /* 0x000000ff02077812 */
        // /* 0x004fca00078ec0ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_u16_u8(unsigned int a) {
    unsigned int out;
    asm volatile("cvt.u16.u8 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U32.BF16.FLOOR.NTZ R9, R2 ;              /* 0x0000000200097305 */
        // /* 0x004e280000407000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rmi_u32_bf16(uint16_t a) {
    unsigned int out;
    asm volatile("cvt.rmi.u32.bf16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U32.BF16.NTZ R9, R2 ;                    /* 0x0000000200097305 */
        // /* 0x004e280000403000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rni_u32_bf16(uint16_t a) {
    unsigned int out;
    asm volatile("cvt.rni.u32.bf16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U32.BF16.CEIL.NTZ R9, R2 ;               /* 0x0000000200097305 */
        // /* 0x004e28000040b000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rpi_u32_bf16(uint16_t a) {
    unsigned int out;
    asm volatile("cvt.rpi.u32.bf16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U32.BF16.TRUNC.NTZ R9, R2 ;              /* 0x0000000200097305 */
        // /* 0x004e28000040f000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rzi_u32_bf16(uint16_t a) {
    unsigned int out;
    asm volatile("cvt.rzi.u32.bf16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U32.F16.FLOOR.NTZ R9, R2 ;               /* 0x0000000200097305 */
        // /* 0x004e280000107000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rmi_u32_f16(uint16_t a) {
    unsigned int out;
    asm volatile("cvt.rmi.u32.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U32.F16.NTZ R9, R2 ;                     /* 0x0000000200097305 */
        // /* 0x004e280000103000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rni_u32_f16(uint16_t a) {
    unsigned int out;
    asm volatile("cvt.rni.u32.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U32.F16.CEIL.NTZ R9, R2 ;                /* 0x0000000200097305 */
        // /* 0x004e28000010b000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rpi_u32_f16(uint16_t a) {
    unsigned int out;
    asm volatile("cvt.rpi.u32.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U32.F16.TRUNC.NTZ R9, R2 ;               /* 0x0000000200097305 */
        // /* 0x004e28000010f000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rzi_u32_f16(uint16_t a) {
    unsigned int out;
    asm volatile("cvt.rzi.u32.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U32.FLOOR.NTZ R9, R2 ;               /* 0x0000000200097305 */
        // /* 0x004e280000207000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rmi_u32_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rmi.u32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.FTZ.U32.NTZ R9, R2 ;                 /* 0x0000000200097305 */
        // /* 0x004e280000213000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rni_ftz_u32_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rni.ftz.u32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U32.NTZ R9, R2 ;                     /* 0x0000000200097305 */
        // /* 0x004e280000203000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rni_u32_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rni.u32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U32.CEIL.NTZ R9, R2 ;                /* 0x0000000200097305 */
        // /* 0x004e28000020b000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rpi_u32_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rpi.u32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.FTZ.U32.TRUNC.NTZ R9, R2 ;           /* 0x0000000200097305 */
        // /* 0x004e28000021f000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rzi_ftz_u32_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rzi.ftz.u32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U32.TRUNC.NTZ R9, R2 ;               /* 0x0000000200097305 */
        // /* 0x004e28000020f000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rzi_u32_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rzi.u32.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   F2I.U32.F64.FLOOR R7, R2 ;                  /* 0x0000000200077311 */
        // /* 0x004e260000305000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rmi_u32_f64(double a) {
    unsigned int out;
    asm volatile("cvt.rmi.u32.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   F2I.U32.F64 R7, R2 ;                        /* 0x0000000200077311 */
        // /* 0x004e260000301000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rni_u32_f64(double a) {
    unsigned int out;
    asm volatile("cvt.rni.u32.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   F2I.U32.F64.CEIL R7, R2 ;                   /* 0x0000000200077311 */
        // /* 0x004e260000309000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rpi_u32_f64(double a) {
    unsigned int out;
    asm volatile("cvt.rpi.u32.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe400078e0204 */
        // /*00b0*/                   F2I.U32.F64.TRUNC R7, R2 ;                  /* 0x0000000200077311 */
        // /* 0x004e26000030d000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rzi_u32_f64(double a) {
    unsigned int out;
    asm volatile("cvt.rzi.u32.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x9910, RZ ;                /* 0x0000991002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_u32_s16(int a) {
    unsigned int out;
    asm volatile("cvt.u32.s16 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   ISETP.LT.AND P0, PT, R2, RZ, PT ;        /* 0x000000ff0200720c */
        // /* 0x004fc80003f01270 */
        // /*00c0*/                   SEL R7, R2, RZ, !P0 ;                    /* 0x000000ff02077207 */
        // /* 0x000fca0004000000 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_sat_u32_s32(int a) {
    unsigned int out;
    asm volatile("cvt.sat.u32.s32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;             /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_u32_s32(int a) {
    unsigned int out;
    asm volatile("cvt.u32.s32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;   /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   ISETP.NE.U32.AND P0, PT, R3.reuse, RZ, PT ;  /* 0x000000ff0300720c */
        // /* 0x044fe40003f05070 */
        // /*00c0*/                   ISETP.GE.AND P1, PT, R3, RZ, PT ;            /* 0x000000ff0300720c */
        // /* 0x000fe40003f26270 */
        // /*00d0*/                   SEL R9, RZ, 0xffffffff, !P0 ;                /* 0xffffffffff097807 */
        // /* 0x000fe40004000000 */
        // /*00e0*/                   SEL R0, RZ, 0xffffffff, !P1 ;                /* 0xffffffffff007807 */
        // /* 0x000fc80004800000 */
        // /*00f0*/                   LOP3.LUT R9, R0, R9, R2, 0xe0, !PT ;         /* 0x0000000900097212 */
        // /* 0x000fca00078ee002 */
        // /*0100*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_sat_u32_s64(long long a) {
    unsigned int out;
    asm volatile("cvt.sat.u32.s64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;             /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_u32_s64(long long a) {
    unsigned int out;
    asm volatile("cvt.u32.s64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x8880, RZ ;                /* 0x0000888002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   MOV R7, R0 ;                             /* 0x0000000000077202 */
        // /* 0x000fca0000000f00 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_u32_s8(int a) {
    unsigned int out;
    asm volatile("cvt.u32.s8 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;   /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;               /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   LOP3.LUT R7, R2, 0xffff, RZ, 0xc0, !PT ;  /* 0x0000ffff02077812 */
        // /* 0x004fca00078ec0ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;              /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_u32_u16(unsigned int a) {
    unsigned int out;
    asm volatile("cvt.u32.u16 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   ISETP.NE.U32.AND P0, PT, R3, RZ, PT ;       /* 0x000000ff0300720c */
        // /* 0x004fc80003f05070 */
        // /*00c0*/                   SEL R9, RZ, 0xffffffff, !P0 ;               /* 0xffffffffff097807 */
        // /* 0x000fc80004000000 */
        // /*00d0*/                   LOP3.LUT R9, R9, R2, RZ, 0xfc, !PT ;        /* 0x0000000209097212 */
        // /* 0x000fca00078efcff */
        // /*00e0*/                   STG.E desc[UR4][R4.64], R9 ;                /* 0x0000000904007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_sat_u32_u64(unsigned long long a) {
    unsigned int out;
    asm volatile("cvt.sat.u32.u64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;             /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_u32_u64(unsigned long long a) {
    unsigned int out;
    asm volatile("cvt.u32.u64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   LOP3.LUT R7, R2, 0xff, RZ, 0xc0, !PT ;   /* 0x000000ff02077812 */
        // /* 0x004fca00078ec0ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_u32_u8(unsigned int a) {
    unsigned int out;
    asm volatile("cvt.u32.u8 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                  /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.U64.BF16.FLOOR R4, R2 ;                  /* 0x0000000200047311 */
        // /* 0x004e260000405800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;              /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_rmi_u64_bf16(uint16_t a) {
    unsigned long long out;
    asm volatile("cvt.rmi.u64.bf16 %0, %1;" : "=l"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                  /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.U64.BF16 R4, R2 ;                        /* 0x0000000200047311 */
        // /* 0x004e260000401800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;              /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_rni_u64_bf16(uint16_t a) {
    unsigned long long out;
    asm volatile("cvt.rni.u64.bf16 %0, %1;" : "=l"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                  /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.U64.BF16.CEIL R4, R2 ;                   /* 0x0000000200047311 */
        // /* 0x004e260000409800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;              /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_rpi_u64_bf16(uint16_t a) {
    unsigned long long out;
    asm volatile("cvt.rpi.u64.bf16 %0, %1;" : "=l"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                  /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.U64.BF16.TRUNC R4, R2 ;                  /* 0x0000000200047311 */
        // /* 0x004e26000040d800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;              /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_rzi_u64_bf16(uint16_t a) {
    unsigned long long out;
    asm volatile("cvt.rzi.u64.bf16 %0, %1;" : "=l"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                  /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.U64.F16.FLOOR R4, R2 ;                   /* 0x0000000200047311 */
        // /* 0x004e260000105800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;              /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_rmi_u64_f16(uint16_t a) {
    unsigned long long out;
    asm volatile("cvt.rmi.u64.f16 %0, %1;" : "=l"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                  /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.U64.F16 R4, R2 ;                         /* 0x0000000200047311 */
        // /* 0x004e260000101800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;              /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_rni_u64_f16(uint16_t a) {
    unsigned long long out;
    asm volatile("cvt.rni.u64.f16 %0, %1;" : "=l"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                  /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.U64.F16.CEIL R4, R2 ;                    /* 0x0000000200047311 */
        // /* 0x004e260000109800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;              /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_rpi_u64_f16(uint16_t a) {
    unsigned long long out;
    asm volatile("cvt.rpi.u64.f16 %0, %1;" : "=l"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                  /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.U64.F16.TRUNC R4, R2 ;                   /* 0x0000000200047311 */
        // /* 0x004e26000010d800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;              /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_rzi_u64_f16(uint16_t a) {
    unsigned long long out;
    asm volatile("cvt.rzi.u64.f16 %0, %1;" : "=l"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.U64.FLOOR R4, R2 ;                   /* 0x0000000200047311 */
        // /* 0x004e260000205800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_rmi_u64_f32(float a) {
    unsigned long long out;
    asm volatile("cvt.rmi.u64.f32 %0, %1;" : "=l"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.U64 R4, R2 ;                         /* 0x0000000200047311 */
        // /* 0x004e260000201800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_rni_ftz_u64_f32(float a) {
    unsigned long long out;
    asm volatile("cvt.rni.ftz.u64.f32 %0, %1;" : "=l"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.U64 R4, R2 ;                         /* 0x0000000200047311 */
        // /* 0x004e260000201800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_rni_u64_f32(float a) {
    unsigned long long out;
    asm volatile("cvt.rni.u64.f32 %0, %1;" : "=l"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.U64.CEIL R4, R2 ;                    /* 0x0000000200047311 */
        // /* 0x004e260000209800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_rpi_u64_f32(float a) {
    unsigned long long out;
    asm volatile("cvt.rpi.u64.f32 %0, %1;" : "=l"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.U64.TRUNC R4, R2 ;                   /* 0x0000000200047311 */
        // /* 0x004e26000020d800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_rzi_ftz_u64_f32(float a) {
    unsigned long long out;
    asm volatile("cvt.rzi.ftz.u64.f32 %0, %1;" : "=l"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.U64.TRUNC R4, R2 ;                   /* 0x0000000200047311 */
        // /* 0x004e26000020d800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_rzi_u64_f32(float a) {
    unsigned long long out;
    asm volatile("cvt.rzi.u64.f32 %0, %1;" : "=l"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                 /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.U64.F64.FLOOR R4, R2 ;                  /* 0x0000000200047311 */
        // /* 0x004e260000305800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;             /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_rmi_u64_f64(double a) {
    unsigned long long out;
    asm volatile("cvt.rmi.u64.f64 %0, %1;" : "=l"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                 /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.U64.F64 R4, R2 ;                        /* 0x0000000200047311 */
        // /* 0x004e260000301800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;             /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_rni_u64_f64(double a) {
    unsigned long long out;
    asm volatile("cvt.rni.u64.f64 %0, %1;" : "=l"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                 /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.U64.F64.CEIL R4, R2 ;                   /* 0x0000000200047311 */
        // /* 0x004e260000309800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;             /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_rpi_u64_f64(double a) {
    unsigned long long out;
    asm volatile("cvt.rpi.u64.f64 %0, %1;" : "=l"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;                 /* 0x0000000805067825 */
        // /* 0x008fe400078e0206 */
        // /*00b0*/                   F2I.U64.F64.TRUNC R4, R2 ;                  /* 0x0000000200047311 */
        // /* 0x004e26000030d800 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;             /* 0x0000000406007986 */
        // /* 0x001fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_rzi_u64_f64(double a) {
    unsigned long long out;
    asm volatile("cvt.rzi.u64.f64 %0, %1;" : "=l"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R9, 0x8, R6 ;              /* 0x0000000809067825 */
        // /* 0x008fe200078e0206 */
        // /*00b0*/                   PRMT R4, R2, 0x9910, RZ ;                /* 0x0000991002047816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   SHF.R.S32.HI R5, RZ, 0x1f, R4 ;          /* 0x0000001fff057819 */
        // /* 0x000fca0000011404 */
        // /*00d0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_u64_s16(int a) {
    unsigned long long out;
    asm volatile("cvt.u64.s16 %0, %1;" : "=l"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R4, desc[UR4][R2.64] ;  /* 0x0000000402047981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R9, 0x8, R6 ;              /* 0x0000000809067825 */
        // /* 0x008fe200078e0206 */
        // /*00b0*/                   SHF.R.S32.HI R5, RZ, 0x1f, R4 ;          /* 0x0000001fff057819 */
        // /* 0x004fca0000011404 */
        // /*00c0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_u64_s32(int a) {
    unsigned long long out;
    asm volatile("cvt.u64.s32 %0, %1;" : "=l"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R6, R9, 0x8, R6 ;                 /* 0x0000000809067825 */
        // /* 0x008fe200078e0206 */
        // /*00b0*/                   ISETP.GE.AND P0, PT, R3.reuse, RZ, PT ;     /* 0x000000ff0300720c */
        // /* 0x044fe40003f06270 */
        // /*00c0*/                   ISETP.LT.AND P1, PT, R3.reuse, RZ, PT ;     /* 0x000000ff0300720c */
        // /* 0x040fe40003f21270 */
        // /*00d0*/                   SEL R11, RZ, 0xffffffff, !P0 ;              /* 0xffffffffff0b7807 */
        // /* 0x000fe40004000000 */
        // /*00e0*/                   SEL R5, R3, RZ, !P1 ;                       /* 0x000000ff03057207 */
        // /* 0x000fe40004800000 */
        // /*00f0*/                   LOP3.LUT R4, R11, R2, RZ, 0xc0, !PT ;       /* 0x000000020b047212 */
        // /* 0x000fca00078ec0ff */
        // /*0100*/                   STG.E.64 desc[UR4][R6.64], R4 ;             /* 0x0000000406007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_sat_u64_s64(long long a) {
    unsigned long long out;
    asm volatile("cvt.sat.u64.s64 %0, %1;" : "=l"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x8, R4 ;                 /* 0x0000000807047825 */
        // /* 0x008fca00078e0204 */
        // /*00b0*/                   STG.E.64 desc[UR4][R4.64], R2 ;             /* 0x0000000204007986 */
        // /* 0x004fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_u64_s64(long long a) {
    unsigned long long out;
    asm volatile("cvt.u64.s64 %0, %1;" : "=l"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R9, 0x8, R6 ;              /* 0x0000000809067825 */
        // /* 0x008fe200078e0206 */
        // /*00b0*/                   PRMT R4, R2, 0x8880, RZ ;                /* 0x0000888002047816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   SHF.R.S32.HI R5, RZ, 0x1f, R4 ;          /* 0x0000001fff057819 */
        // /* 0x000fca0000011404 */
        // /*00d0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_u64_s8(int a) {
    unsigned long long out;
    asm volatile("cvt.u64.s8 %0, %1;" : "=l"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;   /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;               /* 0x0000000805067825 */
        // /* 0x008fc800078e0206 */
        // /*00b0*/                   HFMA2 R5, -RZ, RZ, 0, 0 ;                 /* 0x00000000ff057431 */
        // /* 0x000fe200000001ff */
        // /*00c0*/                   LOP3.LUT R4, R2, 0xffff, RZ, 0xc0, !PT ;  /* 0x0000ffff02047812 */
        // /* 0x004fca00078ec0ff */
        // /*00d0*/                   STG.E.64 desc[UR4][R6.64], R4 ;           /* 0x0000000406007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_u64_u16(unsigned int a) {
    unsigned long long out;
    asm volatile("cvt.u64.u16 %0, %1;" : "=l"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x0020a2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x8, R4 ;              /* 0x0000000807047825 */
        // /* 0x008fc800078e0204 */
        // /*00b0*/                   HFMA2 R3, -RZ, RZ, 0, 0 ;                /* 0x00000000ff037431 */
        // /* 0x001fca00000001ff */
        // /*00c0*/                   STG.E.64 desc[UR4][R4.64], R2 ;          /* 0x0000000204007986 */
        // /* 0x004fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_u64_u32(unsigned int a) {
    unsigned long long out;
    asm volatile("cvt.u64.u32 %0, %1;" : "=l"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R6, R5, 0x8, R6 ;              /* 0x0000000805067825 */
        // /* 0x008fc800078e0206 */
        // /*00b0*/                   HFMA2 R5, -RZ, RZ, 0, 0 ;                /* 0x00000000ff057431 */
        // /* 0x000fe200000001ff */
        // /*00c0*/                   LOP3.LUT R4, R2, 0xff, RZ, 0xc0, !PT ;   /* 0x000000ff02047812 */
        // /* 0x004fca00078ec0ff */
        // /*00d0*/                   STG.E.64 desc[UR4][R6.64], R4 ;          /* 0x0000000406007986 */
        // /* 0x000fe2000c101b04 */
__device__ __forceinline__ unsigned long long cvt_u64_u8(unsigned int a) {
    unsigned long long out;
    asm volatile("cvt.u64.u8 %0, %1;" : "=l"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U8.F16.FLOOR.NTZ R9, R2 ;                /* 0x0000000200097305 */
        // /* 0x004e280000106000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rmi_u8_f16(uint16_t a) {
    unsigned int out;
    asm volatile("cvt.rmi.u8.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U8.F16.NTZ R9, R2 ;                      /* 0x0000000200097305 */
        // /* 0x004e280000102000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rni_u8_f16(uint16_t a) {
    unsigned int out;
    asm volatile("cvt.rni.u8.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U8.F16.CEIL.NTZ R9, R2 ;                 /* 0x0000000200097305 */
        // /* 0x004e28000010a000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rpi_u8_f16(uint16_t a) {
    unsigned int out;
    asm volatile("cvt.rpi.u8.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U8.F16.TRUNC.NTZ R9, R2 ;                /* 0x0000000200097305 */
        // /* 0x004e28000010e000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;                 /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rzi_u8_f16(uint16_t a) {
    unsigned int out;
    asm volatile("cvt.rzi.u8.f16 %0, %1;" : "=r"(out) : "h"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U8.FLOOR.NTZ R9, R2 ;                /* 0x0000000200097305 */
        // /* 0x004e280000206000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rmi_u8_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rmi.u8.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.FTZ.U8.NTZ R0, R2 ;                  /* 0x0000000200007305 */
        // /* 0x004e240000212000 */
        // /*00c0*/                   LOP3.LUT R7, R0, 0xff, RZ, 0xc0, !PT ;   /* 0x000000ff00077812 */
        // /* 0x001fca00078ec0ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rni_ftz_sat_u8_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rni.ftz.sat.u8.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.FTZ.U8.NTZ R0, R2 ;                  /* 0x0000000200007305 */
        // /* 0x004e240000212000 */
        // /*00c0*/                   LOP3.LUT R7, R0, 0xff, RZ, 0xc0, !PT ;   /* 0x000000ff00077812 */
        // /* 0x001fca00078ec0ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rni_ftz_u8_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rni.ftz.u8.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2IP.U8.F32.NTZ R7, RZ, R2, RZ ;         /* 0x00000002ff077243 */
        // /* 0x004fca00000004ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rni_sat_u8_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2IP.U8.F32.NTZ R7, RZ, R2, RZ ;         /* 0x00000002ff077243 */
        // /* 0x004fca00000004ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rni_u8_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rni.u8.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U8.CEIL.NTZ R9, R2 ;                 /* 0x0000000200097305 */
        // /* 0x004e28000020a000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rpi_u8_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rpi.u8.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.FTZ.U8.TRUNC.NTZ R0, R2 ;            /* 0x0000000200007305 */
        // /* 0x004e24000021e000 */
        // /*00c0*/                   LOP3.LUT R7, R0, 0xff, RZ, 0xc0, !PT ;   /* 0x000000ff00077812 */
        // /* 0x001fca00078ec0ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rzi_ftz_sat_u8_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rzi.ftz.sat.u8.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.FTZ.U8.TRUNC.NTZ R0, R2 ;            /* 0x0000000200007305 */
        // /* 0x004e24000021e000 */
        // /*00c0*/                   LOP3.LUT R7, R0, 0xff, RZ, 0xc0, !PT ;   /* 0x000000ff00077812 */
        // /* 0x001fca00078ec0ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rzi_ftz_u8_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rzi.ftz.u8.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2IP.U8.F32.TRUNC.NTZ R7, RZ, R2, RZ ;   /* 0x00000002ff077243 */
        // /* 0x004fca000000c4ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rzi_sat_u8_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rzi.sat.u8.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2IP.U8.F32.TRUNC.NTZ R7, RZ, R2, RZ ;   /* 0x00000002ff077243 */
        // /* 0x004fca000000c4ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rzi_u8_f32(float a) {
    unsigned int out;
    asm volatile("cvt.rzi.u8.f32 %0, %1;" : "=r"(out) : "f"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U8.F64.FLOOR R0, R2 ;                   /* 0x0000000200007311 */
        // /* 0x004e240000304000 */
        // /*00c0*/                   LOP3.LUT R7, R0, 0xff, RZ, 0xc0, !PT ;      /* 0x000000ff00077812 */
        // /* 0x001fca00078ec0ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rmi_u8_f64(double a) {
    unsigned int out;
    asm volatile("cvt.rmi.u8.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U8.F64 R0, R2 ;                         /* 0x0000000200007311 */
        // /* 0x004e240000300000 */
        // /*00c0*/                   LOP3.LUT R7, R0, 0xff, RZ, 0xc0, !PT ;      /* 0x000000ff00077812 */
        // /* 0x001fca00078ec0ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rni_sat_u8_f64(double a) {
    unsigned int out;
    asm volatile("cvt.rni.sat.u8.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U8.F64 R0, R2 ;                         /* 0x0000000200007311 */
        // /* 0x004e240000300000 */
        // /*00c0*/                   LOP3.LUT R7, R0, 0xff, RZ, 0xc0, !PT ;      /* 0x000000ff00077812 */
        // /* 0x001fca00078ec0ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rni_u8_f64(double a) {
    unsigned int out;
    asm volatile("cvt.rni.u8.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U8.F64.CEIL R0, R2 ;                    /* 0x0000000200007311 */
        // /* 0x004e240000308000 */
        // /*00c0*/                   LOP3.LUT R7, R0, 0xff, RZ, 0xc0, !PT ;      /* 0x000000ff00077812 */
        // /* 0x001fca00078ec0ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rpi_u8_f64(double a) {
    unsigned int out;
    asm volatile("cvt.rpi.u8.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U8.F64.TRUNC R0, R2 ;                   /* 0x0000000200007311 */
        // /* 0x004e24000030c000 */
        // /*00c0*/                   LOP3.LUT R7, R0, 0xff, RZ, 0xc0, !PT ;      /* 0x000000ff00077812 */
        // /* 0x001fca00078ec0ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rzi_sat_u8_f64(double a) {
    unsigned int out;
    asm volatile("cvt.rzi.sat.u8.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   F2I.U8.F64.TRUNC R0, R2 ;                   /* 0x0000000200007311 */
        // /* 0x004e24000030c000 */
        // /*00c0*/                   LOP3.LUT R7, R0, 0xff, RZ, 0xc0, !PT ;      /* 0x000000ff00077812 */
        // /* 0x001fca00078ec0ff */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_rzi_u8_f64(double a) {
    unsigned int out;
    asm volatile("cvt.rzi.u8.f64 %0, %1;" : "=r"(out) : "d"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;    /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x9910, RZ ;                  /* 0x0000991002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   ISETP.LT.AND P0, PT, R0.reuse, RZ, PT ;    /* 0x000000ff0000720c */
        // /* 0x040fe40003f01270 */
        // /*00d0*/                   ISETP.GT.AND P1, PT, R0.reuse, 0xff, PT ;  /* 0x000000ff0000780c */
        // /* 0x040fe40003f24270 */
        // /*00e0*/                   SEL R0, R0, RZ, !P0 ;                      /* 0x000000ff00007207 */
        // /* 0x000fc80004000000 */
        // /*00f0*/                   SEL R7, R0, 0xff, !P1 ;                    /* 0x000000ff00077807 */
        // /* 0x000fca0004800000 */
        // /*0100*/                   STG.E desc[UR4][R4.64], R7 ;               /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_sat_u8_s16(int a) {
    unsigned int out;
    asm volatile("cvt.sat.u8.s16 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   LOP3.LUT R7, R2, 0xff, RZ, 0xc0, !PT ;   /* 0x000000ff02077812 */
        // /* 0x004fca00078ec0ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_u8_s16(int a) {
    unsigned int out;
    asm volatile("cvt.u8.s16 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   I2I.U8.S32.SAT R7, R2 ;                  /* 0x0000000200077238 */
        // /* 0x004fca0000000000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_sat_u8_s32(int a) {
    unsigned int out;
    asm volatile("cvt.sat.u8.s32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   LOP3.LUT R7, R2, 0xff, RZ, 0xc0, !PT ;   /* 0x000000ff02077812 */
        // /* 0x004fca00078ec0ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_u8_s32(int a) {
    unsigned int out;
    asm volatile("cvt.u8.s32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;   /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   ISETP.NE.U32.AND P0, PT, R3.reuse, RZ, PT ;  /* 0x000000ff0300720c */
        // /* 0x044fe40003f05070 */
        // /*00c0*/                   ISETP.GE.AND P1, PT, R3, RZ, PT ;            /* 0x000000ff0300720c */
        // /* 0x000fe40003f26270 */
        // /*00d0*/                   SEL R9, RZ, 0xffffffff, !P0 ;                /* 0xffffffffff097807 */
        // /* 0x000fe40004000000 */
        // /*00e0*/                   SEL R0, RZ, 0xffffffff, !P1 ;                /* 0xffffffffff007807 */
        // /* 0x000fc80004800000 */
        // /*00f0*/                   LOP3.LUT R0, R0, R9, R2, 0xe0, !PT ;         /* 0x0000000900007212 */
        // /* 0x000fc800078ee002 */
        // /*0100*/                   VIMNMX.U32 R7, PT, PT, R0, 0xff, PT ;        /* 0x000000ff00077848 */
        // /* 0x000fca0003fe0000 */
        // /*0110*/                   STG.E desc[UR4][R4.64], R7 ;                 /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_sat_u8_s64(long long a) {
    unsigned int out;
    asm volatile("cvt.sat.u8.s64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   LOP3.LUT R7, R2, 0xff, RZ, 0xc0, !PT ;       /* 0x000000ff02077812 */
        // /* 0x004fca00078ec0ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                 /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_u8_s64(long long a) {
    unsigned int out;
    asm volatile("cvt.u8.s64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;    /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x8880, RZ ;                  /* 0x0000888002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   ISETP.LT.AND P0, PT, R0.reuse, RZ, PT ;    /* 0x000000ff0000720c */
        // /* 0x040fe40003f01270 */
        // /*00d0*/                   ISETP.GT.AND P1, PT, R0.reuse, 0xff, PT ;  /* 0x000000ff0000780c */
        // /* 0x040fe40003f24270 */
        // /*00e0*/                   SEL R0, R0, RZ, !P0 ;                      /* 0x000000ff00007207 */
        // /* 0x000fc80004000000 */
        // /*00f0*/                   SEL R7, R0, 0xff, !P1 ;                    /* 0x000000ff00077807 */
        // /* 0x000fca0004800000 */
        // /*0100*/                   STG.E desc[UR4][R4.64], R7 ;               /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_sat_u8_s8(int a) {
    unsigned int out;
    asm volatile("cvt.sat.u8.s8 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   LOP3.LUT R7, R2, 0xff, RZ, 0xc0, !PT ;   /* 0x000000ff02077812 */
        // /* 0x004fca00078ec0ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_u8_s8(int a) {
    unsigned int out;
    asm volatile("cvt.u8.s8 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   PRMT R0, R2, 0x7710, RZ ;                /* 0x0000771002007816 */
        // /* 0x004fc800000000ff */
        // /*00c0*/                   VIMNMX.U32 R7, PT, PT, R0, 0xff, PT ;    /* 0x000000ff00077848 */
        // /* 0x000fca0003fe0000 */
        // /*00d0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_sat_u8_u16(unsigned int a) {
    unsigned int out;
    asm volatile("cvt.sat.u8.u16 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   LOP3.LUT R7, R2, 0xff, RZ, 0xc0, !PT ;   /* 0x000000ff02077812 */
        // /* 0x004fca00078ec0ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_u8_u16(unsigned int a) {
    unsigned int out;
    asm volatile("cvt.u8.u16 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   VIMNMX.U32 R7, PT, PT, R2, 0xff, PT ;    /* 0x000000ff02077848 */
        // /* 0x004fca0003fe0000 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_sat_u8_u32(unsigned int a) {
    unsigned int out;
    asm volatile("cvt.sat.u8.u32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   LOP3.LUT R7, R2, 0xff, RZ, 0xc0, !PT ;   /* 0x000000ff02077812 */
        // /* 0x004fca00078ec0ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_u8_u32(unsigned int a) {
    unsigned int out;
    asm volatile("cvt.u8.u32 %0, %1;" : "=r"(out) : "r"(a));
    return out;
}

        // /*0090*/                   LDG.E.64.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5b00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                 /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   ISETP.NE.U32.AND P0, PT, R3, RZ, PT ;       /* 0x000000ff0300720c */
        // /* 0x004fc80003f05070 */
        // /*00c0*/                   SEL R9, RZ, 0xffffffff, !P0 ;               /* 0xffffffffff097807 */
        // /* 0x000fc80004000000 */
        // /*00d0*/                   LOP3.LUT R9, R9, R2, RZ, 0xfc, !PT ;        /* 0x0000000209097212 */
        // /* 0x000fc800078efcff */
        // /*00e0*/                   VIMNMX.U32 R7, PT, PT, R9, 0xff, PT ;       /* 0x000000ff09077848 */
        // /* 0x000fca0003fe0000 */
        // /*00f0*/                   STG.E desc[UR4][R4.64], R7 ;                /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_sat_u8_u64(unsigned long long a) {
    unsigned int out;
    asm volatile("cvt.sat.u8.u64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}

        // /*0090*/                   LDG.E.U16.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5500 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                  /* 0x0000000407047825 */
        // /* 0x008fe200078e0204 */
        // /*00b0*/                   LOP3.LUT R7, R2, 0xff, RZ, 0xc0, !PT ;       /* 0x000000ff02077812 */
        // /* 0x004fca00078ec0ff */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R7 ;                 /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int cvt_u8_u64(unsigned long long a) {
    unsigned int out;
    asm volatile("cvt.u8.u64 %0, %1;" : "=r"(out) : "l"(a));
    return out;
}


extern "C" __global__ void cvt_basic_kernel(
    const int* __restrict__ in_s32,
    const unsigned int* __restrict__ in_u32,
    const long long* __restrict__ in_s64,
    const unsigned long long* __restrict__ in_u64,
    const float* __restrict__ in_f32,
    const double* __restrict__ in_f64,
    const uint16_t* __restrict__ in_f16,
    const uint16_t* __restrict__ in_bf16,
    unsigned long long* __restrict__ out_acc
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int s32 = ((const volatile int*)in_s32)[tid];
    unsigned int u32 = ((const volatile unsigned int*)in_u32)[tid];
    long long s64 = ((const volatile long long*)in_s64)[tid];
    unsigned long long u64 = ((const volatile unsigned long long*)in_u64)[tid];
    float f32 = ((const volatile float*)in_f32)[tid];
    double f64 = ((const volatile double*)in_f64)[tid];
    uint16_t f16 = ((const volatile uint16_t*)in_f16)[tid];
    uint16_t bf16 = ((const volatile uint16_t*)in_bf16)[tid];
    int s8 = (int)(int8_t)s32;
    int s16 = (int)(int16_t)s32;
    unsigned int u8 = (unsigned int)(uint8_t)u32;
    unsigned int u16 = (unsigned int)(uint16_t)u32;
    unsigned long long acc = 0ull;
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_bf16_bf16(bf16);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_bf16_bf16(bf16);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_bf16_bf16(bf16);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_bf16_bf16(bf16);
    acc ^= (unsigned long long)(unsigned int)cvt_rm_bf16_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_bf16_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rp_bf16_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_bf16_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rm_bf16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_bf16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_ftz_bf16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rp_bf16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_bf16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_ftz_bf16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rm_bf16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_bf16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rp_bf16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_bf16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_bf16_s32(s32);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_bf16_s32(s32);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_bf16_s64(s64);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_bf16_s64(s64);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_bf16_u32(u32);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_bf16_u32(u32);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_bf16_u64(u64);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_bf16_u64(u64);
    acc ^= (unsigned long long)(unsigned int)cvt_rm_f16_bf16(bf16);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_f16_bf16(bf16);
    acc ^= (unsigned long long)(unsigned int)cvt_rp_f16_bf16(bf16);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_f16_bf16(bf16);
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_f16_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_sat_f16_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_f16_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_sat_f16_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_f16_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_sat_f16_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_f16_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_sat_f16_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rm_f16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_f16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_ftz_f16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_sat_f16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rp_f16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_f16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_ftz_f16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_sat_f16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rm_f16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_f16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_sat_f16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rp_f16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_f16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_sat_f16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_f16_s16(s16);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_f16_s16(s16);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_f16_s32(s32);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_sat_f16_s32(s32);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_f16_s32(s32);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_f16_s64(s64);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_sat_f16_s64(s64);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_f16_s64(s64);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_f16_s8(s8);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_f16_s8(s8);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_f16_u16(u16);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_f16_u16(u16);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_f16_u32(u32);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_sat_f16_u32(u32);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_f16_u32(u32);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_f16_u64(u64);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_sat_f16_u64(u64);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_f16_u64(u64);
    acc ^= (unsigned long long)(unsigned int)cvt_rn_f16_u8(u8);
    acc ^= (unsigned long long)(unsigned int)cvt_rz_f16_u8(u8);
    acc ^= (unsigned long long)__float_as_uint(cvt_rm_f32_bf16(bf16));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_f32_bf16(bf16));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_ftz_f32_bf16(bf16));
    acc ^= (unsigned long long)__float_as_uint(cvt_rp_f32_bf16(bf16));
    acc ^= (unsigned long long)__float_as_uint(cvt_rz_f32_bf16(bf16));
    acc ^= (unsigned long long)__float_as_uint(cvt_rz_ftz_f32_bf16(bf16));
    acc ^= (unsigned long long)__float_as_uint(cvt_rmi_f32_f32(f32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rmi_ftz_f32_f32(f32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rmi_sat_f32_f32(f32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rni_f32_f32(f32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rni_ftz_f32_f32(f32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rni_sat_f32_f32(f32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rpi_f32_f32(f32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rpi_ftz_f32_f32(f32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rpi_sat_f32_f32(f32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rzi_f32_f32(f32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rzi_ftz_f32_f32(f32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rzi_sat_f32_f32(f32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rm_f32_f64(f64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_f32_f64(f64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_ftz_f32_f64(f64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_sat_f32_f64(f64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rp_f32_f64(f64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rz_f32_f64(f64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rz_ftz_f32_f64(f64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rz_sat_f32_f64(f64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rm_f32_s16(s16));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_f32_s16(s16));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_ftz_f32_s16(s16));
    acc ^= (unsigned long long)__float_as_uint(cvt_rp_f32_s16(s16));
    acc ^= (unsigned long long)__float_as_uint(cvt_rz_f32_s16(s16));
    acc ^= (unsigned long long)__float_as_uint(cvt_rz_ftz_f32_s16(s16));
    acc ^= (unsigned long long)__float_as_uint(cvt_rm_f32_s32(s32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_f32_s32(s32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_ftz_f32_s32(s32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_ftz_sat_f32_s32(s32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_sat_f32_s32(s32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rp_f32_s32(s32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rz_f32_s32(s32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rz_ftz_f32_s32(s32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rm_f32_s64(s64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_f32_s64(s64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_ftz_f32_s64(s64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_ftz_sat_f32_s64(s64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_sat_f32_s64(s64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rp_f32_s64(s64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rz_f32_s64(s64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rz_ftz_f32_s64(s64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rm_f32_s8(s8));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_f32_s8(s8));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_ftz_f32_s8(s8));
    acc ^= (unsigned long long)__float_as_uint(cvt_rp_f32_s8(s8));
    acc ^= (unsigned long long)__float_as_uint(cvt_rz_f32_s8(s8));
    acc ^= (unsigned long long)__float_as_uint(cvt_rz_ftz_f32_s8(s8));
    acc ^= (unsigned long long)__float_as_uint(cvt_rm_f32_u16(u16));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_f32_u16(u16));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_ftz_f32_u16(u16));
    acc ^= (unsigned long long)__float_as_uint(cvt_rp_f32_u16(u16));
    acc ^= (unsigned long long)__float_as_uint(cvt_rz_f32_u16(u16));
    acc ^= (unsigned long long)__float_as_uint(cvt_rz_ftz_f32_u16(u16));
    acc ^= (unsigned long long)__float_as_uint(cvt_rm_f32_u32(u32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_f32_u32(u32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_ftz_f32_u32(u32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_ftz_sat_f32_u32(u32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_sat_f32_u32(u32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rp_f32_u32(u32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rz_f32_u32(u32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rz_ftz_f32_u32(u32));
    acc ^= (unsigned long long)__float_as_uint(cvt_rm_f32_u64(u64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_f32_u64(u64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_ftz_f32_u64(u64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_ftz_sat_f32_u64(u64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_sat_f32_u64(u64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rp_f32_u64(u64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rz_f32_u64(u64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rz_ftz_f32_u64(u64));
    acc ^= (unsigned long long)__float_as_uint(cvt_rm_f32_u8(u8));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_f32_u8(u8));
    acc ^= (unsigned long long)__float_as_uint(cvt_rn_ftz_f32_u8(u8));
    acc ^= (unsigned long long)__float_as_uint(cvt_rp_f32_u8(u8));
    acc ^= (unsigned long long)__float_as_uint(cvt_rz_f32_u8(u8));
    acc ^= (unsigned long long)__float_as_uint(cvt_rz_ftz_f32_u8(u8));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rm_f64_bf16(bf16));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rn_f64_bf16(bf16));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rp_f64_bf16(bf16));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rz_f64_bf16(bf16));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rmi_f64_f64(f64));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rmi_sat_f64_f64(f64));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rni_f64_f64(f64));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rni_sat_f64_f64(f64));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rpi_f64_f64(f64));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rpi_sat_f64_f64(f64));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rzi_f64_f64(f64));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rzi_sat_f64_f64(f64));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rm_f64_s16(s16));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rn_f64_s16(s16));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rp_f64_s16(s16));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rz_f64_s16(s16));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rm_f64_s32(s32));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rn_f64_s32(s32));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rn_sat_f64_s32(s32));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rp_f64_s32(s32));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rz_f64_s32(s32));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rm_f64_s64(s64));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rn_f64_s64(s64));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rn_sat_f64_s64(s64));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rp_f64_s64(s64));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rz_f64_s64(s64));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rm_f64_s8(s8));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rn_f64_s8(s8));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rp_f64_s8(s8));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rz_f64_s8(s8));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rm_f64_u16(u16));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rn_f64_u16(u16));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rp_f64_u16(u16));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rz_f64_u16(u16));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rm_f64_u32(u32));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rn_f64_u32(u32));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rn_sat_f64_u32(u32));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rp_f64_u32(u32));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rz_f64_u32(u32));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rm_f64_u64(u64));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rn_f64_u64(u64));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rn_sat_f64_u64(u64));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rp_f64_u64(u64));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rz_f64_u64(u64));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rm_f64_u8(u8));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rn_f64_u8(u8));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rp_f64_u8(u8));
    acc ^= (unsigned long long)__double_as_longlong(cvt_rz_f64_u8(u8));
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_s16_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_s16_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_s16_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_s16_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_s16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_ftz_s16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_ftz_sat_s16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_s16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_sat_s16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_s16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_ftz_s16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_ftz_sat_s16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_s16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_sat_s16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_s16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_s16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_sat_s16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_s16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_s16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_sat_s16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_s16_s32(s32);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_s16_s32(s32);
    acc ^= (unsigned long long)(unsigned int)cvt_s16_s64(s64);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_s16_s64(s64);
    acc ^= (unsigned long long)(unsigned int)cvt_s16_s8(s8);
    acc ^= (unsigned long long)(unsigned int)cvt_s16_u16(u16);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_s16_u16(u16);
    acc ^= (unsigned long long)(unsigned int)cvt_s16_u32(u32);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_s16_u32(u32);
    acc ^= (unsigned long long)(unsigned int)cvt_s16_u64(u64);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_s16_u64(u64);
    acc ^= (unsigned long long)(unsigned int)cvt_s16_u8(u8);
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_s32_bf16(bf16);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_s32_bf16(bf16);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_s32_bf16(bf16);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_s32_bf16(bf16);
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_s32_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_s32_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_s32_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_s32_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_s32_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_ftz_s32_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_s32_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_s32_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_ftz_s32_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_s32_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_s32_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_s32_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_s32_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_s32_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_s32_s16(s16);
    acc ^= (unsigned long long)(unsigned int)cvt_s32_s64(s64);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_s32_s64(s64);
    acc ^= (unsigned long long)(unsigned int)cvt_s32_s8(s8);
    acc ^= (unsigned long long)(unsigned int)cvt_s32_u16(u16);
    acc ^= (unsigned long long)(unsigned int)cvt_s32_u32(u32);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_s32_u32(u32);
    acc ^= (unsigned long long)(unsigned int)cvt_s32_u64(u64);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_s32_u64(u64);
    acc ^= (unsigned long long)(unsigned int)cvt_s32_u8(u8);
    acc ^= (unsigned long long)cvt_rmi_s64_bf16(bf16);
    acc ^= (unsigned long long)cvt_rni_s64_bf16(bf16);
    acc ^= (unsigned long long)cvt_rpi_s64_bf16(bf16);
    acc ^= (unsigned long long)cvt_rzi_s64_bf16(bf16);
    acc ^= (unsigned long long)cvt_rmi_s64_f16(f16);
    acc ^= (unsigned long long)cvt_rni_s64_f16(f16);
    acc ^= (unsigned long long)cvt_rpi_s64_f16(f16);
    acc ^= (unsigned long long)cvt_rzi_s64_f16(f16);
    acc ^= (unsigned long long)cvt_rmi_s64_f32(f32);
    acc ^= (unsigned long long)cvt_rni_ftz_s64_f32(f32);
    acc ^= (unsigned long long)cvt_rni_s64_f32(f32);
    acc ^= (unsigned long long)cvt_rpi_s64_f32(f32);
    acc ^= (unsigned long long)cvt_rzi_ftz_s64_f32(f32);
    acc ^= (unsigned long long)cvt_rzi_s64_f32(f32);
    acc ^= (unsigned long long)cvt_rmi_s64_f64(f64);
    acc ^= (unsigned long long)cvt_rni_s64_f64(f64);
    acc ^= (unsigned long long)cvt_rpi_s64_f64(f64);
    acc ^= (unsigned long long)cvt_rzi_s64_f64(f64);
    acc ^= (unsigned long long)cvt_s64_s16(s16);
    acc ^= (unsigned long long)cvt_s64_s32(s32);
    acc ^= (unsigned long long)cvt_s64_s8(s8);
    acc ^= (unsigned long long)cvt_s64_u16(u16);
    acc ^= (unsigned long long)cvt_s64_u32(u32);
    acc ^= (unsigned long long)cvt_s64_u64(u64);
    acc ^= (unsigned long long)cvt_sat_s64_u64(u64);
    acc ^= (unsigned long long)cvt_s64_u8(u8);
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_s8_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_s8_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_s8_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_s8_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_s8_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_ftz_s8_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_ftz_sat_s8_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_s8_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_sat_s8_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_s8_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_ftz_s8_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_ftz_sat_s8_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_s8_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_sat_s8_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_s8_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_s8_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_sat_s8_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_s8_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_s8_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_sat_s8_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_s8_s16(s16);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_s8_s16(s16);
    acc ^= (unsigned long long)(unsigned int)cvt_s8_s32(s32);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_s8_s32(s32);
    acc ^= (unsigned long long)(unsigned int)cvt_s8_s64(s64);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_s8_s64(s64);
    acc ^= (unsigned long long)(unsigned int)cvt_s8_u16(u16);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_s8_u16(u16);
    acc ^= (unsigned long long)(unsigned int)cvt_s8_u32(u32);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_s8_u32(u32);
    acc ^= (unsigned long long)(unsigned int)cvt_s8_u64(u64);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_s8_u64(u64);
    acc ^= (unsigned long long)(unsigned int)cvt_s8_u8(u8);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_s8_u8(u8);
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_u16_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_u16_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_u16_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_u16_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_u16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_ftz_sat_u16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_ftz_u16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_sat_u16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_u16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_u16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_ftz_sat_u16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_ftz_u16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_sat_u16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_u16_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_u16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_sat_u16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_u16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_u16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_sat_u16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_u16_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_u16_s16(s16);
    acc ^= (unsigned long long)(unsigned int)cvt_u16_s16(s16);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_u16_s32(s32);
    acc ^= (unsigned long long)(unsigned int)cvt_u16_s32(s32);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_u16_s64(s64);
    acc ^= (unsigned long long)(unsigned int)cvt_u16_s64(s64);
    acc ^= (unsigned long long)(unsigned int)cvt_u16_s8(s8);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_u16_u32(u32);
    acc ^= (unsigned long long)(unsigned int)cvt_u16_u32(u32);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_u16_u64(u64);
    acc ^= (unsigned long long)(unsigned int)cvt_u16_u64(u64);
    acc ^= (unsigned long long)(unsigned int)cvt_u16_u8(u8);
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_u32_bf16(bf16);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_u32_bf16(bf16);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_u32_bf16(bf16);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_u32_bf16(bf16);
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_u32_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_u32_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_u32_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_u32_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_u32_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_ftz_u32_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_u32_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_u32_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_ftz_u32_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_u32_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_u32_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_u32_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_u32_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_u32_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_u32_s16(s16);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_u32_s32(s32);
    acc ^= (unsigned long long)(unsigned int)cvt_u32_s32(s32);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_u32_s64(s64);
    acc ^= (unsigned long long)(unsigned int)cvt_u32_s64(s64);
    acc ^= (unsigned long long)(unsigned int)cvt_u32_s8(s8);
    acc ^= (unsigned long long)(unsigned int)cvt_u32_u16(u16);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_u32_u64(u64);
    acc ^= (unsigned long long)(unsigned int)cvt_u32_u64(u64);
    acc ^= (unsigned long long)(unsigned int)cvt_u32_u8(u8);
    acc ^= (unsigned long long)cvt_rmi_u64_bf16(bf16);
    acc ^= (unsigned long long)cvt_rni_u64_bf16(bf16);
    acc ^= (unsigned long long)cvt_rpi_u64_bf16(bf16);
    acc ^= (unsigned long long)cvt_rzi_u64_bf16(bf16);
    acc ^= (unsigned long long)cvt_rmi_u64_f16(f16);
    acc ^= (unsigned long long)cvt_rni_u64_f16(f16);
    acc ^= (unsigned long long)cvt_rpi_u64_f16(f16);
    acc ^= (unsigned long long)cvt_rzi_u64_f16(f16);
    acc ^= (unsigned long long)cvt_rmi_u64_f32(f32);
    acc ^= (unsigned long long)cvt_rni_ftz_u64_f32(f32);
    acc ^= (unsigned long long)cvt_rni_u64_f32(f32);
    acc ^= (unsigned long long)cvt_rpi_u64_f32(f32);
    acc ^= (unsigned long long)cvt_rzi_ftz_u64_f32(f32);
    acc ^= (unsigned long long)cvt_rzi_u64_f32(f32);
    acc ^= (unsigned long long)cvt_rmi_u64_f64(f64);
    acc ^= (unsigned long long)cvt_rni_u64_f64(f64);
    acc ^= (unsigned long long)cvt_rpi_u64_f64(f64);
    acc ^= (unsigned long long)cvt_rzi_u64_f64(f64);
    acc ^= (unsigned long long)cvt_u64_s16(s16);
    acc ^= (unsigned long long)cvt_u64_s32(s32);
    acc ^= (unsigned long long)cvt_sat_u64_s64(s64);
    acc ^= (unsigned long long)cvt_u64_s64(s64);
    acc ^= (unsigned long long)cvt_u64_s8(s8);
    acc ^= (unsigned long long)cvt_u64_u16(u16);
    acc ^= (unsigned long long)cvt_u64_u32(u32);
    acc ^= (unsigned long long)cvt_u64_u8(u8);
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_u8_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_u8_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_u8_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_u8_f16(f16);
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_u8_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_ftz_sat_u8_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_ftz_u8_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_sat_u8_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_u8_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_u8_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_ftz_sat_u8_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_ftz_u8_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_sat_u8_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_u8_f32(f32);
    acc ^= (unsigned long long)(unsigned int)cvt_rmi_u8_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_sat_u8_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rni_u8_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rpi_u8_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_sat_u8_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_rzi_u8_f64(f64);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_u8_s16(s16);
    acc ^= (unsigned long long)(unsigned int)cvt_u8_s16(s16);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_u8_s32(s32);
    acc ^= (unsigned long long)(unsigned int)cvt_u8_s32(s32);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_u8_s64(s64);
    acc ^= (unsigned long long)(unsigned int)cvt_u8_s64(s64);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_u8_s8(s8);
    acc ^= (unsigned long long)(unsigned int)cvt_u8_s8(s8);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_u8_u16(u16);
    acc ^= (unsigned long long)(unsigned int)cvt_u8_u16(u16);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_u8_u32(u32);
    acc ^= (unsigned long long)(unsigned int)cvt_u8_u32(u32);
    acc ^= (unsigned long long)(unsigned int)cvt_sat_u8_u64(u64);
    acc ^= (unsigned long long)(unsigned int)cvt_u8_u64(u64);
    out_acc[tid] = acc;
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    constexpr int N = 256;

    int* in_s32;
    unsigned int* in_u32;
    long long* in_s64;
    unsigned long long* in_u64;
    float* in_f32;
    double* in_f64;
    uint16_t* in_f16;
    uint16_t* in_bf16;
    unsigned long long* out_acc;

    ck(cudaMallocManaged(&in_s32, N * sizeof(int)), "cudaMallocManaged in_s32");
    ck(cudaMallocManaged(&in_u32, N * sizeof(unsigned int)), "cudaMallocManaged in_u32");
    ck(cudaMallocManaged(&in_s64, N * sizeof(long long)), "cudaMallocManaged in_s64");
    ck(cudaMallocManaged(&in_u64, N * sizeof(unsigned long long)), "cudaMallocManaged in_u64");
    ck(cudaMallocManaged(&in_f32, N * sizeof(float)), "cudaMallocManaged in_f32");
    ck(cudaMallocManaged(&in_f64, N * sizeof(double)), "cudaMallocManaged in_f64");
    ck(cudaMallocManaged(&in_f16, N * sizeof(uint16_t)), "cudaMallocManaged in_f16");
    ck(cudaMallocManaged(&in_bf16, N * sizeof(uint16_t)), "cudaMallocManaged in_bf16");
    ck(cudaMallocManaged(&out_acc, N * sizeof(unsigned long long)), "cudaMallocManaged out_acc");

    for (int i = 0; i < N; ++i) {
        in_s32[i] = (i * 3 - 123) ^ 0x5a5a1234;
        in_u32[i] = (unsigned int)(i * 7 + 5) ^ 0xa5a51234u;
        in_s64[i] = ((long long)(i * 11 - 5000)) ^ 0x1122334455667788ll;
        in_u64[i] = ((unsigned long long)(i * 13 + 9)) ^ 0x8877665544332211ull;
        float base = (float)(i * 0.5f - 64.0f);
        in_f32[i] = base + 0.25f;
        in_f64[i] = (double)base * 1.25;
        in_f16[i] = (uint16_t)(0x3c00u + (unsigned)(i & 0x03ffu));
        in_bf16[i] = (uint16_t)(0x3f80u + (unsigned)(i & 0x00ffu));
        out_acc[i] = 0;
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    cvt_basic_kernel<<<grid, block>>>(in_s32, in_u32, in_s64, in_u64, in_f32, in_f64, in_f16, in_bf16, out_acc);
    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("acc=%llu\n", (unsigned long long)out_acc[0]);

    cudaFree(in_s32);
    cudaFree(in_u32);
    cudaFree(in_s64);
    cudaFree(in_u64);
    cudaFree(in_f32);
    cudaFree(in_f64);
    cudaFree(in_f16);
    cudaFree(in_bf16);
    cudaFree(out_acc);
    return 0;
}
