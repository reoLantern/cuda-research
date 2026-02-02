// redux_sync_forms.cu
//
// PTX redux.sync forms:
//   redux.sync.add.{u32|s32}   d, a, membermask;
//   redux.sync.min.{u32|s32}   d, a, membermask;
//   redux.sync.max.{u32|s32}   d, a, membermask;
//   redux.sync.{and|or|xor}.b32 d, a, membermask;
//   redux.sync.{min|max}.f32 d, a, membermask;
//   redux.sync.{min|max}.abs.f32 d, a, membermask;
//   redux.sync.{min|max}.NaN.f32 d, a, membermask;
//   redux.sync.{min|max}.abs.NaN.f32 d, a, membermask;
//
// Note: .f32/.abs/.NaN forms require sm_100a or newer.

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// ---- wrappers: each PTX redux.sync form in its own function ----

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   LOP3.LUT R0, R0, 0x1f, RZ, 0xc0, !PT ;   /* 0x0000001f00007812 */
        // /* 0x000fca00078ec0ff */
        // /*00b0*/                   IMAD R5, R0, 0x11, R5 ;                  /* 0x0000001100057824 */
        // /* 0x008fe200078e0205 */
        // /*00c0*/                   VOTE.ANY R0, PT, PT ;                    /* 0x0000000000007806 */
        // /* 0x000fc800038e0100 */
        // /*00d0*/                   LOP3.LUT R9, R2, R5, RZ, 0x3c, !PT ;     /* 0x0000000502097212 */
        // /* 0x004fe400078e3cff */
        // /*00e0*/                   LDC.64 R4, c[0x0][0x388] ;               /* 0x0000e200ff047b82 */
        // /* 0x000e300000000a00 */
        // /*00f0*/                   REDUX.SUM UR6, R9 ;                      /* 0x00000000090673c4 */
        // /* 0x000e62000000c000 */
        // /*0100*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x001fe200078e0204 */
        // /*0110*/                   MOV R7, UR6 ;                            /* 0x0000000600077c02 */
        // /* 0x002fca0008000f00 */
        // /*0120*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t redux_sync_add_u32(uint32_t mask, uint32_t a) {
    uint32_t d;
    asm volatile("redux.sync.add.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(mask));
    return d;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;       /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   LOP3.LUT R0, R0, 0x1f, RZ, 0xc0, !PT ;        /* 0x0000001f00007812 */
        // /* 0x000fca00078ec0ff */
        // /*00b0*/                   IMAD R5, R0, 0x11, R5 ;                       /* 0x0000001100057824 */
        // /* 0x008fe200078e0205 */
        // /*00c0*/                   VOTE.ANY R0, PT, PT ;                         /* 0x0000000000007806 */
        // /* 0x000fc800038e0100 */
        // /*00d0*/                   LOP3.LUT R9, R5, 0x80000000, R2, 0x96, !PT ;  /* 0x8000000005097812 */
        // /* 0x004fe400078e9602 */
        // /*00e0*/                   LDC.64 R4, c[0x0][0x388] ;                    /* 0x0000e200ff047b82 */
        // /* 0x000e300000000a00 */
        // /*00f0*/                   REDUX.SUM.S32 UR6, R9 ;                       /* 0x00000000090673c4 */
        // /* 0x000e62000000c200 */
        // /*0100*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                   /* 0x0000000407047825 */
        // /* 0x001fe200078e0204 */
        // /*0110*/                   MOV R7, UR6 ;                                 /* 0x0000000600077c02 */
        // /* 0x002fca0008000f00 */
        // /*0120*/                   STG.E desc[UR4][R4.64], R7 ;                  /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int32_t redux_sync_add_s32(uint32_t mask, int32_t a) {
    int32_t d;
    asm volatile("redux.sync.add.s32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(mask));
    return d;
}

        // /*00a0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00b0*/                   LOP3.LUT R0, R0, 0x1f, RZ, 0xc0, !PT ;   /* 0x0000001f00007812 */
        // /* 0x000fca00078ec0ff */
        // /*00c0*/                   IMAD R9, R0, 0x11, R9 ;                  /* 0x0000001100097824 */
        // /* 0x008fe200078e0209 */
        // /*00d0*/                   VOTE.ANY R0, PT, PT ;                    /* 0x0000000000007806 */
        // /* 0x000fe200038e0100 */
        // /*00e0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x001fc600078e0204 */
        // /*00f0*/                   LOP3.LUT R9, R2, R9, RZ, 0x3c, !PT ;     /* 0x0000000902097212 */
        // /* 0x004fc800078e3cff */
        // /*0100*/                   CREDUX.MIN UR6, R9 ;                     /* 0x00000000090672cc */
        // /* 0x000fda0000008000 */
        // /*0110*/                   MOV R7, UR6 ;                            /* 0x0000000600077c02 */
        // /* 0x000fca0008000f00 */
        // /*0120*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t redux_sync_min_u32(uint32_t mask, uint32_t a) {
    uint32_t d;
    asm volatile("redux.sync.min.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(mask));
    return d;
}

        // /*00a0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;       /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00b0*/                   LOP3.LUT R0, R0, 0x1f, RZ, 0xc0, !PT ;        /* 0x0000001f00007812 */
        // /* 0x000fca00078ec0ff */
        // /*00c0*/                   IMAD R9, R0, 0x11, R9 ;                       /* 0x0000001100097824 */
        // /* 0x008fe200078e0209 */
        // /*00d0*/                   VOTE.ANY R0, PT, PT ;                         /* 0x0000000000007806 */
        // /* 0x000fe200038e0100 */
        // /*00e0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                   /* 0x0000000407047825 */
        // /* 0x001fc600078e0204 */
        // /*00f0*/                   LOP3.LUT R9, R9, 0x80000000, R2, 0x96, !PT ;  /* 0x8000000009097812 */
        // /* 0x004fc800078e9602 */
        // /*0100*/                   CREDUX.MIN.S32 UR6, R9 ;                      /* 0x00000000090672cc */
        // /* 0x000fda0000008200 */
        // /*0110*/                   MOV R7, UR6 ;                                 /* 0x0000000600077c02 */
        // /* 0x000fca0008000f00 */
        // /*0120*/                   STG.E desc[UR4][R4.64], R7 ;                  /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int32_t redux_sync_min_s32(uint32_t mask, int32_t a) {
    int32_t d;
    asm volatile("redux.sync.min.s32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(mask));
    return d;
}

        // /*00a0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00b0*/                   LOP3.LUT R0, R0, 0x1f, RZ, 0xc0, !PT ;   /* 0x0000001f00007812 */
        // /* 0x000fca00078ec0ff */
        // /*00c0*/                   IMAD R9, R0, 0x11, R9 ;                  /* 0x0000001100097824 */
        // /* 0x008fe200078e0209 */
        // /*00d0*/                   VOTE.ANY R0, PT, PT ;                    /* 0x0000000000007806 */
        // /* 0x000fe200038e0100 */
        // /*00e0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x001fc600078e0204 */
        // /*00f0*/                   LOP3.LUT R9, R2, R9, RZ, 0x3c, !PT ;     /* 0x0000000902097212 */
        // /* 0x004fc800078e3cff */
        // /*0100*/                   CREDUX.MAX UR6, R9 ;                     /* 0x00000000090672cc */
        // /* 0x000fda0000000000 */
        // /*0110*/                   MOV R7, UR6 ;                            /* 0x0000000600077c02 */
        // /* 0x000fca0008000f00 */
        // /*0120*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t redux_sync_max_u32(uint32_t mask, uint32_t a) {
    uint32_t d;
    asm volatile("redux.sync.max.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(mask));
    return d;
}

        // /*00a0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;       /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00b0*/                   LOP3.LUT R0, R0, 0x1f, RZ, 0xc0, !PT ;        /* 0x0000001f00007812 */
        // /* 0x000fca00078ec0ff */
        // /*00c0*/                   IMAD R9, R0, 0x11, R9 ;                       /* 0x0000001100097824 */
        // /* 0x008fe200078e0209 */
        // /*00d0*/                   VOTE.ANY R0, PT, PT ;                         /* 0x0000000000007806 */
        // /* 0x000fe200038e0100 */
        // /*00e0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                   /* 0x0000000407047825 */
        // /* 0x001fc600078e0204 */
        // /*00f0*/                   LOP3.LUT R9, R9, 0x80000000, R2, 0x96, !PT ;  /* 0x8000000009097812 */
        // /* 0x004fc800078e9602 */
        // /*0100*/                   CREDUX.MAX.S32 UR6, R9 ;                      /* 0x00000000090672cc */
        // /* 0x000fda0000000200 */
        // /*0110*/                   MOV R7, UR6 ;                                 /* 0x0000000600077c02 */
        // /* 0x000fca0008000f00 */
        // /*0120*/                   STG.E desc[UR4][R4.64], R7 ;                  /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ int32_t redux_sync_max_s32(uint32_t mask, int32_t a) {
    int32_t d;
    asm volatile("redux.sync.max.s32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(mask));
    return d;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   LOP3.LUT R0, R0, 0x1f, RZ, 0xc0, !PT ;   /* 0x0000001f00007812 */
        // /* 0x000fca00078ec0ff */
        // /*00b0*/                   IMAD R5, R0, 0x11, R5 ;                  /* 0x0000001100057824 */
        // /* 0x008fe200078e0205 */
        // /*00c0*/                   VOTE.ANY R0, PT, PT ;                    /* 0x0000000000007806 */
        // /* 0x000fc800038e0100 */
        // /*00d0*/                   LOP3.LUT R9, R2, R5, RZ, 0x3c, !PT ;     /* 0x0000000502097212 */
        // /* 0x004fe400078e3cff */
        // /*00e0*/                   LDC.64 R4, c[0x0][0x388] ;               /* 0x0000e200ff047b82 */
        // /* 0x000e300000000a00 */
        // /*00f0*/                   REDUX UR6, R9 ;                          /* 0x00000000090673c4 */
        // /* 0x000e620000000000 */
        // /*0100*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x001fe200078e0204 */
        // /*0110*/                   MOV R7, UR6 ;                            /* 0x0000000600077c02 */
        // /* 0x002fca0008000f00 */
        // /*0120*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t redux_sync_and_b32(uint32_t mask, uint32_t a) {
    uint32_t d;
    asm volatile("redux.sync.and.b32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(mask));
    return d;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   LOP3.LUT R0, R0, 0x1f, RZ, 0xc0, !PT ;   /* 0x0000001f00007812 */
        // /* 0x000fca00078ec0ff */
        // /*00b0*/                   IMAD R5, R0, 0x11, R5 ;                  /* 0x0000001100057824 */
        // /* 0x008fe200078e0205 */
        // /*00c0*/                   VOTE.ANY R0, PT, PT ;                    /* 0x0000000000007806 */
        // /* 0x000fc800038e0100 */
        // /*00d0*/                   LOP3.LUT R9, R2, R5, RZ, 0x3c, !PT ;     /* 0x0000000502097212 */
        // /* 0x004fe400078e3cff */
        // /*00e0*/                   LDC.64 R4, c[0x0][0x388] ;               /* 0x0000e200ff047b82 */
        // /* 0x000e300000000a00 */
        // /*00f0*/                   REDUX.OR UR6, R9 ;                       /* 0x00000000090673c4 */
        // /* 0x000e620000004000 */
        // /*0100*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x001fe200078e0204 */
        // /*0110*/                   MOV R7, UR6 ;                            /* 0x0000000600077c02 */
        // /* 0x002fca0008000f00 */
        // /*0120*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t redux_sync_or_b32(uint32_t mask, uint32_t a) {
    uint32_t d;
    asm volatile("redux.sync.or.b32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(mask));
    return d;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   LOP3.LUT R0, R0, 0x1f, RZ, 0xc0, !PT ;   /* 0x0000001f00007812 */
        // /* 0x000fca00078ec0ff */
        // /*00b0*/                   IMAD R5, R0, 0x11, R5 ;                  /* 0x0000001100057824 */
        // /* 0x008fe200078e0205 */
        // /*00c0*/                   VOTE.ANY R0, PT, PT ;                    /* 0x0000000000007806 */
        // /* 0x000fc800038e0100 */
        // /*00d0*/                   LOP3.LUT R9, R2, R5, RZ, 0x3c, !PT ;     /* 0x0000000502097212 */
        // /* 0x004fe400078e3cff */
        // /*00e0*/                   LDC.64 R4, c[0x0][0x388] ;               /* 0x0000e200ff047b82 */
        // /* 0x000e300000000a00 */
        // /*00f0*/                   REDUX.XOR UR6, R9 ;                      /* 0x00000000090673c4 */
        // /* 0x000e620000008000 */
        // /*0100*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x001fe200078e0204 */
        // /*0110*/                   MOV R7, UR6 ;                            /* 0x0000000600077c02 */
        // /* 0x002fca0008000f00 */
        // /*0120*/                   STG.E desc[UR4][R4.64], R7 ;             /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t redux_sync_xor_b32(uint32_t mask, uint32_t a) {
    uint32_t d;
    asm volatile("redux.sync.xor.b32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(mask));
    return d;
}

        // /*00a0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;       /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00b0*/                   LOP3.LUT R0, R6.reuse, 0x1f, RZ, 0xc0, !PT ;  /* 0x0000001f06007812 */
        // /* 0x040fe400078ec0ff */
        // /*00c0*/                   LOP3.LUT P0, RZ, R6, 0x1, RZ, 0xc0, !PT ;     /* 0x0000000106ff7812 */
        // /* 0x000fc6000780c0ff */
        // /*00d0*/                   IMAD R9, R0, 0x11, R9 ;                       /* 0x0000001100097824 */
        // /* 0x008fe200078e0209 */
        // /*00e0*/                   VOTE.ANY R0, PT, PT ;                         /* 0x0000000000007806 */
        // /* 0x000fc800038e0100 */
        // /*00f0*/                   LOP3.LUT R9, R2, 0x7fffff, R9, 0x48, !PT ;    /* 0x007fffff02097812 */
        // /* 0x004fe200078e4809 */
        // /*0100*/                   IMAD.WIDE R2, R7, 0x4, R4 ;                   /* 0x0000000407027825 */
        // /* 0x001fc600078e0204 */
        // /*0110*/                   LOP3.LUT R9, R9, 0x3f000000, RZ, 0xfc, !PT ;  /* 0x3f00000009097812 */
        // /* 0x000fc800078efcff */
        // /*0120*/                   FSEL R9, -R9, R9, P0 ;                        /* 0x0000000909097208 */
        // /* 0x000fc80000000100 */
        // /*0130*/                   CREDUX.MIN.F32 UR6, R9 ;                      /* 0x00000000090672cc */
        // /* 0x000fda0000008400 */
        // /*0140*/                   IMAD.U32 R5, RZ, RZ, UR6 ;                    /* 0x00000006ff057e24 */
        // /* 0x000fca000f8e00ff */
        // /*0150*/                   STG.E desc[UR4][R2.64], R5 ;                  /* 0x0000000502007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float redux_sync_min_f32(uint32_t mask, float a) {
    float d;
    asm volatile("redux.sync.min.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "r"(mask));
    return d;
}

        // /*00a0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;       /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00b0*/                   LOP3.LUT R0, R6.reuse, 0x1f, RZ, 0xc0, !PT ;  /* 0x0000001f06007812 */
        // /* 0x040fe400078ec0ff */
        // /*00c0*/                   LOP3.LUT P0, RZ, R6, 0x1, RZ, 0xc0, !PT ;     /* 0x0000000106ff7812 */
        // /* 0x000fc6000780c0ff */
        // /*00d0*/                   IMAD R9, R0, 0x11, R9 ;                       /* 0x0000001100097824 */
        // /* 0x008fe200078e0209 */
        // /*00e0*/                   VOTE.ANY R0, PT, PT ;                         /* 0x0000000000007806 */
        // /* 0x000fc800038e0100 */
        // /*00f0*/                   LOP3.LUT R9, R2, 0x7fffff, R9, 0x48, !PT ;    /* 0x007fffff02097812 */
        // /* 0x004fe200078e4809 */
        // /*0100*/                   IMAD.WIDE R2, R7, 0x4, R4 ;                   /* 0x0000000407027825 */
        // /* 0x001fc600078e0204 */
        // /*0110*/                   LOP3.LUT R9, R9, 0x3f000000, RZ, 0xfc, !PT ;  /* 0x3f00000009097812 */
        // /* 0x000fc800078efcff */
        // /*0120*/                   FSEL R9, -R9, R9, P0 ;                        /* 0x0000000909097208 */
        // /* 0x000fc80000000100 */
        // /*0130*/                   CREDUX.MAX.F32 UR6, R9 ;                      /* 0x00000000090672cc */
        // /* 0x000fda0000000400 */
        // /*0140*/                   IMAD.U32 R5, RZ, RZ, UR6 ;                    /* 0x00000006ff057e24 */
        // /* 0x000fca000f8e00ff */
        // /*0150*/                   STG.E desc[UR4][R2.64], R5 ;                  /* 0x0000000502007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float redux_sync_max_f32(uint32_t mask, float a) {
    float d;
    asm volatile("redux.sync.max.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "r"(mask));
    return d;
}

        // /*00a0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;       /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00b0*/                   LOP3.LUT R0, R6.reuse, 0x1f, RZ, 0xc0, !PT ;  /* 0x0000001f06007812 */
        // /* 0x040fe400078ec0ff */
        // /*00c0*/                   LOP3.LUT P0, RZ, R6, 0x1, RZ, 0xc0, !PT ;     /* 0x0000000106ff7812 */
        // /* 0x000fc6000780c0ff */
        // /*00d0*/                   IMAD R9, R0, 0x11, R9 ;                       /* 0x0000001100097824 */
        // /* 0x008fe200078e0209 */
        // /*00e0*/                   VOTE.ANY R0, PT, PT ;                         /* 0x0000000000007806 */
        // /* 0x000fc800038e0100 */
        // /*00f0*/                   LOP3.LUT R9, R2, 0x7fffff, R9, 0x48, !PT ;    /* 0x007fffff02097812 */
        // /* 0x004fe200078e4809 */
        // /*0100*/                   IMAD.WIDE R2, R7, 0x4, R4 ;                   /* 0x0000000407027825 */
        // /* 0x001fc600078e0204 */
        // /*0110*/                   LOP3.LUT R9, R9, 0x3f000000, RZ, 0xfc, !PT ;  /* 0x3f00000009097812 */
        // /* 0x000fc800078efcff */
        // /*0120*/                   FSEL R9, -R9, R9, P0 ;                        /* 0x0000000909097208 */
        // /* 0x000fc80000000100 */
        // /*0130*/                   CREDUX.MINABS.F32 UR6, R9 ;                   /* 0x00000000090672cc */
        // /* 0x000fda000000c400 */
        // /*0140*/                   IMAD.U32 R5, RZ, RZ, UR6 ;                    /* 0x00000006ff057e24 */
        // /* 0x000fca000f8e00ff */
        // /*0150*/                   STG.E desc[UR4][R2.64], R5 ;                  /* 0x0000000502007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float redux_sync_min_abs_f32(uint32_t mask, float a) {
    float d;
    asm volatile("redux.sync.min.abs.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "r"(mask));
    return d;
}

        // /*00a0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;       /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00b0*/                   LOP3.LUT R0, R6.reuse, 0x1f, RZ, 0xc0, !PT ;  /* 0x0000001f06007812 */
        // /* 0x040fe400078ec0ff */
        // /*00c0*/                   LOP3.LUT P0, RZ, R6, 0x1, RZ, 0xc0, !PT ;     /* 0x0000000106ff7812 */
        // /* 0x000fc6000780c0ff */
        // /*00d0*/                   IMAD R9, R0, 0x11, R9 ;                       /* 0x0000001100097824 */
        // /* 0x008fe200078e0209 */
        // /*00e0*/                   VOTE.ANY R0, PT, PT ;                         /* 0x0000000000007806 */
        // /* 0x000fc800038e0100 */
        // /*00f0*/                   LOP3.LUT R9, R2, 0x7fffff, R9, 0x48, !PT ;    /* 0x007fffff02097812 */
        // /* 0x004fe200078e4809 */
        // /*0100*/                   IMAD.WIDE R2, R7, 0x4, R4 ;                   /* 0x0000000407027825 */
        // /* 0x001fc600078e0204 */
        // /*0110*/                   LOP3.LUT R9, R9, 0x3f000000, RZ, 0xfc, !PT ;  /* 0x3f00000009097812 */
        // /* 0x000fc800078efcff */
        // /*0120*/                   FSEL R9, -R9, R9, P0 ;                        /* 0x0000000909097208 */
        // /* 0x000fc80000000100 */
        // /*0130*/                   CREDUX.MAXABS.F32 UR6, R9 ;                   /* 0x00000000090672cc */
        // /* 0x000fda0000004400 */
        // /*0140*/                   IMAD.U32 R5, RZ, RZ, UR6 ;                    /* 0x00000006ff057e24 */
        // /* 0x000fca000f8e00ff */
        // /*0150*/                   STG.E desc[UR4][R2.64], R5 ;                  /* 0x0000000502007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float redux_sync_max_abs_f32(uint32_t mask, float a) {
    float d;
    asm volatile("redux.sync.max.abs.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "r"(mask));
    return d;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;           /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   LOP3.LUT R0, R4.reuse, 0x1f, RZ, 0xc0, !PT ;      /* 0x0000001f04007812 */
        // /* 0x040fe400078ec0ff */
        // /*00b0*/                   R2P PR, R4, 0x3 ;                                 /* 0x0000000304007804 */
        // /* 0x000fc60000000000 */
        // /*00c0*/                   IMAD R5, R0, 0x11, R5 ;                           /* 0x0000001100057824 */
        // /* 0x008fca00078e0205 */
        // /*00d0*/                   LOP3.LUT R0, R2, R5, RZ, 0x3c, !PT ;              /* 0x0000000502007212 */
        // /* 0x004fe400078e3cff */
        // /*00e0*/                   LDC.64 R4, c[0x0][0x388] ;                        /* 0x0000e200ff047b82 */
        // /* 0x000e240000000a00 */
        // /*00f0*/                   LOP3.LUT R6, R0.reuse, 0x7fffff, RZ, 0xc0, !PT ;  /* 0x007fffff00067812 */
        // /* 0x040fe400078ec0ff */
        // /*0100*/                   LOP3.LUT R8, R0, 0x3fffff, RZ, 0xc0, !PT ;        /* 0x003fffff00087812 */
        // /* 0x000fe400078ec0ff */
        // /*0110*/                   LOP3.LUT R6, R6, 0x3f000000, RZ, 0xfc, !PT ;      /* 0x3f00000006067812 */
        // /* 0x000fe400078efcff */
        // /*0120*/                   VOTE.ANY R0, PT, PT ;                             /* 0x0000000000007806 */
        // /* 0x000fc400038e0100 */
        // /*0130*/                   FSEL R9, -R6, R6, P0 ;                            /* 0x0000000606097208 */
        // /* 0x000fe40000000100 */
        // /*0140*/               @P1 LOP3.LUT R9, R8, 0x7fc00000, RZ, 0xfc, !PT ;      /* 0x7fc0000008091812 */
        // /* 0x000fc800078efcff */
        // /*0150*/                   CREDUX.MIN.F32.NAN UR6, R9 ;                      /* 0x00000000090672cc */
        // /* 0x000fe2000000a400 */
        // /*0160*/                   IMAD.WIDE R2, R7, 0x4, R4 ;                       /* 0x0000000407027825 */
        // /* 0x001fd800078e0204 */
        // /*0170*/                   IMAD.U32 R5, RZ, RZ, UR6 ;                        /* 0x00000006ff057e24 */
        // /* 0x000fca000f8e00ff */
        // /*0180*/                   STG.E desc[UR4][R2.64], R5 ;                      /* 0x0000000502007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float redux_sync_min_nan_f32(uint32_t mask, float a) {
    float d;
    asm volatile("redux.sync.min.NaN.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "r"(mask));
    return d;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;           /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   LOP3.LUT R0, R4.reuse, 0x1f, RZ, 0xc0, !PT ;      /* 0x0000001f04007812 */
        // /* 0x040fe400078ec0ff */
        // /*00b0*/                   R2P PR, R4, 0x3 ;                                 /* 0x0000000304007804 */
        // /* 0x000fc60000000000 */
        // /*00c0*/                   IMAD R5, R0, 0x11, R5 ;                           /* 0x0000001100057824 */
        // /* 0x008fca00078e0205 */
        // /*00d0*/                   LOP3.LUT R0, R2, R5, RZ, 0x3c, !PT ;              /* 0x0000000502007212 */
        // /* 0x004fe400078e3cff */
        // /*00e0*/                   LDC.64 R4, c[0x0][0x388] ;                        /* 0x0000e200ff047b82 */
        // /* 0x000e240000000a00 */
        // /*00f0*/                   LOP3.LUT R6, R0.reuse, 0x7fffff, RZ, 0xc0, !PT ;  /* 0x007fffff00067812 */
        // /* 0x040fe400078ec0ff */
        // /*0100*/                   LOP3.LUT R8, R0, 0x3fffff, RZ, 0xc0, !PT ;        /* 0x003fffff00087812 */
        // /* 0x000fe400078ec0ff */
        // /*0110*/                   LOP3.LUT R6, R6, 0x3f000000, RZ, 0xfc, !PT ;      /* 0x3f00000006067812 */
        // /* 0x000fe400078efcff */
        // /*0120*/                   VOTE.ANY R0, PT, PT ;                             /* 0x0000000000007806 */
        // /* 0x000fc400038e0100 */
        // /*0130*/                   FSEL R9, -R6, R6, P0 ;                            /* 0x0000000606097208 */
        // /* 0x000fe40000000100 */
        // /*0140*/               @P1 LOP3.LUT R9, R8, 0x7fc00000, RZ, 0xfc, !PT ;      /* 0x7fc0000008091812 */
        // /* 0x000fc800078efcff */
        // /*0150*/                   CREDUX.MAX.F32.NAN UR6, R9 ;                      /* 0x00000000090672cc */
        // /* 0x000fe20000002400 */
        // /*0160*/                   IMAD.WIDE R2, R7, 0x4, R4 ;                       /* 0x0000000407027825 */
        // /* 0x001fd800078e0204 */
        // /*0170*/                   IMAD.U32 R5, RZ, RZ, UR6 ;                        /* 0x00000006ff057e24 */
        // /* 0x000fca000f8e00ff */
        // /*0180*/                   STG.E desc[UR4][R2.64], R5 ;                      /* 0x0000000502007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float redux_sync_max_nan_f32(uint32_t mask, float a) {
    float d;
    asm volatile("redux.sync.max.NaN.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "r"(mask));
    return d;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;           /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   LOP3.LUT R0, R4.reuse, 0x1f, RZ, 0xc0, !PT ;      /* 0x0000001f04007812 */
        // /* 0x040fe400078ec0ff */
        // /*00b0*/                   R2P PR, R4, 0x3 ;                                 /* 0x0000000304007804 */
        // /* 0x000fc60000000000 */
        // /*00c0*/                   IMAD R5, R0, 0x11, R5 ;                           /* 0x0000001100057824 */
        // /* 0x008fca00078e0205 */
        // /*00d0*/                   LOP3.LUT R0, R2, R5, RZ, 0x3c, !PT ;              /* 0x0000000502007212 */
        // /* 0x004fe400078e3cff */
        // /*00e0*/                   LDC.64 R4, c[0x0][0x388] ;                        /* 0x0000e200ff047b82 */
        // /* 0x000e240000000a00 */
        // /*00f0*/                   LOP3.LUT R6, R0.reuse, 0x7fffff, RZ, 0xc0, !PT ;  /* 0x007fffff00067812 */
        // /* 0x040fe400078ec0ff */
        // /*0100*/                   LOP3.LUT R8, R0, 0x3fffff, RZ, 0xc0, !PT ;        /* 0x003fffff00087812 */
        // /* 0x000fe400078ec0ff */
        // /*0110*/                   LOP3.LUT R6, R6, 0x3f000000, RZ, 0xfc, !PT ;      /* 0x3f00000006067812 */
        // /* 0x000fe400078efcff */
        // /*0120*/                   VOTE.ANY R0, PT, PT ;                             /* 0x0000000000007806 */
        // /* 0x000fc400038e0100 */
        // /*0130*/                   FSEL R9, -R6, R6, P0 ;                            /* 0x0000000606097208 */
        // /* 0x000fe40000000100 */
        // /*0140*/               @P1 LOP3.LUT R9, R8, 0x7fc00000, RZ, 0xfc, !PT ;      /* 0x7fc0000008091812 */
        // /* 0x000fc800078efcff */
        // /*0150*/                   CREDUX.MINABS.F32.NAN UR6, R9 ;                   /* 0x00000000090672cc */
        // /* 0x000fe2000000e400 */
        // /*0160*/                   IMAD.WIDE R2, R7, 0x4, R4 ;                       /* 0x0000000407027825 */
        // /* 0x001fd800078e0204 */
        // /*0170*/                   IMAD.U32 R5, RZ, RZ, UR6 ;                        /* 0x00000006ff057e24 */
        // /* 0x000fca000f8e00ff */
        // /*0180*/                   STG.E desc[UR4][R2.64], R5 ;                      /* 0x0000000502007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float redux_sync_min_abs_nan_f32(uint32_t mask, float a) {
    float d;
    asm volatile("redux.sync.min.abs.NaN.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "r"(mask));
    return d;
}

        // /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;           /* 0x0000000402027981 */
        // /* 0x002ea2000c1f5900 */
        // /*00a0*/                   LOP3.LUT R0, R4.reuse, 0x1f, RZ, 0xc0, !PT ;      /* 0x0000001f04007812 */
        // /* 0x040fe400078ec0ff */
        // /*00b0*/                   R2P PR, R4, 0x3 ;                                 /* 0x0000000304007804 */
        // /* 0x000fc60000000000 */
        // /*00c0*/                   IMAD R5, R0, 0x11, R5 ;                           /* 0x0000001100057824 */
        // /* 0x008fca00078e0205 */
        // /*00d0*/                   LOP3.LUT R0, R2, R5, RZ, 0x3c, !PT ;              /* 0x0000000502007212 */
        // /* 0x004fe400078e3cff */
        // /*00e0*/                   LDC.64 R4, c[0x0][0x388] ;                        /* 0x0000e200ff047b82 */
        // /* 0x000e240000000a00 */
        // /*00f0*/                   LOP3.LUT R6, R0.reuse, 0x7fffff, RZ, 0xc0, !PT ;  /* 0x007fffff00067812 */
        // /* 0x040fe400078ec0ff */
        // /*0100*/                   LOP3.LUT R8, R0, 0x3fffff, RZ, 0xc0, !PT ;        /* 0x003fffff00087812 */
        // /* 0x000fe400078ec0ff */
        // /*0110*/                   LOP3.LUT R6, R6, 0x3f000000, RZ, 0xfc, !PT ;      /* 0x3f00000006067812 */
        // /* 0x000fe400078efcff */
        // /*0120*/                   VOTE.ANY R0, PT, PT ;                             /* 0x0000000000007806 */
        // /* 0x000fc400038e0100 */
        // /*0130*/                   FSEL R9, -R6, R6, P0 ;                            /* 0x0000000606097208 */
        // /* 0x000fe40000000100 */
        // /*0140*/               @P1 LOP3.LUT R9, R8, 0x7fc00000, RZ, 0xfc, !PT ;      /* 0x7fc0000008091812 */
        // /* 0x000fc800078efcff */
        // /*0150*/                   CREDUX.MAXABS.F32.NAN UR6, R9 ;                   /* 0x00000000090672cc */
        // /* 0x000fe20000006400 */
        // /*0160*/                   IMAD.WIDE R2, R7, 0x4, R4 ;                       /* 0x0000000407027825 */
        // /* 0x001fd800078e0204 */
        // /*0170*/                   IMAD.U32 R5, RZ, RZ, UR6 ;                        /* 0x00000006ff057e24 */
        // /* 0x000fca000f8e00ff */
        // /*0180*/                   STG.E desc[UR4][R2.64], R5 ;                      /* 0x0000000502007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ float redux_sync_max_abs_nan_f32(uint32_t mask, float a) {
    float d;
    asm volatile("redux.sync.max.abs.NaN.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "r"(mask));
    return d;
}

extern "C" __global__ void redux_sync_forms_kernel(
    const uint32_t* __restrict__ in,
    uint32_t* __restrict__ out,
    uint32_t seed) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t lane = (uint32_t)(threadIdx.x & 31);

    uint32_t mask = __activemask();

    uint32_t a = ((const volatile uint32_t*)in)[tid];
    uint32_t u = a ^ (seed + lane * 17u);
    int32_t s = (int32_t)(u ^ 0x80000000u);

    float f = __uint_as_float((u & 0x007fffffu) | 0x3f000000u);
    float f_signed = (lane & 1u) ? -f : f;
    float f_nan = __uint_as_float(0x7fc00000u | (u & 0x003fffffu));
    float f_mix = (lane & 2u) ? f_nan : f_signed;

    uint32_t acc = 0;
    acc ^= redux_sync_add_u32(mask, u);
    acc ^= (uint32_t)redux_sync_add_s32(mask, s);

    acc ^= redux_sync_min_u32(mask, u);
    acc ^= (uint32_t)redux_sync_min_s32(mask, s);

    acc ^= redux_sync_max_u32(mask, u);
    acc ^= (uint32_t)redux_sync_max_s32(mask, s);

    acc ^= redux_sync_and_b32(mask, u);
    acc ^= redux_sync_or_b32(mask, u);
    acc ^= redux_sync_xor_b32(mask, u);

    acc ^= __float_as_uint(redux_sync_min_f32(mask, f_signed));
    acc ^= __float_as_uint(redux_sync_max_f32(mask, f_signed));
    acc ^= __float_as_uint(redux_sync_min_abs_f32(mask, f_signed));
    acc ^= __float_as_uint(redux_sync_max_abs_f32(mask, f_signed));
    acc ^= __float_as_uint(redux_sync_min_nan_f32(mask, f_mix));
    acc ^= __float_as_uint(redux_sync_max_nan_f32(mask, f_mix));
    acc ^= __float_as_uint(redux_sync_min_abs_nan_f32(mask, f_mix));
    acc ^= __float_as_uint(redux_sync_max_abs_nan_f32(mask, f_mix));

    out[tid] = acc;
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    constexpr int N = 256;

    uint32_t* in;
    uint32_t* out;

    ck(cudaMallocManaged(&in, N * sizeof(uint32_t)), "cudaMallocManaged in");
    ck(cudaMallocManaged(&out, N * sizeof(uint32_t)), "cudaMallocManaged out");

    for (int i = 0; i < N; ++i) {
        in[i] = (uint32_t)((i * 11 + 5) ^ 0xa5a5c33cu);
        out[i] = 0u;
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    redux_sync_forms_kernel<<<grid, block>>>(in, out, 7u);

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("acc=%u\n", (unsigned)out[0]);

    cudaFree(in);
    cudaFree(out);
    return 0;
}
