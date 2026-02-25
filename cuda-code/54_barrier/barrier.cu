// barrier.cu
//
// PTX bar/barrier forms (see ptx.barrier.md):
//   barrier{.cta}.sync{.aligned}      a{, b};
//   barrier{.cta}.arrive{.aligned}    a, b;
//   barrier{.cta}.red.popc{.aligned}.u32  d, a{, b}, {!}c;
//   barrier{.cta}.red.op{.aligned}.pred   p, a{, b}, {!}c;  (op = .and, .or)
//   bar{.cta}.sync      a{, b};
//   bar{.cta}.arrive    a, b;
//   bar{.cta}.red.popc.u32  d, a{, b}, {!}c;
//   bar{.cta}.red.op.pred   p, a{, b}, {!}c;

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// ---- wrappers ----
        // /*0020*/                   S2UR UR6, SR_CTAID.X ;                 /* 0x00000000000679c3 */
        // /* 0x000e220000002500 */
        // /*0030*/                   LDCU.64 UR4, c[0x0][0x358] ;           /* 0x00006b00ff0477ac */
        // /* 0x000e6e0008000a00 */
        // /*0040*/                   BAR.SYNC.DEFER_BLOCKING 0x0 ;          /* 0x0000000000007b1d */
        // /* 0x000ff00000010000 */
        // /*0050*/                   LDC R7, c[0x0][0x360] ;                /* 0x0000d800ff077b82 */
        // /* 0x000e300000000800 */
        // /*0060*/                   LDC.64 R2, c[0x0][0x380] ;             /* 0x0000e000ff027b82 */
        // /* 0x000eb00000000a00 */
        // /*0070*/                   LDC.64 R4, c[0x0][0x388] ;             /* 0x0000e200ff047b82 */
        // /* 0x000ee20000000a00 */
        // /*0080*/                   IMAD R7, R7, UR6, R0 ;                 /* 0x0000000607077c24 */
        // /* 0x001fc8000f8e0200 */
        // /*0090*/                   IMAD.WIDE R2, R7, 0x4, R2 ;            /* 0x0000000407027825 */
        // /* 0x004fcc00078e0202 */
        // /*00a0*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R3 ;           /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ void barrier_cta_sync(uint32_t bar) {
    asm volatile("barrier.cta.sync %0;" :: "r"(bar) : "memory");
}
        // /*0090*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   LDCU UR6, c[0x0][0x364] ;              /* 0x00006c80ff0677ac */
        // /* 0x000e220008000800 */
        // /*00b0*/                   LDC R9, c[0x0][0x368] ;                /* 0x0000da00ff097b82 */
        // /* 0x000e620000000800 */
        // /*00c0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fc800078e0204 */
        // /*00d0*/                   IMAD R0, R0, UR6, RZ ;                 /* 0x0000000600007c24 */
        // /* 0x001fc8000f8e02ff */
        // /*00e0*/                   IMAD R0, R0, R9, -0x2 ;                /* 0xfffffffe00007424 */
        // /* 0x002fca00078e0209 */
        // /*00f0*/                   BAR.SYNC.DEFER_BLOCKING 0x1, R0 ;      /* 0x004000000000791d */
        // /* 0x000fec0000010000 */
        // /*0100*/                   STG.E desc[UR4][R4.64], R3 ;           /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ void barrier_cta_sync_count(uint32_t bar, uint32_t count) {
    asm volatile("barrier.cta.sync %0, %1;" :: "r"(bar), "r"(count) : "memory");
}
        // /*0090*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   LDCU UR6, c[0x0][0x364] ;              /* 0x00006c80ff0677ac */
        // /* 0x000e220008000800 */
        // /*00b0*/                   LDC R9, c[0x0][0x368] ;                /* 0x0000da00ff097b82 */
        // /* 0x000e620000000800 */
        // /*00c0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fc800078e0204 */
        // /*00d0*/                   IMAD R0, R0, UR6, RZ ;                 /* 0x0000000600007c24 */
        // /* 0x001fc8000f8e02ff */
        // /*00e0*/                   IMAD R0, R0, R9, -0x2 ;                /* 0xfffffffe00007424 */
        // /* 0x002fca00078e0209 */
        // /*00f0*/                   BAR.SYNC.DEFER_BLOCKING 0x2, R0 ;      /* 0x008000000000791d */
        // /* 0x000fec0000010000 */
        // /*0100*/                   STG.E desc[UR4][R4.64], R3 ;           /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ void barrier_cta_sync_aligned(uint32_t bar, uint32_t count) {
    asm volatile("barrier.cta.sync.aligned %0, %1;" :: "r"(bar), "r"(count) : "memory");
}
        // /*0090*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   LDCU UR6, c[0x0][0x364] ;              /* 0x00006c80ff0677ac */
        // /* 0x000e220008000800 */
        // /*00b0*/                   LDC R9, c[0x0][0x368] ;                /* 0x0000da00ff097b82 */
        // /* 0x000e620000000800 */
        // /*00c0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fc800078e0204 */
        // /*00d0*/                   IMAD R0, R0, UR6, RZ ;                 /* 0x0000000600007c24 */
        // /* 0x001fc8000f8e02ff */
        // /*00e0*/                   IMAD R0, R0, R9, -0x2 ;                /* 0xfffffffe00007424 */
        // /* 0x002fca00078e0209 */
        // /*00f0*/                   BAR.ARV 0x3, R0 ;                      /* 0x00c000000000791d */
        // /* 0x000fec0000002000 */
        // /*0100*/                   STG.E desc[UR4][R4.64], R3 ;           /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ void barrier_cta_arrive(uint32_t bar, uint32_t count) {
    asm volatile("barrier.cta.arrive %0, %1;" :: "r"(bar), "r"(count) : "memory");
}
        // /*0090*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   LDCU UR6, c[0x0][0x364] ;              /* 0x00006c80ff0677ac */
        // /* 0x000e220008000800 */
        // /*00b0*/                   LDC R9, c[0x0][0x368] ;                /* 0x0000da00ff097b82 */
        // /* 0x000e620000000800 */
        // /*00c0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fc800078e0204 */
        // /*00d0*/                   IMAD R0, R0, UR6, RZ ;                 /* 0x0000000600007c24 */
        // /* 0x001fc8000f8e02ff */
        // /*00e0*/                   IMAD R0, R0, R9, -0x2 ;                /* 0xfffffffe00007424 */
        // /* 0x002fca00078e0209 */
        // /*00f0*/                   BAR.ARV 0x4, R0 ;                      /* 0x010000000000791d */
        // /* 0x000fec0000002000 */
        // /*0100*/                   STG.E desc[UR4][R4.64], R3 ;           /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ void barrier_cta_arrive_aligned(uint32_t bar, uint32_t count) {
    asm volatile("barrier.cta.arrive.aligned %0, %1;" :: "r"(bar), "r"(count) : "memory");
}
        // /*0090*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;            /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   LOP3.LUT P0, RZ, R7.reuse, 0x1, RZ, 0xc0, !PT ;  /* 0x0000000107ff7812 */
        // /* 0x040fe2000780c0ff */
        // /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                      /* 0x0000000407047825 */
        // /* 0x008fd800078e0204 */
        // /*00c0*/                   BAR.RED.POPC.DEFER_BLOCKING 0x5, P0 ;            /* 0x0140000000007b1d */
        // /* 0x000fec0000014000 */
        // /*00d0*/                   B2R.RESULT R0 ;                                  /* 0x000000000000731c */
        // /* 0x000ea400000e4000 */
        // /*00e0*/                   LOP3.LUT R7, R3, R0, RZ, 0x3c, !PT ;             /* 0x0000000003077212 */
        // /* 0x004fca00078e3cff */
        // /*00f0*/                   STG.E desc[UR4][R4.64], R7 ;                     /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t barrier_cta_red_popc_u32(uint32_t bar, uint32_t pred_val) {
    uint32_t d;
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %2, 0;\n\tbarrier.cta.red.popc.u32 %0, %1, p;\n}"
                 : "=r"(d)
                 : "r"(bar), "r"(pred_val)
                 : "memory");
    return d;
}
        // /*0090*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;            /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   LOP3.LUT P0, RZ, R7.reuse, 0x1, RZ, 0xc0, !PT ;  /* 0x0000000107ff7812 */
        // /* 0x040fe2000780c0ff */
        // /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                      /* 0x0000000407047825 */
        // /* 0x008fd800078e0204 */
        // /*00c0*/                   BAR.RED.AND.DEFER_BLOCKING 0x6, P0 ;             /* 0x0180000000007b1d */
        // /* 0x000fec0000014400 */
        // /*00d0*/                   B2R.RESULT RZ, P0 ;                              /* 0x0000000000ff731c */
        // /* 0x000e240000004000 */
        // /*00e0*/                   SEL R0, RZ, 0x2, !P0 ;                           /* 0x00000002ff007807 */
        // /* 0x001fc80004000000 */
        // /*00f0*/                   LOP3.LUT R7, R0, R3, RZ, 0x3c, !PT ;             /* 0x0000000300077212 */
        // /* 0x004fca00078e3cff */
        // /*0100*/                   STG.E desc[UR4][R4.64], R7 ;                     /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t barrier_cta_red_and_pred(uint32_t bar, uint32_t pred_val) {
    uint32_t out;
    asm volatile("{\n\t.reg .pred p_in, p_out;\n\tsetp.ne.u32 p_in, %2, 0;\n\tbarrier.cta.red.and.pred p_out, %1, p_in;\n\tselp.u32 %0, 1, 0, p_out;\n}"
                 : "=r"(out)
                 : "r"(bar), "r"(pred_val)
                 : "memory");
    return out;
}
        // /*0090*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;            /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   LOP3.LUT P0, RZ, R7.reuse, 0x1, RZ, 0xc0, !PT ;  /* 0x0000000107ff7812 */
        // /* 0x040fe2000780c0ff */
        // /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                      /* 0x0000000407047825 */
        // /* 0x008fd800078e0204 */
        // /*00c0*/                   BAR.RED.OR.DEFER_BLOCKING 0x7, P0 ;              /* 0x01c0000000007b1d */
        // /* 0x000fec0000014800 */
        // /*00d0*/                   B2R.RESULT RZ, P0 ;                              /* 0x0000000000ff731c */
        // /* 0x000e240000004000 */
        // /*00e0*/                   SEL R0, RZ, 0x4, !P0 ;                           /* 0x00000004ff007807 */
        // /* 0x001fc80004000000 */
        // /*00f0*/                   LOP3.LUT R7, R0, R3, RZ, 0x3c, !PT ;             /* 0x0000000300077212 */
        // /* 0x004fca00078e3cff */
        // /*0100*/                   STG.E desc[UR4][R4.64], R7 ;                     /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t barrier_cta_red_or_pred(uint32_t bar, uint32_t pred_val) {
    uint32_t out;
    asm volatile("{\n\t.reg .pred p_in, p_out;\n\tsetp.ne.u32 p_in, %2, 0;\n\tbarrier.cta.red.or.pred p_out, %1, p_in;\n\tselp.u32 %0, 1, 0, p_out;\n}"
                 : "=r"(out)
                 : "r"(bar), "r"(pred_val)
                 : "memory");
    return out;
}
        // /*0090*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   LDCU UR6, c[0x0][0x364] ;              /* 0x00006c80ff0677ac */
        // /* 0x000e220008000800 */
        // /*00b0*/                   LDC R9, c[0x0][0x368] ;                /* 0x0000da00ff097b82 */
        // /* 0x000e620000000800 */
        // /*00c0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fc800078e0204 */
        // /*00d0*/                   IMAD R0, R0, UR6, RZ ;                 /* 0x0000000600007c24 */
        // /* 0x001fc8000f8e02ff */
        // /*00e0*/                   IMAD R0, R0, R9, -0x2 ;                /* 0xfffffffe00007424 */
        // /* 0x002fca00078e0209 */
        // /*00f0*/                   BAR.SYNC.DEFER_BLOCKING 0x8, R0 ;      /* 0x020000000000791d */
        // /* 0x000fec0000010000 */
        // /*0100*/                   STG.E desc[UR4][R4.64], R3 ;           /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ void bar_sync(uint32_t bar, uint32_t count) {
    asm volatile("bar.sync %0, %1;" :: "r"(bar), "r"(count) : "memory");
}
        // /*0090*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   LDCU UR6, c[0x0][0x364] ;              /* 0x00006c80ff0677ac */
        // /* 0x000e220008000800 */
        // /*00b0*/                   LDC R9, c[0x0][0x368] ;                /* 0x0000da00ff097b82 */
        // /* 0x000e620000000800 */
        // /*00c0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;            /* 0x0000000407047825 */
        // /* 0x008fc800078e0204 */
        // /*00d0*/                   IMAD R0, R0, UR6, RZ ;                 /* 0x0000000600007c24 */
        // /* 0x001fc8000f8e02ff */
        // /*00e0*/                   IMAD R0, R0, R9, -0x2 ;                /* 0xfffffffe00007424 */
        // /* 0x002fca00078e0209 */
        // /*00f0*/                   BAR.ARV 0x9, R0 ;                      /* 0x024000000000791d */
        // /* 0x000fec0000002000 */
        // /*0100*/                   STG.E desc[UR4][R4.64], R3 ;           /* 0x0000000304007986 */
        // /* 0x004fe2000c101904 */
__device__ __forceinline__ void bar_arrive(uint32_t bar, uint32_t count) {
    asm volatile("bar.arrive %0, %1;" :: "r"(bar), "r"(count) : "memory");
}
        // /*0090*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;            /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   LOP3.LUT P0, RZ, R7.reuse, 0x1, RZ, 0xc0, !PT ;  /* 0x0000000107ff7812 */
        // /* 0x040fe2000780c0ff */
        // /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                      /* 0x0000000407047825 */
        // /* 0x008fd800078e0204 */
        // /*00c0*/                   BAR.RED.POPC.DEFER_BLOCKING 0xa, P0 ;            /* 0x0280000000007b1d */
        // /* 0x000fec0000014000 */
        // /*00d0*/                   B2R.RESULT R0 ;                                  /* 0x000000000000731c */
        // /* 0x000ea400000e4000 */
        // /*00e0*/                   LOP3.LUT R7, R3, R0, RZ, 0x3c, !PT ;             /* 0x0000000003077212 */
        // /* 0x004fca00078e3cff */
        // /*00f0*/                   STG.E desc[UR4][R4.64], R7 ;                     /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t bar_red_popc_u32(uint32_t bar, uint32_t pred_val) {
    uint32_t d;
    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.u32 p, %2, 0;\n\tbar.red.popc.u32 %0, %1, p;\n}"
                 : "=r"(d)
                 : "r"(bar), "r"(pred_val)
                 : "memory");
    return d;
}
        // /*0090*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;            /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   LOP3.LUT P0, RZ, R7.reuse, 0x1, RZ, 0xc0, !PT ;  /* 0x0000000107ff7812 */
        // /* 0x040fe2000780c0ff */
        // /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                      /* 0x0000000407047825 */
        // /* 0x008fd800078e0204 */
        // /*00c0*/                   BAR.RED.AND.DEFER_BLOCKING 0xb, P0 ;             /* 0x02c0000000007b1d */
        // /* 0x000fec0000014400 */
        // /*00d0*/                   B2R.RESULT RZ, P0 ;                              /* 0x0000000000ff731c */
        // /* 0x000e240000004000 */
        // /*00e0*/                   SEL R0, RZ, 0x8, !P0 ;                           /* 0x00000008ff007807 */
        // /* 0x001fc80004000000 */
        // /*00f0*/                   LOP3.LUT R7, R0, R3, RZ, 0x3c, !PT ;             /* 0x0000000300077212 */
        // /* 0x004fca00078e3cff */
        // /*0100*/                   STG.E desc[UR4][R4.64], R7 ;                     /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t bar_red_and_pred(uint32_t bar, uint32_t pred_val) {
    uint32_t out;
    asm volatile("{\n\t.reg .pred p_in, p_out;\n\tsetp.ne.u32 p_in, %2, 0;\n\tbar.red.and.pred p_out, %1, p_in;\n\tselp.u32 %0, 1, 0, p_out;\n}"
                 : "=r"(out)
                 : "r"(bar), "r"(pred_val)
                 : "memory");
    return out;
}
        // /*0090*/                   LDG.E.CONSTANT R3, desc[UR4][R2.64] ;            /* 0x0000000402037981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   LOP3.LUT P0, RZ, R7.reuse, 0x1, RZ, 0xc0, !PT ;  /* 0x0000000107ff7812 */
        // /* 0x040fe2000780c0ff */
        // /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                      /* 0x0000000407047825 */
        // /* 0x008fd800078e0204 */
        // /*00c0*/                   BAR.RED.OR.DEFER_BLOCKING 0xc, P0 ;              /* 0x0300000000007b1d */
        // /* 0x000fec0000014800 */
        // /*00d0*/                   B2R.RESULT RZ, P0 ;                              /* 0x0000000000ff731c */
        // /* 0x000e240000004000 */
        // /*00e0*/                   SEL R0, RZ, 0x10, !P0 ;                          /* 0x00000010ff007807 */
        // /* 0x001fc80004000000 */
        // /*00f0*/                   LOP3.LUT R7, R0, R3, RZ, 0x3c, !PT ;             /* 0x0000000300077212 */
        // /* 0x004fca00078e3cff */
        // /*0100*/                   STG.E desc[UR4][R4.64], R7 ;                     /* 0x0000000704007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t bar_red_or_pred(uint32_t bar, uint32_t pred_val) {
    uint32_t out;
    asm volatile("{\n\t.reg .pred p_in, p_out;\n\tsetp.ne.u32 p_in, %2, 0;\n\tbar.red.or.pred p_out, %1, p_in;\n\tselp.u32 %0, 1, 0, p_out;\n}"
                 : "=r"(out)
                 : "r"(bar), "r"(pred_val)
                 : "memory");
    return out;
}

extern "C" __global__ void barrier_kernel(
    const uint32_t* __restrict__ in,
    uint32_t* __restrict__ out) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    uint32_t count = (uint32_t)(blockDim.x * blockDim.y * blockDim.z) - 2;
    uint32_t pred_val = (uint32_t)(tid & 1);

    uint32_t acc = 0;

    // barrier_cta_sync(0);
    barrier_cta_sync_count(1, count);
    // barrier_cta_sync_aligned(2, count);

    // barrier_cta_arrive(3, count);
    // barrier_cta_arrive_aligned(4, count);

    // acc ^= barrier_cta_red_popc_u32(5, pred_val);
    // acc ^= barrier_cta_red_and_pred(6, pred_val) << 1;
    // acc ^= barrier_cta_red_or_pred(7, pred_val) << 2;

    // bar_sync(8, count);
    // bar_arrive(9, count);

    // acc ^= bar_red_popc_u32(10, pred_val);
    // acc ^= bar_red_and_pred(11, pred_val) << 3;
    // acc ^= bar_red_or_pred(12, pred_val) << 4;

    out[tid] = acc ^ in[tid];
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
        in[i] = (uint32_t)((i * 7 + 3) ^ 0x5a5a1234u);
        out[i] = 0u;
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    barrier_kernel<<<grid, block>>>(in, out);

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("acc=%u\n", (unsigned)out[0]);

    cudaFree(in);
    cudaFree(out);
    return 0;
}
