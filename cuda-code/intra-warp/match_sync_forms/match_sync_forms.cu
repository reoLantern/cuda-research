// match_sync_forms.cu
//
// PTX match.sync forms:
//   match.any.sync.b32  d, a, membermask;
//   match.any.sync.b64  d, a, membermask;
//   match.all.sync.b32  d, a, membermask;
//   match.all.sync.b64  d, a, membermask;
//   match.all.sync.b32  d|p, a, membermask;
//   match.all.sync.b64  d|p, a, membermask;
//
// d is a 32-bit lane mask, a is .b32 or .b64.

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// ---- wrappers: each PTX match.sync form in its own function ----

        // /*00a0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;   /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   LOP3.LUT R0, R0, 0x1f, RZ, 0xc0, !PT ;  /* 0x0000001f00007812 */
        // /* 0x000fca00078ec0ff */
        // /*00c0*/                   IMAD R9, R0, 0x9e37, R9 ;               /* 0x00009e3700097824 */
        // /* 0x008fe200078e0209 */
        // /*00d0*/                   VOTE.ANY R0, PT, PT ;                   /* 0x0000000000007806 */
        // /* 0x000fe200038e0100 */
        // /*00e0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;             /* 0x0000000407047825 */
        // /* 0x001fc600078e0204 */
        // /*00f0*/                   LOP3.LUT R9, R2, R9, RZ, 0x3c, !PT ;    /* 0x0000000902097212 */
        // /* 0x004fcc00078e3cff */
        // /*0100*/                   MATCH.ANY R9, R9 ;                      /* 0x00000000090973a1 */
        // /* 0x000e2400000e8000 */
        // /*0110*/                   STG.E desc[UR4][R4.64], R9 ;            /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ uint32_t match_any_sync_b32(uint32_t mask, uint32_t a) {
    uint32_t d;
    asm volatile("match.any.sync.b32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(mask));
    return d;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;            /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   HFMA2 R5, -RZ, RZ, 0, 4.17232513427734375e-07 ;  /* 0x00000007ff057431 */
        // /* 0x000fe200000001ff */
        // /*00b0*/                   LOP3.LUT R0, R0, 0x1f, RZ, 0xc0, !PT ;           /* 0x0000001f00007812 */
        // /* 0x000fe400078ec0ff */
        // /*00c0*/                   VOTE.ANY R11, PT, PT ;                           /* 0x00000000000b7806 */
        // /* 0x000fc600038e0100 */
        // /*00d0*/                   IMAD R4, R0.reuse, R5, 0x1 ;                     /* 0x0000000100047424 */
        // /* 0x040fe400078e0205 */
        // /*00e0*/                   IMAD R5, R0, 0x9e37, R7 ;                        /* 0x00009e3700057824 */
        // /* 0x008fc600078e0207 */
        // /*00f0*/                   LOP3.LUT R4, R4, R7, RZ, 0x3c, !PT ;             /* 0x0000000704047212 */
        // /* 0x000fe400078e3cff */
        // /*0100*/                   LDC.64 R6, c[0x0][0x388] ;                       /* 0x0000e200ff067b82 */
        // /* 0x000e220000000a00 */
        // /*0110*/                   LOP3.LUT R5, R2, R5, RZ, 0x3c, !PT ;             /* 0x0000000502057212 */
        // /* 0x004fe200078e3cff */
        // /*0120*/                   IMAD.WIDE R2, R9, 0x4, R6 ;                      /* 0x0000000409027825 */
        // /* 0x001fcc00078e0206 */
        // /*0130*/                   MATCH.ANY.U64 R5, R4 ;                           /* 0x00000000040573a1 */
        // /* 0x000e2400000e8200 */
        // /*0140*/                   STG.E desc[UR4][R2.64], R5 ;                     /* 0x0000000502007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ uint32_t match_any_sync_b64(uint32_t mask, uint64_t a) {
    uint32_t d;
    asm volatile("match.any.sync.b64 %0, %1, %2;" : "=r"(d) : "l"(a), "r"(mask));
    return d;
}

        // /*00a0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;   /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   LOP3.LUT R0, R0, 0x1f, RZ, 0xc0, !PT ;  /* 0x0000001f00007812 */
        // /* 0x000fca00078ec0ff */
        // /*00c0*/                   IMAD R9, R0, 0x9e37, R9 ;               /* 0x00009e3700097824 */
        // /* 0x008fe200078e0209 */
        // /*00d0*/                   VOTE.ANY R0, PT, PT ;                   /* 0x0000000000007806 */
        // /* 0x000fe200038e0100 */
        // /*00e0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;             /* 0x0000000407047825 */
        // /* 0x001fc600078e0204 */
        // /*00f0*/                   LOP3.LUT R9, R2, R9, RZ, 0x3c, !PT ;    /* 0x0000000902097212 */
        // /* 0x004fcc00078e3cff */
        // /*0100*/                   MATCH.ALL PT, R9, R9 ;                  /* 0x00000000090973a1 */
        // /* 0x000e2400000e0000 */
        // /*0110*/                   STG.E desc[UR4][R4.64], R9 ;            /* 0x0000000904007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ uint32_t match_all_sync_b32(uint32_t mask, uint32_t a) {
    uint32_t d;
    asm volatile("match.all.sync.b32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(mask));
    return d;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;            /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   HFMA2 R5, -RZ, RZ, 0, 4.17232513427734375e-07 ;  /* 0x00000007ff057431 */
        // /* 0x000fe200000001ff */
        // /*00b0*/                   LOP3.LUT R0, R0, 0x1f, RZ, 0xc0, !PT ;           /* 0x0000001f00007812 */
        // /* 0x000fe400078ec0ff */
        // /*00c0*/                   VOTE.ANY R11, PT, PT ;                           /* 0x00000000000b7806 */
        // /* 0x000fc600038e0100 */
        // /*00d0*/                   IMAD R4, R0.reuse, R5, 0x1 ;                     /* 0x0000000100047424 */
        // /* 0x040fe400078e0205 */
        // /*00e0*/                   IMAD R5, R0, 0x9e37, R7 ;                        /* 0x00009e3700057824 */
        // /* 0x008fc600078e0207 */
        // /*00f0*/                   LOP3.LUT R4, R4, R7, RZ, 0x3c, !PT ;             /* 0x0000000704047212 */
        // /* 0x000fe400078e3cff */
        // /*0100*/                   LDC.64 R6, c[0x0][0x388] ;                       /* 0x0000e200ff067b82 */
        // /* 0x000e220000000a00 */
        // /*0110*/                   LOP3.LUT R5, R2, R5, RZ, 0x3c, !PT ;             /* 0x0000000502057212 */
        // /* 0x004fe200078e3cff */
        // /*0120*/                   IMAD.WIDE R2, R9, 0x4, R6 ;                      /* 0x0000000409027825 */
        // /* 0x001fcc00078e0206 */
        // /*0130*/                   MATCH.ALL.U64 PT, R5, R4 ;                       /* 0x00000000040573a1 */
        // /* 0x000e2400000e0200 */
        // /*0140*/                   STG.E desc[UR4][R2.64], R5 ;                     /* 0x0000000502007986 */
        // /* 0x001fe2000c101904 */
__device__ __forceinline__ uint32_t match_all_sync_b64(uint32_t mask, uint64_t a) {
    uint32_t d;
    asm volatile("match.all.sync.b64 %0, %1, %2;" : "=r"(d) : "l"(a), "r"(mask));
    return d;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;   /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   LOP3.LUT R0, R0, 0x1f, RZ, 0xc0, !PT ;  /* 0x0000001f00007812 */
        // /* 0x000fca00078ec0ff */
        // /*00b0*/                   IMAD R5, R0, 0x9e37, R5 ;               /* 0x00009e3700057824 */
        // /* 0x008fe200078e0205 */
        // /*00c0*/                   VOTE.ANY R0, PT, PT ;                   /* 0x0000000000007806 */
        // /* 0x000fc800038e0100 */
        // /*00d0*/                   LOP3.LUT R9, R2, R5, RZ, 0x3c, !PT ;    /* 0x0000000502097212 */
        // /* 0x004fe400078e3cff */
        // /*00e0*/                   LDC.64 R4, c[0x0][0x388] ;              /* 0x0000e200ff047b82 */
        // /* 0x000e300000000a00 */
        // /*00f0*/                   MATCH.ALL P0, R0, R9 ;                  /* 0x00000000090073a1 */
        // /* 0x000e620000000000 */
        // /*0100*/                   IMAD.WIDE R4, R7, 0x4, R4 ;             /* 0x0000000407047825 */
        // /* 0x001fe200078e0204 */
        // /*0110*/                   SEL R11, RZ, 0x2, !P0 ;                 /* 0x00000002ff0b7807 */
        // /* 0x002fc80004000000 */
        // /*0120*/                   LOP3.LUT R11, R11, R0, RZ, 0x3c, !PT ;  /* 0x000000000b0b7212 */
        // /* 0x000fca00078e3cff */
        // /*0130*/                   STG.E desc[UR4][R4.64], R11 ;           /* 0x0000000b04007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t match_all_sync_b32_p(
    uint32_t mask, uint32_t a, uint32_t* pred_out) {
    uint32_t d;
    uint32_t pred;
    asm volatile("{\n\t.reg .pred p;\n\tmatch.all.sync.b32 %0|p, %2, %3;\n\tselp.u32 %1, 1, 0, p;\n}"
                 : "=r"(d), "=r"(pred)
                 : "r"(a), "r"(mask));
    *pred_out = pred;
    return d;
}

        // /*0090*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;            /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00a0*/                   HFMA2 R5, -RZ, RZ, 0, 4.17232513427734375e-07 ;  /* 0x00000007ff057431 */
        // /* 0x000fe200000001ff */
        // /*00b0*/                   LOP3.LUT R0, R0, 0x1f, RZ, 0xc0, !PT ;           /* 0x0000001f00007812 */
        // /* 0x000fe400078ec0ff */
        // /*00c0*/                   VOTE.ANY R11, PT, PT ;                           /* 0x00000000000b7806 */
        // /* 0x000fc600038e0100 */
        // /*00d0*/                   IMAD R4, R0.reuse, R5, 0x1 ;                     /* 0x0000000100047424 */
        // /* 0x040fe400078e0205 */
        // /*00e0*/                   IMAD R5, R0, 0x9e37, R7 ;                        /* 0x00009e3700057824 */
        // /* 0x008fc600078e0207 */
        // /*00f0*/                   LOP3.LUT R4, R4, R7, RZ, 0x3c, !PT ;             /* 0x0000000704047212 */
        // /* 0x000fe400078e3cff */
        // /*0100*/                   LDC.64 R6, c[0x0][0x388] ;                       /* 0x0000e200ff067b82 */
        // /* 0x000e220000000a00 */
        // /*0110*/                   LOP3.LUT R5, R2, R5, RZ, 0x3c, !PT ;             /* 0x0000000502057212 */
        // /* 0x004fe200078e3cff */
        // /*0120*/                   IMAD.WIDE R2, R9, 0x4, R6 ;                      /* 0x0000000409027825 */
        // /* 0x001fcc00078e0206 */
        // /*0130*/                   MATCH.ALL.U64 P0, R4, R4 ;                       /* 0x00000000040473a1 */
        // /* 0x000e240000000200 */
        // /*0140*/                   SEL R13, RZ, 0x2, !P0 ;                          /* 0x00000002ff0d7807 */
        // /* 0x001fc80004000000 */
        // /*0150*/                   LOP3.LUT R13, R13, R4, RZ, 0x3c, !PT ;           /* 0x000000040d0d7212 */
        // /* 0x000fca00078e3cff */
        // /*0160*/                   STG.E desc[UR4][R2.64], R13 ;                    /* 0x0000000d02007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t match_all_sync_b64_p(
    uint32_t mask, uint64_t a, uint32_t* pred_out) {
    uint32_t d;
    uint32_t pred;
    asm volatile("{\n\t.reg .pred p;\n\tmatch.all.sync.b64 %0|p, %2, %3;\n\tselp.u32 %1, 1, 0, p;\n}"
                 : "=r"(d), "=r"(pred)
                 : "l"(a), "r"(mask));
    *pred_out = pred;
    return d;
}

extern "C" __global__ void match_sync_forms_kernel(
    const uint32_t* __restrict__ in,
    uint32_t* __restrict__ out,
    uint32_t seed) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t lane = (uint32_t)(threadIdx.x & 31);

    uint32_t mask = __activemask();

    uint32_t a32 = in[tid] ^ (seed + lane * 0x9e37u);
    uint64_t a64 = (uint64_t)a32 << 32 | (uint64_t)(seed ^ (lane * 7u + 1u));

    uint32_t a32_all = seed + 0x1234u;
    uint64_t a64_all = ((uint64_t)a32_all << 32) | (uint64_t)a32_all;

    uint32_t acc = 0;
    acc ^= match_any_sync_b32(mask, a32);
    acc ^= match_any_sync_b64(mask, a64);

    acc ^= match_all_sync_b32(mask, a32_all);
    acc ^= match_all_sync_b64(mask, a64_all);

    uint32_t pred = 0;
    acc ^= match_all_sync_b32_p(mask, a32_all, &pred);
    acc ^= (pred << 1);
    acc ^= match_all_sync_b64_p(mask, a64_all, &pred);
    acc ^= (pred << 2);

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
    match_sync_forms_kernel<<<grid, block>>>(in, out, 7u);

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("acc=%u\n", (unsigned)out[0]);

    cudaFree(in);
    cudaFree(out);
    return 0;
}
