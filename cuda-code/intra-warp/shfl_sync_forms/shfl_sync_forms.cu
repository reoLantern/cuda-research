// shfl_sync_forms.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// ---- wrappers: each PTX shfl.sync form in its own function ----

// SASS (sm_103a, isolated kernel):
// /*00b0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
// /* 0x002ea2000c1f5900 */
// /*00c0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
// /* 0x008fe200078e0204 */
// /*00d0*/                   IADD3 R0, PT, PT, R0, UR6, RZ ;          /* 0x0000000600007c10 */
// /* 0x001fc8000fffe0ff */
// /*00e0*/                   LOP3.LUT R0, R0, 0x1f, RZ, 0xc0, !PT ;   /* 0x0000001f00007812 */
// /* 0x000fc800078ec0ff */
// /*00f0*/                   VIMNMX.U32 R0, PT, PT, R0, 0x1, !PT ;    /* 0x0000000100007848 */
// /* 0x000fca0007fe0000 */
// /*0100*/                   SHFL.UP PT, R9, R2, R0, 0x1f ;           /* 0x04001f0002097589 */
// /* 0x004e2800000e0000 */
// /*0110*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
// /* 0x001fe2000c101904 */
__device__ __forceinline__ uint32_t shfl_sync_up_b32(uint32_t mask, uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm volatile("shfl.sync.up.b32 %0, %1, %2, %3, %4;"
                 : "=r"(d)
                 : "r"(a), "r"(b), "r"(c), "r"(mask));
    return d;
}

// SASS (sm_103a, isolated kernel):
// /*00b0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
// /* 0x002ea2000c1f5900 */
// /*00c0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
// /* 0x008fe200078e0204 */
// /*00d0*/                   IADD3 R0, PT, PT, R0, UR6, RZ ;          /* 0x0000000600007c10 */
// /* 0x001fc8000fffe0ff */
// /*00e0*/                   LOP3.LUT R0, R0, 0x1f, RZ, 0xc0, !PT ;   /* 0x0000001f00007812 */
// /* 0x000fc800078ec0ff */
// /*00f0*/                   VIMNMX.U32 R0, PT, PT, R0, 0x1, !PT ;    /* 0x0000000100007848 */
// /* 0x000fca0007fe0000 */
// /*0100*/                   SHFL.DOWN PT, R9, R2, R0, 0x1f ;         /* 0x08001f0002097589 */
// /* 0x004e2800000e0000 */
// /*0110*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
// /* 0x001fe2000c101904 */
__device__ __forceinline__ uint32_t shfl_sync_down_b32(uint32_t mask, uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm volatile("shfl.sync.down.b32 %0, %1, %2, %3, %4;"
                 : "=r"(d)
                 : "r"(a), "r"(b), "r"(c), "r"(mask));
    return d;
}

// SASS (sm_103a, isolated kernel):
// /*00b0*/                   LDG.E.STRONG.SYS R2, desc[UR6][R2.64] ;         /* 0x0000000602027981 */
// /* 0x002ea2000c1f5900 */
// /*00c0*/                   HFMA2 R0, -RZ, RZ, 0, 5.9604644775390625e-08 ;  /* 0x00000001ff007431 */
// /* 0x000fe400000001ff */
// /*00d0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                     /* 0x0000000407047825 */
// /* 0x008fe200078e0204 */
// /*00e0*/                   ULOP3.LUT UR4, UR4, 0x3, URZ, 0xc0, !UPT ;      /* 0x0000000304047892 */
// /* 0x001fcc000f8ec0ff */
// /*00f0*/                   SHF.L.U32 R0, R0, UR4, RZ ;                     /* 0x0000000400007c19 */
// /* 0x000fca00080006ff */
// /*0100*/                   SHFL.BFLY PT, R9, R2, R0, 0x1f ;                /* 0x0c001f0002097589 */
// /* 0x004e2800000e0000 */
// /*0110*/                   STG.E desc[UR6][R4.64], R9 ;                    /* 0x0000000904007986 */
// /* 0x001fe2000c101906 */
__device__ __forceinline__ uint32_t shfl_sync_bfly_b32(uint32_t mask, uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, %3, %4;"
                 : "=r"(d)
                 : "r"(a), "r"(b), "r"(c), "r"(mask));
    return d;
}

// SASS (sm_103a, isolated kernel):
// /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
// /* 0x002ea2000c1f5900 */
// /*00a0*/                   VOTE.ANY R9, PT, PT ;                    /* 0x0000000000097806 */
// /* 0x000fe200038e0100 */
// /*00b0*/                   IMAD R0, R0, 0x3, R5 ;                   /* 0x0000000300007824 */
// /* 0x008fe400078e0205 */
// /*00c0*/                   LDC.64 R4, c[0x0][0x388] ;               /* 0x0000e200ff047b82 */
// /* 0x000e260000000a00 */
// /*00d0*/                   LOP3.LUT R0, R0, 0x1f, RZ, 0xc0, !PT ;   /* 0x0000001f00007812 */
// /* 0x000fe200078ec0ff */
// /*00e0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
// /* 0x001fc800078e0204 */
// /*00f0*/                   SHFL.IDX PT, R9, R2, R0, 0x181f ;        /* 0x00181f0002097589 */
// /* 0x004e2800000e0000 */
// /*0100*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
// /* 0x001fe2000c101904 */
__device__ __forceinline__ uint32_t shfl_sync_idx_b32(uint32_t mask, uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, %3, %4;"
                 : "=r"(d)
                 : "r"(a), "r"(b), "r"(c), "r"(mask));
    return d;
}

// SASS (sm_103a, isolated kernel):
// /*00b0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
// /* 0x002ea2000c1f5900 */
// /*00c0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
// /* 0x008fc800078e0204 */
// /*00d0*/                   VIADD R0, R0, UR6 ;                      /* 0x0000000600007c36 */
// /* 0x001fca0008000000 */
// /*00e0*/                   LOP3.LUT R0, R0, 0x1f, RZ, 0xc0, !PT ;   /* 0x0000001f00007812 */
// /* 0x000fc800078ec0ff */
// /*00f0*/                   VIMNMX.U32 R0, PT, PT, R0, 0x1, !PT ;    /* 0x0000000100007848 */
// /* 0x000fcc0007fe0000 */
// /*0100*/                   SHFL.UP P0, R0, R2, R0, 0x1f ;           /* 0x04001f0002007589 */
// /* 0x004e240000000000 */
// /*0110*/                   SEL R11, RZ, 0x2, !P0 ;                  /* 0x00000002ff0b7807 */
// /* 0x001fc80004000000 */
// /*0120*/                   LOP3.LUT R11, R11, R0, RZ, 0x3c, !PT ;   /* 0x000000000b0b7212 */
// /* 0x000fca00078e3cff */
// /*0130*/                   STG.E desc[UR4][R4.64], R11 ;            /* 0x0000000b04007986 */
// /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t shfl_sync_up_b32_p(
    uint32_t mask, uint32_t a, uint32_t b, uint32_t c, uint32_t* pred_out) {
    uint32_t d;
    uint32_t pred;
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  shfl.sync.up.b32 %0|p, %2, %3, %4, %5;\n"
                 "  selp.u32 %1, 1, 0, p;\n"
                 "}\n"
                 : "=r"(d), "=r"(pred)
                 : "r"(a), "r"(b), "r"(c), "r"(mask));
    *pred_out = pred;
    return d;
}

// SASS (sm_103a, isolated kernel):
// /*00b0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
// /* 0x002ea2000c1f5900 */
// /*00c0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
// /* 0x008fc800078e0204 */
// /*00d0*/                   VIADD R0, R0, UR6 ;                      /* 0x0000000600007c36 */
// /* 0x001fca0008000000 */
// /*00e0*/                   LOP3.LUT R0, R0, 0x1f, RZ, 0xc0, !PT ;   /* 0x0000001f00007812 */
// /* 0x000fc800078ec0ff */
// /*00f0*/                   VIMNMX.U32 R0, PT, PT, R0, 0x1, !PT ;    /* 0x0000000100007848 */
// /* 0x000fcc0007fe0000 */
// /*0100*/                   SHFL.DOWN P0, R0, R2, R0, 0x1f ;         /* 0x08001f0002007589 */
// /* 0x004e240000000000 */
// /*0110*/                   SEL R11, RZ, 0x2, !P0 ;                  /* 0x00000002ff0b7807 */
// /* 0x001fc80004000000 */
// /*0120*/                   LOP3.LUT R11, R11, R0, RZ, 0x3c, !PT ;   /* 0x000000000b0b7212 */
// /* 0x000fca00078e3cff */
// /*0130*/                   STG.E desc[UR4][R4.64], R11 ;            /* 0x0000000b04007986 */
// /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t shfl_sync_down_b32_p(
    uint32_t mask, uint32_t a, uint32_t b, uint32_t c, uint32_t* pred_out) {
    uint32_t d;
    uint32_t pred;
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  shfl.sync.down.b32 %0|p, %2, %3, %4, %5;\n"
                 "  selp.u32 %1, 1, 0, p;\n"
                 "}\n"
                 : "=r"(d), "=r"(pred)
                 : "r"(a), "r"(b), "r"(c), "r"(mask));
    *pred_out = pred;
    return d;
}

// SASS (sm_103a, isolated kernel):
// /*00b0*/                   LDG.E.STRONG.SYS R2, desc[UR6][R2.64] ;         /* 0x0000000602027981 */
// /* 0x002ea2000c1f5900 */
// /*00c0*/                   HFMA2 R0, -RZ, RZ, 0, 5.9604644775390625e-08 ;  /* 0x00000001ff007431 */
// /* 0x000fe400000001ff */
// /*00d0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;                     /* 0x0000000407047825 */
// /* 0x008fe200078e0204 */
// /*00e0*/                   ULOP3.LUT UR4, UR4, 0x3, URZ, 0xc0, !UPT ;      /* 0x0000000304047892 */
// /* 0x001fcc000f8ec0ff */
// /*00f0*/                   SHF.L.U32 R0, R0, UR4, RZ ;                     /* 0x0000000400007c19 */
// /* 0x000fcc00080006ff */
// /*0100*/                   SHFL.BFLY P0, R0, R2, R0, 0x1f ;                /* 0x0c001f0002007589 */
// /* 0x004e240000000000 */
// /*0110*/                   SEL R11, RZ, 0x2, !P0 ;                         /* 0x00000002ff0b7807 */
// /* 0x001fc80004000000 */
// /*0120*/                   LOP3.LUT R11, R11, R0, RZ, 0x3c, !PT ;          /* 0x000000000b0b7212 */
// /* 0x000fca00078e3cff */
// /*0130*/                   STG.E desc[UR6][R4.64], R11 ;                   /* 0x0000000b04007986 */
// /* 0x000fe2000c101906 */
__device__ __forceinline__ uint32_t shfl_sync_bfly_b32_p(
    uint32_t mask, uint32_t a, uint32_t b, uint32_t c, uint32_t* pred_out) {
    uint32_t d;
    uint32_t pred;
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  shfl.sync.bfly.b32 %0|p, %2, %3, %4, %5;\n"
                 "  selp.u32 %1, 1, 0, p;\n"
                 "}\n"
                 : "=r"(d), "=r"(pred)
                 : "r"(a), "r"(b), "r"(c), "r"(mask));
    *pred_out = pred;
    return d;
}

// SASS (sm_103a, isolated kernel):
// /*0090*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
// /* 0x002ea2000c1f5900 */
// /*00a0*/                   VOTE.ANY R9, PT, PT ;                    /* 0x0000000000097806 */
// /* 0x000fe200038e0100 */
// /*00b0*/                   IMAD R0, R0, 0x3, R5 ;                   /* 0x0000000300007824 */
// /* 0x008fe400078e0205 */
// /*00c0*/                   LDC.64 R4, c[0x0][0x388] ;               /* 0x0000e200ff047b82 */
// /* 0x000e260000000a00 */
// /*00d0*/                   LOP3.LUT R0, R0, 0x1f, RZ, 0xc0, !PT ;   /* 0x0000001f00007812 */
// /* 0x000fe200078ec0ff */
// /*00e0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
// /* 0x001fca00078e0204 */
// /*00f0*/                   SHFL.IDX P0, R0, R2, R0, 0x181f ;        /* 0x00181f0002007589 */
// /* 0x004e240000000000 */
// /*0100*/                   SEL R11, RZ, 0x2, !P0 ;                  /* 0x00000002ff0b7807 */
// /* 0x001fc80004000000 */
// /*0110*/                   LOP3.LUT R11, R11, R0, RZ, 0x3c, !PT ;   /* 0x000000000b0b7212 */
// /* 0x000fca00078e3cff */
// /*0120*/                   STG.E desc[UR4][R4.64], R11 ;            /* 0x0000000b04007986 */
// /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t shfl_sync_idx_b32_p(
    uint32_t mask, uint32_t a, uint32_t b, uint32_t c, uint32_t* pred_out) {
    uint32_t d;
    uint32_t pred;
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  shfl.sync.idx.b32 %0|p, %2, %3, %4, %5;\n"
                 "  selp.u32 %1, 1, 0, p;\n"
                 "}\n"
                 : "=r"(d), "=r"(pred)
                 : "r"(a), "r"(b), "r"(c), "r"(mask));
    *pred_out = pred;
    return d;
}

extern "C" __global__ void shfl_sync_forms_kernel(
    const uint32_t* __restrict__ in,
    uint32_t* __restrict__ out,
    uint32_t seed) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t lane = (uint32_t)(threadIdx.x & 31);

    uint32_t a = ((const volatile uint32_t*)in)[tid];
    uint32_t mask = __activemask();

    uint32_t delta = (seed + lane) & 0x1f;
    delta = (delta == 0u) ? 1u : delta;
    uint32_t src_lane = (seed + lane * 3u) & 0x1f;
    uint32_t lane_mask = 1u << (seed & 0x3);

    uint32_t c_full = 0x1f;
    uint32_t c_seg8 = (0x18u << 8) | 0x1f;

    uint32_t acc = 0;
    acc ^= shfl_sync_up_b32(mask, a, delta, c_full);
    acc ^= shfl_sync_down_b32(mask, a, delta, c_full);
    acc ^= shfl_sync_bfly_b32(mask, a, lane_mask, c_full);
    acc ^= shfl_sync_idx_b32(mask, a, src_lane, c_seg8);

    uint32_t pred = 0;
    acc ^= shfl_sync_up_b32_p(mask, a, delta, c_full, &pred);
    acc ^= (pred << 1);
    acc ^= shfl_sync_down_b32_p(mask, a, delta, c_full, &pred);
    acc ^= (pred << 2);
    acc ^= shfl_sync_bfly_b32_p(mask, a, lane_mask, c_full, &pred);
    acc ^= (pred << 3);
    acc ^= shfl_sync_idx_b32_p(mask, a, src_lane, c_seg8, &pred);
    acc ^= (pred << 4);

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
        in[i] = (uint32_t)((i * 7 + 3) ^ 0x5a5a1234u);
        out[i] = 0u;
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    shfl_sync_forms_kernel<<<grid, block>>>(in, out, 7u);

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("acc=%u\n", (unsigned)out[0]);

    cudaFree(in);
    cudaFree(out);
    return 0;
}
