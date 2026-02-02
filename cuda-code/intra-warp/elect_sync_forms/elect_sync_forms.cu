// elect_sync_forms.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// ---- wrappers: each PTX elect.sync form in its own function ----

// SASS (sm_90, isolated kernel):
// /*0060*/                   ELECT PT, UR6, PT ;                      /* 0x000000000006782f */
// /* 0x000fe200038e0000 */
// /*0070*/                   IMAD R7, R7, UR4, R0 ;                   /* 0x0000000407077c24 */
// /* 0x001fe2000f8e0200 */
// /*0080*/                   MOV R9, UR6 ;                            /* 0x0000000600097c02 */
// /* 0x000fe20008000f00 */
// /*0090*/                   ULDC.64 UR4, c[0x0][0x208] ;             /* 0x0000820000047ab9 */
// /* 0x000fc40000000a00 */
// /*00a0*/                   IMAD.WIDE R2, R7, 0x4, R2 ;              /* 0x0000000407027825 */
// /* 0x002fcc00078e0202 */
// /*00b0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
// /* 0x000fe2000c1f5900 */
// /*00c0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
// /* 0x004fca00078e0204 */
// /*00d0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
// /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t elect_sync_laneid(uint32_t mask) {
    uint32_t lane;
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  elect.sync %0|p, %1;\n"
                 "}\n"
                 : "=r"(lane)
                 : "r"(mask));
    return lane;
}

// SASS (sm_90, isolated kernel):
// /*0060*/                   ELECT P0, UR6, PT ;                      /* 0x000000000006782f */
// /* 0x000fc80003800000 */
// /*0070*/                   SEL R9, RZ, 0x40, !P0 ;                  /* 0x00000040ff097807 */
// /* 0x000fe20004000000 */
// /*0080*/                   IMAD R7, R7, UR4, R0 ;                   /* 0x0000000407077c24 */
// /* 0x001fe2000f8e0200 */
// /*0090*/                   MOV R0, UR6 ;                            /* 0x0000000600007c02 */
// /* 0x000fe20008000f00 */
// /*00a0*/                   ULDC.64 UR4, c[0x0][0x208] ;             /* 0x0000820000047ab9 */
// /* 0x000fc60000000a00 */
// /*00b0*/                   LOP3.LUT R9, R9, R0, RZ, 0x3c, !PT ;     /* 0x0000000009097212 */
// /* 0x000fe200078e3cff */
// /*00c0*/                   IMAD.WIDE R2, R7, 0x4, R2 ;              /* 0x0000000407027825 */
// /* 0x002fcc00078e0202 */
// /*00d0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
// /* 0x000fe2000c1f5900 */
// /*00e0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
// /* 0x004fca00078e0204 */
// /*00f0*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
// /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t elect_sync_laneid_pred(uint32_t mask, uint32_t* pred_out) {
    uint32_t lane;
    uint32_t pred;
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  elect.sync %0|p, %2;\n"
                 "  selp.u32 %1, 1, 0, p;\n"
                 "}\n"
                 : "=r"(lane), "=r"(pred)
                 : "r"(mask));
    *pred_out = pred;
    return lane;
}

// SASS (sm_90, isolated kernel):
// /*00a0*/                   LDG.E.STRONG.SYS R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
// /* 0x000ee2000c1f5900 */
// /*00b0*/                   ELECT P0, URZ, PT ;                      /* 0x00000000003f782f */
// /* 0x000fe40003800000 */
// /*00c0*/                   LOP3.LUT R0, R9, 0x1f, R0, 0x78, !PT ;   /* 0x0000001f09007812 */
// /* 0x004fe400078e7800 */
// /*00d0*/                   SEL R9, RZ, 0x1, !P0 ;                   /* 0x00000001ff097807 */
// /* 0x000fe20004000000 */
// /*00e0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
// /* 0x001fc600078e0204 */
// /*00f0*/                   LOP3.LUT R9, R9, R0, R2, 0x96, !PT ;     /* 0x0000000009097212 */
// /* 0x008fca00078e9602 */
// /*0100*/                   STG.E desc[UR4][R4.64], R9 ;             /* 0x0000000904007986 */
// /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t elect_sync_pred_only(uint32_t mask) {
    uint32_t pred;
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  elect.sync _|p, %1;\n"
                 "  selp.u32 %0, 1, 0, p;\n"
                 "}\n"
                 : "=r"(pred)
                 : "r"(mask));
    return pred;
}

extern "C" __global__ void elect_sync_forms_kernel(
    const uint32_t* __restrict__ in,
    uint32_t* __restrict__ out,
    uint32_t seed) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t lane = (uint32_t)(threadIdx.x & 31);

    uint32_t a = ((const volatile uint32_t*)in)[tid];
    uint32_t mask = __activemask();

    uint32_t acc = 0u;
    uint32_t pred = 0u;

    acc ^= elect_sync_laneid(mask);
    acc ^= elect_sync_laneid_pred(mask, &pred) ^ (pred << 6);
    acc ^= elect_sync_pred_only(mask) ^ (a ^ seed ^ lane);

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
    elect_sync_forms_kernel<<<grid, block>>>(in, out, 7u);

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("acc=%u\n", (unsigned)out[0]);

    cudaFree(in);
    cudaFree(out);
    return 0;
}
