// ld_acquire.cu
//
// Covers:
// ld.acquire.scope{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type d, [a]{, cache-policy};

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

        // /*0090*/                   LDG.E.STRONG.SM R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea4000c1eb900 */
        // /*00a0*/                   NOP ;                                   /* 0x0000000000007918 */
        // /* 0x004fe20000000000 */
        // /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;             /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R3 ;            /* 0x0000000304007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t ld_acquire_cta_global_u32(const uint32_t* p) {
    uint32_t d;
    asm volatile("ld.acquire.cta.global.u32 %0, [%1];" : "=r"(d) : "l"(p));
    return d;
}

        // /*0090*/                   LDG.E.STRONG.GPU R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea8000c1ef900 */
        // /*00a0*/                   CCTL.IVALL ;                             /* 0x00000000ff00798f */
        // /* 0x004fe20002000000 */
        // /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R3 ;             /* 0x0000000304007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t ld_acquire_gpu_global_u32(const uint32_t* p) {
    uint32_t d;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(d) : "l"(p));
    return d;
}

        // /*0090*/                   LDG.E.STRONG.SYS R3, desc[UR4][R2.64] ;  /* 0x0000000402037981 */
        // /* 0x002ea8000c1f5900 */
        // /*00a0*/                   CCTL.IVALL ;                             /* 0x00000000ff00798f */
        // /* 0x004fe20002000000 */
        // /*00b0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;              /* 0x0000000407047825 */
        // /* 0x008fca00078e0204 */
        // /*00c0*/                   STG.E desc[UR4][R4.64], R3 ;             /* 0x0000000304007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ uint32_t ld_acquire_sys_global_u32(const uint32_t* p) {
    uint32_t d;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(d) : "l"(p));
    return d;
}

extern "C" __global__ void ld_acquire_kernel(
    const uint32_t* __restrict__ in,
    uint32_t* __restrict__ out) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t* p = in + tid;

    uint32_t acc = 0;
    acc ^= ld_acquire_cta_global_u32(p);
    acc ^= ld_acquire_gpu_global_u32(p);
    acc ^= ld_acquire_sys_global_u32(p);

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
        in[i] = (uint32_t)((i * 29 + 11) ^ 0x2468ace0u);
        out[i] = 0u;
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    ld_acquire_kernel<<<grid, block>>>(in, out);

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("acc=%u\n", (unsigned)out[0]);

    cudaFree(in);
    cudaFree(out);
    return 0;
}
