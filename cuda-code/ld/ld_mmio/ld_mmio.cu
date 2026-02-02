// ld_mmio.cu
//
// Covers:
// ld.mmio.relaxed.sys{.global}.type d, [a];

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

        // /*0080*/                   LDG.E.MMIO.SYS R3, [R2] ;     /* 0x0000000002037381 */
        // /* 0x000ee200001f8900 */
        // /*0090*/                   LDCU.64 UR4, c[0x0][0x358] ;  /* 0x00006b00ff0477ac */
        // /* 0x000ee20008000a00 */
        // /*00a0*/                   IMAD.WIDE R4, R7, 0x4, R4 ;   /* 0x0000000407047825 */
        // /* 0x004fca00078e0204 */
        // /*00b0*/                   STG.E desc[UR4][R4.64], R3 ;  /* 0x0000000304007986 */
        // /* 0x008fe2000c101904 */
__device__ __forceinline__ uint32_t ld_mmio_relaxed_sys_global_u32(const uint32_t* p) {
    uint32_t d;
    asm volatile("ld.mmio.relaxed.sys.global.u32 %0, [%1];" : "=r"(d) : "l"(p));
    return d;
}

extern "C" __global__ void ld_mmio_kernel(
    const uint32_t* __restrict__ in,
    uint32_t* __restrict__ out) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t* p = in + tid;

    uint32_t acc = 0;
    acc ^= ld_mmio_relaxed_sys_global_u32(p);

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
        in[i] = (uint32_t)((i * 31 + 21) ^ 0x11223344u);
        out[i] = 0u;
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    ld_mmio_kernel<<<grid, block>>>(in, out);

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("acc=%u\n", (unsigned)out[0]);

    cudaFree(in);
    cudaFree(out);
    return 0;
}
