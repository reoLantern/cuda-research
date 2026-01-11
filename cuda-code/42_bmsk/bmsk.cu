// bmsk.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

        // /*00a0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;            /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;  /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;            /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   BMSK R9, R2, R5 ;                      /* 0x000000050209721b */
        // /* 0x004fca0000000000 */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;           /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int bmsk_clamp_b32(unsigned int a, unsigned int b) {
    unsigned int out;
    asm volatile("bmsk.clamp.b32 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
}

        // /*00a0*/                   LDG.E.CONSTANT R2, desc[UR4][R2.64] ;  /* 0x0000000402027981 */
        // /* 0x002ea2000c1e9900 */
        // /*00b0*/                   IMAD.WIDE R4, R9, 0x4, R4 ;            /* 0x0000000409047825 */
        // /* 0x008fcc00078e0204 */
        // /*00c0*/                   LDG.E.CONSTANT R5, desc[UR4][R4.64] ;  /* 0x0000000404057981 */
        // /* 0x000ea2000c1e9900 */
        // /*00d0*/                   IMAD.WIDE R6, R9, 0x4, R6 ;            /* 0x0000000409067825 */
        // /* 0x001fe200078e0206 */
        // /*00e0*/                   BMSK.W R9, R2, R5 ;                    /* 0x000000050209721b */
        // /* 0x004fca0000000800 */
        // /*00f0*/                   STG.E desc[UR4][R6.64], R9 ;           /* 0x0000000906007986 */
        // /* 0x000fe2000c101904 */
__device__ __forceinline__ unsigned int bmsk_wrap_b32(unsigned int a, unsigned int b) {
    unsigned int out;
    asm volatile("bmsk.wrap.b32 %0, %1, %2;" : "=r"(out) : "r"(a), "r"(b));
    return out;
}

extern "C" __global__ void bmsk_kernel(
    const unsigned int* __restrict__ in_a,
    const unsigned int* __restrict__ in_b,
    unsigned int* __restrict__ out_clamp,
    unsigned int* __restrict__ out_wrap
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    unsigned int a = in_a[tid];
    unsigned int b = in_b[tid];

    out_clamp[tid] = bmsk_clamp_b32(a, b);
    out_wrap[tid] = bmsk_wrap_b32(a, b);
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    constexpr int N = 256;

    unsigned int *in_a;
    unsigned int *in_b;
    unsigned int *out_clamp;
    unsigned int *out_wrap;

    ck(cudaMallocManaged(&in_a, N * sizeof(unsigned int)), "cudaMallocManaged in_a");
    ck(cudaMallocManaged(&in_b, N * sizeof(unsigned int)), "cudaMallocManaged in_b");
    ck(cudaMallocManaged(&out_clamp, N * sizeof(unsigned int)), "cudaMallocManaged out_clamp");
    ck(cudaMallocManaged(&out_wrap, N * sizeof(unsigned int)), "cudaMallocManaged out_wrap");

    for (int i = 0; i < N; ++i) {
        in_a[i] = (unsigned int)(i % 40);
        in_b[i] = (unsigned int)((i % 8) + 1);
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    bmsk_kernel<<<grid, block>>>(in_a, in_b, out_clamp, out_wrap);

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("bmsk_clamp=%u bmsk_wrap=%u\n", out_clamp[0], out_wrap[0]);

    cudaFree(in_a);
    cudaFree(in_b);
    cudaFree(out_clamp);
    cudaFree(out_wrap);
    return 0;
}
