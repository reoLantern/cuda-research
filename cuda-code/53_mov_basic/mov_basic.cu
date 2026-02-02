// mov_basic.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// ---- wrappers: each PTX mov form in its own function ----

__device__ __forceinline__ uint16_t mov_b16_scalar(uint16_t a) {
    uint16_t d;
    asm volatile("mov.b16 %0, %1;" : "=h"(d) : "h"(a));
    return d;
}

__device__ __forceinline__ uint32_t mov_b32_scalar(uint32_t a) {
    uint32_t d;
    asm volatile("mov.b32 %0, %1;" : "=r"(d) : "r"(a));
    return d;
}

__device__ __forceinline__ uint64_t mov_b64_scalar(uint64_t a) {
    uint64_t d;
    asm volatile("mov.b64 %0, %1;" : "=l"(d) : "l"(a));
    return d;
}

__device__ __forceinline__ uint32_t mov_b32_pack_u16(uint16_t a, uint16_t b) {
    uint32_t d;
    asm volatile("mov.b32 %0, {%1, %2};" : "=r"(d) : "h"(a), "h"(b));
    return d;
}

__device__ __forceinline__ uint32_t mov_b32_unpack_u16(uint32_t a) {
    uint16_t lo;
    uint16_t hi;
    asm volatile("mov.b32 {%0, %1}, %2;" : "=h"(lo), "=h"(hi) : "r"(a));
    return (uint32_t)lo | ((uint32_t)hi << 16);
}

__device__ __forceinline__ uint64_t mov_b64_pack_u32(uint32_t a, uint32_t b) {
    uint64_t d;
    asm volatile("mov.b64 %0, {%1, %2};" : "=l"(d) : "r"(a), "r"(b));
    return d;
}

__device__ __forceinline__ uint64_t mov_b64_unpack_u32(uint64_t a) {
    uint32_t lo;
    uint32_t hi;
    asm volatile("mov.b64 {%0, %1}, %2;" : "=r"(lo), "=r"(hi) : "l"(a));
    return (uint64_t)lo | ((uint64_t)hi << 32);
}

__device__ __forceinline__ uint64_t mov_b64_pack_u16x4(
    uint16_t a, uint16_t b, uint16_t c, uint16_t d) {
    uint64_t out;
    asm volatile("mov.b64 %0, {%1, %2, %3, %4};"
                 : "=l"(out)
                 : "h"(a), "h"(b), "h"(c), "h"(d));
    return out;
}

__device__ __forceinline__ uint64_t mov_b64_unpack_u16x4(uint64_t a) {
    uint16_t v0;
    uint16_t v1;
    uint16_t v2;
    uint16_t v3;
    asm volatile("mov.b64 {%0, %1, %2, %3}, %4;"
                 : "=h"(v0), "=h"(v1), "=h"(v2), "=h"(v3)
                 : "l"(a));
    return (uint64_t)v0 | ((uint64_t)v1 << 16) | ((uint64_t)v2 << 32) | ((uint64_t)v3 << 48);
}

extern "C" __global__ void mov_basic_kernel(
    const uint16_t* __restrict__ in_u16,
    const uint32_t* __restrict__ in_u32,
    const uint64_t* __restrict__ in_u64,
    uint64_t* __restrict__ out_acc) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    int idx16 = tid * 4;
    int idx32 = tid * 2;

    uint16_t u16_0 = ((const volatile uint16_t*)in_u16)[idx16 + 0];
    uint16_t u16_1 = ((const volatile uint16_t*)in_u16)[idx16 + 1];
    uint16_t u16_2 = ((const volatile uint16_t*)in_u16)[idx16 + 2];
    uint16_t u16_3 = ((const volatile uint16_t*)in_u16)[idx16 + 3];

    uint32_t u32_0 = ((const volatile uint32_t*)in_u32)[idx32 + 0];
    uint32_t u32_1 = ((const volatile uint32_t*)in_u32)[idx32 + 1];

    uint64_t u64_0 = ((const volatile uint64_t*)in_u64)[tid];

    uint64_t acc = 0ull;
    acc ^= (uint64_t)mov_b16_scalar(u16_0);
    // acc ^= (uint64_t)mov_b32_scalar(u32_0);
    // acc ^= mov_b64_scalar(u64_0);
    // acc ^= (uint64_t)mov_b32_pack_u16(u16_0, u16_1);
    // acc ^= (uint64_t)mov_b32_unpack_u16(u32_0);
    // acc ^= mov_b64_pack_u32(u32_0, u32_1);
    acc ^= mov_b64_unpack_u32(u64_0);
    // acc ^= mov_b64_pack_u16x4(u16_0, u16_1, u16_2, u16_3);
    // acc ^= mov_b64_unpack_u16x4(u64_0);

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

    uint16_t* in_u16;
    uint32_t* in_u32;
    uint64_t* in_u64;
    uint64_t* out_acc;

    ck(cudaMallocManaged(&in_u16, N * 4 * sizeof(uint16_t)), "cudaMallocManaged in_u16");
    ck(cudaMallocManaged(&in_u32, N * 2 * sizeof(uint32_t)), "cudaMallocManaged in_u32");
    ck(cudaMallocManaged(&in_u64, N * sizeof(uint64_t)), "cudaMallocManaged in_u64");
    ck(cudaMallocManaged(&out_acc, N * sizeof(uint64_t)), "cudaMallocManaged out_acc");

    for (int i = 0; i < N * 4; ++i) {
        in_u16[i] = (uint16_t)((i * 3 + 1) ^ 0x5a5u);
    }
    for (int i = 0; i < N * 2; ++i) {
        in_u32[i] = (uint32_t)((i * 7 + 5) ^ 0xa5a51234u);
    }
    for (int i = 0; i < N; ++i) {
        in_u64[i] = ((uint64_t)(i * 11 + 9)) ^ 0x1122334455667788ull;
        out_acc[i] = 0ull;
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    mov_basic_kernel<<<grid, block>>>(in_u16, in_u32, in_u64, out_acc);

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("acc=%llu\n", (unsigned long long)out_acc[0]);

    cudaFree(in_u16);
    cudaFree(in_u32);
    cudaFree(in_u64);
    cudaFree(out_acc);
    return 0;
}
