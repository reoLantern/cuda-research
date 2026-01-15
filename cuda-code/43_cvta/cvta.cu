// cvta.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

__device__ int cvta_global_data[256];
__device__ __constant__ int cvta_const_data[256];

__device__ __forceinline__ unsigned long long cvta_global_var_u64() {
    unsigned long long out;
    asm volatile("cvta.global.u64 %0, cvta_global_data;" : "=l"(out));
    return out;
}

__device__ __forceinline__ unsigned long long cvta_const_var_u64() {
    unsigned long long out;
    asm volatile("cvta.const.u64 %0, cvta_const_data;" : "=l"(out));
    return out;
}

__device__ __forceinline__ unsigned long long cvta_const_offset_u64() {
    unsigned long long out;
    asm volatile("cvta.const.u64 %0, cvta_const_data+4;" : "=l"(out));
    return out;
}

__device__ __forceinline__ unsigned long long cvta_to_global_u64(const void* p) {
    unsigned long long out;
    asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(out) : "l"(p));
    return out;
}

__device__ __forceinline__ unsigned long long cvta_to_const_u64(const void* p) {
    unsigned long long out;
    asm volatile("cvta.to.const.u64 %0, %1;" : "=l"(out) : "l"(p));
    return out;
}

__device__ __forceinline__ unsigned long long cvta_to_shared_cta_u64(const void* p) {
    unsigned long long out;
    asm volatile("cvta.to.shared::cta.u64 %0, %1;" : "=l"(out) : "l"(p));
    return out;
}

__device__ __forceinline__ unsigned long long cvta_to_local_u64(const void* p) {
    unsigned long long out;
    asm volatile("cvta.to.local.u64 %0, %1;" : "=l"(out) : "l"(p));
    return out;
}

extern "C" __global__ void cvta_kernel(
    const int* __restrict__ in_global,
    unsigned long long* __restrict__ out_global_var,
    unsigned long long* __restrict__ out_const_var,
    unsigned long long* __restrict__ out_const_off,
    unsigned long long* __restrict__ out_to_global,
    unsigned long long* __restrict__ out_to_const,
    unsigned long long* __restrict__ out_to_shared,
    unsigned long long* __restrict__ out_to_local
) {
    __shared__ int smem[256];

    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int lane = tid & 255;

    smem[lane] = in_global[tid];

    int local = in_global[tid];
    const int* gptr = &in_global[tid];
    const int* cptr = &cvta_const_data[lane];
    int* sptr = &smem[lane];
    int* lptr = &local;

    out_global_var[tid] = cvta_global_var_u64();
    // out_const_var[tid] = cvta_const_var_u64();
    // out_const_off[tid] = cvta_const_offset_u64();
    // out_to_global[tid] = cvta_to_global_u64(gptr);
    // out_to_const[tid] = cvta_to_const_u64(cptr);
    // out_to_shared[tid] = cvta_to_shared_cta_u64(sptr);
    // out_to_local[tid] = cvta_to_local_u64(lptr);
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    constexpr int N = 256;

    int* in_global;
    unsigned long long* out_global_var;
    unsigned long long* out_const_var;
    unsigned long long* out_const_off;
    unsigned long long* out_to_global;
    unsigned long long* out_to_const;
    unsigned long long* out_to_shared;
    unsigned long long* out_to_local;

    ck(cudaMallocManaged(&in_global, N * sizeof(int)), "cudaMallocManaged in_global");
    ck(cudaMallocManaged(&out_global_var, N * sizeof(unsigned long long)), "cudaMallocManaged out_global_var");
    ck(cudaMallocManaged(&out_const_var, N * sizeof(unsigned long long)), "cudaMallocManaged out_const_var");
    ck(cudaMallocManaged(&out_const_off, N * sizeof(unsigned long long)), "cudaMallocManaged out_const_off");
    ck(cudaMallocManaged(&out_to_global, N * sizeof(unsigned long long)), "cudaMallocManaged out_to_global");
    ck(cudaMallocManaged(&out_to_const, N * sizeof(unsigned long long)), "cudaMallocManaged out_to_const");
    ck(cudaMallocManaged(&out_to_shared, N * sizeof(unsigned long long)), "cudaMallocManaged out_to_shared");
    ck(cudaMallocManaged(&out_to_local, N * sizeof(unsigned long long)), "cudaMallocManaged out_to_local");

    for (int i = 0; i < N; ++i) {
        in_global[i] = i * 3 + 1;
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    cvta_kernel<<<grid, block>>>(
        in_global,
        out_global_var,
        out_const_var,
        out_const_off,
        out_to_global,
        out_to_const,
        out_to_shared,
        out_to_local
    );

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::printf("cvta_global=0x%llx cvta_to_global=0x%llx\n", out_global_var[0], out_to_global[0]);

    cudaFree(in_global);
    cudaFree(out_global_var);
    cudaFree(out_const_var);
    cudaFree(out_const_off);
    cudaFree(out_to_global);
    cudaFree(out_to_const);
    cudaFree(out_to_shared);
    cudaFree(out_to_local);
    return 0;
}
