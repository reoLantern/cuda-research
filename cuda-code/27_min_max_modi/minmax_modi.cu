// minmax_modi.cu
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

__device__ __forceinline__ float min_f32(float a, float b) {
    float out;
    asm volatile("min.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float min_ftz_f32(float a, float b) {
    float out;
    asm volatile("min.ftz.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float min_nan_f32(float a, float b) {
    float out;
    asm volatile("min.NaN.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float min_ftz_nan_f32(float a, float b) {
    float out;
    asm volatile("min.ftz.NaN.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float min_xorsign_abs_f32(float a, float b) {
    float out;
    asm volatile("min.xorsign.abs.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float min_nan_xorsign_abs_f32(float a, float b) {
    float out;
    asm volatile("min.NaN.xorsign.abs.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float min_ftz_xorsign_abs_f32(float a, float b) {
    float out;
    asm volatile("min.ftz.xorsign.abs.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float min_ftz_nan_xorsign_abs_f32(float a, float b) {
    float out;
    asm volatile("min.ftz.NaN.xorsign.abs.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float max_f32(float a, float b) {
    float out;
    asm volatile("max.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float max_ftz_f32(float a, float b) {
    float out;
    asm volatile("max.ftz.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float max_nan_f32(float a, float b) {
    float out;
    asm volatile("max.NaN.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float max_ftz_nan_f32(float a, float b) {
    float out;
    asm volatile("max.ftz.NaN.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float max_xorsign_abs_f32(float a, float b) {
    float out;
    asm volatile("max.xorsign.abs.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float max_nan_xorsign_abs_f32(float a, float b) {
    float out;
    asm volatile("max.NaN.xorsign.abs.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float max_ftz_xorsign_abs_f32(float a, float b) {
    float out;
    asm volatile("max.ftz.xorsign.abs.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float max_ftz_nan_xorsign_abs_f32(float a, float b) {
    float out;
    asm volatile("max.ftz.NaN.xorsign.abs.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float min3_abs_f32(float a, float b, float c) {
    float out;
    asm volatile("min.abs.f32 %0, %1, %2, %3;" : "=f"(out) : "f"(a), "f"(b), "f"(c));
    return out;
}

__device__ __forceinline__ float min3_ftz_abs_f32(float a, float b, float c) {
    float out;
    asm volatile("min.ftz.abs.f32 %0, %1, %2, %3;" : "=f"(out) : "f"(a), "f"(b), "f"(c));
    return out;
}

__device__ __forceinline__ float min3_nan_abs_f32(float a, float b, float c) {
    float out;
    asm volatile("min.NaN.abs.f32 %0, %1, %2, %3;" : "=f"(out) : "f"(a), "f"(b), "f"(c));
    return out;
}

__device__ __forceinline__ float min3_ftz_nan_abs_f32(float a, float b, float c) {
    float out;
    asm volatile("min.ftz.NaN.abs.f32 %0, %1, %2, %3;" : "=f"(out) : "f"(a), "f"(b), "f"(c));
    return out;
}

extern "C" __global__ void minmax_modi_kernel(
    const float* __restrict__ in_a,
    const float* __restrict__ in_b,
    const float* __restrict__ in_c,
    float* __restrict__ out
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    float a = in_a[tid];
    float b = in_b[tid];
    float c = in_c[tid];

    constexpr int kMin2Out = 8;
    constexpr int kMax2Out = 8;
    constexpr int kMin3Out = 4;
    constexpr int kOutPerThread = kMin2Out + kMax2Out + kMin3Out;

    int o = tid * kOutPerThread;

    out[o + 0] = min_f32(a, b);
    out[o + 1] = min_ftz_f32(a, b);
    out[o + 2] = min_nan_f32(a, b);
    out[o + 3] = min_ftz_nan_f32(a, b);
    out[o + 4] = min_xorsign_abs_f32(a, b);
    out[o + 5] = min_nan_xorsign_abs_f32(a, b);
    out[o + 6] = min_ftz_xorsign_abs_f32(a, b);
    out[o + 7] = min_ftz_nan_xorsign_abs_f32(a, b);

    int om = o + kMin2Out;
    out[om + 0] = max_f32(a, b);
    out[om + 1] = max_ftz_f32(a, b);
    out[om + 2] = max_nan_f32(a, b);
    out[om + 3] = max_ftz_nan_f32(a, b);
    out[om + 4] = max_xorsign_abs_f32(a, b);
    out[om + 5] = max_nan_xorsign_abs_f32(a, b);
    out[om + 6] = max_ftz_xorsign_abs_f32(a, b);
    out[om + 7] = max_ftz_nan_xorsign_abs_f32(a, b);

    int o3 = om + kMax2Out;
    out[o3 + 0] = min3_abs_f32(a, b, c);
    out[o3 + 1] = min3_ftz_abs_f32(a, b, c);
    out[o3 + 2] = min3_nan_abs_f32(a, b, c);
    out[o3 + 3] = min3_ftz_nan_abs_f32(a, b, c);
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

static float f32_from_bits(uint32_t bits) {
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

int main() {
    constexpr int N = 256;
    constexpr int OUT_PER_THREAD = 20;
    constexpr int OUTN = N * OUT_PER_THREAD;
    constexpr int kEdgeN = 8;

    float *in_a, *in_b, *in_c, *out;

    ck(cudaMallocManaged(&in_a, N * sizeof(float)), "cudaMallocManaged in_a");
    ck(cudaMallocManaged(&in_b, N * sizeof(float)), "cudaMallocManaged in_b");
    ck(cudaMallocManaged(&in_c, N * sizeof(float)), "cudaMallocManaged in_c");
    ck(cudaMallocManaged(&out, OUTN * sizeof(float)), "cudaMallocManaged out");

    const float qnan = f32_from_bits(0x7fc00000u);
    const float pos_inf = f32_from_bits(0x7f800000u);
    const float neg_inf = f32_from_bits(0xff800000u);
    const float pos_zero = 0.0f;
    const float neg_zero = f32_from_bits(0x80000000u);
    const float sub_pos = f32_from_bits(0x00000001u);
    const float sub_neg = f32_from_bits(0x80000001u);

    const float a_vals[kEdgeN] = {
        qnan, 1.0f, sub_pos, sub_neg,
        -3.0f, -1.0f, neg_zero, pos_inf
    };
    const float b_vals[kEdgeN] = {
        1.0f, qnan, 1.0f, -1.0f,
        2.0f, -2.0f, pos_zero, neg_inf
    };
    const float c_vals[kEdgeN] = {
        2.0f, 2.0f, 0.5f, -0.5f,
        4.0f, 0.5f, 1.0f, 3.0f
    };

    for (int i = 0; i < N; ++i) {
        if (i < kEdgeN) {
            in_a[i] = a_vals[i];
            in_b[i] = b_vals[i];
            in_c[i] = c_vals[i];
        } else {
            in_a[i] = (float)(i) * 0.25f + 1.0f;
            in_b[i] = (float)(i) * 0.50f + 2.0f;
            in_c[i] = (float)(i) * -0.125f - 0.5f;
        }
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    minmax_modi_kernel<<<grid, block>>>(in_a, in_b, in_c, out);

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    for (int i = 0; i < kEdgeN; ++i) {
        int o = i * OUT_PER_THREAD;
        std::printf("i=%d a=%a b=%a c=%a\n", i, in_a[i], in_b[i], in_c[i]);
        std::printf(
            "  min2 : %a %a %a %a %a %a %a %a\n",
            out[o + 0], out[o + 1], out[o + 2], out[o + 3],
            out[o + 4], out[o + 5], out[o + 6], out[o + 7]
        );
        std::printf(
            "  max2 : %a %a %a %a %a %a %a %a\n",
            out[o + 8], out[o + 9], out[o + 10], out[o + 11],
            out[o + 12], out[o + 13], out[o + 14], out[o + 15]
        );
        std::printf(
            "  min3 : %a %a %a %a\n",
            out[o + 16], out[o + 17], out[o + 18], out[o + 19]
        );
    }

    cudaFree(in_a);
    cudaFree(in_b);
    cudaFree(in_c);
    cudaFree(out);
    return 0;
}
