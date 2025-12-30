// addsub.cu
#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void addsub_kernel(
    const int*   __restrict__ in_i0,
    const int*   __restrict__ in_i1,
    const float* __restrict__ in_f0,
    const float* __restrict__ in_f1,
    int*   __restrict__ out_i,   // 每线程写 4 个 int
    float* __restrict__ out_f    // 每线程写 4 个 float
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);

    int   a = in_i0[tid];
    int   b = in_i1[tid];
    float x = in_f0[tid];
    float y = in_f1[tid];

    // 1) int add/sub (+ imm)
    int i_add     = a + b;
    int i_sub     = a - b;
    int i_add_imm = a + 1;
    int i_sub_imm = a - 1;

    int oi = tid * 4;
    out_i[oi + 0] = i_add;
    out_i[oi + 1] = i_sub;
    out_i[oi + 2] = i_add_imm;
    out_i[oi + 3] = i_sub_imm;

    // 2) float add/sub (+ imm)
    float f_add     = x + y;
    float f_sub     = x - y;
    float f_add_imm = x + 1.0f;
    float f_sub_imm = x - 1.0f;

    int of = tid * 4;
    out_f[of + 0] = f_add;
    out_f[of + 1] = f_sub;
    out_f[of + 2] = f_add_imm;
    out_f[of + 3] = f_sub_imm;
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

int main() {
    constexpr int N = 256;
    constexpr int OUTN = N * 4;

    int *in_i0, *in_i1, *out_i;
    float *in_f0, *in_f1, *out_f;

    ck(cudaMallocManaged(&in_i0,  N * sizeof(int)),   "cudaMallocManaged in_i0");
    ck(cudaMallocManaged(&in_i1,  N * sizeof(int)),   "cudaMallocManaged in_i1");
    ck(cudaMallocManaged(&in_f0,  N * sizeof(float)), "cudaMallocManaged in_f0");
    ck(cudaMallocManaged(&in_f1,  N * sizeof(float)), "cudaMallocManaged in_f1");
    ck(cudaMallocManaged(&out_i,  OUTN * sizeof(int)),   "cudaMallocManaged out_i");
    ck(cudaMallocManaged(&out_f,  OUTN * sizeof(float)), "cudaMallocManaged out_f");

    for (int i = 0; i < N; ++i) {
        in_i0[i] = i * 3 + 1;
        in_i1[i] = i * 5 + 7;
        in_f0[i] = (float)(i) * 0.25f + 1.0f;
        in_f1[i] = (float)(i) * 0.50f + 2.0f;
    }

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);
    addsub_kernel<<<grid, block>>>(in_i0, in_i1, in_f0, in_f1, out_i, out_f);

    ck(cudaGetLastError(), "kernel launch");
    ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // 随便打印一个，避免“完全没用”的感觉（不影响你看 SASS）
    std::printf("out_i[0]=%d out_f[0]=%f\n", out_i[0], out_f[0]);

    cudaFree(in_i0); cudaFree(in_i1);
    cudaFree(in_f0); cudaFree(in_f1);
    cudaFree(out_i); cudaFree(out_f);
    return 0;
}
