// file: branch_simt.cu
#include <cstdio>

__global__ void branch_simt_kernel(float* out, const float* in, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (tid >= n) return;

    float x = in[tid];
    float acc;

    // warp 内制造 divergence
    if (threadIdx.x & 1) {
        // 分支 A：一段 loop
        acc = x;
    #pragma unroll 1
        for (int i = 0; i < 8; ++i) {
            acc = acc * 1.1f + 0.3f;
            acc = acc - 2.0f;
        }
    } else {
        // 分支 B：另一段 loop，结构不同
        acc = -x;
    #pragma unroll 1
        for (int i = 0; i < 8; ++i) {
            acc = acc * 0.9f - 0.5f;
            acc = acc + 1.0f;
        }
    }

    // 公共尾部：所有线程都要跑
    float y = acc * 0.5f + 1.0f;
    out[tid] = y;
}

int main()
{
    const int N = 256;
    const int bytes = N * sizeof(float);

    float* h_in  = new float[N];
    float* h_out = new float[N];

    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<float>(i);
    }

    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);

    branch_simt_kernel<<<grid, block>>>(d_out, d_in, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 8; ++i) {
        printf("tid=%d, in=%f, out=%f\n", i, h_in[i], h_out[i]);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}
