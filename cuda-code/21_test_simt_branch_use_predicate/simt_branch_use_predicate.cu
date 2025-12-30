// file: branch_simple.cu
#include <cstdio>

__global__ void branch_simple_kernel(float* out, const float* in, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (tid >= n) return;

    float x = in[tid];
    float y;

    if (threadIdx.x & 1) {
        float t = x * 1.1f;
        t = t + 0.3f;
        t = t * t;
        t = t - 2.0f;
        y = t;
    } else {
        float t = x * 0.9f;
        t = t - 0.5f;
        t = t * 0.25f;
        t = t + x;
        y = t;
    }

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

    branch_simple_kernel<<<grid, block>>>(d_out, d_in, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // 简单检查一下结果
    for (int i = 0; i < 8; ++i) {
        printf("tid=%d, in=%f, out=%f\n", i, h_in[i], h_out[i]);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}
