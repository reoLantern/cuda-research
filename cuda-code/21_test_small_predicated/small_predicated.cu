// file: masked_branch.cu
#include <cstdio>

__global__ void masked_branch_kernel(float* out, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float x = static_cast<float>(tid);

    float y = x * 2.0f;

    if (threadIdx.x & 1) {
        y = y + 1.0f;
        y = y * 0.5f;
    }
    else {
        y = y - 1.0f;
        y = y * 1.5f;
    }

    out[tid] = y;
}

int main()
{
    const int N = 256;
    const int bytes = N * sizeof(float);

    float* h_out = new float[N];
    float* d_out = nullptr;

    cudaMalloc(&d_out, bytes);

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);

    masked_branch_kernel<<<grid, block>>>(d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 8; ++i) {
        printf("out[%d] = %f\n", i, h_out[i]);
    }

    cudaFree(d_out);
    delete[] h_out;

    return 0;
}
