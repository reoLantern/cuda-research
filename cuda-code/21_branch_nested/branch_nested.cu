// file: nested_branch.cu
#include <cstdio>

__global__ void nested_branch_kernel(float* out, const float* in, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (tid >= n) return;

    float x = in[tid];
    float y = 0.0f;

    // ========= 外层 SIMT 分支 =========
    // 用 threadIdx.x 的第 4 位 (0x10) 做条件：
    // warp 内 0~15 走 else，16~31 走 if，制造 warp 内 divergence。
    if (threadIdx.x & 16) {
        // ========= 内层子分支（希望被编译成 predicate 掩码） =========
        // 这里“原封不动”用你之前那段代码：
        if (threadIdx.x & 1) {
            // 分支 A：一小段“加工流水线”
            float t = x * 1.1f;
            t = t + 0.3f;
            t = t * t;
            t = t - 2.0f;
            y = t;
        } else {
            // 分支 B：结构不同的流水线
            float t = x * 0.9f;
            t = t - 0.5f;
            t = t * 0.25f;
            t = t + x;      // 再和原始 x 混合一下
            y = t;
        }
    } else {
        // 外层的另一条路径：做一段稍微重一点的计算（让外层更倾向用真正的分支）
        float t = -x;
        #pragma unroll 1   // 避免完全展开成特别长的一坨
        for (int i = 0; i < 8; ++i) {
            t = t * 0.8f + 0.6f;
            t = t - 1.5f;
        }
        y = t;
    }

    // ========= 公共尾部 =========
    // 让编译器必须在外层 if 之后 reconverge 再做一段共同的逻辑
    float z = y * 0.5f + 1.0f;
    out[tid] = z;
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

    dim3 block(128);  // 保证 warp 内有完整的 32 线程，0~31/32~63 ...
    dim3 grid((N + block.x - 1) / block.x);

    nested_branch_kernel<<<grid, block>>>(d_out, d_in, N);
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
