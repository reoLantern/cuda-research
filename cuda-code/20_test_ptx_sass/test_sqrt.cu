// test_sqrt.cu
#include <cstdio>
#include <cuda_runtime.h>

// 强制用 inline PTX 生成这几条指令，避免被编译器优化掉/替换掉

// sqrt.approx.f32 d, a;
__device__ float sqrt_approx_f32(float a) {
    float d;
    asm volatile ("sqrt.approx.f32 %0, %1;"
                  : "=f"(d) : "f"(a));
    return d;
}

// sqrt.approx.ftz.f32 d, a;
__device__ float sqrt_approx_ftz_f32(float a) {
    float d;
    asm volatile ("sqrt.approx.ftz.f32 %0, %1;"
                  : "=f"(d) : "f"(a));
    return d;
}

// sqrt.rn.f32 d, a;  // IEEE round-to-nearest-even
__device__ float sqrt_rn_f32(float a) {
    float d;
    asm volatile ("sqrt.rn.f32 %0, %1;"
                  : "=f"(d) : "f"(a));
    return d;
}

// sqrt.rn.ftz.f32 d, a;
__device__ float sqrt_rn_ftz_f32(float a) {
    float d;
    asm volatile ("sqrt.rn.ftz.f32 %0, %1;"
                  : "=f"(d) : "f"(a));
    return d;
}

// sqrt.rn.f64 d, a;
__device__ double sqrt_rn_f64(double a) {
    double d;
    asm volatile ("sqrt.rn.f64 %0, %1;"
                  : "=d"(d) : "d"(a));
    return d;
}

__global__ void test_kernel(float *out_f, double *out_d,
                            float in_f, double in_d) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // 依次调用上面的几个函数，保证这些 PTX 指令真的出现在代码里
        // out_f[0] = sqrt_approx_f32(in_f);
        // out_f[1] = sqrt_approx_ftz_f32(in_f);
        out_f[2] = sqrt_rn_f32(in_f);
        // out_f[3] = sqrt_rn_ftz_f32(in_f);
        // out_d[0] = sqrt_rn_f64(in_d);
    }
}

int main() {
    float  h_out_f[4];
    double h_out_d[1];

    float  *d_out_f;
    double *d_out_d;

    cudaMalloc(&d_out_f, sizeof(h_out_f));
    cudaMalloc(&d_out_d, sizeof(h_out_d));

    float  in_f = 2.0f;
    double in_d = 2.0;

    test_kernel<<<1, 1>>>(d_out_f, d_out_d, in_f, in_d);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out_f, d_out_f, sizeof(h_out_f), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_d, d_out_d, sizeof(h_out_d), cudaMemcpyDeviceToHost);

    cudaFree(d_out_f);
    cudaFree(d_out_d);

    printf("sqrt.approx.f32(%.6f)      = %.9f\n", in_f, h_out_f[0]);
    printf("sqrt.approx.ftz.f32(%.6f)  = %.9f\n", in_f, h_out_f[1]);
    printf("sqrt.rn.f32(%.6f)          = %.9f\n", in_f, h_out_f[2]);
    printf("sqrt.rn.ftz.f32(%.6f)      = %.9f\n", in_f, h_out_f[3]);
    printf("sqrt.rn.f64(%.6f)          = %.15f\n", in_d, h_out_d[0]);

    return 0;
}
