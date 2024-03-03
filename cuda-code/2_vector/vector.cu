#include <stdio.h>
#include <chrono>

#define N 10
#define DIMX 10
#define DIMY 10
#define NN 10 * 65536

__global__ void add(int *a, int *b, int *c)
{
    int t = blockIdx.x;
    if (t < N)
        c[t] = a[t] + b[t];
}

__global__ void add_mat(int *a, int *b, int *c)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x * gridDim.y + y;
    if (x < DIMX && y < DIMY)
        c[offset] = a[offset] + b[offset];
}

__global__ void add_thread(int *a, int *b, int *c)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < NN)
        c[t] = a[t] + b[t];
	// __syncthreads();
}

int main()
{
    /*** vector ***/ {
        int a[N], b[N], c[N];
        int *a_cuda, *b_cuda, *c_cuda;
        // 赋值
        for (int i = 0; i < N; i++)
        {
            a[i] = i - 3;
            b[i] = i / 2 + 1;
        }
        cudaMalloc((void **)&a_cuda, N * sizeof(int));
        cudaMalloc((void **)&b_cuda, N * sizeof(int));
        cudaMalloc((void **)&c_cuda, N * sizeof(int));
        cudaMemcpy(a_cuda, a, N * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(b_cuda, b, N * sizeof(int), cudaMemcpyHostToDevice);

        add<<<N, 1>>>(a_cuda, b_cuda, c_cuda);
        // <<<N,1>>>表示add函数要用N个线程块，每个线程块里用一个线程
        // 1个线程块包含多个warp，但必须分配到1个SM上

        cudaMemcpy(c, c_cuda, N * sizeof(int), cudaMemcpyDeviceToHost);
        printf("a+b=(");
        for (int i = 0; i < N; i++)
            printf("%d,", c[i]);
        printf(")\n");
        cudaFree(a_cuda);
        cudaFree(b_cuda);
        cudaFree(c_cuda);
    }

    /*** matrix ***/ {

        int a[DIMX * DIMY], b[DIMX * DIMY], c[DIMX * DIMY];
        int *a_cuda, *b_cuda, *c_cuda;
        // 赋值
        for (int i = 0; i < DIMX; i++)
        {
            for (int j = 0; j < DIMY; j++)
            {
                a[i * DIMX + j] = i + j - 3;
                b[i * DIMX + j] = (i + j) / 2 + 1;
            }
        }
        cudaMalloc((void **)&a_cuda, DIMX * DIMY * sizeof(int));
        cudaMalloc((void **)&b_cuda, DIMX * DIMY * sizeof(int));
        cudaMalloc((void **)&c_cuda, DIMX * DIMY * sizeof(int));
        cudaMemcpy(a_cuda, a, DIMX * DIMY * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(b_cuda, b, DIMX * DIMY * sizeof(int), cudaMemcpyHostToDevice);
        dim3 grid(DIMX, DIMY);
        add_mat<<<grid, 1>>>(a_cuda, b_cuda, c_cuda);
        cudaMemcpy(c, c_cuda, DIMX * DIMY * sizeof(int), cudaMemcpyDeviceToHost);
        printf("a+b=[");
        for (int i = 0; i < DIMX; i++)
        {
            for (int j = 0; j < DIMY; j++)
            {
                printf("%d,", c[i * DIMX + j]);
            }
            printf("\n");
        }
        printf("]\n");
        cudaFree(a_cuda);
        cudaFree(b_cuda);
        cudaFree(c_cuda);
    }

    /*** long vector ***/ {
        int a[NN], b[NN], c[NN];
        int *a_cuda, *b_cuda, *c_cuda;
        // 赋值
        for (int i = 0; i < NN; i++)
        {
            a[i] = i - 3;
            b[i] = i / 2 + 1;
        }
        auto start = std::chrono::high_resolution_clock::now();
        cudaMalloc((void **)&a_cuda, NN * sizeof(int));
        cudaMalloc((void **)&b_cuda, NN * sizeof(int));
        cudaMalloc((void **)&c_cuda, NN * sizeof(int));
        cudaMemcpy(a_cuda, a, NN * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(b_cuda, b, NN * sizeof(int), cudaMemcpyHostToDevice);
        add_thread<<<NN / 128, 128>>>(a_cuda, b_cuda, c_cuda);
        cudaMemcpy(c, c_cuda, NN * sizeof(int), cudaMemcpyDeviceToHost);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        double seconds = duration.count();
        printf("time=%lfs\n", seconds);
        cudaFree(a_cuda);
        cudaFree(b_cuda);
        cudaFree(c_cuda);
    }
}