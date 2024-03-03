#include <stdio.h>
#define N 10
#define BLOCK 16

__device__ int sum(int *cache, int id)
{
    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (id < i)
        {
            cache[id] += cache[id + i];
        }
        __syncthreads();
        i /= 2;
    }
    return cache[0];
}

__global__ void matrix_dot(int *a, int *b, int *c)
{
    int i = blockIdx.x;
    int j = blockIdx.y;          // (i,j)对应C矩阵中的位置
    __shared__ int cache[BLOCK]; // __shared__被线程块中所有线程共享
    int t = threadIdx.x;
    if (t < N)
        cache[t] = a[i * N + t] * b[t * N + j];
    else
        cache[t] = 0;
    __syncthreads();
    sum(cache, t);
    __syncthreads();
    c[i * N + j] = cache[0];
}

__host__ void dot(int *a, int *b, int *c)
{
    int *a_cuda, *b_cuda, *c_cuda;
    cudaMalloc((void **)&a_cuda, N * N * sizeof(int));
    cudaMalloc((void **)&b_cuda, N * N * sizeof(int));
    cudaMalloc((void **)&c_cuda, N * N * sizeof(int));
    cudaMemcpy(a_cuda, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_cuda, b, N * N * sizeof(int), cudaMemcpyHostToDevice);
    dim3 matrix(N, N);
    matrix_dot<<<matrix, BLOCK>>>(a_cuda, b_cuda, c_cuda);
    cudaMemcpy(c, c_cuda, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(a_cuda);
    cudaFree(b_cuda);
    cudaFree(c_cuda);
}

int main()
{
    int a[N * N], b[N * N], c[N * N];
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            a[i * N + j] = i;
            b[i * N + j] = j;
        }
    }
    dot(a, b, c);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d\t", c[i * N + j]);
        }
        printf("\n");
    }
    return 0;
}
