#include <mma.h>
#include <iostream>

using namespace nvcuda;

#define WMMA_M 8
#define WMMA_N 32
#define WMMA_K 16

__global__ void wmma_ker(half *a, half *b, half *c)
{
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, a, 16);
    wmma::load_matrix_sync(b_frag, b, 16);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the output
    wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
}

int main()
{

    half *d_a, *h_a, *d_b, *h_b;
    half *d_c, *h_c;
    h_c = new half[16 * 16];
    h_b = new half[16 * 16];
    h_a = new half[16 * 16];
    cudaMalloc(&d_a, 16 * 16 * sizeof(half));
    cudaMalloc(&d_b, 16 * 16 * sizeof(half));
    cudaMalloc(&d_c, 16 * 16 * sizeof(float));
    for (int i = 0; i < 16 * 16; i++)
    {
        h_a[i] = 1.0f;
        h_b[i] = 1.0f;
    }
    cudaMemcpy(d_a, h_a, 16 * 16 * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, 16 * 16 * sizeof(half), cudaMemcpyHostToDevice);
    wmma_ker<<<1, 32>>>(d_a, d_b, d_c);
    cudaMemcpy(h_c, d_c, 16 * 16 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 16 * 16; i++)
        std::cout << (float)h_c[i] << ",";
    std::cout << std::endl;
}
