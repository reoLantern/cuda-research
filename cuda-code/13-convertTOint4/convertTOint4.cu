#include <iostream>
#include <iomanip>

#define NUM_ILP 5
#define NUM_ELEM 8

__global__ void pack32to4cvt(int *source, uint32_t *out)
{
    printf("\n------pack32to4cvt------\n");

    uint32_t start;
    uint32_t stop[NUM_ILP];

    int source_reg[NUM_ELEM * NUM_ILP];
    for (int i = 0; i < NUM_ELEM * NUM_ILP; i++)
        source_reg[i] = source[i];
    printf("--loading--");
    for (int i = 0; i < NUM_ELEM * NUM_ILP; i++)
        printf(" %#x", source_reg[i]);
    printf("\n");
    uint32_t out_reg[NUM_ILP];

    asm volatile("mov.u32 %0, %%clock;"
                 : "=r"(start));
#pragma unroll
    for (int i = 0; i < NUM_ILP; i++)
    {
        asm volatile(
            "{ .reg .u32 r4;"
            "cvt.pack.sat.s4.s32.b32   r4, %8, %7, 0;"
            "cvt.pack.sat.s4.s32.b32   r4, %6, %5, r4;"
            "cvt.pack.sat.s4.s32.b32   r4, %4, %3, r4;"
            "cvt.pack.sat.s4.s32.b32   %0, %2, %1, r4;"
            "}"
            : "=r"(out_reg[i])
            : "r"(source_reg[7 + i * NUM_ELEM]), "r"(source_reg[6 + i * NUM_ELEM]), "r"(source_reg[5 + i * NUM_ELEM]), "r"(source_reg[4 + i * NUM_ELEM]),
              "r"(source_reg[3 + i * NUM_ELEM]), "r"(source_reg[2 + i * NUM_ELEM]), "r"(source_reg[1 + i * NUM_ELEM]), "r"(source_reg[0 + i * NUM_ELEM]));
        asm volatile("mov.u32 %0, %%clock;"
                     : "=r"(stop[i]));
    }

    for (size_t i = 0; i < NUM_ILP; i++)
        out[i] = out_reg[i];

    printf("Thread%u pack takes ", threadIdx.x);
    for (int i = 0; i < NUM_ILP; i++)
        printf("%u ", stop[i] - start);
    printf("cycles\n");
}

__global__ void pack32to4logic(int *source, uint32_t *out)
{
    printf("\n------pack32to4logic------\n");

    uint32_t start;
    uint32_t stop[NUM_ILP];

    int source_reg[NUM_ELEM * NUM_ILP];
    for (int i = 0; i < NUM_ELEM * NUM_ILP; i++)
        source_reg[i] = source[i];
    printf("--loading--");
    for (int i = 0; i < NUM_ELEM * NUM_ILP; i++)
        printf(" %#x", source_reg[i]);
    printf("\n");
    uint32_t out_reg[NUM_ILP];

    asm volatile("mov.u32 %0, %%clock;"
                 : "=r"(start));
#pragma unroll
    for (int i = 0; i < NUM_ILP; i++)
    {
        asm volatile(
            "{ .reg .u32 r4;"
            ".reg .u32 r5;"
            "and.b32 r4, %1, 0x0F;"
            "shl.b32 r4, r4, 4;"
            "and.b32 r5, %2, 0x0F;"
            "or.b32 r4, r4, r5;"
            "shl.b32 r4, r4, 4;"
            "and.b32 r5, %3, 0x0F;"
            "or.b32 r4, r4, r5;"
            "shl.b32 r4, r4, 4;"
            "and.b32 r5, %4, 0x0F;"
            "or.b32 r4, r4, r5;"
            "shl.b32 r4, r4, 4;"
            "and.b32 r5, %5, 0x0F;"
            "or.b32 r4, r4, r5;"
            "shl.b32 r4, r4, 4;"
            "and.b32 r5, %6, 0x0F;"
            "or.b32 r4, r4, r5;"
            "shl.b32 r4, r4, 4;"
            "and.b32 r5, %7, 0x0F;"
            "or.b32 r4, r4, r5;"
            "shl.b32 r4, r4, 4;"
            "and.b32 r5, %8, 0x0F;"
            "or.b32 %0, r4, r5;"
            "}"
            : "=r"(out_reg[i])
            : "r"(source_reg[0 + i * NUM_ELEM]), "r"(source_reg[1 + i * NUM_ELEM]), "r"(source_reg[2 + i * NUM_ELEM]), "r"(source_reg[3 + i * NUM_ELEM]),
              "r"(source_reg[4 + i * NUM_ELEM]), "r"(source_reg[5 + i * NUM_ELEM]), "r"(source_reg[6 + i * NUM_ELEM]), "r"(source_reg[7 + i * NUM_ELEM])
            : "memory");
        asm volatile("mov.u32 %0, %%clock;"
                     : "=r"(stop[i]));
    }

    for (size_t i = 0; i < NUM_ILP; i++)
        out[i] = out_reg[i];

    printf("Thread%u pack takes ", threadIdx.x);
    for (int i = 0; i < NUM_ILP; i++)
        printf("%u ", stop[i] - start);
    printf("cycles\n");
}

__global__ void pack32to4simple(int *source, uint32_t *out)
{
    printf("\n------pack32to4simple------\n");

    uint32_t start;
    uint32_t stop[NUM_ILP];

    int source_reg[NUM_ELEM * NUM_ILP];
    for (int i = 0; i < NUM_ELEM * NUM_ILP; i++)
        source_reg[i] = source[i];
    printf("--loading--");
    for (int i = 0; i < NUM_ELEM * NUM_ILP; i++)
        printf(" %#x", source_reg[i]);
    printf("\n");
    uint32_t out_reg[NUM_ILP];

    asm volatile("mov.u32 %0, %%clock;"
                 : "=r"(start));
#pragma unroll
    for (int i = 0; i < NUM_ILP; i++)
    {
        uint32_t reg = 0;
#pragma unroll
        for (int idx = 0; idx < NUM_ELEM; idx++)
            reg = (reg << 4) | (source_reg[idx + i * NUM_ELEM] & 0x0F);
        out_reg[i] = reg;
        asm volatile("mov.u32 %0, %%clock;"
                     : "=r"(stop[i]));
    }

    for (size_t i = 0; i < NUM_ILP; i++)
        out[i] = out_reg[i];

    printf("Thread%u pack takes ", threadIdx.x);
    for (int i = 0; i < NUM_ILP; i++)
        printf("%u ", stop[i] - start);
    printf("cycles\n");
}

__global__ void pack8to4cvt(int8_t *source, uint32_t *out)
{
    printf("\n------pack8to4cvt------\n");

    uint32_t start;
    uint32_t stop[NUM_ILP];

    int source_reg[NUM_ELEM * NUM_ILP];

    for (int i = 0; i < NUM_ELEM * NUM_ILP; i++)
        source_reg[i] = (int)source[i];
    printf("--loading--");
    for (int i = 0; i < NUM_ELEM * NUM_ILP; i++)
        printf(" %#x", source_reg[i]);
    printf("\n");
    uint32_t out_reg[NUM_ILP];

    asm volatile("mov.u32 %0, %%clock;"
                 : "=r"(start));
#pragma unroll
    for (int i = 0; i < NUM_ILP; i++)
    {
        asm volatile(
            "{ .reg .u32 r4;"
            "cvt.pack.sat.s4.s32.b32   r4, %8, %7, 0;"
            "cvt.pack.sat.s4.s32.b32   r4, %6, %5, r4;"
            "cvt.pack.sat.s4.s32.b32   r4, %4, %3, r4;"
            "cvt.pack.sat.s4.s32.b32   %0, %2, %1, r4;"
            "}"
            : "=r"(out_reg[i])
            : "r"(source_reg[7 + i * NUM_ELEM]), "r"(source_reg[6 + i * NUM_ELEM]), "r"(source_reg[5 + i * NUM_ELEM]), "r"(source_reg[4 + i * NUM_ELEM]),
              "r"(source_reg[3 + i * NUM_ELEM]), "r"(source_reg[2 + i * NUM_ELEM]), "r"(source_reg[1 + i * NUM_ELEM]), "r"(source_reg[0 + i * NUM_ELEM]));
        asm volatile("mov.u32 %0, %%clock;"
                     : "=r"(stop[i]));
    }

    for (size_t i = 0; i < NUM_ILP; i++)
        out[i] = out_reg[i];

    printf("Thread%u pack takes ", threadIdx.x);
    for (int i = 0; i < NUM_ILP; i++)
        printf("%u ", stop[i] - start);
    printf("cycles\n");
}

__global__ void pack8to4logic(int8_t *source, uint32_t *out)
{
    printf("\n------pack8to4logic------\n");

    uint32_t start;
    uint32_t stop[NUM_ILP];

    int16_t source_reg[NUM_ELEM * NUM_ILP];

    for (int i = 0; i < NUM_ELEM * NUM_ILP; i++)
        source_reg[i] = (int16_t)source[i];
    printf("--loading--");
    for (int i = 0; i < NUM_ELEM * NUM_ILP; i++)
        printf(" %#x", source_reg[i]);
    printf("\n");
    uint32_t out_reg[NUM_ILP];

    asm volatile("mov.u32 %0, %%clock;"
                 : "=r"(start));
#pragma unroll
    for (int i = 0; i < NUM_ILP; i++)
    {
        asm volatile(
            "{ .reg .u16 a;"
            ".reg .u16 a0;"
            ".reg .u16 b;"
            ".reg .u16 b0;"
            "and.b16 a, %1, 0x0F;"
            "shl.b16 a, a, 4;"
            "and.b16 a0, %2, 0x0F;"
            "or.b16 a, a, a0;"
            "shl.b16 a, a, 4;"
            "and.b16 a0, %3, 0x0F;"
            "or.b16 a, a, a0;"
            "shl.b16 a, a, 4;"
            "and.b16 a0, %4, 0x0F;"
            "or.b16 a, a, a0;"

            "and.b16 b, %5, 0x0F;"
            "shl.b16 b, b, 4;"
            "and.b16 b0, %6, 0x0F;"
            "or.b16 b, b, b0;"
            "shl.b16 b, b, 4;"
            "and.b16 b0, %7, 0x0F;"
            "or.b16 b, b, b0;"
            "shl.b16 b, b, 4;"
            "and.b16 b0, %8, 0x0F;"
            "or.b16 b, b, b0;"

            "mov.b32 %0, {b, a};"
            "}"
            : "=r"(out_reg[i])
            : "h"(source_reg[0 + i * NUM_ELEM]), "h"(source_reg[1 + i * NUM_ELEM]), "h"(source_reg[2 + i * NUM_ELEM]), "h"(source_reg[3 + i * NUM_ELEM]),
              "h"(source_reg[4 + i * NUM_ELEM]), "h"(source_reg[5 + i * NUM_ELEM]), "h"(source_reg[6 + i * NUM_ELEM]), "h"(source_reg[7 + i * NUM_ELEM])
            : "memory");
        asm volatile("mov.u32 %0, %%clock;"
                     : "=r"(stop[i]));
    }

    for (size_t i = 0; i < NUM_ILP; i++)
        out[i] = out_reg[i];

    printf("Thread%u pack takes ", threadIdx.x);
    for (int i = 0; i < NUM_ILP; i++)
        printf("%u ", stop[i] - start);
    printf("cycles\n");
}

__global__ void pack8to4simple(int8_t *source, uint32_t *out)
{
    printf("\n------pack8to4simple------\n");

    uint32_t start;
    uint32_t stop[NUM_ILP];

    int8_t source_reg[NUM_ELEM * NUM_ILP];
    for (int i = 0; i < NUM_ELEM * NUM_ILP; i++)
        source_reg[i] = source[i];
    printf("--loading--");
    for (int i = 0; i < NUM_ELEM * NUM_ILP; i++)
        printf(" %#x", source_reg[i]);
    printf("\n");
    uint32_t out_reg[NUM_ILP];

    asm volatile("mov.u32 %0, %%clock;"
                 : "=r"(start));
#pragma unroll
    for (int i = 0; i < NUM_ILP; i++)
    {
        uint32_t reg = 0;
#pragma unroll
        for (int idx = 0; idx < NUM_ELEM; idx++)
            reg = (reg << 4) | (source_reg[idx + i * NUM_ELEM] & 0x0F);
        out_reg[i] = reg;
        asm volatile("mov.u32 %0, %%clock;"
                     : "=r"(stop[i]));
    }

    for (size_t i = 0; i < NUM_ILP; i++)
        out[i] = out_reg[i];

    printf("Thread%u pack takes ", threadIdx.x);
    for (int i = 0; i < NUM_ILP; i++)
        printf("%u ", stop[i] - start);
    printf("cycles\n");
}

int main()
{
    uint32_t inputData[NUM_ELEM] = {0xfffffffd, 0x00000002, 0x00000001, 0x00000000,
                                    0x00000005, 0x00000003, 0x00000004, 0xfffffffc};
    int inputArray[NUM_ELEM * NUM_ILP];
    for (int i = 0; i < NUM_ELEM * NUM_ILP; i++)
        inputArray[i] = static_cast<int>(inputData[i % NUM_ELEM]);
    printf("\ninputArray=");
    for (int i = 0; i < NUM_ELEM * NUM_ILP; i++)
        printf(" %#x", inputArray[i]);
    printf("\n");
    int *d_inputArray;
    uint32_t *d_outputRegister;
    uint32_t h_outputRegister[NUM_ILP];

    cudaMalloc((void **)&d_inputArray, NUM_ELEM * NUM_ILP * sizeof(uint32_t));
    cudaMalloc((void **)&d_outputRegister, NUM_ILP * sizeof(uint32_t));
    cudaMemcpy(d_inputArray, inputArray, NUM_ELEM * NUM_ILP * sizeof(uint32_t), cudaMemcpyHostToDevice);

    //
    // pack32to4cvt
    //
    cudaMemset(d_outputRegister, 0, NUM_ILP * sizeof(uint32_t));
    pack32to4cvt<<<1, 1>>>(d_inputArray, d_outputRegister);
    cudaMemcpy(&h_outputRegister, d_outputRegister, NUM_ILP * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::cout << "Packed value: 0x" << std::hex << h_outputRegister[0] << std::endl;

    //
    // pack32to4logic
    //
    cudaMemset(d_outputRegister, 0, sizeof(uint32_t));
    pack32to4logic<<<1, 1>>>(d_inputArray, d_outputRegister);
    cudaMemcpy(&h_outputRegister, d_outputRegister, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::cout << "Packed value: 0x" << std::hex << h_outputRegister[0] << std::endl;

    //
    // pack32to4simple
    //
    cudaMemset(d_outputRegister, 0, NUM_ILP * sizeof(uint32_t));
    pack32to4simple<<<1, 1>>>(d_inputArray, d_outputRegister);
    cudaMemcpy(&h_outputRegister, d_outputRegister, NUM_ILP * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::cout << "Packed value: 0x" << std::hex << h_outputRegister[0] << std::endl;

    //
    // pack8to4cvt
    //
    int8_t inputArray8[NUM_ELEM * NUM_ILP];
    for (int i = 0; i < NUM_ELEM * NUM_ILP; i++)
        inputArray8[i] = static_cast<int8_t>(inputData[i % NUM_ELEM]);
    printf("\ninputArray8");
    for (int i = 0; i < NUM_ELEM * NUM_ILP; i++)
        printf(" %#x", inputArray[i]);
    printf("\n");
    int8_t *d_inputArray8;
    cudaMalloc((void **)&d_inputArray8, NUM_ELEM * NUM_ILP * sizeof(int8_t));
    cudaMemcpy(d_inputArray8, inputArray8, NUM_ELEM * NUM_ILP * sizeof(int8_t), cudaMemcpyHostToDevice);

    cudaMemset(d_outputRegister, 0, NUM_ILP * sizeof(uint32_t));
    pack8to4cvt<<<1, 1>>>(d_inputArray8, d_outputRegister);
    cudaMemcpy(&h_outputRegister, d_outputRegister, NUM_ILP * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::cout << "Packed value: 0x" << std::hex << h_outputRegister[0] << std::endl;

    //
    // pack8to4logic
    //
    cudaMemset(d_outputRegister, 0, NUM_ILP * sizeof(uint32_t));
    pack8to4logic<<<1, 1>>>(d_inputArray8, d_outputRegister);
    cudaMemcpy(&h_outputRegister, d_outputRegister, NUM_ILP * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::cout << "Packed value: 0x" << std::hex << h_outputRegister[0] << std::endl;

    //
    // pack8to4simple
    //
    cudaMemset(d_outputRegister, 0, NUM_ILP * sizeof(uint32_t));
    pack8to4simple<<<1, 1>>>(d_inputArray8, d_outputRegister);
    cudaMemcpy(&h_outputRegister, d_outputRegister, NUM_ILP * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::cout << "Packed value: 0x" << std::hex << h_outputRegister[0] << std::endl;

    cudaFree(d_inputArray);
    cudaFree(d_inputArray8);
    cudaFree(d_outputRegister);

    printf("\n");

    return 0;
}
