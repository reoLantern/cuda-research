#include <iostream>
#include <iomanip>
#include <mma.h>

__global__ void cpasync_test(uint32_t *gbl, uint32_t *shrd)
{

    uint32_t gbl_addr[6];
    uint32_t shrd_addr[6];

    __shared__ uint32_t shmem[9];

#pragma unroll
    for (int i = 0; i < 6; i++)
    {
        gbl_addr[i] = gbl[i];
        shrd_addr[i] = shrd[i];
        shmem[i] = i;
    }

    asm volatile("{\n\t"
                 "cp.async.ca.shared.global [%0], [%1], 8;\n\t"
                 "cp.async.cg.shared.global [%2], [%3], 16;\n\t"
                 "cp.async.wait_all;\n\t"
                 "cp.async.ca.shared.global [%4], [%5], 4;\n\t"
                 "cp.async.commit_group;\n\t"
                 "cp.async.cg.shared.global [%6], [%7], 16;\n\t"
                 "cp.async.commit_group;\n\t"
                 "cp.async.cg.shared.global [%8], [%9], 16;\n\t"
                 "cp.async.commit_group;\n\t"
                 "cp.async.wait_group 1;\n\t"
                 "}\n\t"
                 :
                 : "r"(shrd_addr[0]), "r"(gbl_addr[0]), "r"(shrd_addr[1]), "r"(gbl_addr[1]), "r"(shrd_addr[2]), "r"(gbl_addr[2]), "r"(shrd_addr[3]), "r"(gbl_addr[3]), "r"(shrd_addr[4]), "r"(gbl_addr[4])
                 :);

    int a = shmem[2];
    gbl[1] = a;
}

int main()
{
    return 0;
}