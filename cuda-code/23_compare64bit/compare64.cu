// compare64.cu
#include <stdint.h>

#define IDX() int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x)

// 64-bit signed compare kernels
#define DEF_S64(name, OP) \
extern "C" __global__ void name(const int64_t* __restrict__ a, \
                                const int64_t* __restrict__ b, \
                                uint8_t* __restrict__ out) { \
  IDX(); \
  int64_t x = a[idx]; \
  int64_t y = b[idx]; \
  out[idx] = (uint8_t)((x OP y) ? 1 : 0); \
}

// 64-bit unsigned compare kernels
#define DEF_U64(name, OP) \
extern "C" __global__ void name(const uint64_t* __restrict__ a, \
                                const uint64_t* __restrict__ b, \
                                uint8_t* __restrict__ out) { \
  IDX(); \
  uint64_t x = a[idx]; \
  uint64_t y = b[idx]; \
  out[idx] = (uint8_t)((x OP y) ? 1 : 0); \
}

// （可选）32-bit baseline，对照用
#define DEF_S32(name, OP) \
extern "C" __global__ void name(const int32_t* __restrict__ a, \
                                const int32_t* __restrict__ b, \
                                uint8_t* __restrict__ out) { \
  IDX(); \
  int32_t x = a[idx]; \
  int32_t y = b[idx]; \
  out[idx] = (uint8_t)((x OP y) ? 1 : 0); \
}

#define DEF_U32(name, OP) \
extern "C" __global__ void name(const uint32_t* __restrict__ a, \
                                const uint32_t* __restrict__ b, \
                                uint8_t* __restrict__ out) { \
  IDX(); \
  uint32_t x = a[idx]; \
  uint32_t y = b[idx]; \
  out[idx] = (uint8_t)((x OP y) ? 1 : 0); \
}

// ---- 64-bit signed ----
DEF_S64(s64_lt, <)
DEF_S64(s64_le, <=)
DEF_S64(s64_gt, >)
DEF_S64(s64_ge, >=)
DEF_S64(s64_eq, ==)
DEF_S64(s64_ne, !=)

// ---- 64-bit unsigned ----
DEF_U64(u64_lt, <)
DEF_U64(u64_le, <=)
DEF_U64(u64_gt, >)
DEF_U64(u64_ge, >=)
DEF_U64(u64_eq, ==)
DEF_U64(u64_ne, !=)

// ---- optional 32-bit baseline ----
DEF_S32(s32_lt, <)
DEF_U32(u32_lt, <)
