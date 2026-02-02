#### Warp vote：`__all_sync / __any_sync / __ballot_sync`

**接口（CUDA C/C++ 内建函数）**：

```C
int __all_sync   (unsigned mask, int predicate);
int __any_sync   (unsigned mask, int predicate);
unsigned __ballot_sync(unsigned mask, int predicate);
```

**语义**（都在 `mask` 指定的 lane 子集内完成）：

- `__all_sync`：子集内是否 **全部** `predicate != 0`
    
- `__any_sync`：子集内是否 **存在** `predicate != 0`
    
- `__ballot_sync`：返回一个位图：子集内每个 lane 的 predicate 结果（真则对应 bit=1）
    

##### 例子 A：warp 内“流压缩”（stream compaction）

把满足条件的元素紧凑写出，避免每线程一个 atomic：

```C++
__device__ __forceinline__ unsigned lane_id() { return threadIdx.x & 31; }

__global__ void compact(const int* in, int n, int* out, int* outCount) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int v = (idx < n) ? in[idx] : 0;
  int pred = (idx < n) && (v > 0);

  unsigned m = __activemask();
  unsigned votes = __ballot_sync(m, pred);   // 哪些 lane 需要写
  int count = __popc(votes);                 // 子集内写出的数量
  if (count == 0) return;

  int base;
  // 选一个 leader（最低位 1 的 lane）
  int leader = __ffs(votes) - 1;
  if ((int)lane_id() == leader) base = atomicAdd(outCount, count);
  base = __shfl_sync(m, base, leader);       // 广播 base 给所有参与 lane

  // 每个 pred lane 计算自己在 votes 中的排名（rank）
  unsigned laneMask = (1u << lane_id()) - 1;
  int rank = __popc(votes & laneMask);
  if (pred) out[base + rank] = v;
}
```

这个模式里，`__ballot_sync + popc + shfl` 是最经典组合：**vote 负责“筛选/分组”，shuffle 负责“广播/搬运寄存器值”**。

#### Warp match：`__match_any_sync / __match_all_sync`

**接口**：

```C
unsigned __match_any_sync (unsigned mask, T value);
unsigned __match_all_sync (unsigned mask, T value, int *pred);
```

**语义**（直观理解）：

- `__match_any_sync`：把 `mask` 子集按 `value` 相等关系划分组；对每个 lane，返回“与我 value 相等的所有 lane 的位图”。
    
- `__match_all_sync`：额外告诉你“子集内是否全部相等”（通过 `*pred` 返回），并返回匹配掩码。([NVIDIA Docs][1])
    

##### 例子 B：warp 聚合 atomic（按地址分组）

NVIDIA 官方博客用它做 **atomic 聚合**：同一 warp 中访问同一地址的 lane 先用 match 分组，每组只做一次 atomic，再把结果广播回组内，显著减少冲突。

简化版示意：

```C++
unsigned m = __activemask();
unsigned g = __match_any_sync(m, (unsigned long long)ptr); // 同 ptr 的一组
int leader = __ffs(g) - 1;
int old;
if ((threadIdx.x & 31) == leader) old = atomicAdd(ptr, __popc(g));
old = __shfl_sync(m, old, leader);
// 组内每个 lane 根据自己在 g 中的 rank 拿到“原子返回值+偏移”
```

#### Warp reduce：`__reduce_*_sync`

**接口**（一组算子：add/min/max/and/or/xor…）：

```C
int __reduce_add_sync (unsigned mask, int value);
// 以及 u32/u64/s32/s64 等多种重载
```

**语义**：

- 在 `mask` 子集内做归约，并把**归约结果返回给子集内每个参与 lane**（不需要再广播）。
    

##### 例子 C：warp 内求和

```C++
unsigned m = __activemask();
int sum = __reduce_add_sync(m, x);  // 子集内每个 lane 都得到相同 sum
```

如果要做“只让一个 lane 写出”，再配个 leader：

```C++
int leader = __ffs(m) - 1;
if ((threadIdx.x & 31) == leader) out[warpId] = sum;
```

#### Warp shuffle：`__shfl_sync / __shfl_{up,down,xor}_sync`

**接口**：

```C
T __shfl_sync     (unsigned mask, T var, int srcLane, int width=warpSize);
T __shfl_up_sync  (unsigned mask, T var, unsigned delta, int width=warpSize);
T __shfl_down_sync(unsigned mask, T var, unsigned delta, int width=warpSize);
T __shfl_xor_sync (unsigned mask, T var, int laneMask, int width=warpSize);
```

**语义**：

- shuffle 做的是 **“从另一个 lane 读寄存器值”**（或按 up/down/xor 规则搬运），完全不经 shared memory。
    
- `width` 支持把一个 warp 切成若干逻辑小组（如 16/8/4…）在组内 shuffle。
    

##### 例子 D：经典 shuffle 归约

```C++
unsigned m = __activemask();
int v = x;
for (int offset = 16; offset > 0; offset >>= 1) {
  v += __shfl_down_sync(m, v, offset);
}
int sum = v; // 注意：只有 lane0（或每个逻辑小组的 lane0）是最终和
```