```bash
make
make ARCH=80
# 输出：
#   compare64.compute_80.ptx
#   compare64.sm_80.cubin
#   compare64.sm_80.cuobjdump.sass
#   compare64.sm_80.nvdisasm.sass

# 只看某个比较（比如 signed <）
make ARCH=80 sass_fun FUN=s64_lt

```
