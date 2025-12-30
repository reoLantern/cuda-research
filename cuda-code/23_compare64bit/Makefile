# Makefile
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUOBJDUMP ?= $(CUDA_HOME)/bin/cuobjdump
NVDISASM  ?= $(CUDA_HOME)/bin/nvdisasm

# 你的 GPU 对应的 SM，例如：
#   Ampere A100: 80
#   Ampere RTX30: 86
#   Ada RTX40: 89
#   Hopper H100: 90
ARCH ?= 80

SM      := sm_$(ARCH)
COMPUTE := compute_$(ARCH)

SRC  := compare64.cu
BASE := compare64

PTX   := $(BASE).$(COMPUTE).ptx
CUBIN := $(BASE).$(SM).cubin
SASS1 := $(BASE).$(SM).cuobjdump.sass
SASS2 := $(BASE).$(SM).nvdisasm.sass

# 你可以改成 -O0 更“直观”（但不一定更接近真实优化后的实现）
NVCCFLAGS  ?= -O3 -lineinfo
PTXASFLAGS ?= -O3

.PHONY: all ptx cubin sass clean sass_fun

all: ptx sass

ptx: $(PTX)

cubin: $(CUBIN)

sass: $(SASS1) $(SASS2)

$(PTX): $(SRC)
	$(NVCC) $(NVCCFLAGS) -ptx -arch=$(COMPUTE) -o $@ $<

$(CUBIN): $(SRC)
	$(NVCC) $(NVCCFLAGS) -Xptxas $(PTXASFLAGS) -cubin -arch=$(SM) -o $@ $<

# cuobjdump：官方文档里 --dump-sass / -sass 就是导出 CUDA assembly（SASS）:contentReference[oaicite:2]{index=2}
$(SASS1): $(CUBIN)
	$(CUOBJDUMP) --dump-sass $< > $@

# nvdisasm：只吃 cubin，但显示选项更丰富:contentReference[oaicite:3]{index=3}
$(SASS2): $(CUBIN)
	$(NVDISASM) -c $< > $@

# 只导出某个 kernel 的 SASS（用法：make sass_fun FUN=s64_lt）
FUN ?=
sass_fun: $(CUBIN)
	$(CUOBJDUMP) --dump-sass --function $(FUN) $(CUBIN) > $(BASE).$(SM).$(FUN).sass

clean:
	rm -f $(PTX) $(CUBIN) $(SASS1) $(SASS2) $(BASE).$(SM).*\.sass
