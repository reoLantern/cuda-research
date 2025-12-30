# 用法示例：
#   make run
#   make ptx
#   make sass
#   make grep
#   make SM=70 sass      # 指定架构
#   make SM=90a sass     # 视你的 CUDA/工具链支持情况

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
NVDISASM  ?= $(CUDA_HOME)/bin/nvdisasm
CUOBJDUMP ?= $(CUDA_HOME)/bin/cuobjdump

SM ?= 80

SRC  := simt_branch.cu
BIN  := simt_branch.out
PTX  := simt_branch.ptx
CUBIN:= simt_branch.cubin
SASS := simt_branch.sass

# 编译选项：尽量减少“过度优化导致分支被谓词化/折叠”
COMMON_NVCCFLAGS := -std=c++17 -lineinfo
# 生成 SASS 需要真机架构 code=sm_XX
GENCODE := -gencode arch=compute_$(SM),code=sm_$(SM)

# O0 更利于观察结构化控制流；你也可以试试 O3 对比
NVCCFLAGS := $(COMMON_NVCCFLAGS) -O0 -Xptxas -O0 -Xptxas --warn-spills $(GENCODE)

.PHONY: all run ptx cubin sass sass-cuobjdump grep clean

all: $(BIN)

$(BIN): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: $(BIN)
	./$(BIN)

ptx: $(SRC)
	$(NVCC) $(COMMON_NVCCFLAGS) -O0 -arch=compute_$(SM) -ptx -o $(PTX) $<
	@echo "Wrote $(PTX)"

cubin: $(SRC)
	$(NVCC) $(COMMON_NVCCFLAGS) -O0 -Xptxas -O0 $(GENCODE) -cubin -o $(CUBIN) $<
	@echo "Wrote $(CUBIN)"

sass: cubin
	$(NVDISASM) -g $(CUBIN) > $(SASS)
	@echo "Wrote $(SASS)"

# 有些环境 nvdisasm 不方便时，用 cuobjdump 也可以
sass-cuobjdump: cubin
	$(CUOBJDUMP) --dump-sass $(CUBIN) > $(SASS)
	@echo "Wrote $(SASS)"

grep: sass
	@echo "==== BSSY / BSYNC occurrences in $(SASS) ===="
	@grep -nE "BSSY|BSYNC" $(SASS) || true

clean:
	rm -f $(BIN) $(PTX) $(CUBIN) $(SASS)
