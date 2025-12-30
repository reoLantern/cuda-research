CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
NVDISASM  ?= $(CUDA_HOME)/bin/nvdisasm

SM ?= 80

SRC := simt_dyn_depth.cu
BIN := simt_dyn_depth.out
CUBIN := $(BIN).cubin
SASS := $(BIN).sass

GENCODE := -gencode arch=compute_$(SM),code=sm_$(SM)

# 经验：前端 O2/O3 都行，ptxas 用 O0 更利于“看见控制流原貌”
NVCCFLAGS := -std=c++17 -lineinfo -O2 -Xptxas -O0 $(GENCODE)

.PHONY: all cubin sass grep run clean

all: $(BIN)

$(BIN): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: $(BIN)
	./$(BIN)

cubin: $(SRC)
	$(NVCC) $(NVCCFLAGS) -cubin -o $(CUBIN) $<

sass: cubin
	$(NVDISASM) -g $(CUBIN) > $(SASS)

grep: sass
	@echo "==== grep BMOV/BSSY/BSYNC/BREAK/BRA.DIV ===="
	@grep -nE "BMOV|BSSY|BSYNC|BREAK|BRA\\.DIV" $(SASS) || true

clean:
	rm -f $(BIN) $(CUBIN) $(SASS)
