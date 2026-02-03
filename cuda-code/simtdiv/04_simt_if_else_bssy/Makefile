CUDA_HOME ?= /usr/local/cuda-12.6
# CUDA_HOME ?= /usr/local/cuda
NVCC     ?= $(CUDA_HOME)/bin/nvcc
CUOBJDUMP ?= $(CUDA_HOME)/bin/cuobjdump

export PATH := $(CUDA_HOME)/bin:$(PATH)

SM     ?= 50
ARCH   := sm_$(SM)
PTX_ARCH := compute_$(SM)
GENCODE := -gencode arch=$(PTX_ARCH),code=$(ARCH)

NAME   := simt_if_else_bssy
SRC    := $(NAME).cu

BASE   := $(NAME)_$(ARCH)
TARGET := $(BASE).out

all: $(TARGET) $(TARGET).ptx $(TARGET).cubin sass

$(TARGET): $(SRC)
	$(NVCC) -O3 -lineinfo $(GENCODE) $< -o $@

$(TARGET).ptx: $(SRC)
	$(NVCC) -O3 -arch=$(PTX_ARCH) -lineinfo -ptx $< -o $@

$(TARGET).cubin: $(SRC)
	$(NVCC) -O3 -lineinfo $(GENCODE) -cubin $< -o $@

sass: $(TARGET).cubin
	$(CUOBJDUMP) --dump-sass $< > $(BASE).sass

clean:
	rm -f $(TARGET) $(TARGET).ptx $(TARGET).cubin $(BASE).sass
