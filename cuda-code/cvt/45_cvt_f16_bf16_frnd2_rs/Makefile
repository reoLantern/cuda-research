NVCC   := nvcc
SM     ?= 103a
ARCH   := sm_$(SM)
PTX_ARCH := compute_$(SM)
GENCODE := -gencode arch=$(PTX_ARCH),code=$(ARCH)

NAME   := cvt_f16_bf16_frnd2_rs
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
	cuobjdump --dump-sass $< > $(BASE).sass

clean:
	rm -f $(TARGET) $(TARGET).ptx $(TARGET).cubin $(BASE).sass
