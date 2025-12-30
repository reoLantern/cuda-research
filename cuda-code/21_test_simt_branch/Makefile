NVCC  := nvcc
ARCH  := sm_89          # 按你的 GPU 改，比如 sm_75 / sm_86 / sm_90
TARGET := test_simt_branch.out
SRC    := test_simt_branch.cu

all: $(TARGET) $(TARGET).ptx $(TARGET).cubin

$(TARGET): $(SRC)
	$(NVCC) -O3 -arch=$(ARCH) -lineinfo $< -o $@

$(TARGET).ptx: $(SRC)
	$(NVCC) -O3 -arch=$(ARCH) -lineinfo -ptx $< -o $@

$(TARGET).cubin: $(SRC)
	$(NVCC) -O3 -arch=$(ARCH) -lineinfo -cubin $< -o $@

sass: $(TARGET).cubin
	cuobjdump --dump-sass $(TARGET).cubin > $(TARGET).sass

clean:
	rm -f $(TARGET) $(TARGET).ptx $(TARGET).cubin $(TARGET).sass
