# file: Makefile

NVCC  := nvcc
ARCH  := sm_89
TARGET := branch_nested.out
SRC    := branch_nested.cu

# 默认目标：可执行文件 + PTX + CUBIN
all: $(TARGET) $(TARGET).ptx $(TARGET).cubin

# 编译可执行文件
$(TARGET): $(SRC)
	$(NVCC) -O3 -arch=$(ARCH) -lineinfo $< -o $@

# 生成 PTX
$(TARGET).ptx: $(SRC)
	$(NVCC) -O3 -arch=$(ARCH) -lineinfo -ptx $< -o $@

# 生成 CUBIN，用于后面 dump SASS
$(TARGET).cubin: $(SRC)
	$(NVCC) -O3 -arch=$(ARCH) -lineinfo -cubin $< -o $@

# 从 CUBIN 中 dump SASS
sass: $(TARGET).cubin
	cuobjdump --dump-sass $(TARGET).cubin > $(TARGET).sass

clean:
	rm -f $(TARGET) $(TARGET).ptx $(TARGET).cubin $(TARGET).sass
