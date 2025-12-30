# 编译器
NVCC := nvcc

# 目标架构 (请根据你的显卡修改，例如 sm_75, sm_80, sm_86, sm_90)
# 这里默认使用 sm_80 (Ampere A100) 作为示例，因为它很通用
ARCH := -arch=sm_89

# 输出文件名
TARGET := carry_test.out

all: $(TARGET) ptx sass

# 1. 编译生成可执行文件
$(TARGET): carry_test.cu
	$(NVCC) $(ARCH) -O3 -o $@ $<

# 2. 生成 PTX (虚拟汇编) 用于查看逻辑指令
ptx: carry_test.cu
	$(NVCC) $(ARCH) -O3 -ptx -o $(TARGET).ptx $<

# 3. 生成 SASS (硬件汇编) 用于验证是否使用了进位指令
# 需要使用 cuobjdump 工具 (包含在 CUDA Toolkit 中)
sass: $(TARGET)
	cuobjdump -sass $(TARGET) > $(TARGET).sass

clean:
	rm -f $(TARGET) $(TARGET).ptx $(TARGET).sass $(TARGET).lib $(TARGET).exp

.PHONY: all ptx sass clean