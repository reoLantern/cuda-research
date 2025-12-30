# Makefile
TARGET  := warp_ops.out
SRC     := warp_ops.cu

NVCC    ?= nvcc
ARCH    ?= sm_80
COMPUTE := $(subst sm_,compute_,$(ARCH))

# 你可用 make O=0 降低优化，观察更“直”的 PTX/SASS
O       ?= 3

NVCCFLAGS := -std=c++14 -O$(O) -lineinfo -arch=$(ARCH)

.PHONY: all run ptx cubin sass clean

all: $(TARGET) ptx cubin sass

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET)

# 生成 PTX（用 compute_XX 更直观）
ptx: $(TARGET).ptx
$(TARGET).ptx: $(SRC)
	$(NVCC) -std=c++14 -O$(O) -arch=$(COMPUTE) -ptx -o $@ $<

# 生成 cubin
cubin: $(TARGET).cubin
$(TARGET).cubin: $(SRC)
	$(NVCC) -std=c++14 -O$(O) -arch=$(ARCH) -cubin -o $@ $<

# 从 cubin 导出 SASS（如果你更喜欢 nvdisasm，也可以手动替换这一行）
sass: $(TARGET).sass
$(TARGET).sass: $(TARGET).cubin
	cuobjdump --dump-sass $< > $@

clean:
	rm -f $(TARGET) $(TARGET).ptx $(TARGET).cubin $(TARGET).sass
