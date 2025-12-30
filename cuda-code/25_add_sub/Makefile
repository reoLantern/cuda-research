NVCC   := nvcc
SM     ?= 90
ARCH   := sm_$(SM)

NAME   := addsub
SRC    := $(NAME).cu

BASE   := $(NAME)_$(ARCH)
TARGET := $(BASE).out

all: $(TARGET) $(TARGET).ptx $(TARGET).cubin sass

$(TARGET): $(SRC)
	$(NVCC) -O3 -arch=$(ARCH) -lineinfo $< -o $@

$(TARGET).ptx: $(SRC)
	$(NVCC) -O3 -arch=$(ARCH) -lineinfo -ptx $< -o $@

$(TARGET).cubin: $(SRC)
	$(NVCC) -O3 -arch=$(ARCH) -lineinfo -cubin $< -o $@

sass: $(TARGET).cubin
	cuobjdump --dump-sass $< > $(BASE).sass

clean:
	rm -f $(TARGET) $(TARGET).ptx $(TARGET).cubin $(BASE).sass
