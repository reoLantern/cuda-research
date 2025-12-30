NVCC      ?= nvcc
CUOBJDUMP ?= cuobjdump
NVDISASM  ?= nvdisasm

ARCH      ?= sm_80
TARGET    := test_sqrt.out
SRC       := test_sqrt.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -arch=$(ARCH) -Xptxas -v -o $@ $<

ptx: $(SRC)
	$(NVCC) -arch=$(ARCH) -ptx -o $(TARGET).ptx $<
	@echo "Generated PTX: $(TARGET).ptx"

sass: $(TARGET)
	$(CUOBJDUMP) --dump-sass $(TARGET) > $(TARGET).sass
	@echo "Generated SASS: $(TARGET).sass"

sass-nvdisasm: $(TARGET)
	@echo "====== nvdisasm output ======"
	$(NVDISASM) $(TARGET) > $(TARGET).nvdisasm
	@echo "Generated disasm: $(TARGET).nvdisasm"

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(TARGET).ptx $(TARGET).sass $(TARGET).nvdisasm
