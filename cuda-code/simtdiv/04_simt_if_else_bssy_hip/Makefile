ROCM_PATH ?= /opt/rocm
HIPCC    ?= $(ROCM_PATH)/bin/hipcc
OBJDUMP  ?= $(ROCM_PATH)/lib/llvm/bin/llvm-objdump
BUNDLER  ?= $(ROCM_PATH)/lib/llvm/bin/clang-offload-bundler

GFX_RDNA4 ?= gfx1200
GFX_CDNA4 ?= gfx950

NAME := simt_if_else_bssy_hip
SRC  := $(NAME).hip

RDNA4_CO   := $(NAME)_$(GFX_RDNA4).co
RDNA4_OBJ  := $(NAME)_$(GFX_RDNA4).o
RDNA4_SASS := $(NAME)_$(GFX_RDNA4).sass
CDNA4_CO   := $(NAME)_$(GFX_CDNA4).co
CDNA4_OBJ  := $(NAME)_$(GFX_CDNA4).o
CDNA4_SASS := $(NAME)_$(GFX_CDNA4).sass

BUNDLE_PREFIX := hipv4-amdgcn-amd-amdhsa--

all: $(RDNA4_SASS) $(CDNA4_SASS)

$(RDNA4_CO): $(SRC)
	$(HIPCC) -O3 --genco --offload-arch=$(GFX_RDNA4) $< -o $@

$(CDNA4_CO): $(SRC)
	$(HIPCC) -O3 --genco --offload-arch=$(GFX_CDNA4) $< -o $@

$(RDNA4_OBJ): $(RDNA4_CO)
	$(BUNDLER) -unbundle --type=o --input=$< --output=$@ \
		--targets=$(BUNDLE_PREFIX)$(GFX_RDNA4) --allow-missing-bundles

$(CDNA4_OBJ): $(CDNA4_CO)
	$(BUNDLER) -unbundle --type=o --input=$< --output=$@ \
		--targets=$(BUNDLE_PREFIX)$(GFX_CDNA4) --allow-missing-bundles

$(RDNA4_SASS): $(RDNA4_OBJ)
	$(OBJDUMP) --disassemble --mcpu=$(GFX_RDNA4) $< > $@

$(CDNA4_SASS): $(CDNA4_OBJ)
	$(OBJDUMP) --disassemble --mcpu=$(GFX_CDNA4) $< > $@

clean:
	rm -f $(RDNA4_CO) $(RDNA4_OBJ) $(RDNA4_SASS) $(CDNA4_CO) $(CDNA4_OBJ) $(CDNA4_SASS)
