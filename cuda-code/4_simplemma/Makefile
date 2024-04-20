.PHONY:all

nvcc=${CUDA_INSTALL_PATH}/bin/nvcc
cuobjdump=${CUDA_INSTALL_PATH}/bin/cuobjdump

object=simplemma

all:
	$(nvcc) -o $(object) simplemma.cu -gencode=arch=compute_80,code=\"sm_80,compute_80\" && \
	$(cuobjdump) -ptx $(object) > $(object).ptx && $(cuobjdump) -sass $(object) > $(object).sass
