.PHONY:all clean

object=wgmmaasynckernel

CUDA_PATH = ${CUDA_INSTALL_PATH}

nvcc=${CUDA_PATH}/bin/nvcc
cuobjdump=${CUDA_PATH}/bin/cuobjdump

all:
	$(nvcc) -o $(object).out $(object).cu -gencode=arch=compute_90a,code=\"sm_90a,compute_90a\" && \
	$(cuobjdump) -ptx $(object).out > $(object).ptx && $(cuobjdump) -sass $(object).out > $(object).sass

clean:
	rm -rf *.ptx *.sass *.out