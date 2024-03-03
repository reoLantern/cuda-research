.PHONY:all clean

object=cpasynckernel

CUDA_PATH = ${CUDA_INSTALL_PATH}

nvcc=${CUDA_PATH}/bin/nvcc
cuobjdump=${CUDA_PATH}/bin/cuobjdump

all:
	$(nvcc) -o $(object).out $(object).cu -gencode=arch=compute_90,code=\"sm_90,compute_90\" && \
	$(cuobjdump) -ptx $(object).out > $(object).ptx && $(cuobjdump) -sass $(object).out > $(object).sass

clean:
	rm -rf *.ptx *.sass *.out