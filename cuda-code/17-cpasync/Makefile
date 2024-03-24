.PHONY:all clean 2d

object=cpasynckernel1d
object2d=cpasynckernel2d

CUDA_PATH = ${CUDA_INSTALL_PATH}

nvcc=${CUDA_PATH}/bin/nvcc
cuobjdump=${CUDA_PATH}/bin/cuobjdump

all:
	$(nvcc) -o $(object).out $(object).cu -gencode=arch=compute_90a,code=\"sm_90a,compute_90a\" && \
	$(cuobjdump) -ptx $(object).out > $(object).ptx && $(cuobjdump) -sass $(object).out > $(object).sass

2d:
	$(nvcc) -o $(object2d).out $(object2d).cu -gencode=arch=compute_90a,code=\"sm_90a,compute_90a\" && \
	$(cuobjdump) -ptx $(object2d).out > $(object2d).ptx && $(cuobjdump) -sass $(object2d).out > $(object2d).sass

clean:
	rm -rf *.ptx *.sass *.out