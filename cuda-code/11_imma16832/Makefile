.PHONY:all ptxsass clean

object=immakernel

nvcc=${CUDA_PATH}/bin/nvcc
cuobjdump=${CUDA_PATH}/bin/cuobjdump

all:
	$(nvcc) -o $(object) $(object).cu -gencode=arch=compute_86,code=\"sm_86,compute_86\"

ptxsass:
	$(nvcc) -o $(object) $(object).cu -gencode=arch=compute_75,code=\"sm_75,compute_75\" && \
	$(cuobjdump) -ptx $(object) > $(object).ptx && $(cuobjdump) -sass $(object) > $(object).sass

clean:
	rm -rf $(object) $(simplekernel) *.ptx *.sass