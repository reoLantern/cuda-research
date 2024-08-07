.PHONY:asmkernel clean

asmkernel=asmFP16ref


nvcc=${CUDA_INSTALL_PATH}/bin/nvcc
cuobjdump=${CUDA_INSTALL_PATH}/bin/cuobjdump

asmkernel_ampere:
	$(nvcc) -o $(asmkernel) $(asmkernel).cu -gencode=arch=compute_86,code=\"sm_86,compute_86\" && \
	$(cuobjdump) -ptx $(asmkernel) > $(asmkernel).ptx && $(cuobjdump) -sass $(asmkernel) > $(asmkernel).sass

asmkernel_turing:
	# $(nvcc) -o $(asmkernel) $(asmkernel).cu -gencode=arch=compute_75,code=\"sm_75,compute_75\" && \
	$(cuobjdump) -ptx $(asmkernel) > $(asmkernel).ptx && $(cuobjdump) -sass $(asmkernel) > $(asmkernel).sass

clean:
	rm -rf $(object) $(simplekernel) *.o *.ii *.fatbin *.fatbin.c *.reg.c \
	*.cudafe1.c *.cudafe1.cpp *.cudafe1.gpu *.cudafe1.stub.c *.module_id *.ptx *.sass