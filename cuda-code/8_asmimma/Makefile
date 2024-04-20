.PHONY:all debug ptxsass simplekernel asmkernel clean asmkernelhopper simplekernelhopper

object=asmimma
simplekernel=simplekernel
asmkernel=asmFP16ref
asmkernelhopper=asmkernelhopper
simplekernelhopper=simplekernelhopper

nvcc=${CUDA_INSTALL_PATH}/bin/nvcc
cuobjdump=${CUDA_INSTALL_PATH}/bin/cuobjdump

all:
	$(nvcc) -o $(object) $(object).cu -gencode=arch=compute_86,code=\"sm_86,compute_86\"

debug:
	$(nvcc) -g -G -o $(object) $(object).cu -gencode=arch=compute_86,code=\"sm_86,compute_86\"

ptxsass:
	$(nvcc) -o $(object) $(object).cu -gencode=arch=compute_75,code=\"sm_75,compute_75\" && \
	$(cuobjdump) -ptx $(object) > $(object).ptx && $(cuobjdump) -sass $(object) > $(object).sass

ptxsasshopper:
	$(nvcc) -o $(object) $(object).cu -gencode=arch=compute_90,code=\"sm_90,compute_90\" && \
	$(cuobjdump) -ptx $(object) > $(object).ptx && $(cuobjdump) -sass $(object) > $(object).sass

simplekernel:
	$(nvcc) -o $(simplekernel) $(simplekernel).cu -gencode=arch=compute_75,code=\"sm_75,compute_75\" && \
	$(cuobjdump) -ptx $(simplekernel) > $(simplekernel).ptx && $(cuobjdump) -sass $(simplekernel) > $(simplekernel).sass

asmkernel:
	$(nvcc) -o $(asmkernel) $(asmkernel).cu -gencode=arch=compute_86,code=\"sm_86,compute_86\" && \
	$(cuobjdump) -ptx $(asmkernel) > $(asmkernel).ptx && $(cuobjdump) -sass $(asmkernel) > $(asmkernel).sass

asmkernelhopper:
	$(nvcc) -o $(asmkernelhopper) $(asmkernel).cu -gencode=arch=compute_90,code=\"sm_90,compute_90\" && \
	$(cuobjdump) -ptx $(asmkernelhopper) > $(asmkernelhopper).ptx && $(cuobjdump) -sass $(asmkernelhopper) > $(asmkernelhopper).sass

simplekernelhopper:
	$(nvcc) -o $(simplekernelhopper) $(simplekernel).cu -gencode=arch=compute_90,code=\"sm_90,compute_90\" && \
	$(cuobjdump) -ptx $(simplekernelhopper) > $(simplekernelhopper).ptx && $(cuobjdump) -sass $(simplekernelhopper) > $(simplekernelhopper).sass

clean:
	rm -rf $(object) $(simplekernel) *.o *.ii *.fatbin *.fatbin.c *.reg.c \
	*.cudafe1.c *.cudafe1.cpp *.cudafe1.gpu *.cudafe1.stub.c *.module_id *.ptx *.sass