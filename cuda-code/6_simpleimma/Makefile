.PHONY:all debug ptxsass clean

object=simpleimma

all:
	nvcc -o $(object) $(object).cu --gpu-architecture=compute_86

debug:
	nvcc -g -G -o $(object) $(object).cu -gencode=arch=compute_86,code=\"sm_86,compute_86\"

ptxsass:
	nvcc -g -G -o $(object) $(object).cu -gencode=arch=compute_86,code=\"sm_86,compute_86\" && \
	cuobjdump -ptx $(object) > $(object).ptx && cuobjdump -sass $(object) > $(object).sass

clean:
	rm -rf $(object) *.o *.ii *.fatbin *.fatbin.c *.reg.c \
	*.cudafe1.c *.cudafe1.cpp *.cudafe1.gpu *.cudafe1.stub.c *.module_id *.ptx *.sass