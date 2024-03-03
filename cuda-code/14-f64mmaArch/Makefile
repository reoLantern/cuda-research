.PHONY:all clean

object=f64mmaArch

nvcc=${CUDA_PATH}/bin/nvcc
cuobjdump=${CUDA_PATH}/bin/cuobjdump

all:
	$(nvcc) -o $(object).out $(object).cu -gencode=arch=compute_86,code=\"sm_86,compute_86\"

clean:
	rm -rf *.ptx *.sass *.out