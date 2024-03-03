.PHONY:all clean

object=int4bvector

nvcc=${CUDA_PATH}/bin/nvcc

all:
	$(nvcc) -o $(object) $(object).cu -gencode=arch=compute_86,code=\"sm_86,compute_86\"

clean:
	rm -rf $(object) *.ptx *.sass