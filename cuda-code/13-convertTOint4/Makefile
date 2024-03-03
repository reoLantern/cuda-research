.PHONY:all clean

object=convertTOint4

nvcc=${CUDA_PATH}/bin/nvcc
cuobjdump=${CUDA_PATH}/bin/cuobjdump

include=-I/root/data/mmy/work/ArchResearchDevelop/ref/cutlass/include/ -I/root/data/mmy/work/ArchResearchDevelop/ref/cutlass/tools/util/include/

all:
	$(nvcc) -o $(object) $(object).cu -gencode=arch=compute_86,code=\"sm_86,compute_86\" $(include)

clean:
	rm -rf $(object) $(simplekernel) *.ptx *.sass