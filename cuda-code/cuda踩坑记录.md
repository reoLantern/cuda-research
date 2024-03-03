# cuda踩坑记录

## 杂项

- nvcc编译时要标明--gpu-architecture或-arch，格式比较自由，可以是--gpu-architecture=compute_86或--gpu-architecture=sm_86或-arch=sm_86，详见 [NVCC官方手册](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-steering-gpu-code-generation) 或 [为各种 NVIDIA 架构匹配 CUDA arch 和 CUDA gencode](https://zhuanlan.zhihu.com/p/631850036)。需要标明的原因是，warp-level功能只有在新架构中才有。

## c_cpp_properties.json

```json
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "/usr/local/cuda-11.3/include"
            ],
            // "defines": ["__CUDA_ARCH__=860"],
            "compilerPath": "/usr/local/cuda-11.3/bin/nvcc",
            "compilerArgs": [
                "--gpu-architecture=compute_86"
            ],
            "intelliSenseMode": "linux-gcc-x64"
        }
    ],
    "version": 4
}
```

其中，compilerArgs为intelliSense提供了架构参数，这将定义宏__CUDA_ARCH__=860，从而使intelliSense能够识别namespace nvcuda；理论上，可以在json文件中定义"defines": ["__CUDA_ARCH__=860"]从而达到相同的效果，也即代码中注释的部分，但很奇怪没有生效。
