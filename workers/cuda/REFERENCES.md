# NVIDIA implementation references

Only NVIDIA primary sources constrain the CUDA implementation. These references describe API semantics; they do not establish project hardware validation.

* [CUDA 13.2 Windows compiler support](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-installation-guide-microsoft-windows/index.html): MSVC 195x, native x64 and C++17/20. The project uses MSVC C++20 for the control code and CUDA C++17 for the kernels. No unsupported compiler override is used.
* [cuBLAS 13.2 reference](https://docs.nvidia.com/cuda/archive/13.2.0/cublas/index.html): cuBLASLt descriptor, heuristic, algorithm check/configuration, scale pointer and matmul contracts. The actual redistributable component version is separately recorded in the dependency lock and result metadata.
* [cuBLAS 12.9.2 numeric and layout baseline](https://docs.nvidia.com/cuda/archive/12.9.2/cublas/index.html): FP32 pedantic mode, disabling single precision emulation, E4M3 tensor scale semantics, disabled FP8 fast accumulation, TN/alignment constraints on Ada and Blackwell GeForce, FP32 C/D output.
* [NVIDIA LtFp8Matmul sample](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt/LtFp8Matmul): API usage reference. No sample implementation is copied or linked as a runtime dependency.
* [CUDA Math API half2 arithmetic](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF2__ARITHMETIC.html): `__hfma2` performs two half-precision fused multiply-adds. Conversion is outside the hot loop.

The project does not infer Tensor Core utilization from throughput ratios. cuBLASLt type/compute/algorithm configuration is recorded as documented API evidence. Instruction disassembly and profiler evidence are separately recorded when available.
