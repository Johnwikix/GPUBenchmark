# CUDA worker

Native Windows x64 CUDA worker implementing `alu.fp32.fma`, `alu.fp16.fma`, `gemm.fp32.strict`, `gemm.fp16.acc32` and `gemm.fp8.e4m3.acc32`. No FP4 test is included in the first release.

## Build

Use the repository build script from a normal PowerShell prompt. CUDA 13.2.x is required; the currently selected reproducible component versions are in `dependencies.lock.json`. CUDA 11.8 cannot build Blackwell SM120, and compiler override flags are prohibited.

```powershell
.\scripts\build-windows.ps1 -Backends cuda -Configuration Release -CudaRoot 'G:\Tool\Cuda\13.2.2\toolkit'
```

The path is an example for the current validation machine. Supply the actual extracted or installed toolkit path on other machines. The worker contains native SASS for SM75, SM86, SM89 and SM120, plus compute120 PTX for a future legal driver probe. A future GPU is not automatically certified by PTX compatibility.

The CUDA runtime is statically linked. `cublasLt64_13.dll` must be deployed beside `cuda_worker.exe` for GEMM. The DLL is loaded explicitly from that directory with restricted dependency search, so missing BLAS does not disable CUDA enumeration or ALU. Package only the redistributable files actually needed, with hashes and license metadata. Driver DLLs remain provided by the installed NVIDIA driver.

## Validate on hardware

```powershell
.\out\Release\gpubench.exe doctor --backend cuda --json .\results\cuda-doctor.json
.\out\Release\gpubench.exe run --backend cuda --device nvidia:0 --suite alu --dtype fp32,fp16 --json .\results\cuda-alu.json --csv .\results\cuda-alu.csv
.\out\Release\gpubench.exe run --backend cuda --device nvidia:0 --suite gemm --dtype fp32,fp16,fp8 --sizes 1024,2048,4096 --json .\results\cuda-gemm.json --csv .\results\cuda-gemm.csv
.\scripts\validate-device.ps1 -Device nvidia:0 -Profile standard -OutputDir .\results
```

The shared validation script is the reproducible owner-operated entry point. Retain its commands, exit codes, stderr/stdout logs and raw JSON/CSV. Check the support matrix and evidence files for the distinction between implemented, built and physically validated.

## Numerical and measurement contract

FP32 GEMM uses FP32 A/B/C/D with `CUBLAS_COMPUTE_32F_PEDANTIC`. The worker sets `NVIDIA_TF32_OVERRIDE=0` and `CUBLAS_EMULATE_SINGLE_PRECISION=0` before library initialization. FP16 and FP8 use FP32 compute and FP32 C/D. FP8 A/B are OCP E4M3, with distinct non-unit scalar scales A=0.5 and B=1.5 (product 0.75), and fast accumulation explicitly disabled. Validation also compares the output against an intentionally unscaled reference: that negative control must be rejected under the same tolerance policy, recorded as `scaled_reference_observable=true`. Physical TN column-major layout is used for every GEMM. FP8 pads physical M/N/K to multiples of 16; throughput always counts logical `2*M*N*K` operations.

Each GEMM case fully checks exact and random non-square small matrices, then checks up to 128 outputs of each performance input buffer against a CPU reference of the actual quantized/scaled inputs. All output values are checked for finiteness. Quantization error is reported separately from kernel error. The fixed tolerance policy is supplied by the shared core. Algorithm tuning examines up to 32 legal heuristic candidates with three timing trials each and caches the chosen opaque algorithm by device/library/type/shape/workspace. Tuning samples are preserved separately from formal samples. For FP16/FP8 shapes with every dimension at least 512, selection first warms the device for 200ms, then measures three batches of approximately 10ms per candidate (up to 256 calls per batch). Full selected tile/stage/split-K/reduction/swizzle/custom configuration is recorded. Formal measurement uses its own minimum warmup and sample duration; see docs/BENCHMARK_COMPARISON.md and scripts/validate-cuda-throughput.ps1.

ALU searches 4/8/16 independent accumulators, workgroup sizes 128/256 and unroll factors 4/8. Runtime dyadic inputs execute the bounded recurrence `x=fma(x,-1,b)`, with b=0.25 and the second half2 lane b=-0.125. Every initial value is nonzero and the sequence remains finite. Half2 registers remain half through the hot loop. Short complete-reference validation and full-iteration sampled-reference plus all-output finite/checksum validation run before measurement; outputs are rechecked afterwards. The operand generator and chosen tuning configuration are recorded. Certification additionally requires hot-loop instruction and spill checks.

Same-stream CUDA events surround the complete ordered batch, and stop completion is awaited before reading elapsed time. Synchronized wall time is retained for every batch. Setup, upload, allocation, scale preparation, tuning and validation readback are excluded. One device/stream is active at a time. Reuse uses one input/output buffer set; rotate uses two and makes no cold-cache promise. Allocation is limited to half the dynamically available device memory. Explicit shapes are never silently shrunk. Warmup and sampling follow the common minimum 500 ms warmup, seven samples and one second aggregate device time. Cancellation is checked between short tuning/validation/measurement batches.

Unsupported FP8 hardware returns SKIP; no legal library algorithm returns configuration_unsupported rather than converting inputs to a different precision. Failed numerical checks never publish throughput. Asynchronous device loss blocks subsequent work on that device in the worker.
