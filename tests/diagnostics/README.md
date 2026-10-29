# CUDA throughput diagnosis

Optional diagnostic program. Production arithmetic remains unchanged; the new
warm-device batched algorithm selection is documented in
docs/BENCHMARK_COMPARISON.md and results/throughput-optimization/.

From the same x64 MSVC development environment as the production build:

```powershell
cmake -S . -B build/cuda-diagnose -G Ninja -DCMAKE_BUILD_TYPE=Release -DGPUBENCH_BACKEND=cuda -DGPUBENCH_CUDA_DIAGNOSTICS=ON -DCMAKE_CUDA_COMPILER=G:/Tool/Cuda/13.2.2/toolkit/bin/nvcc.exe -DCUDAToolkit_ROOT=G:/Tool/Cuda/13.2.2/toolkit
cmake --build build/cuda-diagnose --target cuda_gemm_probe
```

Deploy the locked cublasLt64_13.dll beside the diagnostic executable. Run with
`<fp32|fp16|fp8> <size> <baseline|stable|tune|standard|nn|fastaccum>`.
Baseline retains the old single-dispatch selection; stable uses the production
warm-device batched selection. Tune also enumerates cuBLASLt algorithm IDs,
tile/stage/custom/swizzle capabilities, checks legality/workspace, and tests legal
configurations within a 45-second search budget. It records whether the budget
was exhausted; it is not a claim to have exhaustively tested every configuration.
It retains the seed algorithm unless a faster candidate is found. The selected
algorithm must pass the same pre/post validation as baseline. This expensive
enumeration is diagnostic only, not enabled in production.
Standard compares FP32
compute without permitting TF32/emulation; nn compares physical A storage while
preserving the logical matrix. fastaccum is FP8-only and never replaces the
production accuracy mode. Non-equivalent modes stay separately labeled.

For actual launch/cubin capture, acquire the fixed CUPTI13.2.86 archive from the
official CUDA13.2.2 manifest, configure GPUBENCH_CUPTI_ROOT to its extracted root,
and deploy its DLLs next to the executable. Pass `profile` as the fifth argument.
Set GPUBENCH_CAPTURE_CUBIN_DIR to an existing output directory to retain selected
module cubins. CUPTI records actual launch symbols; cuobjdump selects their SASS.
Use the separate `ncu` fifth argument when running under Nsight Compute, avoiding
competing CUPTI subscribers. Disable profiler clock/cache controls and do not
publish profiler-instrumented runs as ordinary performance samples.

Nsight Compute hardware counters are currently denied (ERR_NVGPUCTRPERM).
Kernel launch/SASS evidence succeeded without changing driver permissions.
An abandoned FP16-accumulate experiment failed its predetermined numerical
check; no throughput was published and that path is absent from this program.
