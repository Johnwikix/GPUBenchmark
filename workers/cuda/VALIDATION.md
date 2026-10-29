# CUDA validation evidence — 2026-10-20

Follow-up 2026-10-21: warm-device batched FP16/FP8 algorithm selection built and
validated through the production host/worker on the same RTX5070Ti. Odd
non-square and 4096/8192 cases passed, with independent 2MNKR audits and FP32
ALU regression. See docs/BENCHMARK_COMPARISON.md and
results/throughput-optimization/production/. Median at8192: FP16acc32 104.22,
FP8acc32 350.06 TFLOP/s; FP32ALU 55.33. Actual Tensor launch/SASS evidence was
obtained in results/throughput-diagnosis/, but hardware counters remain denied.
The original release package/report below remains historical evidence.

Implemented, native Windows build passed, and physically exercised on **NVIDIA GeForce RTX 5070 Ti**, SM120, UUID `9510a92114f7034232ae2924c362def1`, driver 617.14, Windows build 10.0.26300.0. This is evidence for that exact device/environment; RTX20/30/40 and other RTX50 models are not thereby certified.

Toolchain: CUDA redistributable 13.2.2, nvcc 13.2.86, CUDA runtime 13020, CUDA driver API 13040, cuBLASLt 13.4.1.3 / API version 130401, MSVC 19.51.36260.0. The root dependency lock records download URLs and hashes. No unsupported compiler option, driver update, TDR modification, power limit or clock change was used.

## Build and probe

The actual build command was:

```powershell
.\scripts\build-windows.ps1 -Backends cuda -CudaRoot G:\Tool\Cuda\13.2.2\toolkit -Configuration Release
```

The first sandboxed build failed with MSVC PDB C1902, preserved in `workers/cuda/build-windows.log`. An authorized build outside that sandbox passed, preserved in `build-windows-elevated.log`. Subsequent corrective builds also passed; `build-windows-frozen.log` is the final build log. MSVC `/Zs` host-only syntax checking was performed earlier, and is not counted as CUDA build or hardware proof.

The first tiny FP32 GEMM probe exposed lazy library kernel initialization entering an event interval and triggering the dispatch budget. Its failure is retained in `results/cuda-doctor.json`. Initialization was moved outside the event interval. Final `results/cuda-doctor-frozen.json` records **all five implemented cases passed**; its actual command/exit code is in `cuda-doctor-frozen-command.json`.

## Benchmark evidence

`results/cuda-commands.json` and `results/cuda-additional-commands.json` retain actual commands, exit codes and timestamps. JSON results retain every formal sample, repetition count, logical FLOPs, CUDA event time, synchronized wall time, validation metrics and selected tuning/algorithm configuration. CSV and stdout/stderr logs are adjacent.

* `cuda-alu.json`: FP32 and FP16 ALU, reuse, both passed, exit 0.
* `cuda-gemm.json`: FP32 strict and FP16 acc32 at 1024/2048/4096, all passed, exit 0. Its original FP8 rows use scale product 1 and are retained as earlier evidence only.
* **`cuda-fp8-scaled.json` supersedes the earlier FP8 rows.** It uses A=0.5, B=1.5, product=0.75, with an enforced negative control rejecting the unscaled reference. Every pre/post validation has `scaled_reference_observable=true`. All sizes passed, exit 0.
* `cuda-non-square-rotate.json`: M=31/N=37/K=19, all three GEMM precisions passed, exit 0; FP8 physical shape is 32/48/32. Both rotated buffer groups are independently validated before and after measurement.
* `cuda-alu-rotate.json`: FP32/FP16, both groups validated, both passed, exit 0.

Representative earlier direct-run results (TFLOP/s):

| Case | Logical size | Median | Best | Evidence |
| --- | --- | ---: | ---: | --- |
| ALU FP32 | recorded threads/iterations | 54.70 | 54.96 | cuda-alu.json |
| ALU FP16 | recorded threads/iterations | 55.28 | 56.18 | cuda-alu.json |
| GEMM FP32 strict | 4096³ | 28.15 | 28.76 | cuda-gemm.json |
| GEMM FP16 acc32 | 4096³ | 101.65 | 102.27 | cuda-gemm.json |
| GEMM FP8 E4M3 acc32 | 1024³ | 169.94 | 178.36 | cuda-fp8-scaled.json |
| GEMM FP8 E4M3 acc32 | 2048³ | 293.49 | 294.52 | cuda-fp8-scaled.json |
| GEMM FP8 E4M3 acc32 | 4096³ | 311.34 | 315.15 | cuda-fp8-scaled.json |

These are actual samples, not theoretical throughput. The scaled FP8 1024 and 4096 direct runs are marked **noisy** by the fixed policy; no samples were removed. Final integrated standard-profile evidence may provide a later round and must be identified separately, rather than replacing unfavorable samples.

The FP8 4096 reference max kernel error was `1.1444091796875e-5`; ignoring scales produced max error `48.288272857666016` and was rejected for 108 of 128 sampled outputs. Quantization error is reported independently. A small exact non-square case checks 391 outputs and a random non-square case checks 589 outputs completely.

`calibration_linearity` in the new GEMM results records 1/2/4/8 repetitions, actual FLOPs, full event intervals and wall times independently of formal samples. The FP8 4096 direct run measured approximately 0.392/0.794/1.568/3.578 ms. Queue/launch overhead and GPU clocks prevent exact linearity, especially for tiny shapes; raw values are preserved. Batches projected above 200 ms are skipped in this diagnostic sweep.

## Failure/deployment checks

* `cuda-memory-budget.json`: explicit 32768³ FP32 requires 17,448,304,648 bytes against a 7,882,670,080-byte dynamic allocation budget; **SKIP before allocation**, `configuration_unsupported/memory_budget`, throughput null, exit 3.
* `cuda-missing-blas/`: a separate worker directory contains only the executable. Enumeration passed, ALU FP32/FP16 passed, and GEMM FP32/FP16/FP8 each returned `runtime_missing/cublaslt_dll_missing`, throughput null, exit 3. Commands and logs are retained in that directory. The working deployment was not disturbed.
* `cuda-worker-imports.txt` and `cuda-cublaslt-imports.txt`: `dumpbin /dependents` reports only KERNEL32.dll as ordinary imports for each. Driver modules are loaded dynamically; runtime module paths are in doctor/results. This does not replace a clean Windows deployment test.

## Instruction evidence and remaining work

The final executable SHA256 is `ABBCEF5A1C45948E9D52F03EDB4068BB4291BFA8C275F0E87A92C7EA7D211447`. `results/cuda-isa-evidence.json` records its hash, exact cuobjdump command and the disassembly hash. `cuda-sm120-sass-frozen.txt` contains retained **FFMA** and **HFMA2** instructions and no local LDL/STL spills for the ALU kernels. The original ptxas build log reports SM120 13–37 registers for FP32 and 22–40 for FP16, zero stack/spill bytes. Static instruction counts include compiler unrolling and remainder code; they are not used as dynamic FLOP counts.

Still unverified: RTX40 FP8 and other targeted GeForce generations; clean Windows with no Toolkit/VS/Python; an independent same-conditions performance baseline; cuBLASLt Tensor Core profiler traces. No profiler-verified Tensor Core claim is made. FP4 is outside the current first-release scope. Device-lost stopping is implemented and host scheduler checks cover its propagation; an actual driver reset was not intentionally induced.

Run the root `scripts/validate-device.ps1` on another device to generate a self-contained owner-operated validation directory with environment, telemetry, command exit codes, JSON/CSV, logs and SHA256 hashes.
