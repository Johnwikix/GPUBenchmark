# Support and evidence matrix

Statuses are independent: source implemented, Windows build passed, real device validated. Hardware architecture documentation never establishes a measured score.

| Target | Source | Windows build | Real hardware | Missing condition |
|---|---|---|---|---|
| Host CLI / protocol v1 | implemented | passed native MSVC x64 | not a GPU score | results/build-final-delivery.log; results/host-tests/summary.json |
| OpenCL ALU | implemented | passed native MSVC x64 | RTX5070Ti FP32 passed; FP16 API SKIP | driver617.14 does not expose cl_khr_fp16; results/opencl-initial.json |
| RTX20/30 CUDA | implemented / native SASS built | passed SM75/86 build | untested | physical devices; no generation-wide certification |
| RTX40 FP8 | implemented / native SASS built | passed SM89 build | untested | RTX40 owner validation |
| RTX5070Ti SM120 | implemented | passed CUDA13.2.2 + MSVC19.51 | ALU FP32/FP16 and GEMM FP32 strict/FP16/FP8 passed | driver617.14 / OS26300; raw standard bundles in results/ |
| RX6000 OpenCL | implemented | worker built | untested | RX6000 owner validation |
| RX7000 HIP / WMMA | implemented source | SDK missing; build not reached | untested | ROCm10.1.0 + RX7000 |
| RX9000 HIP / WMMA FP8 | implemented source | SDK missing; build not reached | untested | ROCm10.1.0 + RX9000 |
| Arc A / B | implemented source | SDK missing; build not reached | untested | matched oneAPI DPC++/oneMKL/UR + Arc A and B |
| Intel/AMD iGPU | best_effort | SDK dependent | untested | device capability probe; no name-only matrix claim |

Observed device: NVIDIA GeForce RTX5070Ti, UUID GPU-9510a921-14f7-0342-32ae-2924c362def1, PCI 00000000:01:00.0, driver617.14, 16303MiB, Windows10.0.26300.0 (not yet certified primary OS build). All future evidence must include OS, GPU identity, driver, runtime and original command.

FP4 is deferred, not a failed hardware capability. CUDA source versions and ISA/non-square/rotate/scale/linearity/fault evidence are indexed in workers/cuda/VALIDATION.md and results/cuda-*. The selected standard run is identified in results/RTX5070Ti_REPORT.json. cuBLASLt profiler, external same-condition baseline, other GPUs and clean-machine deployment remain unverified.

2026-10-21 optimization follow-up: RTX5070Ti FP16/FP8 warm-device batched selection built and physically validated (4096/8192 and odd non-square), FP32 ALU regression passed. Actual cuBLASLt launch symbols and Tensor SASS are recorded in results/throughput-diagnosis/. Hardware performance counters remain denied; a full external same-condition binary baseline remains pending. Open-source code comparison and measured limits are documented in BENCHMARK_COMPARISON.md. This does not extend certification to any other GPU.
