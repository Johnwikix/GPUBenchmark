# Measurement and validation contract v1

The governing plan is WINDOWS_CONSUMER_GPU_BENCHMARK_AGENT_PLAN.md v1.1. FP4 was deferred by the user on 2026-10-20 and is not a runnable case. These results measure native consumer GPUs on Windows x64, without changing clocks, driver, power plan or TDR.

ALU performs ordinary fused multiply-add on several independent register accumulators. CUDA uses __fmaf_rn and __hfma2; HIP uses fmaf/half2; SYCL and OpenCL use typed fma, with capability checks. Half accumulators stay half during the loop. Initialization/conversion/final writeback is outside the hot recurrence but within the timed kernel; its work is not added to FLOPs. Every live accumulator is written to prevent dead-code elimination. CUDA uses an exactly representable signed dyadic recurrence x=fma(x,-1,b), with distinct half2 addends and runtime nonzero inputs. OpenCL uses x=fma(x,0.9990234375,b), runtime seed and per-lane signed nonzero addends, full finite checking and CPU recurrence comparison. Results identify each selected kernel/configuration; architecture certification still requires instruction/register/spill evidence.

GEMM computes D=A*B, alpha=1, beta=0, dense. Logical shape, physical padding, transposition and leading dimensions are explicit. FP32 inputs/accumulation/output are FP32, with CUDA pedantic mode or Intel standard compute mode, TF32 and single-precision emulation disabled. FP16 inputs accumulate and output FP32. FP8 inputs are OCP E4M3 with recorded actual tensor scales, FP32 accumulation/output; no FNUZ/E5M2 or precision conversion fallback is counted. Reference values use actual quantized bytes and scales; quantization_error and kernel_error are separate. Non-unit scale validation must use a scale product different from one so ignoring both scales cannot pass by cancellation.

AMD matrix_micro repeatedly computes cooperative 16x16x16 D=A*B+D. Each wave32 instruction performs 8192 FLOPs, multiplied by waves, independent accumulators and iterations, never by lanes again. RDNA3 and RDNA4 use separate fragments/code objects. It is a distinct suite and never substitutes for complete GEMM. Short loops validate complete results against a CPU reference; long loops check all outputs finite and sample their expected values.

GEMM FLOPs per dispatch = 2*M*N*K, using logical dimensions. ALU FLOPs = 2*threads*independent_accumulators*iterations*scalar_lanes. Packed half2 has two scalar lanes and four FLOPs per packed FMA. Batch operations multiply by the fixed repetitions R. All integer products reject uint64 overflow. TFLOP/s = batch_flops / (elapsed_ms * 1e9). Median and best use the same fixed-R raw batch samples, with best defined by minimum valid elapsed time. Unavailable, failed or cancelled items have null throughput; zero is never an unsupported marker.

Single ALU dispatches calibrate toward 5ms and stay within 1–20ms when feasible. GEMM initializes/JITs before checking the conservative 100ms single-dispatch budget. A batch targets about100ms, calibration reduces repetitions when a trial exceeds200ms, and formal repetitions stay fixed. At least500ms warmup, seven formal samples and one second cumulative device time are required. Calibration/tuning samples are distinct from formal samples. Every formal elapsed time, repetition count, operation count and synchronized wall time is retained. No samples are discarded because they are slow. The shared loop flags a range/median above10% as noisy; this is an engineering flag, not a vendor guarantee. Cancellation is consumed on a separate control thread and acted on between short batches.

CUDA/HIP start and stop events surround the entire batch on the same stream. OpenCL spans first kernel start through last kernel end on a profiling in-order queue. Intel uses actual USM-write sentinel kernels on an in-order profiling queue: front sentinel end through back sentinel start, with first GEMM depending on front and back depending on last completion. A returned library event is not assumed to cover all internal kernels. All compulsory internal library/workspace kernels belong inside that interval. When a reliable device interval cannot be read, a host synchronized batch must be labeled host_qpc/host_synchronized_batch and kept out of device elapsed comparisons.

Allocation, upload, external quantization/scale preparation, descriptors, algorithm search, validation readback, IPC, logging and initialization are excluded from formal batch timing. Private vendor logging is disabled and effective environment recorded, without changing the parent's environment. Buffers use reuse or recorded rotation groups. Rotation does not establish cold caches. OpenCL ALU synthesizes register inputs from runtime parameters and rotates output buffers; it does not claim rotating matrix inputs. Allocations are limited to half a dynamic available budget: CUDA/HIP runtime free memory or DXGI local budget/current usage. OpenCL uses an exact LUID adapter match, or a unique vendor adapter for budget accounting only (never for identity deduplication). If no usable dynamic budget exists, the configuration is skipped. Intel oneMKL internal workspace is opaque; reserved budget and this limitation are recorded rather than claiming an exact internal cap.

CPU reference dot products accumulate actual decoded values in long double (on MSVC this has double precision). For FP32 accumulation, u=2^-24, gamma_K=K*u/(1-K*u), for K*u<0.5. A sampled dot passes when abs(actual-reference) <= 1e-6 + 4*u*abs(reference) + 4*gamma_K*sum(abs(a_i*b_i)). K*u>=0.5 is outside useful forward-error certification and must not be used to claim a strong bound. The policy is fixed before running, includes dtype/K and separate kernel/quantization metrics, and is not enlarged after a failure. Exact constructed small cases and signed random non-square/boundary/padded cases catch layout/index errors before sampling performance shapes. CUDA samples up to128 performance elements per buffer and checks every output for finiteness; small cases are fully compared. ALU reference rules are backend-specific and saved with each result (CUDA exact dyadic = zero tolerance; OpenCL half exact and float absolute2e-6). Inputs are never all-zero defaults.

Evidence hierarchy: source implemented, Windows build passed, validated on an identified GPU, ISA inspected, profiler verified, and clean-machine deployment are independent statuses. documented_api_path can document the actual native API used after successful execution; build success alone cannot. Direct ISA evidence from SM120 verifies the ALU instruction family and spill behavior only. A cublas API configuration alone is not profiler proof of a particular tensor kernel. Official performance percentages are not pass criteria; no external same-condition baseline has been claimed when none was measured.


## 2026-10-21 precision and SM120 extension

`gemm.fp16.acc32` remains the default and retains its original random inputs,
FP32 accumulation/output and error policy. `--fp16-accum fp16|both` adds the
independent `gemm.fp16.acc16` CUDA case (CUBLAS_COMPUTE_16F, half scale/C/D).
Its peak input generator is signed dyadic +/-1/8, initialized at runtime with
SplitMix64; exact and varied-magnitude non-square small cases are also checked.
Every formal output must be finite, sampled references must match exactly, and
pre/post validation uses zero absolute/relative/sum-abs tolerance. This structured
peak workload is not a claim of general FP16 accumulation accuracy and is not
mixed with acc32's numerical distribution. The input generator is recorded.

Optional pinned CUTLASS4.3.4 implements SM120 E4M3 GEMM with float accumulation
and float C/D. Unit-block-scale variants use UE8M0 encoding127 (=1), preserving
the same E4M3 input bytes; actual per-tensor scales0.5/1.5 are applied as float
epilogue alpha0.75. Logical GEMM alpha remains1. No quantization, scale creation
or parameter upload is moved into a different throughput count. All necessary
MMA, TMA loads/stores, scale loads, epilogue and descriptor reads stay inside the
complete batch interval. Algorithm search and immutable parameter preparation
are outside formal timing. No cuBLAS FAST_ACCUM mode is enabled; native kernels
hold float accumulator fragments. Exact/random native small cases and the
existing high-precision/negative-scale controls validate the selected path.

MSVC needs conforming preprocessing and `/Zc:__cplusplus` for CUDA tensor-map
alignment. Its aligned-by-value parameter limitation is handled by a pointer
entry to NVIDIA's unchanged kernel, using aligned device Params. Parameter
copies and scale initialization complete on the same nonblocking execution
stream before timing; this is part of correctness, not a timing shortcut.

Raw samples now include UTC start/end milliseconds. `run-gpu-observed.ps1`
records nvidia-smi at100ms intervals; the report correlates matching GPU UUIDs
only within formal sample windows, retaining busy rate, SM clock, power,
temperature and power-cap flags. Driver utilization readings average an internal
window and are not Tensor pipeline occupancy or memory bandwidth percentages.
No driver, clock, TDR or power-limit setting is changed.
