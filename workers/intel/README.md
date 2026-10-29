# Intel worker

Status on 2026-10-20: implementation delivered; **not compiled** and **not run on Intel hardware**. This machine has no `icx`/DPC++ compiler, no oneMKL development/runtime installation, and no Intel GPU. Source review and an expected missing-SDK failure do not establish build compatibility or produce benchmark scores.

The first release contains FP32/FP16 register FMA ALU, FP32 strict GEMM, and FP16 input / FP32 accumulation and output GEMM. FP8 requests on identified consumer Arc A/B return hardware `unsupported`; other architectures remain `unknown`. No conversion path is used. FP4 is outside the revised release. Intel matrix_micro is outside this release.

The worker selects explicit Intel GPU objects from GPU-only SYCL enumeration. It never constructs a CPU or default selector. Arc Pro, Max, Flex and data center products are excluded. Identical UUIDs prove duplicate API views; otherwise separate backend-qualified identities are preserved. Device UUID, PCI address, LUID and PCI device ID are queried only when the matching DPC++ extension aspect exists. An unrecognized model is probed, without assuming a future precision capability.

Initialize a **specific pinned** Windows oneAPI 2026 release that includes DPC++, oneMKL SYCL BLAS, SYCL runtime, Unified Runtime adapter and a supported Intel graphics driver. Run the repository build script from that initialized x64 shell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Backends intel
```

The independent CMake entry is:

```powershell
cmake -S . -B build/intel-Release -G Ninja -DGPUBENCH_BACKEND=intel -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icx
cmake --build build/intel-Release
cmake --install build/intel-Release --prefix out/Release
```

`MKLROOT` must select the same pinned SDK series. CMake uses the installed oneMKL SYCL imported target and fails if it is missing; it does not replace it with the classic CPU BLAS target. Intel's changed imported target names `MKL::MKL_SYCL` and `MKL::MKL_DPCPP` are accepted. The compiler requires the 2026 series and oneMKL at least 2026.0. These guards are source/API expectations, not a verified dependency combination. Lock exact compiler/library/runtime versions, redistribution package hashes, supported host MSVC and Windows SDK after a real build. The target uses the dynamic MSVC CRT to match the SYCL runtime.

After installation, on a device holder's Windows machine:

```powershell
out/Release/gpubench.exe doctor --json results/intel-doctor.json
out/Release/gpubench.exe list
out/Release/gpubench.exe run --device intel:0 --backend intel --suite alu --dtype fp32,fp16 --samples 7 --duration-ms 1000 --warmup-ms 500 --json results/intel-alu.json --csv results/intel-alu.csv
out/Release/gpubench.exe run --device intel:0 --backend intel --suite gemm --dtype fp32,fp16 --m 17 --n 19 --k 13 --json results/intel-boundary.json
out/Release/gpubench.exe run --device intel:0 --backend intel --suite gemm --dtype fp32,fp16 --sizes 1024,2048,4096 --cache reuse --json results/intel-gemm-reuse.json
out/Release/gpubench.exe run --device intel:0 --backend intel --suite gemm --dtype fp32,fp16 --sizes 1024,2048,4096 --cache rotate --json results/intel-gemm-rotate.json
out/Release/gpubench.exe run --device intel:0 --backend intel --suite gemm --dtype fp8 --require-all --json results/intel-fp8-unavailable.json
```

Use the repository validation script for evidence collection, version/path inventory and command exit codes. Preserve the unedited JSON/CSV, stdout/stderr, exact commands and driver versions. Set `MKL_VERBOSE=0`, `ONEMKL_VERBOSE=0` and `MKL_BLAS_COMPUTE_MODE=STANDARD` in the child environment before startup. The launcher must handle DLL startup failure; imports can fail before `main`. Runtime DLLs, SYCL, UR and adapter must be taken from one verified release and deployed according to its redistribution license. This implementation does not copy SDK redistributables or drivers.

ALU scans 64/128/256 workgroup sizes, 4/8/16 independent accumulators, unroll factors 4/8, and scalar/half2 forms where applicable. Runtime input and bounded, nonzero dyadic coefficients stay in registers during the FMA loop. All accumulators are written once. Selected iterations target a short dispatch. Short tests check all outputs; full-length tests check finite values for every output and use CPU fused, per-step-rounded reference values on sampled lanes. Every rotate group is validated before and after sampling. Allocation failure reduces and reports ALU thread geometry. Explicit GEMM dimensions are never reduced.

GEMM uses row-major N/N device-USM pointers, alpha 1, beta 0. Its explicit `oneapi::mkl::blas::compute_mode::standard` precedes the dependency vector; half inputs infer the official mixed overload with float scalars and output. Complete exact and random odd/non-square preflight matrices precede sampling. FP32 random inputs contain bits that a truncated TF32/BF16 implementation would discard. Large outputs get finite checks plus reference samples calculated from actual stored inputs; quantization error is reported separately.

Timing encloses the entire dependency-chained batch in an in-order queue. Both sentinels perform real device-USM writes. The first operation depends on the front marker; the back marker depends on the final completion event. The interval is front `command_end` to back `command_start`, not the profiling duration of oneMKL's returned event. Each raw sample also keeps synchronized wall time. Sentinel writes and timing agreement are checked. Missing/invalid profiling during initial selection uses `host_qpc` / `host_synchronized_batch`; timer scope is fixed before formal sampling, and a later failure never silently changes it.

Allocation budget is half the currently reported SYCL free memory and/or LUID-matched DXGI budget, using the more conservative available value. When neither is available the configuration is skipped with `dynamic_budget_unavailable`; total memory is never presented as a current budget. Three rotate groups are accounted for. oneMKL allocates internal workspace privately: the worker reserves at least 64 MiB and the requested workspace allowance, but cannot enforce a library scratch limit through this API. Reports state the workspace size is opaque and the reserve is not a measured allocation. Dynamic budgets can change after querying.

Remaining acceptance work requires a matched SDK build and Intel hardware: compile/link/initialization, driver/API errors, Arc A/B functional and performance runs, integrated-GPU best-effort behavior, missing DLL/adapter, out-of-memory, cancellation, forced device loss where safely reproducible, and reliable sentinel timestamp coverage on the actual adapter. Inspect the selected ALU hot loop for native half FMA and spills/register usage. Profile FP16 GEMM before claiming XMX. Results intentionally expose only `documented_api_path` and `isa_checked=false`, `profiler_checked=false`, `xmx_verified=false`; no ISA, XMX or hardware certification is claimed.
