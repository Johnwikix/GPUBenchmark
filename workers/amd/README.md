# AMD Windows worker

Implementation status: HIP ALU, hipBLASLt GEMM, strict rocBLAS FP32 fallback,
and independent RDNA3/RDNA4 direct WMMA sources are implemented. This checkout
has **not built these sources with a Windows ROCm SDK**, and has **not executed
them on AMD hardware**. There are no AMD scores. The actual build entry returned
exit 1 because `HIP_PATH`/the ROCm Core SDK was absent; see
[verification/missing-sdk.json](verification/missing-sdk.json). The local CIM
query was denied; its empty array must not be interpreted as proof of no GPU.

The initial candidate is ROCm Core SDK 10.1.0 Windows. It remains a candidate
until compiler, headers, runtime, optional math libraries, kernel databases and
driver are acquired, hashed, and validated as one combination. No SDK DLLs or
kernel databases were copied into this checkout.

## Build

From an x64 VS C++ environment, using the selected SDK's root:

```powershell
.\scripts\build-windows.ps1 -Backends host,amd -Configuration Release -HipRoot 'D:\SDKs\ROCm'
```

The host part uses MSVC. `GPUBENCH_AMD_COMPILER` can explicitly select the SDK's
AMD `amdclang++`, `clang++`, or `clang` driver; discovery checks the selected SDK's
`bin` and `lib/llvm/bin`, never an unrelated LLVM installation on PATH. CMake
records its complete version in the worker build metadata. Device translation
units use `--driver-mode=g++ -x hip -std=c++17 -O3` without fast math.

The HIP ALU object embeds gfx1100/1101/1102/1200/1201 by default. Separate
`wmma_rdna3.hip` and `wmma_rdna4.hip` translation units compile to one `.hsaco`
per target with `--cuda-device-only -mno-wavefrontsize64`. A compiler that rejects
a requested architecture fails the build; explicitly reducing
`GPUBENCH_AMD_ARCHITECTURES` creates a partial artifact whose missing targets
remain `kernel_missing`. Installed code objects live beside the worker in
`workers/amd/kernels`. Do not infer product support from compiler target support.

Both math SDK header sets are optional at compile time. Their DLLs are never
ordinary imports: typed function tables use explicit loading beside the worker.
If the corresponding headers are absent the case reports `api_unsupported`;
with headers but missing DLLs it reports `runtime_missing`. No-BLAS deployment
retains HIP enumeration, ALU and direct WMMA. HIP itself is an ordinary import
and its static registration can run before `main`; the matching runtime and
Windows System32 precedence must be validated independently.

## Cases and limits

| Suite | Precision | Implementation |
| --- | --- | --- |
| ALU | FP32 | `fmaf`, 4/8/16 independent accumulators |
| ALU | FP16 | `__hfma2`, 2 half lanes, half accumulation |
| GEMM | FP32 | hipBLASLt `HIPBLAS_COMPUTE_32F`; rocBLAS `rocblas_default_math` SGEMM fallback |
| GEMM | FP16 | half A/B, FP32 compute/output |
| GEMM | FP8 | OCP E4M3 A/B, explicit nonunit tensor scales 0.5 and 1.5, FP32 compute/output |
| matrix_micro | RDNA3 FP16 | gfx1100/1101/1102 wave32 WMMA |
| matrix_micro | RDNA4 FP16/FP8 | gfx1200/1201 generation-specific wave32 WMMA |

FP4 is outside the initial release. RX6000 HIP is not promised; use the separately
reported OpenCL ALU compatibility backend. Direct micro has no implicit iGPU
prefix whitelist. A future architecture remains unknown until a legal probe.
RX7000 FP8 is hardware unsupported; RX9000 missing GEMM algorithms are backend
kernel/configuration failures, never a hardware capability downgrade.

GEMM uses TN column-major physical descriptors: A is KxM, B is KxN, D is MxN.
The logical operation remains D=A*B, alpha=1, beta=0 with no padding. Inputs are
fixed-seed signed nonzero-containing data. No FAST_TF32/BF16 mode is selected.
One workspace and one ordered HIP stream serve the complete operation.

ALU tunes 18 accumulator/workgroup/unroll candidates. Micro tunes 2/4 independent
accumulators and 64/128/256 threads; its distinct inputs prevent duplicate
accumulator elimination. Both calibrate short kernels toward 5 ms and refuse a
calibrated dispatch over 20 ms. Micro counts exactly
`wave_count * independent_accumulators * iterations * 8192` FLOPs; there is no
extra factor of 32. The complete 16x16 outputs from every wave and accumulator
are preserved, never substituted into GEMM results.

RDNA4 FP16 and FP8 do not share the same K-to-lane input map. FP16 interleaves
groups of four K elements; FP8 assigns contiguous groups of eight. The exact
A/B/D layouts were cross-checked against AMD's official matrix instruction
calculator. Its unmodified output tables and source SHA256 are saved under
`verification/rdna*.csv` and `verification/matrix-mapping-reference.json`.
These are static mapping evidence, not GPU execution or compilation evidence.

The short ALU reference checks every channel (half recurrence rounds each FMA
to half); micro uses an exact dyadic CPU reference for every output after three
instructions. Full iteration runs require finite nonzero-containing outputs.
GEMM verifies all outputs for <=4096 elements and 256 deterministic samples for
larger matrices against actual quantized/scaled inputs. Quantization error is
recorded separately. Post-measurement checks retain raw samples on failure and
erase valid throughput. `native_evidence=documented_api_path` is recorded; it is
not a claim of inspected ISA or completed device certification.

Cache `rotate` uses three distinct device allocations with identical reproducible
inputs; it does not promise cold cache. Allocation checks use half the currently
free HIP memory as the budget. Explicit GEMM dimensions are never silently
shrunk. A single GEMM over 100 ms is refused after short calibration. GEMM
internal library allocations are not known ahead of the API call; allocation
failures remain explicit. Event intervals cover the entire ordered batch,
including library auxiliary kernels, with synchronized wall time retained.

## Device-owner validation

After deploying a matching runtime plus licensed math DLL/kernel-data layout:

```powershell
.\workers\amd\verify-amd.ps1 -Device amd:0 -OutputDir .\results\amd-device
```

The script records executable hashes, exact argument arrays, start/end times,
exit codes, raw stdout/stderr and per-case JSON/CSV. It exercises full small
GEMM checks, rectangular boundaries, ALU, independent WMMA, rotate and standard
sizes. `-Extended` adds 8192. A missing matching algorithm for irregular shapes
must be reviewed as a configuration SKIP; it is not validation success.

Additional acceptance evidence still required:

- Use the selected compiler's `llvm-objdump --disassemble` on each `.hsaco` and
  inspect `v_wmma_f32_16x16x16_f16` / `v_wmma_f32_16x16x16_fp8_fp8` in the correct
  objects. Save commands and full disassembly. Inspect ALU code objects for
  native FP32 FMA / packed FP16 FMA, register pressure and spills. A builtin
  name alone is not this evidence.
- Run HIP-only deployment with optional hipBLASLt/rocBLAS DLLs absent; list,
  ALU and micro must remain available. GEMM must explicitly report missing
  runtime. Inspect `dumpbin /imports amd_worker.exe` to confirm no optional
  math DLL import.
- Validate library kernel-data resolution, versions and actual loaded module
  paths; System32 runtime conflicts must be fixed by correct deployment rather
  than by editing global PATH from the benchmark.
- Exercise cancellation, allocation failure, asynchronous/device loss and a
  clean Windows machine. Never change TDR, clocks, power limits or drivers from
  these scripts. Driver-loss injection requires a separate controlled test.
- Record an actual RDNA3 FP16 and RDNA4 FP16/FP8 run, driver, OS, toolchain and
  original samples. No row becomes hardware certified from compilation alone.

## Primary API references checked during implementation

- [AMD RDNA3 WMMA builtin and fragment mapping](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/develop/compiler-builtins/rdna/rdna3-dense-wmma-builtins.html)
- [AMD RDNA4 WMMA builtin and fragment mapping](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/develop/compiler-builtins/rdna/rdna4-dense-wmma-builtins.html)
- [AMD GPUOpen WMMA generations](https://gpuopen.com/learn/accelerating_generative_ai_on_amd_radeon_gpus/)
- [AMD instruction-level layout calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
- [hipBLASLt type and compute contracts](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/data-type-support.html)
- [hipBLASLt API declarations](https://github.com/ROCm/hipBLASLt/blob/develop/library/include/hipblaslt/hipblaslt.h)
- [HIP public runtime declarations](https://github.com/ROCm/hip/blob/develop/include/hip/hip_runtime_api.h)
- [HIPCC `--genco` translation](https://github.com/ROCm/llvm-project/blob/amd-staging/amd/hipcc/src/hipBin_amd.h)
- [Windows runtime loading limitations](https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md)

Online primary references constrain implementation; installed SDK signatures
and actual hardware validation are still required before release certification.
