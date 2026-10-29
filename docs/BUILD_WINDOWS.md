# Windows native x64 build

Use PowerShell 7.2 or newer (pwsh). The executable itself does not require PowerShell. Host requires MSVC C++ desktop tools, Windows SDK, CMake >=3.25 and Ninja. The script discovers Visual Studio with vswhere and initializes an x64 developer environment. Dependencies are vendored and hashed in dependencies.lock.json; no network is needed for host/OpenCL builds. No Python is required to run the product.

```powershell
.\scripts\build-windows.ps1 -Backends host,opencl -Configuration Release
.\scripts\build-windows.ps1 -Backends cuda -Configuration Release -CudaRoot '<CUDA13.2.2>'
.\scripts\build-windows.ps1 -Backends amd -Configuration Release -HipRoot '<ROCm10.1.0>'
# Initialize the matching oneAPI environment first (setvars.bat in cmd, then powershell).
.\scripts\build-windows.ps1 -Backends intel -Configuration Release
```

Each backend has its own build/<backend>-Release directory. Install output: out/Release/gpubench.exe and out/Release/workers/<backend>/*.exe. Optional workers can be omitted without preventing host startup. Never enable allow-unsupported-compiler. Do not replace drivers or edit global PATH/TDR/power policy.

Environment observed 2026-10-20: native Windows x64 build 26300; VS 18.10.4, MSVC 19.51.36260.0/toolset 14.51.36231; bundled CMake 4.3.1-msvc1 and Ninja 1.11.1.git.kitware.jobserver-1. CUDA 11.8.89 is installed but insufficient for SM120. CUDA12.9 documents MSVC193x/192x, so the installed MSVC1951 is also a mismatch. The initial mismatch was resolved by acquiring CUDA13.2.2 (nvcc13.2.86/cuBLAS13.4.1.3), which explicitly supports MSVC195x; its actual build and RTX5070Ti execution passed. AMD/Intel candidates remain unvalidated. ROCm/oneAPI and AMD/Intel GPUs were not found. CIM inventory was denied in the execution sandbox; OS was queried through the Win32 environment and NVIDIA through nvidia-smi.

Acquisition sources and package hashes for missing SDKs remain null until acquired and verified. This is an explicit incomplete lock, never a claim that a missing package is pinned and tested. Record the exact SDK/compiler/runtime group, acquired package SHA256, driver and smoke logs before publishing a new backend package. Scripts fail with nonzero status for missing prerequisites rather than silently changing backend or configuration.

Final CUDA acquisition: scripts/acquire-cuda.ps1 -Destination G:\Tool\Cuda\13.2.2. The script checks a fixed official manifest SHA256 and each component archive SHA256. Tools remain under the user-selected directory, without driver installation. CMake explicitly lists project/vendor header dependencies because this machine has a localized showIncludes parsing problem; SDK headers remain part of the immutable locked toolchain. Use -Clean for a clean build.
