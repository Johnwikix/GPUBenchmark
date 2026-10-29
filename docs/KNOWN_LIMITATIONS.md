# Known limitations

FP4 is out of scope for the first version by user request, and is rejected at argument parsing. It is not reported as hardware unsupported.

AMD and Intel source has not passed a local compile/link/initialization smoke: ROCm Core SDK10.1.0 and a matched oneAPI2026 compiler/oneMKL/UR group are absent. No RX6000/RX7000/RX9000/ArcA/ArcB device is available. Their source/API/WMMA fragments and runtime packaging need owners to run the provided scripts on those actual environments. These are missing software/hardware validation conditions, not certified supported devices. SDK package hashes remain null for packages not acquired.

The only available NVIDIA GPU is an RTX5070Ti. RTX20/30/40 behavior and Windows primary-version coverage remain untested. Windows build26300 is the observed machine, not a blanket certification for every Windows11 build; Windows10 is not certified. NVIDIA FP16 is tested through CUDA; the current617.14 OpenCL ICD exposes no cl_khr_fp16 and its half compatibility case is accurately skipped.

ALU code uses multiple accumulator/workgroup choices and bounded runtime data. Native instruction inspection is available for the CUDA SM120 build; other vendors' ISA/profiler validation is pending. cuBLASLt configured strict/native paths are not a full Nsight profiler capture of Tensor execution. No same-condition external benchmark baseline has been established. Results are measured effective throughput, not guaranteed physical peak rates.

Memory budgets are snapshots under WDDM and may change between query and allocation. Allocation/API/asynchronous failures remain possible and must be preserved. SYCL oneMKL internal workspace is opaque; the source reserves a budget but cannot guarantee a library-internal allocation cap. Dynamic DXGI/free-memory queries may be unavailable on some runtimes; those configurations are skipped. Cache rotation is buffer rotation, never a promise of cold cache. OpenCL ALU rotates output buffers only.

Clean Windows without VS/Toolkit/Python has not been physically tested. Binary package PE imports and actual loaded module paths are audited; host/CUDA use static C++/CUDA runtimes and a private cuBLASLt. This establishes inspected dependencies, not a clean-machine pass. No AMD/Intel binary package is produced until a real Windows build and runtime manifest exist.

Fault handling includes missing workers, malformed stdout, crash, retained completed rows, output failure and cooperative worker cancellation tests. A fixture never emits performance values. Real driver reset/device loss cannot be safely forced on this display GPU, so that recovery scenario is unverified. Display-driver resets can affect multiple processes despite worker isolation. Non-GPU system/environment failures are reported with raw diagnostics rather than repaired by changing drivers or TDR.
