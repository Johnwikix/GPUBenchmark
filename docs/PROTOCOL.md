# Worker protocol v1 (frozen)

UTF-8 JSON Lines on stdin/stdout; stderr is diagnostics only. Every envelope has protocol_version=1, request_id (integer/string/null), type, payload object. Worker starts with hello, request_id=0, containing build_info plus backend. No GPU pointers or SDK objects cross this boundary.

enumerate {} -> devices {devices:[device...],runtime:{...}}. Device contains id (backend qualified), alias (nvidia:0/amd:0/intel:0 where applicable), backend_index integer, name, vendor_id, device_id, architecture, memory_bytes, integrated, identity object (uuid/luid/pci_bus_id when available).

probe {device:device} -> capability object. run receives {device,suite,dtype,m,n,k,samples,duration_ms,warmup_ms,seed,workspace_mib,cache,native_only}. m/n/k only apply to GEMM. dtype is fp32/fp16/fp8. run -> progress* -> result (payload is the individual result, not a wrapper), or error {code,backend_status,message}. Use gb::base_result, gb::unavailable and gb::measure. Result fields are flat except device/configuration/build/validation metric objects. Timing source must be set by backend; scope device_elapsed_batch or host_synchronized_batch. All unavailable performance fields are null. No simulated scores.

cancel is consumed by the worker control thread and sets an atomic flag; check between short batches. shutdown cancels and joins the execution thread. Only one operation runs at a time. Cancellation acknowledgements are progress responses with the cancel request_id; final run result retains its original request_id. stdout is mutex protected.

Backend interface and entry loop: src/common/worker.hpp. Counting, formats, errors, statistics, result factory, sample loop: src/common/core.hpp. Each backend uses its own CMake directory and links gpubench_common. install worker to workers/<backend> beneath the install prefix. Backends must not modify public headers or schemas independently.
