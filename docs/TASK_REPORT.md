# T00—T09 交付报告

交付日期：2026-10-21（Asia/Shanghai）。计划已按用户要求更新为 v1.1：首版不支持 FP4。原始规格保留于根目录计划，所有未取得的环境与未验证项明确列出。

| 任务 | 已实施 / 已构建 / 真实验证 | 证据与剩余事项 |
|---|---|---|
| T00 | 环境探测、固定依赖与来源/哈希；host/OpenCL/CUDA 实际编译运行 | dependencies.lock.json；CUDA13.2.2/nvcc13.2.86/cuBLAS13.4.1.3 + MSVC19.51；AMD/Intel SDK 缺失，候选包哈希留 null，不冒充锁定验证 |
| T01 | 主程序、CLI、JSONL v1、Job Object、结构化错误、输出、取消已实现并构建 | results/host-tests/summary.json；坏消息、崩溃、缺worker、require-all、写出错误、日志隔离、已完成结果保留、异步取消通过；真实 CTRL_BREAK 与 Ctrl+C 使用相同 handler，退出130并保留成功结果 |
| T02 | 公共计数、统计、参考解码、固定误差策略、校准/warmup/采样、schema 已实施并构建 | common_tests 1/1通过；results/schema-validation.json；FP32/half/E4M3/溢出/null/退出码检查；FP4参考已移出首版 |
| T03 | CUDA FP32/FP16 ALU、strict FP32/FP16/FP8 GEMM 已构建并在 RTX5070Ti 验证 | 最终 standard bundle 全13项passed/exit0；FP8 scale乘积0.75及未缩放负对照通过；SM120 FFMA/HFMA2、零spill记录；RTX40及其他代际实机未验证 |
| T04 | HIP ALU、动态 BLAS、strict SGEMM fallback、RDNA3 FP16/RDNA4 FP16/FP8独立WMMA源码已实施 | workers/amd/verification/missing-sdk.json：实际构建退出1、缺ROCm；官方fragment映射静态证据已保存；未编译、未执行AMD GPU、无AMD分数 |
| T05 | Intel GPU专选、SYCL ALU、oneMKL USM strict/mixed GEMM、sentinel完整区间、wall对照源码已实施 | workers/intel/verification/build-intel.json：实际构建退出1、缺DPC++/oneMKL/UR；未编译、无Intel实测；XMX/ISA、opaque内部workspace待设备验证 |
| T06 | OpenCL动态ICD、GPU枚举、FP32/FP16能力、JIT、调优/参考/完整batch已实施并构建 | RTX5070Ti FP32 reuse/rotate真实通过；FP16因无cl_khr_fp16准确SKIP，不否定硬件；RX6000/核显未取得 |
| T07 | CUDA1/2/4/8 batch、ISA、rotate、非方阵padding、scale负对照、缺BLAS隔离、显存预算、真实取消已验证 | results/cuda-*、results/console-cancel/；真实CTRL_BREAK取消约46ms并保留结果；cuBLAS profiler、同条件外部基线、其他厂商、真实driver reset及干净机仍待验证，不宣称全面性能认证 |
| T08 | host/OpenCL/CUDA独立Windows包、源码包、证据包、PE imports及文件/包SHA256已提供 | dist/Release/；只分发私有cuBLASLt和已构建worker，未复制驱动系统DLL；AMD/Intel二进制/runtime再分发未实施验证；无SDK/VS/Python干净Windows部署未验证 |
| T09 | README、构建/方法/支持矩阵/决策/局限、统一运行验证、实际报告与原始证据已交付 | results/RTX5070Ti_REPORT.html（可直接浏览）和同名JSON；设备持有者运行scripts/validate-device.ps1或AMD专用脚本；脚本要求pwsh7.2+，exe无此依赖 |

本机最终主要结果（4096³，dense，FP32累加/输出，CUDA device_elapsed_batch，中位TFLOP/s）：strict FP32 **28.82**，FP16 **102.48**，OCP E4M3 FP8 **312.16**。ALU普通FMA中位：FP32 **55.17**，FP16 packed **56.61**。这些不是同一种运算，不能将ALU与矩阵列混合排名。

最终报告选用 `results/nvidia-0-standard-reuse-20261021-000632/` 的不可变原始数据（13项正式结果全部校验通过）。各case至少7组、累计设备时间≥1s、warmup≥500ms；本轮上述主项未标noisy。JSON/CSV保留每组elapsed/repetitions/FLOPs、误差、scale、algorithm、memory budget、配置和wall对照。采样不会删除慢值。前期 `results/cuda-gemm.json` 的FP8 scale乘积1数据已明确被修正版本取代，最终报告不使用它作为scale验证证据。

环境：NVIDIA GeForce RTX5070Ti，UUID9510a92114f7034232ae2924c362def1，SM120，driver617.14，Windows build26300 x64。该OS build与其他Windows11/Windows10版本不互相继承认证。源目录不是Git仓库，git_commit为null；source/file SHA256清单提供本次身份，未编造commit。

可复现入口：

```powershell
.\scripts\build-windows.ps1 -Backends host,opencl,cuda -Configuration Release -CudaRoot G:\Tool\Cuda\13.2.2\toolkit
.\scripts\test-host.ps1
.\scripts\validate-device.ps1 -Device nvidia:0 -Backend cuda -Profile standard -OutputDir .\results
.\scripts\generate-report.ps1 -EvidenceDirectory .\results\nvidia-0-standard-reuse-20261021-000632
.\scripts\check-schemas.ps1
.\scripts\package-windows.ps1 -Configuration Release -CudaRoot G:\Tool\Cuda\13.2.2\toolkit
```

工具目录由用户指定，源码通过参数引用，不硬编码安装路径。未修改TDR、驱动、频率、功耗限制或电源计划。用户授权无人值守执行，并要求全部落盘后关闭计算机；关机不写入通用构建/验证脚本。
