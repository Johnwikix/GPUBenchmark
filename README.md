# gpubench

Windows 原生 x64 消费级 GPU 终端 benchmark，C++20 主程序与独立 CUDA、HIP、SYCL、OpenCL worker。FP32/FP16 普通 FMA 与完整 dense GEMM 分别报告；原生 FP8 仅使用 OCP E4M3。**首版暂不支持 FP4**，`--dtype fp4` 返回参数错误 2。

## 五分钟上手

已有发布包：将 host 包解压到目录，再将可选 cuda/opencl 包解压到同一目录，保持 `workers/<backend>` 结构。运行不需要 Python、Visual Studio 或完整 Toolkit。CUDA 包需要兼容的 NVIDIA 驱动；OpenCL 包需要驱动提供 ICD。AMD/Intel 当前只有未构建源码，不能当作可用二进制包。

```powershell
.\gpubench.exe --help
.\gpubench.exe list
.\gpubench.exe doctor --json doctor.json
.\gpubench.exe run --device nvidia:0 --backend cuda --suite alu --dtype fp32,fp16
.\gpubench.exe run --device nvidia:0 --backend cuda --suite gemm --dtype fp32,fp16,fp8 --sizes 1024,2048,4096 --json result.json --csv result.csv
.\scripts\validate-device.ps1 -Device nvidia:0 -Backend cuda -Profile standard -Executable .\gpubench.exe
```

PowerShell 脚本要求 pwsh 7.2+；gpubench.exe 本身不需要 PowerShell。

从源码构建：

```powershell
.\scripts\build-windows.ps1 -Backends host,opencl -Configuration Release
.\scripts\acquire-cuda.ps1 -Destination G:\Tool\Cuda\13.2.2
.\scripts\build-windows.ps1 -Backends cuda -Configuration Release -CudaRoot G:\Tool\Cuda\13.2.2\toolkit
.\scripts\package-windows.ps1 -Configuration Release -CudaRoot G:\Tool\Cuda\13.2.2\toolkit
.\scripts\package-allinone.ps1 -Configuration Release
```

`package-allinone.ps1` 生成单目录免安装整合包 `dist\Release\gpubench-0.1.0-win-x64-allinone.zip`（host+cuda+opencl 合并，静态 MSVC/CUDA 运行时，仅依赖系统 DLL；任意 Windows 10/11 x64 可直接解压运行，CUDA 路径需要 Turing 及以后显卡配 r580+ 驱动）。

工具路径由参数指定；上面的 G: 路径是本次用户选择，不是源码硬编码依赖。实际依赖版本、来源、哈希和待验证项在 [dependencies.lock.json](dependencies.lock.json)。CUDA13.2.2（nvcc13.2.86/cuBLAS13.4.1.3）与本机 MSVC19.51 已完成构建。其他 SDK 未获取，锁文件中的候选版本不冒充验证通过。

`--backend auto` 优先能够证明为同一设备的原生后端；无法证明身份相同时保留 backend 限定 ID。`--device all` 顺序测试。`--sizes` 和 `--m/--n/--k` 互斥，仅控制 GEMM。`--cache rotate` 记录缓冲组数，不承诺冷缓存。首版始终 native-only。

典型 SKIP：缺 worker → not_built；缺可选 BLAS → runtime_missing（ALU 仍可用）；没有 OpenCL FP16 扩展 → api_unsupported，不能推断 GPU 无 FP16；RX7000/ArcA/B 请求 FP8 → hardware unsupported；RX9000 缺算法 → kernel/configuration 不可用，不能改报硬件不支持。所有不可用吞吐为 null，校验失败拒绝发布分数。

退出码：0 至少一项成功且无执行失败；2 参数错误；3 无可运行项；4 执行/校验/写出失败；5 require-all 存在不可用项；130 取消。优先级 130 > 2 > 4 > 5 > 3 > 0。

报告和原始证据见 results/，本地发布包见 dist/Release/。只把真实硬件执行记录列入支持矩阵，不以 mock、编译成功或文档声明替代实机验证。主程序不修改驱动、TDR、电源计划、频率或功耗限制。构建、方法、支持状态和局限分别见 docs/BUILD_WINDOWS.md、METHODOLOGY.md、SUPPORT_MATRIX.md、KNOWN_LIMITATIONS.md。
# FP16 acc16 与原生 SM120 FP8 增量

保留原有 FP16 acc32，新增独立的 `gemm.fp16.acc16`。默认仍为 acc32。

```powershell
.\scripts\acquire-cutlass.ps1 -Destination G:\Tool\Cuda\cutlass-v4.3.4
.\scripts\build-windows.ps1 -Backends host,cuda -CudaRoot G:\Tool\Cuda\13.2.2\toolkit -CutlassRoot G:\Tool\Cuda\cutlass-v4.3.4
.\out\Release\gpubench.exe run --backend cuda --device nvidia:0 --suite gemm --dtype fp16,fp8 --fp16-accum both --cuda-gemm auto --sizes 4096,8192 --json results/precision-comparison.json
```

`--cuda-gemm auto|cublaslt|sm120` 控制 FP8 路径；FP16 仍走 cuBLASLt。
auto 实测多个原生 SM120 候选，仅在更快时选择；显式 sm120 未编译或形状不适用时
返回准确 SKIP。可用 `--sm120-variant 0..7` 显式复现某个 SM120 候选。
CUTLASS 为锁定的可选静态构建依赖，运行时不要求安装 CUTLASS。

acc16 是独立的峰值测试：FP16 输入/累加/输出，输入为有符号 dyadic 值，
使用零容差参考校验。它与 acc32 的通用随机输入分布不同，不能据此宣称相同数值精度。
FP8 始终为 E4M3、FP32 累加/输出、dense，并验证非单位张量缩放。
单位块缩放候选内部 scale 为1，不重新量化输入；实际0.5/1.5张量缩放在FP32 epilogue应用。

同步观察 GPU 利用率、频率、功耗和温度：

```powershell
.\scripts\run-gpu-observed.ps1 -Name my-5070ti-run -BenchmarkArguments @('run','--backend','cuda','--device','nvidia:0','--suite','gemm','--dtype','fp16,fp8','--fp16-accum','both','--sizes','8192','--duration-ms','3000')
```

每100ms记录一次 SMI 遥测；正式样本保存 UTC 时间窗口。GPU 忙碌率不等于 Tensor
管线占用率。结果保留原始样本、命令、退出码、二进制哈希与校验状态。

