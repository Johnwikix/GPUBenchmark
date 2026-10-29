# Benchmark 源码对照与第二轮优化（2026-10-21）

后续官方资料核对见 [NVIDIA_OFFICIAL_PEAKS.md](NVIDIA_OFFICIAL_PEAKS.md)：官方峰值基于boost而非base；CUTLASS注明SM120新FP8指令在FP32累加下相对Ada提升2倍，不能把FP8约450的预期一概归因于half累加或稀疏。该补充不改变下列真实样本或精度设置。

此前偏重本地计时与指令取证，没有充分检查其他 benchmark。本轮直接阅读以下固定提交的源代码，不把 README 分数当作同精度实测基线。

| 项目 | FP16 | FP8 | 计数与计时 |
|---|---|---|---|
| [Microsoft SuperBench 3e8f2b5](https://github.com/microsoft/superbenchmark/blob/3e8f2b5e66fb65ef08bded80e693d7537728bee4/superbench/benchmarks/micro_benchmarks/cublaslt_gemm/cublaslt_utils.cc) | FP32 compute；输出由配置控制 | FP32 compute，显式 FAST_ACCUM=1 | CUDA events包围多次调用，2MNK / 每次平均时间 |
| [binbinsh/gpu-bench d8bca5d](https://github.com/binbinsh/gpu-bench/blob/d8bca5d941248f778cae6e70293ae971e528411a/benchmarks/quantized_benchmark.cu) | 类型选择函数返回 COMPUTE_16F，即half累加 | 独立函数使用FP32 compute/output，NN，未显式配置本项目的非单位缩放验证 | 2MNK×实际迭代数 / 完整事件区间；默认8192，长时间运行 |
| [clpeak-gb10 7688326 GEMM](https://github.com/pontostroy/clpeak-gb10/blob/76883260e9a89b5293ddf0bc4424fabc4f0a3bff/src/cuda/cuda_blas.cpp) | COMPUTE_16F，half C/D | FP32 compute，BF16 C/D | 2MNK；8个候选实测选优；按设备选矩阵尺寸 |
| [clpeak-gb10 同提交微基准](https://github.com/pontostroy/clpeak-gb10/blob/76883260e9a89b5293ddf0bc4424fabc4f0a3bff/src/cuda/cuda_kernels/wmma_fp16.cu) | 寄存器WMMA循环，声明4条FP32累加链 | 另有寄存器mma.sync循环 | 按指令循环统计，非完整GEMM；FP16存在死代码问题，见下文 |

上述源文件、完整提交、原始URL和SHA256保存在 `results/throughput-diagnosis/external/`。未运行第三方安装脚本，未把其源码并入产品。不能把某项目所有分数归为同一种语义；需要逐case检查累加、输出、稀疏、FAST_ACCUM、尺寸、验证和计时。

这些GEMM均采用有效FLOPs **2×M×N×K**，与本项目一致。本项目独立复算原始样本未发现漏算FMA的乘二。不能直接给分数乘二，也不能相加ALU和Tensor GEMM成绩。

## 微基准死代码证据

clpeak-gb10 的FP16循环写c0/c1/c2/c3；收尾先把c1写入c0，再把c3写入c2，最后把c2写入c0。输出不再依赖循环中的c0/c1/c2，但host仍按四条链统计，见同提交 `src/cuda/wmma.cpp` 的workPerWI。

本机以CUDA13.2.86、`-O3 -arch=sm_120 -cubin`复编该固定源码并用nvdisasm查看，256次循环体只有两条HMMA.16816.F32，即一条16×16×16 WMMA链；循环外还有4条HMMA。声明四条循环链、实际保留一条，存在接近4倍虚高风险。这里是**本机复编产物的指令证据**，不是作者原二进制或其设备的跑分复现。cubin、SASS和构建日志已保留。第三方高分也必须经过机器码与计数检查。

## 实现与实验

独立诊断新增 `tune`，枚举算法ID，查询tile/stage/custom/swizzle，检查合法性及workspace，再实测。45秒预算耗尽时明确记录，不称为穷尽搜索。FP16检查8601个配置、实测743个；FP8检查29647个、实测3707个。没有找到同精度翻倍的配置。这种昂贵枚举仅用于诊断。

正式FP16/FP8大矩阵选优新增200ms设备预热，再对每候选测3组约10ms的调用批次，减少低频启动和单次事件噪声。小案例和FP32原选法保留。算法记录新增tile/stage/split-K/reduction/swizzle/custom及选中配置，每组调优重复次数单独保存。正式分数仍经过独立500ms预热、至少7组且累计至少1秒计时。FP32 compute/output、dense、FAST_ACCUM=false和固定校验容差均未改变。

本轮诊断中位数（TFLOP/s，所有计时前后校验通过）：

| 配置 | FP16 acc32 | FP8 E4M3 acc32 |
|---|---:|---:|
| 4096³，旧单次选优重新建立基线 | 96.89 | 310.43 |
| 4096³，扩大配置搜索 | 102.04 | 308.30 |
| 4096³，预热后批次选优 | 102.57 | 313.41 |
| 8192³，预热后批次选优 | 104.36 | 356.17 |
| 6144³，旧选优 | 103.56 | 344.87 |
| 12288³，旧选优 | 未测 | 364.46 |
| 14336³，旧选优 | 未测 | 361.75 |

冷启动旧选优的一轮96.89不能代表所有旧版表现，历史FP16已测102–103。因此不能宣称本次稳定提升6%，也未证明FP16峰值显著提高。不同尺寸单独报告，不静默更改用户显式尺寸。

独立诊断JSON的measurement复用公共计时器，implementation字段不是正式认证结果。正式host/worker回归保存在 `results/throughput-optimization/production/`：4096³中位数FP16 **100.76**、FP8 **313.65**；8192³为 **104.22 / 350.06**；FP32 ALU回归 **55.33**。奇数非方阵FP16/FP8及全部性能case均通过前后校验，退出码0。原始命令、环境/二进制/库哈希、JSON/CSV、协议日志与独立2MNKR审计保留。

新增设备持有者入口：

```powershell
.\scripts\build-windows.ps1 -Backends cuda -CudaRoot G:\Tool\Cuda\13.2.2\toolkit
.\scripts\validate-cuda-throughput.ps1
# 额外尺寸需显式指定，超预算会返回不可用：
.\scripts\validate-cuda-throughput.ps1 -Sizes 4096,8192,12288
```

可读报告和独立复算汇总为 `results/throughput-optimization/REPORT.html`、summary.json。复算入口 `python tests/diagnostics/summarize_optimization.py`。

## 尚未解决

FP16约220不能直接验收本计划acc32项；half累加需要另定义case、误差策略与输出，不能覆盖现有acc32。目前未交付通过验证的acc16分数。

FP8 **450仍未实测达到**。通过校验的本轮最佳尺寸诊断约364；不能把其他程序的低位输出、FAST_ACCUM或微基准分数混用。此前实际QMMA/SASS证据仍有效，计数器仍受ERR_NVGPUCTRPERM限制。其他benchmark在本机完整同条件二进制复现、硬件计数器归因仍未完成，不能声称已找到所有差距来源。

原dist/Release完整归档与首份报告保留为历史版本，本轮通过单独优化补丁包交付。没有修改驱动权限、TDR、系统功耗或频率。
