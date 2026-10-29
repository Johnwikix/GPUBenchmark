# RTX5070Ti 吞吐疑问诊断：2026-10-21

结论：没有发现统一除以2的计数/时间错误；FP16和FP8确实启动了Tensor kernel。上次总览只突出GEMM主项，容易与整卡算术峰值混淆，需要同时看suite、输入/累加精度、完整GEMM开销及实际频率。FP32 GEMM仍有优化空间，不能把正确性通过当成峰值达成。

## 真实复现与计数审计

命令及退出码：results/throughput-diagnosis/command.json。原始结果：repro-4096.json/CSV/log。固定4096³、至少7组/1秒设备时间，三项均校验通过：

| case | 中位TFLOP/s | batch repeats | 原始计时中位ms |
|---|---:|---:|---:|
| FP32 strict GEMM | 28.89145 | 21 | 99.89869 |
| FP16 input / FP32 accumulation/output | 98.53476 | 78 | 108.79651 |
| E4M3 input / FP32 compute/output | 316.17857 | 259 | 112.58413 |

count-audit.json 使用独立Python计算核对：每条原始sample的FLOPs精确等于2*M*N*K*R，所有中位TFLOP/s与独立重算一致。同步wall/device比值为1.00066、1.00059、1.00022，没有时间多算一倍或FMA漏乘2的证据。

此前正式ALU FP32实际55.17 TFLOP/s，接近用户的56预期。该项是多独立寄存器FMA，与完整GEMM约29不是同一suite。FP16 ALU使用普通half2 FMA，也不等于Tensor FP16 GEMM峰值。

## 假设、单变量对照与结果

假设顺序：精度/峰值口径不一致；库算法/布局限制；FP8 fast-accum准确性设置；最后排查统一计数或时间倍数错误。计数已用独立审计排除，随后使用独立诊断target逐项改变一个变量，不改变生产worker的精度合同。

| 对照 | 中位TFLOP/s | 判断 |
|---|---:|---|
| FP32 TN pedantic 4096 | 28.95 | 复现基线 |
| FP32 TN standard、TF32/emulation仍关闭 | 28.94 | pedantic不是减半原因 |
| FP32 NN、其余相同，4096 | 34.35 | 布局影响实际性能，当前生产TN限制了搜索范围 |
| FP16 NN、其余保持FP32 accumulate，4096 | 102.18 | 未出现2倍提升 |
| FP8 FAST_ACCUM=1，4096 | 310.27 | 开关不是当前差距的主要解释；仅诊断，未替换生产准确性设置 |
| FP32 NN，8192 | 34.94 | 更大尺寸仍未达到普通FMA峰值 |
| FP16 baseline，8192 | 103.34 | 同一acc32合同仍约100 |
| FP8 baseline，8192 | 355.19 | 比4096更高，但没有测到450 |

所有上表数字均来自原始GPU样本且通过前后参考验证；comparison-commands.json、size-commands.json记录命令/退出码。部分小对照使用独立诊断程序，同生产代码共用分配、参考、算法搜索与事件计时；其结果不覆盖原发布报告。曾尝试单独FP16累加实验，但未通过其预先设置的误差阈值，未发布吞吐；失败日志保留为fp16-acc16.stderr.log，原型已移除，未放宽阈值或混入acc32成绩。

## Tensor执行证据

使用官方CUPTI13.2.86记录实际cuLaunchKernel/cuLaunchKernelEx，并在module-loaded回调保存对应cubin；再用锁定的cuobjdump反汇编。actual-kernel-evidence.json记录实际启动符号、选中kernel的独立SASS、SHA256；不是只看整个库里是否存在某类指令。

| 精度 | 实际启动的kernel族 | 选中函数中的机器指令 |
|---|---|---|
| FP32 GEMM | cutlass_80_simt_sgemm_128x256_8x4_tn_align1 | FFMA；普通CUDA Core路径 |
| FP16 GEMM | cutlass_80_tensorop_s16816gemm_f16_64x64_32x6_tn_align8（不同调优轮可能选64x256变体） | HMMA.16816.F32；Tensor路径 |
| FP8 GEMM | nvjet_sm120_qqsss_mma_128x128x64_6_64x64x64_tmaAB_alignCD4_bz_TNNN | QMMA.SF.16832.F32.E4M3.E4M3.E8；SM120原生Tensor路径 |

完整GEMM按2MNK统计有效逻辑工作量，事件覆盖整个batch。它包含实现必须的设备工作，不是只测一条Tensor指令。CUDA ALU和Tensor GEMM是不同工作负载的独立结果，不能把两列分数相加成“总算力”。

## 峰值口径与未解决部分

[NVIDIA RTX Blackwell白皮书v1.1，表5（PDF第52页）](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf#page=52)明确把FP16 accumulate和FP32 accumulate、dense和sparse分开。RTX5070Ti标称boost下，FP16 Tensor的FP16 accumulate dense为175.8，FP32 accumulate dense为87.9 TFLOP/s；实际boost更高时数值会相应变化。因此用户约220的预期不能直接用来验收当前FP16 acc32项。不能为了显示220把FP32累加偷偷改成FP16累加。

FP8的剩余差距尚未完全归因：已确认原生QMMA、输入E4M3和F32寄存器累加指令，尺寸增大提高到355，但不能仅凭白皮书或dtype比值解释成某个固定倍率。实际库选的是SM120 QMMA.SF路径，不能把旧架构/另一指令族的吞吐比直接当测量修正系数。需要继续分析实际频率、功耗限制、kernel流水线/内存和硬件计数器，才能量化各自影响。

Nsight Compute2026.1.1已按官方SHA256校验获取，在不锁频、不修改cache的设置下尝试采集；明确失败ERR_NVGPUCTRPERM，完整命令与日志见fp16-ncu-command.json、fp16-ncu.log。没有修改驱动计数器权限、时钟、功耗、TDR或电源计划，也没有用失败的counter采集宣称利用率。上述kernel启动及SASS证据不依赖计数器权限，已成功获取。

本诊断阶段主程序/正式worker计数与精度未修改。后续源码对照、扩大搜索和正式选优改进见 [BENCHMARK_COMPARISON.md](BENCHMARK_COMPARISON.md)。可选GPUBENCH_CUDA_DIAGNOSTICS target默认关闭；原发布归档仍是原版本。
