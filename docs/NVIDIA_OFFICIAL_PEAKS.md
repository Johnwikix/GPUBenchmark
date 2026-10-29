# NVIDIA 官方峰值的条件与算法口径（2026-10-21）

## 能够直接确认的官方条件

[RTX Blackwell 白皮书v1.1，表5，PDF第52页及第53页脚注](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf#page=52)的RTX5070Ti峰值基于 **GPU Boost Clock 2452MHz**。官方产品页另外列出的base clock是2300MHz；不能把该峰值表说成基频实测。

| 输入 / 累加 | Dense TFLOP/s | 启用稀疏后的有效TFLOP/s |
|---|---:|---:|
| FP16 / FP16 | 175.8 | 351.5 |
| FP16 / FP32 | 87.9 | 175.8 |
| FP8 / FP16 | 351.5 | 703 |
| FP8 / FP32 | 175.8 | 351.5 |

表中斜线前后是dense与sparsity两种条件，不是最小/最大值。FP32非Tensor峰值为43.9。上述数值忠实转录该白皮书，不擅自改表。

[NVIDIA性能背景指南](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/)说明峰值的计算方法：每SM每周期multiply-add数 × 2 FLOPs/FMA × SM数 × 时钟。因此它是矩阵乘加执行单元的理论吞吐规格，不对应“某个 cuBLAS 算法ID在某个矩阵上实际测得该分数”。检索到的规格表未提供M/N/K、cuBLAS版本、算法ID、tile、正式样本或用于得出峰值的可执行benchmark。

## 必须补上的SM120 FP8信息

[NVIDIA CUTLASS官方SM120文档](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms)给出更新的指令级说明：

* `mma.sync.aligned.kind::f8f6f4`：支持F8/F6/F4混合输入；FP32累加吞吐相对Ada FP8为2倍。
* `mma.sync.aligned.kind::mxf8f6f4.block_scale`：同样标注FP32累加相对Ada FP8为2倍。
* SM120只支持TN；采用TMA warp-specialized kernel，支持ping-pong与cooperative调度，数据搬运、计算与epilogue重叠；cluster为1×1×1。

这意味着不能把白皮书中FP8 acc32的旧半速倍率当成所有SM120 FP8路径的统一硬件上限。这里有不同官方文档在指令覆盖和吞吐表述上的差异；没有找到NVIDIA对表5与新SM120指令逐项对应的明确勘误，不能替官方断言究竟是哪一行印错。

基于CUTLASS指令吞吐比做**推算**，SM120全速dense FP8 FP32累加可按约2048 FLOPs/SM/cycle估计：70SM×2048×2.452GHz≈351.52TFLOP/s。若实测平均SM频率为3.10GHz，则推算约444.42，而不是把450一概归为FP16累加或稀疏。该数字是指令吞吐推算，不是本机已经实现的跑分。

FP16则应独立比较：按表5，在3.10GHz推算，dense FP16 acc32约111.10，acc16约222.21。先前实测acc32约104与“应该220”之间主要有累加语义差异；仍需基于运行期间实际平均频率计算利用率，而不能拿跑分前后空闲频率或一个最高频率样本作分母。

## 官方可用实现入口

[CUTLASS mma_sm120.hpp](https://github.com/NVIDIA/cutlass/blob/91843b71d90737e0a9b86f7d9df213cff5a3526f/include/cute/arch/mma_sm120.hpp)包含E4M3×E4M3→F32的完整 `kind::f8f6f4.m16n8k32` 包装。

[官方87a示例](https://github.com/NVIDIA/cutlass/blob/91843b71d90737e0a9b86f7d9df213cff5a3526f/examples/87_blackwell_geforce_gemm_blockwise/87a_blackwell_geforce_fp8_bf16_gemm_blockwise.cu)展示FP8 blockwise scaling、float累加、TMA persistent GEMM和epilogue优化。它默认BF16输出及块级缩放，与本项目float输出/标量缩放不同，不能原样跑完就作为同条件acc32基线。另有79c的MXFP8示例；同样不能把MX块缩放与当前scalar E4M3 case混成一项。

## 当前实机证据与不能推断的结论

已有cuBLASLt实际启动的SM120 kernel使用 `QMMA.SF.16832.F32.E4M3.E4M3.E8`，因此本程序已经有原生Tensor路径，不能因为约350–364而说没调用Tensor。

本次用锁定CUDA13.2.86，分别编译有/无`kind::f8f6f4`的两个最小FP8指令kernel；两者在sm_120a都生成 `QMMA.16832.F32.E4M3.E4M3`。这是**编译/指令映射证据，没有运行这些kernel或发布新分数**。它也表明不能仅添加PTX前缀或改`sm_120`为`sm_120a`便声称获得2倍实测提升。库kernel中的SF路径需要结合内部缩放与数据移动分析，不能仅凭SASS名称反推唯一PTX源码。

官方文档不能单独解释本机FP8从约350到约450之间的全部差距。后续同条件基线需要：保留FP32累加/float输出/非单位scale/FAST_ACCUM=false，比较原生SM120 GEMM调度、tile与流水线，记录负载期间频率，并完成数值校验。不要为贴合峰值而改计数或降低精度。

原始PDF、表5渲染、固定提交的官方源码、URL/SHA256和本次最小编译产物保存在 `results/official-peaks-research/`。本次未修改正式worker或历史发布包。
