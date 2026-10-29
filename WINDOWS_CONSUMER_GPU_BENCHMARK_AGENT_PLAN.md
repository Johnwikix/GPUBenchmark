# Windows 消费级 GPU 浮点吞吐 Benchmark Agent 执行计划

版本 1.2（2026-10-21：用户授权独立FP16累加测试和官方SM120优化；FP4继续延后）  
范围基准日期 2026-10-20  
项目暂定名 gpubench  
交付目标 Windows 原生 x64 终端程序及可复现的源码 构建脚本 测试记录

本计划是编码任务规格。执行 agent 应完成实现、构建、可用设备上的验证和发布包，而不是再次返回一份实现建议。硬件和 Windows 验证不足时，继续完成不依赖该条件的工作，并明确记录尚未验证的项目。禁止把文档支持、编译成功、模拟结果或 mock 测试表述为实际 GPU 跑分通过。

## 1 执行指令和范围

### 2026-10-21 用户授权的增量范围

保留原有 `gemm.fp16.acc32` 与 `gemm.fp8.e4m3.acc32`；新增独立
`gemm.fp16.acc16`（FP16 A/B、FP16累加、FP16 C/D，dense）。默认仍选择acc32，
使用 `--fp16-accum fp32|fp16|both` 显式选择。acc16是峰值吞吐测试，性能输入
固定为SplitMix64有符号±1/8 dyadic值；以实际量化输入的CPU高精度参考做零容差
校验，同时完整验证小型带不同幅度的非方阵。该输入分布与acc32通用随机输入
分开记录，不将acc16表述为通用数值精度认证。失败不扩大容差、不发布有效分数。

允许引入锁定CUTLASS4.3.4的可选静态构建依赖，通过官方SM120 collective builder
实现FP8 E4M3→FP32累加/输出的完整GEMM；不引入SM100内核，不改变FP4延后范围。
`--cuda-gemm auto|cublaslt|sm120` 控制路径；auto只在原生SM120候选实测更快时
采用，显式sm120缺编译内核/不支持形状时准确SKIP。缩放应用、算法与内核证据记录。

测试期间同步采集GPU忙碌率、SM频率、功耗、温度，正式样本保存UTC起止时间，
与遥测时间对齐。GPU忙碌率不是Tensor管线利用率；计数器权限不足须继续准确说明。
220/450是优化目标，不是修改运算量、精度、计时或校验规则的理由。

### 1.1 给执行 agent 的启动指令

```text
完整阅读本计划和工作区已有 AGENTS.md，按任务 T00 至 T09 实施 gpubench。
仅面向 Windows 原生 x64 和消费级 GPU。以本计划的数据语义、计时规则、
错误分类和验收清单为完成标准。不要扩展到 Linux、WSL、数据中心 GPU、
训练框架或图形界面。
先完成 T00 环境与版本锁定以及 T01 公共协议，再并行推进各后端。
普通实现选择自行决定并记入决策记录；不得因为没有某家 GPU 就停止其他工作。
没有真实设备时只报告实现/构建状态，生成可由设备持有者直接运行的验证包。
必须保留真实命令、退出码、版本、原始样本和失败原因，不得编造 TFLOPS。
最终交付可构建源码、Windows 包、使用说明、验证矩阵和明确的未完成项。
```

### 1.2 产品范围

主要验收目标：

| 厂商 | 主要系列 | 基础测试 | 原生低精度矩阵测试 |
| --- | --- | --- | --- |
| NVIDIA | GeForce RTX 20 和 RTX 30 | FP32 FP16 ALU 和 GEMM | FP16 |
| NVIDIA | GeForce RTX 40 | FP32 FP16 ALU 和 GEMM | FP16 FP8 E4M3 |
| NVIDIA | GeForce RTX 50 | FP32 FP16 ALU 和 GEMM | FP16 FP8 E4M3 |
| AMD | Radeon RX 6000 | OpenCL FP32 FP16 ALU，按驱动能力启用 | 不承诺 HIP GEMM 或 WMMA |
| AMD | Radeon RX 7000 | FP32 FP16 ALU 和可用 GEMM | FP16 |
| AMD | Radeon RX 9000 | FP32 FP16 ALU 和可用 GEMM | FP16 FP8 E4M3 |
| Intel | Arc A 和 Arc B 消费级独显 | FP32 FP16 ALU 和 GEMM | FP16 |

这张表表示实现目标，不等于所有型号均已通过实机认证。NVIDIA 精度依据见 [S01]、[S02]；AMD 依据见 [S04]、[S05]；Intel 原生与转换实现的区别见 [S08]。

笔记本对应架构、Ryzen 核显、Intel Iris Xe 和 Core Ultra 核显允许按运行时能力执行，初始状态为 best_effort。不要只按营销名称认定矩阵单元存在；也不要因设备不在名称白名单里拒绝所有能力探测。更老消费卡仅进入可用的兼容路径。

明确排除：NVIDIA 数据中心产品、RTX PRO 等专业产品，AMD Instinct 和 Radeon PRO，Intel Max/Flex/D-Series/Arc Pro，服务器与云实例专项适配，以及非 Windows 执行平台。不为这些设备建立首版测试目标。

首版暂不支持 FP4，NVFP4/MXFP4 的实现、参考解码、CLI case、验证与发布验收均延后；显式请求 --dtype fp4 返回参数错误（退出码 2），不报告为硬件不支持。不实现 CDNA3 FNUZ、Intel FP8 转换路径。未来新架构默认 unknown，不能把“当前消费级 Intel 不支持”硬编码成永远按厂商拒绝。

### 1.3 系统范围

- 主验收平台：Windows 11 x64。AMD ROCm 路径优先在官方列出的 Windows 11 25H2 验证。[S06]
- Windows 10 和其他 Windows 11 版本：兼容尝试，不自动继承主平台认证。若正式承诺 Windows 10，必须单独锁定仍支持它的工具链并验证；DPC++ 2026.2 已移除 Windows 10 支持。[S10]
- 不以 WSL、Linux 容器或 Linux 编译成功代替 Windows 原生验收。
- 程序运行不要求 Python、VS 或完整开发 SDK；无法做到独立运行的后端必须明确声明所需 runtime。
- 主程序不能因缺失任何厂商 SDK/DLL 而无法启动。
- 不自动安装或替换显卡驱动，不修改 TDR 注册表、系统电源计划、GPU 频率或功耗限制。

## 2 已确定的技术路线

### 2.1 语言与依赖

主程序使用 C++20 和 CMake；公共接口只依赖 Windows 和少量固定版本的开源库。设备代码可使用该工具链验证过的 C++17/20 子集。

固定如下分工：

| 组件 | 实现 | 用途 |
| --- | --- | --- |
| gpubench.exe | MSVC C++20 | CLI 调度 统计 输出 worker 生命周期 |
| cuda_worker.exe | CUDA C++ 加 cuBLASLt | NVIDIA ALU 与全部首版 GEMM |
| amd_worker.exe | HIP C++，显式加载 hipBLASLt 和按需 rocBLAS | AMD 原生 ALU GEMM 和 WMMA 微基准 |
| intel_worker.exe | DPC++ SYCL 加 oneMKL BLAS | Intel ALU 和 FP32 FP16 GEMM |
| opencl_worker.exe | C++ 加 Khronos OpenCL headers/loader | RX6000 核显及原生后端不可用时的 ALU 兼容测试 |

必须实现 OpenCL 兼容 worker，但其功能只要求设备枚举和 FP32/FP16 ALU，不扩展到通用矩阵库。

基础路径不要求CUTLASS；2026-10-21用户授权的可选SM120路径允许锁定CUTLASS。oneDNN、rocWMMA、Vulkan、Direct3D、Triton、PyTorch不作为首版依赖。矩阵手写内核仅用于明确的补缺任务，不从头编写覆盖三家的高性能 GEMM 库。

NVIDIA 的正式 FP16 ALU 走 CUDA。OpenCL 未报告 cl_khr_fp16 时不得强行启用实验 half 路径，不能由此认定 GPU 缺少 FP16 硬件。[S17]

### 2.2 独立 worker 进程

三个原生后端分别构建为独立进程，避免一份可执行文件同时链接不同 SDK 的初始化和 DLL 导入链。OpenCL 同样独立。主程序通过 UTF-8 JSON Lines 协议控制 worker。

- 每个 worker 的协议版本为 1，启动后先发送 hello 和构建信息。
- 消息包含 protocol_version、request_id、type、payload。
- 命令至少包含 enumerate、probe、run、cancel、shutdown。
- 响应至少包含 hello、devices、capability、progress、result、error。
- stdout 只输出协议；诊断日志写 stderr。
- launcher 在子进程环境中明确关闭厂商库的 stdout 日志，或重定向到独立文件；正式测量关闭详细日志。至少覆盖 HIPBLASLT_LOG_LEVEL、HIPBLASLT_LOG_MASK、HIPBLASLT_LOG_FILE 及所用 Intel/NVIDIA 库的 verbose 设置，保存有效配置，不修改用户的全局环境。hipBLASLt 在未指定日志文件时可能直接写 stdout。[S14]
- worker 读取控制消息的线程与执行线程分离，使 cancel 不必等整个测试结束才被读取。
- 不通过协议传 C++/SYCL/CUDA 对象或跨进程 GPU 指针；worker 内完成分配、计算、验证。
- 正常保持 worker 存活以复用上下文。进程启动、SDK 初始化和 IPC 不进入设备计时。
- 主程序使用 Windows Job Object 管理子进程；Ctrl+C 先协作取消，超时再终止 worker 并保留已完成结果。
- 后端进程异常不应使主程序丢失其他结果。进程隔离不能保证隔离整个显示驱动复位，设备丢失后应停止该设备上的后续测试。

worker 的目录各自保存匹配的 runtime。AMD 文档特别说明：System32 可能优先于 PATH，普通导入 DLL 及 HIP 静态注册可能在 main 前执行；不能用 main 内临时 LoadLibrary 或随意拼 PATH 掩盖版本冲突。[S07]

amd_worker 的基础枚举、HIP ALU 和 direct WMMA 不得普通导入可选 hipBLASLt/rocBLAS DLL。使用显式加载与函数表，或可捕获加载失败的延迟加载；若固定工具链不适用，分出独立 GEMM worker。HIP runtime 完整而 BLAS 缺失时，基础 worker 必须仍可运行，并仅将 GEMM 报为 runtime_missing。这不替代 HIP runtime 本身在 main 前的正确部署。其他被定义为可选的依赖遵守同一规则。

### 2.3 构建基线

T00 先发现已有环境并锁定实际能编译、运行的版本。以下是起始候选，不是未验证的兼容性承诺：

| 后端 | 起始候选 | 必须锁定的信息 |
| --- | --- | --- |
| CUDA | CUDA 12.9.2 或经验证更新的稳定补丁 | toolkit cuBLASLt nvcc host MSVC GPU 架构 |
| AMD | ROCm Core SDK 10.1.0 官方 Windows 包 | HIP 编译器 runtime hipBLASLt rocBLAS kernel 数据 驱动 |
| Intel | 同一受支持 oneAPI 发布系列的 DPC++ 和 oneMKL | compiler oneMKL SYCL Unified Runtime adapter 驱动 |
| OpenCL | 固定提交的 Khronos headers/loader | headers loader GPU ICD 版本 |

CUDA 工具链仍需支持 Blackwell GeForce SM120，不引入 SM100 数据中心内核。[S02]、[S03] ROCm 10.1.0 已有 Windows HIP/hipBLASLt 组件；不要按旧 SDK 文档判定 Windows 不可用。[S06]

将选定版本、下载来源、包哈希、编译器完整版本和验证结果写入 dependencies.lock.json。不得发布依赖 latest 的构建；不得使用 allow-unsupported-compiler 绕过主机编译器不匹配。

主程序与每个 worker 使用独立 CMake build directory。AMD/Intel 编译器不覆盖主程序的 CMAKE_CXX_COMPILER。安装路径从参数或发现逻辑获取，禁止硬编码开发机用户名或 SDK 路径。

## 3 测试语义与数据约定

### 3.1 首版 suite

| suite | 含义 | 首版要求 |
| --- | --- | --- |
| alu | 寄存器中反复执行普通浮点 FMA | FP32 FP16 必须实现 |
| gemm | 完整 dense 矩阵乘法 | FP32 FP16 以及硬件支持的 FP8 |
| matrix_micro | 反复执行矩阵指令，减少数据搬运 | 首版必须实现 AMD RDNA3 FP16、RDNA4 FP16/FP8；作为独立项目或库缺项时的补缺路径 |

matrix_micro 永远单独成行，不能替换 gemm 的数值。首版不要求 NVIDIA/Intel 也实现 matrix_micro；其缺失不得伪造分数。后续扩展可以统一此 suite。

AMD matrix_micro 的源码与目标架构构建不取决于开发机是否缺少库算法。用户显式选择时独立运行；suite=all 可同时给出 GEMM 和 micro 各自的结果。没有 AMD 实机时保留硬件验证未完成状态。

默认仅输出原生计算路径。允许记录非原生支持信息，但不运行“低精度存储转高精度计算”的模拟测试来填满四列。

### 3.2 数值合同

| case_id | A 输入 | B 输入 | 累加 | 输出 | 缩放 |
| --- | --- | --- | --- | --- | --- |
| alu.fp32.fma | FP32 | FP32 | FP32 | FP32 checksum | 无 |
| alu.fp16.fma | FP16 | FP16 | FP16 | 保存或转换最终 checksum | 无 |
| gemm.fp32.strict | FP32 | FP32 | FP32 | FP32 | 无 |
| gemm.fp16.acc32 | FP16 | FP16 | FP32 | FP32 | 无 |
| gemm.fp16.acc16 | FP16 | FP16 | FP16 | FP16 | 无；独立dyadic峰值输入 |
| gemm.fp8.e4m3.acc32 | OCP E4M3 | OCP E4M3 | FP32 | FP32 | 每张量缩放，按 API 实现 |
| matrix_micro.fp16.acc32 | FP16 | FP16 | FP32 | FP32 校验结果 | 无 |
| matrix_micro.fp8.e4m3.acc32 | OCP E4M3 | OCP E4M3 | FP32 | FP32 校验结果 | 无，直接使用有限可表示输入 |

- GEMM 定义为 D = A*B，alpha=1，beta=0。逻辑矩阵形状和布局在报告中明确；物理转置/打包可以因 API 而异。
- 数值含义中的 alpha=1 不妨碍后端按 SDK 规则正确应用实际输入量化 scale。
- FP32 主项禁用 TF32、BF16 分解模拟和其他替代计算模式。
- FP16 ALU 的累加器保持 half，避免无意提升到 float；packed 变体记录 lanes_per_instruction。
- 默认FP16 GEMM保留FP32累加/输出；FP16 accumulate单独作为gemm.fp16.acc16，不混入acc32。
- FP8 首版仅要求 A/B 都为 E4M3。E5M2 和混合输入为后续扩展，避免承诺所有类型笛卡尔积。
- 不启用结构化稀疏；density 固定 dense。
- 不以启发式 TFLOPS 比值推断执行资源；保留 API 配置、内核身份和可用的指令证据。
- 普通浮点乘加中的融合舍入属于预期语义，跨实现校验允许合法归约顺序差异。
- matrix_micro 的独立 accumulator 反复执行 D = A*B + D；输入和循环次数必须使结果保持有限，短循环用参考计算完整校验。它不执行完整的大矩阵 GEMM。

cuBLAS 的低精度类型、scale 和 compute mode 依据 [S01]；Intel 的严格 compute mode 依据 [S09]。

### 3.3 运算量

GEMM 每次逻辑运算量按 2*M*N*K FLOPs，重复 R 次乘 R。TFLOP/s = ops / (elapsed_seconds * 1e12)。

需要 padding 时同时记录 logical_shape 和 physical_shape；主指标按用户逻辑形状计算有效吞吐，不能用填充后的运算量冒充用户完成的工作量。

ALU 按实际执行的 FMA 次数计算：每个标量 FMA 为 2 FLOPs，两个 half 的 packed FMA 为 4 FLOPs。loop overhead、转换、checksum、scale 生成不另算作主 FLOPs。维度和乘法计数必须做溢出检查。

AMD 16x16x16 WMMA 的每个 wave 每次协作矩阵运算为 8192 FLOPs；不能在乘 wave 数后再乘 32 lanes。其他形状按正式 ISA 定义，不能仅从函数名猜计数。[S05]

### 3.4 精度能力与执行状态分离

硬件能力：

```text
hardware_support = supported | unsupported | unknown
backend_status = ready | not_built | runtime_missing | driver_incompatible |
                 kernel_missing | api_unsupported | configuration_unsupported | probe_failed
result_status = ok | skipped | failed | cancelled
implementation = native | converted | unknown
```

result 为 skipped/failed/cancelled 时吞吐字段为 null，不用 0 或 NaN。native 需要记录证据来源，例如 documented_api_path、isa_checked、profiler_checked；编译成功本身不是 native 执行证据。

关键规则：

| 情况 | 必须得到的结果 |
| --- | --- |
| 任意设备请求 FP4 | 首版范围外，参数错误（退出码 2），不推断硬件能力 |
| RX7900 请求 FP8 | hardware unsupported，SKIP |
| Arc A/B 请求 FP8 | hardware unsupported，SKIP |
| RX9070 的 hipBLASLt 没有 matching algorithm | hardware supported，backend kernel_missing 或 configuration_unsupported；不能改报硬件不支持 |
| HIP 本身缺失 | AMD 原生项 runtime_missing；可继续 OpenCL ALU |
| FP16 OpenCL 扩展缺失 | 当前后端 api_unsupported；不据此否定硬件 |
| 未识别的新架构 | unknown，按合法 API probe 进一步判断 |
| 数值校验失败 | failed，拒绝发布该项的有效吞吐 |

## 4 目录与公共接口

建议目录：

```text
CMakeLists.txt
CMakePresets.json
dependencies.lock.json
src/host/
src/common/
workers/cuda/
workers/amd/
workers/intel/
workers/opencl/
kernels/opencl/
schemas/
tests/unit/
tests/integration/
scripts/build-windows.ps1
scripts/package-windows.ps1
scripts/validate-device.ps1
docs/BUILD_WINDOWS.md
docs/METHODOLOGY.md
docs/SUPPORT_MATRIX.md
docs/DECISIONS.md
docs/KNOWN_LIMITATIONS.md
results/
```

schemas 至少包含 protocol-v1.schema.json 和 result-v1.schema.json。公共代码包含配置解析、类型定义、CPU 参考计算、格式解码、计数、统计、JSON/CSV 输出和错误码，不依赖任何 GPU SDK。

设备 ID 优先使用可取得的 LUID/UUID/PCI 标识。显示 nvidia:0、amd:0、intel:0 等友好别名，但同时保存真实身份。不同 API 同名设备不能仅靠名称和枚举序号去重；无法证明身份一致时保留 backend 限定 ID。

默认每次只测一张设备，不并发运行多个 GPU benchmark。多卡 all 模式顺序执行，防止资源争抢改变结果。

结果至少包含以下分组：

| 分组 | 必须保存的字段 |
| --- | --- |
| 构建与环境 | schema_version app_version git_commit OS build compiler backend runtime library driver |
| 设备 | name vendor_id device_id architecture LUID/UUID 若可用 memory_bytes discrete/integrated |
| 测试定义 | suite case_id input_a input_b accumulation output scaling density implementation |
| 配置 | M N K layout repetitions kernel_variant workgroup_size workspace_bytes algorithm_id 若可用 seed cache_policy |
| 微基准计数 | threads wave_count instruction_tile iterations independent_accumulators lanes_per_instruction，每项按 suite 适用性保存 |
| 计时 | timing_source timing_scope sample_times_ms raw_samples median_ms best_ms wall_ms measurement_duration |
| 验证 | validation_status reference_kind sampled_elements error_metrics tolerance_policy native_evidence |
| 状态 | hardware_support backend_status result_status reason_code reason_message |
| 性能 | median_tflops best_tflops；不可用时为 null |

允许扩展字段，schema_version=1 的已定义字段含义不得在不同后端变化。用 UTF-8 输出，终端非 TTY 时禁用进度重绘，JSON 不混入彩色控制符。

## 5 设备内核与各后端实施规范

### 5.1 共用 ALU 设计

- 输入在运行时初始化，使用固定种子和有限非零值。
- 每线程多个独立 accumulator，扫描合理的 accumulator 数量、展开因子和 workgroup 大小。
- 热循环在寄存器中执行；最后一次写回所有必要结果，防止优化消除。
- 采用数值有界的递推，并在完整迭代长度下验证不会溢出、产生 NaN 或导致所有通道无效。
- 不用仅一条依赖链代表峰值吞吐。
- 不能靠 volatile 全局访存或每步写回防 DCE。
- 正式认证检查热循环的指令、寄存器压力和 spill；FP16 不能是 half 转 float 再做 FP32 FMA。
- 调优范围和选中配置进入结果；不要假定一种线程配置适合所有架构。
- 基础正确性使用短迭代与边界输入，性能运行对全迭代结果再做 finite/checksum 检查。

### 5.2 NVIDIA

1. cudaGetDeviceProperties 枚举；首版编入经工具链支持的 SM75、86、89、120 代码。未知架构按现有兼容代码和 probe 处理，不伪装成已验证目标。
2. FP32 使用显式 float FMA；FP16 主变体用 __hfma2。
3. GEMM 统一优先 cuBLASLt；在 warmup/计时前创建 descriptor、矩阵布局、scale、workspace，搜索多个合法候选算法。
4. FP32 case 使用 FP32 A/B/C/D 和 CUBLAS_COMPUTE_32F_PEDANTIC；worker 初始化前显式排除 single-precision emulation，并禁用 TF32。保存有效数学模式。首版没有 TF32 case，因此不需要在同一 worker 内切换全局 TF32 override。
5. FP16 acc32使用FP32 compute/输出；显式acc16使用FP16 compute/输出并独立校验和报告。
6. FP8 主路径 OCP E4M3，显式 scale；遵守固定 SDK 对 GeForce 的 TN 和对齐约束。fast accumulation 等会改变数值特征的开关保持关闭并记录。
8. SM120 使用 GeForce 支持的算法，不构建 SM100a 数据中心内核。
9. 无合法算法时返回配置不可用，不偷偷转为 FP16。
10. 正式计时用同一 CUDA stream 的 start/stop events 覆盖整个 batch；读取前等待 stop 完成。

FP8 输入量化、scale 准备全部在计时外，验证使用实际量化输入与非单位 scale。[S01]

### 5.3 AMD

1. 先通过 HIP runtime 枚举和短 kernel probe；对 RX6000 不承诺 HIP 可用，不能用 gfx103X 包存在代替官方产品支持。[S06]
2. 原生 ALU 用 HIP FMA/half2；若不可用，OpenCL worker 提供独立、明示后端的 ALU 项。
3. GEMM 优先 hipBLASLt。FP32 在该库缺合法 native kernel 时允许使用 rocBLAS SGEMM，并明确 library/algorithm；不能用 TF32 代替。
4. RX7000 的 FP16、RX9000 的 FP16/FP8 逐项探测。消费级 RDNA4 使用 OCP E4M3，不使用 FNUZ。
5. 对 HIP 支持但 hipBLASLt 不能完成目标矩阵 case 的设备，先排查安装的 kernel 数据、布局、workspace、尺寸与版本。
6. 首版必须编译 RDNA3 FP16、RDNA4 FP16/FP8 的 HIP WMMA matrix_micro，不因开发机库覆盖完整或没有 AMD 设备而省略。运行时可以作为库缺项的补缺路径，也可显式独立运行；它有自己的 case_id、计数和校验，不填入 gemm 成绩。
7. WMMA 每代独立 translation unit/code object。不得把 RDNA3 fragment 布局复制到 RDNA4，也不得仅用 gfx11/gfx12 字符串前缀开放所有集显。
8. 直接 WMMA 选择 wave32、多个独立 accumulator、短 dispatch，保存可校验结果。用官方 builtin/ISA 定义检查 wave 级计数。
9. 首批 GPU code object 目标为 gfx1100/1101/1102、gfx1200/1201；只为实际 SDK 支持的目标构建。
10. HIP events 覆盖同一 stream 完整 batch；发生设备丢失结束该 worker。

当前 ROCm Windows 组件可用不代表任意 GPU/shape 的库算法都存在。该区别必须在 capability 和结果中保留。[S05]、[S06]、[S18]

### 5.4 Intel

1. 只枚举 Intel GPU，禁止 SYCL 默认选择器回退 CPU。
2. 记录 GPU backend、driver、aspect::fp16、half_fp_config、device USM、queue profiling 能力。
3. 自写 FP32/FP16 ALU。half 类型存在并不足以认定原生 FMA，发布认证需指令检查。
4. oneMKL 使用 USM GEMM，明确 device USM；不用已弃用的 buffer 接口作为新实现基础。half 输入、float 输出使用官方 mixed precision overload。[S15]
5. FP32 显式 compute_mode::standard，避免用户默认模式改成 TF32/BF16 分解；FP16 A/B=half、C/D=float。
6. 首版无 oneDNN；Arc A/B 的 FP8 请求直接给出原生不支持结果。
7. FP16 GEMM 的 XMX 路径在认证时检查；核显可执行 vector GEMM 时按真实实现标记，不称为 XMX。
8. 自写 ALU 的单 kernel event 可用于设备计时；oneMKL 返回 event 主要用于完成依赖，不假定它的单个 command 时长覆盖库内全部 kernels。
9. GEMM 使用 profiling 开启的 in_order queue，以前后两个具有真实 USM 写入的短 sentinel kernels 包住 batch；首笔 GEMM 显式依赖前标记，后标记依赖最后一笔 GEMM completion。区间为前标记 end 到后标记 start。验证依赖确实覆盖整个运算。
10. 可对固定 SDK 条件启用 profiling_tag 扩展，但不把实验扩展设为唯一计时方法。无法取得可靠设备区间时只返回标注清楚的 wall timing，不能混入 device timing 汇总。
11. 同时保留同步 QPC/steady_clock wall batch 时间用于自检。
12. DPC++、oneMKL、SYCL、Unified Runtime/adapter 必须作为一组验证，不能混搭 ABI 代际。

计时标记的依据和实验扩展边界见 [S12]；oneMKL compute mode 见 [S09]。joint_matrix 是后续可选项，不是首版缺省依赖。

### 5.5 OpenCL 兼容后端

1. 使用 OpenCL 1.2 级基础 API 和 GPU ICD 枚举，排除 CPU/OpenCL 软件设备。
2. FP32 查询 CL_DEVICE_SINGLE_FP_CONFIG；FP16 查询 cl_khr_fp16 与 CL_DEVICE_HALF_FP_CONFIG。
3. native FMA 要求相应 FMA 能力，不能把 CL_FP_SOFT_FLOAT 当原生；再做编译、运行、验证 probe。[S16]
4. FP16 仅在扩展实际报告后启用 pragma 和 half/half2 kernel；保存 build log。
5. 动态编译的源代码与编译选项进入可复现记录。未报告的 NVIDIA FP16 扩展不能强开。
6. 按 scalar half 与 half2 分变体调优，记录实际使用者；发布认证检查机器指令。
7. queue 开启 profiling。batch 设备区间使用首个 kernel start 到最后一个 kernel end，保留依赖并检查队列顺序。
8. 无 OpenCL ICD 时只禁用该 worker，不使整个程序失败。
9. 同一设备已有原生 ALU 结果时，OpenCL 作为可选对照，不覆盖原生后端成绩。
10. RX6000/核显没有可用 GEMM 时明确显示 suite 不可用，不能从 ALU 值推算矩阵 TFLOPS。

## 6 计时 稳定性与正确性

### 6.1 默认工程参数

以下为程序默认策略，不是 GPU 厂商保证值：

| 参数 | 默认值 |
| --- | --- |
| random seed | 20261020 |
| GEMM 默认尺寸 | 1024 2048 4096；8192 由用户显式选择或 extended profile 启用 |
| warmup 总时长目标 | 每个配置至少 500 ms，且已完成 JIT 和库初始化 |
| ALU 单 kernel 校准目标 | 约 5 ms，允许在 1 至 20 ms 范围调节 |
| 单 batch 目标 | 约 100 ms，优先控制在 200 ms 内 |
| 单次 GEMM 调度预算 | 100 ms 的保守软阈值，先从小尺寸校准 |
| 正式 samples | 至少 7 组，累计设备计时至少 1 s |
| 缓存策略 | 默认 reuse；提供 rotate 选项并记录缓冲组数 |
| 总 GPU 分配预算 | 不超过动态可用预算的一半，分配失败时回退更小配置 |
| 默认并发 | 单设备 单执行队列 顺序测试 |
| 取消响应目标 | 正常情况下一个短 batch 后响应 |

对用户明确指定的矩阵形状，不得静默缩小并仍按原尺寸计数。可以 SKIP 并说明预算问题，或者在显式 auto-size 模式下生成新 case 并报告 actual shape。

WDDM 默认 TDR 等待为 2 秒，涉及任务完成或抢占，不是“单 kernel 小于 2 秒就一定安全”。通过短 dispatch 和短 batch 累积数秒样本，不提交单个数秒循环或巨大 graph，不要求关闭 TDR。[S11]

### 6.2 统一计时口径

主指标 timing_scope=device_elapsed_batch：完整 batch 在设备队列上的经过时间，包含队列执行间隙，不宣称为所有计算单元 busy time 之和。

- 包含该 GEMM 实现每次运行必需的内部计算、workspace 使用及辅助 kernels。
- 排除初始化、分配、输入上传、外部量化、算法搜索、正确性 readback、日志和 IPC。
- 不能为了提高成绩将每次 GEMM 必需的计算搬出计时而仍声称相同运算。
- CUDA/HIP events、OpenCL events、SYCL sentinel 都需捕获完整 batch。
- 若只有 host synchronized wall time，timing_source=host_qpc、timing_scope=host_synchronized_batch。单列成绩，不与 device_elapsed_batch 合并排名。
- 记录有效重复次数以及每组原始 elapsed_ms。median_tflops 按同一采样定义计算，best_tflops 单列。
- 校准后固定该配置的 batch 重复次数 R；正式采样期间不混用不同 R 直接取原始时间中位数。若因时长预算需要重新校准，生成新的配置记录。
- 初始化 warmup 与正式 profiling 分离；详细 profiler/verbose dump 只在诊断轮启用。[S13]
- 不使用 all-zero 数据作为默认。缓存 reuse/rotate、随机种子、实际内存预算进入结果。
- rotate 只说明轮换输入缓冲；不知道缓存容量时不能承诺完全冷缓存。
- 调优测量和正式测量分开。每项保存已搜索的配置数量、workspace 上限和最终选择。
- 热稳定性不够或波动过大时标记 noisy，保留样本，不通过删除不利样本伪造稳定结果。

### 6.3 正确性

必须实现 CPU 高精度参考和低精度格式解码，测试数据量控制在验证成本合理的范围。

- 小矩阵完整验证，包括非方阵、合法的非整齐边界与需要 padding 的情况。
- 性能矩阵至少抽样输出元素；参考使用实际量化后输入和实际 scale。
- 单独记录 quantization_error，不与 kernel_error 混为一个阈值。
- FP8 验证 E4M3 解码、舍入和缩放；不要求首版实现未使用的 E5M2/FNUZ。
- 使用预先定义、与 dtype 和 K 相关的绝对/相对/归一化误差规则；把公式和阈值写入 METHODOLOGY.md。
- 同时包含数值构造明确的精确案例和随机案例；不能只用宽松的统一误差阈值掩盖布局错误。
- 若 FP32/FP16/FP8 路径的硬件累加规则导致误差差异，需要给出依据和固定的验证策略，不允许每次失败后临时扩大阈值。
- 验证失败的配置不得进入正式性能结果。保存诊断和可重现 seed。

### 6.4 必须覆盖的故障行为

缺 SDK runtime、驱动不兼容、DLL 版本错配、GPU 不存在、无合法算法、显存不足、kernel build 失败、异步执行错误、设备丢失、用户取消、输出文件写入失败。

失败时保留已完成样本和明确错误。主程序不无限重启失败 worker；同一设备出现设备丢失后结束该设备的测试。

退出码：0 表示至少一项成功且无执行失败，正常硬件 SKIP 可共存；2 参数/配置错误；3 没有可运行测试；4 校验或执行失败；5 --require-all 下有不可用请求；130 用户取消。若多个条件同时出现，按 130、2、4、5、3、0 的顺序判定，并记录详细 summary。

## 7 CLI 合同

以下是必须实现的接口，所有例子都是计划中的命令：

```powershell
gpubench.exe --help
gpubench.exe list
gpubench.exe doctor --json doctor.json
gpubench.exe run --device all --suite alu --dtype fp32,fp16
gpubench.exe run --device nvidia:0 --suite gemm --dtype fp32,fp16,fp8
gpubench.exe run --device amd:0 --suite gemm --dtype fp16,fp8
gpubench.exe run --device intel:0 --suite gemm --dtype fp32,fp16
gpubench.exe run --device amd:0 --suite matrix_micro --dtype fp16,fp8
gpubench.exe run --device all --suite all --sizes 1024,2048,4096 --json result.json --csv result.csv
gpubench.exe run --device nvidia:0 --suite gemm --dtype fp8 --require-all
```

参数要求：

- --backend auto|cuda|amd|intel|opencl；auto 选择该设备的原生后端优先，OpenCL fallback 显式显示。
- --dtype fp32,fp16,fp8 是简写，必须展开为数值合同中的完整 case；fp4 为延后项，当前返回参数错误。
- --suite all 运行所选设备的适用 suite；未实现的可选 suite 不伪装为失败，显式要求某 suite 时给出不可用原因。
- --sizes 为方阵集合；另支持 --m --n --k 指定非方阵，二者互斥。
- 形状参数只控制 GEMM；suite=all 时不改变 ALU 或 micro 的运算定义。只选择 ALU/micro 却传 GEMM 形状参数时给出参数错误。micro 使用报告中的 instruction_tile、wave_count 与 iterations。
- --samples、--duration-ms、--warmup-ms、--seed、--workspace-mib、--cache reuse|rotate、--native-only、--require-all。
- 首版始终 native-only；可以保留该显式参数，不提供模拟计算开关。
- --json 和 --csv 可同时输出；写出失败必须反映在退出状态。
- 测试进行时定期输出阶段和进度，非 TTY 使用可读日志。
- doctor 输出检测到的驱动、编译/运行库版本、实际 DLL 路径、缺失依赖、设备能力和建议安装来源；不修改机器环境。

终端简表至少包含 Device、Suite、Case、Accum、Backend、Timing、Median TFLOP/s、Best TFLOP/s、Validation、Status。不要把 unknown 和 unsupported 显示成 0。

## 8 任务分解与完成条件

### T00 环境与版本锁定

依赖：无。

- [ ] 读取现有仓库说明和 AGENTS.md；若无项目，在指定工作目录创建 gpubench。
- [ ] 检测 Windows 版本、GPU、编译器、CMake、Ninja、SDK、驱动和可用权限。
- [ ] 选定每个后端的兼容版本组合，执行最小编译/链接/初始化 smoke。
- [ ] 生成 dependencies.lock.json、docs/BUILD_WINDOWS.md 和初始 SUPPORT_MATRIX.md。
- [ ] 记录可编译、可运行、待硬件验证三种不同状态。
- [ ] 非 Windows 环境仍可编写源码、运行 host 逻辑测试、准备 Windows CI/脚本；不据此标记 Windows 通过。

完成条件：后续 agent 有明确的构建入口和版本，缺失环境被精确定位；不要求在本任务自动替换系统驱动。

### T01 主程序与公共协议

依赖：T00。

- [ ] 实现 CLI、JSONL worker 协议、超时/取消、Job Object、结构化错误。
- [ ] 定义并验证 protocol/result schemas。
- [ ] 实现公共计数、统计、参考计算接口、JSON/CSV 输出。
- [ ] 用 mock worker 测试缺后端、异常退出、坏消息、取消和已有结果保留。
- [ ] 验证子进程独立日志配置，继承的库 verbose 环境变量不会污染 JSONL。
- [ ] list/doctor 在无 GPU 和无 SDK 的环境仍能正常解释状态。

完成条件：协议冻结为 v1，各后端无需共享编译器或 SDK。mock 仅用于调度测试，永远不产生真实 GPU 分数。

### T02 共用测量与验证逻辑

依赖：T01，可与后端开发并行。

- [ ] 实现短运行校准、warmup、重复采样、workspace/内存预算与缓存策略。
- [ ] 实现 FP32/FP16/OCP E4M3 的参考解码和验证。
- [ ] 测试 FLOP 计数、计时单位、median/best、溢出、null 结果与退出码。
- [ ] 固定各 case 的 validation policy，覆盖非单位 scale 与 layout 错误。
- [ ] 加入完整运行 metadata 和原始样本记录。

完成条件：公共逻辑无需 GPU 也可测试；后端按同一数值与测量合同接入。

### T03 NVIDIA 后端

依赖：T01，集成 T02。

- [ ] CUDA worker 构建、枚举、probe、FP32/FP16 ALU。
- [ ] cuBLASLt FP32 strict、FP16 acc32、FP8 E4M3。
- [ ] 多候选算法搜索及缓存，正确对齐与 scale packing。
- [ ] CUDA 完整 batch 计时与异步错误处理。
- [ ] 至少覆盖一种 RTX40 和一种 RTX50 的 FP8 实机验收；其他代际按获得设备情况记录。

完成条件：目标能力表内每项有实现或明确软件配置限制；缺少对应设备时标为 implemented_not_hardware_validated，不能宣称支持已验收。

### T04 AMD 后端

依赖：T01，集成 T02。

- [ ] HIP worker、FP32/FP16 ALU、FP32/FP16 GEMM。
- [ ] RDNA4 OCP FP8 GEMM。
- [ ] 正确区分硬件支持与 runtime/kernel/API 不可用。
- [ ] 按 5.3 必须实现并编译 RDNA3 FP16、RDNA4 FP16/FP8 的独立 WMMA matrix_micro。
- [ ] HIP runtime 完整而 hipBLASLt/rocBLAS 均不可加载时，枚举、HIP ALU 和已实现的 WMMA 仍可运行，GEMM 准确报缺库。
- [ ] 目标 code objects 分架构构建和检查。
- [ ] 至少一张 RX7000、RX9000 的硬件验证；没有设备时提供验证脚本和待验证记录。

完成条件：RX9000 FP8 不因 hipBLASLt 缺 kernel 被误标为硬件不支持；matrix_micro 与 GEMM 从语义到结果都分开。

### T05 Intel 后端

依赖：T01，集成 T02。

- [ ] SYCL worker，GPU 精确选择，禁止 CPU fallback。
- [ ] FP32/FP16 ALU；oneMKL USM FP32 strict/FP16 GEMM。
- [ ] 完整 batch GPU 区间计时与 wall 对照。
- [ ] FP8 在 Arc A/B 上输出正确 SKIP。
- [ ] 至少一张 Arc A 和一张 Arc B 验证；iGPU 单列 best effort。

完成条件：无 oneDNN 依赖；FP32 无替代计算模式；oneMKL 内部多 kernel 不被漏计。

### T06 OpenCL 兼容后端

依赖：T01，集成 T02。

- [ ] 驱动 ICD 枚举、FP32/FP16 能力检查、JIT kernel 与 build logs。
- [ ] FMA/half2 变体、短 batch profiling、正确性验证。
- [ ] 无 ICD、无 fp16 扩展、软件设备的错误路径。
- [ ] 至少一张 RX6000 验证；可用核显设备单列兼容记录。
- [ ] 验证原生 worker 不可用时，主程序可明确选择 OpenCL ALU 而不伪造 GEMM。

完成条件：兼容包只依赖可取得的运行库/驱动，不要求安装完整 ROCm/oneAPI 开发 SDK。

### T07 集成和性能可信度验证

依赖：T02 至 T06 中相关任务。

- [ ] 验证 1/2/4/8 次 GEMM 的计数和时间随工作量合理变化。
- [ ] 检查自写 ALU 机器码，排除 DCE、精度提升和严重 spill。
- [ ] 在能够取得 profiler/ISA 的机器上确认对应 Tensor/WMMA/XMX 路径，保存证据。
- [ ] 对照官方/独立同条件基线，解释显著性能差距，不强设所有卡必须达到某理论百分比。
- [ ] 主显示 GPU 上完成标准测试、Ctrl+C、缺库、显存不足与 worker crash 场景。
- [ ] 父进程预设 HIPBLASLT_LOG_LEVEL=2 且不设 HIPBLASLT_LOG_FILE 时，AMD worker 的 JSONL 仍完整，诊断日志可独立读取。
- [ ] 检查同一配置多轮噪声、cache 模式、驱动和电源状态记录。
- [ ] 仅测试必要的代表性设备与风险；发现具体风险后再扩大测试范围。

完成条件：所有公布分数均有真实数据、正确性通过、明确计时口径。无实机的后端只完成源码/构建验收，不进入性能认证。

### T08 Windows 发布包

依赖：对应后端通过构建与 smoke。

- [ ] 主 CLI 包和各可选后端包独立。
- [ ] 记录 runtime 再分发文件来源、版本、哈希与必要 kernel 数据。
- [ ] 检查 PE import 和运行时动态加载模块，不只猜 DLL 文件名。
- [ ] 兼容 DLL 位于 worker 私有目录；不复制显卡驱动系统 DLL，不改 System32。
- [ ] 处理 MSVC runtime，禁止把整个 SDK bin 无差别打包。
- [ ] 在没有 Toolkit/VS/Python 的干净 Windows 机器测试，或明确交付为需要指定 runtime 的包。
- [ ] 生成包清单和 SHA256；只创建本地发布产物，不自动上传公共发布站点。

完成条件：包的部署前提可被重复验证；不能仅因开发机能运行就标为便携。

### T09 文档与最终交付

依赖：全部已开展任务。

- [ ] README 含五分钟上手、CLI 例子、典型 SKIP 和安装要求。
- [ ] BUILD_WINDOWS 含每个 worker 的实际构建命令和版本。
- [ ] METHODOLOGY 含三类 suite、精度、FLOPs、计时、scale、缓存和验证规则。
- [ ] SUPPORT_MATRIX 按具体 GPU + OS + driver + runtime 记录 tested/untested/best_effort。
- [ ] KNOWN_LIMITATIONS 区分软件缺项、硬件不支持和缺少实机验证。
- [ ] 提供真实 results/*.json、CSV、日志和 ISA/profiler 证据；无数据项不生成假示例数值。
- [ ] 提供 validate-device.ps1，使持有另一家 GPU 的人可以直接执行同一套验收。
- [ ] 最终报告逐项列出已完成、构建通过、硬件通过、待验证和失败原因。

完成条件：下一位 agent 或用户无需从聊天重建参数、版本或步骤。

## 9 并行分工

T00/T01 由集成 agent 完成并冻结公共协议，随后：

| 工作流 | 负责目录 | 允许并行的任务 |
| --- | --- | --- |
| 集成与方法 | src/host src/common schemas tests scripts docs | T02 T06 调度与集成 |
| NVIDIA | workers/cuda | T03 |
| AMD | workers/amd | T04 |
| Intel | workers/intel | T05 |

共用 schema 或数值合同的变更由集成 agent 统一应用，不让各后端自行修改字段含义。缺少一家的工具链/硬件时继续其他工作；最终明确剩余验证需求，不通过模拟结果“完成”硬件任务。

## 10 构建与验收入口

T01/T08 必须让下列入口实际可用，PowerShell 参数名可保持如下：

```powershell
.\scripts\build-windows.ps1 -Backends host,opencl -Configuration Release
.\scripts\build-windows.ps1 -Backends cuda -Configuration Release
.\scripts\build-windows.ps1 -Backends amd -Configuration Release
.\scripts\build-windows.ps1 -Backends intel -Configuration Release
.\scripts\package-windows.ps1 -Configuration Release
.\scripts\validate-device.ps1 -Device nvidia:0 -Profile standard -OutputDir .\results
.\scripts\validate-device.ps1 -Device amd:0 -Profile standard -OutputDir .\results
.\scripts\validate-device.ps1 -Device intel:0 -Profile standard -OutputDir .\results
```

脚本在环境缺失时输出可操作的诊断和非零退出码，不偷偷切换到其他后端或 Debug 构建。

最终完成检查：

- [ ] Windows 原生 x64；主 CLI 不依赖任一厂商 SDK 启动。
- [ ] 三家目标后端和 OpenCL 兼容路径按合同实现。
- [ ] FP8 只在已确认的原生路径运行；FP4 已延后，不进入首版构建、CLI case 或验收。
- [ ] FP32 无 TF32/BF16 模拟混入；FP16 ALU 无隐式升精度。
- [ ] 所有矩阵结果明确输入、累加、输出、scale 与 dense。
- [ ] 无合法 native 路径时有准确 SKIP，不用零分代替。
- [ ] GPU 时间覆盖完整运算；无仅测 CPU 提交或漏测内部 kernel。
- [ ] 正确性失败不发布有效吞吐。
- [ ] 默认测试不修改 TDR、不超长单 dispatch、可以取消。
- [ ] 支持矩阵只登记实际完成的硬件验收。
- [ ] 发布包具备依赖清单和干净 Windows 验证状态。

## 11 官方依据

资料用于约束实现，不替代实际编译和设备验证。后续执行若采用更高版本，应先检查对应版本官方变更，更新锁文件和验证记录，而不是直接继承旧结论。

[S01]: https://docs.nvidia.com/cuda/archive/12.9.2/cublas/index.html
[S02]: https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html
[S03]: https://docs.nvidia.com/cuda/archive/12.9.2/cuda-installation-guide-microsoft-windows/index.html
[S04]: https://rocm.docs.amd.com/en/latest/reference/precision-support.html
[S05]: https://gpuopen.com/learn/accelerating_generative_ai_on_amd_radeon_gpus/
[S06]: https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html
[S07]: https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md
[S08]: https://uxlfoundation.github.io/oneDNN/v3.13/dev_guide_data_types.html
[S09]: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2026-0/compute-modes.html
[S10]: https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2026.html
[S11]: https://learn.microsoft.com/en-us/windows-hardware/drivers/display/timeout-detection-and-recovery
[S12]: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_profiling_tag.asciidoc
[S13]: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_performance_measurement_methodology_guidelines.html
[S14]: https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/env-variables.html
[S15]: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2026-0/gemm.html
[S16]: https://registry.khronos.org/OpenCL/specs/unified/refpages/man/html/clGetDeviceInfo.html
[S17]: https://docs.nvidia.com/datacenter/tesla/tesla-release-notes-615-71-09/index.html
[S18]: https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/reference/data-type-support.html

- [NVIDIA cuBLAS 类型和计算模式][S01]
- [Blackwell GeForce 工具链发布说明][S02]
- [CUDA Windows 工具链要求][S03]
- [AMD 架构精度支持][S04]
- [RDNA3 和 RDNA4 WMMA 能力及实现说明][S05]
- [ROCm 10.1.0 Windows 平台 产品及组件范围][S06]
- [AMD Windows DLL 搜索与加载约束][S07]
- [Intel GPU 原生精度与转换支持][S08]
- [oneMKL 严格计算模式][S09]
- [DPC++ Windows 系统要求与版本变化][S10]
- [Windows WDDM TDR][S11]
- [SYCL 队列区间计时扩展][S12]
- [GEMM 测量方法参考][S13]
- [hipBLASLt 日志与环境变量][S14]
- [oneMKL USM GEMM 支持类型与接口][S15]
- [OpenCL 设备浮点能力查询][S16]
- [NVIDIA OpenCL FP16 一般性限制 不是消费级驱动安装要求][S17]
- [hipBLASLt 精度和架构组合][S18]

