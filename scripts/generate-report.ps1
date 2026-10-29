#requires -Version 7.2
[CmdletBinding()]
param([Parameter(Mandatory)][string]$EvidenceDirectory,[string]$ReportPath=(Join-Path (Split-Path $PSScriptRoot -Parent) 'results/RTX5070Ti_REPORT.html'))
$ErrorActionPreference='Stop'
function HtmlEncode($Value){return [Net.WebUtility]::HtmlEncode([string]$Value)}
function F($Value){if($null -eq $Value){return '—'};return ([double]$Value).ToString('F3',[Globalization.CultureInfo]::InvariantCulture)}
try{
 $docs=@();$rows=@();$valid=@();$commands=Get-Content -Raw -LiteralPath (Join-Path $EvidenceDirectory 'commands.json')|ConvertFrom-Json
 foreach($name in @('alu','gemm','non-square')){$path=Join-Path $EvidenceDirectory ($name+'.json');$doc=Get-Content -Raw -LiteralPath $path|ConvertFrom-Json;$docs+=@{name=$name;document=$doc;path=$path};foreach($r in $doc.results){$rows+=@{source=$name;result=$r};if($r.result_status -eq 'ok' -and $r.validation_status -eq 'passed'){$valid+=$r}}}
 if(-not $valid.Count){throw 'No validated real hardware results: refusing to generate a performance report'}
 $first=$valid[0];$device=$first.device;$build=$first.build
 if($device.name -notmatch 'RTX 5070 Ti' -or @($valid|Where-Object backend -ne 'cuda').Count){throw 'This report template is specifically for RTX5070Ti CUDA evidence; refusing mislabeled device/backend claims'}
 $telemetry=Import-Csv -LiteralPath (Join-Path $EvidenceDirectory 'telemetry-before.csv')|Select-Object -First 1
 $driver=($telemetry.PSObject.Properties|Where-Object {$_.Name.Trim() -eq 'driver_version'}).Value.Trim()
 $body=[Text.StringBuilder]::new()
 [void]$body.AppendLine('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RTX 5070 Ti gpubench 实测报告</title><style>body{font:16px/1.6 system-ui,sans-serif;color:#172532;background:#fafbfc;max-width:1200px;margin:36px auto;padding:0 24px}h1{font-size:30px}h2{font-size:21px;margin-top:32px}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{border:1px solid #d9dfe5;padding:8px 10px;text-align:left}th{background:#edf2f6}code,pre{font:13px/1.5 Consolas,monospace}pre{white-space:pre-wrap;background:#edf2f6;padding:14px;overflow-wrap:anywhere}.note{border-left:4px solid #236a93;padding:8px 16px;background:#eff7fb}.num{text-align:right;font-variant-numeric:tabular-nums}a{color:#155b96}footer{color:#506170;font-size:13px;margin-top:36px}</style>')
 [void]$body.AppendLine('<h1>RTX 5070 Ti：FP32 / FP16 / FP8 实测</h1>')
 [void]$body.AppendLine('<p class="note">本报告只引用本机真实执行且校验通过的结果。ALU 普通 FMA 与 GEMM 矩阵运算分别比较。FP32 为严格 FP32，FP16 GEMM 和 FP8 E4M3 GEMM 均使用 FP32 累加/输出；FP4 已按用户要求延后。数据不代表其他型号或驱动已认证。</p>')
 [void]$body.AppendLine('<h2>环境与版本</h2><table><tbody>')
 $envRows=[ordered]@{'GPU'=$device.name;'Identity'=($device.identity|ConvertTo-Json -Compress);'Architecture'=$device.architecture;'Driver'=$driver;'OS'=($build.os|ConvertTo-Json -Compress);'Compiler'=$build.compiler;'Runtime'=($build.runtime|ConvertTo-Json -Depth 5 -Compress);'Evidence directory'=$EvidenceDirectory;'Generated UTC'=[DateTime]::UtcNow.ToString('o')}
 foreach($key in $envRows.Keys){[void]$body.AppendLine('<tr><th>'+(HtmlEncode $key)+'</th><td>'+(HtmlEncode $envRows[$key])+'</td></tr>')}
 [void]$body.AppendLine('</tbody></table><h2>正式性能样本</h2><p>单位 TFLOP/s。Median 为固定 batch 重复次数下全部有效原始 elapsed_ms 的中位数，Best 为最短样本单列；不删除慢样本。每配置至少 7 组、累计设备计时至少 1 秒，warmup 至少 500ms。时间来自同一 CUDA stream 的完整 batch events，保留同步 wall time。校验失败、不可用或取消时吞吐为 null。</p>')
 [void]$body.AppendLine('<table><thead><tr><th>Suite / Case</th><th>Logical shape</th><th>Accum</th><th>Median</th><th>Best</th><th>Samples</th><th>Device ms</th><th>Noisy</th><th>Validation / Status</th></tr></thead><tbody>')
 foreach($entry in $rows | Where-Object source -ne 'non-square'){$r=$entry.result;$shape=if($r.suite -eq 'gemm'){"$($r.configuration.m) × $($r.configuration.n) × $($r.configuration.k)"}else{"$($r.configuration.threads) threads; $($r.configuration.independent_accumulators) accum; $($r.configuration.lanes_per_instruction) lanes"};[void]$body.AppendLine('<tr><td>'+(HtmlEncode $r.case_id)+'</td><td>'+(HtmlEncode $shape)+'</td><td>'+(HtmlEncode $r.accumulation)+'</td><td class="num">'+(F $r.median_tflops)+'</td><td class="num">'+(F $r.best_tflops)+'</td><td>'+$r.raw_samples.Count+'</td><td>'+(F $r.measurement_duration)+'</td><td>'+(HtmlEncode $r.noisy)+'</td><td>'+(HtmlEncode ($r.validation_status+' / '+$r.result_status))+'</td></tr>')}
 [void]$body.AppendLine('</tbody></table><h2>正确性与可信度</h2><ul><li>GEMM 先执行有正负值的精确构造和随机非方阵小案例完整校验，再抽样性能矩阵；reference 使用真实量化输入与实际 scale。所有输出检查 finite，量化误差与 kernel error 分开保存。</li><li>FP8 scale A=0.5、B=1.5（乘积 0.75）；同一固定 tolerance 必须拒绝故意忽略 scale 的 reference。不会用两个 scale 互相抵消的例子证明缩放正确。</li><li>FP32 使用 CUBLAS_COMPUTE_32F_PEDANTIC，禁 TF32 和单精度模拟；FP8 fast accumulation 关闭。算法搜索、初始化、量化和 readback 在正式计时外。</li><li>SM120 的 FFMA / HFMA2 反汇编及寄存器、spill 构建记录已保存。cuBLASLt 的实际 API 配置与算法 ID 已保存；完整 Tensor profiler 认证和同条件外部基线仍未完成。</li><li>补充证据包含 1/2/4/8 次 GEMM batch 计数/时间、reuse/rotate、奇数非方阵 padding、缺 BLAS 时 ALU 可运行及超显存预算在分配前 SKIP。噪声标记保留，不换用最佳轮次冒充中位数。</li><li>OpenCL 兼容 FP32 独立保存；NVIDIA 驱动未暴露 cl_khr_fp16，OpenCL FP16 为 API SKIP，CUDA FP16 实测照常有效。AMD/Intel 仅源码实现，缺 SDK/GPU，未声称构建或硬件通过。</li></ul>')
 [void]$body.AppendLine('<h2>本轮命令与退出码</h2><p>工作目录为项目根目录。完整可复现参数和时间见 commands.json，stdout/stderr 和 worker JSONL 位于同一证据目录。</p>')
 foreach($cmd in $commands){[void]$body.AppendLine('<pre>'+(HtmlEncode ($cmd.executable+' '+(($cmd.arguments|ForEach-Object{'"'+$_+'"'}) -join ' ')))+"`nexit="+$cmd.exit_code+'</pre>')}
 [void]$body.AppendLine('<h2>原始样本</h2>')
 foreach($entry in $rows){$r=$entry.result;[void]$body.AppendLine('<details><summary>'+(HtmlEncode ($entry.source+' / '+$r.case_id+' / '+($r.configuration.logical_shape|ConvertTo-Json -Compress)))+'</summary><pre>'+(HtmlEncode ($r.raw_samples|ConvertTo-Json -Depth 10))+'</pre></details>')}
 [void]$body.AppendLine('<h2>证据文件与限制</h2><p>该报告与 JSON/CSV、原始样本、commands.json、worker 日志、PE imports、SHA256 清单一起交付。原生 Windows 构建已完成；无 VS/Toolkit/Python 的干净机器部署、RTX40 和 AMD/Intel 真机仍待验证。此机器是 Windows build26300、RTX5070Ti/617.14，不能外推为所有 Windows/GPU 组合的支持承诺。</p><footer>gpubench 0.1.0 · Plan v1.1 · 原始数据优先；报告仅作可读呈现。未修改驱动、TDR、功耗或频率。</footer></html>')
 $parent=Split-Path $ReportPath -Parent;if($parent){New-Item -ItemType Directory -Force $parent|Out-Null}
 $body.ToString()|Set-Content -Encoding utf8 -LiteralPath $ReportPath
 $summary=@{report=$ReportPath;evidence=$EvidenceDirectory;validated_results=$valid.Count;device=$device;created_utc=[DateTime]::UtcNow.ToString('o');results=@($rows|ForEach-Object{@{source=$_.source;case_id=$_.result.case_id;configuration=$_.result.configuration;median_tflops=$_.result.median_tflops;best_tflops=$_.result.best_tflops;status=$_.result.result_status;validation=$_.result.validation_status}})}
 $summary|ConvertTo-Json -Depth 40|Set-Content -Encoding utf8 -LiteralPath ([IO.Path]::ChangeExtension($ReportPath,'.json'))
 Write-Output "Report: $ReportPath"
 exit 0
}catch{Write-Error $_ -ErrorAction Continue;exit 1}
