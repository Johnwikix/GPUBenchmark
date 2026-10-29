#requires -Version 7.2
[CmdletBinding()]
param([ValidateSet('Release','Debug','RelWithDebInfo')][string]$Configuration='Release',[string]$CudaRoot)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$installed=Join-Path $repo "out/$Configuration"
$dist=Join-Path $repo "dist/$Configuration"
try{
 foreach($r in @('gpubench.exe','workers/cuda/cuda_worker.exe','workers/cuda/cublasLt64_13.dll','workers/opencl/opencl_worker.exe')){
  if(-not(Test-Path -LiteralPath (Join-Path $installed $r))){throw "Missing build output: $r. Build host, cuda and opencl backends first."}
 }
 $stage=Join-Path $dist 'gpubench-0.1.0-win-x64-allinone'
 if(Test-Path -LiteralPath $stage){throw "Package staging already exists: $stage. Use a fresh Configuration/output directory; do not silently mix releases."}
 New-Item -ItemType Directory -Force $stage | Out-Null
 Copy-Item -LiteralPath (Join-Path $installed 'gpubench.exe') -Destination $stage
 foreach($backend in @('cuda','opencl')){
  $subdir=Join-Path $stage "workers/$backend"
  New-Item -ItemType Directory -Force $subdir | Out-Null
  Copy-Item -Path (Join-Path $installed "workers/$backend/*") -Destination $subdir -Recurse
 }
 foreach($name in @('VALIDATION.md','evidence.json')){Copy-Item -LiteralPath (Join-Path $repo "workers/cuda/$name") -Destination (Join-Path $stage 'workers/cuda')}
 $license=$null
 if($CudaRoot){foreach($n in @('LICENSE','LICENSE.txt')){if(Test-Path -LiteralPath (Join-Path $CudaRoot $n)){$license=Join-Path $CudaRoot $n}}}
 if(-not $license){$prior=Join-Path $dist 'gpubench-cuda-0.1.0-win-x64/NVIDIA-LICENSE.txt';if(Test-Path -LiteralPath $prior){$license=$prior}}
 if(-not $license){throw 'NVIDIA redistribution license not found; supply -CudaRoot or package the cuda backend first'}
 Copy-Item -LiteralPath $license -Destination (Join-Path $stage 'NVIDIA-LICENSE.txt')
 $readme=@'
# gpubench 0.1.0 便携整合包（Windows x64 免安装）

单目录、免安装、免依赖：静态链接 MSVC 运行时与 CUDA 运行时，
不要求 Visual Studio、VC Redist、CUDA Toolkit、Python 或 PowerShell。
解压到任意目录后直接运行 gpubench.exe，workers\ 目录必须与其保持同级。

## 快速开始

    gpubench.exe list            列出检测到的 GPU
    gpubench.exe doctor          后端与设备诊断（--json doctor.json 可导出）
    gpubench.exe run --device nvidia:0 --backend cuda --suite all --dtype fp32,fp16,fp8 --sizes 1024,2048,4096 --json result.json --csv result.csv

测量过程中控制台保持一行实时进度，逐项完成后打印结果表格，
最后输出 Summary（各精度峰值与退出码）。Ctrl+C 可随时取消。
--json / --csv 是机器可读输出，与控制台表格相互独立。

## 运行要求

- Windows 10/11 x64。
- CUDA 路径（--backend cuda）：NVIDIA Turing（GTX 16 / RTX 20）及之后的
  消费级显卡（含 RTX 30/40/50），驱动 r580 及以上。只需显卡驱动。
- OpenCL 路径（--backend opencl）：任意厂商驱动提供 OpenCL ICD。
- 没有 NVIDIA GPU 的机器程序本身仍可运行：list/doctor 正常工作，
  CUDA 相关项报告 SKIP，不会崩溃。

## 说明

- workers\cuda\cublasLt64_13.dll 是随包分发的 NVIDIA 私有运行时，
  许可见 NVIDIA-LICENSE.txt；不要单独分发该 DLL。
- 程序不修改驱动、TDR、电源计划、频率或功耗限制。
- FP8 仅原生 OCP E4M3；首版不支持 FP4。
- 退出码：0 至少一项成功且无执行失败；2 参数错误；3 无可运行项；
  4 执行/校验/写出失败；5 --require-all 存在不可用项；130 已取消。
'@
 Set-Content -Encoding utf8 (Join-Path $stage 'README.md') $readme
 $vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
 $vs=& $vswhere -latest -products '*' -property installationPath | Select-Object -First 1
 $tools=Get-ChildItem -Directory -LiteralPath (Join-Path $vs 'VC/Tools/MSVC') | Sort-Object Name -Descending | Select-Object -First 1
 $dumpbin=Join-Path $tools.FullName 'bin/Hostx64/x64/dumpbin.exe'
 if(-not(Test-Path -LiteralPath $dumpbin)){throw 'dumpbin required to audit actual PE imports'}
 $imports=Join-Path $stage 'pe-imports'
 New-Item -ItemType Directory -Force $imports | Out-Null
 foreach($binary in Get-ChildItem -Recurse -File -LiteralPath $stage | Where-Object Extension -In @('.exe','.dll')){
  & $dumpbin /headers /dependents $binary.FullName > (Join-Path $imports ($binary.Name+'.txt'))
  if($LASTEXITCODE -ne 0){throw "PE audit failed: $($binary.FullName)"}
 }
 $manifest=[ordered]@{schema_version=1;app_version='0.1.0';backend='allinone';configuration=$Configuration;platform='Windows x64';created_utc=[DateTime]::UtcNow.ToString('o');clean_windows_verified=$false;deployment='Self-contained single directory: static MSVC runtime, static cudart, private locked cuBLASLt bundled. CUDA path requires an NVIDIA display driver r580+ (Turing or newer consumer GPU); OpenCL path requires a vendor OpenCL ICD. No toolkit, VC redist, Python or PowerShell required';files=@()}
 foreach($file in Get-ChildItem -Recurse -File -LiteralPath $stage){
  $relative=[IO.Path]::GetRelativePath($stage,$file.FullName)
  $provenance=if($file.Name -eq 'cublasLt64_13.dll'){'NVIDIA CUDA 13.2 toolkit (locked runtime)'}else{'gpubench source/build'}
  $manifest.files+=@{path=$relative;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant();bytes=$file.Length;source=$provenance;version=$file.VersionInfo.FileVersion}
 }
 $manifest | ConvertTo-Json -Depth 10 | Set-Content -Encoding utf8 (Join-Path $stage 'manifest.json')
 $zip=$stage+'.zip'
 Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -CompressionLevel Optimal
 $hashFile=Join-Path $dist 'SHA256.json'
 $release=@()
 if(Test-Path -LiteralPath $hashFile){$release=@(Get-Content -Raw $hashFile | ConvertFrom-Json)}
 $release=@($release)+[pscustomobject]@{path=[IO.Path]::GetFileName($zip);sha256=(Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant();bytes=(Get-Item -LiteralPath $zip).Length}
 $release | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 $hashFile
 Write-Output "All-in-one package: $zip (clean-machine verification pending)"
 exit 0
}catch{Write-Error $_ -ErrorAction Continue;exit 1}
