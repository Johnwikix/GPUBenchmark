[CmdletBinding()]
param(
 [string]$Executable,
 [string]$Device='amd:0',
 [string]$OutputDir,
 [switch]$Extended
)
$ErrorActionPreference='Stop'
$repo=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if(-not $Executable){$Executable=Join-Path $repo 'out/Release/gpubench.exe'}
if(-not $OutputDir){$OutputDir=Join-Path $repo ('results/amd-'+[DateTime]::UtcNow.ToString('yyyyMMdd-HHmmss'))}
$Executable=[IO.Path]::GetFullPath($Executable)
if(-not(Test-Path -LiteralPath $Executable -PathType Leaf)){throw "gpubench missing: $Executable; run scripts/build-windows.ps1 -Backends host,amd -Configuration Release -HipRoot <SDK>."}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$OutputDir=(Resolve-Path -LiteralPath $OutputDir).Path
$records=[Collections.Generic.List[object]]::new()
function Invoke-Evidence([string]$Name,[string[]]$Arguments){
 $stdout=Join-Path $OutputDir ($Name+'.stdout.txt')
 $stderr=Join-Path $OutputDir ($Name+'.stderr.txt')
 $start=[DateTime]::UtcNow
 & $Executable @Arguments 1> $stdout 2> $stderr
 $code=$LASTEXITCODE
 $records.Add([ordered]@{name=$Name;executable=$Executable;arguments=$Arguments;started_utc=$start.ToString('o');finished_utc=[DateTime]::UtcNow.ToString('o');exit_code=$code;stdout=$stdout;stderr=$stderr})
 Write-Host "$Name exit=$code; evidence=$OutputDir"
}
Invoke-Evidence -Name 'doctor' -Arguments @('doctor','--json',(Join-Path $OutputDir 'doctor.json'))
Invoke-Evidence -Name 'list' -Arguments @('list')
$cases=@(
 @{name='alu';args=@('--suite','alu','--dtype','fp32,fp16')},
 @{name='small-full-gemm';args=@('--suite','gemm','--dtype','fp32,fp16,fp8','--sizes','64')},
 @{name='rectangular-boundary';args=@('--suite','gemm','--dtype','fp32,fp16','--m','31','--n','37','--k','29')},
 @{name='matrix-micro';args=@('--suite','matrix_micro','--dtype','fp16,fp8')},
 @{name='rotate-gemm';args=@('--suite','gemm','--dtype','fp32,fp16,fp8','--sizes','64','--cache','rotate')},
 @{name='standard-gemm';args=@('--suite','gemm','--dtype','fp32,fp16,fp8','--sizes','1024,2048,4096')}
)
if($Extended){$cases+=@{name='extended-gemm';args=@('--suite','gemm','--dtype','fp32,fp16,fp8','--sizes','8192')}}
foreach($case in $cases){
 $arguments=@('run','--backend','amd','--device',$Device)+$case.args+@('--samples','7','--duration-ms','1000','--warmup-ms','500','--seed','20261020','--json',(Join-Path $OutputDir ($case.name+'.json')),'--csv',(Join-Path $OutputDir ($case.name+'.csv')))
 Invoke-Evidence -Name $case.name -Arguments $arguments
}
$summary=[ordered]@{
 schema_version=1;device=$Device;executable=$Executable;executable_sha256=(Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash.ToLowerInvariant();records=$records;
 note='This script records real commands, exit codes, samples and numerical validation. Inspect each result: unsupported SKIP is not a numerical PASS. Disassembly/register/spill review, optional-DLL absence, driver-loss/cancel and clean Windows deployment remain separate acceptance checks.'
}
$summary|ConvertTo-Json -Depth 12|Set-Content -LiteralPath (Join-Path $OutputDir 'verification-summary.json') -Encoding utf8
if(@($records|Where-Object {$_.exit_code -eq 130}).Count){exit 130}
if(@($records|Where-Object {$_.exit_code -eq 2}).Count){exit 2}
if(@($records|Where-Object {$_.exit_code -eq 4}).Count){exit 4}
if(@($records|Where-Object {$_.exit_code -notin @(0,3,5)}).Count){exit 1}
if(-not @($records|Where-Object {$_.name -eq 'alu' -and $_.exit_code -eq 0}).Count){exit 3}
exit 0
