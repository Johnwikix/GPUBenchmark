#requires -Version 7.2
[CmdletBinding()]
param([string]$Executable=(Join-Path (Split-Path $PSScriptRoot -Parent) 'build/cuda-Release/workers/cuda/sm120_kernel_check.exe'),[string]$OutputDirectory=(Join-Path (Split-Path $PSScriptRoot -Parent) ('results/sm120-kernel-checks-'+(Get-Date -Format 'yyyyMMdd-HHmmss'))),[string]$ComputeSanitizer)
$ErrorActionPreference='Stop'
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$records=[Collections.Generic.List[object]]::new();$failed=$false
foreach($variant in 0..7){foreach($size in @(128,2048)){
 $name="variant-$variant-size-$size"
 & $Executable $variant $size 1> (Join-Path $OutputDirectory ($name+'.stdout.log')) 2> (Join-Path $OutputDirectory ($name+'.stderr.log'))
 $code=$LASTEXITCODE;$records.Add(@{executable=$Executable;arguments=@("$variant","$size");exit_code=$code});if($code -ne 0){$failed=$true}
}}
if($ComputeSanitizer){
 & $ComputeSanitizer --tool memcheck --error-exitcode 9 $Executable 5 128 padded *> (Join-Path $OutputDirectory 'sanitizer.log')
 $code=$LASTEXITCODE;$records.Add(@{executable=$ComputeSanitizer;arguments=@('--tool','memcheck','--error-exitcode','9',$Executable,'5','128','padded');exit_code=$code});if($code -ne 0){$failed=$true}
}
@{commands=$records;passed=(-not $failed);note='Known-answer real GPU correctness tests only; no throughput scores.'} | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $OutputDirectory 'commands.json') -Encoding utf8
if($failed){exit 4};Write-Output 'SM120 known-answer checks passed';exit 0
