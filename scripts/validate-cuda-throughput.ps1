#requires -Version 7.2
[CmdletBinding()]
param([string]$Executable=(Join-Path (Split-Path $PSScriptRoot -Parent) 'out/Release/gpubench.exe'),[string]$Device='nvidia:0',[int[]]$Sizes=@(4096,8192),[string]$OutputDirectory=(Join-Path (Split-Path $PSScriptRoot -Parent) ('results/cuda-throughput-'+(Get-Date -Format 'yyyyMMdd-HHmmss'))))
$ErrorActionPreference='Stop'
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$Executable=[IO.Path]::GetFullPath($Executable)
$OutputDirectory=[IO.Path]::GetFullPath($OutputDirectory)
$records=[Collections.Generic.List[object]]::new()
function Run-Recorded([string]$Name,[string[]]$Arguments){
 $path=Join-Path $OutputDirectory ($Name+'.json')
 $args2=$Arguments+@('--json',$path,'--csv',(Join-Path $OutputDirectory ($Name+'.csv')),'--log-dir',(Join-Path $OutputDirectory 'workers'))
 $start=[DateTime]::UtcNow.ToString('o')
 & $Executable @args2 1> (Join-Path $OutputDirectory ($Name+'.stdout.log')) 2> (Join-Path $OutputDirectory ($Name+'.stderr.log'))
 $code=$LASTEXITCODE
 $records.Add(@{executable=$Executable;arguments=$args2;exit_code=$code;start_utc=$start;end_utc=[DateTime]::UtcNow.ToString('o')})
 $records | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $OutputDirectory 'commands.json') -Encoding utf8
 if($code -ne 0){throw "$Name exit=$code; inspect retained stderr/results"}
 $doc=Get-Content $path -Raw | ConvertFrom-Json
 foreach($r in $doc.results){
  if($r.result_status -ne 'ok' -or $r.validation_status -ne 'passed'){throw "$Name did not validate"}
  if($r.suite -eq 'gemm'){
   if($r.accumulation -ne 'fp32' -or $r.output -ne 'fp32' -or $r.density -ne 'dense' -or $r.configuration.fast_accum){throw 'Precision contract mismatch'}
   $shape=$r.configuration.logical_shape
   foreach($s in $r.raw_samples){$expected=[decimal]2*$shape.m*$shape.n*$shape.k*$s.repetitions;if([decimal]$s.flops -ne $expected){throw 'Independent 2MNKR audit failed'}}
  }
  Write-Output "$Name $($r.case_id) median=$($r.median_tflops) TFLOP/s validated"
 }
}
try{
 $worker=Join-Path (Split-Path $Executable -Parent) 'workers/cuda/cuda_worker.exe'
 $dll=Join-Path (Split-Path $worker -Parent) 'cublasLt64_13.dll'
 @{utc=[DateTime]::UtcNow.ToString('o');os=[Environment]::OSVersion.VersionString;executable_sha256=(Get-FileHash $Executable).Hash;worker_sha256=(Get-FileHash $worker).Hash;library_sha256=(Get-FileHash $dll).Hash;library_version=(Get-Item $dll).VersionInfo.FileVersion} | ConvertTo-Json | Set-Content (Join-Path $OutputDirectory 'environment.json') -Encoding utf8
 & nvidia-smi -q 1> (Join-Path $OutputDirectory 'gpu-before.txt') 2> (Join-Path $OutputDirectory 'gpu-before.stderr.log')
 $common=@('run','--backend','cuda','--device',$Device,'--native-only','--samples','7','--duration-ms','1000','--warmup-ms','500','--seed','20261020','--cache','reuse','--workspace-mib','256')
 Run-Recorded 'non-square' ($common+@('--suite','gemm','--dtype','fp16,fp8','--m','31','--n','19','--k','37'))
 foreach($size in $Sizes){Run-Recorded "gemm-$size" ($common+@('--suite','gemm','--dtype','fp16,fp8','--sizes',"$size"))}
 Run-Recorded 'alu-fp32-regression' ($common+@('--suite','alu','--dtype','fp32'))
 & nvidia-smi -q 1> (Join-Path $OutputDirectory 'gpu-after.txt') 2> (Join-Path $OutputDirectory 'gpu-after.stderr.log')
 @{status='passed';independent_flops_audit='2*M*N*K*repetitions';commands=$records.Count} | ConvertTo-Json | Set-Content (Join-Path $OutputDirectory 'summary.json') -Encoding utf8
 Get-ChildItem $OutputDirectory -Recurse -File | ForEach-Object {@{path=[IO.Path]::GetRelativePath($OutputDirectory,$_.FullName);sha256=(Get-FileHash $_.FullName).Hash}} | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $OutputDirectory 'SHA256.json') -Encoding utf8
 Write-Output "Evidence: $OutputDirectory"
 exit 0
}catch{Write-Error $_ -ErrorAction Continue;exit 4}
