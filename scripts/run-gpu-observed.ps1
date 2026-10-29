#requires -Version 7.2
[CmdletBinding()]
param([Parameter(Mandatory)][string]$Name,[Parameter(Mandatory)][string[]]$BenchmarkArguments,[string]$Executable=(Join-Path (Split-Path $PSScriptRoot -Parent) 'out/Release/gpubench.exe'),[string]$OutputRoot=(Join-Path (Split-Path $PSScriptRoot -Parent) 'results/sm120-acc16'),[int]$GpuIndex=0)
$ErrorActionPreference='Stop'
$dir=Join-Path ([IO.Path]::GetFullPath($OutputRoot)) $Name
if(Test-Path $dir){throw "Evidence directory already exists: $dir"}
New-Item -ItemType Directory -Path $dir -Force | Out-Null
$Executable=[IO.Path]::GetFullPath($Executable)
$args2=$BenchmarkArguments+@('--json',(Join-Path $dir 'result.json'),'--csv',(Join-Path $dir 'result.csv'),'--log-dir',(Join-Path $dir 'workers'))
$telemetry=$null
try{
 $smi=(Get-Command nvidia-smi).Source
 $fields='timestamp,uuid,utilization.gpu,utilization.memory,clocks.sm,power.draw,power.limit,temperature.gpu,pstate,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown'
 $telemetry=Start-Process -FilePath $smi -ArgumentList @("--id=$GpuIndex","--query-gpu=$fields",'--format=csv','--loop-ms=100') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $dir 'gpu-telemetry.csv') -RedirectStandardError (Join-Path $dir 'gpu-telemetry.stderr.log')
 $start=[DateTimeOffset]::UtcNow
 $psi=[Diagnostics.ProcessStartInfo]::new($Executable);$psi.UseShellExecute=$false;$psi.CreateNoWindow=$true;$psi.RedirectStandardOutput=$true;$psi.RedirectStandardError=$true
 foreach($arg in $args2){$psi.ArgumentList.Add($arg)}
 $process=[Diagnostics.Process]::Start($psi);$stdout=$process.StandardOutput.ReadToEndAsync();$stderr=$process.StandardError.ReadToEndAsync()
 $process.WaitForExit();$code=$process.ExitCode;$end=[DateTimeOffset]::UtcNow
 $stdout.GetAwaiter().GetResult() | Set-Content (Join-Path $dir 'stdout.log') -Encoding utf8
 $stderr.GetAwaiter().GetResult() | Set-Content (Join-Path $dir 'stderr.log') -Encoding utf8
 @{executable=$Executable;arguments=$args2;exit_code=$code;start_utc=$start.ToString('o');end_utc=$end.ToString('o');start_unix_ms=$start.ToUnixTimeMilliseconds();end_unix_ms=$end.ToUnixTimeMilliseconds();timezone=[TimeZoneInfo]::Local.Id;utc_offset_minutes=[TimeZoneInfo]::Local.GetUtcOffset([DateTime]::Now).TotalMinutes;telemetry_command=@($smi,"--id=$GpuIndex","--query-gpu=$fields",'--format=csv','--loop-ms=100');host_sha256=(Get-FileHash $Executable).Hash;worker_sha256=(Get-FileHash (Join-Path (Split-Path $Executable -Parent) 'workers/cuda/cuda_worker.exe')).Hash} | ConvertTo-Json -Depth 8 | Set-Content (Join-Path $dir 'command.json') -Encoding utf8
 Write-Output "$Name exit=$code"
 if(Test-Path (Join-Path $dir 'result.json')){(Get-Content (Join-Path $dir 'result.json') -Raw|ConvertFrom-Json).results | Select-Object case_id,median_tflops,validation_status,result_status,reason_message | Format-Table -AutoSize}
 exit $code
}finally{if($telemetry -and -not $telemetry.HasExited){$telemetry.Kill();$telemetry.WaitForExit()}}
