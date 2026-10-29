#requires -Version 7.2
[CmdletBinding()]
param([Parameter(Mandatory)][string]$Device,[ValidateSet('standard','extended')][string]$Profile='standard',[string]$OutputDir=(Join-Path (Split-Path $PSScriptRoot -Parent) 'results'),[string]$Executable=(Join-Path (Split-Path $PSScriptRoot -Parent) 'out/Release/gpubench.exe'),[ValidateSet('auto','cuda','amd','intel','opencl')][string]$Backend='auto',[ValidateSet('reuse','rotate')][string]$Cache='reuse')
$ErrorActionPreference='Stop'
if(-not(Test-Path -LiteralPath $Executable)){Write-Error 'gpubench.exe not found. Build host and optional backend first, or supply -Executable.';exit 1}
$stamp=Get-Date -Format 'yyyyMMdd-HHmmss'
$directory=Join-Path $OutputDir ($Device.Replace(':','-')+'-'+$Profile+'-'+$Cache+'-'+$stamp)
New-Item -ItemType Directory -Force -Path $directory | Out-Null
$Executable=[IO.Path]::GetFullPath($Executable)
$commands=[System.Collections.Generic.List[object]]::new()
$environment=[ordered]@{time_utc=[DateTime]::UtcNow.ToString('o');os=[Environment]::OSVersion.VersionString;architecture=[Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString();powershell=$PSVersionTable.PSVersion.ToString();executable=$Executable;executable_sha256=(Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash;profile=$Profile;cache=$Cache;device=$Device;backend=$Backend}
try{$environment.os_details=Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber; $environment.adapters=Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID,DriverVersion}catch{$environment.inventory_error=$_.Exception.Message}
$environment | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 (Join-Path $directory 'environment.json')
function Invoke-Recorded([string]$Name,[string[]]$Arguments){
 $start=[DateTime]::UtcNow
 & $Executable @Arguments 1> (Join-Path $directory ($Name+'.stdout.log')) 2> (Join-Path $directory ($Name+'.stderr.log'))
 $code=$LASTEXITCODE
 $commands.Add([ordered]@{name=$Name;executable=$Executable;arguments=$Arguments;start_utc=$start.ToString('o');end_utc=[DateTime]::UtcNow.ToString('o');exit_code=$code})
 $commands | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 (Join-Path $directory 'commands.json')
 Write-Output "$Name exit=$code"
 return $code
}
function Save-Telemetry([string]$Name){$smi=Get-Command nvidia-smi -ErrorAction SilentlyContinue;if($smi){& $smi.Source --query-gpu=timestamp,name,uuid,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.gr,clocks.mem,memory.used,memory.total --format=csv 1> (Join-Path $directory ($Name+'.csv')) 2> (Join-Path $directory ($Name+'.stderr.log'))}}
try{
 Save-Telemetry 'telemetry-before'
 $common=@('--backend',$Backend,'--device',$Device,'--log-dir',(Join-Path $directory 'workers'))
 $null=Invoke-Recorded 'doctor' @('doctor','--backend',$Backend,'--json',(Join-Path $directory 'doctor.json'),'--log-dir',(Join-Path $directory 'workers'))
 $null=Invoke-Recorded 'list' @('list','--backend',$Backend,'--log-dir',(Join-Path $directory 'workers'))
 $sizes=if($Profile -eq 'extended'){'1024,2048,4096,8192'}else{'1024,2048,4096'}
 $base=@('--samples','7','--duration-ms','1000','--warmup-ms','500','--seed','20261020','--cache',$Cache,'--native-only')
 $null=Invoke-Recorded 'alu' (@('run')+$common+@('--suite','alu','--dtype','fp32,fp16')+$base+@('--json',(Join-Path $directory 'alu.json'),'--csv',(Join-Path $directory 'alu.csv')))
 $null=Invoke-Recorded 'gemm' (@('run')+$common+@('--suite','gemm','--dtype','fp32,fp16,fp8','--sizes',$sizes)+$base+@('--json',(Join-Path $directory 'gemm.json'),'--csv',(Join-Path $directory 'gemm.csv')))
 $null=Invoke-Recorded 'non-square' (@('run')+$common+@('--suite','gemm','--dtype','fp32,fp16','--m','128','--n','96','--k','80')+$base+@('--json',(Join-Path $directory 'non-square.json'),'--csv',(Join-Path $directory 'non-square.csv')))
 if($Device.StartsWith('amd:') -or $Backend -eq 'amd'){$null=Invoke-Recorded 'matrix-micro' (@('run')+$common+@('--suite','matrix_micro','--dtype','fp16,fp8')+$base+@('--json',(Join-Path $directory 'matrix-micro.json'),'--csv',(Join-Path $directory 'matrix-micro.csv')))}
 Save-Telemetry 'telemetry-after'
 $bad=@($commands | Where-Object {$_.exit_code -notin @(0,3,5)})
 $ok=@($commands | Where-Object {$_.name -eq 'alu' -or $_.name -eq 'gemm' -or $_.name -eq 'matrix-micro'} | Where-Object exit_code -eq 0)
 $exitCode=if($bad.Count){4}elseif(-not $ok.Count){3}else{0}
 [ordered]@{exit_code=$exitCode;directory=$directory;commands=$commands;note='Only result_status=ok with validation_status=passed is a valid performance result. SKIP does not certify hardware. ISA/profiler and clean-machine deployment remain separate checks.'} | ConvertTo-Json -Depth 10 | Set-Content -Encoding utf8 (Join-Path $directory 'validation-summary.json')
 Get-ChildItem -Recurse -File -LiteralPath $directory | ForEach-Object {[ordered]@{path=[IO.Path]::GetRelativePath($directory,$_.FullName);sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash;bytes=$_.Length}} | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 (Join-Path $directory 'SHA256.json')
 Write-Output "Evidence: $directory"
 exit $exitCode
}catch{$_ | Out-String | Set-Content -Encoding utf8 (Join-Path $directory 'validation-error.log');Write-Error $_ -ErrorAction Continue;exit 4}
