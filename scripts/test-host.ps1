#requires -Version 7.2
[CmdletBinding()]
param([string]$Configuration='Release')
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$exe=Join-Path $repo "out/$Configuration/gpubench.exe"
$fixture=Join-Path $repo "build/host-$Configuration/protocol_fixture.exe"
$testroot=Join-Path $repo 'results/host-tests'
New-Item -ItemType Directory -Force $testroot | Out-Null
$records=[System.Collections.Generic.List[object]]::new()
$savedLevel=$env:HIPBLASLT_LOG_LEVEL
$savedFile=$env:HIPBLASLT_LOG_FILE
function Check-Command([string]$Name,[string[]]$Arguments,[int]$Expected){
 & $exe @Arguments 1> (Join-Path $testroot ($Name+'.stdout.log')) 2> (Join-Path $testroot ($Name+'.stderr.log'))
 $actual=$LASTEXITCODE
 $records.Add(@{name=$Name;arguments=$Arguments;expected=$Expected;actual=$actual;passed=($actual -eq $Expected)})
 if($actual -ne $Expected){throw "$Name expected=$Expected actual=$actual"}
}
try{
 Check-Command 'help' @('--help') 0
 Check-Command 'invalid-accumulation' @('run','--suite','gemm','--dtype','fp16','--fp16-accum','fp8') 2
 Check-Command 'acc16-requires-fp16' @('run','--suite','gemm','--dtype','fp8','--fp16-accum','fp16') 2
 Check-Command 'acc16-requires-gemm' @('run','--suite','alu','--dtype','fp16','--fp16-accum','both') 2
 Check-Command 'invalid-cuda-path' @('run','--cuda-gemm','invalid') 2
 Check-Command 'invalid-shape' @('run','--suite','alu','--sizes','128') 2
 Check-Command 'partial-shape' @('run','--suite','gemm','--m','128') 2
 Check-Command 'overflow-shape' @('run','--suite','gemm','--m','18446744073709551615','--n','2','--k','2') 2
 Check-Command 'fp4-deferred' @('run','--dtype','fp4') 2
 Check-Command 'missing-worker' @('run','--backend','cuda','--workers-dir',(Join-Path $testroot 'absent'),'--device','nvidia:0','--suite','alu','--dtype','fp32','--json',(Join-Path $testroot 'missing.json')) 3
 Check-Command 'require-all' @('run','--backend','cuda','--workers-dir',(Join-Path $testroot 'absent'),'--device','nvidia:0','--suite','alu','--dtype','fp32','--require-all','--json',(Join-Path $testroot 'required.json')) 5
 Check-Command 'output-failed' @('doctor','--workers-dir',(Join-Path $testroot 'absent'),'--json',(Join-Path $testroot 'required.json/invalid.json')) 4
 $workers=Join-Path $testroot 'workers'
 $fixtureDir=Join-Path $workers 'cuda'
 New-Item -ItemType Directory -Force $fixtureDir | Out-Null
 Copy-Item -LiteralPath $fixture -Destination (Join-Path $fixtureDir 'cuda_worker.exe') -Force
 $env:GPUBENCH_FIXTURE_MODE='bad'
 Check-Command 'bad-worker' @('doctor','--backend','cuda','--workers-dir',$workers,'--json',(Join-Path $testroot 'bad.json'),'--log-dir',(Join-Path $testroot 'logs')) 0
 $bad=Get-Content -Raw (Join-Path $testroot 'bad.json') | ConvertFrom-Json
 if($bad.backends.cuda.reason_code -ne 'bad_worker_message'){throw 'bad stdout was not classified'}
 $env:GPUBENCH_FIXTURE_MODE='crash'
 Check-Command 'crashed-worker' @('doctor','--backend','cuda','--workers-dir',$workers,'--json',(Join-Path $testroot 'crash.json'),'--log-dir',(Join-Path $testroot 'logs')) 0
 $env:GPUBENCH_FIXTURE_MODE=''
 $env:HIPBLASLT_LOG_LEVEL='2'
 Remove-Item Env:HIPBLASLT_LOG_FILE -ErrorAction SilentlyContinue
 Check-Command 'log-isolation' @('doctor','--backend','cuda','--workers-dir',$workers,'--json',(Join-Path $testroot 'logs.json'),'--log-dir',(Join-Path $testroot 'logs')) 0
 $logs=Get-Content -Raw (Join-Path $testroot 'logs.json') | ConvertFrom-Json
 if($logs.backends.cuda.effective_environment.HIPBLASLT_LOG_LEVEL -ne '0'){throw 'child log isolation failed'}
 if($env:HIPBLASLT_LOG_LEVEL -ne '2'){throw 'parent environment modified'}
 $env:GPUBENCH_FIXTURE_MODE='partial'
 Check-Command 'preserve-after-crash' @('run','--backend','cuda','--workers-dir',$workers,'--device','all','--suite','alu','--dtype','fp32,fp16','--json',(Join-Path $testroot 'partial.json'),'--log-dir',(Join-Path $testroot 'logs')) 4
 $partial=Get-Content -Raw (Join-Path $testroot 'partial.json') | ConvertFrom-Json
 if($partial.results.Count -ne 2 -or $partial.results[0].reason_code -ne 'test_fixture_no_score' -or $partial.results[1].result_status -ne 'failed'){throw 'completed result not preserved after worker crash'}
 $env:GPUBENCH_FIXTURE_MODE=''
 # Fixture never emits GPU scores; direct JSONL test verifies control-thread cancellation.
 $psi=[Diagnostics.ProcessStartInfo]::new($fixture);$psi.UseShellExecute=$false;$psi.RedirectStandardInput=$true;$psi.RedirectStandardOutput=$true;$psi.RedirectStandardError=$true;$psi.CreateNoWindow=$true
 $p=[Diagnostics.Process]::Start($psi);$null=$p.StandardOutput.ReadLine()
 $p.StandardInput.WriteLine('{"protocol_version":1,"request_id":1,"type":"run","payload":{}}')
 Start-Sleep -Milliseconds 30
 $p.StandardInput.WriteLine('{"protocol_version":1,"request_id":2,"type":"cancel","payload":{}}')
 $p.StandardInput.WriteLine('{"protocol_version":1,"request_id":3,"type":"shutdown","payload":{}}')
 $p.StandardInput.Close();$text=$p.StandardOutput.ReadToEnd();$p.WaitForExit(5000)|Out-Null
 $text | Set-Content -Encoding utf8 (Join-Path $testroot 'cancel.protocol.jsonl')
 if($text -notmatch 'cancelled'){throw 'asynchronous worker cancellation failed'}
 $records.Add(@{name='worker-async-cancel';passed=$true})
 $records | ConvertTo-Json -Depth 10 | Set-Content -Encoding utf8 (Join-Path $testroot 'summary.json')
 Write-Output 'host integration checks passed; no GPU scores were generated by fixture'
 exit 0
}catch{$records | ConvertTo-Json -Depth 10 | Set-Content -Encoding utf8 (Join-Path $testroot 'summary.json');Write-Error $_ -ErrorAction Continue;exit 1}
finally{Remove-Item Env:GPUBENCH_FIXTURE_MODE -ErrorAction SilentlyContinue;$env:HIPBLASLT_LOG_LEVEL=$savedLevel;$env:HIPBLASLT_LOG_FILE=$savedFile}
