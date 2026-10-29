#requires -Version 7.2
[CmdletBinding()]
param([Parameter(Mandatory)][string]$Destination)
$ErrorActionPreference='Stop'
$commit='1810164f27f2c7a499fcfc86681392662375327d'
if(-not(Test-Path -LiteralPath $Destination)){
 & git clone --depth 1 --branch v4.3.4 https://github.com/NVIDIA/cutlass.git $Destination
 if($LASTEXITCODE -ne 0){throw 'CUTLASS clone failed'}
}
$actual=& git -C $Destination rev-parse HEAD
if($LASTEXITCODE -ne 0 -or $actual -ne $commit){throw "Expected CUTLASS4.3.4 commit $commit; found $actual"}
& git -C $Destination diff HEAD --exit-code -- include tools/util/include
if($LASTEXITCODE -ne 0){throw 'Pinned CUTLASS headers have local changes'}
Write-Output "Verified CUTLASS4.3.4 $commit at $Destination"
