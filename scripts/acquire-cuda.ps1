#requires -Version 7.2
[CmdletBinding()]
param([string]$Destination=(Join-Path (Split-Path $PSScriptRoot -Parent) 'dependencies/cuda-13.2.2'))
$ErrorActionPreference='Stop'
$version='13.2.2'
$base='https://developer.download.nvidia.com/compute/cuda/redist/'
New-Item -ItemType Directory -Force $Destination | Out-Null
$manifestPath=Join-Path $Destination 'redistrib.json'
if(-not(Test-Path $manifestPath)){Invoke-WebRequest ($base+"redistrib_$version.json") -OutFile $manifestPath}
if((Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne '89d93e6de640d47eb43c63437fe1be426ed21dd58cc79933a29acbca33c6eeb8'){throw 'Pinned NVIDIA13.2.2 manifest SHA256 mismatch'}
$manifest=Get-Content -Raw $manifestPath | ConvertFrom-Json
$packages=@('cuda_nvcc','cuda_crt','cuda_cudart','cuda_cccl','libnvvm','libcublas','cuda_nvdisasm','cuda_cuobjdump')
$root=Join-Path $Destination 'toolkit'
New-Item -ItemType Directory -Force $root | Out-Null
$records=@()
foreach($name in $packages){
 $entry=$manifest.$name.'windows-x86_64'
 if(-not $entry){throw "No Windows x64 archive for $name"}
 $archive=Join-Path $Destination ($name+'.zip')
 if(-not(Test-Path $archive)){Write-Output "Downloading $name ($($entry.size) bytes)";Invoke-WebRequest ($base+$entry.relative_path) -OutFile $archive}
 $hash=(Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
 if($hash -ne $entry.sha256){throw "SHA256 mismatch: $name; remove the corrupt archive before retrying"}
 $expanded=Join-Path $Destination ('extract/'+$name)
 if(-not(Test-Path $expanded)){Expand-Archive -LiteralPath $archive -DestinationPath $expanded}
 $payload=Get-ChildItem -Directory -LiteralPath $expanded | Select-Object -First 1
 if(-not $payload){throw "Unexpected archive layout: $name"}
 foreach($item in Get-ChildItem -LiteralPath $payload.FullName){if($item.Name -notin @('LICENSE','LICENSE.txt')){Copy-Item -LiteralPath $item.FullName -Destination $root -Recurse -Force}}
 $records+=@{name=$name;version=$manifest.$name.version;url=($base+$entry.relative_path);sha256=$hash;size=(Get-Item -LiteralPath $archive).Length;license=$manifest.$name.license}
}
$records | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 (Join-Path $Destination 'acquired-packages.json')
& (Join-Path $root 'bin/nvcc.exe') --version
if($LASTEXITCODE -ne 0){throw 'nvcc smoke failed'}
Write-Output "Workspace-local toolkit: $root. No drivers, registry or system SDKs were changed."
