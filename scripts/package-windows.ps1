#requires -Version 7.2
[CmdletBinding()]
param([ValidateSet('Release','Debug','RelWithDebInfo')][string]$Configuration='Release',[string]$CudaRoot,[string]$RuntimeManifest)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$installed=Join-Path $repo "out/$Configuration"
$dist=Join-Path $repo "dist/$Configuration"
try{
 if(-not(Test-Path -LiteralPath (Join-Path $installed 'gpubench.exe'))){throw 'Build host before packaging'}
 New-Item -ItemType Directory -Force $dist | Out-Null
 $vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
 $vs=& $vswhere -latest -products '*' -property installationPath | Select-Object -First 1
 $tools=Get-ChildItem -Directory -LiteralPath (Join-Path $vs 'VC/Tools/MSVC') | Sort-Object Name -Descending | Select-Object -First 1
 $dumpbin=Join-Path $tools.FullName 'bin/Hostx64/x64/dumpbin.exe'
 if(-not(Test-Path -LiteralPath $dumpbin)){throw 'dumpbin required to audit actual PE imports'}
 $release=@()
 $packages=@('host')+@(Get-ChildItem -Directory -LiteralPath (Join-Path $installed 'workers') -ErrorAction SilentlyContinue | ForEach-Object Name)
 foreach($backend in $packages){
  $stage=Join-Path $dist ('gpubench-'+$backend+'-0.1.0-win-x64')
  if(Test-Path -LiteralPath $stage){throw "Package staging already exists: $stage. Use a fresh Configuration/output directory; do not silently mix releases."}
  New-Item -ItemType Directory -Force $stage | Out-Null
  if($backend -eq 'host'){
   Copy-Item -LiteralPath (Join-Path $installed 'gpubench.exe') -Destination $stage
   foreach($name in @('README.md','dependencies.lock.json','WINDOWS_CONSUMER_GPU_BENCHMARK_AGENT_PLAN.md')){Copy-Item -LiteralPath (Join-Path $repo $name) -Destination $stage}
   foreach($name in @('docs','schemas','scripts')){Copy-Item -LiteralPath (Join-Path $repo $name) -Destination $stage -Recurse}
   $reports=Join-Path $stage 'reports';New-Item -ItemType Directory -Force $reports | Out-Null
   foreach($name in @('RTX5070Ti_REPORT.html','RTX5070Ti_REPORT.json')){Copy-Item -LiteralPath (Join-Path $repo "results/$name") -Destination $reports}
   $licenses=Join-Path $stage 'licenses';New-Item -ItemType Directory -Force $licenses | Out-Null
   Copy-Item -LiteralPath (Join-Path $repo 'third_party/nlohmann/LICENSE.MIT') -Destination (Join-Path $licenses 'nlohmann-json-MIT.txt')
   Copy-Item -LiteralPath (Join-Path $repo 'third_party/OpenCL-Headers/LICENSE') -Destination (Join-Path $licenses 'OpenCL-Headers-LICENSE.txt')
  }else{
   $subdir=Join-Path $stage "workers/$backend"
   New-Item -ItemType Directory -Force $subdir | Out-Null
   Copy-Item -Path (Join-Path $installed "workers/$backend/*") -Destination $subdir -Recurse
   if($backend -eq 'cuda'){foreach($name in @('VALIDATION.md','evidence.json')){Copy-Item -LiteralPath (Join-Path $repo "workers/cuda/$name") -Destination $subdir}}
   if($backend -eq 'cuda'){
    if(-not $CudaRoot){$CudaRoot=$env:CUDA_PATH}
    if(-not $CudaRoot){throw 'CUDA package requires -CudaRoot matching dependencies.lock.json for runtime provenance'}
    $dll=Join-Path $CudaRoot 'bin/x64/cublasLt64_13.dll'
    if(-not(Test-Path -LiteralPath $dll)){throw 'Locked cuBLASLt runtime missing; no SDK-wide DLL glob is allowed'}
    Copy-Item -LiteralPath $dll -Destination $subdir -Force
    $license=Join-Path $CudaRoot 'LICENSE';if(-not(Test-Path -LiteralPath $license)){$license=Join-Path $CudaRoot 'LICENSE.txt'}
    $archiveLicense=Join-Path (Split-Path $CudaRoot -Parent) 'extract/libcublas'
    $found=Get-ChildItem -Recurse -File -LiteralPath $archiveLicense -ErrorAction SilentlyContinue | Where-Object Name -Match '^LICENSE' | Select-Object -First 1
    if($found){Copy-Item -LiteralPath $found.FullName -Destination (Join-Path $stage 'NVIDIA-LICENSE.txt')}elseif(Test-Path $license){Copy-Item -LiteralPath $license -Destination (Join-Path $stage 'NVIDIA-LICENSE.txt')}else{throw 'NVIDIA redistribution license not found'}
   }
   if($backend -in @('amd','intel')){throw "No validated runtime redistribution implementation for $backend yet. Source and validation scripts are delivered; binary packaging requires SDK smoke and licensed runtime manifest implementation."}
  }
  $imports=Join-Path $stage 'pe-imports';New-Item -ItemType Directory -Force $imports | Out-Null
  $files=@(Get-ChildItem -Recurse -File -LiteralPath $stage)
  foreach($binary in $files | Where-Object Extension -In @('.exe','.dll')){
   & $dumpbin /headers /dependents $binary.FullName > (Join-Path $imports ($binary.Name+'.txt'))
   if($LASTEXITCODE -ne 0){throw "PE audit failed: $($binary.FullName)"}
  }
  $manifest=[ordered]@{schema_version=1;app_version='0.1.0';backend=$backend;configuration=$Configuration;platform='Windows x64';created_utc=[DateTime]::UtcNow.ToString('o');clean_windows_verified=$false;deployment=if($backend -eq 'opencl'){'Requires vendor display driver with OpenCL ICD; system OpenCL/driver DLLs are not redistributed'}elseif($backend -eq 'cuda'){'Requires compatible NVIDIA display driver; static MSVC/cudart, private locked cuBLASLt; clean-machine verification pending'}else{'Static MSVC C++ runtime; Windows system DLLs only; no GPU SDK required'};files=@()}
  foreach($file in Get-ChildItem -Recurse -File -LiteralPath $stage){$relative=[IO.Path]::GetRelativePath($stage,$file.FullName);$provenance=if($file.Name -eq 'cublasLt64_13.dll'){Join-Path $CudaRoot 'bin/x64/cublasLt64_13.dll'}else{'gpubench source/build'};$manifest.files+=@{path=$relative;sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant();bytes=$file.Length;source=$provenance;version=$file.VersionInfo.FileVersion}}
  $manifest | ConvertTo-Json -Depth 10 | Set-Content -Encoding utf8 (Join-Path $stage 'manifest.json')
  $zip=$stage+'.zip'
  Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip -CompressionLevel Optimal
  $release+=@{path=[IO.Path]::GetFileName($zip);sha256=(Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant();bytes=(Get-Item -LiteralPath $zip).Length}
 }
 $release | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 (Join-Path $dist 'SHA256.json')
 Write-Output "Packages: $dist (clean-machine verification pending)"
 exit 0
}catch{Write-Error $_ -ErrorAction Continue;exit 1}
