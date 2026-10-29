#requires -Version 7.2
[CmdletBinding()]
param([string[]]$Backends=@('host','opencl'),[ValidateSet('Release','Debug','RelWithDebInfo')][string]$Configuration='Release',[string]$CudaRoot,[string]$CutlassRoot,[string]$HipRoot,[string]$OneApiRoot,[string]$VCToolsVersion,[string]$CMakePath,[switch]$Clean)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
try {
 $vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
 if(-not(Test-Path -LiteralPath $vswhere)){throw 'Missing vswhere. Install Visual Studio C++ desktop tools and Windows SDK.'}
 $vs=(& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
 if(-not $vs){throw 'MSVC x64 C++ tools not found.'}
 $devshell=Join-Path $vs 'Common7\Tools\Launch-VsDevShell.ps1'
 $devargs='-arch=x64 -host_arch=x64'
 if($VCToolsVersion){$devargs+=' -vcvars_ver='+$VCToolsVersion}
 if($VCToolsVersion){
  Import-Module (Join-Path $vs 'Common7\Tools\Microsoft.VisualStudio.DevShell.dll')
  Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments $devargs
 }else{& $devshell -SkipAutomaticLocation -Arch amd64 -HostArch amd64 -NoLogo}
 # Stable English /showIncludes is needed for Ninja dependency tracking on
 # Chinese Windows installations; localized output can silently lose headers.
 $env:VSLANG='1033'
 if(-not $CMakePath){$cmd=Get-Command cmake -ErrorAction SilentlyContinue;if($cmd){$CMakePath=$cmd.Source}else{$CMakePath=Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'}}
 if(-not(Test-Path -LiteralPath $CMakePath)){throw 'CMake >=3.25 missing; supply -CMakePath.'}
 foreach($backend in $Backends){
  if($backend -notin @('host','cuda','amd','intel','opencl')){throw "Unknown backend $backend"}
  $build=Join-Path $repo "build/$backend-$Configuration"
  $args=@('-S',$repo,'-B',$build,'-G','Ninja',"-DCMAKE_BUILD_TYPE=$Configuration","-DGPUBENCH_BACKEND=$backend")
  if($backend -eq 'cuda'){
   if(-not $CudaRoot){$CudaRoot=$env:CUDA_PATH}
   if(-not $CudaRoot -or -not(Test-Path "$CudaRoot/bin/nvcc.exe")){throw 'CUDA >=12.9.2 required; supply -CudaRoot. CUDA 11.8 cannot build SM120.'}
   $nvccVersion=(& "$CudaRoot/bin/nvcc.exe" --version | Out-String)
   if($nvccVersion -notmatch 'release (\d+)\.(\d+)'){throw 'Cannot read nvcc version'}
   if([int]$Matches[1] -lt 12 -or ([int]$Matches[1] -eq 12 -and [int]$Matches[2] -lt 9)){throw 'CUDA >=12.9.2 required; installed toolkit is older.'}
   $args+="-DCMAKE_CUDA_COMPILER=$CudaRoot/bin/nvcc.exe";$args+="-DCUDAToolkit_ROOT=$CudaRoot"
  }
  if($backend -eq 'cuda' -and $CutlassRoot){& (Join-Path $PSScriptRoot 'acquire-cutlass.ps1') -Destination $CutlassRoot; if($LASTEXITCODE -ne 0){throw 'Pinned CUTLASS verification failed'};$args+="-DGPUBENCH_CUTLASS_ROOT=$CutlassRoot"}
  if($backend -eq 'amd'){
   if(-not $HipRoot){$HipRoot=$env:HIP_PATH}
   if(-not $HipRoot -or -not(Test-Path -LiteralPath $HipRoot)){throw 'ROCm Core SDK 10.1.0 Windows missing; supply -HipRoot (HIP, headers, kernel data).'}
   $args+="-DGPUBENCH_ROCM_ROOT=$HipRoot"
  }
  if($backend -eq 'intel'){
   if($OneApiRoot){$env:ONEAPI_ROOT=$OneApiRoot}
   $icx=Get-Command icx -ErrorAction SilentlyContinue
   if(-not $icx){throw 'Matched DPC++/oneMKL/UR environment missing. Run the pinned oneAPI setvars.bat, or supply -OneApiRoot and initialize that release.'}
   $args+="-DCMAKE_CXX_COMPILER=$($icx.Source)"
  }
  & $CMakePath @args
  if($LASTEXITCODE -ne 0){throw "CMake configure failed: $backend ($LASTEXITCODE)"}
  $buildArgs=@('--build',$build,'--config',$Configuration)
  if($Clean){$buildArgs+='--clean-first'}
  & $CMakePath @buildArgs
  if($LASTEXITCODE -ne 0){throw "Build failed: $backend ($LASTEXITCODE)"}
  & $CMakePath --install $build --config $Configuration --prefix (Join-Path $repo "out/$Configuration")
  if($LASTEXITCODE -ne 0){throw "Install failed: $backend ($LASTEXITCODE)"}
  if($backend -eq 'host'){
   $ctest=Join-Path (Split-Path $CMakePath -Parent) 'ctest.exe'
   & $ctest --test-dir $build --output-on-failure
   if($LASTEXITCODE -ne 0){throw 'Host tests failed'}
  }
 }
 exit 0
} catch {Write-Error $_ -ErrorAction Continue;exit 1}
