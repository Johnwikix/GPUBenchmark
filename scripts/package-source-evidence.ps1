#requires -Version 7.2
[CmdletBinding()]
param([string]$OutputDirectory=(Join-Path (Split-Path $PSScriptRoot -Parent) 'dist/Release'),[switch]$Replace)
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
try{
 New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
 $sourcePaths=@('CMakeLists.txt','CMakePresets.json','.gitignore','README.md','dependencies.lock.json','WINDOWS_CONSUMER_GPU_BENCHMARK_AGENT_PLAN.md','src','workers','kernels','schemas','tests','scripts','docs','third_party')
 $manifest=@()
 foreach($name in $sourcePaths){$path=Join-Path $repo $name;foreach($file in Get-ChildItem -Recurse -File -LiteralPath $path){$manifest+=@{path=[IO.Path]::GetRelativePath($repo,$file.FullName);sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant();bytes=$file.Length}}}
 $manifestPath=Join-Path $repo 'results/source-manifest.json'
 $manifest|ConvertTo-Json -Depth 5|Set-Content -Encoding utf8 $manifestPath
 $sourceZip=Join-Path $OutputDirectory 'gpubench-source-0.1.0.zip'
 if((Test-Path -LiteralPath $sourceZip) -and -not $Replace){throw "Source archive exists: $sourceZip; use -Replace to regenerate this local artifact"}
 Compress-Archive -Path @($sourcePaths|ForEach-Object{Join-Path $repo $_}) -DestinationPath $sourceZip -CompressionLevel Optimal -Force:$Replace
 $evidenceZip=Join-Path $OutputDirectory 'gpubench-evidence-20261021.zip'
 if((Test-Path -LiteralPath $evidenceZip) -and -not $Replace){throw "Evidence archive exists: $evidenceZip; use -Replace to regenerate this local artifact"}
 # A transcript redirected by this running packager may be write-only/locked.
 # All benchmark commands, samples and worker logs are retained.
 $evidencePaths=@(Get-ChildItem -LiteralPath (Join-Path $repo 'results') | Where-Object Name -NotLike 'package-source-evidence*.log' | ForEach-Object FullName)
 Compress-Archive -Path $evidencePaths -DestinationPath $evidenceZip -CompressionLevel Optimal -Force:$Replace
 $hashes=@(Get-ChildItem -File -LiteralPath $OutputDirectory -Filter '*.zip' | ForEach-Object{@{path=$_.Name;sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant();bytes=$_.Length}})
 $hashes|ConvertTo-Json -Depth 6|Set-Content -Encoding utf8 (Join-Path $OutputDirectory 'SHA256.json')
 Write-Output "Source/evidence archives: $OutputDirectory"
 exit 0
}catch{Write-Error $_ -ErrorAction Continue;exit 1}
