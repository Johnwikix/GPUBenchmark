#requires -Version 7.2
[CmdletBinding()]
param([string]$ResultsDirectory=(Join-Path (Split-Path $PSScriptRoot -Parent) 'results'))
$ErrorActionPreference='Stop'
$repo=Split-Path $PSScriptRoot -Parent
$protocolSchema=Get-Content -Raw -LiteralPath (Join-Path $repo 'schemas/protocol-v1.schema.json')
$resultSchema=Get-Content -Raw -LiteralPath (Join-Path $repo 'schemas/result-v1.schema.json')
$count=0
try{
 foreach($file in Get-ChildItem -Recurse -File -LiteralPath $ResultsDirectory -Filter '*.protocol.jsonl'){
  foreach($line in Get-Content -LiteralPath $file.FullName){if(-not $line.Trim()){continue};$m=$line|ConvertFrom-Json;if($m.PSObject.Properties.Name -contains 'message'){$line=$m.message|ConvertTo-Json -Depth 100 -Compress};if(-not(Test-Json -Json $line -Schema $protocolSchema)){throw "Protocol schema failed: $($file.FullName)"};$count++}
 }
 foreach($file in Get-ChildItem -Recurse -File -LiteralPath $ResultsDirectory -Filter '*.json'){
  $document=Get-Content -Raw -LiteralPath $file.FullName | ConvertFrom-Json
  if(($document.PSObject.Properties.Name -contains 'results') -and ($document.PSObject.Properties.Name -contains 'app_version')){
   foreach($result in $document.results){if(-not(Test-Json -Json ($result|ConvertTo-Json -Depth 100 -Compress) -Schema $resultSchema)){throw "Result schema failed: $($file.FullName)"};$count++}
  }
 }
 @{checked=$count;status='passed';utc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json|Set-Content -Encoding utf8 (Join-Path $ResultsDirectory 'schema-validation.json')
 Write-Output "Validated $count protocol messages/results"
 exit 0
}catch{Write-Error $_ -ErrorAction Continue;exit 1}
