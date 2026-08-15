$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$modelDir = Join-Path $root "models\beit3"
New-Item -ItemType Directory -Force $modelDir | Out-Null

$spm = Join-Path $modelDir "beit3.spm"
$checkpoint = Join-Path $modelDir "beit3_large_patch16_384_coco_retrieval.pth"

$spmUrl = "https://github.com/addf400/files/releases/download/beit3/beit3.spm"
$checkpointUrl = "https://github.com/addf400/files/releases/download/beit3/beit3_large_patch16_384_coco_retrieval.pth"

if (-not (Test-Path $spm)) {
    Write-Host "Downloading beit3.spm..."
    curl.exe -L --fail --retry 3 -o $spm $spmUrl
}

if (-not (Test-Path $checkpoint)) {
    Write-Host "Downloading BEiT-3 Large COCO retrieval checkpoint (~1.35 GB)..."
    curl.exe -L --fail --retry 3 -o $checkpoint $checkpointUrl
}

Write-Host "Assets:"
Get-Item $spm, $checkpoint | Select-Object FullName, Length
Write-Host "BEiT-3 assets are ready under models\beit3."
