$ErrorActionPreference = "Stop"

$root = Resolve-Path (Join-Path $PSScriptRoot "..")
$modelDir = Join-Path $root "models\beit3"
New-Item -ItemType Directory -Force $modelDir | Out-Null

$spm = Join-Path $modelDir "beit3.spm"
$checkpoint = Join-Path $modelDir "beit3_large_patch16_384_coco_retrieval.pth"
$spmUrl = "https://github.com/addf400/files/releases/download/beit3/beit3.spm"
$checkpointUrl = "https://github.com/addf400/files/releases/download/beit3/beit3_large_patch16_384_coco_retrieval.pth"

function Test-PyTorchCheckpoint([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $script = @"
import sys, torch
p = sys.argv[1]
try:
    obj = torch.load(p, map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError("checkpoint root is not a dict")
    if "model" not in obj:
        raise RuntimeError("checkpoint does not contain a 'model' state dict")
except Exception as exc:
    print(f"INVALID: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
print("VALID")
"@
    $tmp = Join-Path $env:TEMP "aic2026_validate_beit3.py"
    Set-Content -LiteralPath $tmp -Value $script -Encoding UTF8
    try {
        python $tmp $Path | Out-Null
        return ($LASTEXITCODE -eq 0)
    } finally {
        Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue
    }
}

if (-not (Test-Path -LiteralPath $spm)) {
    Write-Host "Downloading beit3.spm..."
    curl.exe -L --fail --retry 5 -o $spm $spmUrl
}

$checkpointValid = Test-PyTorchCheckpoint $checkpoint
if (-not $checkpointValid) {
    if (Test-Path -LiteralPath $checkpoint) {
        $broken = "$checkpoint.broken"
        $i = 1
        while (Test-Path -LiteralPath $broken) {
            $broken = "$checkpoint.broken.$i"
            $i++
        }
        Move-Item -LiteralPath $checkpoint -Destination $broken
        Write-Host "Existing checkpoint is invalid; moved to $broken"
    }

    Write-Host "Downloading BEiT-3 Large COCO retrieval checkpoint (~1.35 GB)..."
    curl.exe -L --fail --retry 5 -o $checkpoint $checkpointUrl

    if (-not (Test-PyTorchCheckpoint $checkpoint)) {
        throw "Downloaded BEiT-3 checkpoint failed torch.load validation: $checkpoint"
    }
}

Write-Host "Assets:"
Get-Item $spm, $checkpoint | Select-Object FullName, Length
Write-Host "BEiT-3 assets are valid under models\beit3."
