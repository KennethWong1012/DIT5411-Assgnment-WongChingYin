param(
    [double]$Hours = 8,
    [int]$Epochs = 4,
    [int]$Batch = 256
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUNBUFFERED = '1'
$env:TF_CPP_MIN_LOG_LEVEL = '2'

if (!(Test-Path -LiteralPath './data/prepared/metadata.json')) {
    python prepare.py data/upstream.zip
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

python train.py --benchmark 40 --batch $Batch --output run/benchmark
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python train.py --hours $Hours --epochs $Epochs --batch $Batch
exit $LASTEXITCODE
