# GPU training chain (serial): clean ResNet-18 -> PGD-AT(7 steps) -> PGD-AT(10 steps)
# Logs: logs/gpu_chain.log ; per-epoch CSVs: results/logs/*.csv
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = $root
$env:PYTHONUNBUFFERED = "1"
$py = Join-Path $root ".venv\Scripts\python.exe"
New-Item -ItemType Directory -Force -Path results\logs, results\runs, checkpoints, logs | Out-Null

function Run($name, $argsLine) {
    Write-Host "=== $name  $(Get-Date -Format 'HH:mm:ss') ==="
    $cmd = "`"$py`" -u -m src.train $argsLine > logs\train_$name.log 2>&1"
    cmd /c $cmd
    Write-Host "=== $name done  $(Get-Date -Format 'HH:mm:ss') exit=$LASTEXITCODE ==="
}

Run "std_r18" "--mode std --model resnet18 --epochs 10 --tag std_r18 --bs 128 --lr 0.1 --wd 5e-4 --workers 0 --seed 0 --eval-subset 512 --save-every-epoch"
Run "at_r18_eps8_s7" "--mode pgd_at --model resnet18 --epochs 2 --tag at_r18_eps8_s7 --init checkpoints/std_r18.pt --eps 8 --steps 7 --bs 128 --lr 0.05 --wd 5e-4 --workers 0 --seed 0 --eval-subset 512 --save-every-epoch"
Run "at_r18_eps8_s10" "--mode pgd_at --model resnet18 --epochs 1 --tag at_r18_eps8_s10 --init checkpoints/std_r18.pt --eps 8 --steps 10 --bs 128 --lr 0.05 --wd 5e-4 --workers 0 --seed 0 --eval-subset 512"
Write-Host "GPU chain finished $(Get-Date -Format 'HH:mm:ss')"
