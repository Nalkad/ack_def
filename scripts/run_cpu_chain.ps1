# CPU training chain (serial), runs in parallel with the GPU chain.
# VGG-16 (surrogate for the transfer experiment) -> AT(eps=4) -> TRADES(beta=6) -> AT(20 steps)
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

function Wait-Ckpt($path, $label) {
    while (-not (Test-Path $path)) {
        Write-Host "waiting for $label ..."
        Start-Sleep -Seconds 30
    }
}

Run "std_vgg16" "--mode std --model vgg16 --epochs 10 --tag std_vgg16 --bs 128 --lr 0.05 --wd 5e-4 --workers 0 --seed 0 --threads 10 --device cpu --eval-subset 0"
Wait-Ckpt "checkpoints/std_r18.pt" "std_r18.pt"
Run "at_r18_eps4_s7" "--mode pgd_at --model resnet18 --epochs 1 --tag at_r18_eps4_s7 --init checkpoints/std_r18.pt --eps 4 --steps 7 --bs 128 --lr 0.05 --wd 5e-4 --workers 0 --seed 0 --threads 10 --device cpu --eval-subset 0"
Run "trades_r18_b6" "--mode trades --model resnet18 --epochs 1 --tag trades_r18_b6 --init checkpoints/std_r18.pt --eps 8 --trades_steps 10 --beta 6 --bs 128 --lr 0.05 --wd 5e-4 --workers 0 --seed 0 --threads 10 --device cpu --eval-subset 0"
Run "at_r18_eps8_s20" "--mode pgd_at --model resnet18 --epochs 1 --tag at_r18_eps8_s20 --init checkpoints/std_r18.pt --eps 8 --steps 20 --bs 128 --lr 0.05 --wd 5e-4 --workers 0 --seed 0 --threads 10 --device cpu --eval-subset 0"
Write-Host "CPU chain finished $(Get-Date -Format 'HH:mm:ss')"
