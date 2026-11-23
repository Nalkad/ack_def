# Final pipeline: figures -> qualitative -> report -> validation -> packages.
#   pwsh -ExecutionPolicy Bypass -File scripts\finalize.ps1 -Sid 2024123456 -Name 张三
param(
    [string]$Sid = "____________",
    [string]$Name = "____________",
    [string]$Major = "____________"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONPATH = (Get-Location).Path
$env:PYTHONUNBUFFERED = "1"
$py = ".venv\Scripts\python.exe"

function Step($msg) { Write-Host "=== $msg  $(Get-Date -Format 'HH:mm:ss') ===" }

Step "rebuild run metadata"
& $py tools\rebuild_runs.py

Step "Fix targeted-attack rows"
$tags = @()
Get-ChildItem results\defense_*.csv | ForEach-Object {
    $tags += $_.BaseName -replace '^defense_', ''
}
if ($tags.Count -gt 0) { & $py tools\fix_targeted.py --tag @tags }

Step "Figures (data-driven)"
& $py -m src.figures

Step "Figures (model-dependent)"
& $py -m src.qualitative

Step "Report"
& $py report\build_report.py --out "report\报告_对抗攻击与对抗防御.docx" --sid $Sid --name $Name --major $Major

Step "Validate"
& $py report\validate.py "report\报告_对抗攻击与对抗防御.docx"

Step "Package"
& $py scripts\package.py --sid $Sid --name $Name

Step "DONE"
