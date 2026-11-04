# 一键复现（PowerShell）
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

& $py scripts\download_data.py --train 40 --valid 10 --test 10 --workers 3
& $py scripts\prepare_data.py --patch 256 --stride 256
& $py scripts\audit_dataset.py

foreach ($model in @("unet", "msf", "full")) {
    & $py scripts\train.py --model $model --epochs 12 --max-train 900 --base 16 `
        --batch-size 8 --threads 8 --channels-last
}

& $py scripts\evaluate.py --models unet msf full --threshold 0.5
& $py scripts\analyze_results.py
& $py scripts\plot_training.py
Write-Host "全部完成：results/ 与 report/"
