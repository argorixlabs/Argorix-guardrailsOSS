$ErrorActionPreference = "Stop"

$project = Split-Path -Parent $MyInvocation.MyCommand.Path
$translated = Join-Path $project "data_es\bogdanminko__Catch_the_prompt_injection_or_jailbreak_or_benign\train.parquet"
$partial = Join-Path $project "data_es\bogdanminko__Catch_the_prompt_injection_or_jailbreak_or_benign\train.partial.parquet"
$log = Join-Path $project "logs\v2_prepare_after_translate.log"

while ((-not (Test-Path $translated)) -or (Test-Path $partial)) {
    Start-Sleep -Seconds 60
}

Set-Location $project
& "env\Scripts\python.exe" "prepare_finetune_dataset.py" `
    --output "data_finetune\guardrail_es_v2.parquet" `
    --max-variants-per-row 2 `
    --max-train-per-label 160000 `
    --max-eval-per-label 12000 `
    --max-train-per-source-label 90000 `
    *> $log
