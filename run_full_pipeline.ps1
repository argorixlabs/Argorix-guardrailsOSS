$ErrorActionPreference = "Stop"

$Python = "env\Scripts\python.exe"

& $Python translate.py `
  --profile finetune `
  --max-rows-per-label 60000 `
  --max-safe-rows 60000 `
  --batch-size 64 `
  --checkpoint-every 500 `
  --max-input-tokens 256 `
  --max-new-tokens 256

& $Python prepare_finetune_dataset.py `
  --output data_finetune\guardrail_es.parquet `
  --max-variants-per-row 2 `
  --max-train-per-label 120000 `
  --max-eval-per-label 10000 `
  --max-train-per-source-label 60000
