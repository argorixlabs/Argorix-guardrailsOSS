#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/Proyectos/GuardrailsGovernance

VENV="/home/$(whoami)/.venvs/guardrails-governance"
source "$VENV/bin/activate"

python train_qlora_guardrail.py \
  --dataset data_finetune/guardrail_es_v2.parquet \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir models/guardrail-qwen25-1_5b-qlora-v2 \
  --max-eval-samples 3000 \
  --epochs 1 \
  --max-steps 4000 \
  --eval-steps 250 \
  --save-steps 250 \
  --logging-steps 20 \
  --train-batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --max-length 512 \
  --bf16
