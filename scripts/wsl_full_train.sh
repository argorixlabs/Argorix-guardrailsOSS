#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/Proyectos/GuardrailsGovernance

VENV="/home/$(whoami)/.venvs/guardrails-governance"
source "$VENV/bin/activate"

python train_qlora_guardrail.py \
  --dataset data_finetune/guardrail_es.parquet \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir models/guardrail-qwen25-1_5b-qlora-v1 \
  --max-eval-samples 2000 \
  --epochs 1 \
  --max-steps 3000 \
  --eval-steps 250 \
  --save-steps 250 \
  --logging-steps 20 \
  --train-batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --max-length 512 \
  --bf16
