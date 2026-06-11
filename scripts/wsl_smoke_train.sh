#!/usr/bin/env bash
set -euo pipefail

cd /mnt/d/Proyectos/GuardrailsGovernance

VENV="/home/$(whoami)/.venvs/guardrails-governance"
python3 -m venv "$VENV"
source "$VENV/bin/activate"

python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
python -m pip install transformers datasets accelerate peft trl bitsandbytes pandas pyarrow safetensors

python train_qlora_guardrail.py \
  --dataset data_finetune/guardrail_es.parquet \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir models/guardrail-qwen25-1_5b-qlora-smoke \
  --max-train-samples 2000 \
  --max-eval-samples 200 \
  --max-steps 20 \
  --eval-steps 10 \
  --save-steps 10 \
  --logging-steps 2 \
  --train-batch-size 2 \
  --eval-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --max-length 512 \
  --bf16
