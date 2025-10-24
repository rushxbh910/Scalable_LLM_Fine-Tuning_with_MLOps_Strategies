#!/usr/bin/env bash
set -euo pipefail
# Simulate global batch 32 using micro-batches of 8 and grad accumulation steps 4
litgpt finetune_full --config configs/tiny-llama-full.yaml --train.global_batch_size 32 --train.micro_batch_size 8 --train.accumulate_grad_batches 4 | tee reports/results/tinyllama_grad_accum.log
python scripts/utils/parse_litgpt_log.py reports/results/tinyllama_grad_accum.log reports/results/summary.json grad_accum
