#!/usr/bin/env bash
set -euo pipefail
litgpt finetune_full --config configs/tiny-llama-full.yaml --train.global_batch_size 16 --train.micro_batch_size 16 --precision bf16-mixed | tee reports/results/tinyllama_reduced_precision.log
python scripts/utils/parse_litgpt_log.py reports/results/tinyllama_reduced_precision.log reports/results/summary.json reduced_precision
