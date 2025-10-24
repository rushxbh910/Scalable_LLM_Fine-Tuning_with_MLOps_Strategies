#!/usr/bin/env bash
set -euo pipefail
litgpt finetune_full --config configs/tiny-llama-full.yaml --train.global_batch_size 8 --train.micro_batch_size 8 | tee reports/results/tinyllama_reduced_batch.log
python scripts/utils/parse_litgpt_log.py reports/results/tinyllama_reduced_batch.log reports/results/summary.json reduced_batch
