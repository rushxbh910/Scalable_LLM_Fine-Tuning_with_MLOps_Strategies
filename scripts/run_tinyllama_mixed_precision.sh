#!/usr/bin/env bash
set -euo pipefail
litgpt finetune_full --config configs/tiny-llama-full.yaml --train.global_batch_size 16 --train.micro_batch_size 16 --precision 16-mixed | tee reports/results/tinyllama_mixed_precision.log
python scripts/utils/parse_litgpt_log.py reports/results/tinyllama_mixed_precision.log reports/results/summary.json mixed_precision
