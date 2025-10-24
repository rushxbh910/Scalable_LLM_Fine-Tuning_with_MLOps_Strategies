#!/usr/bin/env bash
set -euo pipefail
litgpt finetune --config configs/tiny-llama-lora.yaml | tee reports/results/tinyllama_lora.log
python scripts/utils/parse_litgpt_log.py reports/results/tinyllama_lora.log reports/results/summary.json lora
