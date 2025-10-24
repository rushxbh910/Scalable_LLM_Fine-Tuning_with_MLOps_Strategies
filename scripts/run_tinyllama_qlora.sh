#!/usr/bin/env bash
set -euo pipefail
litgpt finetune --config configs/tiny-llama-lora.yaml --quantize bnb.nf4 | tee reports/results/tinyllama_qlora.log
python scripts/utils/parse_litgpt_log.py reports/results/tinyllama_qlora.log reports/results/summary.json qlora
