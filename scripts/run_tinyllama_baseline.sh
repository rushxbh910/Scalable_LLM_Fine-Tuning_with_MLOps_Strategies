#!/usr/bin/env bash
set -euo pipefail
# Baseline (expected to OOM on many GPUs at batch_size=32); keep logs for report
litgpt finetune_full --config configs/tiny-llama-full.yaml --train.global_batch_size 32 --train.micro_batch_size 32 || true
