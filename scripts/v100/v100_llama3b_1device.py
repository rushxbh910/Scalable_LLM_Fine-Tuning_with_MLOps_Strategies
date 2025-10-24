import os
# This is expected to OOM on a single V100; run to demonstrate the limitation (keep logs)
os.system("litgpt finetune_full --config configs/open-llama-3b-full.yaml --devices 1 --precision bf16-mixed --train.global_batch_size 1 --train.micro_batch_size 1 || true")
