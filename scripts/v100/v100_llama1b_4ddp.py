import os
os.system("litgpt finetune_full --config configs/tiny-llama-full.yaml --devices 4 --strategy ddp --precision bf16-mixed --train.global_batch_size 32 --train.micro_batch_size 8")
