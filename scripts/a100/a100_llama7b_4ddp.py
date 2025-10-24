import os
os.system("litgpt finetune_full --config configs/open-llama-7b-full.yaml --devices 4 --strategy ddp --precision bf16-mixed --train.global_batch_size 16 --train.micro_batch_size 4")
