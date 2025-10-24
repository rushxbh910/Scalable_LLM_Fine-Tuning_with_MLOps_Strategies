import os
os.system("litgpt finetune_full --config configs/open-llama-3b-full.yaml --devices 4 --strategy fsdp --precision bf16-mixed --train.global_batch_size 4 --train.micro_batch_size 1 --train.limit_train_batches 0.2")
