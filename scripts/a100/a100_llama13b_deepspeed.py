import os
# stage=3 + CPU optimizer offload
os.system("litgpt finetune_full --config configs/open-llama-13b-full.yaml --devices 4 --strategy deepspeed_stage_3_offload --precision bf16-mixed --train.global_batch_size 8 --train.micro_batch_size 2")
