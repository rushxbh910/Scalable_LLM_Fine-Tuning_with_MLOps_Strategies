import os
from lightning import Trainer
from lightning.pytorch.strategies import SingleDeviceStrategy
from litgpt import scripts as lg  # placeholder import; CLI is primary path
# Minimal launcher: use CLI under the hood for stability
os.system("litgpt finetune_full --config configs/open-llama-7b-full.yaml --devices 1 --precision bf16-mixed --train.global_batch_size 4 --train.micro_batch_size 4")
