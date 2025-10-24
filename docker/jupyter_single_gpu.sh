#!/usr/bin/env bash
set -euo pipefail
docker pull quay.io/jupyter/pytorch-notebook:cuda12-pytorch-2.4
docker run -d --gpus all -p 8888:8888 --name jupyter quay.io/jupyter/pytorch-notebook:cuda12-pytorch-2.4
sleep 3
docker logs jupyter | tail -n 20
