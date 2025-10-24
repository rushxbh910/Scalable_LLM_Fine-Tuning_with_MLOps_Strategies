# 🔧 Scalable LLM Fine‑Tuning Using MLOps Strategies

**Author:** Rushabh Bhatt

This repository is a ready‑to‑run, reproducible project for scalable fine‑tuning of large language models (LLMs) on **Chameleon Cloud** using **LitGPT** and **PyTorch Lightning**. It consolidates single‑GPU techniques, parameter‑efficient fine‑tuning, and multi‑GPU scaling (DDP/FSDP/DeepSpeed) across **V100 (32GB)** and **A100 (80GB)** GPUs. It mirrors the flow of the *Large‑scale model training on Chameleon* tutorial while providing turnkey scripts and configs.

---

## 📑 Table of Contents
1. [Overview](#overview)  
2. [Experiment Setup](#experiment-setup)  
3. [Quick Start](#quick-start)  
4. [Single‑GPU Optimization Techniques](#single-gpu-optimization-techniques)  
5. [Larger Models & PEFT](#larger-models--parameter-efficient-fine-tuning)  
6. [Multi‑GPU Scaling (DDP / FSDP / DeepSpeed)](#multi-gpu-scaling-ddp--fsdp--deepspeed)  
7. [Expected Screenshots](#expected-screenshots)  
8. [Key Insights](#key-insights)  
9. [References](#references)

---

## 🧠 Overview
This project explores efficient training of large‑scale language models by combining:
- Gradient accumulation  
- Mixed/reduced precision (bf16/fp16)  
- LoRA / QLoRA  
- Data‑parallel and sharded strategies (DDP, FSDP)  
- DeepSpeed CPU optimizer offload  
- Batch size tuning

---

## ⚙️ Experiment Setup
- **LitGPT version:** 0.5.7  
- **Lightning:** < 2.5  
- **GPUs:** V100 (32GB) and A100 (80GB)  
- **Cloud:** Chameleon  
- **Models:** TinyLlama 1.1B, OpenLLaMA 3B, 7B, 13B  
- **Parallelism:** DDP and FSDP (plus DeepSpeed for 13B)  
- **Repo layout:** see `configs/`, `scripts/`, `scripts/a100`, `scripts/v100`, `reports/`

> Add your screenshots from each run to `reports/screenshots/` using the exact filenames in [Expected Screenshots](#expected-screenshots). Some single‑GPU baselines are intentionally stress/oom tests—keep the logs.

---

## ⚡ Quick Start

### 0) Python env
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Get models & configs
```bash
# TinyLlama + OpenLLaMA families (run what you need)
litgpt download TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
litgpt download openlm-research/open_llama_3b
litgpt download openlm-research/open_llama_7b
litgpt download openlm-research/open_llama_13b

# Configs are in ./configs
```

### 2) Single‑GPU (TinyLlama‑1.1B) experiments
From repo root:
```bash
# Baseline (likely OOM by design; keep the log)
bash scripts/run_tinyllama_baseline.sh

# Reduced batch size
bash scripts/run_tinyllama_reduced_batch.sh

# Gradient Accumulation
bash scripts/run_tinyllama_grad_accum.sh

# Reduced precision (bf16/float16)
bash scripts/run_tinyllama_reduced_precision.sh

# Mixed precision
bash scripts/run_tinyllama_mixed_precision.sh

# LoRA / QLoRA
bash scripts/run_tinyllama_lora.sh
bash scripts/run_tinyllama_qlora.sh
```

### 3) Multi‑GPU (A100 80GB)
Open **two terminals** on the GPU host. In Terminal‑A, run the trainer; in Terminal‑B, keep `nvtop` open.
```bash
python scripts/a100/a100_llama7b_1device.py
python scripts/a100/a100_llama7b_4ddp.py
python scripts/a100/a100_llama7b_4fsdp.py
python scripts/a100/a100_llama7b_4fsdp_8batch.py
python scripts/a100/a100_llama13b_deepspeed.py
```

### 4) Multi‑GPU (V100 32GB)
```bash
python scripts/v100/v100_llama1b_1device.py
python scripts/v100/v100_llama1b_4ddp.py
python scripts/v100/v100_llama1b_4fsdp.py
python scripts/v100/v100_llama3b_4fsdp.py
# Optional: demonstrate OOM contrast
python scripts/v100/v100_llama3b_1device.py
```

### 5) Dockerized Jupyter (optional)
```bash
bash docker/jupyter_single_gpu.sh
# Then open the printed http://<FLOATING_IP>:8888/... link
```

---

## 🧪 Single‑GPU Optimization Techniques

Experiments on **TinyLlama‑1.1B** with different techniques. Numbers below are sample outputs (replace with your exact results parsed into `reports/results/summary.json`).

| Technique          | Time (s) | Memory (GB) |
|-------------------|----------|-------------|
| Baseline          | ❌ OOM   | >80         |
| Reduced Batch     | 165.70   | 30.41       |
| Gradient Accum.   | 57.30    | 34.82       |
| Reduced Precision | 36.86    | 20.10       |
| Mixed Precision   | 38.67    | 31.32       |

📷 **Screenshot:** `reports/screenshots/01_tinyllama_techniques.pdf`

**Why these effects?**  
- **Reduced batch size:** lowers activation memory per step → fits on smaller VRAM; more steps per epoch.  
- **Gradient accumulation:** simulates large global batch with small micro‑batches → similar memory to small batch while matching big‑batch dynamics.  
- **Reduced precision (bf16/fp16):** halves tensor precision → less memory, faster math on capable GPUs; slight numerical differences.  
- **Mixed precision:** keeps critical parts in fp32 and others in lower precision → balances stability with speed/memory.

---

## 📈 Larger Models & Parameter‑Efficient Fine‑Tuning

### Larger Models
| Model | Optimizer | Status | Memory |
|------:|-----------|--------|--------|
| 3B    | Adam      | ✅     | High        |
| 7B    | Adam      | ✅     | Very High   |
| 13B   | Adam      | ❌ OOM | N/A         |
| 13B   | SGD       | ✅     | Moderate    |

📷 **Screenshot:** `reports/screenshots/02_larger_models.pdf`

### Parameter‑Efficient Fine‑Tuning
| Model | LoRA | QLoRA | Time (s) | Tokens/sec | Mem (GB) |
|------:|:---:|:-----:|---------:|-----------:|---------:|
| 1.1B  | ✅   | ❌     | 42.31    | 8532.61    | 8.00     |
| 3B    | ✅   | ❌     | 89.23    | 3766.41    | 17.02    |
| 7B    | ✅   | ✅     | 136.61   | 1814.15    | 21.25    |
| 13B   | ✅   | ❌     | 226.19   | 1485.77    | 51.46    |

📷 **Screenshot:** `reports/screenshots/03_param_efficient_finetuning.pdf`

---

## 🖥️ Multi‑GPU Scaling (DDP / FSDP / DeepSpeed)

### DDP
| Setup        | Time  | GPUs Used | Memory/GPU |
|--------------|-------|-----------|------------|
| Single V100  | Slow  | 1         | 32 GB      |
| 4× V100 DDP  | Fast  | 4         | Reduced    |

📷 **Screenshot:** `reports/screenshots/04_ddp_vs_single.pdf`

### FSDP
| Model | Time   | GPUs | Memory    | Note                    |
|------:|--------|-----:|-----------|-------------------------|
| 1.1B  | Fast   | 4    | Low       | Efficient sharding      |
| 3B    | Medium | 4    | Moderate  | 3B fits cleanly w/ FSDP |

📷 **Screenshot:** `reports/screenshots/05_fsdp_training.pdf`

### DeepSpeed (CPU offload)
Used for 13B runs to reduce GPU memory by offloading optimizer state to host RAM; expect PCIe latency trade‑offs.

---

## 🖼️ Expected Screenshots
Drop the following into `reports/screenshots/`:

**TinyLlama‑1.1B (single‑GPU):**
- `baseline_end_metrics.png`
- `reduced_batch_end_metrics.png`
- `grad_accum_end_metrics.png`
- `reduced_precision_end_metrics.png`
- `mixed_precision_end_metrics.png`

**Multi‑GPU (capture both an `nvtop` during run and the trainer “end” line):**
- `a100_7b_1dev_nvtop.png` + `a100_7b_1dev_end.png`
- `a100_7b_4ddp_nvtop.png` + `a100_7b_4ddp_end.png`
- `a100_7b_4fsdp_nvtop.png` + `a100_7b_4fsdp_end.png`
- `a100_7b_4fsdp_8batch_nvtop.png` + `a100_7b_4fsdp_8batch_end.png`
- `a100_13b_deepspeed_nvtop.png` + `a100_13b_deepspeed_end.png`

**V100:**
- `v100_1b_1dev_nvtop.png` + `v100_1b_1dev_end.png`
- `v100_1b_4ddp_nvtop.png` + `v100_1b_4ddp_end.png`
- `v100_1b_4fsdp_nvtop.png` + `v100_1b_4fsdp_end.png`
- `v100_3b_4fsdp_nvtop.png` + `v100_3b_4fsdp_end.png`

> Populate `reports/results/summary.json` (auto‑generated by `scripts/utils/parse_litgpt_log.py`) to centralize timings and max memory per run.

---

## 🔍 Key Insights
| Technique         | Memory | Speed | Best Use Case                        |
|------------------|--------|-------|--------------------------------------|
| Reduced Batch     | ↓↓↓    | ↓     | Fit on limited VRAM                  |
| Gradient Accum.   | ↑      | ↑     | Simulate large global batch          |
| Reduced Precision | ↓↓↓    | ↑↑↑   | Best overall trade‑off               |
| Mixed Precision   | ↓↓     | ↑↑    | Balanced speed + numerical stability |
| DDP               | ↓/GPU  | ↑↑    | Parallelize large training           |
| FSDP              | ↓↓↓    | ↑     | Train bigger models on 4× GPU        |
| LoRA / QLoRA      | ↓↓↓↓   | ↑     | Efficient fine‑tuning of LLMs        |
| DeepSpeed Offload | ↓↓↓    | →/↓   | Free GPU mem; accept PCIe overhead   |

---

## 🔗 References
- **LitGPT:** https://github.com/Lightning-AI/litgpt  
- **Chameleon Cloud:** https://www.chameleoncloud.org/  
- **OpenLLaMA:** https://github.com/openlm-research/open_llama

---

## 📎 Reproducing the Chameleon Tutorial Flow
This repo tracks the tutorial’s sequence (leases, A100/V100 picks, LitGPT 0.5.7, Lightning < 2.5). See inline comments in `scripts/` for experiment parity.

---
## 📄 Embedded PDF Screenshots
_The PDFs you placed in `reports/screenshots/` are embedded below. If your Markdown viewer doesn’t render inline PDFs, the fallback links will still work._
> No PDFs were found in `reports/screenshots/`. Add files ending with `.pdf` and re-run this step.
