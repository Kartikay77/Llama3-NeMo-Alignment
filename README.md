# Llama-3 Alignment Pipeline using NVIDIA NeMo & DPO

## 🚀 Project Overview
This repository contains a production-grade **RLHF (Reinforcement Learning from Human Feedback)** pipeline designed to align **Llama-3-8B** using **Direct Preference Optimization (DPO)** within the **NVIDIA NeMo Framework**.

The pipeline is engineered for efficiency, utilizing **LoRA (Low-Rank Adaptation)** and **BF16 mixed-precision** to enable fine-tuning on constrained compute environments while maintaining model performance.

## 🛠️ Key Technical Features
* **Framework:** NVIDIA NeMo (NeMo-Aligner)
* **Algorithm:** Direct Preference Optimization (DPO) - Stable alignment without a reward model.
* **Efficiency:** PEFT/LoRA integration to reduce trainable parameters by ~98%.
* **Infrastructure:** Configured for Multi-GPU scaling via PyTorch Lightning.

## 🔬 Experimental Design: Hyperparameter Ablation
This project includes configuration for an ablation study on the **KL-Divergence Penalty (`beta`)**, a critical hyperparameter in DPO that controls the drift from the reference model.

| Experiment | Beta Value | Hypothesis |
| :--- | :--- | :--- |
| **Exp A** | `0.1` | **Balanced:** Expected optimal trade-off between instruction-following and diversity. |
| **Exp B** | `0.5` | **Conservative:** Higher penalty keeps the model too close to base, potentially reducing helpfulness. |
| **Exp C** | `0.05` | **Unstable:** Lower penalty allows drift, increasing the risk of "reward hacking" or hallucinations. |

## 📂 Project Structure
* `data_prep/`: Scripts to format Anthropic-HH dataset into NeMo JSONL format.
* `configs/`: Hydra-based YAML configurations for DPO training and LoRA adapters.
* `scripts/`: Bash execution scripts for checkpoint conversion and training jobs.

## 💻 Usage
### 1. Prepare Data
```bash
python data_prep/prepare_data.py
```
### 2. Convert Llama-3 Checkpoint

Converts HuggingFace weights to .nemo format.
```
python scripts/convert_checkpoint.py
```

### 3. Run DPO Training
```
# Run with Beta=0.1 (Optimized)
bash scripts/run_dpo.sh
```

## Built to demonstrate competency in LLM Post-Training, NVIDIA AI Stack, and Systems Engineering.
---

#### **2. `dpo_config.yaml` (The "Brains")**
This proves you understand NeMo's configuration system.

```yaml
name: llama3_dpo_beta_0.1

trainer:
  devices: 1
  num_nodes: 1
  accelerator: gpu
  precision: bf16
  max_epochs: 1
  log_every_n_steps: 10
  val_check_interval: 0.1
  enable_checkpointing: True

model:
  restore_from_path: "/checkpoints/llama3_8b_base.nemo"
  
  # LoRA Configuration (Parameter Efficient Fine-Tuning)
  peft:
    peft_scheme: "lora"
    lora_tuning:
      adapter_dim: 32   # Rank
      alpha: 16         # Scaling
      dropout: 0.05
      target_modules: ['gate_proj', 'o_proj', 'k_proj', 'q_proj', 'up_proj', 'v_proj', 'down_proj']

  # DPO Configuration
  dpo:
    ref_policy_kl_penalty: 0.1  # The Beta Hyperparameter
    loss_type: "dpo"
    log_prob_forward_micro_batch_size: 1

  data:
    train_ds:
      file_path: "data/dpo_train.jsonl"
      batch_size: 1
      shuffle: true
      max_seq_length: 2048
      num_workers: 4

  optim:
    name: fused_adam
    lr: 5e-6
    weight_decay: 0.01
    sched:
      name: CosineAnnealing
      warmup_steps: 100
```
## 4. run_dpo.sh (The Execution Logic)
```
#!/bin/bash

# Configuration
PROJECT_DIR=$(pwd)
DATA_DIR="$PROJECT_DIR/data"
CONFIG_PATH="$PROJECT_DIR/configs"

echo "--- Starting NeMo DPO Training Pipeline ---"

# Step 1: Ensure Data Exists
if [ ! -f "$DATA_DIR/dpo_train.jsonl" ]; then
    echo "Data not found. Running preparation script..."
    python data_prep/prepare_data.py
fi

# Step 2: Run Training (Example command for Slurm/Cluster)
# Note: This command assumes a NeMo Docker container environment
python /opt/NeMo-Aligner/examples/nlp/gpt/train_gpt_dpo.py \
    --config-path=$CONFIG_PATH \
    --config-name=dpo_config.yaml \
    model.dpo.ref_policy_kl_penalty=0.1 \
    exp_manager.name="llama3_dpo_run_beta_0.1" \
    exp_manager.exp_dir="$PROJECT_DIR/results"

echo "Training job submitted."
```
