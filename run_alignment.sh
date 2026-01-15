#!/bin/bash

# --- STEP 1: INSTALL NVIDIA NEMO (If running on fresh cloud instance) ---
# pip install nemo_toolkit['all']

# --- STEP 2: CONVERT HUGGINGFACE LLAMA-3 TO NEMO FORMAT ---
# (You need a HuggingFace Token for Llama-3 access)
echo "Converting HF Llama-3 to NeMo format..."
python /opt/NeMo/scripts/checkpoint_converters/convert_llama_hf_to_nemo.py \
    --input_name_or_path="meta-llama/Meta-Llama-3-8B" \
    --output_path="/workspace/llama3_8b.nemo" \
    --precision="bf16"

# --- STEP 3: RUN DPO TRAINING (The Experiment) ---
echo "Starting DPO Training with Beta=0.1..."

# We use the 'neMo-aligner' training script
python /opt/NeMo-Aligner/examples/nlp/gpt/train_gpt_dpo.py \
    --config-path=. \
    --config-name=dpo_config.yaml \
    model.dpo.ref_policy_kl_penalty=0.1 \
    exp_manager.name="dpo_experiment_beta_0.1" \
    exp_manager.exp_dir="/workspace/results"