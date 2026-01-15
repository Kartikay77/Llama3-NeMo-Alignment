import json
from datasets import load_dataset

# 1. Download Anthropic's Helpful/Harmless dataset (standard for DPO)
print("Downloading dataset...")
dataset = load_dataset("anthropic/hh-rlhf", split="train[:5000]") # Small subset for demo

# 2. Format for NVIDIA NeMo
# NeMo expects: {"prompt": "...", "chosen_response": "...", "rejected_response": "..."}
formatted_data = []

print("Formatting data...")
for item in dataset:
    # The dataset comes as "chosen" and "rejected" full conversation strings.
    # We need to split the last assistant response from the prompt.
    
    # Simple splitting logic (Anthropic data usually ends with "Assistant: ")
    chosen_split = item["chosen"].rpartition("\n\nAssistant:")
    rejected_split = item["rejected"].rpartition("\n\nAssistant:")
    
    prompt = chosen_split[0] + "\n\nAssistant:"
    chosen_response = chosen_split[2]
    rejected_response = rejected_split[2]

    entry = {
        "prompt": prompt,
        "chosen_response": chosen_response, 
        "rejected_response": rejected_response
    }
    formatted_data.append(entry)

# 3. Save as JSONL (JSON Lines)
with open("dpo_train.jsonl", "w") as f:
    for entry in formatted_data:
        json.dump(entry, f)
        f.write("\n")

print(f"Saved {len(formatted_data)} samples to dpo_train.jsonl")