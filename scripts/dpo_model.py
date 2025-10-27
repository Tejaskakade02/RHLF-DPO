# # scripts/dpo_model.py
# import os
# import json
# import torch
# from tqdm import tqdm
# from datasets import Dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from trl import DPOTrainer, DPOConfig

# # -----------------------------
# # 1. Paths
# # -----------------------------
# POLICY_MODEL_PATH = "models/policy"
# DATA_PATH = "data/processed"
# OUTPUT_PATH = "models/dpo"
# os.makedirs(OUTPUT_PATH, exist_ok=True)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"🔥 Using device: {device}")

# # -----------------------------
# # 2. Load models
# # -----------------------------
# model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_PATH).to(device)
# ref_model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_PATH).to(device)

# tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_PATH)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # -----------------------------
# # 3. Load accepted/rejected pairs
# # -----------------------------
# def load_jsonl(path):
#     with open(path, "r", encoding="utf-8") as f:
#         return [json.loads(line) for line in f]

# accepted = load_jsonl(os.path.join(DATA_PATH, "accepted_data.jsonl"))
# rejected = load_jsonl(os.path.join(DATA_PATH, "rejected_data.jsonl"))

# pairs = []
# for acc, rej in zip(accepted, rejected):
#     if acc["prompt"] == rej["prompt"]:
#         pairs.append({
#             "prompt": acc["prompt"],
#             "chosen": acc["response"],
#             "rejected": rej["response"]
#         })

# print(f"✅ Loaded {len(pairs)} preference pairs for DPO training.")
# dataset = Dataset.from_list(pairs)

# # -----------------------------
# # 4. Tokenize
# # -----------------------------
# max_length = 512

# def tokenize_fn(batch):
#     chosen_enc = tokenizer(batch["prompt"] + "\n" + batch["chosen"],
#                            truncation=True, padding="max_length", max_length=max_length)
#     rejected_enc = tokenizer(batch["prompt"] + "\n" + batch["rejected"],
#                              truncation=True, padding="max_length", max_length=max_length)
#     return {
#         "prompt": batch["prompt"],
#         "chosen_input_ids": chosen_enc["input_ids"],
#         "chosen_attention_mask": chosen_enc["attention_mask"],
#         "rejected_input_ids": rejected_enc["input_ids"],
#         "rejected_attention_mask": rejected_enc["attention_mask"],
#     }

# tokenized_dataset = dataset.map(tokenize_fn, batched=False)

# # -----------------------------
# # 5. DPO Config
# # -----------------------------
# dpo_config = DPOConfig(
#     output_dir=OUTPUT_PATH,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     learning_rate=1e-5,
#     num_train_epochs=1,
#     logging_steps=50,
#     save_steps=200,
#     warmup_ratio=0.1,
#     bf16=False,   # disable bf16 for CPU
#     fp16=False,   # disable fp16 for CPU
#     remove_unused_columns=False,
#     report_to="none"
# )

# # -----------------------------
# # 6. DPO Trainer
# # -----------------------------
# trainer = DPOTrainer(
#     model=model,
#     ref_model=ref_model,
#     args=dpo_config,
#     train_dataset=tokenized_dataset,
# )

# # -----------------------------
# # 7. Train & Save
# # -----------------------------
# print("🚀 Starting DPO Fine-Tuning...")
# trainer.train()

# trainer.save_model(OUTPUT_PATH)
# tokenizer.save_pretrained(OUTPUT_PATH)
# print(f"✅ DPO fine-tuned model saved at {OUTPUT_PATH}")


# scripts/dpo_model.py (Enhanced Training Config)
import os
import json
import torch
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

# -----------------------------
# 1. Paths
# -----------------------------
POLICY_MODEL_PATH = "models/policy"
DATA_PATH = "data/processed"
OUTPUT_PATH = "models/dpo"
os.makedirs(OUTPUT_PATH, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

# -----------------------------
# 2. Load models
# -----------------------------
print("📦 Loading tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained(POLICY_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_PATH).to(device)
ref_model = AutoModelForCausalLM.from_pretrained(POLICY_MODEL_PATH).to(device)
print("✅ Models loaded successfully!")

# -----------------------------
# 3. Load accepted/rejected pairs
# -----------------------------
def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

accepted = load_jsonl(os.path.join(DATA_PATH, "accepted_data.jsonl"))
rejected = load_jsonl(os.path.join(DATA_PATH, "rejected_data.jsonl"))

pairs = []
for acc, rej in zip(accepted, rejected):
    if acc["prompt"] == rej["prompt"]:
        pairs.append({
            "prompt": acc["prompt"],
            "chosen": acc["response"],
            "rejected": rej["response"]
        })

print(f"✅ Loaded {len(pairs)} preference pairs for DPO training.")
dataset = Dataset.from_list(pairs)

# Shuffle the dataset for better learning stability
dataset = dataset.shuffle(seed=42)

# -----------------------------
# 4. Tokenization
# -----------------------------
max_length = 512

def tokenize_fn(batch):
    chosen_enc = tokenizer(
        batch["prompt"] + "\n" + batch["chosen"],
        truncation=True, padding="max_length", max_length=max_length
    )
    rejected_enc = tokenizer(
        batch["prompt"] + "\n" + batch["rejected"],
        truncation=True, padding="max_length", max_length=max_length
    )
    return {
        "prompt": batch["prompt"],
        "chosen_input_ids": chosen_enc["input_ids"],
        "chosen_attention_mask": chosen_enc["attention_mask"],
        "rejected_input_ids": rejected_enc["input_ids"],
        "rejected_attention_mask": rejected_enc["attention_mask"],
    }

tokenized_dataset = dataset.map(tokenize_fn, batched=False)
print("🧩 Tokenization complete!")

# -----------------------------
# 5. DPO Config (Improved)
# -----------------------------
dpo_config = DPOConfig(
    output_dir=OUTPUT_PATH,
    per_device_train_batch_size=2,           # ↑ Slightly larger batch for smoother updates
    gradient_accumulation_steps=4,
    learning_rate=5e-6,                      # ↓ Lower LR for stability
    num_train_epochs=3,                      # ↑ Multi-epoch fine-tuning
    warmup_ratio=0.15,                       # Gradual warmup for stable start
    save_strategy="epoch",                   # Save after each epoch
    logging_steps=20,
    bf16=False,
    fp16=False,
    remove_unused_columns=False,
    report_to="none",
)

# -----------------------------
# 6. DPO Trainer
# -----------------------------
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=tokenized_dataset,
)

# -----------------------------
# 7. Train & Save
# -----------------------------
print("🚀 Starting Enhanced DPO Fine-Tuning...")
train_result = trainer.train()

print("✅ Training finished!")
trainer.save_model(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"✅ DPO fine-tuned model saved at {OUTPUT_PATH}")
print(f"📊 Final Training Loss: {train_result.metrics['train_loss']:.4f}")
