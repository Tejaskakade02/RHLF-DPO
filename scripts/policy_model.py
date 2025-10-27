# scripts/policy_model.py
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
import os, json, torch
from tqdm import tqdm

# Paths
DATA_PROCESSED_PATH = "data/processed"
MODEL_SAVE_PATH = "models/policy"
os.makedirs(DATA_PROCESSED_PATH, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Config — choose one:
model_name = "EleutherAI/pythia-410m"  # smaller option
# model_name = "gpt2-medium"           # or use this

# Load dataset (5000 samples)
print("Loading dataset...")
dataset = load_dataset("yahma/alpaca-cleaned")
dataset["train"] = dataset["train"].select(range(min(5000, len(dataset["train"]))))

# Tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Format examples using requested prompt template
def format_data(example):
    prompt = f"### Instruction:\n{example['instruction']}\n### Response:\n{example['output']}"
    return {"text": prompt, "instruction": example["instruction"], "response": example["output"]}

dataset = dataset.map(format_data)

# Tokenize and mask prompt tokens in labels
max_length = 256

def tokenize_and_make_labels(example):
    split_marker = "### Response:\n"
    full_text = example["text"]
    if split_marker in full_text:
        prompt_text, _ = full_text.split(split_marker, 1)
        prompt_text = prompt_text + split_marker
    else:
        prompt_text = ""

    prompt_ids = tokenizer(prompt_text, truncation=False, padding=False)["input_ids"]
    enc = tokenizer(full_text, truncation=True, padding="max_length", max_length=max_length)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    labels = input_ids.copy()
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = -100

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

tokenized = dataset["train"].map(tokenize_and_make_labels, batched=False, remove_columns=dataset["train"].column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Training args (requested values)
training_args = TrainingArguments(
    output_dir=MODEL_SAVE_PATH,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_total_limit=1,
    logging_steps=50,
    learning_rate=5e-5,
    report_to="none",
    fp16=torch.cuda.is_available()
)

model = AutoModelForCausalLM.from_pretrained(model_name)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator
)

print("Starting supervised fine-tuning (SFT)...")
trainer.train()

trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print(f"✅ Policy model saved at {MODEL_SAVE_PATH}")

# Generate accepted/rejected candidate pairs
print("Generating candidate responses for differentiator stage...")
accepted, rejected = [], []
gen_count = min(300, len(dataset["train"]))  # you can increase

for sample in tqdm(dataset["train"].select(range(gen_count))):
    instr = sample["instruction"]
    formatted = f"### Instruction:\n{instr}\n### Response:\n"
    input_ids = tokenizer.encode(formatted, return_tensors="pt").to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=80,
        num_return_sequences=2,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
    def extract_resp(full_text):
        if "### Response:" in full_text:
            return full_text.split("### Response:")[-1].strip()
        return full_text.strip()

    resp0 = extract_resp(decoded[0])
    resp1 = extract_resp(decoded[1])

    if len(resp0) >= len(resp1):
        acc, rej = resp0, resp1
    else:
        acc, rej = resp1, resp0

    accepted.append({"prompt": instr, "response": acc, "label": "accepted"})
    rejected.append({"prompt": instr, "response": rej, "label": "rejected"})

# Save
with open(os.path.join(DATA_PROCESSED_PATH, "accepted_data.jsonl"), "w", encoding="utf-8") as fa:
    for e in accepted:
        fa.write(json.dumps(e, ensure_ascii=False) + "\n")
with open(os.path.join(DATA_PROCESSED_PATH, "rejected_data.jsonl"), "w", encoding="utf-8") as fr:
    for e in rejected:
        fr.write(json.dumps(e, ensure_ascii=False) + "\n")

print("✅ Saved accepted_data.jsonl and rejected_data.jsonl.")
