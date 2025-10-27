# scripts/reward_model.py
import os, json, torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW

DATA_PATH = "data/processed"
MODEL_SAVE_PATH = "models/reward"
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

accepted_file = os.path.join(DATA_PATH, "accepted_data.jsonl")
rejected_file = os.path.join(DATA_PATH, "rejected_data.jsonl")

if not os.path.exists(accepted_file) or not os.path.exists(rejected_file):
    raise FileNotFoundError("Missing accepted_data.jsonl or rejected_data.jsonl — run policy_model.py first.")

with open(accepted_file, "r", encoding="utf-8") as fa:
    accepted = [json.loads(line) for line in fa.readlines()]
with open(rejected_file, "r", encoding="utf-8") as fr:
    rejected = [json.loads(line) for line in fr.readlines()]

print(f"Loaded {len(accepted)} accepted and {len(rejected)} rejected samples.")

class RewardDataset(Dataset):
    def __init__(self, accepted, rejected, tokenizer, max_length=256):
        self.data = list(zip(accepted, rejected))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        acc, rej = self.data[idx]
        prompt = acc["prompt"]
        acc_text = f"{prompt}\n{acc['response']}"
        rej_text = f"{prompt}\n{rej['response']}"

        acc_tokens = self.tokenizer(acc_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        rej_tokens = self.tokenizer(rej_text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        acc_tokens = {k: v.squeeze(0) for k, v in acc_tokens.items()}
        rej_tokens = {k: v.squeeze(0) for k, v in rej_tokens.items()}
        return acc_tokens, rej_tokens

base_model = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModel.from_pretrained(base_model).to(device)
reward_head = torch.nn.Linear(model.config.hidden_size, 1).to(device)
optimizer = AdamW(list(model.parameters()) + list(reward_head.parameters()), lr=1e-5)

dataset = RewardDataset(accepted, rejected, tokenizer)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

print("Training reward model...")
for epoch in range(2):
    total_loss = 0
    for acc_tokens, rej_tokens in tqdm(loader, desc=f"Epoch {epoch+1}"):
        acc_tokens = {k: v.to(device) for k, v in acc_tokens.items()}
        rej_tokens = {k: v.to(device) for k, v in rej_tokens.items()}

        acc_emb = model(**acc_tokens).last_hidden_state[:, 0, :]
        rej_emb = model(**rej_tokens).last_hidden_state[:, 0, :]
        acc_reward = reward_head(acc_emb)
        rej_reward = reward_head(rej_emb)

        loss = -torch.log(torch.sigmoid(acc_reward - rej_reward)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(loader):.4f}")

torch.save({
    "model_state_dict": model.state_dict(),
    "reward_head_state_dict": reward_head.state_dict(),
    "tokenizer": base_model
}, os.path.join(MODEL_SAVE_PATH, "reward_model.pt"))

print(f"Saved reward model at {MODEL_SAVE_PATH}/reward_model.pt")
