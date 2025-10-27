# 🧠 RLHF Training Pipeline with Direct Preference Optimization (DPO)

This project implements a **Reinforcement Learning with Human Feedback (RLHF)** pipeline using **Direct Preference Optimization (DPO)** instead of PPO, consisting of:

1. **Policy Model** — fine-tuned on instructions
2. **Reward Model** — learns to prefer better responses
3. **DPO Fine-tuning** — aligns the policy model using human preference pairs
4. **Testing** — to validate and chat with the final DPO model

Built entirely with **PyTorch** + **Hugging Face Transformers** 🚀

---

## 🗂️ Folder Structure

```
RLHF-DPO Project/
│
├── data/
│   ├── raw/                     # Downloaded raw datasets (Yahma/Alpaca-Cleaned)
│   ├── processed/
│   │   ├── accepted_data.jsonl  # Human-approved (good) responses
│   │   ├── rejected_data.jsonl  # Human-rejected (bad) responses
│   │   ├── preference_data.jsonl # Combined preference pairs for DPO
│
├── models/
│   ├── policy/                  # Fine-tuned base model
│   ├── reward/                  # Trained reward model checkpoint
│   ├── dpo/                     # DPO fine-tuned model
│
├── scripts/
│   ├── policy_model.py          # Step 1: Train policy model
│   ├── reward_model.py          # Step 2: Train reward model
│   ├── dpo_model.py             # Step 3: DPO fine-tuning
│   ├── test_dpo_model.py        # Step 4: Test DPO model
│
├── requirements.txt
├── .venv/
└── README.md
```

---

## ⚙️ Environment Setup

### 1️⃣ Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it:

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`:**

```
torch
transformers
datasets
tqdm
trl
```

---

## 🧩 Data Setup

Before running the training scripts, you must create the folders and download the dataset.

### 1️⃣ Create Folders

```bash
mkdir -p data/raw data/processed models/policy models/reward models/dpo scripts
```

### 2️⃣ Download Dataset (Yahma/Alpaca-Cleaned)

This dataset will serve as the base for policy and reward model training.

Make sure you have **Git LFS** installed:

```bash
git lfs install
```

Then, download the dataset into the `data/raw` directory:

```bash
cd data/raw
git clone https://huggingface.co/datasets/yahma/alpaca-cleaned
cd ../../
```

After this step, your `data/raw/alpaca-cleaned` folder will contain the original instruction–response pairs.

---

## 🧠 RLHF Training Flow (DPO-Based)

### 🟢 Step 1: Train Policy Model

Fine-tune the base model (like GPT-2) on the Alpaca dataset.

```bash
python scripts/policy_model.py
```

➡️ Output: `models/policy/`

---

### 🟡 Step 2: Train Reward Model

Train a DistilBERT-based reward model on **accepted vs rejected** responses.

```bash
python scripts/reward_model.py
```

➡️ Output: `models/reward/reward_model.pt`

---

### 🔴 Step 3: DPO Fine-Tuning

Perform **Direct Preference Optimization (DPO)** using the trained policy model and preference pairs.

```bash
python scripts/dpo_model.py
```

➡️ Output: `models/dpo/`

---

### 🧪 Step 4: Test DPO Model

Interactively test or evaluate the fine-tuned DPO model.

```bash
python scripts/test_dpo_model.py
```

🧠 Example Output:

```
Prompt: Explain reinforcement learning simply.
Response: Reinforcement learning is when an AI learns from feedback on which actions work better.
```

---

## 🧪 Optional: Run All Steps in Sequence

To automate the full DPO-based RLHF flow:

```bash
python scripts/policy_model.py && \
python scripts/reward_model.py && \
python scripts/dpo_model.py && \
python scripts/test_dpo_model.py
```

---

## ⚡ GPU Check

Ensure CUDA is available before training:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `True`, GPU training is enabled ✅

---

## 🏁 Summary

| Step | Script              | Description           | Output                          |
| ---- | ------------------- | --------------------- | ------------------------------- |
| 1️⃣  | `policy_model.py`   | Fine-tunes base LLM   | `models/policy/`                |
| 2️⃣  | `reward_model.py`   | Trains reward scorer  | `models/reward/reward_model.pt` |
| 3️⃣  | `dpo_model.py`      | DPO fine-tuning       | `models/dpo/`                   |
| 4️⃣  | `test_dpo_model.py` | Chat & test DPO model | Console output                  |

---

## ❤️ Credits

Built using:

* [PyTorch](https://pytorch.org/)
* [Hugging Face Transformers](https://huggingface.co/transformers)
* [Yahma/Alpaca-Cleaned Dataset](https://huggingface.co/datasets/yahma/alpaca-cleaned)
* [TRL (Hugging Face)](https://huggingface.co/docs/trl)
* DPO concept inspired by [Direct Preference Optimization (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290)
