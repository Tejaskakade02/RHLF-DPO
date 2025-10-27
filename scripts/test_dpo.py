# # scripts/test_dpo.py
# import os
# import torch
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from math import exp

# # -----------------------------
# # 1. Paths & Device
# # -----------------------------
# BASE_MODEL_PATH = "models/policy"
# DPO_MODEL_PATH = "models/dpo"

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"🔥 Using device: {device}")

# # -----------------------------
# # 2. Load models & tokenizer
# # -----------------------------
# print("📦 Loading tokenizer and models...")
# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH).to(device)
# dpo_model = AutoModelForCausalLM.from_pretrained(DPO_MODEL_PATH).to(device)
# print("✅ All models loaded successfully!\n")

# # -----------------------------
# # 3. Helper functions
# # -----------------------------
# def generate_response(model, prompt, max_new_tokens=120):
#     """Generate text from a model given a prompt."""
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=max_new_tokens,
#         temperature=0.7,
#         top_p=0.9,
#         do_sample=True,
#         pad_token_id=tokenizer.eos_token_id
#     )
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return response[len(prompt):].strip()

# def compute_confidence(model, text):
#     """Compute a simple normalized confidence score based on token likelihood."""
#     inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
#     with torch.no_grad():
#         outputs = model(**inputs, labels=inputs["input_ids"])
#     loss = outputs.loss.item()
#     confidence = exp(-loss)  # higher = more confident
#     return confidence

# # -----------------------------
# # 4. Test prompts
# # -----------------------------
# prompts = [
#     "Explain what reinforcement learning is in simple terms.",
#     "Write a short inspirational message for students preparing for exams.",
#     "List three benefits of renewable energy and explain one in detail.",
#     "Describe how AI models can learn from human feedback.",
#     "Give advice on how to stay productive while working remotely."
# ]

# print("🚀 Running DPO Validation Test...\n")

# # -----------------------------
# # 5. Compare Models
# # -----------------------------
# for i, prompt in enumerate(prompts, 1):
#     print(f"🧠 Prompt {i}: {prompt}\n")

#     # Base model generation
#     base_out = generate_response(base_model, prompt)
#     base_conf = compute_confidence(base_model, prompt + " " + base_out)

#     # DPO model generation
#     dpo_out = generate_response(dpo_model, prompt)
#     dpo_conf = compute_confidence(dpo_model, prompt + " " + dpo_out)

#     # Print comparison
#     print("💬 Base Model Response:")
#     print(base_out)
#     print(f"📈 Confidence: {base_conf:.4f}\n")

#     print("🔥 DPO Fine-Tuned Model Response:")
#     print(dpo_out)
#     print(f"📈 Confidence: {dpo_conf:.4f}\n")

#     # Summary
#     if dpo_conf > base_conf:
#         print("✅ DPO model shows higher confidence and likely better alignment.")
#     else:
#         print("⚠️ DPO model not significantly more confident on this prompt.")
#     print("-" * 100)

# print("\n✅ Test completed! Compare outputs and confidence scores above.\n")


# scripts/test_dpo.py
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from math import exp

# -----------------------------
# 1. Paths & Device
# -----------------------------
BASE_MODEL_PATH = "models/policy"
DPO_MODEL_PATH = "models/dpo"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔥 Using device: {device}")

# -----------------------------
# 2. Load models & tokenizer
# -----------------------------
print("📦 Loading tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH).to(device)
dpo_model = AutoModelForCausalLM.from_pretrained(DPO_MODEL_PATH).to(device)
print("✅ All models loaded successfully!\n")

# -----------------------------
# 3. Helper functions
# -----------------------------
def generate_response(model, prompt, max_new_tokens=120):
    """Generate text from a model given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

def compute_confidence(model, text):
    """Compute a simple normalized confidence score based on token likelihood."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()
    confidence = exp(-loss)  # higher = more confident
    return confidence

# -----------------------------
# 4. Test prompts
# -----------------------------
prompts = [
    "Explain what reinforcement learning is in simple terms.",
    "Write a short inspirational message for students preparing for exams.",
    "List three benefits of renewable energy and explain one in detail.",
    "Describe how AI models can learn from human feedback.",
    "Give advice on how to stay productive while working remotely."
]

print("🚀 Running DPO Validation Test...\n")

# -----------------------------
# 5. Compare Models
# -----------------------------
for i, prompt in enumerate(prompts, 1):
    print(f"🧠 Prompt {i}: {prompt}\n")

    base_out = generate_response(base_model, prompt)
    base_conf = compute_confidence(base_model, prompt + " " + base_out)

    dpo_out = generate_response(dpo_model, prompt)
    dpo_conf = compute_confidence(dpo_model, prompt + " " + dpo_out)

    print("💬 Base Model Response:")
    print(base_out)
    print(f"📈 Confidence: {base_conf:.4f}\n")

    print("🔥 DPO Fine-Tuned Model Response:")
    print(dpo_out)
    print(f"📈 Confidence: {dpo_conf:.4f}\n")

    if dpo_conf > base_conf:
        print("✅ DPO model shows higher confidence and likely better alignment.")
    else:
        print("⚠️ DPO model not significantly more confident on this prompt.")
    print("-" * 100)

print("\n✅ Test completed! Compare outputs and confidence scores above.\n")

# -----------------------------
# 6. Interactive Chat Mode
# -----------------------------
def chat_with_models():
    """Simple interactive chat with option to switch between base and DPO models."""
    print("\n💬 Interactive Chat Mode Activated!")
    print("Type 'switch' to toggle between Base and DPO model.")
    print("Type 'exit' to quit the chat.\n")

    current_model = dpo_model
    current_name = "DPO"

    while True:
        user_input = input(f"🧑 You ({current_name} mode): ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting chat. Goodbye!")
            break
        elif user_input.lower() in ["switch", "swap"]:
            if current_model == dpo_model:
                current_model = base_model
                current_name = "Base"
            else:
                current_model = dpo_model
                current_name = "DPO"
            print(f"🔁 Switched to {current_name} model.\n")
            continue

        response = generate_response(current_model, user_input)
        conf = compute_confidence(current_model, user_input + " " + response)

        print(f"\n🤖 {current_name} Model:\n{response}")
        print(f"📈 Confidence: {conf:.4f}\n")

# Uncomment to start chat after testing
# chat_with_models()
