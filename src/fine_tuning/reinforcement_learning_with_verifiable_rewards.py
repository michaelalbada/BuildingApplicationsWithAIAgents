# rl_verifiable_reward_helpdesk.py
# Reinforcement fine-tuning (PPO) with an objective reward

import os, re, torch, random
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments)
from peft import LoraConfig, get_peft_model
from trl import PPOTrainer, PPOConfig

# 1️⃣  Base model & tokenizer ---------------------------------------------------
BASE = "microsoft/Phi-3-mini-4k-instruct"          # 3.8 B, MIT licence
tok  = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True, cache_dir="~/hf")

policy_base = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2️⃣  Add LoRA adapters for parameter-efficient RL ----------------------------
lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM"
)
policy = get_peft_model(policy_base, lora_cfg)

# 3️⃣  Toy dataset: prompt + correct label -------------------------------------
raw = [
    {"prompt": "Hi, I cannot access my invoice.",  "label": "billing"},
    {"prompt": "My login keeps failing.",          "label": "technical"},
    {"prompt": "Refund the overcharge on order X", "label": "billing"},
    {"prompt": "The site is very slow today.",     "label": "technical"},
]
ds = Dataset.from_list(raw)

# 4️⃣  Deterministic reward function -------------------------------------------
def reward_fn(pred: str, gold: str) -> float:
    """Return +1 if model output contains the correct label, else 0."""
    return float(gold in pred.lower())

# 5️⃣  PPO configuration --------------------------------------------------------
ppo_cfg = PPOConfig(
    batch_size     = 4,
    ppo_epochs     = 4,
    learning_rate  = 5e-6,
    init_kl_coef   = 0.05,     # keeps policy near the base model
)

train_args = TrainingArguments(
    output_dir      = "phi3-mini-helpdesk-ppo",
    per_device_train_batch_size = 4,
    logging_steps   = 5,
    save_strategy   = "no",
    bf16            = True,
)

trainer = PPOTrainer(
    config    = ppo_cfg,
    model     = policy,
    tokenizer = tok,
    dataset   = ds,
    args      = train_args
)

# 6️⃣  Main training loop -------------------------------------------------------
for epoch in range(3):
    # shuffle prompts each epoch for robustness
    samples = ds.shuffle(seed=epoch)
    queries, labels = zip(*[(s["prompt"], s["label"]) for s in samples])

    # tokenise prompts (padding to longest within batch)
    inputs = tok(list(queries), return_tensors="pt", padding=True).to(policy.device)

    # generate responses
    with torch.no_grad():
        generated = policy.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tok.eos_token_id,
        )
    responses = tok.batch_decode(
        generated[:, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    # compute rewards
    rewards = [reward_fn(resp, gold) for resp, gold in zip(responses, labels)]

    # PPO update
    trainer.step(queries=queries, responses=responses, rewards=rewards)

    print(f"Epoch {epoch+1} | mean reward: {sum(rewards)/len(rewards):.2f}")

# 7️⃣  Save the LoRA-adapted policy --------------------------------------------
policy.save_pretrained("phi3-mini-helpdesk-ppo")
tok.save_pretrained("phi3-mini-helpdesk-ppo")