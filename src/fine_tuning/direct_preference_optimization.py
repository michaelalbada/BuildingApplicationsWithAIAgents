# fine_tune_helpdesk_dpo.py
import torch, os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer
import logging

BASE_SFT_CKPT = "microsoft/Phi-3-mini-4k-instruct"
DPO_DATA      = "training_data/dpo_it_help_desk_training_data.jsonl"                   # -> path or HF dataset
OUTPUT_DIR    = "phi3-mini-helpdesk-dpo"

# 1️⃣ Model + tokenizer
tok = AutoTokenizer.from_pretrained(BASE_SFT_CKPT, padding_side="right",
                                    trust_remote_code=True)

logger = logging.getLogger(__name__)
if not os.path.exists(BASE_SFT_CKPT):
    logger.warning("Local path not found; will attempt to download '%s' from the Hub.", BASE_SFT_CKPT)

base = AutoModelForCausalLM.from_pretrained(
    BASE_SFT_CKPT,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True
)

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
   target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                    "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base, lora_cfg)
print("✅  Phi-3 loaded:", mdl.config.hidden_size, "hidden dim")

ds = load_dataset("json", data_files=DPO_DATA, split="train")

# 4️⃣ Trainer
train_args = TrainingArguments(
    output_dir      = OUTPUT_DIR,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    learning_rate   = 5e-6,
    num_train_epochs= 3,
    fp16            = True,
    logging_steps   = 10,
    save_strategy   = "epoch",
    bf16            = True,
    report_to       = "tensorboard"
)

dpo_cfg = DPOConfig( beta=0.1 )

trainer = DPOTrainer(
    model,
    ref_model=None,
    args=train_args,
    train_dataset=ds,
    tokenizer=tok,
    dpo_config=dpo_cfg,
)

trainer.train()
trainer.save_model()
tok.save_pretrained(OUTPUT_DIR)
