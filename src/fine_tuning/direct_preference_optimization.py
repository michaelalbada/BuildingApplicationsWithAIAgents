# fine_tune_helpdesk_dpo.py
import torch, os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
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

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,               # quantize weights to 4 bits :contentReference[oaicite:0]{index=0}
    bnb_4bit_use_double_quant=True,  # optional: nested quantization :contentReference[oaicite:1]{index=1}
    bnb_4bit_compute_dtype=torch.bfloat16
)

base = AutoModelForCausalLM.from_pretrained(
    BASE_SFT_CKPT,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
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
print("✅  Phi-3 loaded:", model.config.hidden_size, "hidden dim")

ds = load_dataset("json", data_files=DPO_DATA, split="train")

# 4️⃣ Trainer
train_args = TrainingArguments(
    output_dir      = OUTPUT_DIR,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    learning_rate   = 5e-6,
    num_train_epochs= 3,
    logging_steps   = 10,
    save_strategy   = "epoch",
    bf16            = True,
    report_to       = None,
)

dpo_args = DPOConfig(
    output_dir              = "phi3-mini-helpdesk-dpo",
    per_device_train_batch_size  = 4,
    gradient_accumulation_steps  = 4,
    learning_rate           = 5e-6,
    num_train_epochs        = 3.0,
    bf16                    = True,
    logging_steps           = 10,
    save_strategy           = "epoch",
    report_to               = None,
    beta                    = 0.1,
    loss_type               = "sigmoid",
    label_smoothing         = 0.0,
    max_prompt_length       = 4096,
    max_completion_length   = 4096,
    max_length              = 8192,
    padding_value           = tok.pad_token_id,
    label_pad_token_id      = tok.pad_token_id,
    truncation_mode         = "keep_end",
    generate_during_eval    = False,
    disable_dropout         = False,
    reference_free          = True,
    model_init_kwargs       = None,
    ref_model_init_kwargs   = None,
)

trainer = DPOTrainer(
    model,
    ref_model=None,
    args=dpo_args,
    train_dataset=ds,
)

trainer.train()
trainer.save_model()
tok.save_pretrained(OUTPUT_DIR)
