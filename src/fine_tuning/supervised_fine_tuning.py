# fine_tune_function_calling.py
"""A clean, modular script to fine‑tune an LLM for function‑calling with LoRA.

This script consolidates the steps from the Bonus Unit notebook into a single
Python entry‑point.  Usage (basic):

    HF_TOKEN=<your_token> python fine_tune_function_calling.py \
        --model google/gemma-2-2b-it \
        --dataset Jofthomas/hermes-function-calling-thinking-V1 \
        --output_dir gemma-2-2B-function-call-ft

See `--help` for all options.
"""
from __future__ import annotations

import argparse
import os
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, PeftConfig, PeftModel, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

###############################################################################
# Special tokens & chat template helpers
###############################################################################

class ChatmlSpecialTokens(str, Enum):
    tools = "<tools>"
    eotools = "</tools>"
    think = "<think>"
    eothink = "</think>"
    tool_call = "<tool_call>"
    eotool_call = "</tool_call>"
    tool_response = "<tool_response>"
    eotool_response = "</tool_response>"
    pad_token = "<pad>"
    eos_token = "<eos>"

    @classmethod
    def list(cls) -> List[str]:
        return [c.value for c in cls]

CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}"
    "{% for message in messages %}"
    "{{ '<start_of_turn>' + message['role'] + '\n' + message['content']|trim + '<end_of_turn><eos>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
)

###############################################################################
# Dataset preprocessing
###############################################################################

def _merge_system_into_first_user(messages: List[Dict[str, str]]) -> None:
    """Merges a leading system message into the subsequent user message."""
    if messages and messages[0]["role"] == "system":
        system_content = messages[0]["content"]
        messages.pop(0)
        if not messages or messages[0]["role"] != "human":
            raise ValueError("Expected a user message after system message.")
        messages[0][
            "content"
        ] = (
            f"{system_content}Also, before making a call to a function take the time "
            "to plan the function to take. Make that thinking process between "
            "<think>{your thoughts}</think>\n\n" + messages[0]["content"]
        )


def build_preprocess_fn(tokenizer):
    """Returns a function that maps raw samples to tokenized prompts."""
    def _preprocess(sample):
        messages = sample["messages"].copy()
        _merge_system_into_first_user(messages)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": prompt}

    return _preprocess


def load_and_prepare_dataset(ds_name: str, tokenizer, max_train: int, max_eval: int) -> DatasetDict:
    """Loads the dataset and applies preprocessing & train/test split."""
    raw = load_dataset(ds_name).rename_column("conversations", "messages")
    processed = raw.map(build_preprocess_fn(tokenizer), remove_columns="messages")
    split = processed["train"].train_test_split(test_size=0.1, seed=42)
    split["train"] = split["train"].select(range(max_train))
    split["test"] = split["test"].select(range(max_eval))
    return split

###############################################################################
# Model & Tokenizer helpers
###############################################################################

def build_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        pad_token=ChatmlSpecialTokens.pad_token.value,
        additional_special_tokens=ChatmlSpecialTokens.list(),
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


def build_model(model_name: str, tokenizer, load_4bit: bool = False):
    kwargs = {
        "attn_implementation": "eager",
        "device_map": "auto",
    }
    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.resize_token_embeddings(len(tokenizer))
    return model

###############################################################################
# PEFT / LoRA helpers
###############################################################################

def build_lora_config(r: int = 16, alpha: int = 64, dropout: float = 0.05) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=[
            "gate_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "lm_head",
            "embed_tokens",
        ],
        task_type=TaskType.CAUSAL_LM,
    )

###############################################################################
# Training
###############################################################################

def train(
    model,
    tokenizer,
    dataset: DatasetDict,
    peft_cfg: LoraConfig,
    output_dir: str,
    epochs: int = 1,
    lr: float = 1e-4,
    batch_size: int = 1,
    grad_accum: int = 4,
    max_seq_len: int = 1500,
):
    train_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        save_strategy="no",
        eval_strategy="epoch",
        logging_steps=5,
        learning_rate=lr,
        num_train_epochs=epochs,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        report_to=None,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=True,
        max_seq_length=max_seq_len,
    )

    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )

    trainer.train()
    trainer.save_model()
    return trainer

###############################################################################
# CLI
###############################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine‑tune an LLM for function calling with LoRA.")
    parser.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct", help="Base model name or path")
    parser.add_argument("--dataset", default="Jofthomas/hermes-function-calling-thinking-V1", help="HF dataset")
    parser.add_argument("--output_dir", default="gemma-2-2B-function-call-ft", help="Where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_train", type=int, default=100, help="Subset train rows for quick runs")
    parser.add_argument("--max_eval", type=int, default=10, help="Subset eval rows for quick runs")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hf_username", default=None, help="HF username for pushing the model")
    parser.add_argument("--load_4bit", action="store_true", help="Load base model in 4‑bit quantised mode")
    return parser.parse_args()


def maybe_push_to_hub(trainer: SFTTrainer, tokenizer, username: str, output_dir: str):
    if not username:
        print("No HF username provided, skipping push_to_hub.")
        return
    repo = f"{username}/{Path(output_dir).name}"
    print(f"\nPushing adapter & tokenizer to https://huggingface.co/{repo} …")
    trainer.push_to_hub(repo)
    tokenizer.push_to_hub(repo, token=os.environ.get("HF_TOKEN"))

###############################################################################
# Entry‑point
###############################################################################

def main():
    args = parse_args()

    tokenizer = build_tokenizer(args.model)
    model = build_model(args.model, tokenizer, load_4bit=args.load_4bit)

    dataset = load_and_prepare_dataset(
        args.dataset, tokenizer, max_train=args.max_train, max_eval=args.max_eval
    )

    lora_cfg = build_lora_config()
    results = train(
        model,
        tokenizer,
        dataset,
        lora_cfg,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
    )

    print("\nTraining complete! 🎉")
    print(results)


if __name__ == "__main__":
    main()
