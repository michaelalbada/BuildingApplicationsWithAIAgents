"""
Production GRPO Training Script for Function Calling
Trains on full Glaive dataset with proper evaluation and monitoring
"""
import torch
import os
import json
import re
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, DataCollatorWithPadding
from trl import GRPOConfig, GRPOTrainer
from bitsandbytes.optim import AdamW8bit
import wandb

# ============================================================================
# Configuration
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Function Calling Training")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-0.5B-Instruct",
                       help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/grpo_production",
                       help="Directory to save checkpoints")
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="glaiveai/glaive-function-calling-v2",
                       help="Dataset to use for training")
    parser.add_argument("--max_train_samples", type=int, default=None,
                       help="Maximum number of training samples (None = use all)")
    parser.add_argument("--validation_split", type=float, default=0.05,
                       help="Percentage of data to use for validation")
    
    # Training arguments
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                       help="Batch size per device (you have H100, can go higher!)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                       help="Learning rate")
    parser.add_argument("--num_generations", type=int, default=4,
                       help="Number of completions to generate per prompt")
    
    # Reward function weights
    parser.add_argument("--reward_quality_weight", type=float, default=0.8,
                       help="Weight for quality reward")
    parser.add_argument("--reward_format_weight", type=float, default=0.2,
                       help="Weight for format reward")
    
    # Monitoring
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="grpo-function-calling",
                       help="W&B project name")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                       help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    
    # Testing
    parser.add_argument("--test_mode", action="store_true",
                       help="Run in test mode with small dataset")
    
    return parser.parse_args()


# ============================================================================
# Reward Functions
# ============================================================================


# ---------- Utilities for sharper argument scoring ----------

_NUM_TOL = 1e-6

def _infer_type(x):
    if isinstance(x, bool): return "bool"
    if isinstance(x, int) or isinstance(x, float): return "number"
    if isinstance(x, str): return "string"
    if isinstance(x, list): return "list"
    if isinstance(x, dict): return "object"
    if x is None: return "null"
    return "unknown"

def _coerce_like(v, ref):
    """Coerce v toward the type of ref when it's 'stringly typed'."""
    rt = _infer_type(ref)
    if rt == "number":
        if isinstance(v, (int, float)): return v
        if isinstance(v, str):
            xs = v.strip().lower()
            # handle booleans or 'nan' gracefully as mismatch later
            try:
                if "." in xs: return float(xs)
                return int(xs)
            except Exception:
                return v
        return v
    if rt == "bool":
        if isinstance(v, bool): return v
        if isinstance(v, str):
            xs = v.strip().lower()
            if xs in ("true","1","yes","y","t"): return True
            if xs in ("false","0","no","n","f"): return False
        return v
    # for strings/lists/objects leave as-is
    return v

def _close_number(a, b, tol=_NUM_TOL):
    try:
        return abs(float(a) - float(b)) <= max(tol, 1e-6 * max(1.0, abs(float(b))))
    except Exception:
        return False

def _eq_string(a: str, b: str):
    return a.strip().lower() == b.strip().lower()

def _arg_key_score(k: str, pred_v, true_v) -> float:
    """
    Score a single argument key in [0,1].
    - Exact structural/type/value matches earn 1.0
    - Type mismatches earn small but nonzero if value is 'coercibly' close
    - Lists: Jaccard over stringified items (order-insensitive) if shallow
    - Dicts: shallow exact match only (keeps it simple & fast)
    """
    pt = _infer_type(pred_v)
    tt = _infer_type(true_v)

    # Type-aware coercion for "stringly typed" predictions
    pred_c = _coerce_like(pred_v, true_v)
    pc = _infer_type(pred_c)

    # Numbers
    if tt == "number":
        if pc == "number" and _close_number(pred_c, true_v):
            return 1.0
        # wrong type but close after best-effort? partial
        if pc == "number":
            # numeric but off → small credit
            return 0.25
        return 0.0

    # Booleans
    if tt == "bool":
        if isinstance(pred_c, bool):
            return 1.0 if pred_c == true_v else 0.0
        return 0.0

    # Strings
    if tt == "string":
        if isinstance(pred_c, str):
            return 1.0 if _eq_string(pred_c, true_v) else 0.0
        # simple coercions (e.g., number->string) rarely desirable; give nothing
        return 0.0

    # Lists (shallow)
    if tt == "list":
        if isinstance(pred_v, list) and isinstance(true_v, list):
            # shallow set match on stringified items
            ps = set(map(lambda x: json.dumps(x, sort_keys=True), pred_v))
            ts = set(map(lambda x: json.dumps(x, sort_keys=True), true_v))
            if len(ts) == 0:
                return 1.0 if len(ps) == 0 else 0.0
            inter = len(ps & ts)
            return inter / max(1, len(ts))
        return 0.0

    # Objects (shallow exact)
    if tt == "object":
        if isinstance(pred_v, dict) and isinstance(true_v, dict):
            return 1.0 if pred_v == true_v else 0.0
        return 0.0

    # Null/unknown → exact only
    return 1.0 if pred_v == true_v else 0.0


def reward_correct_function_call(
    completions: List[str],
    prompts: List[str],
    format_weight: float = 0.1,
    allow_extra: bool = True,
) -> List[float]:
    """
    Sharpened reward with strong, graded per-argument credit.

    Structure (clamped to [-1, 1]):
      Base penalties:
        - No JSON with 'name' → -1.0 + small format bonus
      Positive components:
        + Name match: +0.20
        + Args quality: up to +0.75 (per-key graded average)
        + Valid JSON presence (parses): +0.05 (tiny—no longer a big win)
        + Format bonus (tiny): format_weight * format_score

      Negatives:
        - Empty args when GT expects args: -0.10
        - Missing required keys: per-key penalty baked into per-key average
        - Wrong name (when GT known): -0.20
        - Extra keys (if allow_extra=True): small penalty capped at -0.05 total

    This pushes the policy to fix arguments, not just emit JSON or name.
    """
    rewards: List[float] = []

    for completion, prompt in zip(completions, prompts):
        # ---------------- Ground truth from prompt ----------------
        gt_hit = _find_first_json_with_name(prompt or "")
        gt_name, gt_args = None, {}
        if gt_hit:
            _, _, gt = gt_hit
            gt_name = gt.get("name")
            gt_args = gt.get("arguments", gt.get("parameters", {}))
            if not isinstance(gt_args, dict):
                gt_args = {}

        # ---------------- Prediction from completion ----------------
        fmt = reward_format_compliance([completion])[0]  # reuse your existing format scorer (tiny)
        pred_hit = _find_first_json_with_name(completion or "")

        if not pred_hit:
            # No parsable function JSON
            rewards.append(max(-1.0, min(1.0, -1.0 + format_weight * fmt)))
            continue

        _, _, pred = pred_hit
        pred_name = pred.get("name")
        pred_args = pred.get("arguments", pred.get("parameters", {}))
        if not isinstance(pred_args, dict):
            pred_args = {}

        reward = 0.0

        # tiny valid-JSON presence (we do NOT want this to dominate)
        reward += 0.05

        # Name matching
        if gt_name is not None and pred_name is not None:
            if str(pred_name) == str(gt_name):
                reward += 0.20
            else:
                reward -= 0.20

        # Argument scoring (dominant signal)
        if isinstance(gt_args, dict) and len(gt_args) > 0:
            if len(pred_args) == 0:
                reward -= 0.10  # discourage lazy empty dicts
                arg_score = 0.0
            else:
                per_key_scores = []
                for k, true_v in gt_args.items():
                    if k in pred_args:
                        per_key_scores.append(_arg_key_score(k, pred_args[k], true_v))
                    else:
                        per_key_scores.append(0.0)  # missing = 0 for that key

                # Average across required keys only
                if len(per_key_scores) > 0:
                    arg_avg = sum(per_key_scores) / len(per_key_scores)
                else:
                    arg_avg = 0.0

                # Map to up-to +0.75
                reward += 0.75 * arg_avg

                # Light penalty for hallucinated extras (keeps exploration but discourages junk)
                if allow_extra and len(pred_args) > len(gt_args):
                    extra = max(0, len(pred_args) - len(gt_args))
                    reward -= min(0.05, 0.01 * extra)
        else:
            # If GT has no args, don't push on args
            pass

        # Tiny format bonus so we don't overfit to formatting
        reward += format_weight * fmt

        rewards.append(max(-1.0, min(1.0, reward)))

    return rewards

# ---------- JSON extraction (FIRST_JSON with "name" heuristic) ----------
def _find_first_json_with_name(s: str):
    """
    Return (start, end, obj) for the first balanced JSON object in s that contains the key "name".
    Scans for '{', then balances braces to find the matching '}', then checks for "name".
    """
    n = len(s)
    i = 0
    while i < n:
        i = s.find("{", i)
        if i == -1:
            return None
        # balance braces
        depth = 0
        for j in range(i, n):
            ch = s[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[i:j+1]
                    if '"name"' in candidate or "'name'" in candidate:
                        try:
                            obj = json.loads(candidate)
                            return i, j+1, obj
                        except Exception:
                            break  # keep scanning after i
                    break
        i += 1
    return None

# ---------- Semantic normalization (basic, shallow) ----------
def _norm_scalar(x):
    # Trim whitespace for strings
    if isinstance(x, str):
        xs = x.strip()
        # booleans
        if xs.lower() == "true":  return True
        if xs.lower() == "false": return False
        # numbers
        try:
            if "." in xs:
                return float(xs)
            return int(xs)
        except Exception:
            return xs
    return x  # leave as-is for numbers/bools/dicts/lists

def _semantic_equal(a, b):
    # BASIC: shallow equality only; nested dicts/lists must match exactly as JSON (no partial credit inside)
    if isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
        return a == b
    return _norm_scalar(a) == _norm_scalar(b)

def _score_args_partial(pred_args: dict, true_args: dict, allow_extra=True) -> float:
    """
    Returns a fraction in [0,1] = (#correct top-level keys) / (#true keys).
    Extra keys in pred_args are ignored if allow_extra=True.
    """
    if not isinstance(pred_args, dict) or not isinstance(true_args, dict) or len(true_args) == 0:
        return 0.0
    correct = 0
    for k, v in true_args.items():
        if k in pred_args and _semantic_equal(pred_args[k], v):
            correct += 1
    return correct / max(1, len(true_args))

def reward_correct_function_call_old(
    completions: List[str],
    prompts: List[str],
    format_weight: float = 0.2,
    allow_extra: bool = True,
) -> List[float]:
    """
    Inline parses:
      - ground truth from the *prompt* (FIRST_JSON containing "name")
      - model output JSON from the completion (FIRST_JSON containing "name")
    Rewards (clamped [-1,1]):
      +0.20 valid JSON in completion
      +0.30 correct function name
      +up to +0.50 partial credit over args (shallow semantic)
      -1.00 if no function call at all
      -0.30 if wrong function name
    Plus a small format bonus (your existing reward_format_compliance) scaled by format_weight.
    """
    rewards = []

    for completion, prompt in zip(completions, prompts):
        # ---------------- Ground truth from prompt ----------------
        gt_hit = _find_first_json_with_name(prompt or "")
        if not gt_hit:
            # If no GT found in prompt, give neutral small reward window (format only).
            gt_name, gt_args = None, {}
        else:
            _, _, gt = gt_hit
            gt_name = gt.get("name")
            gt_args = gt.get("arguments", gt.get("parameters", {}))
            if not isinstance(gt_args, dict):
                gt_args = {}

        # ---------------- Prediction from completion ----------------
        base = -1.0  # default: no function call found
        pred_hit = _find_first_json_with_name(completion or "")
        if not pred_hit:
            # Add small format bonus even if no JSON (to keep signal flowing)
            fmt = reward_format_compliance([completion])[0]
            rewards.append(max(-1.0, min(1.0, base + format_weight * fmt)))
            continue

        # Has some JSON
        base = 0.0
        _, _, pred = pred_hit
        fmt = reward_format_compliance([completion])[0]

        # + valid JSON bonus
        reward = 0.20

        pred_name = pred.get("name")
        pred_args = pred.get("arguments", pred.get("parameters", {}))
        if not isinstance(pred_args, dict):
            pred_args = {}

        # Name match
        if gt_name is not None and pred_name is not None:
            if str(pred_name) == str(gt_name):
                reward += 0.30
                # Arg partial credit
                arg_frac = _score_args_partial(pred_args, gt_args, allow_extra=allow_extra)
                reward += 0.50 * arg_frac
            else:
                reward -= 0.30  # wrong function
        # If gt_name unknown, skip name penalty/bonus but still count format

        # Format bonus (small)
        reward += format_weight * fmt

        rewards.append(max(-1.0, min(1.0, reward)))

    return rewards

def format_for_grpo(example):
    system = example["system"]
    chat = example["chat"]

    # Find the FIRST_JSON with "name" inside the chat
    hit = _find_first_json_with_name(chat)
    if hit:
        start, _, _ = hit
        chat_without_answer = chat[:start]  # cut before the function call JSON
    else:
        chat_without_answer = chat  # fallback

    prompt = system + "\n\n" + chat_without_answer

    # Keep the *raw* chat around so reward can parse the GT from it inline
    return {"prompt": prompt, "raw_chat": chat}

def reward_tool_call_quality(completions: List[str], **kwargs) -> List[float]:
    """
    Granular reward function for function calling quality.
    
    Rewards:
    - Correct tool name + valid JSON + required params: +1.0
    - Correct tool name + valid JSON + missing params: +0.5
    - Correct tool name + invalid JSON: +0.2
    - Wrong tool name + valid JSON: -0.3
    - No tool call or completely invalid: -1.0
    """
    rewards = []
    
    for completion in completions:
        reward = 0.0
        completion_lower = completion.lower()
        
        # Check for function call tags
        has_functioncall = '<functioncall>' in completion_lower
        has_json_structure = '{' in completion and '}' in completion
        
        if not has_functioncall and not has_json_structure:
            # No function call detected
            rewards.append(-1.0)
            continue
        
        if has_functioncall:
            reward += 0.3  # Bonus for using proper tags
        
        # Try to extract and parse JSON
        try:
            # Find JSON in completion
            start = completion.find('{')
            end = completion.rfind('}') + 1
            
            if start == -1 or end <= start:
                rewards.append(0.2 if has_functioncall else -0.5)
                continue
            
            json_str = completion[start:end]
            tool_call = json.loads(json_str)
            
            # Valid JSON - check structure
            if 'name' in tool_call:
                reward += 0.5  # Has function name
                
                # Check for parameters
                if 'arguments' in tool_call or 'parameters' in tool_call:
                    params = tool_call.get('arguments', tool_call.get('parameters', {}))
                    if isinstance(params, dict) and len(params) > 0:
                        reward += 0.2  # Has parameters
                else:
                    reward += 0.1  # Missing parameters but has structure
            else:
                reward -= 0.2  # Invalid structure
            
        except json.JSONDecodeError:
            # Invalid JSON but tried to call function
            reward = 0.2 if has_functioncall else -0.5
        except Exception:
            reward = -0.3
        
        rewards.append(min(max(reward, -1.0), 1.0))  # Clamp to [-1, 1]
    
    return rewards


def reward_format_compliance(completions: List[str], **kwargs) -> List[float]:
    """
    Reward function for format compliance and readability.
    """
    rewards = []
    
    for completion in completions:
        reward = 0.0
        
        # Check for proper XML-style tags
        if '<functioncall>' in completion.lower() and '</functioncall>' in completion.lower():
            reward += 0.4
        
        # Check for balanced braces
        open_braces = completion.count('{')
        close_braces = completion.count('}')
        if open_braces == close_braces and open_braces > 0:
            reward += 0.3
        
        # Check for proper JSON key formatting
        if '"name"' in completion or "'name'" in completion:
            reward += 0.2
        
        # Penalize overly long completions (likely hallucinating)
        if len(completion) > 1000:
            reward -= 0.2
        
        # Bonus for concise, well-formatted responses
        if 100 < len(completion) < 500:
            reward += 0.1
        
        rewards.append(reward)
    
    return rewards


def combined_reward(completions: List[str], **kwargs) -> List[float]:
    """Weighted combination of multiple reward signals."""
    quality_weight = kwargs.get('quality_weight', 0.8)
    format_weight = kwargs.get('format_weight', 0.2)
    
    quality_rewards = reward_tool_call_quality(completions, **kwargs)
    format_rewards = reward_format_compliance(completions, **kwargs)
    
    combined = [
        quality_weight * q + format_weight * f
        for q, f in zip(quality_rewards, format_rewards)
    ]
    
    return combined

class ForceSaveCallback(TrainerCallback):
    """Force save checkpoints at regular intervals"""
    
    def on_step_end(self, args, state, control, **kwargs):
        """Save checkpoint every save_steps"""
        if state.global_step % args.save_steps == 0 and state.global_step > 0:
            control.should_save = True
        return control

# ============================================================================
# Dataset Preparation
# ============================================================================

def prepare_dataset(args):
    """Load and prepare dataset for training."""
    print(f"\n{'='*60}")
    print("Loading Dataset")
    print(f"{'='*60}\n")
    
    if args.test_mode:
        print("🧪 TEST MODE: Loading small sample...")
        dataset = load_dataset("json", data_files="training_data/glaive_sample_100.jsonl", split="train")
        dataset = dataset.select(range(20))  # Just 20 examples for testing
    else:
        print(f"Loading full dataset: {args.dataset_name}")
        dataset = load_dataset(args.dataset_name, split="train")
        
        if args.max_train_samples:
            print(f"Limiting to {args.max_train_samples} samples")
            dataset = dataset.select(range(args.max_train_samples))
    
    print(f"Dataset size: {len(dataset):,} examples")
    
    print("Formatting dataset...")
    dataset = dataset.map(format_for_grpo, remove_columns=["system", "chat"])

    
    # Split into train/validation
    if args.validation_split > 0:
        print(f"Creating {args.validation_split*100:.0f}% validation split...")
        split = dataset.train_test_split(test_size=args.validation_split, seed=42)
        train_dataset = split['train']
        eval_dataset = split['test']
        print(f"  Train: {len(train_dataset):,} examples")
        print(f"  Eval:  {len(eval_dataset):,} examples")
    else:
        train_dataset = dataset
        eval_dataset = None
    
    return train_dataset, eval_dataset

def _tok(batch, tokenizer=None):
    enc = tokenizer(
        batch["prompt"],
        padding=False,          # we'll pad in the collator
        truncation=True,
        max_length=2048,
        return_tensors=None,
    )
    # carry raw_chat through for reward_func
    enc["raw_chat"] = batch["raw_chat"]
    return enc

class KeepRawChatCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # gather passthrough strings (these keys exist thanks to processing_class)
        prompts   = [f.get("prompt", "")   for f in features]
        raw_chats = [f.get("raw_chat", "") for f in features]

        # strip non-tensor fields before padding
        to_pad = [
            {k: v for k, v in f.items() if k not in ("prompt", "raw_chat")}
            for f in features
        ]

        batch = super().__call__(to_pad)

        # reattach in the shapes TRL expects/uses
        batch["inputs"]    = [{"prompt": p} for p in prompts]  # <-- avoids "string indices" error
        batch["prompts"]   = prompts
        batch["raw_chats"] = raw_chats
        return batch

class PromptPassthroughCollator:
    def __call__(self, features):
        # features are dicts with at least "prompt" and "raw_chat"
        batch = []
        for f in features:
            batch.append({
                "prompt":   f.get("prompt", ""),
                "raw_chat": f.get("raw_chat", ""),
            })
        return batch

class GRPOProcessing:
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        enc = self.tokenizer(
            examples["prompt"],
            padding=False,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        # keep passthrough fields at the example-level
        # (these will be lifted by the collator)
        enc["prompt"] = examples["prompt"]
        enc["raw_chat"] = examples["raw_chat"]
        return enc

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

# ============================================================================
# Training
# ============================================================================

def main():
    args = parse_args()
    
    # Print configuration
    print(f"\n{'='*60}")
    print("GRPO Function Calling Training")
    print(f"{'='*60}\n")
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Batch size: {args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Generations per prompt: {args.num_generations}")
    
    if args.test_mode:
        print("\n⚠️  RUNNING IN TEST MODE")
    
    # Check GPU
    print(f"\n{'='*60}")
    print("Hardware")
    print(f"{'='*60}\n")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.model_name.split('/')[-1]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=vars(args)
        )
    
    # Load dataset
    train_dataset, eval_dataset = prepare_dataset(args)
    
    # Configure training
    print(f"\n{'='*60}")
    print("Training Configuration")
    print(f"{'='*60}\n")
    
    # Configure evaluation strategy
    has_eval = eval_dataset is not None and len(eval_dataset) > 0
    
    # Ensure save_steps is a multiple of eval_steps when using load_best_model_at_end
    save_steps = args.save_steps
    eval_steps = args.eval_steps
    if has_eval and save_steps % eval_steps != 0:
        # Adjust save_steps to be a multiple of eval_steps
        save_steps = ((save_steps // eval_steps) + 1) * eval_steps
        print(f"⚙️  Adjusted save_steps to {save_steps} (must be multiple of eval_steps={eval_steps})")
    
    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        remove_unused_columns=False,
        eval_strategy="steps" if has_eval else "no",
        eval_steps=eval_steps if has_eval else None,
        save_strategy="steps",
        save_steps=save_steps,
        num_generations=args.num_generations,
        report_to="wandb" if args.use_wandb else "tensorboard",
        bf16=True,
        save_total_limit=5,
        load_best_model_at_end=False,
        save_safetensors=True,
        metric_for_best_model=None,
    )
    
    # Create reward function with configured weights
    def old_reward_func(completions: List[str], **kwargs) -> List[float]:
        return combined_reward(
            completions,
            quality_weight=args.reward_quality_weight,
            format_weight=args.reward_format_weight,
            **kwargs
        )
    
    def reward_func(completions: List[str], **kwargs) -> List[float]:
        # Try to recover the raw chat text from batch (column survives into kwargs in many setups)
        raw_chats = kwargs.get("raw_chat") or kwargs.get("raw_chats")
        prompts = kwargs.get("prompts") or kwargs.get("queries") or kwargs.get("inputs_text")

        # If raw_chat not routed through kwargs by GRPO, fall back to prompts (less ideal)
        ground_truth_carrier = raw_chats if raw_chats is not None else prompts
        if ground_truth_carrier is None:
            ground_truth_carrier = [""] * len(completions)

        return reward_correct_function_call(
            completions=completions,
            prompts=ground_truth_carrier,
            format_weight=args.reward_format_weight,
            allow_extra=True,
        )

    torch._dynamo.config.cache_size_limit = 64
    
    use_load_in_8bit = False  # set True to test 8-bit loading
    model_obj = None
    if use_load_in_8bit:
        from transformers import AutoModelForCausalLM
        print("Loading model in 8-bit to save memory...")
        model_obj = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    if model_obj is not None:
        try:
            model_obj.gradient_checkpointing_enable()
            print("✓ gradient checkpointing enabled on model object")
        except Exception as ex:
            print("⚠️ couldn't enable gradient checkpointing on model object:", ex)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, padding_side="left", truncation_side="left")
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    save_callback = ForceSaveCallback()
    processing = tokenizer

    trainer = GRPOTrainer(
        model=model_obj if model_obj is not None else args.model_name,
        reward_funcs=reward_func,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processing,
        args=config,
        callbacks=[save_callback],
    )
    trainer.tokenizer = tokenizer 
    trainer.data_collator = PromptPassthroughCollator()
    trainer.model.config.padding_side = "left"
    trainer.model.generation_config.padding_side = "left"

    m = trainer.model
    m.resize_token_embeddings(len(tokenizer))
    m.config.pad_token_id = tokenizer.pad_token_id
    m.config.eos_token_id = tokenizer.eos_token_id
    m.config.bos_token_id = getattr(tokenizer, "bos_token_id", None)
    m.config.padding_side = "left"
    if hasattr(m, "generation_config"):
        m.generation_config.pad_token_id = tokenizer.pad_token_id
        m.generation_config.eos_token_id = tokenizer.eos_token_id
        m.generation_config.padding_side = "left"

    try:
        trainer.model.config.pad_token_id = tokenizer.pad_token_id
        trainer.model.config.bos_token_id = tokenizer.bos_token_id
        trainer.model.config.eos_token_id = tokenizer.eos_token_id
        if hasattr(trainer.model, "generation_config"):
            gc = trainer.model.generation_config
            gc.pad_token_id = tokenizer.pad_token_id
            gc.bos_token_id = tokenizer.bos_token_id
            gc.eos_token_id = tokenizer.eos_token_id
            gc.padding_side = "left"
    except Exception as ex:
        print("⚠️  Could not sync special tokens on model:", ex)

    if model_obj is not None:
        try:
            trainer.optimizer = AdamW8bit(model_obj.parameters(), lr=args.learning_rate)
            print("✓ AdamW8bit optimizer assigned")
        except Exception as ex:
            print("⚠️ Couldn't set AdamW8bit optimizer; falling back to trainer default:", ex)

    trainer.model.gradient_checkpointing_enable()

    print(f"✓ Trainer initialized")
    print(f"✓ Model loaded on: {trainer.model.device}")
    
    # Train
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")
    
    # Now trainer.model exists. Apply memory/tuning changes **after** init:
    # 1) gradient checkpointing (saves large amounts of activation memory)
    try:
        trainer.model.gradient_checkpointing_enable()
    except Exception as ex:
        print("⚠️  Could not enable gradient_checkpointing:", ex)

    # 2) optional torch.compile to reduce overhead (call AFTER model is on device)
    try:
        # limit dynamo cache to reduce memory overhead (tweak if needed)
        import torch._dynamo as dynamo
        dynamo.config.cache_size_limit = 64
        trainer.model = torch.compile(trainer.model, mode="reduce-overhead")
    except Exception as ex:
        print("⚠️  torch.compile not applied (okay):", ex)

    # 3) set a memory-efficient optimizer (bitsandbytes AdamW8bit)
    try:
        trainer.optimizer = AdamW8bit(trainer.model.parameters(), lr=args.learning_rate)
    except Exception as ex:
        print("⚠️  Couldn't set AdamW8bit optimizer; falling back to trainer default:", ex)

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    except Exception as ex:
        print("⚠️  Failed to clear CUDA cache and reset peak memory stats", ex)

    training_failed = False
    try:

        sample = [train_dataset[i]["prompt"] for i in range(3)]
        enc = tokenizer(sample, padding=True, return_tensors="pt", truncation=True, max_length=256)
        print("padding_side:", tokenizer.padding_side)
        print("pad_token_id:", tokenizer.pad_token_id, "eos_token_id:", tokenizer.eos_token_id)
        # Left padding ⇒ first column mostly pad_token_id; last column non-pad.
        print("first col ids:", enc["input_ids"][:, 0].tolist())
        print("last  col ids:", enc["input_ids"][:, -1].tolist())

        result = trainer.train()
        
        print(f"\n{'='*60}")
        print("✓ TRAINING COMPLETED!")
        print(f"{'='*60}\n")
        print(f"Final checkpoint: {args.output_dir}")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Saving current state...")
        training_failed = True

    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        print("Attempting to save current state...")
        import traceback
        traceback.print_exc()
        training_failed = True

    finally:
        # ALWAYS save the model, even if training crashed
        print("\n💾 Saving final model...")
        final_model_path = os.path.join(args.output_dir, "final_model")
        try:
            trainer.save_model(final_model_path)
            print(f"✓ Model saved to: {final_model_path}")
            
            # Save training info
            info = {
                "model": args.model_name,
                "dataset": args.dataset_name,
                "train_samples": len(train_dataset),
                "timestamp": datetime.now().isoformat(),
                "completed": "partial" if training_failed else "full",
            }
            
            with open(os.path.join(args.output_dir, "training_info.json"), "w") as f:
                json.dump(info, f, indent=2)
                
        except Exception as save_error:
            print(f"⚠️  Could not save model: {save_error}")

        if args.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
