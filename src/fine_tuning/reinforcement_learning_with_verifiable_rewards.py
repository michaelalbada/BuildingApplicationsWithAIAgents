from datasets import load_dataset
from trl import GRPOTrainer
import torch
import re
import json
from typing import List, Dict, Any

DPO_DATA = "training_data/rlvr_it_help_desk_training_data.jsonl"

dataset = load_dataset("json", data_files=DPO_DATA, split="train")

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
    labels = kwargs.get('label', [])
    expected_params = kwargs.get('required_params', [])  # Optional: list of required param names

    rewards = []
    num_generations = kwargs.get('num_generations', getattr(trainer.args, 'num_generations', 1))

    for i, completion in enumerate(completions):
        # Get the label for this completion
        label_idx = i // num_generations
        if label_idx >= len(labels):
            rewards.append(-1.0)
            continue

        label = labels[label_idx]
        expected_tool = label.lower().strip()

        # Extract tool call from completion
        tool_match = re.search(
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
            completion,
            re.DOTALL
        )

        if not tool_match:
            # No tool call found
            rewards.append(-1.0)
            continue

        tool_json_str = tool_match.group(1)

        # Try to parse JSON
        try:
            tool_call = json.loads(tool_json_str)
        except json.JSONDecodeError:
            # Invalid JSON - but at least tried to call a tool
            # Check if tool name appears in the malformed string
            if expected_tool in tool_json_str.lower():
                rewards.append(0.2)
            else:
                rewards.append(-0.5)
            continue

        # Valid JSON - now check tool name
        tool_name = tool_call.get('name', '').lower().strip()

        # Check if tool name is correct
        tool_name_correct = (
            expected_tool in tool_name or
            tool_name in expected_tool or
            tool_name == expected_tool
        )

        if not tool_name_correct:
            # Wrong tool but valid JSON
            rewards.append(-0.3)
            continue

        # Correct tool name + valid JSON
        # Now check parameters if provided
        if expected_params and label_idx < len(expected_params):
            required = expected_params[label_idx]
            provided_params = tool_call.get('parameters', tool_call.get('arguments', {}))

            if isinstance(provided_params, dict):
                has_all_required = all(
                    param in provided_params and provided_params[param] not in [None, '', []]
                    for param in required
                )

                if has_all_required:
                    rewards.append(1.0)  # Perfect!
                else:
                    rewards.append(0.5)  # Correct tool but missing params
            else:
                rewards.append(0.5)  # Parameters in wrong format
        else:
            # No parameter checking - just validate tool name
            rewards.append(1.0)

    return rewards


def reward_format_compliance(completions: List[str], **kwargs) -> List[float]:
    """
    Reward function specifically for format compliance.
    Checks for proper XML tags, JSON structure, etc.
    """
    rewards = []

    for completion in completions:
        reward = 0.0

        # Check for proper tool_call tags
        if '<tool_call>' in completion and '</tool_call>' in completion:
            reward += 0.3

        # Check for balanced braces
        if completion.count('{') == completion.count('}'):
            reward += 0.2

        # Check for quotes around keys (basic JSON formatting)
        tool_match = re.search(r'<tool_call>(.*?)</tool_call>', completion, re.DOTALL)
        if tool_match:
            tool_content = tool_match.group(1).strip()
            # Basic check: should have "name" with quotes
            if '"name"' in tool_content or "'name'" in tool_content:
                reward += 0.3

            # Check if parseable
            try:
                json.loads(tool_content)
                reward += 0.2  # Bonus for valid JSON
            except:
                pass

        rewards.append(reward)

    return rewards


# Combine multiple reward functions with weights
def combined_reward(completions: List[str], **kwargs) -> List[float]:
    """Weighted combination of multiple reward signals."""
    quality_rewards = reward_tool_call_quality(completions, **kwargs)
    format_rewards = reward_format_compliance(completions, **kwargs)

    # Weight quality more heavily than format
    combined = [
        0.8 * q + 0.2 * f
        for q, f in zip(quality_rewards, format_rewards)
    ]

    return combined


trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=combined_reward,
    train_dataset=dataset,
    args={
        "num_generations": 4,  # Generate multiple completions per prompt
        "learning_rate": 5e-6,
        "per_device_train_batch_size": 1,
    }
)

print(f"Training on device: {trainer.model.device}")

trainer.train()
