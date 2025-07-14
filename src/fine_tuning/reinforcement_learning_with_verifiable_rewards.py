from datasets import load_dataset
from trl import GRPOTrainer
import torch
import re

DPO_DATA = "training_data/rlvr_it_help_desk_training_data.jsonl"

dataset = load_dataset("json", data_files=DPO_DATA, split="train")

# Reward function: parse tool call and check if it matches the label
def reward_tool_match(completions, **kwargs):
    labels = kwargs['label']  # List of labels for the batch
    rewards = []
    for i, completion in enumerate(completions):
        label = labels[i // trainer.args.num_generations]  # Since multiple generations per prompt
        
        # Extract tool name from completion
        tool_match = re.search(r'<tool_call>\s*(\{.*\})\s*</tool_call>', completion, re.DOTALL)
        if tool_match:
            try:
                tool_call = json.loads(tool_match.group(1))
                tool_name = tool_call.get('name', '').lower()
                if label in tool_name:
                    rewards.append(1.0)
                else:
                    rewards.append(-1.0)
            except json.JSONDecodeError:
                rewards.append(-1.0)
        else:
            rewards.append(-1.0)
    return rewards

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_tool_match,
    train_dataset=dataset,
)

print(f"Training on device: {trainer.model.device}")

trainer.train()