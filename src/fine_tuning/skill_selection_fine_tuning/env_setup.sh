#!/bin/bash
# GRPO Training Environment Bootstrap Script
# For Ubuntu with Python 3.10 and GPU

set -e  # Exit on error

echo "=== GRPO Training Environment Setup ==="
echo "Starting setup for RL fine-tuning..."

# 1. Update system packages
echo -e "\n[1/8] Updating system packages..."
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential

# 2. Check NVIDIA driver and CUDA
echo -e "\n[2/8] Checking GPU setup..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA driver found"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
    echo "⚠ WARNING: nvidia-smi not found. Install NVIDIA drivers first!"
    echo "Visit: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"
fi

# 3. Create virtual environment
echo -e "\n[3/8] Creating virtual environment..."
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# 4. Install PyTorch with CUDA support
echo -e "\n[4/8] Installing PyTorch with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch can see GPU
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# 5. Install TRL and core dependencies
echo -e "\n[5/8] Installing TRL (Transformer Reinforcement Learning)..."
pip install trl transformers datasets accelerate peft bitsandbytes

# 6. Install additional training utilities
echo -e "\n[6/8] Installing training utilities..."
pip install wandb tensorboard scipy scikit-learn

# Optional: Login to Hugging Face (for gated models)
echo -e "\n[7/8] Hugging Face Hub setup..."
read -p "Do you want to login to Hugging Face Hub? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install huggingface_hub
    huggingface-cli login
fi

# 7. Setup weights and biases (optional but recommended)
echo -e "\n[8/8] W&B setup (optional)..."
read -p "Do you want to login to Weights & Biases for experiment tracking? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    wandb login
fi

# 8. Verify installation
echo -e "\n=== Verification ==="
python3 << 'EOF'
import sys
print(f"Python version: {sys.version}")

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

import transformers
print(f"Transformers version: {transformers.__version__}")

import trl
print(f"TRL version: {trl.__version__}")

import datasets
print(f"Datasets version: {datasets.__version__}")

print("\n✓ All core packages installed successfully!")
EOF

# 9. Create project structure
echo -e "\n=== Creating project structure ==="
mkdir -p training_data
mkdir -p models
mkdir -p logs
mkdir -p checkpoints

# 10. Download Glaive dataset
echo -e "\n=== Downloading training data ==="
python3 << 'EOF'
from datasets import load_dataset
import os

print("Downloading Glaive function calling dataset...")
dataset = load_dataset("glaiveai/glaive-function-calling-v2", split="train")
print(f"Dataset size: {len(dataset)} examples")

# Save a small sample locally for testing
sample = dataset.select(range(100))
os.makedirs("training_data", exist_ok=True)
sample.to_json("training_data/glaive_sample_100.jsonl")
print("✓ Saved 100 examples to training_data/glaive_sample_100.jsonl")
EOF

echo -e "\n=== Setup Complete! ==="
echo "Your environment is ready for GRPO training."
echo ""
echo "To activate your environment:"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Review your training script"
echo "  2. Test with small dataset: training_data/glaive_sample_100.jsonl"
echo "  3. Run full training with glaiveai/glaive-function-calling-v2"
echo ""
echo "Useful commands:"
echo "  nvidia-smi           # Monitor GPU usage"
echo "  tensorboard --logdir logs  # View training metrics"
echo "  wandb login          # Setup experiment tracking"
