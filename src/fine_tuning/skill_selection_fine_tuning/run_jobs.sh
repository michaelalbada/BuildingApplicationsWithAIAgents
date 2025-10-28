#!/bin/bash
# Training launch scripts for different scenarios

# ==============================================================================
# 1. TEST RUN (Quick validation - 5 minutes)
# ==============================================================================
test_run() {
    echo "🧪 Running test training..."
    python grpo_production.py \
        --test_mode \
        --num_train_epochs 1 \
        --per_device_train_batch_size 2 \
        --num_generations 2 \
        --logging_steps 1 \
        --output_dir ./checkpoints/test_run
}

# ==============================================================================
# 2. SMALL SCALE (Quick training - 1-2 hours)
# ==============================================================================
small_scale() {
    echo "📊 Running small-scale training..."
    python grpo_production.py \
        --max_train_samples 5000 \
        --num_train_epochs 2 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --num_generations 4 \
        --learning_rate 5e-6 \
        --validation_split 0.05 \
        --logging_steps 10 \
        --eval_steps 100 \
        --save_steps 200 \
        --output_dir ./checkpoints/small_scale \
        --use_wandb
}

# ==============================================================================
# 3. FULL TRAINING (Production quality - 8-12 hours on H100)
# ==============================================================================
full_training() {
    echo "🚀 Running FULL production training..."
    python grpo_production.py \
        --model_name "Qwen/Qwen2-0.5B-Instruct" \
        --dataset_name "glaiveai/glaive-function-calling-v2" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 4 \
        --num_generations 6 \
        --learning_rate 5e-6 \
        --validation_split 0.05 \
        --logging_steps 10 \
        --eval_steps 200 \
        --save_steps 500 \
        --output_dir ./checkpoints/full_training \
        --use_wandb \
        --wandb_project "grpo-function-calling-production"
}

# ==============================================================================
# 4. LARGER MODEL (For more capability - requires more memory)
# ==============================================================================
large_model() {
    echo "💪 Training larger model..."
    python grpo_production.py \
        --model_name "Qwen/Qwen2-1.5B-Instruct" \
        --dataset_name "glaiveai/glaive-function-calling-v2" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 8 \
        --num_generations 4 \
        --learning_rate 3e-6 \
        --validation_split 0.05 \
        --logging_steps 10 \
        --eval_steps 200 \
        --save_steps 500 \
        --reward_format_weight 0.05 \
        --output_dir ./checkpoints/qwen_1.5b \
        --use_wandb \
        --wandb_project "grpo-function-calling-production"
}

# ==============================================================================
# 5. RESUME FROM CHECKPOINT
# ==============================================================================
resume_training() {
    CHECKPOINT_DIR=$1
    if [ -z "$CHECKPOINT_DIR" ]; then
        echo "Usage: resume_training <checkpoint_directory>"
        return 1
    fi
    
    echo "🔄 Resuming training from: $CHECKPOINT_DIR"
    python grpo_production.py \
        --model_name "$CHECKPOINT_DIR" \
        --dataset_name "glaiveai/glaive-function-calling-v2" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 4 \
        --num_generations 6 \
        --learning_rate 5e-6 \
        --validation_split 0.05 \
        --logging_steps 10 \
        --eval_steps 200 \
        --save_steps 500 \
        --output_dir "$CHECKPOINT_DIR" \
        --use_wandb
}

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

# Monitor training
monitor() {
    echo "📊 Monitoring options:"
    echo ""
    echo "1. TensorBoard:"
    echo "   tensorboard --logdir ./checkpoints --port 6006"
    echo "   Then visit: http://localhost:6006"
    echo ""
    echo "2. Watch GPU usage:"
    echo "   watch -n 1 nvidia-smi"
    echo ""
    echo "3. Tail training logs:"
    echo "   tail -f ./checkpoints/*/logs/*.log"
}

# Clean old checkpoints
clean_checkpoints() {
    read -p "⚠️  Delete all checkpoints? This cannot be undone! (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf ./checkpoints/*
        echo "✓ Checkpoints deleted"
    fi
}

# Show training status
status() {
    echo "📊 Training Status"
    echo "=================="
    echo ""
    echo "Checkpoints:"
    du -sh ./checkpoints/* 2>/dev/null || echo "  No checkpoints found"
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
}

# ==============================================================================
# MAIN MENU
# ==============================================================================

show_menu() {
    echo ""
    echo "╔════════════════════════════════════════════════╗"
    echo "║   GRPO Function Calling Training Launcher     ║"
    echo "╚════════════════════════════════════════════════╝"
    echo ""
    echo "Training Options:"
    echo "  1) test_run       - Quick test (5 min)"
    echo "  2) small_scale    - Small training (1-2 hours)"
    echo "  3) full_training  - Full production (8-12 hours) ⭐"
    echo "  4) large_model    - Train 1.5B model"
    echo ""
    echo "Utilities:"
    echo "  m) monitor        - Show monitoring commands"
    echo "  s) status         - Check training status"
    echo "  c) clean          - Clean checkpoints"
    echo "  q) quit"
    echo ""
}

# If script is sourced, show menu
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    # Script is being executed
    case "$1" in
        test) test_run ;;
        small) small_scale ;;
        full) full_training ;;
        large) large_model ;;
        resume) resume_training "$2" ;;
        monitor) monitor ;;
        status) status ;;
        clean) clean_checkpoints ;;
        *)
            show_menu
            echo "Usage: $0 {test|small|full|large|resume|monitor|status|clean}"
            ;;
    esac
else
    # Script is being sourced, functions are available
    echo "✓ Training functions loaded. Available commands:"
    echo "  test_run, small_scale, full_training, large_model"
    echo "  monitor, status, clean_checkpoints"
fi
