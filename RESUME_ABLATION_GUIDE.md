# Resume Ablation Training Guide

## Overview

The `train_ablation.py` script now supports **two modes**:

1. **Fresh Start Mode** - Train from scratch using a pretrained base model
2. **Resume Mode** - Continue training from a saved ablation checkpoint

## Usage Examples

### Mode 1: Fresh Start (Training from Scratch)

Start ablation training from a pretrained base model:

```bash
# Full Denoising variant
python train_ablation.py \
  --variant full_denoising \
  --pretrained_path /kaggle/working/checkpoints/CR_best.pth \
  --data_list_filepath /kaggle/working/data.csv \
  --save_model_dir ./checkpoints/ablation_full_denoising \
  --max_epochs 20 \
  --batch_sz 8 \
  --lr 0.0001

# No Denoising variant
python train_ablation.py \
  --variant no_denoising \
  --pretrained_path /kaggle/working/checkpoints/CR_best.pth \
  --data_list_filepath /kaggle/working/data.csv \
  --save_model_dir ./checkpoints/ablation_no_denoising \
  --max_epochs 20 \
  --batch_sz 8 \
  --lr 0.0001

# Original variant (with speckle gating)
python train_ablation.py \
  --variant original \
  --pretrained_path /kaggle/working/checkpoints/CR_best.pth \
  --data_list_filepath /kaggle/working/data.csv \
  --save_model_dir ./checkpoints/ablation_original \
  --max_epochs 20 \
  --batch_sz 8 \
  --lr 0.0001
```

**Key Points:**
- `--pretrained_path` must point to your **original CloudRemovalCrossAttention model** checkpoint
- This checkpoint should be from your main training (e.g., from `train_CR_kaggle.py`)
- All variants should use the **same pretrained checkpoint** for fair comparison

### Mode 2: Resume Training (Continue from Checkpoint)

Resume ablation training from a previously saved checkpoint:

```bash
# Resume full_denoising from epoch 10
python train_ablation.py \
  --variant full_denoising \
  --resume_checkpoint ./checkpoints/ablation_full_denoising/ablation_full_denoising_epoch_10.pth \
  --data_list_filepath /kaggle/working/data.csv \
  --save_model_dir ./checkpoints/ablation_full_denoising \
  --max_epochs 20

# Resume no_denoising from epoch 5
python train_ablation.py \
  --variant no_denoising \
  --resume_checkpoint ./checkpoints/ablation_no_denoising/ablation_no_denoising_epoch_5.pth \
  --data_list_filepath /kaggle/working/data.csv \
  --save_model_dir ./checkpoints/ablation_no_denoising \
  --max_epochs 20
```

**Key Points:**
- `--resume_checkpoint` takes **priority** over `--pretrained_path`
- Training will continue from the next epoch after the checkpoint (e.g., epoch 11 if resuming from epoch 10)
- The optimizer and learning rate scheduler states are restored
- The best validation PSNR is carried over

## Understanding Checkpoint Types

### Pretrained Checkpoint (Base Model)
- **File**: `CR_best.pth`, `CR_epoch_XX.pth`, etc.
- **Source**: Main training script (`train_CR_kaggle.py`)
- **Purpose**: Initialize shared modules for ablation study
- **Used in**: Fresh start mode only
- **Example path**: `/kaggle/working/checkpoints/CR_best.pth`

### Ablation Training Checkpoint
- **File**: `ablation_{variant}_epoch_{epoch}.pth`
- **Source**: Automatically saved during ablation training
- **Purpose**: Resume ablation training from specific epoch
- **Used in**: Resume mode only
- **Example path**: `./checkpoints/ablation_full_denoising/ablation_full_denoising_epoch_10.pth`

## What Gets Loaded in Each Mode

### Fresh Start Mode
Loads from `--pretrained_path`:
- ✅ Model weights (shared modules: encoders, cross-attention, decoder)
- ❌ Optimizer state (fresh initialization)
- ❌ Learning rate scheduler state (fresh initialization)
- ❌ Training progress (starts from epoch 0)

### Resume Mode
Loads from `--resume_checkpoint`:
- ✅ Complete model state (all weights)
- ✅ Optimizer state (momentum, learning rates, etc.)
- ✅ Learning rate scheduler state
- ✅ Training progress (epoch number, best PSNR)
- ✅ Previous configuration (for verification)

## Common Scenarios

### Scenario 1: Kaggle Session Timeout
Your Kaggle session timed out at epoch 10/20:

```bash
# Find the latest checkpoint
ls -lah ./checkpoints/ablation_full_denoising/

# Resume from epoch 10
python train_ablation.py \
  --variant full_denoising \
  --resume_checkpoint ./checkpoints/ablation_full_denoising/ablation_full_denoising_epoch_10.pth \
  --data_list_filepath /kaggle/working/data.csv \
  --max_epochs 20
```

### Scenario 2: Change Hyperparameters Mid-Training
You want to reduce learning rate after epoch 10:

```bash
python train_ablation.py \
  --variant full_denoising \
  --resume_checkpoint ./checkpoints/ablation_full_denoising/ablation_full_denoising_epoch_10.pth \
  --data_list_filepath /kaggle/working/data.csv \
  --max_epochs 20 \
  --lr 0.00005  # Reduced learning rate for fine-tuning
```

> ⚠️ **Note**: The optimizer state from the checkpoint will be loaded first, then the new learning rate will override it. This allows you to change hyperparameters when resuming.

### Scenario 3: Starting All Three Variants
Train all variants from the same pretrained base model:

```bash
# 1. Full denoising
python train_ablation.py --variant full_denoising --pretrained_path /path/to/CR_best.pth --max_epochs 20

# 2. No denoising
python train_ablation.py --variant no_denoising --pretrained_path /path/to/CR_best.pth --max_epochs 20

# 3. Original (gating)
python train_ablation.py --variant original --pretrained_path /path/to/CR_best.pth --max_epochs 20
```

## Troubleshooting

### Error: "FileNotFoundError"
**Problem**: Can't find checkpoint file

**Solution**: Check the exact path
```bash
# On Kaggle, find your checkpoints
find /kaggle -name "*.pth" -type f 2>/dev/null

# In local directory
ls -lah ./checkpoints/ablation_*/
```

### Error: "Either --pretrained_path or --resume_checkpoint must be provided"
**Problem**: Neither checkpoint path was specified

**Solution**: You must provide one of:
- `--pretrained_path` for fresh start
- `--resume_checkpoint` for resuming

### Warning: "Checkpoint variant does not match requested variant"
**Problem**: You're loading a checkpoint from a different variant

**Example**: Loading `ablation_full_denoising_epoch_10.pth` with `--variant no_denoising`

**Solution**: Make sure the variant matches:
```bash
# Correct
python train_ablation.py --variant full_denoising --resume_checkpoint ablation_full_denoising_epoch_10.pth

# Incorrect
python train_ablation.py --variant no_denoising --resume_checkpoint ablation_full_denoising_epoch_10.pth
```

## Checkpoint Saving Behavior

Checkpoints are automatically saved:
- Every `--save_freq` epochs (default: 5)
- At the final epoch
- When a new best validation PSNR is achieved

**Saved files:**
```
checkpoints/ablation_full_denoising/
├── ablation_full_denoising_epoch_5.pth    # Regular checkpoint
├── ablation_full_denoising_epoch_10.pth   # Regular checkpoint
├── ablation_full_denoising_epoch_15.pth   # Regular checkpoint
└── ablation_full_denoising_best.pth       # Best model so far
```

## Best Practices

1. **Fair Comparison**: All variants should start from the **same pretrained checkpoint**
2. **Save Frequently**: Set `--save_freq 5` or less to avoid losing progress
3. **Resume from Best**: To continue training past max_epochs, resume from the `_best.pth` checkpoint
4. **Keep Logs**: Training logs are saved in `./checkpoints/ablation_{variant}/logs/`
5. **Variant Consistency**: Always match the `--variant` argument with the checkpoint variant

## Quick Reference

| Mode | Required Argument | Optional Arguments | Training Starts From |
|------|------------------|-------------------|---------------------|
| Fresh Start | `--pretrained_path` | `--max_epochs`, `--lr`, etc. | Epoch 0 |
| Resume | `--resume_checkpoint` | `--max_epochs` (to extend training) | Next epoch after checkpoint |

## Example Workflow

### Initial Training (Fresh Start)
```bash
# Start training full_denoising variant
python train_ablation.py \
  --variant full_denoising \
  --pretrained_path /kaggle/working/checkpoints/CR_best.pth \
  --max_epochs 20 \
  --save_freq 5
```

### Session Interrupted at Epoch 10
```bash
# Resume from epoch 10 checkpoint
python train_ablation.py \
  --variant full_denoising \
  --resume_checkpoint ./checkpoints/ablation_full_denoising/ablation_full_denoising_epoch_10.pth \
  --max_epochs 20
```

### Extend Training Beyond Original max_epochs
```bash
# Continue training to epoch 30 (resume from epoch 20)
python train_ablation.py \
  --variant full_denoising \
  --resume_checkpoint ./checkpoints/ablation_full_denoising/ablation_full_denoising_epoch_20.pth \
  --max_epochs 30
```
