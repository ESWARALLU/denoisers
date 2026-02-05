# Checkpoint Loading Issue - Diagnosis and Solution

## Problem Summary

You're getting a `FileNotFoundError` when trying to continue training from epoch 10:
```
FileNotFoundError: [Errno 2] No such file or directory: '/checkpoints/ablation_full_denoising/ablation_full_denoising_epoch_10.pth'
```

## Root Cause Analysis

### 1. **Checkpoint Type Confusion** ✓ You were correct!

There are **TWO different types of checkpoints** in this ablation training:

#### A. **Pretrained Checkpoint** (Required for initialization)
- **Purpose**: Initialize the shared modules (encoders, cross-attention, decoder) from your original trained model
- **What it should be**: A checkpoint from your **base CloudRemovalCrossAttention model** trained on the full dataset
- **When used**: Only at the **start** of ablation training (line 338 in train_ablation.py)
- **Expected path parameter**: `--pretrained_path` (e.g., path to your original model's best checkpoint)
- **File naming**: Could be anything like `best_model.pth`, `CR_epoch_75.pth`, etc.

#### B. **Ablation Training Checkpoint** (Saved during ablation training)
- **Purpose**: Resume ablation training from a specific epoch
- **What it contains**: Model state + optimizer state + epoch number + training config
- **When created**: Automatically saved every `--save_freq` epochs during ablation training (line 245)
- **File naming**: `ablation_{variant}_epoch_{epoch}.pth` (e.g., `ablation_full_denoising_epoch_10.pth`)
- **Where saved**: `--save_model_dir` directory (default: `./checkpoints/ablation`)

### 2. **Path Issue**

The path in your error message is `/checkpoints/ablation_full_denoising/ablation_full_denoising_epoch_10.pth`

This is an **absolute path** pointing to the root filesystem. However:
- On **Kaggle**, the checkpoints would be in `/kaggle/working/checkpoints/` or similar
- On **Windows**, absolute paths don't work this way (they'd be `C:\checkpoints\...`)
- The script expects checkpoints in a **relative path** like `./checkpoints/ablation_full_denoising/`

## Solutions

### Solution 1: If you're trying to START ablation training (from scratch)

You need a **pretrained checkpoint** from your original model:

```bash
python train_ablation.py \
  --variant full_denoising \
  --pretrained_path /path/to/your/original/CR_best_model.pth \
  --data_list_filepath /kaggle/working/data.csv \
  --save_model_dir ./checkpoints/ablation_full_denoising \
  --max_epochs 20 \
  --batch_sz 8
```

**Key points:**
- `--pretrained_path` should point to your **original CloudRemovalCrossAttention model** checkpoint
- This checkpoint was likely saved from conversations like "01dd7087" or "cee8caf7" (GLF-CR training)
- Common names: `best_model.pth`, `CR_epoch_XX.pth`, `glf_cr_best.pth`, etc.

### Solution 2: If you're trying to RESUME ablation training (continue from epoch 10)

The current `train_ablation.py` script **does NOT support resuming from a previous ablation training checkpoint**. It only supports:
1. Loading a pretrained original model checkpoint
2. Training from scratch

To resume from epoch 10, you would need to:

**Option A**: Modify the script to add resume functionality
```python
parser.add_argument('--resume_checkpoint', type=str, default=None,
                    help='Path to ablation checkpoint to resume training from')
```

Then load epoch number, optimizer state, etc. from the checkpoint.

**Option B**: Continue training with a different approach
- Check if you have an existing checkpoint at `./checkpoints/ablation_full_denoising/ablation_full_denoising_epoch_10.pth`
- If yes, create a resume script
- If no, you'll need to restart training from scratch with the pretrained model

## What You Should Do Now

### Step 1: Locate your pretrained checkpoint

Find your **original trained model** checkpoint. Likely locations on Kaggle:
```bash
# Check common locations
ls -lah /kaggle/working/checkpoints/
ls -lah /kaggle/working/*.pth
ls -lah ./checkpoints/
```

Likely names:
- `best_model.pth`
- `CR_epoch_XX.pth`
- `glf_cr_best.pth`
- Any checkpoint from your main training (not ablation)

### Step 2A: If you have a pretrained checkpoint - Start fresh ablation training

```bash
python train_ablation.py \
  --variant full_denoising \
  --pretrained_path /path/to/found/checkpoint.pth \
  --data_list_filepath /kaggle/working/data.csv \
  --save_model_dir ./checkpoints/ablation_full_denoising \
  --max_epochs 20 \
  --batch_sz 8 \
  --lr 0.0001
```

### Step 2B: If you DON'T have a pretrained checkpoint

You need to:
1. First train the base CloudRemovalCrossAttention model (using `train_CR_kaggle.py` or similar)
2. Get a good checkpoint (e.g., best_model.pth)
3. Then use that checkpoint for ablation training

## Understanding the Ablation Training Flow

```
[Original Model Training]
     ↓
  best_model.pth (pretrained checkpoint)
     ↓
[Ablation Training - Variant 1]
     ↓
  ablation_full_denoising_epoch_5.pth
  ablation_full_denoising_epoch_10.pth
  ablation_full_denoising_best.pth
```

The ablation study compares different noise-handling strategies, but all variants **must start from the same pretrained base model** for a fair comparison.

## Quick Diagnostic Commands (Run in Kaggle)

```bash
# Check what checkpoints exist
find /kaggle -name "*.pth" -type f 2>/dev/null

# Check current directory checkpoints
ls -lah ./checkpoints/ 2>/dev/null || echo "No ./checkpoints directory"

# Check working directory
ls -lah /kaggle/working/*.pth 2>/dev/null || echo "No .pth files in working"

# Check if the path in error exists
ls -lah /checkpoints/ablation_full_denoising/ 2>/dev/null || echo "Path does not exist"
```
