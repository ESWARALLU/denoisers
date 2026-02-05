# Testing Ablation Study Results

## Quick Start

### Test Your Full Denoising Model

```bash
python test_ablation.py \
  --variant full_denoising \
  --checkpoint_path ./checkpoints/ablation_full_denoising/ablation_full_denoising_best.pth \
  --data_list_filepath /kaggle/working/data.csv
```

### Test All Three Variants

```bash
# 1. Full Denoising
python test_ablation.py \
  --variant full_denoising \
  --checkpoint_path ./checkpoints/ablation_full_denoising/ablation_full_denoising_best.pth

# 2. No Denoising
python test_ablation.py \
  --variant no_denoising \
  --checkpoint_path ./checkpoints/ablation_no_denoising/ablation_no_denoising_best.pth

# 3. Original (with gating)
python test_ablation.py \
  --variant original \
  --checkpoint_path ./checkpoints/ablation_original/ablation_original_best.pth
```

## Command Line Arguments

### Required Arguments
- `--variant`: Ablation variant (`original`, `no_denoising`, `full_denoising`)
- `--checkpoint_path`: Path to checkpoint file (`.pth`)

### Optional Arguments
- `--data_list_filepath`: Path to data CSV (default: `/kaggle/working/data.csv`)
- `--input_data_folder`: Path to image data (default: `/kaggle/input/sen12ms-cr-winter`)
- `--results_dir`: Directory to save results (default: `/kaggle/working/results`)
- `--notes`: Additional notes to include in results

## Output

### Console Output
```
============================================================
ABLATION TEST: FULL_DENOISING VARIANT
============================================================
Variant: full_denoising
Checkpoint: ./checkpoints/ablation_full_denoising/ablation_full_denoising_best.pth
Data CSV: /kaggle/working/data.csv
============================================================

Loading checkpoint: ./checkpoints/ablation_full_denoising/ablation_full_denoising_best.pth
✓ Loaded from ablation checkpoint
  Epoch: 20
  Validation PSNR: 32.45 dB
  Best Validation PSNR: 32.45 dB
✓ Model loaded successfully

============================================================
Testing Ablation Variant: full_denoising
============================================================
Testing full_denoising: 100%|██████████| 7117/7117 [15:23<00:00, 7.70it/s]
---------------------------------------------------------------------
============================================================
Test Results for FULL_DENOISING variant:
  PSNR: 32.41 dB
  SSIM: 0.9234
  Total Images: 7117
============================================================

✓ Saved results to /kaggle/working/results/ablation_full_denoising_20260205_230530.json
✓ Updated summary: /kaggle/working/results/ablation_summary.csv
```

### Saved Files

#### 1. Detailed JSON Results
Location: `/kaggle/working/results/ablation_{variant}_{timestamp}.json`

```json
{
    "timestamp": "20260205_230530",
    "variant": "full_denoising",
    "checkpoint": "./checkpoints/ablation_full_denoising/ablation_full_denoising_best.pth",
    "avg_psnr": 32.41,
    "avg_ssim": 0.9234,
    "num_images": 7117,
    "notes": "",
    "per_image": [
        {
            "image": ["ROIs1868_spring_s1_146_p202"],
            "psnr": 34.56,
            "ssim": 0.9456
        },
        ...
    ]
}
```

#### 2. Summary CSV
Location: `/kaggle/working/results/ablation_summary.csv`

```csv
timestamp,variant,avg_psnr,avg_ssim,num_images,checkpoint,notes
20260205_230530,full_denoising,32.4100,0.9234,7117,./checkpoints/ablation_full_denoising/ablation_full_denoising_best.pth,
20260205_231245,no_denoising,30.8900,0.9012,7117,./checkpoints/ablation_no_denoising/ablation_no_denoising_best.pth,
20260205_232001,original,31.5600,0.9123,7117,./checkpoints/ablation_original/ablation_original_best.pth,
```

This CSV makes it easy to compare all variants in Excel or pandas!

## Comparison Workflow

### Step 1: Train all variants
```bash
# Already done for full_denoising
# Train the other two:
python train_ablation.py --variant no_denoising --pretrained_path <base_checkpoint> --max_epochs 20
python train_ablation.py --variant original --pretrained_path <base_checkpoint> --max_epochs 20
```

### Step 2: Test all variants
```bash
python test_ablation.py --variant full_denoising --checkpoint_path ./checkpoints/ablation_full_denoising/ablation_full_denoising_best.pth
python test_ablation.py --variant no_denoising --checkpoint_path ./checkpoints/ablation_no_denoising/ablation_no_denoising_best.pth
python test_ablation.py --variant original --checkpoint_path ./checkpoints/ablation_original/ablation_original_best.pth
```

### Step 3: Compare results
```python
import pandas as pd

# Load summary
df = pd.read_csv('/kaggle/working/results/ablation_summary.csv')
print(df[['variant', 'avg_psnr', 'avg_ssim']])

# Output:
#          variant  avg_psnr  avg_ssim
# 0  full_denoising    32.41    0.9234
# 1   no_denoising    30.89    0.9012
# 2       original    31.56    0.9123
```

## Testing Specific Epoch Checkpoints

You can also test specific epoch checkpoints (not just the best):

```bash
# Test epoch 10
python test_ablation.py \
  --variant full_denoising \
  --checkpoint_path ./checkpoints/ablation_full_denoising/ablation_full_denoising_epoch_10.pth \
  --notes "Testing epoch 10 checkpoint"

# Test epoch 15
python test_ablation.py \
  --variant full_denoising \
  --checkpoint_path ./checkpoints/ablation_full_denoising/ablation_full_denoising_epoch_15.pth \
  --notes "Testing epoch 15 checkpoint"
```

## Troubleshooting

### Error: Checkpoint not found
**Problem**: `FileNotFoundError: Checkpoint not found`

**Solution**: Check the exact path to your checkpoint
```bash
ls -lah ./checkpoints/ablation_full_denoising/
```

### Error: CSV not found
**Problem**: `FileNotFoundError: CSV not found`

**Solution**: Specify the correct CSV path
```bash
python test_ablation.py \
  --variant full_denoising \
  --checkpoint_path <checkpoint_path> \
  --data_list_filepath /kaggle/working/data.csv
```

### CUDA Out of Memory
**Problem**: GPU memory error during testing

**Solution**: The script uses batch_size=1, but if you still have issues:
```bash
# Clear GPU cache first
python -c "import torch; torch.cuda.empty_cache()"

# Then run test
python test_ablation.py ...
```

## Additional Features

### Add Notes to Results
```bash
python test_ablation.py \
  --variant full_denoising \
  --checkpoint_path <checkpoint> \
  --notes "After 20 epochs with lr=1e-4"
```

### Custom Results Directory
```bash
python test_ablation.py \
  --variant full_denoising \
  --checkpoint_path <checkpoint> \
  --results_dir ./my_results
```

## Expected Performance

Based on validation results, you should expect test set performance similar to:
- **Full Denoising**: ~32-33 dB PSNR
- **No Denoising**: ~30-31 dB PSNR  
- **Original (Gating)**: ~31-32 dB PSNR

The exact numbers will depend on your training and test set.
