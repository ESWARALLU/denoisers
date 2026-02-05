#!/bin/bash
# Example Commands for Resume Ablation Training
# This file demonstrates how to use the new resume functionality

echo "========================================"
echo "Resume Ablation Training Examples"
echo "========================================"

# Example 1: Fresh Start - Full Denoising Variant
echo -e "\n[Example 1] Starting full_denoising variant from scratch\n"
cat << 'EOF'
python train_ablation.py \
  --variant full_denoising \
  --pretrained_path /kaggle/working/checkpoints/CR_best.pth \
  --data_list_filepath /kaggle/working/data.csv \
  --save_model_dir ./checkpoints/ablation_full_denoising \
  --max_epochs 20 \
  --batch_sz 8 \
  --lr 0.0001 \
  --save_freq 5
EOF

# Example 2: Fresh Start - No Denoising Variant
echo -e "\n[Example 2] Starting no_denoising variant from scratch\n"
cat << 'EOF'
python train_ablation.py \
  --variant no_denoising \
  --pretrained_path /kaggle/working/checkpoints/CR_best.pth \
  --data_list_filepath /kaggle/working/data.csv \
  --save_model_dir ./checkpoints/ablation_no_denoising \
  --max_epochs 20 \
  --batch_sz 8 \
  --lr 0.0001 \
  --save_freq 5
EOF

# Example 3: Fresh Start - Original Variant (with gating)
echo -e "\n[Example 3] Starting original variant from scratch\n"
cat << 'EOF'
python train_ablation.py \
  --variant original \
  --pretrained_path /kaggle/working/checkpoints/CR_best.pth \
  --data_list_filepath /kaggle/working/data.csv \
  --save_model_dir ./checkpoints/ablation_original \
  --max_epochs 20 \
  --batch_sz 8 \
  --lr 0.0001 \
  --save_freq 5
EOF

# Example 4: Resume from Checkpoint - Full Denoising
echo -e "\n[Example 4] Resume full_denoising from epoch 10\n"
cat << 'EOF'
python train_ablation.py \
  --variant full_denoising \
  --resume_checkpoint ./checkpoints/ablation_full_denoising/ablation_full_denoising_epoch_10.pth \
  --data_list_filepath /kaggle/working/data.csv \
  --save_model_dir ./checkpoints/ablation_full_denoising \
  --max_epochs 20
EOF

# Example 5: Resume from Checkpoint - No Denoising
echo -e "\n[Example 5] Resume no_denoising from epoch 10\n"
cat << 'EOF'
python train_ablation.py \
  --variant no_denoising \
  --resume_checkpoint ./checkpoints/ablation_no_denoising/ablation_no_denoising_epoch_10.pth \
  --data_list_filepath /kaggle/working/data.csv \
  --save_model_dir ./checkpoints/ablation_no_denoising \
  --max_epochs 20
EOF

# Example 6: Extend Training Beyond Original max_epochs
echo -e "\n[Example 6] Extend training to 30 epochs (resume from epoch 20)\n"
cat << 'EOF'
python train_ablation.py \
  --variant full_denoising \
  --resume_checkpoint ./checkpoints/ablation_full_denoising/ablation_full_denoising_epoch_20.pth \
  --data_list_filepath /kaggle/working/data.csv \
  --save_model_dir ./checkpoints/ablation_full_denoising \
  --max_epochs 30
EOF

# Example 7: Resume with Modified Learning Rate
echo -e "\n[Example 7] Resume with reduced learning rate for fine-tuning\n"
cat << 'EOF'
python train_ablation.py \
  --variant full_denoising \
  --resume_checkpoint ./checkpoints/ablation_full_denoising/ablation_full_denoising_best.pth \
  --data_list_filepath /kaggle/working/data.csv \
  --save_model_dir ./checkpoints/ablation_full_denoising_finetune \
  --max_epochs 30 \
  --lr 0.00005
EOF

echo -e "\n========================================"
echo "Use these commands in your Kaggle notebook or terminal"
echo "========================================"
