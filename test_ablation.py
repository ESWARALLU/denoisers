"""
Test script for ablation study variants (full_denoising, no_denoising, original).

Usage:
    python test_ablation.py --variant full_denoising --checkpoint_path ./checkpoints/ablation_full_denoising/ablation_full_denoising_best.pth
"""

import os
import torch
import torch.nn as nn
import argparse
import json
from datetime import datetime

# Fix PyTorch 2.6 UnpicklingError
import torch.serialization
import argparse as _argparse
torch.serialization.add_safe_globals([_argparse.Namespace])

# optional progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from metrics import PSNR, SSIM
from dataloader import AlignedDataset, get_train_val_test_filelists

# Import all ablation variants
from net_CR_CrossAttention import CloudRemovalCrossAttention
from net_CR_NoDenoising import CloudRemovalCrossAttentionNoDenoising
from net_CR_FullDenoising import CloudRemovalCrossAttentionFullDenoising


##########################################################
def get_model(variant):
    """Get the appropriate model based on variant selection"""
    if variant == 'original':
        return CloudRemovalCrossAttention()
    elif variant == 'no_denoising':
        return CloudRemovalCrossAttentionNoDenoising()
    elif variant == 'full_denoising':
        return CloudRemovalCrossAttentionFullDenoising()
    else:
        raise ValueError(f"Unknown variant: {variant}")


def test(model, opts, variant):
    """Test the ablation model
    Args:
        model: The model to test
        opts: Configuration options
        variant: Ablation variant name
    """
    _, _, test_filelist = get_train_val_test_filelists(opts.data_list_filepath)

    data = AlignedDataset(opts, test_filelist)

    dataloader = torch.utils.data.DataLoader(
        dataset=data,
        batch_size=1,               # FORCE batch=1 for consistency
        shuffle=False,
        num_workers=0,              # Avoid worker issues
        pin_memory=True
    )

    test_set_size = len(test_filelist)
    print(f"Test set size: {test_set_size} images")

    total_psnr = 0.0
    total_ssim = 0.0
    results_per_image = []
    processed_images = 0

    iterator = tqdm(dataloader, total=len(dataloader), desc=f'Testing {variant}') if tqdm else dataloader

    with torch.no_grad():
        for inputs in iterator:
            # Handle both old and new key names
            cloudy_key = 'cloudy_optical' if 'cloudy_optical' in inputs else 'cloudy_data'
            sar_key = 'sar' if 'sar' in inputs else 'SAR_data'
            cloudfree_key = 'cloudfree_optical' if 'cloudfree_optical' in inputs else 'cloudfree_data'
            
            cloudy_data = inputs[cloudy_key].cuda()
            cloudfree_data = inputs[cloudfree_key].cuda()
            SAR_data = inputs[sar_key].cuda()
            file_names = inputs.get('file_name', inputs.get('image_name', ['unknown']))

            # Forward pass
            pred = model(cloudy_data, SAR_data)

            psnr_val = PSNR(pred, cloudfree_data)
            ssim_val = SSIM(pred, cloudfree_data)

            psnr_val = float(psnr_val.item()) if hasattr(psnr_val, "item") else float(psnr_val)
            ssim_val = float(ssim_val.item()) if hasattr(ssim_val, "item") else float(ssim_val)

            total_psnr += psnr_val
            total_ssim += ssim_val

            results_per_image.append({
                "image": file_names,
                "psnr": psnr_val,
                "ssim": ssim_val
            })

            processed_images += 1

            if tqdm:
                iterator.set_postfix({
                    "PSNR": f"{psnr_val:.3f}",
                    "SSIM": f"{ssim_val:.3f}",
                    "Done": processed_images
                })

    avg_psnr = total_psnr / processed_images
    avg_ssim = total_ssim / processed_images

    return avg_psnr, avg_ssim, results_per_image


##########################################################
def main():
    parser = argparse.ArgumentParser(description='Test ablation study variants')

    # Ablation variant selection
    parser.add_argument('--variant', type=str, required=True,
                        choices=['original', 'no_denoising', 'full_denoising'],
                        help='Ablation variant to test')
    
    # Data and checkpoint paths
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to ablation checkpoint (e.g., ablation_full_denoising_best.pth)')
    parser.add_argument('--input_data_folder', type=str, default='/kaggle/input/sen12ms-cr-winter')
    parser.add_argument('--data_list_filepath', type=str, default='/kaggle/working/data.csv')
    
    # Data loading parameters
    parser.add_argument('--batch_sz', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--load_size', type=int, default=256)
    parser.add_argument('--crop_size', type=int, default=128)
    
    # Other settings
    parser.add_argument('--is_test', type=bool, default=True)
    parser.add_argument('--is_use_cloudmask', type=bool, default=False)
    parser.add_argument('--cloud_threshold', type=float, default=0.2)
    parser.add_argument('--results_dir', type=str, default='/kaggle/working/results',
                        help='Directory to save results')
    parser.add_argument('--notes', type=str, default='', help='Additional notes')

    opts = parser.parse_args()

    # Choose CSV properly
    working_csv = '/kaggle/working/data.csv'
    input_csv = os.path.join(opts.input_data_folder, 'data.csv')

    if opts.data_list_filepath == working_csv:
        if os.path.exists(working_csv):
            opts.data_list_filepath = working_csv
        elif os.path.exists(input_csv):
            print(f"Using dataset CSV: {input_csv}")
            opts.data_list_filepath = input_csv
        else:
            raise FileNotFoundError("No CSV available")
    else:
        if not os.path.exists(opts.data_list_filepath):
            raise FileNotFoundError(f"CSV not found: {opts.data_list_filepath}")

    print("\n" + "="*60)
    print(f"ABLATION TEST: {opts.variant.upper()} VARIANT")
    print("="*60)
    print(f"Variant: {opts.variant}")
    print(f"Checkpoint: {opts.checkpoint_path}")
    print(f"Data CSV: {opts.data_list_filepath}")
    print("="*60 + "\n")

    # Load model based on variant
    model = get_model(opts.variant).cuda()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Load checkpoint
    print(f"Loading checkpoint: {opts.checkpoint_path}")
    if not os.path.exists(opts.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {opts.checkpoint_path}")
    
    checkpoint = torch.load(
        opts.checkpoint_path,
        map_location="cuda",
        weights_only=False     # REQUIRED FIX for PyTorch 2.6
    )

    # Extract model state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"✓ Loaded from ablation checkpoint")
        
        # Print checkpoint info
        if 'epoch' in checkpoint:
            print(f"  Epoch: {checkpoint['epoch']}")
        if 'val_psnr' in checkpoint:
            print(f"  Validation PSNR: {checkpoint['val_psnr']:.2f} dB")
        if 'best_val_psnr' in checkpoint:
            print(f"  Best Validation PSNR: {checkpoint['best_val_psnr']:.2f} dB")
    elif "network" in checkpoint:
        state_dict = checkpoint["network"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    print("✓ Model loaded successfully\n")

    # Run test
    print("="*60)
    print(f"Testing Ablation Variant: {opts.variant}")
    print("="*60)
    print(f"{'Image':40s} | {'PSNR':>10s} | {'SSIM':>8s}")
    print("-"*65)

    avg_psnr, avg_ssim, results_per_image = test(model, opts, opts.variant)

    print("-"*65)
    print("="*60)
    print(f"Test Results for {opts.variant.upper()} variant:")
    print(f"  PSNR: {avg_psnr:.4f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    print(f"  Total Images: {len(results_per_image)}")
    print("="*60)

    # Save results
    os.makedirs(opts.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_json = os.path.join(opts.results_dir, f"ablation_{opts.variant}_{timestamp}.json")
    with open(out_json, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "variant": opts.variant,
            "checkpoint": opts.checkpoint_path,
            "avg_psnr": avg_psnr,
            "avg_ssim": avg_ssim,
            "num_images": len(results_per_image),
            "notes": opts.notes,
            "per_image": results_per_image
        }, f, indent=4)

    print(f"\n✓ Saved results to {out_json}")
    
    # Also save a summary CSV for easy comparison
    summary_csv = os.path.join(opts.results_dir, "ablation_summary.csv")
    
    # Check if summary exists, if not create with header
    if not os.path.exists(summary_csv):
        with open(summary_csv, 'w') as f:
            f.write("timestamp,variant,avg_psnr,avg_ssim,num_images,checkpoint,notes\n")
    
    # Append this result
    with open(summary_csv, 'a') as f:
        f.write(f"{timestamp},{opts.variant},{avg_psnr:.4f},{avg_ssim:.4f},{len(results_per_image)},{opts.checkpoint_path},{opts.notes}\n")
    
    print(f"✓ Updated summary: {summary_csv}\n")


if __name__ == "__main__":
    main()
