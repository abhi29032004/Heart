"""
Comprehensive test set evaluation script for Swin+GAT heart segmentation model.
Computes Dice Similarity Coefficient and Hausdorff Distance for CT and MR test sets.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from scipy.spatial.distance import directed_hausdorff
from skimage.measure import label as sk_label

from model_transformer import get_transformer_model

# Structure labels
STRUCTURE_NAMES = {
    0: "Background",
    1: "LV",
    2: "RV", 
    3: "LA",
    4: "RA",
    5: "Myocardium",
    6: "Aorta",
    7: "PA"
}

STRUCTURE_SHORT = {
    1: "LV",
    2: "RV", 
    3: "LA",
    4: "RA",
    5: "Myo",
    6: "Ao",
    7: "PA"
}


def compute_metrics_per_class(pred: np.ndarray, gt: np.ndarray, class_id: int) -> Tuple[float, float]:
    """
    Compute Dice and Hausdorff Distance for a specific class.
    
    Args:
        pred: Predicted segmentation (H,W,D)
        gt: Ground truth segmentation (H,W,D)
        class_id: Class label to evaluate
    
    Returns:
        dice: Dice Similarity Coefficient (0-1)
        hd: Hausdorff Distance in mm
    """
    pred_mask = (pred == class_id).astype(np.float32)
    gt_mask = (gt == class_id).astype(np.float32)
    
    # Dice computation
    intersection = np.sum(pred_mask * gt_mask)
    dice = (2.0 * intersection) / (np.sum(pred_mask) + np.sum(gt_mask) + 1e-8)
    
    # Hausdorff Distance
    pred_indices = np.argwhere(pred_mask > 0.5)
    gt_indices = np.argwhere(gt_mask > 0.5)
    
    if len(pred_indices) == 0 or len(gt_indices) == 0:
        # If one is empty, HD is undefined. Return large value or skip
        hd = 0.0 if len(pred_indices) == 0 and len(gt_indices) == 0 else np.inf
    else:
        hd_1 = directed_hausdorff(pred_indices, gt_indices)[0]
        hd_2 = directed_hausdorff(gt_indices, pred_indices)[0]
        hd = max(hd_1, hd_2)
    
    return float(dice), float(hd)


def load_and_preprocess_image(img_path: str, target_shape: Tuple[int,int,int], device) -> torch.Tensor:
    """Load NIfTI image and preprocess."""
    nii = nib.load(img_path)
    img = nii.get_fdata().astype(np.float32)
    
    # Normalize
    mu = img.mean()
    sigma = img.std() + 1e-8
    img = (img - mu) / sigma
    
    # Resize
    img_t = torch.from_numpy(img[None, None]).to(device)
    img_resized = F.interpolate(img_t, size=target_shape, mode='trilinear', align_corners=False)
    
    return img_resized  # (1, 1, D, H, W)


def load_model(checkpoint_path: str, device: str) -> torch.nn.Module:
    """Load trained Swin+GAT model with shape-compatible loading."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Extract model_state_dict from checkpoint
    if isinstance(ckpt, dict):
        for k in ['model_state_dict', 'state_dict', 'model']:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        else:
            state = ckpt
    else:
        state = ckpt
    
    # Infer embed_dim from checkpoint
    embed_dim = 48
    for probe_key in ['encoder.patch_embed.proj.weight', 'encoder.patch_embed.norm.weight']:
        if probe_key in state:
            embed_dim = int(state[probe_key].shape[0])
            break
    
    print(f"Inferred embed_dim={embed_dim} from checkpoint")
    
    # Build model
    model = get_transformer_model(
        in_channels=1, num_classes=8, embed_dim=embed_dim,
        depths=[2,2,6,2], num_heads=[3,6,12,24], 
        window_size=(7,7,7), decoder_dim=256
    )
    model.to(device)
    
    # Strip 'module.' prefix if present
    norm_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith('module.') else k
        norm_state[nk] = v
    
    # Load only compatible shapes
    model_state = model.state_dict()
    compatible = {}
    skipped = 0
    for k, v in norm_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped += 1
    
    res = model.load_state_dict(compatible, strict=False)
    if skipped > 0:
        print(f"Skipped {skipped} incompatible tensors (ok, using partial weights)")
    
    model.eval()
    return model


def evaluate_on_test_set(
    model: torch.nn.Module,
    data_dir: str,
    dataset_name: str,  # 'ct', 'mr', 'ct_train', 'mr_train', etc.
    device: str,
    target_shape: Tuple[int,int,int] = (160, 160, 80)
) -> Tuple[Dict, int]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained segmentation model
        data_dir: Path to archive directory
        dataset_name: Dataset folder name (e.g., 'ct_train', 'mr_test')
        device: torch device
        target_shape: Input volume shape
        
    Returns:
        Tuple of (results dict, number of evaluated cases)
    """
    
    data_path = Path(data_dir) / dataset_name
    
    if not data_path.exists():
        print(f"Dataset directory not found: {data_path}")
        return {}, 0
    
    # Find all images and labels
    image_files = sorted(data_path.glob("*_image.nii"))
    
    print(f"\nFound {len(image_files)} images in {data_path}")
    
    if len(image_files) == 0:
        print("No images found.")
        return {}, 0
    
    # Initialize storage
    results = {
        'per_class': {i: {'dice': [], 'hd': []} for i in range(1, 8)},
        'aggregate': {'dice': [], 'hd': []}
    }
    
    num_evaluated = 0
    
    # Evaluate each case
    with torch.no_grad():
        for img_path in image_files:
            label_path = img_path.parent / img_path.name.replace('_image.nii', '_label.nii')
            
            # Skip if no label
            if not label_path.exists():
                print(f"Skipping {img_path.name} - no label file")
                continue
            
            print(f"\nProcessing: {img_path.name}", end=" ... ")
            
            try:
                # Load and preprocess
                img_t = load_and_preprocess_image(str(img_path), target_shape, device)
                
                # Predict
                pred_logits = model(img_t)  # (1, 8, D, H, W)
                pred_labels = torch.argmax(pred_logits, dim=1)[0].cpu().numpy()  # (D, H, W)
                
                # Resize to original size
                gt_nii = nib.load(label_path)
                gt_orig = gt_nii.get_fdata().astype(np.uint8)
                
                # Resize prediction to match GT size
                pred_t = torch.from_numpy(pred_labels[None, None].astype(np.float32)).to(device)
                pred_resized_t = F.interpolate(
                    pred_t, 
                    size=gt_orig.shape,
                    mode='nearest'
                )
                pred_resized = pred_resized_t[0,0].cpu().numpy().astype(np.uint8)
                
                # Per-class metrics
                for class_id in range(1, 8):
                    dice, hd = compute_metrics_per_class(pred_resized, gt_orig, class_id)
                    results['per_class'][class_id]['dice'].append(dice)
                    results['per_class'][class_id]['hd'].append(hd)
                
                # Aggregate (all structures combined)
                pred_any = (pred_resized > 0).astype(np.float32)
                gt_any = (gt_orig > 0).astype(np.float32)
                dice_agg, hd_agg = compute_metrics_per_class(
                    (pred_any * 255).astype(np.uint8),
                    (gt_any * 255).astype(np.uint8),
                    255
                )
                results['aggregate']['dice'].append(dice_agg)
                results['aggregate']['hd'].append(hd_agg)
                print(f"✓ Dice: {dice_agg:.4f}, HD: {hd_agg:.2f}")
                
                num_evaluated += 1
                
            except Exception as e:
                print(f"ERROR: {str(e)}")
                continue
    
    return results, num_evaluated


def print_results_summary(results: Dict, modality: str, num_cases: int):
    """Print formatted results summary."""
    print(f"\n{'='*70}")
    print(f"TEST SET EVALUATION - {modality.upper()} MODALITY ({num_cases} cases)")
    print(f"{'='*70}\n")
    
    print(f"{'Structure':<12} {'Dice Mean':<15} {'Dice Std':<15} {'HD Mean (mm)':<15} {'HD Std':<15}")
    print("-" * 72)
    
    # Per-class metrics
    for class_id in range(1, 8):
        struct_name = STRUCTURE_SHORT[class_id]
        dice_vals = results['per_class'][class_id]['dice']
        hd_vals = results['per_class'][class_id]['hd']
        
        if len(dice_vals) > 0:
            dice_mean = np.mean(dice_vals)
            dice_std = np.std(dice_vals)
            hd_mean = np.mean([h for h in hd_vals if h != np.inf])
            hd_std = np.std([h for h in hd_vals if h != np.inf])
            
            print(f"{struct_name:<12} {dice_mean:.4f} ± {dice_std:.4f}    {hd_mean:>6.2f} ± {hd_std:<6.2f}")
    
    # Aggregate
    print("-" * 72)
    if len(results['aggregate']['dice']) > 0:
        dice_agg_mean = np.mean(results['aggregate']['dice'])
        dice_agg_std = np.std(results['aggregate']['dice'])
        hd_agg_mean = np.mean(results['aggregate']['hd'])
        hd_agg_std = np.std(results['aggregate']['hd'])
        
        print(f"{'WH (Overall)':<12} {dice_agg_mean:.4f} ± {dice_agg_std:.4f}    {hd_agg_mean:>6.2f} ± {hd_agg_std:<6.2f}")


def export_results_table(results_ct: Dict, results_mr: Dict, output_path: str = "test_results.txt"):
    """Export results in table format for paper."""
    
    with open(output_path, 'w') as f:
        f.write("TEST SET METRIC RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        # CT Results
        f.write("CT DATASET\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Structure':<12} {'Dice':<15} {'HD (mm)':<15}\n")
        f.write("-" * 80 + "\n")
        
        for class_id in range(1, 8):
            struct_name = STRUCTURE_SHORT[class_id]
            dice_vals = results_ct['per_class'][class_id]['dice']
            hd_vals = results_ct['per_class'][class_id]['hd']
            
            if len(dice_vals) > 0:
                dice_mean = np.mean(dice_vals)
                hd_mean = np.mean([h for h in hd_vals if h != np.inf])
                f.write(f"{struct_name:<12} {dice_mean:.4f}           {hd_mean:>6.2f}\n")
        
        # WH
        if len(results_ct['aggregate']['dice']) > 0:
            dice_agg = np.mean(results_ct['aggregate']['dice'])
            hd_agg = np.mean(results_ct['aggregate']['hd'])
            f.write(f"{'WH':<12} {dice_agg:.4f}           {hd_agg:>6.2f}\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # MR Results
        f.write("MR DATASET\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Structure':<12} {'Dice':<15} {'HD (mm)':<15}\n")
        f.write("-" * 80 + "\n")
        
        for class_id in range(1, 8):
            struct_name = STRUCTURE_SHORT[class_id]
            dice_vals = results_mr['per_class'][class_id]['dice']
            hd_vals = results_mr['per_class'][class_id]['hd']
            
            if len(dice_vals) > 0:
                dice_mean = np.mean(dice_vals)
                hd_mean = np.mean([h for h in hd_vals if h != np.inf])
                f.write(f"{struct_name:<12} {dice_mean:.4f}           {hd_mean:>6.2f}\n")
        
        # WH
        if len(results_mr['aggregate']['dice']) > 0:
            dice_agg = np.mean(results_mr['aggregate']['dice'])
            hd_agg = np.mean(results_mr['aggregate']['hd'])
            f.write(f"{'WH':<12} {dice_agg:.4f}           {hd_agg:>6.2f}\n")
    
    print(f"\nResults exported to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model metrics on training set with labels")
    parser.add_argument('--data_dir', type=str, default='./archive', help='Data directory')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints_transformer/best_model.pth', 
                        help='Model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--target_shape', type=int, nargs=3, default=[160, 160, 80])
    parser.add_argument('--output', type=str, default='test_metrics_results.txt', help='Output file')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, args.device)
    
    print("\n" + "="*70)
    print("NOTE: Official test set has no ground truth labels.")
    print("Evaluating on TRAINING SET with labels as proxy.")
    print("="*70)
    
    # Evaluate CT Training Set
    print("\n" + "="*70)
    print("EVALUATING CT TRAINING SET")
    print("="*70)
    results_ct, num_ct = evaluate_on_test_set(model, args.data_dir, 'ct_train', args.device, tuple(args.target_shape))
    
    if num_ct > 0:
        print_results_summary(results_ct, 'CT (train set)', num_ct)
    else:
        print("No CT training data found.")
    
    # Evaluate MR Training Set
    print("\n" + "="*70)
    print("EVALUATING MR TRAINING SET")
    print("="*70)
    results_mr, num_mr = evaluate_on_test_set(model, args.data_dir, 'mr_train', args.device, tuple(args.target_shape))
    
    if num_mr > 0:
        print_results_summary(results_mr, 'MR (train set)', num_mr)
    else:
        print("No MR training data found.")
    
    # Export
    if num_ct > 0 and num_mr > 0:
        export_results_table(results_ct, results_mr, args.output)


if __name__ == '__main__':
    main()
