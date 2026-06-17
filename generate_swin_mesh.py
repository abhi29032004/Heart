"""
Generate Swin Transformer predicted meshes (via segmentation) for comparison.
This generates meshes from the model's segmentation output using Marching Cubes.
"""

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import pyvista as pv
from skimage.measure import marching_cubes
import os

from model_transformer import get_transformer_model

# ============================================================================
# CONFIGURATION
# ============================================================================
IMG_PATH = r"C:\Users\ravis\Desktop\heart_testing\archive\mr_train\mr_train_1006_image.nii"
CHECKPOINT_PATH = r"C:\Users\ravis\Desktop\heart\checkpoints_transformer\best_model.pth"
OUTPUT_FOLDER = "mc_comparison_meshes"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SHAPE = (160, 160, 80)

# ============================================================================
# FUNCTIONS
# ============================================================================

def normalize_volume(volume):
    """Z-score normalization."""
    mu = volume.mean()
    sigma = volume.std() + 1e-8
    return (volume - mu) / sigma


def load_nifti(path):
    """Load NIfTI file."""
    return nib.load(path).get_fdata()


def resize_volume(vol, target_shape):
    """Resize volume using trilinear interpolation."""
    t = torch.from_numpy(vol.astype(np.float32))[None, None]
    out = F.interpolate(t, size=target_shape, mode='trilinear', align_corners=False)[0,0].cpu().numpy()
    if out.max() > out.min():
        out = (out - out.min()) / (out.max() - out.min())
    return out


def load_checkpoint(ckpt_path, device):
    """Load checkpoint and infer configuration."""
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    
    # Infer embed_dim from patch embedding
    embed_dim = 48  # Based on checkpoint inspection
    for k, v in state.items():
        if 'patch_embed.proj.weight' in k:
            embed_dim = v.shape[0]  # Output channels of proj layer
            break
    
    print(f"Inferred embed_dim from checkpoint: {embed_dim}")
    return ckpt, embed_dim


def compute_metrics(mesh):
    """Compute surface roughness and edge uniformity metrics."""
    try:
        mesh_with_curv = mesh.compute_mean_curvature()
        curvatures = np.abs(mesh_with_curv['mean_curvature'])
        mean_curv = np.mean(curvatures)
    except:
        mean_curv = None
    
    # Edge uniformity
    try:
        edges = mesh.extract_all_edges()
        edge_lengths = []
        for i in range(edges.n_cells):
            pts = edges.get_cell(i).GetPoints()
            p1 = np.array(pts.GetPoint(0))
            p2 = np.array(pts.GetPoint(1))
            edge_lengths.append(np.linalg.norm(p2 - p1))
        
        edge_lengths = np.array(edge_lengths)
        if len(edge_lengths) > 0:
            cv = np.std(edge_lengths) / (np.mean(edge_lengths) + 1e-9)
        else:
            cv = None
    except:
        cv = None
    
    return mean_curv, cv


def generate_swin_mesh(img_path, ckpt_path, device, target_shape, output_folder):
    """Generate mesh from Swin model's segmentation output."""
    
    print("\n" + "="*70)
    print("GENERATING SWIN TRANSFORMER MESH (from segmentation)")
    print("="*70)
    
    # Load image
    print(f"\n[1/4] Loading image: {img_path}")
    img = load_nifti(img_path)
    img = resize_volume(img, target_shape)
    img_orig = img.copy()
    img = normalize_volume(img)
    
    # Load model
    print(f"[2/4] Loading model from: {ckpt_path}")
    ckpt, embed_dim = load_checkpoint(ckpt_path, device)
    
    model = get_transformer_model(
        in_channels=1,
        num_classes=8,
        embed_dim=embed_dim,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(7, 7, 7),
        decoder_dim=256
    )
    
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    # Remove 'module.' prefix if present and filter out mesh_gat keys
    cleaned_state = {}
    for k, v in state.items():
        # Skip mesh_gat related keys to avoid mismatch
        if 'mesh_gat' in k:
            continue
        new_k = k[7:] if k.startswith('module.') else k
        cleaned_state[new_k] = v
    
    # Load only encoder and decoder weights
    model.load_state_dict(cleaned_state, strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device} (encoder and decoder only)")
    
    # Run inference
    print(f"[3/4] Running inference...")
    img_tensor = torch.from_numpy(img).float().to(device)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    
    with torch.no_grad():
        output = model(img_tensor)
        
        # Handle both single output and tuple output
        if isinstance(output, (list, tuple)):
            seg_logits = output[0]
        else:
            seg_logits = output
        
        # Get predicted segmentation
        seg_probs = torch.softmax(seg_logits, dim=1)  # (1, 8, D, H, W)
        seg_pred = torch.argmax(seg_probs, dim=1)[0].cpu().numpy()  # (D, H, W)
    
    print(f"Segmentation shape: {seg_pred.shape}")
    print(f"Classes present: {np.unique(seg_pred)}")
    
    # Create unified heart mask
    cardiac_mask = (seg_pred > 0).astype(np.uint8)
    
    # Generate mesh from segmentation
    print(f"[4/4] Creating mesh from segmentation...")
    try:
        verts, faces, _, _ = marching_cubes(cardiac_mask, level=0.5, step_size=1)
        formatted_faces = np.column_stack((np.full(len(faces), 3), faces)).flatten()
        mesh = pv.PolyData(verts, formatted_faces)
        
        print(f"Mesh created from segmentation: {mesh.n_points} vertices, {mesh.n_cells} faces")
        
        # Compute metrics
        mean_curv, edge_cv = compute_metrics(mesh)
        
        # Save meshes
        os.makedirs(output_folder, exist_ok=True)
        
        # Raw segmentation-based mesh
        output_path_raw = os.path.join(output_folder, "WHOLE_HEART_SWIN_SEGMC.obj")
        mesh.save(output_path_raw)
        print(f"✅ Saved RAW segmentation mesh: {output_path_raw}")
        if mean_curv:
            print(f"   Metrics - Curvature: {mean_curv:.4f}, Edge CV: {edge_cv:.4f if edge_cv else 'N/A'}")
        else:
            print(f"   Metrics - Could not compute surface metrics")
        
        # Smoothed version
        mesh_smooth = mesh.smooth(n_iter=50)
        output_path_smooth = os.path.join(output_folder, "WHOLE_HEART_SWIN_SEGMC_SMOOTH.obj")
        mesh_smooth.save(output_path_smooth)
        print(f"✅ Saved SMOOTHED segmentation mesh: {output_path_smooth}")
        
        mean_curv_s, edge_cv_s = compute_metrics(mesh_smooth)
        if mean_curv_s:
            print(f"   Metrics - Curvature: {mean_curv_s:.4f}, Edge CV: {edge_cv_s:.4f if edge_cv_s else 'N/A'}")
        else:
            print(f"   Metrics - Could not compute surface metrics")
        
        return mesh, mesh_smooth
        
    except Exception as e:
        print(f"❌ Error creating mesh: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    print("Checking prerequisites...")
    if not os.path.exists(IMG_PATH):
        print(f"❌ Image not found: {IMG_PATH}")
        return
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        print(f"   Checked: {CHECKPOINT_PATH}")
        return
    
    # Generate meshes
    mesh_raw, mesh_smooth = generate_swin_mesh(IMG_PATH, CHECKPOINT_PATH, DEVICE, TARGET_SHAPE, OUTPUT_FOLDER)
    
    if mesh_raw is not None:
        print("\n" + "="*70)
        print("MESH GENERATION COMPLETE")
        print("="*70)
        print("\n✅ Comparison files ready for visualization:")
        print("   - WHOLE_HEART_RAW_MC.obj (Raw Marching Cubes)")
        print("   - WHOLE_HEART_SMOOTHED_MC.obj (Smoothed MC)")
        print("   - WHOLE_HEART_SWIN_SEGMC.obj (Swin segmentation → MC)")
        print("   - WHOLE_HEART_SWIN_SEGMC_SMOOTH.obj (Swin segmentation → MC → Smoothed)")
        print("\nUse ParaView or similar to compare side-by-side!")
        print("="*70 + "\n")
    else:
        print("❌ Failed to generate meshes")


if __name__ == "__main__":
    main()
