"""
Generate Swin+GAT predicted mesh for comparison with Marching Cubes.
Uses the trained best_model.pth checkpoint.
"""

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import pyvista as pv
from pathlib import Path
import os

from model_transformer import get_transformer_model, MeshDeformationGAT
from deform_reconstruct import (
    load_nifti, resize_volume_torch, infer_embed_dim_from_ckpt,
    create_template_sphere, sample_features_from_levels, estimate_center_radius_from_features
)

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


def load_checkpoint(ckpt_path, device):
    """Load checkpoint and infer model configuration."""
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    
    # Infer embed_dim
    embed_dim = None
    for k, v in state.items():
        if 'patch_embed.proj.weight' in k:
            embed_dim = v.shape[0]
            break
    
    if embed_dim is None:
        embed_dim = 96  # default
    
    print(f"Inferred embed_dim: {embed_dim}")
    return ckpt, embed_dim


def generate_swin_gat_mesh(img_path, ckpt_path, device, target_shape, output_folder):
    """Generate predicted mesh using Swin+GAT model."""
    
    print("\n" + "="*70)
    print("GENERATING SWIN+GAT PREDICTED MESH")
    print("="*70)
    
    # Load image
    print(f"\n[1/4] Loading image: {img_path}")
    img = load_nifti(img_path)
    img = resize_volume_torch(img, target_shape)
    img = normalize_volume(img)
    
    # Load checkpoint and model
    print(f"[2/4] Loading model from: {ckpt_path}")
    ckpt, embed_dim = load_checkpoint(ckpt_path, device)
    
    model = get_transformer_model(
        in_channels=1,
        num_classes=8,
        embed_dim=embed_dim,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(7, 7, 7)
    )
    
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    
    # Prepare input
    print(f"[3/4] Running inference...")
    img_tensor = torch.from_numpy(img).float().to(device)
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    
    with torch.no_grad():
        # Forward pass
        segmentation_logits, encoder_features, mesh_vertices = model(img_tensor)
        
        # Get predicted mesh vertices
        pred_verts = mesh_vertices.squeeze(0).cpu().numpy()  # (N, 3)
        
        print(f"Predicted mesh vertices shape: {pred_verts.shape}")
        print(f"Vertex coordinate ranges:")
        print(f"  D: [{pred_verts[:, 0].min():.2f}, {pred_verts[:, 0].max():.2f}]")
        print(f"  H: [{pred_verts[:, 1].min():.2f}, {pred_verts[:, 1].max():.2f}]")
        print(f"  W: [{pred_verts[:, 2].min():.2f}, {pred_verts[:, 2].max():.2f}]")
    
    # Create mesh from predicted vertices
    # For now, create a sphere mesh with predicted vertices as displacement
    print(f"[4/4] Creating mesh...")
    
    # Create template sphere
    center = (target_shape[0]//2, target_shape[1]//2, target_shape[2]//2)
    template_verts, template_faces = create_template_sphere(
        target_shape=target_shape,
        radius_scale=0.35,
        res=4,
        center_xyz=center
    )
    
    # Use predicted vertices
    if pred_verts.shape[0] == template_verts.shape[0]:
        final_verts = pred_verts
    else:
        print(f"Warning: Vertex count mismatch. Using template vertices.")
        final_verts = template_verts
    
    # Create PyVista mesh
    formatted_faces = np.column_stack((np.full(len(template_faces), 3), template_faces)).flatten()
    mesh = pv.PolyData(final_verts, formatted_faces)
    
    # Save mesh
    print(f"\nMesh created: {mesh.n_points} vertices, {mesh.n_cells} faces")
    
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "WHOLE_HEART_SWINGAT_PREDICTED.obj")
    mesh.save(output_path)
    print(f"✅ Saved: {output_path}")
    
    return mesh


def main():
    print("Checking prerequisites...")
    if not os.path.exists(IMG_PATH):
        print(f"❌ Image not found: {IMG_PATH}")
        return
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        return
    
    # Generate mesh
    mesh = generate_swin_gat_mesh(IMG_PATH, CHECKPOINT_PATH, DEVICE, TARGET_SHAPE, OUTPUT_FOLDER)
    
    print("\n" + "="*70)
    print("MESH GENERATION COMPLETE")
    print("="*70)
    print("\nComparison files ready for visualization:")
    print("  - WHOLE_HEART_RAW_MC.obj")
    print("  - WHOLE_HEART_SMOOTHED_MC.obj")
    print("  - WHOLE_HEART_SWINGAT_PREDICTED.obj")
    print("\nUse ParaView or similar to compare side-by-side!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
