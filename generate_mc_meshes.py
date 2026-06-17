"""
Generate Marching Cubes meshes for comparison with Swin+GAT predictions.
Shows both raw MC (artifacts) and smoothed MC for comparison.
"""

import nibabel as nib
import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
IMG_PATH = r"C:\Users\ravis\Desktop\heart_testing\archive\mr_train\mr_train_1006_image.nii"
MASK_PATH = r"C:\Users\ravis\Desktop\heart_testing\archive\mr_train\mr_train_1006_label.nii"

OUTPUT_FOLDER = "mc_comparison_meshes"
SAVE_MESHES = True

# Cardiac structures
STRUCTURES = {
    205: {"name": "Myocardium_LV",      "color": "mediumblue"},
    420: {"name": "Left_Atrium",        "color": "red"},
    500: {"name": "Left_Ventricle",     "color": "orange"},
    550: {"name": "Right_Atrium",       "color": "brown"},
    600: {"name": "Right_Ventricle",    "color": "lightgreen"},
    820: {"name": "Ascending_Aorta",    "color": "darkred"},
    850: {"name": "Pulmonary_Artery",   "color": "cyan"}
}

# ============================================================================
# FUNCTIONS
# ============================================================================

def create_raw_mc_mesh(mask_data, label_id, name):
    """Create raw Marching Cubes mesh (WITH artifacts)."""
    print(f"  - Processing RAW MC for {name} (Label {label_id})...", end=" ")
    binary_mask = (mask_data == label_id)
    
    if not np.any(binary_mask):
        print("⚠️  Not found.")
        return None
    
    try:
        verts, faces, _, _ = marching_cubes(binary_mask, level=0.5, step_size=1)
        formatted_faces = np.column_stack((np.full(len(faces), 3), faces)).flatten()
        mesh = pv.PolyData(verts, formatted_faces)
        print(f"✅ Created ({mesh.n_points} vertices, {mesh.n_cells} faces)")
        return mesh
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def create_smoothed_mc_mesh(mask_data, label_id, name, smooth_iter=50):
    """Create smoothed Marching Cubes mesh (standard post-processing)."""
    print(f"  - Processing SMOOTHED MC for {name} (Label {label_id})...", end=" ")
    binary_mask = (mask_data == label_id)
    
    if not np.any(binary_mask):
        print("⚠️  Not found.")
        return None
    
    try:
        verts, faces, _, _ = marching_cubes(binary_mask, level=0.5, step_size=1)
        formatted_faces = np.column_stack((np.full(len(faces), 3), faces)).flatten()
        mesh = pv.PolyData(verts, formatted_faces)
        mesh = mesh.smooth(n_iter=smooth_iter)
        print(f"✅ Created ({mesh.n_points} vertices, {mesh.n_cells} faces)")
        return mesh
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def compute_surface_roughness(mesh):
    """Compute mean surface curvature as roughness metric."""
    try:
        mesh_with_curv = mesh.compute_mean_curvature()
        curvatures = np.abs(mesh_with_curv['mean_curvature'])
        mean_curv = np.mean(curvatures)
        return mean_curv
    except:
        return None


def compute_edge_uniformity(mesh):
    """Compute coefficient of variation in edge lengths."""
    edges = mesh.extract_all_edges()
    edge_lengths = []
    for i in range(edges.n_cells):
        pts = edges.get_cell(i).GetPoints()
        p1 = np.array(pts.GetPoint(0))
        p2 = np.array(pts.GetPoint(1))
        edge_lengths.append(np.linalg.norm(p2 - p1))
    
    edge_lengths = np.array(edge_lengths)
    if len(edge_lengths) > 0:
        cv = np.std(edge_lengths) / np.mean(edge_lengths)
        return cv
    return None


def create_whole_heart_mesh(mask_data):
    """Create a single unified mesh of the entire heart."""
    print(f"  - Creating WHOLE HEART mesh (union of all structures)...", end=" ")
    
    # Binary mask where any cardiac structure exists
    cardiac_mask = np.zeros_like(mask_data)
    for label_id in STRUCTURES.keys():
        cardiac_mask[mask_data == label_id] = 1
    
    try:
        verts, faces, _, _ = marching_cubes(cardiac_mask, level=0.5, step_size=1)
        formatted_faces = np.column_stack((np.full(len(faces), 3), faces)).flatten()
        mesh = pv.PolyData(verts, formatted_faces)
        print(f"✅ Created ({mesh.n_points} vertices, {mesh.n_cells} faces)")
        return mesh
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def main():
    # Verify files exist
    if not os.path.exists(MASK_PATH):
        print(f"❌ Error: Could not find {MASK_PATH}")
        return
    
    # Load mask
    mask_obj = nib.load(MASK_PATH)
    mask_data = mask_obj.get_fdata()
    
    # Create output folder
    if SAVE_MESHES and not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    print("\n" + "="*70)
    print("GENERATING MARCHING CUBES MESHES FOR COMPARISON")
    print("="*70)
    
    # Storage for comparison
    mc_raw_meshes = {}
    mc_smooth_meshes = {}
    metrics = {}
    
    print("\n[1/2] Generating RAW Marching Cubes meshes (WITH artifacts)...")
    print("-"*70)
    for label_id, info in STRUCTURES.items():
        name = info['name']
        mesh = create_raw_mc_mesh(mask_data, label_id, name)
        if mesh:
            mc_raw_meshes[name] = mesh
            roughness = compute_surface_roughness(mesh)
            edge_cv = compute_edge_uniformity(mesh)
            metrics[f"{name}_raw"] = {"roughness": roughness, "edge_cv": edge_cv}
    
    # Generate WHOLE HEART meshes
    print("\n  *** WHOLE HEART (Union of all structures) ***")
    whole_heart_raw = create_whole_heart_mesh(mask_data)
    if whole_heart_raw:
        mc_raw_meshes["WHOLE_HEART"] = whole_heart_raw
        roughness = compute_surface_roughness(whole_heart_raw)
        edge_cv = compute_edge_uniformity(whole_heart_raw)
        metrics["WHOLE_HEART_raw"] = {"roughness": roughness, "edge_cv": edge_cv}
    
    print("\n[2/2] Generating SMOOTHED Marching Cubes meshes (post-processing)...")
    print("-"*70)
    for label_id, info in STRUCTURES.items():
        name = info['name']
        mesh = create_smoothed_mc_mesh(mask_data, label_id, name, smooth_iter=50)
        if mesh:
            mc_smooth_meshes[name] = mesh
            roughness = compute_surface_roughness(mesh)
            edge_cv = compute_edge_uniformity(mesh)
            metrics[f"{name}_smooth"] = {"roughness": roughness, "edge_cv": edge_cv}
    
    # Generate SMOOTHED whole heart mesh
    print("\n  *** WHOLE HEART (Union of all structures - SMOOTHED) ***")
    if whole_heart_raw:
        whole_heart_smooth = whole_heart_raw.smooth(n_iter=50)
        mc_smooth_meshes["WHOLE_HEART"] = whole_heart_smooth
        roughness = compute_surface_roughness(whole_heart_smooth)
        edge_cv = compute_edge_uniformity(whole_heart_smooth)
        metrics["WHOLE_HEART_smooth"] = {"roughness": roughness, "edge_cv": edge_cv}
    
    # Save meshes
    if SAVE_MESHES:
        print("\n[3/3] Saving meshes to disk...")
        print("-"*70)
        
        for name, mesh in mc_raw_meshes.items():
            filename = os.path.join(OUTPUT_FOLDER, f"{name}_RAW_MC.obj")
            mesh.save(filename)
            print(f"  ✅ Saved: {filename}")
        
        for name, mesh in mc_smooth_meshes.items():
            filename = os.path.join(OUTPUT_FOLDER, f"{name}_SMOOTHED_MC.obj")
            mesh.save(filename)
            print(f"  ✅ Saved: {filename}")
    
    # Print comparison metrics
    print("\n" + "="*70)
    print("SURFACE ROUGHNESS COMPARISON (Mean Absolute Curvature)")
    print("="*70)
    print(f"{'Structure':<30} {'Raw MC':<15} {'Smoothed MC':<15} {'Reduction'}")
    print("-"*70)
    for name in mc_raw_meshes.keys():
        raw_curv = metrics.get(f"{name}_raw", {}).get("roughness")
        smooth_curv = metrics.get(f"{name}_smooth", {}).get("roughness")
        if raw_curv and smooth_curv:
            reduction = (1 - smooth_curv/raw_curv) * 100
            print(f"{name:<30} {raw_curv:<15.4f} {smooth_curv:<15.4f} {reduction:.1f}%")
    
    print("\n" + "="*70)
    print("EDGE UNIFORMITY COMPARISON (Coefficient of Variation)")
    print("="*70)
    print(f"{'Structure':<30} {'Raw MC':<15} {'Smoothed MC':<15} {'Reduction'}")
    print("-"*70)
    for name in mc_raw_meshes.keys():
        raw_cv = metrics.get(f"{name}_raw", {}).get("edge_cv")
        smooth_cv = metrics.get(f"{name}_smooth", {}).get("edge_cv")
        if raw_cv and smooth_cv:
            reduction = (1 - smooth_cv/raw_cv) * 100
            print(f"{name:<30} {raw_cv:<15.4f} {smooth_cv:<15.4f} {reduction:.1f}%")
    
    print("\n" + "="*70)
    print(f"All meshes saved to: {os.path.abspath(OUTPUT_FOLDER)}")
    print("="*70)
    print("\nNext steps:")
    print("1. Generate your Swin+GAT predicted meshes")
    print("2. Compare side-by-side visually in ParaView or similar")
    print("3. The artifacts in RAW MC should be clearly visible")
    print("4. Smoothing reduces artifacts but still not as clean as Swin+GAT")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
