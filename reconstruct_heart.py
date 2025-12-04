r"""
3D Heart Reconstruction and Interactive Visualization

Loads a trained 3D Swin Transformer + GAT checkpoint, runs segmentation on a
NIfTI CT/MR volume, reconstructs per-structure 3D meshes, optionally refines
meshes with a Graph Attention Network, and launches an interactive PyVista
viewer. Also saves meshes (.vtp) and a screenshot.

Classes (labels):
  0: Background
  1: Left Ventricle (LV)
  2: Right Ventricle (RV)
  3: Left Atrium (LA)
  4: Right Atrium (RA)
  5: Myocardium
  6: Aorta (AO)
  7: Pulmonary Artery (PA)

Usage (Windows PowerShell):
    python reconstruct_heart.py --input .\archive\ct_test\ct_test_2001_image.nii \
        --checkpoint .\checkpoints_transformer\best_model.pth --device auto

If --input is omitted, the script will use the first file in archive/ct_test or mr_test.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    import nibabel as nib
except Exception as e:  # pragma: no cover
    nib = None

try:
    from skimage import measure, morphology
except Exception:
    measure = None
    morphology = None

# Light-weight scipy.ndimage ops for 3D morphology and smoothing
try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None

# PyVista/VTK are used for interactive 3D visualization
try:
    import pyvista as pv
except Exception:
    pv = None

# Local imports
from model_transformer import get_transformer_model

# ------------ Constants ------------
STRUCTURE_NAMES: Dict[int, str] = {
    1: "LV", 2: "RV", 3: "LA", 4: "RA", 5: "Myocardium", 6: "Aorta", 7: "PA"
}
STRUCTURE_COLORS: Dict[int, Tuple[float, float, float]] = {
    1: (0.85, 0.1, 0.1),      # LV - red
    2: (0.1, 0.8, 0.2),       # RV - green
    3: (0.2, 0.4, 0.9),       # LA - blue
    4: (0.95, 0.75, 0.2),     # RA - yellow
    5: (0.9, 0.2, 0.8),       # Myocardium - magenta
    6: (0.55, 0.85, 0.95),    # Aorta - cyan-ish
    7: (0.8, 0.15, 0.05),     # PA - dark red/orange
}

DEFAULT_TARGET_SHAPE = (160, 160, 80)  # D, H, W as in training

# ------------ Utilities ------------

def find_default_input(data_dir: str) -> Optional[str]:
    """Pick the first *_image.nii from ct_test or mr_test."""
    candidates = []
    for sub in ("ct_test", "mr_test"):
        folder = Path(data_dir) / sub
        if folder.exists():
            for f in sorted(folder.glob("*_image.nii")):
                if f.is_file():
                    candidates.append(str(f))
    return candidates[0] if candidates else None


def ensure_dirs(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_nifti(path: str) -> np.ndarray:
    assert nib is not None, "nibabel is required. Please install nibabel."
    nii = nib.load(path)
    return nii.get_fdata()


def resize_volume(volume: np.ndarray, target_shape: Tuple[int, int, int], is_label: bool=False) -> np.ndarray:
    """Resize a 3D volume to target shape using scipy.ndimage.zoom-like behavior.
    We avoid importing scipy here to keep dependencies minimal at runtime. If
    scipy is available via dataset module, you can switch to that; however,
    PyTorch's interpolate can also be used for images.
    """
    # Use PyTorch interpolate for reliability and speed
    vol = torch.from_numpy(volume.astype(np.float32))[None, None]  # (1,1,D,H,W)
    mode = 'nearest' if is_label else 'trilinear'
    vol = F.interpolate(vol, size=target_shape, mode=mode, align_corners=False if mode != 'nearest' else None)
    out = vol[0, 0].cpu().numpy()
    if is_label:
        out = np.rint(out).astype(np.int64)  # preserve discrete labels if used
    return out


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    v = volume.astype(np.float32)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax > vmin:
        v = (v - vmin) / (vmax - vmin)
    return v


def preprocess_image(img_path: str, target_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Load, resize and normalize a NIfTI image.
    Returns (image_dhw_float_in_0_1, original_image_dhw)
    """
    vol = load_nifti(img_path)  # (D,H,W) or (H,W,D) depending on dataset; ours uses (D,H,W)
    # If axis order is ambiguous, try to detect overly long first axis; but we'll assume (D,H,W)
    orig = vol.copy()
    vol = resize_volume(vol, target_shape, is_label=False)
    vol = normalize_volume(vol)
    return vol, orig


# ------------ Model loading & inference ------------

def load_model(checkpoint_path: str, device: torch.device, embed_dim: int = -1,
               depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24), window_size=(7, 7, 7),
               decoder_dim: int = 256):
    """Create the transformer model and load weights from checkpoint.
    - If embed_dim == -1, auto-infer it from the checkpoint.
    - Loads only compatible tensor shapes to avoid size-mismatch errors.
    Returns the model in eval mode.
    """
    if not os.path.isfile(checkpoint_path):
        # Try to suggest alternatives
        parent = Path(checkpoint_path).parent
        suggestions = "\n  ".join([p.name for p in parent.glob("*.pth")]) if parent.exists() else ""
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Available in folder:\n  {suggestions}"
        )

    ckpt = torch.load(checkpoint_path, map_location=device)
    # Try to extract the actual state_dict from common training wrappers
    state = None
    if isinstance(ckpt, dict):
        # Preferred keys in order
        for k in [
            'model_state_dict',          # torch.save({'model_state_dict': model.state_dict(), ...})
            'state_dict',                # common convention
            'model',                     # some trainers
            'net', 'weights'
        ]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        if state is None:
            # If dict looks like a state_dict already (keys with dots and tensors)
            looks_like_state = all(isinstance(v, torch.Tensor) for v in ckpt.values())
            state = ckpt if looks_like_state else None
    else:
        # Some code saves state_dict directly
        state = ckpt if isinstance(ckpt, dict) else None

    if state is None:
        raise RuntimeError(
            "Could not locate a state_dict in the checkpoint. Supported keys: "
            "model_state_dict, state_dict, model, net, weights."
        )
    # Strip 'module.' prefix if present
    norm_state = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith('module.') else k
        norm_state[nk] = v

    # Auto-infer embed_dim when not specified
    inferred_embed = None
    for probe_key in (
        'encoder.patch_embed.proj.weight',
        'encoder.patch_embed.norm.weight',
    ):
        if probe_key in norm_state:
            t = norm_state[probe_key]
            inferred_embed = int(t.shape[0])
            break

    used_embed_dim = embed_dim if embed_dim != -1 else (inferred_embed or 96)
    if embed_dim == -1:
        print(f"[load_model] Auto-inferred embed_dim={used_embed_dim} from checkpoint")
    elif inferred_embed is not None and inferred_embed != embed_dim:
        print(f"[load_model] WARNING: CLI embed_dim={embed_dim} != checkpoint embed_dim≈{inferred_embed}. Loading compatible tensors only.")

    # Build model with the chosen embed_dim
    model = get_transformer_model(
        in_channels=1,
        num_classes=8,
        embed_dim=used_embed_dim,
        depths=list(depths),
        num_heads=list(num_heads),
        window_size=tuple(window_size),
        decoder_dim=decoder_dim,
    )
    model.to(device)

    # Load only matching shapes to avoid RuntimeError on mismatches
    model_state = model.state_dict()
    compatible = {}
    skipped = []
    for k, v in norm_state.items():
        if k in model_state and model_state[k].shape == v.shape:
            compatible[k] = v
        else:
            skipped.append(k)

    res = model.load_state_dict(compatible, strict=False)
    if getattr(res, 'missing_keys', []):
        print(f"[load_model] Missing keys: {len(res.missing_keys)} (ok)")
    if getattr(res, 'unexpected_keys', []):
        print(f"[load_model] Unexpected keys: {len(res.unexpected_keys)} (ok)")
    if skipped:
        print(f"[load_model] Skipped incompatible tensors: {len(skipped)}")

    model.eval()
    return model


def predict(model: torch.nn.Module, image_dhw: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Run a forward pass and return (labels, probs_np).
    labels: (D,H,W) int16 in 0..7
    probs_np: (8,D,H,W) float32 per-class probabilities
    """
    with torch.no_grad():
        x = torch.from_numpy(image_dhw[None, None].astype(np.float32)).to(device)
        logits = model(x)  # (1,8,D,H,W)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)  # (1,D,H,W)
        labels = pred[0].cpu().numpy().astype(np.int16)
        probs_np = probs[0].cpu().numpy().astype(np.float32)
    return labels, probs_np


def extract_deep_features(model: torch.nn.Module, image_dhw: np.ndarray, device: torch.device) -> torch.Tensor:
    """Run encoder to obtain deepest feature map for GAT refinement.
    Returns tensor of shape (1, C, D', H', W').
    """
    with torch.no_grad():
        x = torch.from_numpy(image_dhw[None, None].astype(np.float32)).to(device)
        encoder_features = model.encoder(x)
        deep = encoder_features[-1]  # (1,C,D',H',W')
    return deep


# ------------ Mesh reconstruction & optional GAT refinement ------------

def marching_cubes_from_label(label_dhw: np.ndarray, class_id: int, step_size: int = 1,
                               min_voxels: int = 500) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Run marching cubes on a single class mask and return (verts, faces).
    Returns None if not enough voxels.
    """
    assert measure is not None, "scikit-image is required for marching cubes. Please install scikit-image."
    mask = (label_dhw == class_id)
    voxels = int(mask.sum())
    if min_voxels and voxels < min_voxels:
        return None
    # marching_cubes requires a binary volume with both foreground and background present
    if not (mask.any() and (~mask).any()):
        return None
    # Optional small-object removal for cleaner surfaces
    try:
        if morphology is not None:
            mask = morphology.remove_small_objects(mask, min_size=min_voxels, connectivity=3)
    except Exception:
        pass

    vol = mask.astype(np.float32)
    # Slight Gaussian smoothing of the binary mask improves surface quality
    try:
        if ndi is not None:
            vol = ndi.gaussian_filter(vol, sigma=1.0)
    except Exception:
        pass
    try:
        verts, faces, normals, _ = measure.marching_cubes(vol, level=0.5, step_size=step_size, spacing=(1.0, 1.0, 1.0))
        return verts, faces.astype(np.int32)
    except Exception as e:
        print(f"[marching_cubes] class {class_id} failed: {e}")
        return None


def sample_vertex_features(deep_feat: torch.Tensor, verts: np.ndarray, orig_shape: Tuple[int,int,int]) -> torch.Tensor:
    """Sample deepest encoder features at vertex locations using trilinear sampling.
    Args:
        deep_feat: (1, C, D', H', W')
        verts: (N, 3) in (z,y,x) voxel coordinates at original resolution
        orig_shape: (D, H, W) original resolution used for model input
    Returns:
        (N, C) torch tensor of sampled features
    """
    device = deep_feat.device
    _, C, Dp, Hp, Wp = deep_feat.shape
    D, H, W = orig_shape

    # Map original voxel coords -> deepest feature coords by global downsample factor
    # Total downsample ~ 4 (patch) * 2 * 2 * 2 = 32
    ds = 32.0
    vz = (verts[:, 0] / ds)
    vy = (verts[:, 1] / ds)
    vx = (verts[:, 2] / ds)

    # Normalize to [-1, 1] for grid_sample with index order (x, y, z) mapped to (W, H, D)
    # grid expects (z,y,x) in normalized coords last dim order is (x,y,z)
    gz = (vz / max(Dp - 1, 1)) * 2 - 1
    gy = (vy / max(Hp - 1, 1)) * 2 - 1
    gx = (vx / max(Wp - 1, 1)) * 2 - 1

    grid = np.stack([gx, gy, gz], axis=1).astype(np.float32)  # (N, 3) order (x,y,z)
    grid = torch.from_numpy(grid).to(device)[None, None, None]  # (1,1,1,N,3)
    # Reshape to sample N points along H_out dimension
    grid = grid.permute(0, 1, 3, 2, 4)  # (1,1,N,1,3)

    sampled = F.grid_sample(deep_feat, grid, align_corners=True, mode='bilinear', padding_mode='border')
    # sampled shape: (1, C, 1, N, 1) -> pick D_out=0, keep H_out=N
    feat = sampled[0, :, 0, :, 0].transpose(0, 1).contiguous()  # (N, C)
    return feat


def refine_with_gat_if_available(model: torch.nn.Module, deep_feat: torch.Tensor,
                                 verts: np.ndarray, faces: np.ndarray,
                                 orig_shape: Tuple[int,int,int]) -> np.ndarray:
    """Use model.mesh_gat to refine vertices based on deep features.
    If any step fails, falls back to original vertices.
    """
    try:
        # sample features per vertex
        vert_feat = sample_vertex_features(deep_feat, verts, orig_shape)  # (N,C)
        v = torch.from_numpy(verts.astype(np.float32)).to(deep_feat.device)
        f = torch.from_numpy(faces.astype(np.int64)).to(deep_feat.device)
        with torch.no_grad():
            refined = model.refine_mesh(v, vert_feat, f).cpu().numpy()
        return refined
    except Exception as e:
        print(f"[GAT refine] fallback (reason: {e})")
        return verts


# ------------ Visualization ------------

def to_pyvista_mesh(verts: np.ndarray, faces: np.ndarray) -> Any:
    # PyVista expects a faces array with counts: [3, i0, i1, i2, 3, j0, j1, j2, ...]
    if pv is None:
        raise RuntimeError("pyvista is required for visualization. Please install pyvista and vtk.")
    face_sizes = np.full((faces.shape[0], 1), 3, dtype=np.int32)
    faces_pv = np.hstack([face_sizes, faces.astype(np.int32)]).reshape(-1)
    mesh = pv.PolyData(verts[:, [2, 1, 0]], faces_pv)  # swap to (x,y,z) for VTK
    # Optional Taubin/Laplacian smoothing and normal recomputation for nicer look
    try:
        if hasattr(mesh, 'smooth_taubin'):
            mesh = mesh.smooth_taubin(n_iter=30, pass_band=0.1)
        else:
            mesh = mesh.smooth(n_iter=30, relaxation_factor=0.1, feature_smoothing=True, boundary_smoothing=True)
    except Exception:
        pass
    mesh.compute_normals(inplace=True)
    return mesh


# ------------ Label post-processing for cleaner heart shapes ------------

def keep_largest_cc(mask: np.ndarray) -> np.ndarray:
    if ndi is None:
        return mask
    try:
        labeled, n = ndi.label(mask)
        if n <= 1:
            return mask
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # background
        keep = sizes.argmax()
        return (labeled == keep)
    except Exception:
        return mask


def postprocess_labels(labels: np.ndarray,
                       probs: Optional[np.ndarray] = None,
                       prob_thresh: Optional[Dict[int, float]] = None,
                       class_min_voxels: Optional[Dict[int, int]] = None,
                       morph_radius: int = 2) -> np.ndarray:
    """Clean up raw argmax labels to look more like anatomical structures.
    - Optional per-class probability thresholding
    - Keep largest connected component per class
    - 3D closing/opening to remove spikes and holes
    - 3D hole filling
    """
    cleaned = np.zeros_like(labels, dtype=np.int16)
    if prob_thresh is None:
        # Slightly conservative thresholds to suppress stray voxels
        prob_thresh = {1: 0.40, 2: 0.40, 3: 0.35, 4: 0.35, 5: 0.45, 6: 0.30, 7: 0.30}
    if class_min_voxels is None:
        class_min_voxels = {1: 3000, 2: 3000, 3: 1500, 4: 1500, 5: 4000, 6: 800, 7: 800}

    selem = None
    if morphology is not None and hasattr(morphology, 'ball'):
        try:
            selem = morphology.ball(max(1, int(morph_radius)))
        except Exception:
            selem = None

    for cid in range(1, 8):
        mask = (labels == cid)
        # probability gating
        if probs is not None and cid < probs.shape[0]:
            mask &= (probs[cid] >= prob_thresh.get(cid, 0.35))

        # Remove tiny speckles early
        try:
            if morphology is not None:
                mask = morphology.remove_small_objects(mask, min_size=class_min_voxels.get(cid, 500), connectivity=3)
        except Exception:
            pass

        # Morphological closing -> opening to smooth boundaries
        try:
            if ndi is not None:
                if selem is not None:
                    mask = ndi.binary_closing(mask, structure=selem)
                    mask = ndi.binary_opening(mask, structure=selem)
                else:
                    mask = ndi.binary_closing(mask)
        except Exception:
            pass

        # Fill holes and keep largest component
        try:
            if ndi is not None:
                mask = ndi.binary_fill_holes(mask)
        except Exception:
            pass
        mask = keep_largest_cc(mask)

        cleaned[mask] = cid

    return cleaned


def visualize_interactive(meshes: Dict[int, Any], screenshot_path: str) -> None:
    """Interactive PyVista viewer with:
    - Per-structure toggles (checkboxes and number-key shortcuts)
    - Global opacity slider
    - Solo/All buttons to focus on a single structure
    - Quick Screenshot button
    """
    plotter = pv.Plotter(window_size=(1200, 850))
    plotter.set_background('black')
    plotter.enable_anti_aliasing('fxaa')

    # Add meshes
    actors: Dict[int, object] = {}
    default_opacity = 0.9
    for cid, mesh in meshes.items():
        color = STRUCTURE_COLORS.get(cid, (0.8, 0.8, 0.8))
        actor = plotter.add_mesh(
            mesh,
            color=color,
            opacity=default_opacity,
            smooth_shading=True,
            specular=0.15,
            name=STRUCTURE_NAMES[cid],
        )
        actors[cid] = actor

    # Legend, axes, bounds
    plotter.add_axes(line_width=1, labels_off=False)
    plotter.show_bounds(grid='front', location='outer', ticks='outside', color='white')
    plotter.add_text("3D Heart Reconstruction (Swin + GAT)", position='upper_left', font_size=12, color='white')
    plotter.camera_position = 'yz'
    plotter.enable_eye_dome_lighting()

    # ---------- Widgets ----------
    # Per-structure checkboxes
    toggles: Dict[int, int] = {}
    for i, cid in enumerate(meshes.keys()):
        def make_cb(_cid: int):
            def _cb(state):
                actors[_cid].SetVisibility(bool(state))
            return _cb
        # Stagger vertically along the left
        wid = plotter.add_checkbox_button_widget(
            make_cb(cid), value=True, position=(10, int(10 + i * 34))
        )
        toggles[cid] = wid
        plotter.add_text(
            f"{STRUCTURE_NAMES[cid]}", position=(40, int(10 + i * 34) + 3), color='white', font_size=10
        )

    # Global opacity slider
    def on_opacity(value):
        for a in actors.values():
            a.GetProperty().SetOpacity(float(value))
    plotter.add_slider_widget(
        on_opacity,
        rng=[0.2, 1.0],
        value=default_opacity,
        title='Opacity',
        pointa=(0.02, 0.06),
        pointb=(0.32, 0.06),
        style='modern',
        title_height=0.02,
    )

    # Solo cycle (keyboard) and show-all bindings
    ids = list(meshes.keys())
    solo_idx = {'i': -1}  # mutable closure

    def solo_next():
        if not ids:
            return
        solo_idx['i'] = (solo_idx['i'] + 1) % len(ids)
        cid = ids[solo_idx['i']]
        for k, a in actors.items():
            a.SetVisibility(k == cid)
        print(f"Solo: {STRUCTURE_NAMES[cid]}")

    def show_all():
        for a in actors.values():
            a.SetVisibility(True)
        print("Show all structures")

    # Save screenshot function (can be bound to a key)
    def save_shot():
        try:
            ensure_dirs(os.path.dirname(screenshot_path))
            plotter.screenshot(screenshot_path)
            print(f"✓ Screenshot saved: {screenshot_path}")
        except Exception as e:
            print(f"[visualize] Could not save screenshot: {e}")

    # Add on-screen controls text instead of unsupported button widgets
    controls_text = (
        "Controls: [S] Solo cycle  [A] Show all  [P] Save PNG  [1-7] Toggle structures  [0] Show all"
    )
    plotter.add_text(controls_text, position='lower_left', font_size=10, color='white')

    # Keyboard bindings for controls
    plotter.add_key_event('s', solo_next)
    plotter.add_key_event('S', solo_next)
    plotter.add_key_event('a', show_all)
    plotter.add_key_event('A', show_all)
    plotter.add_key_event('p', save_shot)
    plotter.add_key_event('P', save_shot)

    # Keyboard shortcuts: 1..7 to toggle, 0 to show all
    def make_toggle_key(_cid: int):
        def _handler():
            vis = actors[_cid].GetVisibility() == 0
            actors[_cid].SetVisibility(vis)
        return _handler

    for idx, cid in enumerate(ids, start=1):
        key = str(idx) if idx < 10 else None
        if key:
            plotter.add_key_event(key, make_toggle_key(cid))
    plotter.add_key_event('0', show_all)

    # Pre-save one screenshot before showing (useful in headless runs)
    try:
        ensure_dirs(os.path.dirname(screenshot_path))
        plotter.screenshot(screenshot_path)
        print(f"✓ Screenshot saved: {screenshot_path}")
    except Exception as e:
        print(f"[visualize] Could not save screenshot: {e}")

    plotter.show()


# ------------ Main pipeline ------------

def run_reconstruction(args):
    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else (
                          args.device if args.device in ('cpu',) or args.device.startswith('cuda') else 'cpu'))

    if args.input is None:
        default = find_default_input(args.data_dir)
        if default is None:
            raise FileNotFoundError("No input volume provided and none found in archive/ct_test or mr_test.")
        args.input = default
        print(f"Using default input: {args.input}")

    # Prepare output
    out_dir = Path(args.output_dir)
    ensure_dirs(str(out_dir))

    # Load and preprocess image
    print("\nLoading and preprocessing volume...")
    img_proc, img_orig = preprocess_image(args.input, tuple(args.target_shape))

    # Build model & load weights
    print("\nInitializing model and loading checkpoint...")
    model = load_model(args.checkpoint, device=device, embed_dim=args.embed_dim,
                       depths=tuple(args.depths), num_heads=tuple(args.num_heads),
                       window_size=tuple(args.window_size), decoder_dim=args.decoder_dim)

    # Predict labels
    print("\nRunning inference...")
    labels, probs = predict(model, img_proc, device)
    print(f"Prediction shape: {labels.shape}; unique labels: {np.unique(labels).tolist()}")

    # Save predicted labels for debugging/inspection
    try:
        npy_path = Path(args.output_dir) / f"{Path(args.input).stem}_labels.npy"
        ensure_dirs(str(npy_path.parent))
        np.save(str(npy_path), labels.astype(np.int16))
        if nib is not None:
            aff = np.eye(4)
            nib.save(nib.Nifti1Image(labels.astype(np.int16), aff), str(npy_path.with_suffix('.nii')))
        print(f"Saved label volume: {npy_path}")
    except Exception as e:
        print(f"[save labels] failed: {e}")

    # Optional deep features for GAT refinement
    deep_feat = None
    if args.refine_gat:
        print("Extracting deep features for GAT refinement...")
        deep_feat = extract_deep_features(model, img_proc, device)

    # Reconstruct meshes and optionally refine
    # Clean labels to obtain smoother, more anatomical shapes
    labels_clean = postprocess_labels(labels, probs=probs)

    print("\nReconstructing meshes...")
    meshes_pv: Dict[int, Any] = {}
    mesh_save_dir = out_dir / 'meshes'
    ensure_dirs(str(mesh_save_dir))

    for cid in range(1, 8):
        mc = marching_cubes_from_label(labels_clean, cid, step_size=args.mc_step, min_voxels=args.min_voxels)
        if mc is None:
            print(f" - {STRUCTURE_NAMES[cid]}: skipped (too small or empty)")
            continue
        verts, faces = mc

        if args.refine_gat and deep_feat is not None:
            verts = refine_with_gat_if_available(model, deep_feat, verts, faces, img_proc.shape)

        mesh = to_pyvista_mesh(verts, faces)
        # light decimation to remove noise while preserving anatomy
        try:
            mesh = mesh.decimate_pro(target_reduction=0.10, preserve_topology=True)
        except Exception:
            pass
        meshes_pv[cid] = mesh

        # Save mesh as .vtp
        out_vtp = mesh_save_dir / f"{Path(args.input).stem}_{STRUCTURE_NAMES[cid]}.vtp"
        try:
            mesh.save(str(out_vtp))
            print(f"   ✓ Saved mesh: {out_vtp}")
        except Exception as e:
            print(f"   [save mesh] {STRUCTURE_NAMES[cid]} failed: {e}")

    if not meshes_pv:
        print("No meshes were generated. Nothing to visualize.")
        return

    screenshot = out_dir / f"{Path(args.input).stem}_heart3d.png"
    print("\nLaunching interactive viewer... Close the window to finish.")
    visualize_interactive(meshes_pv, str(screenshot))


def build_argparser():
    p = argparse.ArgumentParser(description="3D Heart Reconstruction with Swin Transformer + GAT")
    p.add_argument('--input', type=str, default=None, help='Path to NIfTI image (.nii). Defaults to first in archive/ct_test.')
    p.add_argument('--data_dir', type=str, default='./archive', help='Root data dir to search default inputs.')
    p.add_argument('--checkpoint', type=str, default='./checkpoints_transformer/best_model.pth', help='Model checkpoint path (.pth).')
    p.add_argument('--output_dir', type=str, default='./predictions_transformer', help='Where to save meshes and screenshot.')

    p.add_argument('--device', type=str, default='auto', help="'auto', 'cpu', or 'cuda[:index]'")

    # Model config (must match training)
    # embed_dim: use -1 for auto-detect from checkpoint
    p.add_argument('--embed_dim', type=int, default=-1)
    p.add_argument('--depths', type=int, nargs=4, default=[2,2,6,2])
    p.add_argument('--num_heads', type=int, nargs=4, default=[3,6,12,24])
    p.add_argument('--window_size', type=int, nargs=3, default=[7,7,7])
    p.add_argument('--decoder_dim', type=int, default=256)

    # Preprocessing
    p.add_argument('--target_shape', type=int, nargs=3, default=list(DEFAULT_TARGET_SHAPE), help='(D H W) used during training')

    # Meshing & refinement
    p.add_argument('--mc_step', type=int, default=1, help='Marching cubes step size (larger is faster, less detail).')
    p.add_argument('--min_voxels', type=int, default=500, help='Ignore components smaller than this voxel count.')
    p.add_argument('--refine_gat', action='store_true', help='Apply GAT vertex refinement using deep features.')

    return p


if __name__ == '__main__':
    parser = build_argparser()
    args = parser.parse_args()
    run_reconstruction(args)
