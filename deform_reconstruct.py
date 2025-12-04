"""
Direct mesh deformation inference: Swin encoder features + GAT to deform a
smooth template mesh into a heart-like shape. No marching cubes from
segmentation is used at inference.

This expects you have trained the GAT head using train_gat_deform.py, which
produces a checkpoint containing the 'mesh_gat' state dict.
"""

from __future__ import annotations

import os
from pathlib import Path
import argparse
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F

import pyvista as pv
import nibabel as nib

from model_transformer import get_transformer_model, MeshDeformationGAT


STRUCTURE_COLOR = (0.85, 0.2, 0.2)  # single mesh color


def ensure_dirs(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_nifti(path: str) -> np.ndarray:
    return nib.load(path).get_fdata()


def resize_volume_torch(vol: np.ndarray, target_shape: Tuple[int,int,int]) -> np.ndarray:
    t = torch.from_numpy(vol.astype(np.float32))[None, None]
    out = F.interpolate(t, size=target_shape, mode='trilinear', align_corners=False)[0,0].cpu().numpy()
    vmin, vmax = out.min(), out.max()
    if vmax > vmin:
        out = (out - vmin) / (vmax - vmin)
    return out


def infer_embed_dim_from_ckpt(ckpt_path: str) -> int:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    for k, v in state.items():
        if k.endswith('encoder.patch_embed.proj.weight') or 'patch_embed.proj.weight' in k:
            return v.shape[0]
    return 96


def create_template_sphere(target_shape: Tuple[int,int,int], radius_scale: float = 0.35, res: int = 4,
                           center_xyz: Tuple[float,float,float] | None = None,
                           radius_abs: float | None = None):
    D, H, W = target_shape
    r = (radius_scale * min(D, H, W)) if radius_abs is None else float(radius_abs)
    center = (W/2.0, H/2.0, D/2.0) if center_xyz is None else (float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2]))
    sphere = pv.Sphere(radius=r, center=center, theta_resolution=16*res, phi_resolution=16*res)
    verts_vtk = sphere.points
    faces_vtk = sphere.faces.reshape(-1, 4)[:, 1:]
    verts = np.stack([verts_vtk[:, 2], verts_vtk[:, 1], verts_vtk[:, 0]], axis=1)
    return verts.astype(np.float32), faces_vtk.astype(np.int64)


def create_template_ellipsoid(center_xyz: Tuple[float, float, float], half_extents_xyz: Tuple[float, float, float],
                              rotation_R: np.ndarray, res: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Create an ellipsoid mesh given PCA parameters.
    Args:
        center_xyz: (cx, cy, cz) in voxel space
        half_extents_xyz: (ex, ey, ez) semi-axes lengths along PCA axes
        rotation_R: 3x3 rotation matrix; columns are PCA axes in x,y,z order
        res: sphere base resolution multiplier
    Returns:
        verts_dhw, faces (triangles)
    """
    # Start from a unit sphere at origin in xyz
    sphere = pv.Sphere(radius=1.0, center=(0.0, 0.0, 0.0), theta_resolution=16*res, phi_resolution=16*res)
    pts = sphere.points  # (N,3) xyz on unit sphere
    # Scale along principal axes, then rotate and translate
    S = np.diag(np.asarray(half_extents_xyz, dtype=np.float32))  # (3,3)
    R = np.asarray(rotation_R, dtype=np.float32)  # (3,3)
    transformed = (pts @ S.T) @ R.T  # still centered at origin
    transformed += np.asarray(center_xyz, dtype=np.float32)
    faces_vtk = sphere.faces.reshape(-1, 4)[:, 1:]
    # Convert xyz -> dhw order for feature sampling
    verts_dhw = np.stack([transformed[:, 2], transformed[:, 1], transformed[:, 0]], axis=1).astype(np.float32)
    return verts_dhw, faces_vtk.astype(np.int64)

def estimate_center_radius_from_features(deep: torch.Tensor, target_shape: Tuple[int,int,int], perc: float = 0.90,
                                         min_radius: float = 6.0, scale: float = 0.45) -> Tuple[Tuple[float,float,float], float]:
    """Estimate template center/radius from encoder deep features.
    We compute a saliency map by L2 norm over channels, upsample to target_shape,
    then use a high-percentile region to compute center and radius.
    """
    device = deep.device
    with torch.no_grad():
        sal = torch.norm(deep[0], dim=0, p=2, keepdim=True).unsqueeze(0)  # (1,1,Dp,Hp,Wp)
        sal_up = F.interpolate(sal, size=target_shape, mode='trilinear', align_corners=False)[0,0].cpu().numpy()
    v = sal_up
    v = v - v.min()
    if v.max() > 0:
        v = v / v.max()
    thr = np.quantile(v, perc)
    mask = v >= thr
    if not np.any(mask):
        D, H, W = target_shape
        return (W/2.0, H/2.0, D/2.0), max(min_radius, 0.35*min(D,H,W))
    coords = np.argwhere(mask)  # (k,3) (z,y,x)
    cz, cy, cx = coords.mean(axis=0)
    mins = coords.min(axis=0); maxs = coords.max(axis=0)
    size = (maxs - mins + 1).astype(np.float32)
    diag = float(np.linalg.norm(size))
    radius = max(min_radius, scale * 0.5 * diag)
    return (float(cx), float(cy), float(cz)), radius


def estimate_center_radius_from_segmentation(x_vol: torch.Tensor, embed_dim: int, ckpt_path: str, device: torch.device,
                                             num_classes: int = 8, prob_thr: float = 0.30,
                                             scale: float = 0.55,
                                             debug: Dict | None = None) -> Tuple[Tuple[float,float,float], float]:
    """Use the trained segmentation decoder to get a coarse foreground mask and
    compute center and radius from its bounding box. Returns (center_xyz, radius).
    """
    seg_model = get_transformer_model(in_channels=1, num_classes=num_classes, embed_dim=embed_dim,
                                      depths=[2,2,6,2], num_heads=[3,6,12,24], window_size=(7,7,7), decoder_dim=256)
    seg_model.to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    cleaned = { (k[7:] if k.startswith('module.') else k): v for k, v in state.items() }
    seg_model.load_state_dict(cleaned, strict=False)
    seg_model.eval()
    with torch.no_grad():
        logits = seg_model(x_vol)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        prob = torch.softmax(logits, dim=1)
        fg = prob[:, 1:].max(dim=1).values  # (B, D, H, W)
        mask = (fg >= prob_thr)
        m = mask[0].cpu().numpy()
    if debug is not None:
        debug['coarse_mask'] = m.astype(np.uint8)
    if not np.any(m):
        raise RuntimeError("Coarse segmentation mask empty")
    coords = np.argwhere(m)
    cz, cy, cx = coords.mean(axis=0)
    mins = coords.min(axis=0); maxs = coords.max(axis=0)
    size = (maxs - mins + 1).astype(np.float32)
    diag = float(np.linalg.norm(size))
    radius = max(6.0, scale * 0.5 * diag)
    return (float(cx), float(cy), float(cz)), radius


def pca_from_mask(mask_dhw: np.ndarray, shrink: float = 0.95) -> Tuple[Tuple[float,float,float], Tuple[float,float,float], np.ndarray]:
    """Compute PCA-based center, half-extents, and rotation from a binary mask in DHW order.
    Returns center_xyz, half_extents_xyz, rotation_R (xyz order).
    """
    idx = np.argwhere(mask_dhw > 0)
    if idx.size == 0:
        raise ValueError('Empty mask for PCA init')
    # Convert to xyz for PCA
    z, y, x = idx[:, 0].astype(np.float32), idx[:, 1].astype(np.float32), idx[:, 2].astype(np.float32)
    coords_xyz = np.stack([x, y, z], axis=1)
    center = coords_xyz.mean(0)
    centered = coords_xyz - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # sort by descending eigenvalue
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # Project to get extents along each PCA axis
    proj = centered @ eigvecs  # (N,3)
    mins = proj.min(0)
    maxs = proj.max(0)
    half_extents = 0.5 * (maxs - mins)
    half_extents = np.maximum(half_extents * float(shrink), 6.0)  # keep non-degenerate and slightly shrunken
    center_xyz = (float(center[0]), float(center[1]), float(center[2]))
    return center_xyz, (float(half_extents[0]), float(half_extents[1]), float(half_extents[2])), eigvecs.astype(np.float32)


def sample_vertex_features(deep_feat: torch.Tensor, verts_dhw: np.ndarray, orig_shape: Tuple[int,int,int]) -> torch.Tensor:
    device = deep_feat.device
    _, C, Dp, Hp, Wp = deep_feat.shape
    D, H, W = orig_shape
    vz = torch.from_numpy(verts_dhw[:, 0] / max(D-1,1)).to(device)
    vy = torch.from_numpy(verts_dhw[:, 1] / max(H-1,1)).to(device)
    vx = torch.from_numpy(verts_dhw[:, 2] / max(W-1,1)).to(device)
    vz = vz * (Dp-1); vy = vy * (Hp-1); vx = vx * (Wp-1)
    gz = vz / max(Dp-1,1) * 2 - 1
    gy = vy / max(Hp-1,1) * 2 - 1
    gx = vx / max(Wp-1,1) * 2 - 1
    grid = torch.stack([gx, gy, gz], dim=1).view(1, 1, -1, 1, 3)
    sampled = F.grid_sample(deep_feat, grid, align_corners=True, mode='bilinear', padding_mode='border')
    # Output (1, C, 1, N, 1) -> pick D_out=0, keep H_out=N
    feat = sampled[0, :, 0, :, 0].transpose(0, 1).contiguous()
    return feat


def maybe_norm_features(feat: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == 'l2':
        return F.normalize(feat, p=2, dim=1, eps=1e-6)
    if mode == 'ln':
        return F.layer_norm(feat, normalized_shape=(feat.shape[1],))
    return feat


def sample_features_from_levels(enc_feats: list[torch.Tensor], verts_dhw: np.ndarray, orig_shape: Tuple[int,int,int],
                                levels: str = 'deep', norm: str = 'none') -> torch.Tensor:
    if levels == 'multi' and len(enc_feats) >= 2:
        f_deep = sample_vertex_features(enc_feats[-1], verts_dhw, orig_shape)
        f_prev = sample_vertex_features(enc_feats[-2], verts_dhw, orig_shape)
        f_deep = maybe_norm_features(f_deep, norm)
        f_prev = maybe_norm_features(f_prev, norm)
        return torch.cat([f_deep, f_prev], dim=1)
    f = sample_vertex_features(enc_feats[-1], verts_dhw, orig_shape)
    return maybe_norm_features(f, norm)


def to_pyvista_mesh(verts: np.ndarray, faces: np.ndarray) -> 'pv.PolyData':
    face_sizes = np.full((faces.shape[0], 1), 3, dtype=np.int32)
    faces_pv = np.hstack([face_sizes, faces.astype(np.int32)]).reshape(-1)
    mesh = pv.PolyData(verts[:, [2,1,0]], faces_pv)  # to (x,y,z)
    mesh.compute_normals(inplace=True)
    return mesh


def run(args):
    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else args.device)
    ensure_dirs(args.output_dir)

    vol = load_nifti(args.input)
    vol = resize_volume_torch(vol, tuple(args.target_shape))

    embed_dim = infer_embed_dim_from_ckpt(args.encoder_ckpt)
    model = get_transformer_model(in_channels=1, num_classes=8, embed_dim=embed_dim,
                                  depths=[2,2,6,2], num_heads=[3,6,12,24], window_size=(7,7,7), decoder_dim=256)
    model.to(device)

    # Load encoder weights
    ckpt = torch.load(args.encoder_ckpt, map_location=device)
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    cleaned = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith('module.') else k
        if nk.startswith('encoder.'):
            cleaned[nk] = v
    model.load_state_dict(cleaned, strict=False)

    # Load trained GAT weights and align architecture
    feat_levels = args.feat_levels
    feat_norm = args.feat_norm
    if os.path.isfile(args.gat_ckpt):
        gat_state = torch.load(args.gat_ckpt, map_location=device)
        meta = {k: gat_state.get(k, None) for k in ['vertex_feat_dim','hidden','heads','layers','feat_levels','feat_norm']}
        deepest_dim = int(embed_dim * 2 ** (len([2,2,6,2]) - 1))
        prev_dim = int(embed_dim * 2 ** (len([2,2,6,2]) - 2))
        if meta['vertex_feat_dim'] is None:
            # infer from levels
            vdim = deepest_dim if feat_levels != 'multi' else deepest_dim + prev_dim
        else:
            vdim = int(meta['vertex_feat_dim'])
        hidden = int(meta['hidden']) if meta['hidden'] is not None else 256
        heads = int(meta['heads']) if meta['heads'] is not None else 4
        layers = int(meta['layers']) if meta['layers'] is not None else 3
        if isinstance(meta.get('feat_levels', None), str):
            feat_levels = meta['feat_levels']
        if isinstance(meta.get('feat_norm', None), str):
            feat_norm = meta['feat_norm']
        # Replace mesh_gat if dimensions mismatch
        if model.mesh_gat.input_proj.in_features != vdim or model.mesh_gat.output_proj[0].in_features != hidden:
            model.mesh_gat = MeshDeformationGAT(vertex_feature_dim=vdim, hidden_dim=hidden, num_heads=heads, num_layers=layers).to(device)
        # Load state dict
        if 'mesh_gat' in gat_state:
            model.mesh_gat.load_state_dict(gat_state['mesh_gat'], strict=False)
            loaded_keys = list(gat_state['mesh_gat'].keys())
        else:
            model.mesh_gat.load_state_dict(gat_state, strict=False)
            loaded_keys = list(gat_state.keys())
        print(f"Loaded GAT weights from {args.gat_ckpt} with {len(loaded_keys)} keys | feat_levels={feat_levels} feat_norm={feat_norm}")
    else:
        print(f"WARNING: GAT checkpoint not found at {args.gat_ckpt}. Results may be poor.")

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(vol[None, None].astype(np.float32)).to(device)
        enc_feats = model.encoder(x)

    # Template initialization
    debug = {}
    t_center = None; t_radius = None
    if args.init in ('seg', 'auto'):
        try:
            t_center, t_radius = estimate_center_radius_from_segmentation(
                x, embed_dim, args.encoder_ckpt, device,
                num_classes=8, prob_thr=args.seg_prob_thr, scale=args.seg_radius_scale, debug=debug)
            print(f"Init from segmentation: center={tuple(round(c,1) for c in t_center)}, radius={t_radius:.1f}")
        except Exception as e:
            if args.init == 'seg':
                print(f"Segmentation-based init failed: {e}")
                raise
            else:
                print(f"Seg init failed, falling back to feature-based: {e}")
                t_center = None; t_radius = None

    if t_center is None or t_radius is None:
        t_center, t_radius = estimate_center_radius_from_features(enc_feats[-1], tuple(args.target_shape), perc=args.center_percentile)
        print(f"Init from features: center={tuple(round(c,1) for c in t_center)}, radius={t_radius:.1f}")

    # If we have a coarse mask and PCA init is requested/auto, build ellipsoid; otherwise sphere
    use_pca = (args.init in ('pca', 'auto')) and ('coarse_mask' in debug)
    if use_pca:
        try:
            center_xyz, half_extents_xyz, R = pca_from_mask(debug['coarse_mask'], shrink=args.pca_shrink)
            print(f"PCA init: center={tuple(round(c,1) for c in center_xyz)}, half_extents={tuple(round(e,1) for e in half_extents_xyz)}")
            t_verts, t_faces = create_template_ellipsoid(center_xyz, half_extents_xyz, R, res=args.template_res)
        except Exception as e:
            print(f"PCA init failed ({e}); using sphere")
            t_verts, t_faces = create_template_sphere(tuple(args.target_shape), args.radius_scale, args.template_res,
                                                      center_xyz=t_center, radius_abs=t_radius)
    else:
        t_verts, t_faces = create_template_sphere(tuple(args.target_shape), args.radius_scale, args.template_res,
                                                  center_xyz=t_center, radius_abs=t_radius)
    if args.debug:
        try:
            if 'coarse_mask' in debug:
                out_mask = os.path.join(args.output_dir, f"{Path(args.input).stem}_coarse_mask.nii.gz")
                nib.Nifti1Image(debug['coarse_mask'].astype(np.uint8), affine=np.eye(4)).to_filename(out_mask)
            out_tpl = os.path.join(args.output_dir, f"{Path(args.input).stem}_template.vtp")
            to_pyvista_mesh(t_verts, t_faces).save(out_tpl)
        except Exception:
            pass
    v0 = torch.from_numpy(t_verts).to(device)
    f = torch.from_numpy(t_faces).to(device)
    vfeat = sample_features_from_levels(enc_feats, t_verts, tuple(args.target_shape), levels=feat_levels, norm=feat_norm)
    with torch.no_grad():
        print(f"Feature stats | mean={float(vfeat.mean()):.4f} std={float(vfeat.std()):.4f} min={float(vfeat.min()):.4f} max={float(vfeat.max()):.4f}")

    with torch.no_grad():
        v_pred_t = model.mesh_gat(v0, vfeat, f)
        disp = (v_pred_t - v0).detach().cpu().numpy()
        print(f"GAT displacement | mean={np.abs(disp).mean():.3f}, max={np.abs(disp).max():.3f}")
        v_pred = v_pred_t.cpu().numpy()

    mesh = to_pyvista_mesh(v_pred, t_faces)
    # Optional smoothing for nicer visuals
    try:
        mesh = mesh.smooth_taubin(n_iter=30, pass_band=0.1)
    except Exception:
        pass
    out_vtp = os.path.join(args.output_dir, f"{Path(args.input).stem}_deform.vtp")
    mesh.save(out_vtp)
    print(f"Saved mesh: {out_vtp}")

    # Visualize
    pl = pv.Plotter(window_size=(1000, 800))
    if args.show_template:
        tpl_mesh = to_pyvista_mesh(t_verts, t_faces)
        try:
            tpl_mesh = tpl_mesh.smooth_taubin(n_iter=10, pass_band=0.2)
        except Exception:
            pass
        pl.add_mesh(tpl_mesh, color=(0.2, 0.6, 0.9), opacity=0.35, smooth_shading=True, label='template')
    pl.add_mesh(mesh, color=STRUCTURE_COLOR, opacity=0.9, smooth_shading=True, label='deformed')
    pl.add_axes()
    pl.show_bounds(grid='front')
    screenshot = os.path.join(args.output_dir, f"{Path(args.input).stem}_deform.png")
    try:
        pl.screenshot(screenshot)
        print(f"Saved screenshot: {screenshot}")
    except Exception:
        pass
    pl.show()


def build_argparser():
    p = argparse.ArgumentParser(description="Direct mesh deformation inference (Swin+GAT)")
    p.add_argument('--input', type=str, required=True, help='NIfTI image path')
    p.add_argument('--encoder_ckpt', type=str, default='./checkpoints_transformer/best_model.pth')
    p.add_argument('--gat_ckpt', type=str, default='./checkpoints_transformer/gat_deform_epoch_10.pth')
    p.add_argument('--output_dir', type=str, default='./predictions_transformer/deform')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--target_shape', type=int, nargs=3, default=[160,160,80])
    p.add_argument('--radius_scale', type=float, default=0.35)
    p.add_argument('--template_res', type=int, default=4)
    p.add_argument('--center_percentile', type=float, default=0.90, help='Percentile of deep-feature saliency to localize center')
    p.add_argument('--init', type=str, choices=['auto','seg','feature','pca'], default='auto', help='Template initialization method (pca uses coarse seg)')
    p.add_argument('--seg_prob_thr', type=float, default=0.30, help='Foreground prob threshold for coarse seg init')
    p.add_argument('--seg_radius_scale', type=float, default=0.55, help='Scale for radius from coarse seg bbox diagonal')
    p.add_argument('--pca_shrink', type=float, default=0.95, help='Shrink factor for PCA semi-axes to leave room for deformation')
    p.add_argument('--show_template', action='store_true', help='Also render the initial template for debugging')
    p.add_argument('--debug', action='store_true', help='Save coarse mask/template for debugging')
    # Feature controls (auto uses settings from GAT checkpoint if present)
    p.add_argument('--feat_levels', type=str, choices=['auto','deep','multi'], default='auto')
    p.add_argument('--feat_norm', type=str, choices=['auto','none','l2','ln'], default='auto')
    return p


if __name__ == '__main__':
    args = build_argparser().parse_args()
    run(args)
