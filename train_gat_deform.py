"""
Train the GAT-based direct deformation head using Chamfer distance and mesh
regularizers. The Swin encoder provides image features; the GAT predicts
per-vertex displacements of a smooth template mesh to match the heart surface.

Ground-truth surface is obtained from labels via marching cubes during
training only. Inference does not require segmentation.

This script freezes the Swin encoder by default and optimizes only the GAT
parameters for stability and speed. You can unfreeze later if desired.
"""

from __future__ import annotations

import os
from pathlib import Path
import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from skimage import measure
import pyvista as pv

from model_transformer import get_transformer_model, MeshDeformationGAT
from dataset import HeartSegmentationDataset
from mesh_losses import chamfer_distance, laplacian_smoothness, edge_length_loss


def create_template_sphere(target_shape: Tuple[int,int,int], radius_scale: float = 0.35, res: int = 4,
                           center_xyz: Tuple[float,float,float] | None = None,
                           radius_abs: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Create a sphere template. If center/radius provided, use them.
    Returns (verts[N,3], faces[F,3]) in voxel coords (z,y,x).
    """
    D, H, W = target_shape
    r = radius_scale * float(min(D, H, W)) if radius_abs is None else float(radius_abs)
    if center_xyz is None:
        center = (W/2.0, H/2.0, D/2.0)
    else:
        center = (float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2]))
    sphere = pv.Sphere(radius=r, center=center, theta_resolution=16*res, phi_resolution=16*res)
    verts_vtk = sphere.points
    faces_vtk = sphere.faces.reshape(-1, 4)[:, 1:]
    verts = np.stack([verts_vtk[:, 2], verts_vtk[:, 1], verts_vtk[:, 0]], axis=1)
    return verts.astype(np.float32), faces_vtk.astype(np.int64)

def estimate_center_radius_from_label(label: np.ndarray, fallback_shape: Tuple[int,int,int],
                                      min_radius: float = 6.0, scale: float = 0.5) -> Tuple[Tuple[float,float,float], float]:
    """Estimate heart center (x,y,z) and radius from union mask.
    scale controls fraction of bbox size to radius.
    """
    mask = (label > 0)
    if not np.any(mask):
        D, H, W = fallback_shape
        return (W/2.0, H/2.0, D/2.0), max(min_radius, 0.35*min(D,H,W))
    coords = np.argwhere(mask)  # (k,3) in (z,y,x)
    cz, cy, cx = coords.mean(axis=0)
    mins = coords.min(axis=0); maxs = coords.max(axis=0)
    size = (maxs - mins + 1).astype(np.float32)
    # Use half of bbox diagonal scaled
    diag = float(np.linalg.norm(size))
    radius = max(min_radius, scale * 0.5 * diag)
    return (float(cx), float(cy), float(cz)), radius


def sample_vertex_features(deep_feat: torch.Tensor, verts_dhw: np.ndarray, orig_shape: Tuple[int,int,int]) -> torch.Tensor:
    """Sample deepest encoder features at given vertex positions.
    deep_feat: (1, C, Dp, Hp, Wp)
    verts_dhw: (N,3) in (z,y,x) at original resolution orig_shape
    Returns (N, C)
    """
    device = deep_feat.device
    _, C, Dp, Hp, Wp = deep_feat.shape
    D, H, W = orig_shape
    vz = torch.from_numpy(verts_dhw[:, 0] / max(D-1,1)).to(device)
    vy = torch.from_numpy(verts_dhw[:, 1] / max(H-1,1)).to(device)
    vx = torch.from_numpy(verts_dhw[:, 2] / max(W-1,1)).to(device)

    # map to feature grid coordinates
    vz = vz * (Dp-1); vy = vy * (Hp-1); vx = vx * (Wp-1)
    gz = vz / max(Dp-1,1) * 2 - 1
    gy = vy / max(Hp-1,1) * 2 - 1
    gx = vx / max(Wp-1,1) * 2 - 1
    grid = torch.stack([gx, gy, gz], dim=1).view(1, 1, -1, 1, 3)  # (N=1, D_out=1, H_out=N, W_out=1, 3)
    sampled = F.grid_sample(deep_feat, grid, align_corners=True, mode='bilinear', padding_mode='border')
    # sampled shape: (1, C, 1, N, 1) -> index D_out=0, keep H_out=:
    feat = sampled[0, :, 0, :, 0].transpose(0, 1).contiguous()  # (N, C)
    return feat


def maybe_norm_features(feat: torch.Tensor, mode: str) -> torch.Tensor:
    """Normalize per-vertex features.
    mode in {none, l2, ln}. Returns tensor of same shape.
    """
    if mode == 'l2':
        return F.normalize(feat, p=2, dim=1, eps=1e-6)
    if mode == 'ln':
        return F.layer_norm(feat, normalized_shape=(feat.shape[1],))
    return feat


def sample_features_from_levels(enc_feats: list[torch.Tensor], verts_dhw: np.ndarray, orig_shape: Tuple[int,int,int],
                                levels: str = 'deep', norm: str = 'none') -> torch.Tensor:
    """Sample per-vertex features from encoder feature maps.
    levels: 'deep' (last level only) or 'multi' (concatenate last 2 levels).
    Returns (N, C_total).
    """
    if levels == 'multi' and len(enc_feats) >= 2:
        f_deep = sample_vertex_features(enc_feats[-1], verts_dhw, orig_shape)
        f_prev = sample_vertex_features(enc_feats[-2], verts_dhw, orig_shape)
        f_deep = maybe_norm_features(f_deep, norm)
        f_prev = maybe_norm_features(f_prev, norm)
        return torch.cat([f_deep, f_prev], dim=1)
    # default: deep only
    f = sample_vertex_features(enc_feats[-1], verts_dhw, orig_shape)
    return maybe_norm_features(f, norm)


def debug_feature_sampling_stats(enc_feats: list[torch.Tensor], verts_dhw: np.ndarray, orig_shape: Tuple[int,int,int],
                                 levels: str = 'deep', norm: str = 'none') -> Tuple[torch.Tensor, dict]:
    """Return features and a dict of sampling diagnostics without altering training.
    Diagnostics include: feature mean/std, mean vertex-norm, std of per-vertex norms,
    and normalized grid range [-1,1] min/max per axis for the deepest level.
    """
    # sample features according to settings
    feat = sample_features_from_levels(enc_feats, verts_dhw, orig_shape, levels=levels, norm=norm)
    stats: dict = {}
    # Deepest level grid stats
    deep = enc_feats[-1]
    device = deep.device
    _, C, Dp, Hp, Wp = deep.shape
    D, H, W = orig_shape
    vz = torch.from_numpy(verts_dhw[:, 0] / max(D-1,1)).to(device)
    vy = torch.from_numpy(verts_dhw[:, 1] / max(H-1,1)).to(device)
    vx = torch.from_numpy(verts_dhw[:, 2] / max(W-1,1)).to(device)
    vz = vz * (Dp-1); vy = vy * (Hp-1); vx = vx * (Wp-1)
    gz = (vz / max(Dp-1,1) * 2 - 1).detach()
    gy = (vy / max(Hp-1,1) * 2 - 1).detach()
    gx = (vx / max(Wp-1,1) * 2 - 1).detach()
    stats['grid_min'] = (float(gx.min().cpu()), float(gy.min().cpu()), float(gz.min().cpu()))
    stats['grid_max'] = (float(gx.max().cpu()), float(gy.max().cpu()), float(gz.max().cpu()))
    # feature stats
    with torch.no_grad():
        norms = feat.norm(dim=1)
        stats['feat_mean'] = float(feat.mean().cpu())
        stats['feat_std'] = float(feat.std().cpu())
        stats['norm_mean'] = float(norms.mean().cpu())
        stats['norm_std'] = float(norms.std().cpu())
        # pairwise difference on a small random subset for variance sanity
        N = feat.shape[0]
        k = min(256, N)
        if k >= 2:
            idx = torch.randperm(N, device=feat.device)[:k]
            a = feat[idx[:k//2]]; b = feat[idx[k//2: k//2 + (k//2)]]
            stats['pair_mean_l2'] = float((a - b).norm(dim=1).mean().cpu())
    return feat, stats


def marching_cubes_union(label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a union-of-classes surface from label volume using marching cubes.
    Returns (verts[N,3], faces[F,3]) in (z,y,x) voxel coordinates.
    """
    mask = (label > 0).astype(np.uint8)
    if mask.sum() < 1000:
        raise RuntimeError("Too few foreground voxels for surface extraction")
    vol = mask.astype(np.float32)
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.5, spacing=(1.0,1.0,1.0))
    return verts.astype(np.float32), faces.astype(np.int64)


def point_sample_from_mesh(verts: np.ndarray, faces: np.ndarray, n_points: int = 5000) -> np.ndarray:
    """Uniformly sample points on a triangular mesh surface using area weights."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    # triangle areas
    areas = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) * 0.5
    probs = areas / (areas.sum() + 1e-8)
    idx = np.random.choice(len(faces), size=n_points, p=probs)
    v0 = v0[idx]; v1 = v1[idx]; v2 = v2[idx]
    u = np.random.rand(n_points, 1)
    v = np.random.rand(n_points, 1)
    mask = (u + v > 1)
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    pts = v0 + u * (v1 - v0) + v * (v2 - v0)
    return pts.astype(np.float32)


def infer_embed_dim_from_ckpt(ckpt_path: str) -> int:
    """Infer embed_dim from checkpoint weight shapes (patch_embed.proj.out_channels)."""
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    for k, v in state.items():
        if k.endswith('encoder.patch_embed.proj.weight') or 'patch_embed.proj.weight' in k:
            return v.shape[0]
    return 96  # fallback


def train(args):
    device = torch.device('cuda' if (args.device == 'auto' and torch.cuda.is_available()) else args.device)

    # Resolve embed dim to match encoder checkpoint
    embed_dim = infer_embed_dim_from_ckpt(args.encoder_ckpt)
    print(f"Using embed_dim={embed_dim} inferred from checkpoint")

    # Build model and load encoder weights
    model = get_transformer_model(in_channels=1, num_classes=8, embed_dim=embed_dim,
                                  depths=[2,2,6,2], num_heads=[3,6,12,24], window_size=(7,7,7), decoder_dim=256)
    model.to(device)

    ckpt = torch.load(args.encoder_ckpt, map_location=device)
    state = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith('module.'):
            nk = nk[7:]
        new_state[nk] = v
    # load only encoder weights
        missing, unexpected = model.load_state_dict({k: v for k, v in new_state.items() if k.startswith('encoder.')}, strict=False)
        # Most of 'missing' will be decoder and mesh_gat (we didn't load them on purpose).
        missing_enc = [k for k in missing if k.startswith('encoder.')]
        print(f"Loaded encoder weights. Missing(total)={len(missing)} | Missing(encoder)={len(missing_enc)} | Unexpected={len(unexpected)}")
        if len(missing_enc) > 0:
            print("Warning: Some encoder weights are missing. First 10:")
            print("  ", missing_enc[:10])

    # Freeze encoder & decoder; train only GAT head
    for n, p in model.named_parameters():
        if not n.startswith('mesh_gat'):
            p.requires_grad = False

    # Optionally replace the mesh_gat with higher capacity to experiment
    # Determine vertex feature dimension based on chosen feature levels
    deepest_dim = int(embed_dim * 2 ** (len([2,2,6,2]) - 1))
    if args.feat_levels == 'multi':
        vertex_feat_dim = deepest_dim + int(embed_dim * 2 ** (len([2,2,6,2]) - 2))
    else:
        vertex_feat_dim = deepest_dim
    if (args.hidden != 256) or (args.heads != 4) or (args.layers != 3) or (vertex_feat_dim != model.mesh_gat.input_proj.in_features):
        model.mesh_gat = MeshDeformationGAT(vertex_feature_dim=vertex_feat_dim, hidden_dim=args.hidden,
                                            num_heads=args.heads, num_layers=args.layers).to(device)
        # ensure other params remain frozen
        for n, p in model.named_parameters():
            if not n.startswith('mesh_gat'):
                p.requires_grad = False

    optimizer = Adam([p for p in model.mesh_gat.parameters() if p.requires_grad], lr=args.lr)

    # LR scheduler
    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr)
    elif args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None

    # Dataset (use CT and MR train)
    # Build modality datasets based on flags
    datasets = []
    names = []
    if args.use_ct:
        ds_ct = HeartSegmentationDataset(args.data_dir, modality='ct', mode='train', target_shape=tuple(args.target_shape))
        datasets.append(ds_ct); names.append('CT')
    if args.use_mr:
        ds_mr = HeartSegmentationDataset(args.data_dir, modality='mr', mode='train', target_shape=tuple(args.target_shape))
        datasets.append(ds_mr); names.append('MR')
    assert len(datasets) > 0, "At least one modality must be enabled (use --use_ct/--use_mr)"

    dataset = ConcatDataset(datasets)

    # Balanced sampling across modalities so CT is used even if fewer samples
    lengths = [len(ds) for ds in datasets]
    total = sum(lengths)
    # Each modality gets equal mass 1/len(datasets), distributed uniformly across its samples
    weights = []
    for i, ds in enumerate(datasets):
        per_sample = (1.0 / len(datasets)) / max(len(ds), 1)
        weights.extend([per_sample] * len(ds))

    sampler = WeightedRandomSampler(weights=torch.tensor(weights, dtype=torch.double), num_samples=total, replacement=True)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=False, num_workers=0)

    print(f"\n[DeformTrain] Modalities: {names} | counts: {lengths} | total: {total}")

    os.makedirs(args.save_dir, exist_ok=True)

    # Prepare logging
    log_path = os.path.join(args.save_dir, 'gat_deform_logs.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('epoch,loss,loss_ch,loss_lap,loss_edge,lr,offset_mean,feat_mean,feat_std\n')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_ch = 0.0; total_lap = 0.0; total_edge = 0.0
        total_off = 0.0; n_batches = 0
        for batch in loader:
            img = batch['image'].to(device)  # (1,1,D,H,W)
            label = batch['label'].numpy()[0]  # (D,H,W) numpy for MC

            # Ground truth surface points from union
            try:
                gt_verts, gt_faces = marching_cubes_union(label)
            except Exception:
                continue
            gt_pts = point_sample_from_mesh(gt_verts, gt_faces, n_points=args.points)
            gt_pts_t = torch.from_numpy(gt_pts).to(device)

            # Encoder features (list)
            with torch.no_grad():
                enc_feats = model.encoder(img)  # list of 4 tensors

            # Template sphere (label-aware placement)
            t_center, t_radius = estimate_center_radius_from_label(label, tuple(args.target_shape), scale=0.55)
            t_verts_np, t_faces_np = create_template_sphere(tuple(args.target_shape), radius_scale=args.radius_scale,
                                                            res=args.template_res, center_xyz=t_center, radius_abs=t_radius)
            v0 = torch.from_numpy(t_verts_np).to(device)
            f = torch.from_numpy(t_faces_np).to(device)

            # Per-vertex features (deep or multi-level) + normalization
            if args.log_feature_detail:
                vfeat, fstats = debug_feature_sampling_stats(enc_feats, t_verts_np, tuple(args.target_shape),
                                                             levels=args.feat_levels, norm=args.feat_norm)
            else:
                vfeat = sample_features_from_levels(enc_feats, t_verts_np, tuple(args.target_shape),
                                                    levels=args.feat_levels, norm=args.feat_norm)

            # Predict displacement (mesh_gat only)
            v_pred = model.mesh_gat(v0, vfeat, f)  # (N,3)

            # Losses
            loss_ch = chamfer_distance(v_pred, gt_pts_t)
            loss_lap = laplacian_smoothness(v_pred, f)
            loss_edge = edge_length_loss(v_pred, f)
            # Regularization warmup to avoid staying spherical
            if args.reg_warmup_epochs > 0 and epoch <= args.reg_warmup_epochs:
                warm = float(epoch) / float(max(1, args.reg_warmup_epochs))
                w_lap = args.w_lap * warm
                w_edge = args.w_edge * warm
            else:
                w_lap = args.w_lap
                w_edge = args.w_edge
            loss = args.w_ch * loss_ch + w_lap * loss_lap + w_edge * loss_edge

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.mesh_gat.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            total_ch += float(loss_ch.detach().cpu())
            total_lap += float(loss_lap.detach().cpu())
            total_edge += float(loss_edge.detach().cpu())
            off = (v_pred - v0).norm(dim=1).mean().detach().cpu().item()
            total_off += off
            n_batches += 1

            # Optional quick exit for smoke test
            if args.dry_run:
                break

        if scheduler is not None:
            scheduler.step()

        denom = max(n_batches, 1)
        avg = total_loss / denom
        avg_ch = total_ch / denom
        avg_lap = total_lap / denom
        avg_edge = total_edge / denom
        avg_off = total_off / denom
        cur_lr = optimizer.param_groups[0]['lr']
        # Feature stats from last batch
        f_mean = float(vfeat.mean().detach().cpu().item()) if 'vfeat' in locals() else 0.0
        f_std = float(vfeat.std().detach().cpu().item()) if 'vfeat' in locals() else 0.0
        msg = f"Epoch {epoch}/{args.epochs}  loss={avg:.4f} (ch={avg_ch:.4f}, lap={avg_lap:.4f}, edge={avg_edge:.4f})  off={avg_off:.3f}  lr={cur_lr:.2e}"
        if args.log_feature_detail and 'fstats' in locals():
            gm = ','.join(f"{v:.2f}" for v in fstats['grid_min'])
            gM = ','.join(f"{v:.2f}" for v in fstats['grid_max'])
            msg += f" | feat(mean={fstats['feat_mean']:.4f}, std={fstats['feat_std']:.4f}, norm={fstats['norm_mean']:.3f}±{fstats['norm_std']:.3f}) grid[min=({gm}) max=({gM})] pair_l2={fstats.get('pair_mean_l2',0.0):.3f}"
        print(msg)
        with open(log_path, 'a') as f_csv:
            f_csv.write(f"{epoch},{avg:.6f},{avg_ch:.6f},{avg_lap:.6f},{avg_edge:.6f},{cur_lr:.6e},{avg_off:.6f},{f_mean:.6f},{f_std:.6f}\n")

        # Save
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_path = os.path.join(args.save_dir, f"gat_deform_epoch_{epoch}.pth")
            torch.save({
                'mesh_gat': model.mesh_gat.state_dict(),
                'embed_dim': embed_dim,
                'vertex_feat_dim': vertex_feat_dim,
                'hidden': args.hidden,
                'heads': args.heads,
                'layers': args.layers,
                'feat_levels': args.feat_levels,
                'feat_norm': args.feat_norm,
                'w_ch': args.w_ch,
                'w_lap': args.w_lap,
                'w_edge': args.w_edge,
            }, save_path)
            print(f"Saved: {save_path}")
        if args.dry_run:
            break


def build_argparser():
    p = argparse.ArgumentParser(description="Train GAT direct deformation head with Chamfer distance")
    p.add_argument('--data_dir', type=str, default='./archive')
    p.add_argument('--encoder_ckpt', type=str, default='./checkpoints_transformer/best_model.pth', help='Checkpoint containing encoder weights')
    p.add_argument('--save_dir', type=str, default='./checkpoints_transformer')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--epochs', type=int, default=30)
    # Use a moderate LR for stability with strong regularization
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--lr_scheduler', type=str, choices=['none','cosine','step'], default='cosine')
    p.add_argument('--min_lr', type=float, default=1e-5)
    p.add_argument('--step_size', type=int, default=10)
    p.add_argument('--gamma', type=float, default=0.5)
    p.add_argument('--save_every', type=int, default=5)
    p.add_argument('--target_shape', type=int, nargs=3, default=[160,160,80])
    p.add_argument('--points', type=int, default=5000)
    p.add_argument('--radius_scale', type=float, default=0.35)
    p.add_argument('--template_res', type=int, default=4)
    p.add_argument('--w_ch', type=float, default=1.0)
    # Stronger regularization by default so Lap/Edge contribute meaningfully
    p.add_argument('--w_lap', type=float, default=0.015)
    p.add_argument('--w_edge', type=float, default=0.008)
    # Warmup disabled so the mesh stays inflated/smooth from the start
    p.add_argument('--reg_warmup_epochs', type=int, default=0, help='Linearly ramp regularizers from 0 to target over N epochs')
    p.add_argument('--grad_clip', type=float, default=1.0)
    # Feature controls
    p.add_argument('--feat_levels', type=str, choices=['deep','multi'], default='deep')
    p.add_argument('--feat_norm', type=str, choices=['none','l2','ln'], default='ln')
    p.add_argument('--log_feature_detail', action='store_true', help='Print detailed feature and grid sampling diagnostics each epoch')
    # Capacity knobs
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--use_ct', action='store_true', default=True)
    p.add_argument('--use_mr', action='store_true', default=True)
    # Debugging
    p.add_argument('--dry_run', action='store_true', help='Run a single batch for quick sanity check')
    return p


if __name__ == '__main__':
    args = build_argparser().parse_args()
    train(args)
