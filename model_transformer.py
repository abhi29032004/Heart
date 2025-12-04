"""
3D Swin Transformer + GAT-based Heart Segmentation Model
Replaces U-Net with a Transformer backbone for global context
Adds Graph Attention Network for mesh refinement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from typing import Optional, Tuple


# ============================================================================
# 3D Swin Transformer Components
# ============================================================================

class PatchEmbed3D(nn.Module):
    """
    3D Image to Patch Embedding
    Splits volume into non-overlapping patches and projects to embedding dimension
    """
    def __init__(self, patch_size=4, in_chans=1, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        # Use 3D convolution for patch embedding
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: (B, C, D, H, W)
        x = self.proj(x)  # (B, embed_dim, D//P, H//P, W//P)
        B, C, D, H, W = x.shape
        x = rearrange(x, 'b c d h w -> b (d h w) c')  # (B, N, C) where N = D*H*W
        x = self.norm(x)
        return x, (D, H, W)


class WindowAttention3D(nn.Module):
    """
    Window-based Multi-head Self Attention for 3D volumes
    Uses shifted windows to capture cross-window connections
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wd, Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads)
        )
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
    
    def forward(self, x, mask=None):
        # x: (B*num_windows, window_size^3, C)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Add relative position bias (simplified - full implementation would compute coords)
        # For now, skip detailed relative position bias to keep code concise
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock3D(nn.Module):
    """
    Swin Transformer Block for 3D volumes
    Consists of Window-MSA, Shifted Window-MSA, and MLP
    """
    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x, D, H, W):
        # x: (B, D*H*W, C)
        B, L, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)
        
        # Adjust window size if volume is smaller
        Wd = min(self.window_size[0], D)
        Wh = min(self.window_size[1], H)
        Ww = min(self.window_size[2], W)
        
        # Adjust shift size accordingly
        Sd = min(self.shift_size[0], Wd // 2)
        Sh = min(self.shift_size[1], Wh // 2)
        Sw = min(self.shift_size[2], Ww // 2)
        
        # Cyclic shift for shifted window attention
        if any(s > 0 for s in [Sd, Sh, Sw]):
            shifted_x = torch.roll(x, shifts=(-Sd, -Sh, -Sw), dims=(1, 2, 3))
        else:
            shifted_x = x
        
        # Pad if not divisible by window size
        pad_d = (Wd - D % Wd) % Wd
        pad_h = (Wh - H % Wh) % Wh
        pad_w = (Ww - W % Ww) % Ww
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            shifted_x = F.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        
        D_p, H_p, W_p = D + pad_d, H + pad_h, W + pad_w
        
        # Partition windows
        x_windows = rearrange(
            shifted_x, 
            'b (d wd) (h wh) (w ww) c -> (b d h w) (wd wh ww) c',
            wd=Wd, wh=Wh, ww=Ww
        )
        
        # Window attention
        attn_windows = self.attn(x_windows)
        
        # Merge windows
        shifted_x = rearrange(
            attn_windows,
            '(b d h w) (wd wh ww) c -> b (d wd) (h wh) (w ww) c',
            b=B, d=D_p//Wd, h=H_p//Wh, w=W_p//Ww, wd=Wd, wh=Wh, ww=Ww
        )
        
        # Remove padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            shifted_x = shifted_x[:, :D, :H, :W, :]
        
        # Reverse cyclic shift
        if any(s > 0 for s in [Sd, Sh, Sw]):
            x = torch.roll(shifted_x, shifts=(Sd, Sh, Sw), dims=(1, 2, 3))
        else:
            x = shifted_x
        
    # Use reshape instead of view to handle non-contiguous tensors
        x = x.reshape(B, D * H * W, C)
        
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x


class PatchMerging3D(nn.Module):
    """
    Patch Merging Layer for downsampling
    Merges 2x2x2 neighboring patches and applies linear projection
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)
    
    def forward(self, x, D, H, W):
        # x: (B, D*H*W, C)
        B, L, C = x.shape
        x = x.view(B, D, H, W, C)
        # If any spatial dimension is odd, pad to make them even so 2x2x2 downsampling works
        pad_d = (2 - (D % 2)) % 2
        pad_h = (2 - (H % 2)) % 2
        pad_w = (2 - (W % 2)) % 2

        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            # convert to (B, C, D, H, W) for F.pad
            x_perm = x.permute(0, 4, 1, 2, 3).contiguous()
            x_perm = F.pad(x_perm, (0, pad_w, 0, pad_h, 0, pad_d))
            # back to (B, D_p, H_p, W_p, C)
            x = x_perm.permute(0, 2, 3, 4, 1)
            D = D + pad_d
            H = H + pad_h
            W = W + pad_w

        # Downsample by concatenating 2x2x2 patches
        x0 = x[:, 0::2, 0::2, 0::2, :]  # (B, D/2, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # (B, D/2, H/2, W/2, 8C)
        x = x.view(B, -1, 8 * C)
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x, (D // 2, H // 2, W // 2)


class SwinTransformer3DEncoder(nn.Module):
    """
    3D Swin Transformer Encoder
    Hierarchical architecture with 4 stages
    """
    def __init__(self, in_chans=1, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7), mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(patch_size=4, in_chans=in_chans, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_dim = int(embed_dim * 2 ** i_layer)
            layer = nn.ModuleList([
                SwinTransformerBlock3D(
                    dim=layer_dim,
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=(0, 0, 0) if (i % 2 == 0) else tuple(ws // 2 for ws in window_size),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate
                )
                for i in range(depths[i_layer])
            ])
            
            # Add patch merging except for last layer
            if i_layer < self.num_layers - 1:
                downsample = PatchMerging3D(dim=layer_dim)
            else:
                downsample = None
            
            self.layers.append(nn.ModuleDict({
                'blocks': layer,
                'downsample': downsample
            }))
    
    def forward(self, x):
        # x: (B, 1, D, H, W)
        x, (D, H, W) = self.patch_embed(x)  # (B, N, C)
        x = self.pos_drop(x)
        
        features = []
        for i, layer_dict in enumerate(self.layers):
            blocks = layer_dict['blocks']
            downsample = layer_dict['downsample']
            
            # Apply transformer blocks
            for blk in blocks:
                x = blk(x, D, H, W)
            
            # Store feature for skip connection
            features.append(x.view(x.shape[0], D, H, W, -1).permute(0, 4, 1, 2, 3))  # (B, C, D, H, W)
            
            # Downsample
            if downsample is not None:
                x, (D, H, W) = downsample(x, D, H, W)
        
        return features


# ============================================================================
# Graph Attention Network for Mesh Refinement
# ============================================================================

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT)
    Memory-efficient edge-based attention over mesh graphs.
    Accepts either a dense adjacency matrix (N,N) or an edge index of shape
    (E,2) or (2,E). Prefer edge index to avoid O(N^2) memory.
    """
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.1, alpha=0.2, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.head_dim = out_features // num_heads
        assert self.head_dim * num_heads == out_features, "out_features must be divisible by num_heads"

        # Linear transformations for each head
        self.W = nn.ModuleList([
            nn.Linear(in_features, self.head_dim, bias=False) for _ in range(num_heads)
        ])

        # Attention mechanism parameters
        self.a = nn.ModuleList([
            nn.Linear(2 * self.head_dim, 1, bias=False) for _ in range(num_heads)
        ])

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)

        if not concat:
            self.final_proj = nn.Linear(self.head_dim, out_features)

    def _to_edge_index(self, adj_or_edges, N, device):
        # Return edge_index as (2, E) LongTensor
        if adj_or_edges.dim() == 2 and adj_or_edges.shape == (N, N):
            edge_index = (adj_or_edges > 0).nonzero(as_tuple=False).T.contiguous()
            return edge_index.long()
        if adj_or_edges.dim() == 2 and adj_or_edges.shape[1] == 2:
            ei = adj_or_edges.T.contiguous()
            return ei.long()
        if adj_or_edges.dim() == 2 and adj_or_edges.shape[0] == 2:
            return adj_or_edges.long()
        raise ValueError("adj_or_edges must be (N,N) adjacency, (E,2), or (2,E) edge index")

    def forward(self, h, adj_or_edges):
        """
        Args:
            h: Node features (N, in_features)
            adj_or_edges: (N,N) dense adjacency or edge index (E,2) or (2,E)
        Returns:
            out: Updated node features (N, out_features)
        """
        device = h.device
        N = h.size(0)
        edge_index = self._to_edge_index(adj_or_edges, N, device)  # (2, E)
        src = edge_index[0]
        dst = edge_index[1]

        head_outputs = []
        for head_idx in range(self.num_heads):
            Wh = self.W[head_idx](h)  # (N, head_dim)
            Wh_i = Wh[src]  # (E, head_dim)
            Wh_j = Wh[dst]  # (E, head_dim)
            # Attention logits e_ij
            e_ij = self.leakyrelu(self.a[head_idx](torch.cat([Wh_i, Wh_j], dim=1)).squeeze(1))  # (E,)
            # Softmax over neighbors of i (src)
            exp_e = torch.exp(e_ij.clamp_max(20.0))
            denom = torch.zeros(N, device=device, dtype=exp_e.dtype)
            denom.scatter_add_(0, src, exp_e)
            alpha = exp_e / (denom[src] + 1e-9)
            alpha = self.dropout_layer(alpha)
            # Aggregate: h'_i = sum_j alpha_ij * Wh_j
            h_prime = torch.zeros(N, self.head_dim, device=device, dtype=Wh.dtype)
            h_prime.index_add_(0, src, alpha.unsqueeze(1) * Wh_j)
            head_outputs.append(h_prime)

        if self.concat:
            out = torch.cat(head_outputs, dim=1)  # (N, out_features)
        else:
            out = torch.mean(torch.stack(head_outputs), dim=0)
            out = self.final_proj(out)

        return F.elu(out)


class MeshDeformationGAT(nn.Module):
    """
    GAT-based Mesh Deformation/Refinement Module
    Refines mesh vertices using graph attention on neighboring vertices
    """
    def __init__(self, vertex_feature_dim=128, hidden_dim=256, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        
        # Initial projection
        self.input_proj = nn.Linear(vertex_feature_dim, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim
            concat = True if i < num_layers - 1 else False
            
            self.gat_layers.append(
                GraphAttentionLayer(
                    in_features=in_dim,
                    out_features=out_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    concat=concat
                )
            )
        
        # Output: predict vertex displacement
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3)  # 3D displacement (dx, dy, dz)
        )
    
    def forward(self, vertices, vertex_features, faces):
        """
        Args:
            vertices: Initial vertex positions (N, 3)
            vertex_features: Features extracted from segmentation (N, vertex_feature_dim)
            faces: Face connectivity (F, 3)
        Returns:
            refined_vertices: Refined vertex positions (N, 3)
        """
        N = vertices.shape[0]
        
        # Build adjacency matrix from faces
        adj = self._build_adjacency(faces, N)
        
        # Project features
        h = self.input_proj(vertex_features)  # (N, hidden_dim)
        
        # Apply GAT layers
        for gat_layer in self.gat_layers:
            h = gat_layer(h, adj)
        
        # Predict displacement
        displacement = self.output_proj(h)  # (N, 3)
        
        # Refine vertices
        refined_vertices = vertices + displacement
        
        return refined_vertices
    
    def _build_adjacency(self, faces, num_vertices):
        """
        Build edge index from face connectivity (directed, with self-loops).
        Returns edge_index of shape (2, E).
        """
        device = faces.device
        f = faces.long()
        e01 = torch.stack([f[:, 0], f[:, 1]], dim=0)
        e12 = torch.stack([f[:, 1], f[:, 2]], dim=0)
        e20 = torch.stack([f[:, 2], f[:, 0]], dim=0)
        edges = torch.cat([e01, e12, e20], dim=1)
        # add reverse directions
        edges = torch.cat([edges, edges.flip(0)], dim=1)
        # add self-loops
        ar = torch.arange(num_vertices, device=device)
        self_loops = torch.stack([ar, ar], dim=0)
        edges = torch.cat([edges, self_loops], dim=1)
        # remove duplicate directed edges
        edges = edges.t().unique(dim=0).t().contiguous()
        return edges


# ============================================================================
# Hybrid Decoder (Transformer features + Conv upsampling)
# ============================================================================

class TransformerDecoder3D(nn.Module):
    """
    Decoder that fuses Swin Transformer features with convolutional upsampling
    Uses skip connections from encoder features
    """
    def __init__(self, encoder_dims=[96, 192, 384, 768], decoder_dim=256, num_classes=8):
        super().__init__()
        self.encoder_dims = encoder_dims
        self.decoder_dim = decoder_dim
        
        # Upsample blocks - we need len(encoder_dims) - 1 blocks
        # to go from deepest to shallowest
        self.up_blocks = nn.ModuleList()
        for i in range(len(encoder_dims) - 1):
            if i == 0:
                # First block processes deepest feature
                in_dim = encoder_dims[-1]
            else:
                in_dim = decoder_dim
            
            # Skip connection from the corresponding encoder stage
            skip_dim = encoder_dims[-(i+2)]
            
            self.up_blocks.append(nn.ModuleDict({
                'up': nn.ConvTranspose3d(in_dim, decoder_dim, kernel_size=2, stride=2),
                'conv': nn.Sequential(
                    nn.Conv3d(decoder_dim + skip_dim, decoder_dim, kernel_size=3, padding=1),
                    nn.BatchNorm3d(decoder_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(decoder_dim, decoder_dim, kernel_size=3, padding=1),
                    nn.BatchNorm3d(decoder_dim),
                    nn.ReLU(inplace=True)
                )
            }))
        
        # Final upsampling to original resolution
        self.final_up = nn.ConvTranspose3d(decoder_dim, decoder_dim // 2, kernel_size=4, stride=4)
        self.final_conv = nn.Conv3d(decoder_dim // 2, num_classes, kernel_size=1)
    
    def forward(self, encoder_features):
        # encoder_features: list of (B, C, D, H, W) from encoder stages [stage0, stage1, stage2, stage3]
        # Process from deepest to shallowest
        x = encoder_features[-1]  # Start with deepest feature (stage3)
        
        for i, block_dict in enumerate(self.up_blocks):
            # Get skip connection from the corresponding stage
            skip_idx = -(i + 2)  # stage2, stage1, stage0
            skip = encoder_features[skip_idx]
            
            # Upsample
            x = block_dict['up'](x)
            # If spatial sizes don't match due to padding/odd sizes, interpolate to the skip size
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            
            # Concatenate with skip
            x = torch.cat([x, skip], dim=1)
            
            # Convolve
            x = block_dict['conv'](x)
        
        # Final upsampling and classification
        x = self.final_up(x)
        x = self.final_conv(x)
        
        return x


# ============================================================================
# Complete Model: SwinTransformer + GAT
# ============================================================================

class SwinTransformerGATHeartModel(nn.Module):
    """
    Complete 3D Heart Segmentation Model with:
    - 3D Swin Transformer Encoder (global context)
    - Hybrid Decoder with skip connections
    - GAT-based Mesh Refinement (optional, for post-processing)
    
    Input: (B, 1, D, H, W) - 3D CT/MRI volume
    Output: (B, 8, D, H, W) - 8-class segmentation logits
    """
    def __init__(self, in_chans=1, num_classes=8, embed_dim=96, depths=[2, 2, 6, 2], 
                 num_heads=[3, 6, 12, 24], window_size=(7, 7, 7), decoder_dim=256):
        super().__init__()
        
        # Encoder: Swin Transformer
        self.encoder = SwinTransformer3DEncoder(
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size
        )
        
        # Calculate encoder output dimensions
        encoder_dims = [int(embed_dim * 2 ** i) for i in range(len(depths))]
        
        # Decoder: Conv-based with skip connections
        self.decoder = TransformerDecoder3D(
            encoder_dims=encoder_dims,
            decoder_dim=decoder_dim,
            num_classes=num_classes
        )
        
        # Optional: GAT for mesh refinement (used in post-processing)
        # Vertex features are sampled from the deepest encoder feature map
        deepest_dim = encoder_dims[-1]
        self.mesh_gat = MeshDeformationGAT(
            vertex_feature_dim=deepest_dim,
            hidden_dim=256,
            num_heads=4,
            num_layers=3
        )
    
    def forward(self, x):
        """
        Forward pass for segmentation
        Args:
            x: (B, 1, D, H, W) input volume
        Returns:
            logits: (B, num_classes, D, H, W) segmentation logits
        """
        # Encode
        encoder_features = self.encoder(x)
        
        # Decode
        logits = self.decoder(encoder_features)
        
        return logits
    
    def refine_mesh(self, vertices, features, faces):
        """
        Post-processing: refine mesh using GAT
        Args:
            vertices: (N, 3) initial mesh vertices
            features: (N, F) per-vertex features from segmentation
            faces: (F, 3) face connectivity
        Returns:
            refined_vertices: (N, 3) refined mesh vertices
        """
        return self.mesh_gat(vertices, features, faces)


def get_transformer_model(in_channels=1, num_classes=8, embed_dim=96, depths=[2, 2, 6, 2],
                          num_heads=[3, 6, 12, 24], window_size=(7, 7, 7), decoder_dim=256):
    """
    Factory function to create the Swin Transformer + GAT model
    
    Args:
        in_channels: Input channels (1 for CT/MRI)
        num_classes: Number of segmentation classes (8: background + 7 structures)
        embed_dim: Base embedding dimension (default 96)
        depths: Number of blocks in each stage
        num_heads: Number of attention heads in each stage
        window_size: Window size for local attention
        decoder_dim: Decoder feature dimension
    
    Returns:
        model: SwinTransformerGATHeartModel instance
    """
    model = SwinTransformerGATHeartModel(
        in_chans=in_channels,
        num_classes=num_classes,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        decoder_dim=decoder_dim
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"3D Swin Transformer + GAT Model Created:")
    print(f"  Architecture: Swin Transformer Encoder + Hybrid Decoder + GAT Mesh Refinement")
    print(f"  Input channels: {in_channels}")
    print(f"  Output classes: {num_classes}")
    print(f"  Embedding dim: {embed_dim}")
    print(f"  Depths: {depths}")
    print(f"  Num heads: {num_heads}")
    print(f"  Window size: {window_size}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}\n")
    
    # Create model (smaller config for testing)
    test_embed_dim = 48
    test_depths = [2, 2, 2, 2]
    model = get_transformer_model(
        in_channels=1,
        num_classes=8,
        embed_dim=test_embed_dim,  # Smaller for testing
        depths=test_depths,
        num_heads=[3, 6, 12, 24],
        window_size=(4, 4, 4),  # Smaller window for 32x32x32 input
        decoder_dim=128
    )
    model = model.to(device)
    model.eval()
    
    # Test forward pass
    batch_size = 1
    D, H, W = 32, 32, 32  # Small test volume
    x = torch.randn(batch_size, 1, D, H, W).to(device)
    
    print(f"\nTesting forward pass...")
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test mesh refinement (synthetic example)
    print(f"\nTesting GAT mesh refinement...")
    num_vertices = 100
    num_faces = 50
    
    vertices = torch.randn(num_vertices, 3).to(device)
    # Use deepest encoder channel dimension for vertex features to match MeshDeformationGAT
    deepest_dim_test = int(test_embed_dim * 2 ** (len(test_depths) - 1))
    vertex_features = torch.randn(num_vertices, deepest_dim_test).to(device)
    faces = torch.randint(0, num_vertices, (num_faces, 3)).to(device)
    
    refined = model.refine_mesh(vertices, vertex_features, faces)
    print(f"Refined vertices shape: {refined.shape}")
    print(f"Average displacement: {(refined - vertices).abs().mean():.4f}")
    
    print("\n✓ Model tests passed!")
