#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2025/10/15
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    """
    seq: [batch_size, ]
    embed_seq: [batch_size, dim]
    """

    def forward(self, seq):
        device = seq.device
        half = self.dim // 2
        embed_seq = math.log(10000) / (half - 1)
        embed_seq = torch.exp(torch.arange(half, device=device) * -embed_seq)
        embed_seq = seq[:, None].float() * embed_seq[None, :]
        embed_seq = torch.cat([torch.sin(embed_seq), torch.cos(embed_seq)], dim=-1)
        if self.dim % 2 == 1:  # pad if odd
            embed_seq = F.pad(embed_seq, (0, 1))

        return embed_seq


class PosEmbedding(nn.Module):
    """
    Sinusoidal position embedding for geographic coordinates (latitude, longitude).
    Uses the existing SinusoidalPosEmb from ModelHelper for embedding.
    Applies sinusoidal embedding to each coordinate separately, then concatenates.
    """

    def __init__(self, embed_dim: int):
        """
        Args:
            embed_dim: Dimension of position embeddings (total for both coordinates)
        """
        super().__init__()
        self.embed_dim = embed_dim
        # Each coordinate (lat, lon) gets embed_dim // 2 dimensions
        self.dim_per_coord = embed_dim // 2
        # Use existing SinusoidalPosEmb for embedding
        self.lat_embedder = SinusoidalPosEmb(self.dim_per_coord)
        self.lon_embedder = SinusoidalPosEmb(self.dim_per_coord)

    def forward(self, pos_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos_coords: [B, 2, H, W] - position coordinates (latitude, longitude)
        Returns:
            [B, embed_dim, H, W] - embedded position coordinates
        """
        B, _, H, W = pos_coords.shape

        # Extract lat and lon
        lat = pos_coords[:, 0, :, :]  # [B, H, W]
        lon = pos_coords[:, 1, :, :]  # [B, H, W]

        # Normalize coordinates to reasonable range for embedding
        # Latitude: [-90, 90] -> scale to [0, 180] for embedding
        lat_scaled = (lat + 90.0)  # [B, H, W], range [0, 180]
        # Longitude: [-180, 180] -> scale to [0, 360] for embedding
        lon_scaled = (lon + 180.0)  # [B, H, W], range [0, 360]

        # Flatten spatial dimensions for embedding
        # SinusoidalPosEmb expects [batch_size] input
        lat_flat = lat_scaled.view(-1)  # [B*H*W]
        lon_flat = lon_scaled.view(-1)  # [B*H*W]

        # Apply sinusoidal embedding using existing SinusoidalPosEmb
        lat_embed_flat = self.lat_embedder(lat_flat)  # [B*H*W, dim_per_coord]
        lon_embed_flat = self.lon_embedder(lon_flat)  # [B*H*W, dim_per_coord]

        # Reshape back to spatial dimensions
        lat_embed = lat_embed_flat.view(B, H, W, self.dim_per_coord)  # [B, H, W, dim_per_coord]
        lon_embed = lon_embed_flat.view(B, H, W, self.dim_per_coord)  # [B, H, W, dim_per_coord]

        # Concatenate along channel dimension
        pos_embed = torch.cat([lat_embed, lon_embed], dim=-1)  # [B, H, W, embed_dim]

        # Permute to [B, embed_dim, H, W]
        pos_embed = pos_embed.permute(0, 3, 1, 2)

        return pos_embed


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)


def evaluate(true_ys: torch.Tensor, pred_ys: torch.Tensor):
    mse = F.mse_loss(pred_ys, true_ys)
    rmse = torch.sqrt(mse)
    r2 = 1 - torch.sum((true_ys - pred_ys) ** 2) / torch.sum((true_ys - torch.mean(true_ys)) ** 2)
    print(f"Evaluation MSE: {mse.item():.6f}")
    print(f"Evaluation RMSE: {rmse.item():.6f}")
    print(f"Evaluation R2: {r2.item():.6f}")


def calc_masked_mse(preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Calculate the masked mean squared error between predictions and targets.
    Args:
        preds (torch.Tensor): [B, C, H, W]
        targets (torch.Tensor): [B, C, H, W]
        masks (torch.Tensor): [B, 1, H, W]
        C is the number of channels
    Returns:
        mse (torch.Tensor): scalar tensor representing the masked MSE
    """
    mse = (preds - targets) ** 2 * masks
    mse = (mse.sum(dim=(1, 2, 3)) / torch.clamp(masks.sum(dim=(1, 2, 3)), min=1.0)).mean()

    return mse
