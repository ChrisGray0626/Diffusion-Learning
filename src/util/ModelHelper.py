#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2025/10/15
"""
import math

import torch
import torch.nn.functional as functional
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
            embed_seq = functional.pad(embed_seq, (0, 1))

        return embed_seq


class PosEmbedding(nn.Module):

    def __init__(self, embed_dim: int,
                 lat_min: float = -90.0, lat_max: float = 90.0,
                 lon_min: float = -180.0, lon_max: float = 180.0):
        super().__init__()
        # Ensure embed_dim is even and >= 4 to avoid padding/truncate
        assert embed_dim >= 4, f"embed_dim must be >= 4, got {embed_dim}"
        assert embed_dim % 2 == 0, f"embed_dim must be even, got {embed_dim}"

        self.embed_dim = embed_dim
        # Each coordinate gets half of the total embedding dimension
        self.coord_embed_dim = embed_dim // 2

        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_range = lat_max - lat_min
        self.lon_range = lon_max - lon_min

        # Pre-compute frequencies for sinusoidal embedding
        freq_num = max(1, self.coord_embed_dim // 2)
        freq_base = math.log(10000.0) / max(1, freq_num - 1)
        # Register freq as buffer so it moves with the model to the correct device
        self.register_buffer('freq', torch.exp(torch.arange(freq_num, dtype=torch.float32) * -freq_base))

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos: [B, 2, H, W]
        Returns:
            [B, embed_dim, H, W]
        """

        B, _, H, W = pos.shape

        # Handle lat and lon
        lats = pos[:, 0, :, :]  # [B, H, W]
        lons = pos[:, 1, :, :]  # [B, H, W]
        # Normalize to [0, 1] based on actual boundaries
        lats = (lats - self.lat_min) / self.lat_range
        lons = (lons - self.lon_min) / self.lon_range
        # Scale to appropriate range for embedding for better frequency coverage
        lats = lats * 100.0  # [B, H, W]
        lons = lons * 100.0  # [B, H, W]

        # Calculate embeddings using sinusoidal frequencies
        # Multiply coordinates with frequencies: [B, H, W] * [freq_num] -> [B, H, W, freq_num]
        embed_lats = lats[..., None] * self.freq[None, None, None, :]  # [B, H, W, freq_num]
        embed_lats = torch.cat([torch.sin(embed_lats), torch.cos(embed_lats)], dim=-1)  # [B, H, W, coord_embed_dim]
        emb_lons = lons[..., None] * self.freq[None, None, None, :]  # [B, H, W, freq_num]
        emb_lons = torch.cat([torch.sin(emb_lons), torch.cos(emb_lons)], dim=-1)  # [B, H, W, coord_embed_dim]

        # Concatenate along channel dimension
        embed_pos = torch.cat([embed_lats, emb_lons], dim=-1)  # [B, H, W, embed_dim]
        embed_pos = embed_pos.permute(0, 3, 1, 2)  # [B, H, W, embed_dim] -> [B, embed_dim, H, W]

        return embed_pos


class BaseResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = None

    def forward(self, x):
        return x + self.net(x)


class SimpleResBlock(nn.Module):
    """
    Simple Residual Block For 1D inputs
    """

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
    mse = functional.mse_loss(pred_ys, true_ys)
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
