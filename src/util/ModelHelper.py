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

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = None

    def forward(self, x, *args):
        return x + self.net(x)


class SimpleResBlock(BaseResBlock):

    def __init__(self, hidden_dim: int):
        super().__init__(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )


def evaluate(true_ys: torch.Tensor, pred_ys: torch.Tensor, masks: torch.Tensor = None):
    """
    Evaluate predictions against true values with optional masking.
    Args:
        true_ys (torch.Tensor): [B, 1, H, W] or [B, H, W]
        pred_ys (torch.Tensor): [B, 1, H, W] or [B, H, W]
        masks (torch.Tensor, optional): [B, 1, H, W] or [B, H, W], 1=valid, 0=invalid
    """
    # Ensure 4D format [B, 1, H, W]
    if true_ys.dim() == 3:
        true_ys = true_ys.unsqueeze(1)
    if pred_ys.dim() == 3:
        pred_ys = pred_ys.unsqueeze(1)
    if masks is not None and masks.dim() == 3:
        masks = masks.unsqueeze(1)

    mse = calc_mse(pred_ys, true_ys, masks)
    bias = calc_bias(pred_ys, true_ys, masks)
    rmse = torch.sqrt(mse)
    ubrmse = torch.sqrt(torch.clamp(mse - bias ** 2, min=0.0))
    r2 = calc_r2(pred_ys, true_ys, masks)

    print(f"Evaluation MSE: {mse.item():.6f}")
    print(f"Evaluation RMSE: {rmse.item():.6f}")
    print(f"Evaluation ubRMSE: {ubrmse.item():.6f}")
    print(f"Evaluation Bias: {bias.item():.6f}")
    print(f"Evaluation R2: {r2.item():.6f}")


def calc_mse(preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
    if masks is not None:
        mse = (preds - targets) ** 2 * masks
        mse = (mse.sum(dim=(1, 2, 3)) / torch.clamp(masks.sum(dim=(1, 2, 3)), min=1.0)).mean()
    else:
        mse = functional.mse_loss(preds, targets)

    return mse


def calc_bias(preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
    if masks is not None:
        bias = (preds - targets) * masks
        bias = (bias.sum(dim=(1, 2, 3)) / torch.clamp(masks.sum(dim=(1, 2, 3)), min=1.0)).mean()
    else:
        bias = torch.mean(preds - targets)

    return bias


def calc_r2(preds: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
    if masks is not None:
        # Flatten tensors for easier computation
        preds_flat = preds.view(preds.shape[0], -1)  # [B, C*H*W]
        targets_flat = targets.view(targets.shape[0], -1)  # [B, C*H*W]
        masks_flat = masks.view(masks.shape[0], -1)  # [B, H*W]

        # Calculate R2 for each sample in the batch
        r2_list = []
        for b in range(preds.shape[0]):
            # Get valid pixels for this sample
            valid_mask = masks_flat[b] > 0  # [H*W]

            if valid_mask.sum() == 0:
                # No valid pixels, R2 is undefined
                r2_list.append(torch.tensor(float('nan'), device=preds.device))
                continue

            pred_valid = preds_flat[b][valid_mask]  # [num_valid]
            target_valid = targets_flat[b][valid_mask]  # [num_valid]

            # Calculate SS_res (sum of squared residuals)
            ss_res = torch.sum((target_valid - pred_valid) ** 2)

            # Calculate SS_tot (total sum of squares)
            target_mean = torch.mean(target_valid)
            ss_tot = torch.sum((target_valid - target_mean) ** 2)

            # Handle case where all targets are the same (ss_tot = 0)
            if ss_tot < 1e-8:
                if ss_res < 1e-8:
                    r2 = torch.tensor(1.0, device=preds.device)  # Perfect prediction when all values are the same
                else:
                    r2 = torch.tensor(float('inf'),
                                      device=preds.device)  # Undefined: predictions differ but true values are constant
            else:
                r2 = 1 - ss_res / ss_tot

            r2_list.append(r2)

        # Return mean R2 across batch
        r2_tensor = torch.stack(r2_list)
        # Handle NaN values: if all are NaN, return NaN; otherwise return mean of non-NaN values
        if torch.isnan(r2_tensor).all():
            return torch.tensor(float('nan'), device=preds.device)
        else:
            return torch.nanmean(r2_tensor)
    else:
        # No mask: flatten and calculate R2 on all values
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        ss_res = torch.sum((targets_flat - preds_flat) ** 2)
        ss_tot = torch.sum((targets_flat - torch.mean(targets_flat)) ** 2)

        if ss_tot < 1e-8:
            if ss_res < 1e-8:
                return torch.tensor(1.0, device=targets.device)
            else:
                return torch.tensor(float('inf'), device=targets.device)
        else:
            return 1 - ss_res / ss_tot


class EarlyStopping:

    def __init__(self, patience: int = 5, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Args:
            patience: 验证集损失不再改善时等待的epoch数
            min_delta: 被认为是改善的最小变化量
            restore_best_weights: 是否在早停时恢复最佳权重
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False
