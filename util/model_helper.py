#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2025/10/15
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


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
