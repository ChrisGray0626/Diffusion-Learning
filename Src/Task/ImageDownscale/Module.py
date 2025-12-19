#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2025/11/12
"""
import torch
from torch import nn

from Util.ModelHelper import BaseResBlock


class ChannelAttention(nn.Module):

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.dim = dim
        self.reduction = reduction

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_dim = max(dim // reduction, 1)
        # Shared MLP
        self.MLP = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, dim, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Average pooling: [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        avg_out = self.avg_pool(x).view(B, C)
        # Max pooling: [B, C, H, W] -> [B, C, 1, 1] -> [B, C]
        max_out = self.max_pool(x).view(B, C)

        avg_out = self.MLP(avg_out)
        max_out = self.MLP(max_out)

        out = avg_out + max_out
        out = self.sigmoid(out)

        out = out.view(B, C, 1, 1)

        return x * out


class ResBlock(BaseResBlock):

    def __init__(self, hidden_dim):
        super().__init__(hidden_dim)
        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
