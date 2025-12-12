#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Custom modules for RSImageDownscale task
  @Author Chris
  @Date 2025/11/12
"""
from datetime import datetime
from typing import List

import torch
from torch import nn


class TimeEmbedding(nn.Module):
    """
    增强的时间嵌入：结合 DOY 周期、年份趋势和多频率 Fourier 编码
    """

    def __init__(self, hidden_dim: int = 512, num_fourier: int = 8, max_doy: int = 366):
        """
        Args:
            hidden_dim: 输出隐藏维度
            num_fourier: Fourier 频率数量
            max_doy: 最大 DOY 值（通常为 366，考虑闰年）
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_fourier = num_fourier
        self.max_doy = max_doy

        # 1. DOY 年周期编码（单一频率，基础周期）
        self.doy_dim = 2

        # 2. 年份趋势编码
        self.year_dim = 1

        # 3. 多频率 Fourier 编码（指数频率）
        self.fourier_dim = 2 * num_fourier

        # 总维度
        total_dim = self.doy_dim + self.year_dim + self.fourier_dim

        # 投影到隐藏维度
        self.proj = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, dates: List[str]) -> torch.Tensor:
        """
        Args:
            dates: List[str] - 日期字符串列表，格式为 'YYYYMMDD'，如 ['20160101', '20160102', ...]
        Returns:
            [B, hidden_dim] - 时间嵌入
        """
        # 从日期字符串提取时间特征
        doys = []
        years = []
        for date_str in dates:
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            doys.append(date_obj.timetuple().tm_yday)
            years.append(date_obj.year)

        # 转换为 tensor（使用第一个参数的设备）
        device = next(self.proj.parameters()).device
        doy = torch.tensor(doys, dtype=torch.float32, device=device)
        year = torch.tensor(years, dtype=torch.float32, device=device)

        emb_list = []

        # 1. DOY 年周期编码（基础周期，单一频率）
        doy_norm = doy / self.max_doy
        doy_emb = torch.stack([
            torch.sin(2 * torch.pi * doy_norm),
            torch.cos(2 * torch.pi * doy_norm)
        ], dim=1)  # [B, 2]
        emb_list.append(doy_emb)

        # 2. 年份趋势编码（线性归一化）
        year_min = year.min().item() if len(year) > 0 else 2016
        year_max = year.max().item() if len(year) > 0 else 2020
        year_norm = (year - year_min) / max(year_max - year_min, 1.0)
        year_emb = year_norm.unsqueeze(1)  # [B, 1]
        emb_list.append(year_emb)

        # 3. 多频率 Fourier 编码（指数频率：2^k）
        # 从 year 和 doy 计算绝对时间索引（用于捕捉长期趋势）
        base_year = 2016
        time_index = (year - base_year) * 365 + doy
        t = time_index.unsqueeze(1)  # [B, 1]

        fourier_list = []
        for k in range(self.num_fourier):
            freq = 2 ** k
            fourier_list.append(torch.sin(2 * torch.pi * freq * t / 365))
            fourier_list.append(torch.cos(2 * torch.pi * freq * t / 365))
        fourier_emb = torch.cat(fourier_list, dim=1)  # [B, 2*num_fourier]
        emb_list.append(fourier_emb)

        # 拼接所有编码
        emb = torch.cat(emb_list, dim=1)  # [B, total_dim]

        # 投影到隐藏维度
        out = self.proj(emb)  # [B, hidden_dim]

        return out


class SpatialEmbedding(nn.Module):
    """
    Random Fourier Features (RFF) 空间嵌入：对标准化后的坐标进行多频率嵌入
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 512, num_freqs: int = 64):
        """
        Args:
            input_dim: 输入维度（通常为2，表示经纬度）
            hidden_dim: 输出隐藏维度
            num_freqs: 频率数量
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_freqs = num_freqs

        # 随机采样频率矩阵（RFF的核心）
        # 使用高斯分布采样，scale=1.0 是常见选择
        B = torch.randn(input_dim, num_freqs) * 1.0
        self.register_buffer('B', B)

        # 投影层
        self.proj = nn.Linear(num_freqs * 2, hidden_dim)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos: [B, input_dim] - 标准化后的坐标（[-1, 1]范围）
        Returns:
            [B, hidden_dim] - 空间嵌入
        """
        # RFF 变换：cos(2π * pos @ B)
        # pos: [B, input_dim], B: [input_dim, num_freqs]
        proj = 2 * torch.pi * pos @ self.B  # [B, num_freqs]

        # 使用 cos 和 sin（更稳定的表示）
        cos_emb = torch.cos(proj)  # [B, num_freqs]
        sin_emb = torch.sin(proj)  # [B, num_freqs]

        # 拼接
        rff = torch.cat([cos_emb, sin_emb], dim=-1)  # [B, num_freqs * 2]

        # 投影到隐藏维度
        out = self.proj(rff)  # [B, hidden_dim]

        return out


class FeatureAttention(nn.Module):
    """
    简化的特征注意力模块：只使用MultiheadAttention，不使用完整的Transformer
    适用于短序列（如3个token），更高效且参数更少
    """

    def __init__(self, hidden_dim: int = 512, num_heads: int = 8,
                 num_layers: int = 2, dropout: float = 0.1):
        """
        Args:
            hidden_dim: 隐藏维度
            num_heads: 注意力头数
            num_layers: 注意力层数
            dropout: Dropout比率
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # 多层注意力（每层都是独立的）
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # LayerNorm（每层一个）
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, num_tokens, hidden_dim] - 输入token序列
        Returns:
            [B, num_tokens, hidden_dim] - 编码后的token序列
        """
        x = tokens

        for attention, norm in zip(self.attention_layers, self.layer_norms):
            # Self-attention
            attn_out, _ = attention(x, x, x)  # [B, num_tokens, hidden_dim]

            # Residual connection + LayerNorm
            x = norm(x + attn_out)

        return x
