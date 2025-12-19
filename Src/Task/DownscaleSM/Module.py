#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Module for Task
  @Author Chris
  @Date 2025/11/12
"""
from datetime import datetime
from typing import List

import torch
from torch import nn


class TimeEmbedding(nn.Module):

    def __init__(self, hidden_dim: int, num_fourier: int = 8, max_doy: int = 366):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_fourier = num_fourier
        self.max_doy = max_doy

        # DOY 年周期编码（单一频率，基础周期）
        self.doy_dim = 2
        # 年份趋势编码
        self.year_dim = 1
        # 多频率 Fourier 编码（指数频率）
        self.fourier_dim = 2 * num_fourier

        total_dim = self.doy_dim + self.year_dim + self.fourier_dim
        self.proj = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, dates: List[str]) -> torch.Tensor:
        # 从日期字符串提取时间特征
        doys = []
        years = []
        for date in dates:
            date_obj = datetime.strptime(date, '%Y%m%d')
            doys.append(date_obj.timetuple().tm_yday)
            years.append(date_obj.year)

        # 转换为 tensor（使用第一个参数的设备）
        device = next(self.proj.parameters()).device
        doy = torch.tensor(doys, dtype=torch.float32, device=device)
        year = torch.tensor(years, dtype=torch.float32, device=device)

        emb_list = []

        # DOY 年周期编码（基础周期，单一频率）
        doy_norm = doy / self.max_doy
        doy_emb = torch.stack([
            torch.sin(2 * torch.pi * doy_norm),
            torch.cos(2 * torch.pi * doy_norm)
        ], dim=1)  # [B, 2]
        emb_list.append(doy_emb)

        # 年份趋势编码（线性归一化）
        year_min = year.min().item() if len(year) > 0 else 2016
        year_max = year.max().item() if len(year) > 0 else 2020
        year_norm = (year - year_min) / max(year_max - year_min, 1.0)
        year_emb = year_norm.unsqueeze(1)  # [B, 1]
        emb_list.append(year_emb)

        # 多频率 Fourier 编码（指数频率：2^k）
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

        emb = torch.cat(emb_list, dim=1)  # [B, total_dim]
        out = self.proj(emb)  # [B, hidden_dim]

        return out


class SpatialEmbedding(nn.Module):

    def __init__(self, hidden_dim: int, num_fourier: int = 6,
                 lon_min: float = -120, lon_max: float = -104,
                 lat_min: float = 35, lat_max: float = 49):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_fourier = num_fourier
        self.lon_min, self.lon_max = lon_min, lon_max
        self.lat_min, self.lat_max = lat_min, lat_max

        # 特征维度：2 (原始 lon, lat) + 4 * num_fourier (每个频率对 lon/lat 的 sin/cos)
        spatial_dim = 2 + 4 * num_fourier

        # 投影到隐藏维度
        self.proj = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        lon = pos[:, 0]
        lat = pos[:, 1]

        # 归一化到 [-1, 1]
        lon_n = 2 * (lon - self.lon_min) / (self.lon_max - self.lon_min) - 1
        lat_n = 2 * (lat - self.lat_min) / (self.lat_max - self.lat_min) - 1

        emb_list = [lon_n.unsqueeze(1), lat_n.unsqueeze(1)]

        # 多频率 Fourier 编码（分别处理 lon 和 lat）
        for k in range(self.num_fourier):
            freq = 2 ** k
            emb_list.append(torch.sin(freq * torch.pi * lon_n).unsqueeze(1))
            emb_list.append(torch.cos(freq * torch.pi * lon_n).unsqueeze(1))
            emb_list.append(torch.sin(freq * torch.pi * lat_n).unsqueeze(1))
            emb_list.append(torch.cos(freq * torch.pi * lat_n).unsqueeze(1))

        spatial_emb = torch.cat(emb_list, dim=1)
        out = self.proj(spatial_emb)

        return out


class FiLM(nn.Module):

    def __init__(self, hidden_dim: int):

        super().__init__()
        self.hidden_dim = hidden_dim

        self.scale_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.shift_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale = self.scale_net(condition)
        shift = self.shift_net(condition)
        out = scale * x + shift

        return out


class FiLMResBlock(nn.Module):

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.film = FiLM(hidden_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        x_modulated = self.film(x, condition)
        out = x + self.net(x_modulated)

        return out
