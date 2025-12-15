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
    增强的空间嵌入：使用确定性指数频率，分别处理经度和纬度
    结合原始坐标和多频率 Fourier 编码
    """

    def __init__(self, hidden_dim: int = 512, num_fourier: int = 6,
                 lon_min: float = -120, lon_max: float = -104,
                 lat_min: float = 35, lat_max: float = 49):
        """
        Args:
            hidden_dim: 输出隐藏维度
            num_fourier: Fourier 频率数量
            lon_min, lon_max, lat_min, lat_max: 空间范围
        """
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
        """
        Args:
            pos: [B, 2] - 原始坐标 [lon, lat]（未标准化）
        Returns:
            [B, hidden_dim] - 空间嵌入
        """
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

        # 拼接
        spatial_emb = torch.cat(emb_list, dim=1)

        # 投影到隐藏维度
        out = self.proj(spatial_emb)

        return out


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM): 使用条件信息调制特征
    """

    def __init__(self, hidden_dim: int = 512):
        """
        Args:
            hidden_dim: 特征维度
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # 从条件信息生成 scale 和 shift
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
        """
        Args:
            x: [B, hidden_dim] - 输入特征
            condition: [B, hidden_dim] - 条件信息（用于生成 scale 和 shift）
        Returns:
            [B, hidden_dim] - 调制后的特征
        """
        # 从条件信息生成 scale 和 shift
        scale = self.scale_net(condition)
        shift = self.shift_net(condition)

        # FiLM 调制：scale * x + shift
        out = scale * x + shift

        return out


class FiLMResBlock(nn.Module):
    """
    FiLM + ResBlock: 使用 FiLM 进行条件调制，然后通过 ResBlock 处理
    """

    def __init__(self, hidden_dim: int = 512):
        """
        Args:
            hidden_dim: 隐藏维度
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # ResBlock 网络
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # FiLM 模块
        self.film = FiLM(hidden_dim)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, hidden_dim] - 输入特征
            condition: [B, hidden_dim] - 条件信息（时间、空间等）
        Returns:
            [B, hidden_dim] - 输出特征
        """
        # FiLM 调制
        x_modulated = self.film(x, condition)

        # ResBlock
        out = x + self.net(x_modulated)

        return out
