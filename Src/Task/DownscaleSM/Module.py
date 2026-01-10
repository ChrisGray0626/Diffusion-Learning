#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Module for Task
  @Author Chris
  @Date 2025/11/12
"""
from datetime import datetime
from typing import List

import numpy as np
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from Constant import RANGE
from Util.ModelHelper import SinusoidalPosEmb


class NoisePredictor(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self, input_feature_num: int,
                 hidden_dim: int = 512, timestep_emb_dim: int = 128,
                 res_block_num: int = 3):
        super().__init__()
        self.input_feature_num = input_feature_num
        self.hidden_dim = hidden_dim

        input_dim = input_feature_num + 1
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Timestep Embedding
        self.timestep_embedding = nn.Sequential(
            SinusoidalPosEmb(timestep_emb_dim),
            nn.Linear(timestep_emb_dim, timestep_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(timestep_emb_dim * 4, timestep_emb_dim * 4),
        )
        self.emb_timestep2hidden = nn.Linear(timestep_emb_dim * 4, hidden_dim)

        # Time Embedding
        self.time_embedding = TimeEmbedding(
            hidden_dim=hidden_dim,
            num_fourier=8
        )

        # Spatial Embedding
        lon_min, lat_min, lon_max, lat_max = RANGE
        self.spatial_embedding = SpatialEmbedding(
            hidden_dim=hidden_dim,
            num_fourier=6,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max
        )

        # Insitu Stats Embedding
        self.insitu_stats_embedding = InsituStatsEmbedding(
            hidden_dim=hidden_dim,
            stats_dim=4
        )

        # Condition Fusion
        self.condition_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Residual Block
        self.res_blocks = nn.ModuleList([
            FiLMResBlock(hidden_dim=hidden_dim)
            for _ in range(res_block_num)
        ])

        # Output Layer
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, diffused_ys: torch.Tensor, xs: torch.Tensor, timesteps: torch.Tensor,
                pos: torch.Tensor, dates: List[str],
                insitu_stats: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([xs, diffused_ys], dim=1)
        x = self.input_layer(inputs)

        # Embed Timestep
        embed_timesteps = self.timestep_embedding(timesteps)
        embed_timesteps = self.emb_timestep2hidden(embed_timesteps)

        # Embed Time
        embed_time = self.time_embedding(dates)

        # Embed Spatial
        embed_spatial = self.spatial_embedding(pos)

        # Embed Insitu Stats
        embed_insitu_stats = self.insitu_stats_embedding(insitu_stats)

        # Fuse Condition
        condition = torch.cat([embed_timesteps, embed_time, embed_spatial, embed_insitu_stats], dim=1)
        condition = self.condition_fusion(condition)

        # Residual Blocks with FiLM
        for res_block in self.res_blocks:
            x = res_block(x, condition)

        out = self.output_layer(x)

        return out


class TimeEmbedding(nn.Module):

    def __init__(self, hidden_dim: int, num_fourier: int = 8, max_doy: int = 366):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_fourier = num_fourier
        self.max_doy = max_doy
        self.doy_dim = 2
        self.year_dim = 1
        self.fourier_dim = 2 * num_fourier
        total_dim = self.doy_dim + self.year_dim + self.fourier_dim
        self.proj = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, dates: List[str]) -> torch.Tensor:
        doys, years = [], []
        for date in dates:
            date_obj = datetime.strptime(date, '%Y%m%d')
            doys.append(date_obj.timetuple().tm_yday)
            years.append(date_obj.year)

        device = next(self.proj.parameters()).device
        doy = torch.tensor(doys, dtype=torch.float32, device=device)
        year = torch.tensor(years, dtype=torch.float32, device=device)

        emb_list = []
        doy_norm = doy / self.max_doy
        doy_emb = torch.stack([torch.sin(2 * torch.pi * doy_norm), torch.cos(2 * torch.pi * doy_norm)], dim=1)
        emb_list.append(doy_emb)

        year_min = year.min().item() if len(year) > 0 else 2016
        year_max = year.max().item() if len(year) > 0 else 2020
        year_norm = (year - year_min) / max(year_max - year_min, 1.0)
        emb_list.append(year_norm.unsqueeze(1))

        base_year = 2016
        time_index = (year - base_year) * 365 + doy
        t = time_index.unsqueeze(1)
        fourier_list = []
        for k in range(self.num_fourier):
            freq = 2 ** k
            fourier_list.append(torch.sin(2 * torch.pi * freq * t / 365))
            fourier_list.append(torch.cos(2 * torch.pi * freq * t / 365))
        emb_list.append(torch.cat(fourier_list, dim=1))

        return self.proj(torch.cat(emb_list, dim=1))


class SpatialEmbedding(nn.Module):

    def __init__(self, hidden_dim: int, num_fourier: int = 6,
                 lon_min: float = -120, lon_max: float = -104,
                 lat_min: float = 35, lat_max: float = 49):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_fourier = num_fourier
        self.lon_min, self.lon_max = lon_min, lon_max
        self.lat_min, self.lat_max = lat_min, lat_max
        spatial_dim = 2 + 4 * num_fourier
        self.proj = nn.Sequential(
            nn.Linear(spatial_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        lon, lat = pos[:, 0], pos[:, 1]
        lon_n = 2 * (lon - self.lon_min) / (self.lon_max - self.lon_min) - 1
        lat_n = 2 * (lat - self.lat_min) / (self.lat_max - self.lat_min) - 1

        emb_list = [lon_n.unsqueeze(1), lat_n.unsqueeze(1)]
        for k in range(self.num_fourier):
            freq = 2 ** k
            emb_list.append(torch.sin(freq * torch.pi * lon_n).unsqueeze(1))
            emb_list.append(torch.cos(freq * torch.pi * lon_n).unsqueeze(1))
            emb_list.append(torch.sin(freq * torch.pi * lat_n).unsqueeze(1))
            emb_list.append(torch.cos(freq * torch.pi * lat_n).unsqueeze(1))

        return self.proj(torch.cat(emb_list, dim=1))


class InsituStatsEmbedding(nn.Module):

    def __init__(self, hidden_dim: int, stats_dim: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stats_dim = stats_dim
        self.proj = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, insitu_stats: torch.Tensor) -> torch.Tensor:
        return self.proj(insitu_stats)


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


class BiasCorrector:

    def __init__(self, n_estimators: int = 300, max_depth: int = None,
                 max_features: int = 4, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.model = None

    def train(self, pred_ys: np.ndarray, insitus: np.ndarray,
              aux_feats: np.ndarray, verbose: bool = True):
        X = np.column_stack([pred_ys, aux_feats]).astype(np.float32)
        y = insitus.astype(np.float32)

        x_train, x_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        n_samples = len(X)
        n_est = min(self.n_estimators, max(50, n_samples // 10))

        self.model = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=self.max_depth,
            max_features=self.max_features,
            n_jobs=-1,
            random_state=self.random_state,
            oob_score=n_samples > 50
        )
        self.model.fit(X, y)

        if verbose:
            y_pred_val = self.model.predict(x_val)
            mse = mean_squared_error(y_val, y_pred_val)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val, y_pred_val)

            print(f"RF Bias Corrector Training:")
            print(f"  Training samples: {len(x_train):,}, Validation samples: {len(x_val):,}")
            print(f"  Validation RMSE: {rmse:.6f}, R2: {r2:.4f}")
            if hasattr(self.model, 'oob_score_') and self.model.oob_score_ is not None:
                print(f"  OOB Score: {self.model.oob_score_:.4f}")

        return self.model

    def predict(self, pred_ys: np.ndarray, aux_feats: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = np.column_stack([pred_ys, aux_feats]).astype(np.float32)
        return self.model.predict(X).astype(np.float32)
