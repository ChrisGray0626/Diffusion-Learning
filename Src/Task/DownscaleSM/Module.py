#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Module for Task
  @Author Chris
  @Date 2025/11/12
"""
import os
from datetime import datetime
from typing import List

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn

from Constant import CHECKPOINT_DIR_PATH


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


class BiasCorrector:

    def __init__(self, model_name: str = "rf_bias_corrector", n_estimators: int = 300, max_depth: int = None,
                 max_features: int = 4, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.model = None
        self.model_dir = os.path.join(CHECKPOINT_DIR_PATH, "DownscaleSM", "RF")
        self.model_path = os.path.join(self.model_dir, f"{model_name}.joblib")

        if os.path.exists(self.model_path):
            self.load()

    def train(self, pred_ys: np.ndarray, insitus: np.ndarray, insitu_masks: np.ndarray,
              aux_feats: np.ndarray):
        valid_mask = insitu_masks > 0
        pred_ys = pred_ys[valid_mask]
        insitus = insitus[valid_mask]
        aux_feats = aux_feats[valid_mask]

        X = np.column_stack([pred_ys, aux_feats]).astype(np.float32)
        y = insitus.astype(np.float32)

        # 划分训练集和验证集
        x_train, x_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        # 训练随机森林模型
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            n_jobs=-1,
            random_state=self.random_state,
            oob_score=True
        )
        self.model.fit(x_train, y_train)

        y_pred_val = self.model.predict(x_val)
        mse = mean_squared_error(y_val, y_pred_val)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred_val)

        print(f"\nRF Bias Corrector Training:")
        print(f"  Training samples: {len(x_train):,}, Validation samples: {len(x_val):,}")
        print(f"  Validation RMSE: {rmse:.6f}, R2: {r2:.4f}")
        print(f"  OOB Score: {self.model.oob_score_:.4f}")

        # Save Model
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.model, self.model_path)

        return self.model

    def load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"RF bias corrector model not found: {self.model_path}")

        self.model = joblib.load(self.model_path)
        return self.model

    def predict(self, pred_ys: np.ndarray, aux_feats: np.ndarray) -> np.ndarray:
        X = np.column_stack([pred_ys, aux_feats]).astype(np.float32)
        return self.model.predict(X).astype(np.float32)
