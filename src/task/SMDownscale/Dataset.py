#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Dataset for Task
  @Author Chris
  @Date 2025/11/12
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from Constant import DEM_NAME, RESOLUTION_36KM, TIFF_SUFFIX, STANDARD_GRID_36KM_PATH, \
    NDVI_NAME, PRECIPITATION_NAME, ALBEDO_NAME, LST_NAME, SM_NAME, PROCESSED_DIR_PATH
from util.TiffHandler import read_tiff, read_tiff_data
from util.Util import get_valid_dates


class RSImageDownscaleDataset(Dataset):

    def __init__(self):
        self._load_data()
        self._filter_valid()
        self._normalize_feat()
        self.date_num = len(set(self.dates))

    def _load_data(self):
        dem_path = os.path.join(PROCESSED_DIR_PATH, DEM_NAME, RESOLUTION_36KM, f'{DEM_NAME}{TIFF_SUFFIX}')
        dem_data = read_tiff_data(dem_path).astype(np.float32)

        _, lon_grid, lat_grid = read_tiff(STANDARD_GRID_36KM_PATH, dst_epsg_code=4326)
        pos_grid = np.stack([lat_grid, lon_grid], axis=-1).astype(np.float32)

        dates = get_valid_dates()
        xs = []
        ys = []

        for date in dates:
            xs_date, ys_date = self._load_single_date_data(date, dem_data)
            xs.append(xs_date)
            ys.append(ys_date)

        xs = np.array(xs).astype(np.float32)
        ys = np.array(ys).astype(np.float32)

        date_num, H, W = xs.shape[:3]

        self.xs = xs.reshape(date_num * H * W, -1).astype(np.float32)
        self.ys = ys.reshape(date_num * H * W, -1).astype(np.float32)

        pos_expanded = np.tile(pos_grid[np.newaxis, :, :, :], (date_num, 1, 1, 1))
        pos_flat = pos_expanded.reshape(date_num * H * W, -1).astype(np.float32)

        # 存储原始坐标（不归一化），让 SpatialEmbedding 模块内部处理
        self.pos = pos_flat[:, [1, 0]]  # [lon, lat] 顺序

        dates = np.repeat(dates, H * W)
        self.dates = np.array(dates, dtype=object)

    # TODO _load_single_date_data
    def _load_single_date_data(self, date, dem_data):
        data_dict = {}
        for data_type in [NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, SM_NAME]:
            file_path = os.path.join(PROCESSED_DIR_PATH, data_type, RESOLUTION_36KM, f'{date}{TIFF_SUFFIX}')
            data_dict[data_type] = read_tiff_data(file_path).astype(np.float32)

        xs = np.stack([
            data_dict[NDVI_NAME],
            data_dict[LST_NAME],
            data_dict[ALBEDO_NAME],
            data_dict[PRECIPITATION_NAME],
            dem_data,
        ], axis=-1)

        ys = data_dict[SM_NAME][:, :, np.newaxis]

        return xs, ys

    def _filter_valid(self):
        valid = ~np.isnan(self.xs).any(axis=1) & ~np.isnan(self.ys).any(axis=1)

        self.xs = self.xs[valid]
        self.pos = self.pos[valid]
        self.ys = self.ys[valid, 0]
        self.dates = self.dates[valid]

    def _normalize_feat(self):
        feature_names = [NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, DEM_NAME]

        for i, _ in enumerate(feature_names):
            mean_val = self.xs[:, i].mean()
            std_val = self.xs[:, i].std()
            if std_val > 0:
                self.xs[:, i] = (self.xs[:, i] - mean_val) / std_val

        self.y_mean = self.ys.mean()
        self.y_std = self.ys.std()
        if self.y_std > 0:
            self.ys = (self.ys - self.y_mean) / self.y_std
        else:
            self.y_mean = 0.0
            self.y_std = 1.0

    def denormalize_y(self, ys):
        return ys * self.y_std + self.y_mean

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        xs = torch.from_numpy(self.xs[idx]).float()
        ys = torch.tensor(self.ys[idx], dtype=torch.float32)
        pos = torch.from_numpy(self.pos[idx]).float()
        date = str(self.dates[idx])

        return xs, ys, pos, date

# TODO 1km Dataset for Prediction
