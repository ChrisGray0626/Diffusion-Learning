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

    def _load_data(self):
        # Load DEM data
        dem_path = os.path.join(PROCESSED_DIR_PATH, DEM_NAME, RESOLUTION_36KM, f'{DEM_NAME}{TIFF_SUFFIX}')
        dem_data = read_tiff_data(dem_path).astype(np.float32)

        # Load position data
        _, lon_grid, lat_grid = read_tiff(STANDARD_GRID_36KM_PATH, dst_epsg_code=4326)
        pos_grid = np.stack([lon_grid, lat_grid], axis=-1).astype(np.float32)

        # Load multi-date data
        dates = get_valid_dates()
        xs_list = []
        ys_list = []
        for date in dates:
            data_dict = {}
            for data_type in [NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, SM_NAME]:
                file_path = os.path.join(PROCESSED_DIR_PATH, data_type, RESOLUTION_36KM, f'{date}{TIFF_SUFFIX}')
                data_dict[data_type] = read_tiff_data(file_path).astype(np.float32)

            xs_date = np.stack([
                data_dict[NDVI_NAME],
                data_dict[LST_NAME],
                data_dict[ALBEDO_NAME],
                data_dict[PRECIPITATION_NAME],
                dem_data,
            ], axis=-1)
            ys_date = data_dict[SM_NAME][:, :, np.newaxis]

            xs_list.append(xs_date)
            ys_list.append(ys_date)

        xs = np.array(xs_list, dtype=np.float32)
        ys = np.array(ys_list, dtype=np.float32)

        # Reshape X & Y data
        date_num, H, W = xs.shape[:3]
        self.xs = xs.reshape(date_num * H * W, -1).astype(np.float32)
        self.ys = ys.reshape(date_num * H * W, -1).astype(np.float32)

        # Reshape position data
        pos_expanded = np.tile(pos_grid[np.newaxis, :, :, :], (date_num, 1, 1, 1))
        pos_flat = pos_expanded.reshape(date_num * H * W, -1).astype(np.float32)

        self.pos = pos_flat

        # Reshape date data
        dates = np.repeat(dates, H * W)
        self.dates = np.array(dates, dtype=object)

    def _filter_valid(self):
        valid = ~np.isnan(self.xs).any(axis=1) & ~np.isnan(self.ys).any(axis=1)

        self.xs = self.xs[valid]
        self.pos = self.pos[valid]
        self.ys = self.ys[valid, 0]
        self.dates = self.dates[valid]

    def _normalize_feat(self):
        for i in range(self.xs.shape[1]):
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
