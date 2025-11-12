#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Remote Sensing Image Downscaling Dataset
  @Author Chris
  @Date 2025/11/12
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from constant import DATA_DIR_PATH, DEM_NAME, RESOLUTION_36KM, TIFF_SUFFIX, STANDARD_GRID_36KM_PATH, \
    NDVI_NAME, PRECIPITATION_NAME, ALBEDO_NAME, LST_NAME, SM_NAME
from util.TiffHandler import read_tiff, read_tiff_data


class RSImageDownscaleDataset(Dataset):
    DEFAULT_RANGES = {
        NDVI_NAME: (-2.0, 10.0),
        ALBEDO_NAME: (0.0, 1.0),
        SM_NAME: (0.02, 1.0),
    }

    def __init__(self):
        dem_path = os.path.join(DATA_DIR_PATH, DEM_NAME, f'{RESOLUTION_36KM}{TIFF_SUFFIX}')
        self.dem_data = read_tiff_data(dem_path).astype(np.float32)
        self.image_height, self.image_width = self.dem_data.shape

        _, self.lon_grid, self.lat_grid = read_tiff(STANDARD_GRID_36KM_PATH, dst_epsg_code=4326)
        self.pos = np.stack([self.lat_grid, self.lon_grid], axis=-1).astype(np.float32)  # [H, W, 2]

        ndvi_dir = os.path.join(DATA_DIR_PATH, NDVI_NAME, RESOLUTION_36KM)
        tif_files = glob.glob(os.path.join(ndvi_dir, f'*{TIFF_SUFFIX}'))
        self.dates = sorted([os.path.splitext(os.path.basename(f))[0] for f in tif_files])

        # Build X & Y
        self.xs = []
        self.ys = []
        self.masks = []

        for date in self.dates:
            xs_date, ys_date, mask_date = self._load_date_data(date)
            self.xs.append(xs_date)
            self.ys.append(ys_date)
            self.masks.append(mask_date)

        self.xs = np.array(self.xs).astype(np.float32)  # [num_dates, H, W, 5]
        self.ys = np.array(self.ys).astype(np.float32)  # [num_dates, H, W, 1]
        self.masks = np.array(self.masks).astype(np.float32)  # [num_dates, H, W, 1]

        # Filter out samples with no valid data
        mask_sums = self.masks.sum(axis=(1, 2, 3))  # [num_dates]
        valid_indices = mask_sums > 0  # [num_dates]

        if valid_indices.sum() < len(self.dates):
            num_filtered = len(self.dates) - valid_indices.sum()
            print(f"\nFiltering out {num_filtered} samples with no valid data:")
            for i, date in enumerate(self.dates):
                if not valid_indices[i]:
                    print(f"  - {date}")

        # Keep only valid samples
        self.dates = [self.dates[i] for i in range(len(self.dates)) if valid_indices[i]]
        self.xs = self.xs[valid_indices]  # [num_valid_dates, H, W, 5]
        self.ys = self.ys[valid_indices]  # [num_valid_dates, H, W, 1]
        self.masks = self.masks[valid_indices]  # [num_valid_dates, H, W, 1]

        # Normalize data
        self._normalize_data()

        # Apply mask again after normalization
        self.xs = self.xs * self.masks
        self.ys = self.ys * self.masks

    def _load_date_data(self, date):
        dem = self.dem_data.copy()
        data_dict = {}

        for data_type in [NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, SM_NAME]:
            file_path = os.path.join(DATA_DIR_PATH, data_type, RESOLUTION_36KM, f'{date}{TIFF_SUFFIX}')
            data_dict[data_type] = read_tiff_data(file_path).astype(np.float32)

        xs = np.stack([
            data_dict[NDVI_NAME],
            data_dict[LST_NAME],
            data_dict[ALBEDO_NAME],
            data_dict[PRECIPITATION_NAME],
            dem,
        ], axis=-1)  # [H, W, 5]

        ys = data_dict[SM_NAME][:, :, np.newaxis]  # [H, W, 1]

        valid_xs = ~np.isnan(xs).any(axis=-1)
        valid_ys = ~np.isnan(ys).squeeze()
        mask = (valid_xs & valid_ys).astype(np.float32)[:, :, np.newaxis]  # [H, W, 1]

        xs = np.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)
        ys = np.nan_to_num(ys, nan=0.0, posinf=0.0, neginf=0.0)

        return xs, ys, mask

    def _normalize_data(self):
        """Normalize data: variables with DEFAULT_RANGES use minmax, others use z-score"""
        feature_names = [NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, DEM_NAME]

        # Normalize input features
        for i, name in enumerate(feature_names):
            valid_data = self.xs[:, :, :, i][self.masks.squeeze() > 0]
            if len(valid_data) == 0:
                continue

            if name in self.DEFAULT_RANGES:
                # MinMax normalization
                min_val, max_val = self.DEFAULT_RANGES[name]
                x_range = max_val - min_val
                if x_range > 0:
                    self.xs[:, :, :, i] = (self.xs[:, :, :, i] - min_val) / x_range
                    self.xs[:, :, :, i] = np.clip(self.xs[:, :, :, i], 0.0, 1.0)
            else:
                # Z-score normalization
                mean_val = valid_data.mean()
                std_val = valid_data.std()
                if std_val > 0:
                    self.xs[:, :, :, i] = (self.xs[:, :, :, i] - mean_val) / std_val

        # Normalize output feature (SM)
        valid_sm = self.ys[:, :, :, 0][self.masks.squeeze() > 0]
        if len(valid_sm) > 0:
            if SM_NAME in self.DEFAULT_RANGES:
                # MinMax normalization
                min_val, max_val = self.DEFAULT_RANGES[SM_NAME]
                y_range = max_val - min_val
                if y_range > 0:
                    self.ys[:, :, :, 0] = (self.ys[:, :, :, 0] - min_val) / y_range
                    self.ys[:, :, :, 0] = np.clip(self.ys[:, :, :, 0], 0.0, 1.0)
            else:
                # Z-score normalization
                mean_val = valid_sm.mean()
                std_val = valid_sm.std()
                if std_val > 0:
                    self.ys[:, :, :, 0] = (self.ys[:, :, :, 0] - mean_val) / std_val

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        xs = torch.from_numpy(self.xs[idx]).permute(2, 0, 1)  # [H, W, 5] -> [5, H, W]
        ys = torch.from_numpy(self.ys[idx]).permute(2, 0, 1)  # [H, W, 1] -> [1, H, W]
        pos = torch.from_numpy(self.pos).permute(2, 0, 1)  # [H, W, 2] -> [2, H, W]
        masks = torch.from_numpy(self.masks[idx]).permute(2, 0, 1)  # [H, W, 1] -> [1, H, W]

        return xs, ys, pos, masks
