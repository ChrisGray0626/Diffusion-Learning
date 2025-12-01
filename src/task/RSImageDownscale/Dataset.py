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

from Constant import DEM_NAME, RESOLUTION_36KM, TIFF_SUFFIX, STANDARD_GRID_36KM_PATH, \
    NDVI_NAME, PRECIPITATION_NAME, ALBEDO_NAME, LST_NAME, SM_NAME, PROCESSED_DIR_PATH
from util.TiffHandler import read_tiff, read_tiff_data
from util.Util import get_valid_dates


class RSImageDownscaleDataset(Dataset):

    def __init__(self):
        dem_path = os.path.join(PROCESSED_DIR_PATH, DEM_NAME, RESOLUTION_36KM, f'{DEM_NAME}{TIFF_SUFFIX}')
        self.dem_data = read_tiff_data(dem_path).astype(np.float32)
        self.image_height, self.image_width = self.dem_data.shape

        _, self.lon_grid, self.lat_grid = read_tiff(STANDARD_GRID_36KM_PATH, dst_epsg_code=4326)
        self.pos = np.stack([self.lat_grid, self.lon_grid], axis=-1).astype(np.float32)  # [H, W, 2]

        ndvi_dir = os.path.join(PROCESSED_DIR_PATH, NDVI_NAME, RESOLUTION_36KM)
        tif_files = glob.glob(os.path.join(ndvi_dir, f'*{TIFF_SUFFIX}'))
        self.dates = get_valid_dates()

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

        # Filter out samples with valid ratio < 0.3
        mask_sums = self.masks.sum(axis=(1, 2, 3))  # [num_dates]
        total_pixels = self.image_height * self.image_width
        valid_ratios = mask_sums / total_pixels  # [num_dates]
        valid_indices = valid_ratios >= 0.3  # [num_dates]

        self.dates = [self.dates[i] for i in range(len(self.dates)) if valid_indices[i]]
        self.xs = self.xs[valid_indices]  # [num_valid_dates, H, W, 5]
        self.ys = self.ys[valid_indices]  # [num_valid_dates, H, W, 1]
        self.masks = self.masks[valid_indices]  # [num_valid_dates, H, W, 1]

        # Normalize data
        self._normalize_data()

        # Apply mask again after normalization
        self.xs = self.xs * self.masks
        self.ys = self.ys * self.masks

    def denormalize_sm(self, sm_normalized: np.ndarray) -> np.ndarray:
        if not hasattr(self, 'sm_mean') or not hasattr(self, 'sm_std'):
            return sm_normalized
        return sm_normalized * self.sm_std + self.sm_mean

    def _load_date_data(self, date):
        dem = self.dem_data.copy()
        data_dict = {}

        for data_type in [NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, SM_NAME]:
            file_path = os.path.join(PROCESSED_DIR_PATH, data_type, RESOLUTION_36KM, f'{date}{TIFF_SUFFIX}')
            data_dict[data_type] = read_tiff_data(file_path).astype(np.float32)

        xs = np.stack([
            data_dict[NDVI_NAME],
            data_dict[LST_NAME],
            data_dict[ALBEDO_NAME],
            data_dict[PRECIPITATION_NAME],
            dem,
        ], axis=-1)  # [H, W, 5]

        ys = data_dict[SM_NAME][:, :, np.newaxis]  # [H, W, 1]

        valid_xs = ~np.isnan(xs).any(axis=-1)  # True where xs has no NaN (valid)
        valid_ys = ~np.isnan(ys).squeeze()  # True where ys has no NaN (valid)
        mask = (valid_xs & valid_ys).astype(np.float32)[:, :, np.newaxis]  # [H, W, 1], 1=valid, 0=invalid

        xs = np.nan_to_num(xs, nan=0.0, posinf=0.0, neginf=0.0)
        ys = np.nan_to_num(ys, nan=0.0, posinf=0.0, neginf=0.0)

        return xs, ys, mask

    def _normalize_data(self):
        """Normalize all input features (X) and SM (Y) using z-score for stable diffusion training"""
        feature_names = [NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, DEM_NAME]

        # Normalize input features using z-score
        for i, name in enumerate(feature_names):
            valid_data = self.xs[:, :, :, i][self.masks.squeeze() > 0]
            if len(valid_data) == 0:
                continue

            mean_val = valid_data.mean()
            std_val = valid_data.std()
            if std_val > 0:
                self.xs[:, :, :, i] = (self.xs[:, :, :, i] - mean_val) / std_val

        # Normalize SM (Y) using z-score for stable diffusion training
        # Note: We save normalization params for denormalization during evaluation
        valid_sm = self.ys[:, :, :, 0][self.masks.squeeze() > 0]
        if len(valid_sm) > 0:
            self.sm_mean = valid_sm.mean()
            self.sm_std = valid_sm.std()
            if self.sm_std > 0:
                self.ys[:, :, :, 0] = (self.ys[:, :, :, 0] - self.sm_mean) / self.sm_std
            else:
                self.sm_mean = 0.0
                self.sm_std = 1.0
        else:
            self.sm_mean = 0.0
            self.sm_std = 1.0

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        xs = torch.from_numpy(self.xs[idx]).permute(2, 0, 1)  # [H, W, 5] -> [5, H, W]
        ys = torch.from_numpy(self.ys[idx]).permute(2, 0, 1)  # [H, W, 1] -> [1, H, W]
        pos = torch.from_numpy(self.pos).permute(2, 0, 1)  # [H, W, 2] -> [2, H, W]
        masks = torch.from_numpy(self.masks[idx]).permute(2, 0, 1)  # [H, W, 1] -> [1, H, W]

        return xs, ys, pos, masks
