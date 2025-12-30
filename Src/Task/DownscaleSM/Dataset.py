#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Dataset for Task
  @Author Chris
  @Date 2025/11/12
"""

from functools import cached_property

import numpy as np
import torch
from torch.utils.data import Dataset

from Constant import *
from Util.TiffUtil import read_tiff, read_tiff_data
from Util.Util import get_valid_dates


class TrainDataset(Dataset):

    def __init__(self):
        self._load_data()
        self._filter_valid()
        self._norm()

    def _load_data(self):
        # Load DEM data
        dem_path = os.path.join(PROCESSED_DIR_PATH, DEM_NAME, RESOLUTION_36KM, f'{DEM_NAME}{TIFF_SUFFIX}')
        dem_data = read_tiff_data(dem_path).astype(np.float32)

        # Load multi-date data
        dates = get_valid_dates()
        self.dates = np.array(dates, dtype=object)
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

        # Load & Reshape position data
        _, lon_grid, lat_grid = read_tiff(REF_GRID_36KM_PATH, dst_epsg_code=4326)
        pos_grid = np.stack([lon_grid, lat_grid], axis=-1).astype(np.float32)
        pos_expanded = np.tile(pos_grid[np.newaxis, :, :, :], (date_num, 1, 1, 1))
        self.pos = pos_expanded.reshape(date_num * H * W, -1).astype(np.float32)

        self.date_indices = np.repeat(np.arange(date_num), H * W).astype(np.int32)

    def _filter_valid(self):
        valid = ~np.isnan(self.xs).any(axis=1) & ~np.isnan(self.ys).any(axis=1)

        self.xs = self.xs[valid]
        self.pos = self.pos[valid]
        self.ys = self.ys[valid, 0]
        self.date_indices = self.date_indices[valid]

    def _norm(self):
        self.x_mean = self.xs.mean(axis=0).astype(np.float32)
        self.x_std = self.xs.std(axis=0).astype(np.float32)
        self.x_std[self.x_std == 0] = 1.0
        self.xs = (self.xs - self.x_mean) / self.x_std

        self.y_mean = self.ys.mean()
        self.y_std = self.ys.std()
        self.ys = (self.ys - self.y_mean) / self.y_std

    def denorm_y(self, ys):
        if isinstance(ys, torch.Tensor):
            y_std = torch.tensor(self.y_std, dtype=ys.dtype, device=ys.device)
            y_mean = torch.tensor(self.y_mean, dtype=ys.dtype, device=ys.device)
            return ys * y_std + y_mean
        else:
            return ys * self.y_std + self.y_mean

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        xs = torch.from_numpy(self.xs[idx]).float()
        ys = torch.tensor(self.ys[idx], dtype=torch.float32)
        pos = torch.from_numpy(self.pos[idx]).float()
        date_idx = self.date_indices[idx]
        date = str(self.dates[date_idx])

        return xs, ys, pos, date


class InferenceDataset(Dataset):

    def __init__(self, date: str, resolution: str):
        self.date = date
        self.resolution = resolution
        self._load_data()
        self._filter_valid()
        self.train_dataset = TrainDataset()

    def _load_data(self):
        # Load DEM data
        dem_path = os.path.join(PROCESSED_DIR_PATH, DEM_NAME, self.resolution, f'{DEM_NAME}{TIFF_SUFFIX}')
        dem_data = read_tiff_data(dem_path).astype(np.float32)

        data_dict = {}
        for data_type in [NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME]:
            file_path = os.path.join(PROCESSED_DIR_PATH, data_type, self.resolution, f'{self.date}{TIFF_SUFFIX}')
            data_dict[data_type] = read_tiff_data(file_path).astype(np.float32)

        xs_date = np.stack([
            data_dict[NDVI_NAME],
            data_dict[LST_NAME],
            data_dict[ALBEDO_NAME],
            data_dict[PRECIPITATION_NAME],
            dem_data,
        ], axis=-1)  # [H, W, C]

        H, W = xs_date.shape[:2]
        self.xs = xs_date.reshape(H * W, -1).astype(np.float32)

        # Load & Reshape position data
        if self.resolution == RESOLUTION_1KM:
            ref_grid_path = REF_GRID_1KM_PATH
        else:
            ref_grid_path = REF_GRID_36KM_PATH
        _, grid_lon, grid_lat = read_tiff(ref_grid_path, dst_epsg_code=4326)
        pos_grid = np.stack([grid_lon, grid_lat], axis=-1).astype(np.float32)
        self.pos = pos_grid.reshape(H * W, -1).astype(np.float32)

        # Row & Col Index
        rows, cols = np.meshgrid(np.arange(H, dtype=np.int32), np.arange(W, dtype=np.int32), indexing='ij')
        self.rows = rows.flatten()
        self.cols = cols.flatten()

    def _filter_valid(self):
        valid = ~np.isnan(self.xs).any(axis=1)

        self.xs = self.xs[valid]
        self.pos = self.pos[valid]
        self.rows = self.rows[valid]
        self.cols = self.cols[valid]

    def norm_x(self, xs: torch.Tensor) -> torch.Tensor:
        x_mean = torch.tensor(self.train_dataset.x_mean, dtype=xs.dtype, device=xs.device)
        x_std = torch.tensor(self.train_dataset.x_std, dtype=xs.dtype, device=xs.device)

        return (xs - x_mean) / x_std

    def denorm_y(self, ys: torch.Tensor) -> torch.Tensor:
        return self.train_dataset.denorm_y(ys)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        xs = torch.from_numpy(self.xs[idx]).float()
        pos = torch.from_numpy(self.pos[idx]).float()
        date = self.date
        row = self.rows[idx]
        col = self.cols[idx]

        return xs, pos, date, row, col


class InsituStatsDataset:
    def __init__(self, resolution: str = RESOLUTION_36KM):
        self.resolution = resolution
        self._load_data()
        self._norm()

    def _load_data(self):
        insitu_dir = os.path.join(PROCESSED_DIR_PATH, IN_SITU_NAME, self.resolution)
        self.dates = np.array(get_valid_dates(), dtype=str)

        insitu_stats_list = []
        for date in self.dates:
            insitu_path = os.path.join(insitu_dir, f'{date}{TIFF_SUFFIX}')
            if not os.path.exists(insitu_path):
                insitu_stats_list.append(np.zeros(4, dtype=np.float32))
                continue

            insitu_data = read_tiff_data(insitu_path).astype(np.float32)
            raw_stats = self._calc_insitu_stats_from_data(insitu_data)
            insitu_stats_list.append(raw_stats)

        self.insitu_stats = np.array(insitu_stats_list, dtype=np.float32)

    @staticmethod
    def _calc_insitu_stats_from_data(insitu_data: np.ndarray) -> np.ndarray:
        valid_insitu = insitu_data[~np.isnan(insitu_data)]
        if len(valid_insitu) > 0:
            raw_stats = np.array([
                np.mean(valid_insitu),
                np.std(valid_insitu),
                np.percentile(valid_insitu, 25),
                np.percentile(valid_insitu, 75),
            ], dtype=np.float32)
        else:
            raw_stats = np.zeros(4, dtype=np.float32)
        return raw_stats

    def _norm(self):
        stats_mean = self.insitu_stats.mean(axis=0)
        stats_std = self.insitu_stats.std(axis=0)
        stats_std[stats_std == 0] = 1.0
        self.insitu_stats = (self.insitu_stats - stats_mean) / stats_std

    def get_stats(self, date: str) -> np.ndarray:
        date_idx = np.where(self.dates == date)[0]
        if len(date_idx) == 0:
            return np.zeros(4, dtype=np.float32)
        return self.insitu_stats[date_idx[0]]

    @cached_property
    def stats_dict(self) -> dict:
        return {str(date): self.insitu_stats[i] for i, date in enumerate(self.dates)}


class InsituDataset(Dataset):
    def __init__(self, resolution: str):
        self.resolution = resolution
        self.insitu_dir = os.path.join(PROCESSED_DIR_PATH, IN_SITU_NAME, self.resolution)

        ref_grid_path = REF_GRID_1KM_PATH if self.resolution == RESOLUTION_1KM else REF_GRID_36KM_PATH
        _, lon_grid, lat_grid = read_tiff(ref_grid_path, dst_epsg_code=4326)
        self.H, self.W = lon_grid.shape
        self._cache = {}

    def _load_date_data(self, date: str):
        if date in self._cache:
            return self._cache[date]
        insitu_path = os.path.join(self.insitu_dir, f'{date}{TIFF_SUFFIX}')
        insitu_map = read_tiff_data(insitu_path).astype(np.float32).flatten() if os.path.exists(
            insitu_path) else np.full(self.H * self.W, np.nan, dtype=np.float32)
        insitu_mask = (~np.isnan(insitu_map)).astype(np.float32)
        self._cache[date] = (insitu_map, insitu_mask)
        return self._cache[date]

    def get_data_by_date(self, date: str) -> tuple:
        insitu_map, insitu_mask = self._load_date_data(date)
        rows, cols = np.meshgrid(np.arange(self.H, dtype=np.int32), np.arange(self.W, dtype=np.int32), indexing='ij')
        return insitu_map, insitu_mask, rows.flatten(), cols.flatten()

    def get_data_by_date_row_col(self, dates: list, rows: np.ndarray, cols: np.ndarray) -> tuple:
        unique_dates = list(set(dates))
        date_data_cache = {date: self._load_date_data(date) for date in unique_dates}
        indices = rows.astype(np.int32) * self.W + cols.astype(np.int32)
        matched_insitus, matched_insitu_masks = [], []
        for i, date in enumerate(dates):
            insitu_map, insitu_mask = date_data_cache[date]
            matched_insitus.append(insitu_map[indices[i]])
            matched_insitu_masks.append(insitu_mask[indices[i]])
        return np.array(matched_insitus), np.array(matched_insitu_masks)


class InferenceResultDataset(Dataset):
    def __init__(self, resolution: str):
        self.resolution = resolution
        self.inference_dir = os.path.join(INFERENCE_DIR_PATH, self.resolution)
        ref_grid_path = REF_GRID_1KM_PATH if self.resolution == RESOLUTION_1KM else REF_GRID_36KM_PATH
        _, lon_grid, lat_grid = read_tiff(ref_grid_path, dst_epsg_code=4326)
        self.H, self.W = lon_grid.shape
        rows, cols = np.meshgrid(np.arange(self.H, dtype=np.int32), np.arange(self.W, dtype=np.int32), indexing='ij')
        self.rows_flat = rows.flatten()
        self.cols_flat = cols.flatten()
        self._cache = {}

    def _load_date_data(self, date: str):
        if date in self._cache:
            return self._cache[date]
        inference_tiff_path = os.path.join(self.inference_dir, f"{date}{TIFF_SUFFIX}")
        if not os.path.exists(inference_tiff_path):
            pred_map = np.full(self.H * self.W, np.nan, dtype=np.float32)
        else:
            pred_map = read_tiff_data(inference_tiff_path).astype(np.float32).flatten()
        valid_mask = ~np.isnan(pred_map)
        self._cache[date] = (pred_map, valid_mask)
        return self._cache[date]

    def get_data_by_date(self, date: str) -> tuple:
        pred_map, pred_mask = self._load_date_data(date)
        return pred_map, pred_mask, self.rows_flat, self.cols_flat
