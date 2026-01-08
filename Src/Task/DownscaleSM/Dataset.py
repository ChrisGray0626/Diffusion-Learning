#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Dataset for Task
  @Author Chris
  @Date 2025/11/12
"""
from typing import Dict, Optional, Hashable, TypeVar, Generic, Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from Constant import *
from Util.TiffUtil import read_tiff, read_tiff_data
from Util.Util import get_valid_dates


class TrainDataset(Dataset):

    def __init__(self, resolution: str = RESOLUTION_36KM):
        self.resolution = resolution
        self.data_store = DataStore(resolution=self.resolution)
        self.grid_info_store = GridInfoStore(resolution=self.resolution)
        self.data_names = [NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, DEM_NAME]
        self._load_data()
        self._filter_valid()
        self._norm()

    def _load_data(self):
        dates = get_valid_dates()
        self.dates = np.array(dates, dtype=object)

        grid_info = self.grid_info_store.get()

        xs_list, ys_list = [], []

        for date in dates:
            xs_date = np.stack([
                self.data_store.get(name, date) for name in self.data_names
            ], axis=-1)
            ys_date = self.data_store.get(SM_NAME, date)[:, :, np.newaxis]

            xs_list.append(xs_date)
            ys_list.append(ys_date)

        xs = np.array(xs_list, dtype=np.float32)
        ys = np.array(ys_list, dtype=np.float32)

        date_num = len(dates)
        H = grid_info["H"]
        W = grid_info["W"]
        self.xs = xs.reshape(date_num * H * W, -1).astype(np.float32)
        self.ys = ys.reshape(date_num * H * W, -1).astype(np.float32)

        pos_grid = grid_info["pos"]
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
        self.data_store = DataStore(resolution=self.resolution)
        self.insitu_stats_store = InsituStatsStore(resolution=self.resolution)
        self.grid_info_store = GridInfoStore(resolution=self.resolution)
        self.data_names = [NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, DEM_NAME]
        self.train_dataset = TrainDataset()

        self._load_data()
        self._filter_valid()
        self._norm()

    def _load_data(self):
        xs = np.stack([self.data_store.get(name, self.date) for name in self.data_names], axis=-1)

        grid_info = self.grid_info_store.get()
        H, W = grid_info["H"], grid_info["W"]

        self.xs = xs.reshape(H * W, -1).astype(np.float32)
        self.pos = grid_info["pos"].reshape(H * W, -1).astype(np.float32)
        self.rows = grid_info["rows"].flatten()
        self.cols = grid_info["cols"].flatten()

        self.insitu_stats = self.insitu_stats_store.get(self.date)

    def _filter_valid(self):
        valid = ~np.isnan(self.xs).any(axis=1)

        self.xs = self.xs[valid]
        self.pos = self.pos[valid]
        self.rows = self.rows[valid]
        self.cols = self.cols[valid]

    def _norm(self):
        x_mean = self.train_dataset.x_mean
        x_std = self.train_dataset.x_std
        self.xs = (self.xs - x_mean) / x_std

    def denorm_y(self, ys: torch.Tensor) -> torch.Tensor:
        return self.train_dataset.denorm_y(ys)

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        xs = self.xs[idx]
        pos = self.pos[idx]
        date = self.date
        row = self.rows[idx]
        col = self.cols[idx]

        return xs, pos, date, row, col


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


# TODO 只评估实际参与训练的点
class InferenceEvaluationDataset(Dataset):

    def __init__(self, resolution: str):
        self.resolution = resolution
        self.inference_result_store = InferenceResultStore(resolution=self.resolution)
        self.data_store = DataStore(resolution=self.resolution)
        self.grid_info_store = GridInfoStore(resolution=self.resolution)

        grid_info = self.grid_info_store.get()
        self.H, self.W = grid_info["H"], grid_info["W"]
        self.rows = grid_info["rows"]
        self.cols = grid_info["cols"]

    def get(self, date: str) -> tuple:
        pred_map = self.inference_result_store.get(date)
        insitu_map = self.data_store.get(IN_SITU_NAME, date)

        pred_mask = (~np.isnan(pred_map)).astype(np.float32)
        insitu_mask = (~np.isnan(insitu_map)).astype(np.float32)
        valid_masks = (pred_mask > 0) & (insitu_mask > 0)

        return pred_map, insitu_map, valid_masks, self.rows, self.cols

    def get_all(self) -> tuple:
        dates = get_valid_dates()
        all_pred_map, all_insitus_map, all_valid_masks, all_rows, all_cols, all_dates = [], [], [], [], [], []

        for date in dates:
            pred_map, insitus_map, valid_masks, rows, cols = self.get(date)

            all_pred_map.append(pred_map)
            all_insitus_map.append(insitus_map)
            all_valid_masks.append(valid_masks)
            all_rows.append(rows)
            all_cols.append(cols)
            all_dates.extend([date] * self.H * self.W)

        pred_map = np.concatenate(all_pred_map)
        insitus_map = np.concatenate(all_insitus_map)
        valid_masks = np.concatenate(all_valid_masks)
        rows = np.concatenate(all_rows)
        cols = np.concatenate(all_cols)

        return pred_map, insitus_map, valid_masks, all_dates, rows, cols


class DataCoverageDataset(Dataset):

    def __init__(self, resolution: str):
        self.resolution = resolution
        self.data_store = DataStore(resolution=self.resolution)
        self.grid_info_store = GridInfoStore(resolution=self.resolution)

        grid_info = self.grid_info_store.get()
        self.H, self.W = grid_info["H"], grid_info["W"]
        self.rows = grid_info["rows"]
        self.cols = grid_info["cols"]
        self.feature_names = [NDVI_NAME, LST_NAME, ALBEDO_NAME, PRECIPITATION_NAME, DEM_NAME, SM_NAME, IN_SITU_NAME]

    def get(self, date: str) -> tuple:
        features = {}
        for name in self.feature_names:
            try:
                if name == DEM_NAME:
                    features[name] = self.data_store.get(name, None)
                else:
                    features[name] = self.data_store.get(name, date)
            except FileNotFoundError:
                features[name] = np.full((self.H, self.W), np.nan, dtype=np.float32)

        train_valid = ~np.isnan(features[NDVI_NAME]) & ~np.isnan(features[LST_NAME]) & \
                      ~np.isnan(features[ALBEDO_NAME]) & ~np.isnan(features[PRECIPITATION_NAME]) & \
                      ~np.isnan(features[DEM_NAME]) & ~np.isnan(features[SM_NAME])
        insitu_valid = ~np.isnan(features[IN_SITU_NAME])

        all_positions_mask = train_valid | insitu_valid

        rows = self.rows[all_positions_mask]
        cols = self.cols[all_positions_mask]
        feature_values = {name: features[name][all_positions_mask] for name in self.feature_names}
        train_valid_mask = train_valid[all_positions_mask]
        insitu_valid_mask = insitu_valid[all_positions_mask]

        return rows, cols, feature_values, train_valid_mask, insitu_valid_mask

    def get_all(self) -> tuple:
        dates = get_valid_dates()
        all_rows, all_cols, all_dates = [], [], []
        all_feature_values = {name: [] for name in self.feature_names}
        all_train_valid, all_insitu_valid = [], []

        for date in dates:
            rows, cols, feature_values, train_valid, insitu_valid = self.get(date)

            all_rows.append(rows)
            all_cols.append(cols)
            all_dates.extend([date] * len(rows))
            for name in self.feature_names:
                all_feature_values[name].append(feature_values[name])
            all_train_valid.append(train_valid)
            all_insitu_valid.append(insitu_valid)

        rows = np.concatenate(all_rows)
        cols = np.concatenate(all_cols)
        feature_values = {name: np.concatenate(all_feature_values[name]) for name in self.feature_names}
        train_valid = np.concatenate(all_train_valid)
        insitu_valid = np.concatenate(all_insitu_valid)

        return rows, cols, feature_values, train_valid, insitu_valid, all_dates


T = TypeVar("T")


class BaseDataStore(Generic[T]):

    def __init__(self):
        self._cache: Dict[Hashable, T] = {}

    def _get(self, key: Hashable, loader: Callable[[], T], cache_used: bool = True) -> T:
        if cache_used and key in self._cache:
            return self._cache[key]

        data = loader()

        if cache_used:
            self._cache[key] = data

        return data

    def clear_cache(self) -> None:
        self._cache.clear()


class InsituStatsStore(BaseDataStore[np.ndarray]):

    def __init__(self, resolution: str):
        super().__init__()
        self.resolution = resolution
        self.data_store = DataStore(resolution=self.resolution)

    def get(self, date: str, cache_used: bool = True) -> np.ndarray:
        return self._get(date, lambda: self._load(date), cache_used=cache_used)

    def _load(self, date: str) -> np.ndarray:
        insitus = self.data_store.get(IN_SITU_NAME, date)

        return self._calc_insitu_stats_from_data(insitus)

    @staticmethod
    def _calc_insitu_stats_from_data(insitus: np.ndarray) -> np.ndarray:
        valid_insitu = insitus[~np.isnan(insitus)]
        if len(valid_insitu) > 0:
            insitu_stats = np.array([
                np.mean(valid_insitu),
                np.std(valid_insitu),
                np.percentile(valid_insitu, 25),
                np.percentile(valid_insitu, 75),
            ], dtype=np.float32)
        else:
            insitu_stats = np.zeros(4, dtype=np.float32)
        return insitu_stats


class InferenceResultStore(BaseDataStore[np.ndarray]):

    def __init__(self, resolution: str):
        super().__init__()
        self.resolution = resolution

    def get(self, date: Optional[str] = None, cache_used: bool = True) -> np.ndarray:
        key = (date, self.resolution)
        return self._get(key, lambda: self._load(date), cache_used=cache_used)

    def _load(self, date: str) -> np.ndarray:
        file_path = self._build_path(date)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        data = read_tiff_data(file_path).astype(np.float32)

        return data

    def _build_path(self, date: str) -> str:
        return os.path.join(INFERENCE_DIR_PATH, self.resolution, f"{date}{TIFF_SUFFIX}")


class DataStore(BaseDataStore[np.ndarray]):

    def __init__(self, resolution: str):
        super().__init__()
        self.resolution = resolution

    def get(self, name: str, date: Optional[str] = None, cache_used: bool = True) -> np.ndarray:
        key = (name, date, self.resolution)
        return self._get(key, lambda: self._load(name, date), cache_used=cache_used)

    def _load(self, name: str, date: Optional[str]) -> np.ndarray:
        file_path = self._build_path(name, date)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        data = read_tiff_data(file_path).astype(np.float32)

        return data

    def _build_path(self, name: str, date: Optional[str]) -> str:
        if name == DEM_NAME:
            return os.path.join(PROCESSED_DIR_PATH, name, self.resolution, f"{name}{TIFF_SUFFIX}")
        return os.path.join(PROCESSED_DIR_PATH, name, self.resolution, f"{date}{TIFF_SUFFIX}")


class GridInfoStore(BaseDataStore[Dict]):

    def __init__(self, resolution: str):
        super().__init__()
        self.resolution = resolution

    def get(self, cache_used: bool = True) -> Dict:
        key = self.resolution
        return self._get(key, lambda: self._load(), cache_used=cache_used)

    def _load(self) -> Dict:
        if self.resolution == RESOLUTION_1KM:
            ref_path = REF_GRID_1KM_PATH
        else:
            ref_path = REF_GRID_36KM_PATH

        _, lons, lats = read_tiff(ref_path, dst_epsg_code=4326)
        H, W = lons.shape
        rows, cols = np.meshgrid(np.arange(H, dtype=np.int32),
                                 np.arange(W, dtype=np.int32), indexing="ij")
        grid_info = {
            "lons": lons,
            "lats": lats,
            "rows": rows,
            "cols": cols,
            "H": H,
            "W": W,
            "pos": np.stack([lons, lats], axis=-1).astype(np.float32),
        }
        return grid_info
