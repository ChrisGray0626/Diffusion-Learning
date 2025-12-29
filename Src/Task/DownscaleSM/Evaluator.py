#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Evaluator for Soil Moisture Downscaling Results
@Author Chris
@Date 2025/12/12
"""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Evaluator:
    def __init__(self, min_site_num: int = 2, min_date_num: int = 2):
        self.min_site_num = min_site_num
        self.min_date_num = min_date_num

    def evaluate_by_date(self, pred_ys: np.ndarray, insitus: np.ndarray,
                         insitu_masks: np.ndarray, dates: List[str],
                         true_ys: Optional[np.ndarray] = None) -> pd.DataFrame:
        if pred_ys.ndim == 2:
            pred_ys = pred_ys.flatten()
        if insitus.ndim == 2:
            insitus = insitus.flatten()
        if insitu_masks.ndim == 2:
            insitu_masks = insitu_masks.flatten()
        if true_ys is not None and true_ys.ndim == 2:
            true_ys = true_ys.flatten()

        data_dict = {
            'Date': dates,
            'PredY': pred_ys,
            'Insitu': insitus,
            'InsituMask': insitu_masks
        }
        if true_ys is not None:
            data_dict['TrueY'] = true_ys

        df = pd.DataFrame(data_dict)
        df_result = pd.DataFrame(
            df.groupby('Date').apply(self._calc_metrics_by_date, include_groups=False).dropna().tolist())
        df_result = df_result.sort_values('InSitu_Corr_R2', ascending=False, na_position='last')

        return df_result

    def _calc_metrics_by_date(self, group):
        pred_y = group['PredY'].values
        insitu = group['Insitu'].values
        insitu_mask = group['InsituMask'].values

        valid_insitu_count = insitu_mask.sum()

        if valid_insitu_count < self.min_site_num:
            return None

        result = {
            'Date': group.name,
            'Total_Points': len(group),
            'Valid_InSitu_Points': int(valid_insitu_count)
        }

        if 'TrueY' in group.columns:
            true_y = group['TrueY'].values
            true_y_metrics = self._calc_metrics(pred_y, true_y)
            result.update({f'TrueY_{k}': v for k, v in true_y_metrics.items()})

        insitu_orig_metrics = self._calc_metrics(pred_y, insitu, insitu_mask)
        result.update({f'InSitu_Orig_{k}': v for k, v in insitu_orig_metrics.items()})

        insitu_valid = insitu[insitu_mask > 0]
        systematic_bias = np.mean(pred_y[insitu_mask > 0] - insitu_valid)
        pred_corrected = pred_y - systematic_bias
        insitu_corr_metrics = self._calc_metrics(pred_corrected, insitu, insitu_mask)
        result.update({f'InSitu_Corr_{k}': v for k, v in insitu_corr_metrics.items()})

        return result

    def evaluate_by_site(self, pred_ys: np.ndarray, insitus: np.ndarray,
                         insitu_masks: np.ndarray, dates: List[str],
                         rows: np.ndarray, cols: np.ndarray,
                         true_ys: Optional[np.ndarray] = None) -> pd.DataFrame:
        if pred_ys.ndim == 2:
            pred_ys = pred_ys.flatten()
        if insitus.ndim == 2:
            insitus = insitus.flatten()
        if insitu_masks.ndim == 2:
            insitu_masks = insitu_masks.flatten()
        if true_ys is not None and true_ys.ndim == 2:
            true_ys = true_ys.flatten()
        if rows.ndim > 1:
            rows = rows.flatten()
        if cols.ndim > 1:
            cols = cols.flatten()

        data_dict = {
            'Row': rows,
            'Col': cols,
            'Date': dates,
            'PredY': pred_ys,
            'Insitu': insitus,
            'InsituMask': insitu_masks
        }
        if true_ys is not None:
            data_dict['TrueY'] = true_ys

        df = pd.DataFrame(data_dict)
        df_with_insitu = df[df['InsituMask'] > 0].copy()

        results = df_with_insitu.groupby(['Row', 'Col']).apply(self._calc_metrics_by_site,
                                                               include_groups=False).dropna()
        df_result = pd.DataFrame(list(results))
        df_result = df_result.sort_values('InSitu_Corr_R2', ascending=False, na_position='last')

        return df_result

    def _calc_metrics_by_site(self, group):
        pred_y = group['PredY'].values
        insitu = group['Insitu'].values
        insitu_mask = group['InsituMask'].values

        valid_dates_count = len(group)

        if valid_dates_count < self.min_date_num:
            return None

        row, col = group.name
        result = {
            'Row': int(row),
            'Col': int(col),
            'Valid_InSitu_Dates': valid_dates_count
        }

        if 'TrueY' in group.columns:
            true_y = group['TrueY'].values
            true_y_metrics = self._calc_metrics(pred_y, true_y)
            result.update({f'TrueY_{k}': v for k, v in true_y_metrics.items()})

        insitu_orig_metrics = self._calc_metrics(pred_y, insitu, insitu_mask)
        result.update({f'InSitu_Orig_{k}': v for k, v in insitu_orig_metrics.items()})

        insitu_valid = insitu[insitu_mask > 0]
        systematic_bias = np.mean(pred_y[insitu_mask > 0] - insitu_valid)
        pred_corrected = pred_y - systematic_bias
        insitu_corr_metrics = self._calc_metrics(pred_corrected, insitu, insitu_mask)
        result.update({f'InSitu_Corr_{k}': v for k, v in insitu_corr_metrics.items()})

        return result

    @staticmethod
    def _calc_metrics(pred, true, mask=None):
        if mask is not None:
            pred = pred[mask > 0]
            true = true[mask > 0]
        mse = np.mean((pred - true) ** 2)
        bias = np.mean(pred - true)
        ubrmse = np.sqrt(np.clip(mse - bias ** 2, 0, None))
        ss_res = np.sum((true - pred) ** 2)
        ss_tot = np.sum((true - np.mean(true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-8 else (1.0 if ss_res < 1e-8 else np.nan)
        return {'ubRMSE': ubrmse, 'Bias': bias, 'R2': r2 if np.isfinite(r2) else np.nan}

    def evaluate_by_spatial_distribution(self, df_site_results: pd.DataFrame, figsize: tuple = (16, 6)):
        rows = df_site_results['Row'].values
        cols = df_site_results['Col'].values
        error_values = df_site_results['InSitu_Corr_ubRMSE'].values
        r2_values = df_site_results['InSitu_Corr_R2'].values

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot ubRMSE
        error_grid = self._build_metric_grid(error_values, rows, cols)
        im1 = ax1.imshow(error_grid, cmap='YlOrRd', aspect='auto', origin='upper')
        ax1.set_title('ubRMSE (Pred vs InSitu)')
        ax1.set_xlabel('Column Index')
        ax1.set_ylabel('Row Index')
        plt.colorbar(im1, ax=ax1, label='ubRMSE')

        # Plot R²
        r2_grid = self._build_metric_grid(r2_values, rows, cols)
        im2 = ax2.imshow(r2_grid, cmap='cividis', aspect='auto', origin='upper', vmin=0, vmax=1)
        ax2.set_title('R² Score (Pred vs InSitu)')
        ax2.set_xlabel('Column Index')
        ax2.set_ylabel('Row Index')
        plt.colorbar(im2, ax=ax2, label='R²')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def _build_metric_grid(metric_values: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        max_row, max_col = rows.max(), cols.max()
        metric_grid = np.full((max_row + 1, max_col + 1), np.nan, dtype=np.float32)
        for i in range(len(rows)):
            metric_grid[int(rows[i]), int(cols[i])] = metric_values[i]
        return metric_grid

    @staticmethod
    def print_result(df_results: pd.DataFrame):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(df_results.to_string(index=False))
