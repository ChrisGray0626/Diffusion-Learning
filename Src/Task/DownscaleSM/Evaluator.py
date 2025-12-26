#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Evaluator for Soil Moisture Downscaling Results
@Author Chris
@Date 2025/12/12
"""

from typing import List, Optional

import numpy as np
import pandas as pd


class Evaluator:
    def __init__(self, min_insitu_points: int = 2):
        self.min_insitu_points = min_insitu_points

    def evaluate(self, pred_ys: np.ndarray, insitus: np.ndarray,
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
            df.groupby('Date').apply(self._calc_group_metrics, include_groups=False).dropna().tolist()
        )
        df_result = df_result.sort_values('InSitu_Corr_R2', ascending=False, na_position='last')

        return df_result

    def _calc_group_metrics(self, group):
        pred_y = group['PredY'].values
        insitu = group['Insitu'].values
        insitu_mask = group['InsituMask'].values

        valid_insitu_count = insitu_mask.sum()

        if valid_insitu_count < self.min_insitu_points:
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

    @staticmethod
    def print_result(df_results: pd.DataFrame):
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(df_results.to_string(index=False))
