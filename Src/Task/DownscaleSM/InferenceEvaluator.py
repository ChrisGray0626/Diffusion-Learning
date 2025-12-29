#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Evaluate Inference Result
@Author Chris
@Date 2025/12/12
"""

import os

import numpy as np

from Constant import RESOLUTION_1KM, RESOLUTION_36KM, REF_GRID_1KM_PATH, REF_GRID_36KM_PATH, \
    INFERENCE_DIR_PATH, TIFF_SUFFIX
from Task.DownscaleSM.Dataset import InsituDataset
from Task.DownscaleSM.Evaluator import Evaluator
from Util.TiffUtil import read_tiff_data, read_tiff
from Util.Util import get_valid_dates

RESOLUTION = RESOLUTION_36KM
if RESOLUTION == RESOLUTION_1KM:
    REF_GRID_PATH = REF_GRID_1KM_PATH
else:
    REF_GRID_PATH = REF_GRID_36KM_PATH


def main():
    print(f"\n{'=' * 60}")
    print(f"Evaluating Inference Results: {RESOLUTION}")
    print("=" * 60)

    # Load inference results
    pred_ys, pred_dates, pred_pos, pred_rows, pred_cols = load_inference_results(RESOLUTION)

    # Load and match insitu data
    insitu_dataset = InsituDataset(resolution=RESOLUTION)
    insitus, insitu_masks = insitu_dataset.get_data_by_date_row_col(pred_dates, pred_rows, pred_cols)

    # Filter to only valid insitu points
    valid_mask = insitu_masks > 0
    pred_ys = pred_ys[valid_mask]
    insitus = insitus[valid_mask]
    insitu_masks = insitu_masks[valid_mask]
    dates = [pred_dates[i] for i in range(len(pred_dates)) if valid_mask[i]]
    rows = pred_rows[valid_mask]
    cols = pred_cols[valid_mask]

    # Initialize evaluator
    evaluator = Evaluator(min_site_num=2, min_date_num=2)

    # Evaluate by Date
    df_result_date = evaluator.evaluate_by_date(
        pred_ys, insitus, insitu_masks, dates, true_ys=None
    )
    print("\n" + "=" * 60)
    print(f"Evaluation by Date: {RESOLUTION} Inference Results vs InSitu Data")
    print("=" * 60)
    evaluator.print_result(df_result_date)

    # Evaluate by Site
    df_result_site = evaluator.evaluate_by_site(
        pred_ys, insitus, insitu_masks, dates, rows, cols, true_ys=None
    )
    print("\n" + "=" * 60)
    print(f"Evaluation by Site: {RESOLUTION} Inference Results vs InSitu Data")
    print("=" * 60)
    evaluator.print_result(df_result_site)

    # Evaluate by Spatial Distribution
    evaluator.evaluate_by_spatial_distribution(df_result_site)


# TODO Inference Result Dataset?
def load_inference_results(resolution: str) -> tuple:
    inference_dir = os.path.join(INFERENCE_DIR_PATH, resolution)
    dates = get_valid_dates()

    if resolution == RESOLUTION_1KM:
        ref_grid_path = REF_GRID_1KM_PATH
    else:
        ref_grid_path = REF_GRID_36KM_PATH

    _, lon_grid, lat_grid = read_tiff(ref_grid_path, dst_epsg_code=4326)
    H, W = lon_grid.shape
    pos_flat = np.stack([lon_grid.flatten(), lat_grid.flatten()], axis=-1)
    rows, cols = np.meshgrid(np.arange(H, dtype=np.int32), np.arange(W, dtype=np.int32), indexing='ij')
    rows_flat, cols_flat = rows.flatten(), cols.flatten()

    all_pred_ys, all_dates, all_pos, all_rows, all_cols = [], [], [], [], []

    for date in dates:
        inference_tiff_path = os.path.join(inference_dir, f"{date}{TIFF_SUFFIX}")
        if not os.path.exists(inference_tiff_path):
            continue

        pred_map = read_tiff_data(inference_tiff_path).astype(np.float32)
        pred_flat = pred_map.flatten()
        valid_mask = ~np.isnan(pred_flat)

        all_pred_ys.append(pred_flat[valid_mask])
        all_dates.extend([date] * valid_mask.sum())
        all_pos.append(pos_flat[valid_mask])
        all_rows.append(rows_flat[valid_mask])
        all_cols.append(cols_flat[valid_mask])

    if len(all_pred_ys) == 0:
        raise ValueError(f"No inference data found for resolution {resolution}")

    return (
        np.concatenate(all_pred_ys),
        all_dates,
        np.concatenate(all_pos),
        np.concatenate(all_rows),
        np.concatenate(all_cols)
    )


if __name__ == "__main__":
    main()
