#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Evaluate Inference Result
@Author Chris
@Date 2025/12/12
"""

import numpy as np
from tqdm import tqdm

from Constant import RESOLUTION_1KM
from Task.DownscaleSM.Dataset import InsituDataset, InferenceResultDataset
from Task.DownscaleSM.Evaluator import Evaluator
from Util.Util import get_valid_dates

RESOLUTION = RESOLUTION_1KM


def main():
    inference_result_dataset = InferenceResultDataset(resolution=RESOLUTION)
    insitu_dataset = InsituDataset(resolution=RESOLUTION)
    evaluator = Evaluator(min_site_num=2, min_date_num=2)

    dates = get_valid_dates()
    all_pred_ys, all_insitus, all_insitu_masks, all_dates, all_rows, all_cols = [], [], [], [], [], []

    for date in tqdm(dates):
        pred_map, pred_mask, pred_rows, pred_cols = inference_result_dataset.get_data_by_date(date)
        insitu_map, insitu_mask, _, _ = insitu_dataset.get_data_by_date(date)

        valid_mask = (pred_mask > 0) & (insitu_mask > 0)
        if not np.any(valid_mask):
            continue

        all_pred_ys.append(pred_map[valid_mask])
        all_insitus.append(insitu_map[valid_mask])
        all_insitu_masks.append(insitu_mask[valid_mask])
        all_dates.extend([date] * valid_mask.sum())
        all_rows.append(pred_rows[valid_mask])
        all_cols.append(pred_cols[valid_mask])

    pred_ys = np.concatenate(all_pred_ys)
    insitus = np.concatenate(all_insitus)
    insitu_masks = np.concatenate(all_insitu_masks)
    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)

    df_result_date = evaluator.evaluate_by_date(pred_ys, insitus, insitu_masks, all_dates)
    print("\n" + "=" * 60)
    print(f"Evaluation by Date: {RESOLUTION} Inference Results vs InSitu Data")
    print("=" * 60)
    evaluator.print_result(df_result_date)

    df_result_site = evaluator.evaluate_by_site(pred_ys, insitus, insitu_masks, all_dates, rows, cols)
    print("\n" + "=" * 60)
    print(f"Evaluation by Site: {RESOLUTION} Inference Results vs InSitu Data")
    print("=" * 60)
    evaluator.print_result(df_result_site)

    evaluator.evaluate_by_spatial_distribution(df_result_site, height=insitu_dataset.H, width=insitu_dataset.W)


if __name__ == "__main__":
    main()
