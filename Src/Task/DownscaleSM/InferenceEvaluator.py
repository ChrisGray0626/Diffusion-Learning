#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Evaluate Inference Result
@Author Chris
@Date 2025/12/12
"""
import os

from Constant import RESULT_DIR_PATH, RESOLUTION_36KM
from Task.DownscaleSM.Dataset import InferenceEvaluationDataset
from Task.DownscaleSM.Evaluator import Evaluator

RESOLUTION = RESOLUTION_36KM


def main():
    dataset = InferenceEvaluationDataset(resolution=RESOLUTION)
    evaluator = Evaluator(min_site_num=2, min_date_num=2)

    pred_map, insitu_map, insitu_masks, dates, rows, cols = dataset.get_all()
    # Evaluate by Date
    print("\n" + "=" * 60)
    print(f"Evaluation by Date: {RESOLUTION} Inference Results vs InSitu Data")
    print("=" * 60)
    df_result_date = evaluator.evaluate_by_date(pred_map, insitu_map, insitu_masks, dates)
    dst_file_path = os.path.join(RESULT_DIR_PATH, f"Evaluation_By_Date_{RESOLUTION}.csv")
    df_result_date.to_csv(dst_file_path, index=False)

    # Evaluate by Site
    print("\n" + "=" * 60)
    print(f"Evaluation by Site: {RESOLUTION} Inference Results vs InSitu Data")
    print("=" * 60)
    df_result_site = evaluator.evaluate_by_site(pred_map, insitu_map, insitu_masks, dates, rows, cols)
    dst_file_path = os.path.join(RESULT_DIR_PATH, f"Evaluation_By_Site_{RESOLUTION}.csv")
    df_result_site.to_csv(dst_file_path, index=False)

    # Evaluate by Spatial Distribution
    grid_info = dataset.grid_info_store.get()
    evaluator.evaluate_by_spatial_distribution(df_result_site, height=grid_info['H'], width=grid_info['W'])


if __name__ == "__main__":
    main()
