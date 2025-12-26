#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Evaluate Inference Results
@Author Chris
@Date 2025/12/12
"""

import os

import numpy as np

from Constant import INFERENCE_DIR_PATH, RESOLUTION_1KM, REF_GRID_1KM_PATH, REF_GRID_36KM_PATH, \
    TIFF_SUFFIX, PROCESSED_DIR_PATH, IN_SITU_NAME
from Task.DownscaleSM.Evaluator import Evaluator
from Util.TiffUtil import read_tiff_data
from Util.Util import get_valid_dates

RESOLUTION = RESOLUTION_1KM
if RESOLUTION == RESOLUTION_1KM:
    REF_GRID_PATH = REF_GRID_1KM_PATH
else:
    REF_GRID_PATH = REF_GRID_36KM_PATH

INFERENCE_DIR_PATH = os.path.join(INFERENCE_DIR_PATH, RESOLUTION)


def main():
    dates = get_valid_dates()

    all_inference_ys = []
    all_insitus = []
    all_insitu_masks = []
    all_dates = []

    for date in dates:
        inference_tiff_path = os.path.join(INFERENCE_DIR_PATH, f"{date}{TIFF_SUFFIX}")
        if not os.path.exists(inference_tiff_path):
            continue

        inference_map = read_tiff_data(inference_tiff_path).astype(np.float32)
        insitu_path = os.path.join(PROCESSED_DIR_PATH, IN_SITU_NAME, RESOLUTION, f'{date}{TIFF_SUFFIX}')
        insitu_map = read_tiff_data(insitu_path).astype(np.float32)

        inference_map = inference_map.flatten()
        insitu_map = insitu_map.flatten()
        insitu_masks = (~np.isnan(insitu_map)).astype(np.float32)

        all_inference_ys.append(inference_map)
        all_insitus.append(insitu_map)
        all_insitu_masks.append(insitu_masks)
        all_dates.extend([date] * len(inference_map))

    inference_ys = np.concatenate(all_inference_ys)
    insitus = np.concatenate(all_insitus)
    insitu_masks = np.concatenate(all_insitu_masks)

    evaluator = Evaluator(min_insitu_points=2)
    df_result = evaluator.evaluate(inference_ys, insitus, insitu_masks, all_dates, true_ys=None)

    print("\n" + "=" * 60)
    print("Evaluation: Inference Results vs InSitu Data")
    print("=" * 60)
    evaluator.print_result(df_result)


if __name__ == "__main__":
    main()
