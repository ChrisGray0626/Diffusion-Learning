#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Diffusers-based Soil Moisture Downscaling Corrector
@Author Chris
@Date 2025/12/12
"""

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from Constant import *
from Task.DownscaleSM.Dataset import InferenceDataset, CorrectionDataset, GridInfoStore
from Task.DownscaleSM.Module import BiasCorrector
from Util.TiffUtil import write_tiff
from Util.Util import get_valid_dates

BATCH_SIZE = 16384
SM_MIN = 0.02
SM_MAX = 0.5

RESOLUTION = RESOLUTION_1KM


def main():
    grid_info = GridInfoStore(RESOLUTION).get()
    correction_dir_path = os.path.join(CORRECTION_DIR_PATH, RESOLUTION)
    os.makedirs(correction_dir_path, exist_ok=True)

    # Collect Data for Bias Correction Training
    all_pred_ys_insitu = []
    all_insitus = []
    all_aux_feats_insitu = []

    dates = get_valid_dates()
    for date in tqdm(dates, desc="Collecting"):
        inference_dataset = InferenceDataset(date=date, resolution=RESOLUTION)

        # Load inference result from InferenceResultStore
        from Task.DownscaleSM.Dataset import InferenceResultStore
        inference_result_store = InferenceResultStore(resolution=RESOLUTION)
        pred_map = inference_result_store.get(date)
        pred_ys = pred_map[inference_dataset.rows, inference_dataset.cols]

        # Collect in-situ data for training
        insitu_mask = inference_dataset.insitu_mask > 0
        if insitu_mask.sum() > 0:
            all_pred_ys_insitu.append(pred_ys[insitu_mask])
            all_insitus.append(inference_dataset.insitu[insitu_mask])
            all_aux_feats_insitu.append(inference_dataset.xs[insitu_mask])

    # Train Bias Corrector
    all_pred_ys_insitu_concat = np.concatenate(all_pred_ys_insitu)
    all_insitus_concat = np.concatenate(all_insitus)
    all_aux_feats_insitu_concat = np.concatenate(all_aux_feats_insitu)

    rf_corrector = BiasCorrector()
    rf_corrector.train(
        pred_ys=all_pred_ys_insitu_concat,
        insitus=all_insitus_concat,
        aux_feats=all_aux_feats_insitu_concat,
        verbose=True
    )

    # Release Memory
    del all_pred_ys_insitu, all_insitus, all_aux_feats_insitu
    del all_pred_ys_insitu_concat, all_insitus_concat, all_aux_feats_insitu_concat

    # Correction
    for date in tqdm(dates, desc="Correction"):
        correction_dataset = CorrectionDataset(date=date, resolution=RESOLUTION)

        all_pred_ys_corrected = []
        data_loader = DataLoader(correction_dataset, batch_size=min(BATCH_SIZE, len(correction_dataset)), shuffle=False)
        for batch_xs, batch_pred_ys, _, _ in data_loader:
            batch_xs = batch_xs.numpy().astype(np.float32)
            batch_pred_ys = batch_pred_ys.numpy().astype(np.float32)

            batch_pred_ys_corrected = rf_corrector.predict(
                pred_ys=batch_pred_ys,
                aux_feats=batch_xs
            )

            all_pred_ys_corrected.append(batch_pred_ys_corrected)

        pred_ys_corrected = np.concatenate(all_pred_ys_corrected)

        # Clip
        pred_ys_corrected = np.clip(pred_ys_corrected, SM_MIN, SM_MAX)

        # Save Corrected Result
        pred_map = np.full((grid_info["H"], grid_info["W"]), np.nan, dtype=np.float32)
        pred_map[correction_dataset.rows, correction_dataset.cols] = pred_ys_corrected
        dst_file_path = os.path.join(correction_dir_path, f"{date}{TIFF_SUFFIX}")
        write_tiff(pred_map, dst_file_path, transform=grid_info["transform"], crs=grid_info["crs"])


if __name__ == "__main__":
    main()
