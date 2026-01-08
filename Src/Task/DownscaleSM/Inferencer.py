#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Diffusers-based Soil Moisture Downscaling Inferencer
@Author Chris
@Date 2025/12/12
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Constant import *
from Task.DownscaleSM.Dataset import InferenceDataset, GridInfoStore
from Task.DownscaleSM.Module import BiasCorrector
from Task.DownscaleSM.Trainer import NoisePredictor, build_scheduler, reverse_diffuse
from Util.TiffUtil import write_tiff, read_tiff_meta
from Util.Util import get_valid_dates, build_device

INFERENCE_STEP_NUM = 50
BATCH_SIZE = 16384
SM_MIN = 0.02
SM_MAX = 0.5

RESOLUTION = RESOLUTION_36KM
REF_GRID_PATH = REF_GRID_1KM_PATH if RESOLUTION == RESOLUTION_1KM else REF_GRID_36KM_PATH


def main():
    device = build_device()
    print(f"Device: {device}")
    model = build_model()
    rf_corrector = BiasCorrector()
    transform, crs, _, _ = read_tiff_meta(REF_GRID_PATH)
    for date in tqdm(get_valid_dates()):
        dataset = InferenceDataset(date=date, resolution=RESOLUTION)

        # Inference
        pred_ys = inference(model=model, dataset=dataset, device=device)

        # Correction
        pred_ys = rf_corrector.predict(
            pred_ys=pred_ys,
            aux_feats=dataset.xs
        )

        # Clip
        pred_ys = np.clip(pred_ys, SM_MIN, SM_MAX)

        # Convert to map
        grid_info = GridInfoStore(RESOLUTION).get()
        pred_map = np.full((grid_info["H"], grid_info["W"]), np.nan, dtype=np.float32)
        pred_map[dataset.rows, dataset.cols] = pred_ys

        # Write TIFF
        dst_dir_path = os.path.join(INFERENCE_DIR_PATH, RESOLUTION)
        os.makedirs(dst_dir_path, exist_ok=True)
        dst_file_path = os.path.join(dst_dir_path, f"{date}{TIFF_SUFFIX}")
        write_tiff(pred_map, dst_file_path, transform=transform, crs=crs)


def build_model() -> NoisePredictor:
    model_save_path = os.path.join(CHECKPOINT_DIR_PATH, "DownscaleSM/Diffusers")
    model = NoisePredictor.from_pretrained(model_save_path)

    return model


@torch.no_grad()
def inference(model: NoisePredictor, dataset: InferenceDataset, device: str) -> np.ndarray:
    model = model.to(device)
    scheduler = build_scheduler()
    model.eval()

    pred_ys_list = []
    data_loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)), shuffle=False)  # type: ignore[arg-type]
    insitu_stats = torch.from_numpy(dataset.insitu_stats).to(device).unsqueeze(0)
    for batch_xs, batch_pos, batch_dates, batch_rows, batch_cols in tqdm(data_loader):
        batch_xs = batch_xs.to(device)
        batch_pos = batch_pos.to(device)
        B = batch_xs.shape[0]
        batch_insitu_stats = insitu_stats.expand(B, -1)

        batch_pred_ys = reverse_diffuse(model, scheduler, batch_xs, batch_pos, batch_dates, INFERENCE_STEP_NUM,
                                        device=device, insitu_stats=batch_insitu_stats)
        batch_pred_ys = dataset.denorm_y(batch_pred_ys.reshape(-1))
        batch_pred_ys = batch_pred_ys.cpu().numpy()

        pred_ys_list.append(batch_pred_ys)

    pred_ys = np.concatenate(pred_ys_list)
    return pred_ys


if __name__ == "__main__":
    main()
