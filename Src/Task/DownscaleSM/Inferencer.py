#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Description Diffusers-based Soil Moisture Downscaling Inferencer
@Author Chris
@Date 2025/12/12
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from Constant import *
from Task.DownscaleSM.Dataset import InferenceDataset, InsituStatsDataset
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

    insitu_stats_dataset = InsituStatsDataset(resolution=RESOLUTION)
    transform, crs, height, width = read_tiff_meta(REF_GRID_PATH)
    dates = get_valid_dates()
    for date in tqdm(dates):
        dataset = InferenceDataset(date=date, resolution=RESOLUTION)
        insitu_stats = insitu_stats_dataset.get_stats(date)

        pred_map = inference(
            dataset=dataset,
            height=height,
            width=width,
            insitu_stats=insitu_stats,
            device=device,
        )

        dst_dir_path = os.path.join(INFERENCE_DIR_PATH, RESOLUTION)
        os.makedirs(dst_dir_path, exist_ok=True)
        dst_file_path = os.path.join(dst_dir_path, f"{date}{TIFF_SUFFIX}")
        write_tiff(pred_map, dst_file_path, transform=transform, crs=crs)


def build_model() -> NoisePredictor:
    model_save_path = os.path.join(CHECKPOINT_DIR_PATH, "DownscaleSM/Diffusers")
    model = NoisePredictor.from_pretrained(model_save_path)

    return model


@torch.no_grad()
def inference(dataset: Dataset, height: int, width: int, insitu_stats: np.ndarray, device: str) -> np.ndarray:
    model = build_model().to(device)
    scheduler = build_scheduler()
    model.eval()

    data_loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)), shuffle=False)  # type: ignore[arg-type]
    pred_map = np.full((height, width), np.nan, dtype=np.float32)
    pred_ys, rows_list, cols_list = [], [], []

    for batch_xs, batch_pos, batch_dates, batch_rows, batch_cols in tqdm(data_loader):
        batch_xs = dataset.norm_x(batch_xs.to(device))
        batch_pos = batch_pos.to(device)
        B = batch_xs.shape[0]
        batch_insitu_stats = torch.from_numpy(insitu_stats).float().to(device).unsqueeze(0).expand(B, -1)

        batch_pred_ys = reverse_diffuse(model, scheduler, batch_xs, batch_pos, batch_dates, INFERENCE_STEP_NUM,
                                        device=device, insitu_stats=batch_insitu_stats)
        batch_pred_ys = dataset.denorm_y(batch_pred_ys.reshape(-1))
        batch_pred_ys = torch.clamp(batch_pred_ys, SM_MIN, SM_MAX).cpu().numpy()

        pred_ys.append(batch_pred_ys)
        rows_list.append(batch_rows.numpy())
        cols_list.append(batch_cols.numpy())

    pred_map[np.concatenate(rows_list), np.concatenate(cols_list)] = np.concatenate(pred_ys)
    return pred_map


if __name__ == "__main__":
    main()
