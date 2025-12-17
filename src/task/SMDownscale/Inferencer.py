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
from task.SMDownscale.Dataset import InferenceDataset, TrainDataset
from task.SMDownscale.Trainer import NoisePredictor, build_scheduler, reverse_diffuse
from util.TiffUtil import write_tiff, read_tiff_meta
from util.Util import get_valid_dates, build_device

# Inference settings
INFERENCE_STEP_NUM = 50
BATCH_SIZE = 8192

RESOLUTION = RESOLUTION_1KM
if RESOLUTION == RESOLUTION_1KM:
    REF_GRID_PATH = REF_GRID_1KM_PATH
else:
    REF_GRID_PATH = REF_GRID_36KM_PATH


def main():
    train_dataset = TrainDataset()
    denorm_fn = train_dataset.denormalize_y
    x_mean = torch.from_numpy(train_dataset.x_mean).float()
    x_std = torch.from_numpy(train_dataset.x_std).float()

    def norm_fn(x: torch.Tensor) -> torch.Tensor:
        return (x - x_mean.to(x.device)) / x_std.to(x.device)

    transform, crs, height, width = read_tiff_meta(REF_GRID_PATH)
    dates = get_valid_dates()
    for date in tqdm(dates):
        dataset = InferenceDataset(
            date=date,
            resolution=RESOLUTION,
        )
        pred_map = inference(
            dataset=dataset,
            height=height,
            width=width,
            norm_fn=norm_fn,
            denorm_fn=denorm_fn,
        )

        dst_dir_path = os.path.join(INFERENCE_DIR_PATH, RESOLUTION)
        os.makedirs(dst_dir_path, exist_ok=True)
        dst_file_path = os.path.join(dst_dir_path, f"{date}{TIFF_SUFFIX}")
        write_tiff(pred_map, dst_file_path, transform=transform, crs=crs)


def build_model() -> NoisePredictor:
    model_save_path = os.path.join(PROJ_PATH, "Checkpoint/SMDownscale/Diffusers")
    model = NoisePredictor.from_pretrained(model_save_path)

    return model


@torch.no_grad()
def inference(dataset: Dataset, height: int, width: int, norm_fn, denorm_fn) -> np.ndarray:
    device = build_device()
    model = build_model().to(device)
    scheduler = build_scheduler()
    model.eval()

    data_loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)), shuffle=False)  # type: ignore[arg-type]

    pred_map = np.full((height, width), np.nan, dtype=np.float32)
    pred_ys = []
    grid_indices = []
    for batch_xs, batch_pos, batch_dates, batch_grid_indices in tqdm(data_loader):
        batch_xs = norm_fn(batch_xs.to(device))
        batch_pos = batch_pos.to(device)

        batch_pred_ys = reverse_diffuse(
            model, scheduler, batch_xs, batch_pos, batch_dates, INFERENCE_STEP_NUM, device=device
        )
        batch_pred_ys = batch_pred_ys.cpu().numpy()
        batch_pred_ys = denorm_fn(batch_pred_ys).reshape(-1)

        pred_ys.append(batch_pred_ys)
        grid_indices.append(batch_grid_indices)

    pred_ys = np.concatenate(pred_ys)
    grid_indices = np.concatenate(grid_indices)

    # Convert H, W index to Map
    h_indices = grid_indices // width
    w_indices = grid_indices % width
    pred_map[h_indices, w_indices] = pred_ys

    return pred_map


if __name__ == "__main__":
    main()
