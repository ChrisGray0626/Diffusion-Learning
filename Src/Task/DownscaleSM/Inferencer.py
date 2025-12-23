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
from Task.DownscaleSM.Dataset import InferenceDataset, TrainDataset
from Task.DownscaleSM.Trainer import NoisePredictor, build_scheduler, reverse_diffuse
from Util.TiffUtil import write_tiff, read_tiff_meta
from Util.Util import get_valid_dates, build_device

# Inference settings
INFERENCE_STEP_NUM = 50
BATCH_SIZE = 16384

# Valid range for soil moisture
SM_MIN = 0.02
SM_MAX = 0.5

RESOLUTION = RESOLUTION_1KM
if RESOLUTION == RESOLUTION_1KM:
    REF_GRID_PATH = REF_GRID_1KM_PATH
else:
    REF_GRID_PATH = REF_GRID_36KM_PATH


def main():
    print(f"Device: {build_device()}")

    train_dataset = TrainDataset()
    x_mean = torch.from_numpy(train_dataset.x_mean).float()
    x_std = torch.from_numpy(train_dataset.x_std).float()
    y_mean = torch.tensor(train_dataset.y_mean, dtype=torch.float32)
    y_std = torch.tensor(train_dataset.y_std, dtype=torch.float32)

    def norm_fn(x: torch.Tensor) -> torch.Tensor:
        return (x - x_mean.to(x.device)) / x_std.to(x.device)

    def denorm_fn(y: torch.Tensor, device: torch.device) -> torch.Tensor:
        return y * y_std.to(device) + y_mean.to(device)

    insitu_stats_dict = train_dataset.insitu_stats_dict
    transform, crs, height, width = read_tiff_meta(REF_GRID_PATH)
    dates = get_valid_dates()
    for date in tqdm(dates):
        insitu_stats = insitu_stats_dict.get(date, None)
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
            insitu_stats=insitu_stats,
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
def inference(dataset: Dataset, height: int, width: int, norm_fn, denorm_fn,
              insitu_stats: np.ndarray | None) -> np.ndarray:
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

        if insitu_stats is not None:
            B = batch_xs.shape[0]
            batch_insitu_stats = torch.from_numpy(insitu_stats).float().to(device)
            batch_insitu_stats = batch_insitu_stats.unsqueeze(0).expand(B, -1)
        else:
            batch_insitu_stats = None

        batch_pred_ys = reverse_diffuse(
            model, scheduler, batch_xs, batch_pos, batch_dates, INFERENCE_STEP_NUM,
            device=device, insitu_stats=batch_insitu_stats  # type: ignore[arg-type]
        )
        batch_pred_ys = batch_pred_ys.reshape(-1)
        # Denormalize
        batch_pred_ys = denorm_fn(batch_pred_ys, device)
        # Clip
        batch_pred_ys = torch.clamp(batch_pred_ys, SM_MIN, SM_MAX)
        batch_pred_ys = batch_pred_ys.cpu().numpy()

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
