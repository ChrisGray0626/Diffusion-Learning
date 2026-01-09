#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Diffusers-Based Soil Moisture Downscaling Trainer
  @Author Chris
  @Date 2025/11/12
"""
from typing import List

import torch
import torch.nn.functional as F
from diffusers import DDPMScheduler
from torch.utils.data import Dataset, DataLoader

from Constant import *
from Task.DownscaleSM.Dataset import TrainDataset, InsituStatsStore
from Task.DownscaleSM.Module import NoisePredictor
from Util.ModelHelper import EarlyStopping
from Util.Util import build_device

# Dataset setting
INPUT_FEATURE_NUM = 5

# Diffusion setting
STEP_TOTAL_NUM = 1000
BETA_START = 1e-4
BETA_END = 0.02

# Model setting
HIDDEN_DIM = 512
TIMESTEP_EMB_DIM = 128

# Train setting
TOTAL_EPOCH = 60
BATCH_SIZE = 64
LR = 2e-4

# Early stopping setting
PATIENCE = 5
MIN_DELTA = 1e-6


def main():
    dataset = TrainDataset()
    insitu_stats_store = InsituStatsStore(resolution=RESOLUTION_36KM)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    model = NoisePredictor(
        input_feature_num=INPUT_FEATURE_NUM,
        hidden_dim=HIDDEN_DIM,
        timestep_emb_dim=TIMESTEP_EMB_DIM,
        res_block_num=3,
    )

    trainer = Trainer(model, train_dataset, val_dataset, insitu_stats_store)
    model = trainer.run()

    model_save_path = os.path.join(CHECKPOINT_DIR_PATH, "DownscaleSM", "Diffusers")
    model.save_pretrained(model_save_path)


class Trainer:

    def __init__(self, model: NoisePredictor,
                 train_dataset: Dataset, val_dataset: Dataset, insitu_stats_store: InsituStatsStore):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.insitu_stats_store = insitu_stats_store

        self.scheduler = build_scheduler()
        self.total_epoch = TOTAL_EPOCH
        self.batch_size = BATCH_SIZE
        self.lr = LR
        self.device = build_device()
        self.early_stopping = build_early_stopping()

        self.model = model.to(self.device)

    def evaluate_epoch(self, data_loader: DataLoader, optimizer: torch.optim.Optimizer = None) -> float:
        total_loss = 0.0
        total_samples = 0

        for batch_xs, batch_ys, batch_pos, batch_dates in data_loader:
            batch_xs = batch_xs.to(self.device)
            batch_ys = batch_ys.to(self.device).unsqueeze(1)
            batch_pos = batch_pos.to(self.device)

            batch_insitu_stats = torch.stack([
                torch.from_numpy(self.insitu_stats_store.get(date)).float()
                for date in batch_dates
            ]).to(self.device)

            B = batch_xs.shape[0]

            sampled_timesteps = torch.randint(
                0, self.scheduler.config['num_train_timesteps'],
                (B,), device=self.device, dtype=torch.long
            )

            noises = torch.randn_like(batch_ys)
            diffused_ys = self.scheduler.add_noise(
                original_samples=batch_ys,
                noise=noises,
                timesteps=sampled_timesteps  # type: ignore[arg-type]
            )

            pred_noise = self.model.forward(
                diffused_ys, batch_xs, sampled_timesteps,
                pos=batch_pos, dates=batch_dates,
                insitu_stats=batch_insitu_stats
            )

            diffusion_loss = F.mse_loss(pred_noise, noises)
            loss = diffusion_loss

            total_loss += loss.item() * B
            total_samples += B

            # If train phrase, perform optimization step
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        avg_loss = total_loss / max(total_samples, 1)

        return avg_loss

    def run(self):
        # Optimizer
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # Learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.total_epoch, eta_min=1e-6
        )

        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        for epoch in range(self.total_epoch):
            # Training phase
            self.model.train()
            train_loss = self.evaluate_epoch(train_loader, optimizer=opt)
            current_lr = opt.param_groups[0]['lr']

            # Validation phase
            self.model.eval()
            val_loss = self.evaluate_epoch(val_loader, optimizer=None)

            print(
                f"Epoch {epoch + 1}/{self.total_epoch} Train Loss: {train_loss:.6f} Val Loss: {val_loss:.6f} LR: {current_lr:.6f}")

            # Early stopping check
            if self.early_stopping(val_loss, self.model):
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. Best val loss: {self.early_stopping.best_loss:.6f}")
                break

            # Update learning rate
            lr_scheduler.step()

        return self.model


def build_scheduler() -> DDPMScheduler:
    return DDPMScheduler(
        num_train_timesteps=STEP_TOTAL_NUM,
        beta_start=BETA_START,
        beta_end=BETA_END,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=False,
    )


def build_early_stopping() -> EarlyStopping:
    return EarlyStopping(
        patience=PATIENCE,
        min_delta=MIN_DELTA,
        restore_best_weights=True
    )


@torch.no_grad()
def reverse_diffuse(model: NoisePredictor, scheduler: DDPMScheduler,
                    xs: torch.Tensor, pos: torch.Tensor, dates: List[str],
                    inference_step_num: int, device: str,
                    insitu_stats: torch.Tensor) -> torch.Tensor:
    model.eval()
    B = xs.shape[0]

    ys = torch.randn(B, 1, device=device)
    scheduler.set_timesteps(inference_step_num)

    for timestep in scheduler.timesteps:
        timesteps = torch.full((B,), timestep.item(), device=device, dtype=torch.long)
        insitu_stats = insitu_stats.to(device)
        pred_noises = model.forward(ys, xs, timesteps, pos=pos, dates=dates,
                                    insitu_stats=insitu_stats)
        step_out = scheduler.step(model_output=pred_noises, timestep=timestep, sample=ys)
        ys = step_out.prev_sample

    return ys


if __name__ == "__main__":
    main()
