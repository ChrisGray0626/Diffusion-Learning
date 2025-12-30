#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Diffusers-Based Soil Moisture Downscaling Trainer
  @Author Chris
  @Date 2025/11/12
"""
import os
from typing import List

import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.utils.data import Dataset, DataLoader

from Constant import RANGE, CHECKPOINT_DIR_PATH, REF_GRID_36KM_PATH, RESOLUTION_36KM
from Task.DownscaleSM.Dataset import TrainDataset, InsituStatsDataset, InsituDataset
from Task.DownscaleSM.Evaluator import Evaluator
from Task.DownscaleSM.Module import TimeEmbedding, SpatialEmbedding, FiLMResBlock
from Util.ModelHelper import SinusoidalPosEmb, EarlyStopping
from Util.TiffUtil import pos2grid_index
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

# Inference setting
# Less inference steps for faster sampling
INFERENCE_STEP_NUM = 50

# Train setting
TOTAL_EPOCH = 60
BATCH_SIZE = 64
LR = 2e-4

# Early stopping setting
PATIENCE = 5
MIN_DELTA = 1e-6

# Valid range for soil moisture
SM_MIN = 0.02
SM_MAX = 0.5

# Control Flag
TRAINING = True
TEST = True


def main():
    device = build_device()
    scheduler = build_scheduler()
    model_save_path = os.path.join(CHECKPOINT_DIR_PATH, "DownscaleSM", "Diffusers")

    dataset = TrainDataset()
    insitu_stats_dataset = InsituStatsDataset()
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Training Phase
    if TRAINING:
        print("\n" + "=" * 60)
        print(f"Training ...")
        print("=" * 60)
        early_stopping = build_early_stopping()
        model = NoisePredictor(
            input_feature_num=INPUT_FEATURE_NUM,
            hidden_dim=HIDDEN_DIM,
            timestep_emb_dim=TIMESTEP_EMB_DIM,
            res_block_num=3,
        ).to(device)
        trainer = Trainer(model, scheduler, train_dataset, val_dataset, device=device, early_stopping=early_stopping,
                          insitu_stats_dataset=insitu_stats_dataset,
                          total_epoch=TOTAL_EPOCH, batch_size=BATCH_SIZE, lr=LR)
        model = trainer.run()
        model.save_pretrained(model_save_path)

    # Testing Phase
    if TEST:
        print("\n" + "=" * 60)
        print(f"Testing ...")
        print("=" * 60)
        model = NoisePredictor.from_pretrained(model_save_path).to(device)
        test(model, scheduler, test_dataset, insitu_stats_dataset, device)


class NoisePredictor(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self, input_feature_num: int,
                 hidden_dim: int = 512, timestep_emb_dim: int = 128,
                 res_block_num: int = 3):
        super().__init__()
        self.input_feature_num = input_feature_num
        self.hidden_dim = hidden_dim

        input_dim = input_feature_num + 1
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # Timestep Embedding
        self.timestep_embedding = nn.Sequential(
            SinusoidalPosEmb(timestep_emb_dim),
            nn.Linear(timestep_emb_dim, timestep_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(timestep_emb_dim * 4, timestep_emb_dim * 4),
        )
        self.emb_timestep2hidden = nn.Linear(timestep_emb_dim * 4, hidden_dim)

        # Time Embedding
        self.time_embedding = TimeEmbedding(
            hidden_dim=hidden_dim,
            num_fourier=8
        )

        # Spatial Embedding
        lon_min, lat_min, lon_max, lat_max = RANGE
        self.spatial_embedding = SpatialEmbedding(
            hidden_dim=hidden_dim,
            num_fourier=6,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max
        )

        # InSitu (站点) 全局统计特征嵌入（同一日期的全局信息）
        self.insitu_stats_embedding = nn.Sequential(
            nn.Linear(4, hidden_dim // 2),  # 8个统计特征 -> hidden_dim//2
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Condition Fusion (now includes global InSitu embedding only)
        self.condition_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),  # timestep + time + spatial + insitu_global
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Residual Block
        self.res_blocks = nn.ModuleList([
            FiLMResBlock(hidden_dim=hidden_dim)
            for _ in range(res_block_num)
        ])

        # Output Layer
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, diffused_ys: torch.Tensor, xs: torch.Tensor, timesteps: torch.Tensor,
                pos: torch.Tensor, dates: List[str],
                insitu_stats: torch.Tensor = None) -> torch.Tensor:
        inputs = torch.cat([xs, diffused_ys], dim=1)
        x = self.input_layer(inputs)

        # Embed Timestep
        embed_timesteps = self.timestep_embedding(timesteps)
        embed_timesteps = self.emb_timestep2hidden(embed_timesteps)

        # Embed Time
        embed_time = self.time_embedding(dates)

        # Embed Spatial
        embed_spatial = self.spatial_embedding(pos)

        # Embed InSitu Stats
        if insitu_stats is not None:
            embed_insitu_stats = self.insitu_stats_embedding(insitu_stats)
        else:
            B = x.shape[0]
            embed_insitu_stats = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)

        # Fuse Condition (now includes global InSitu only)
        condition = torch.cat([embed_timesteps, embed_time, embed_spatial, embed_insitu_stats], dim=1)
        condition = self.condition_fusion(condition)

        # Residual Blocks with FiLM
        for res_block in self.res_blocks:
            x = res_block(x, condition)

        out = self.output_layer(x)

        return out


class Trainer:

    def __init__(self, model: NoisePredictor, scheduler: DDPMScheduler,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 device: str,
                 early_stopping: EarlyStopping,
                 insitu_stats_dataset: InsituStatsDataset,
                 total_epoch: int,
                 batch_size: int, lr: float):
        self.model = model
        self.scheduler = scheduler
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.early_stopping = early_stopping
        self.insitu_stats_dataset = insitu_stats_dataset
        self.total_epoch = total_epoch
        self.batch_size = batch_size
        self.lr = lr

    def evaluate_epoch(self, data_loader: DataLoader, optimizer: torch.optim.Optimizer = None) -> float:
        total_loss = 0.0
        total_samples = 0

        for batch_xs, batch_ys, batch_pos, batch_dates in data_loader:
            batch_xs = batch_xs.to(self.device)
            batch_ys = batch_ys.to(self.device).unsqueeze(1)
            batch_pos = batch_pos.to(self.device)

            batch_insitu_stats = torch.stack([
                torch.from_numpy(self.insitu_stats_dataset.get_stats(date)).float()
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

            diffusion_loss = nn.functional.mse_loss(pred_noise, noises)
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
                    insitu_stats: torch.Tensor = None) -> torch.Tensor:
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


def test(model: NoisePredictor, scheduler: DDPMScheduler, dataset: Dataset,
         insitu_stats_dataset: InsituStatsDataset, device: str):
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)  # type: ignore[arg-type]
    xs, true_ys, pos, dates = next(iter(data_loader))
    xs = xs.to(device)
    pos = pos.to(device)
    insitu_stats = torch.stack([
        torch.from_numpy(insitu_stats_dataset.get_stats(date)).float()
        for date in dates
    ]).to(device)

    # Inference
    with torch.no_grad():
        pred_ys = reverse_diffuse(
            model, scheduler, xs, pos, dates, INFERENCE_STEP_NUM, device=device,
            insitu_stats=insitu_stats
        )

    # Denormalize & Clip
    pred_ys = dataset.dataset.denorm_y(pred_ys)
    pred_ys = torch.clamp(pred_ys, SM_MIN, SM_MAX)
    pred_ys = pred_ys.cpu().numpy()
    true_ys = dataset.dataset.denorm_y(true_ys)
    true_ys = true_ys.numpy()

    # Evaluate
    insitu_dataset = InsituDataset(resolution=RESOLUTION_36KM)
    rows, cols = pos2grid_index(pos.cpu().numpy(), REF_GRID_36KM_PATH)
    insitus, insitu_masks = insitu_dataset.get_data_by_date_row_col(dates, rows, cols)
    evaluator = Evaluator(min_site_num=2, min_date_num=2)

    # Evaluate by Date
    df_result_date = evaluator.evaluate_by_date(pred_ys, insitus, insitu_masks, dates, true_ys=true_ys)
    print("\n" + "=" * 60)
    print("Evaluation by Date: Test Results vs InSitu Data")
    print("=" * 60)
    evaluator.print_result(df_result_date)

    # Evaluate by Site
    df_result_site = evaluator.evaluate_by_site(pred_ys, insitus, insitu_masks, dates, rows, cols, true_ys=true_ys)
    print("\n" + "=" * 60)
    print("Evaluation by Site: Test Results vs InSitu Data")
    print("=" * 60)
    evaluator.print_result(df_result_site)

    # Evaluate by Spatial Distribution
    evaluator.evaluate_by_spatial_distribution(df_result_site)


if __name__ == "__main__":
    main()
