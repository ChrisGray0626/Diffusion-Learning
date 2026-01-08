#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Diffusers-Based Soil Moisture Downscaling Trainer
  @Author Chris
  @Date 2025/11/12
"""
from typing import List

import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.utils.data import Dataset, DataLoader

from Constant import *
from Task.DownscaleSM.Dataset import TrainDataset, InsituStatsStore, InsituDataset, GridInfoStore
from Task.DownscaleSM.Evaluator import Evaluator
from Task.DownscaleSM.Module import TimeEmbedding, SpatialEmbedding, InsituStatsEmbedding, FiLMResBlock, \
    BiasCorrector
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
CORRECT_TRAINING = True


def main():
    device = build_device()
    scheduler = build_scheduler()
    model_save_path = os.path.join(CHECKPOINT_DIR_PATH, "DownscaleSM", "Diffusers")

    dataset = TrainDataset()
    insitu_stats_store = InsituStatsStore(resolution=RESOLUTION_36KM)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
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
                          insitu_stats_store=insitu_stats_store,
                          total_epoch=TOTAL_EPOCH, batch_size=BATCH_SIZE, lr=LR)
        model = trainer.run()
        model.save_pretrained(model_save_path)

    model = NoisePredictor.from_pretrained(model_save_path).to(device)

    # Correction Phase
    if CORRECT_TRAINING:
        print("\n" + "=" * 60)
        print(f"Correcting Training Data ...")
        print("=" * 60)
        correct_train(model, scheduler, dataset, insitu_stats_store, device)

    # Testing Phase
    print("\n" + "=" * 60)
    print(f"Testing ...")
    print("=" * 60)
    test(model, scheduler, dataset, insitu_stats_store, device)


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

        # Insitu Stats Embedding
        self.insitu_stats_embedding = InsituStatsEmbedding(
            hidden_dim=hidden_dim,
            stats_dim=4
        )

        # Condition Fusion
        self.condition_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
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
                insitu_stats: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([xs, diffused_ys], dim=1)
        x = self.input_layer(inputs)

        # Embed Timestep
        embed_timesteps = self.timestep_embedding(timesteps)
        embed_timesteps = self.emb_timestep2hidden(embed_timesteps)

        # Embed Time
        embed_time = self.time_embedding(dates)

        # Embed Spatial
        embed_spatial = self.spatial_embedding(pos)

        # Embed Insitu Stats
        embed_insitu_stats = self.insitu_stats_embedding(insitu_stats)

        # Fuse Condition
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
                 insitu_stats_store: InsituStatsStore,
                 total_epoch: int,
                 batch_size: int, lr: float):
        self.model = model
        self.scheduler = scheduler
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.early_stopping = early_stopping
        self.insitu_stats_store = insitu_stats_store
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


def correct_train(model: NoisePredictor, scheduler: DDPMScheduler, dataset: TrainDataset,
                  insitu_stats_store: InsituStatsStore, device: str):
    CORRECT_BATCH_SIZE = 10000
    data_loader = DataLoader(dataset, batch_size=CORRECT_BATCH_SIZE, shuffle=False)

    all_pred_ys, all_pos, all_dates, all_xs = [], [], [], []
    with torch.no_grad():
        for batch_xs, batch_true_ys, batch_pos, batch_dates in data_loader:
            batch_xs = batch_xs.to(device)
            batch_pos = batch_pos.to(device)
            batch_insitu_stats = torch.stack([
                torch.from_numpy(insitu_stats_store.get(date)).float()
                for date in batch_dates
            ]).to(device)

            batch_pred_ys = reverse_diffuse(
                model, scheduler, batch_xs, batch_pos, list(batch_dates), INFERENCE_STEP_NUM, device=device,
                insitu_stats=batch_insitu_stats
            )

            all_pred_ys.append(batch_pred_ys)
            all_pos.append(batch_pos)
            all_dates.extend(list(batch_dates))
            all_xs.append(batch_xs)

    pred_ys = torch.cat(all_pred_ys, dim=0)
    pos = torch.cat(all_pos, dim=0)
    xs = torch.cat(all_xs, dim=0)

    pred_ys = dataset.denorm_y(pred_ys)

    pred_ys = pred_ys.cpu().numpy()
    pos = pos.cpu().numpy()
    xs = xs.cpu().numpy()

    # Get InSitu Data
    rows, cols = pos2grid_index(pos, REF_GRID_36KM_PATH)
    insitu_dataset = InsituDataset(resolution=RESOLUTION_36KM)
    insitus, insitu_masks = insitu_dataset.get_data_by_date_row_col(all_dates, rows.flatten(), cols.flatten())

    # RF Bias Corrector
    rf_corrector = BiasCorrector()
    rf_corrector.train(
        pred_ys=pred_ys,
        insitus=insitus.flatten(),
        insitu_masks=insitu_masks.flatten(),
        aux_feats=xs
    )


def test(model: NoisePredictor, scheduler: DDPMScheduler, train_dataset: TrainDataset,
         insitu_stats_store: InsituStatsStore, device: str):
    TEST_BATCH_SIZE = 10000
    data_loader = DataLoader(train_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

    all_pred_ys, all_true_ys, all_pos, all_dates, all_xs = [], [], [], [], []
    with torch.no_grad():
        for batch_xs, batch_true_ys, batch_pos, batch_dates in data_loader:
            batch_xs = batch_xs.to(device)
            batch_pos = batch_pos.to(device)
            batch_insitu_stats = torch.stack([
                torch.from_numpy(insitu_stats_store.get(date)).float()
                for date in batch_dates
            ]).to(device)

            batch_pred_ys = reverse_diffuse(
                model, scheduler, batch_xs, batch_pos, list(batch_dates), INFERENCE_STEP_NUM, device=device,
                insitu_stats=batch_insitu_stats
            )

            all_pred_ys.append(batch_pred_ys)
            all_true_ys.append(batch_true_ys.to(device))
            all_pos.append(batch_pos)
            all_dates.extend(list(batch_dates))
            all_xs.append(batch_xs)

    # Concatenate all batches
    pred_ys = torch.cat(all_pred_ys, dim=0)
    true_ys = torch.cat(all_true_ys, dim=0)
    pos = torch.cat(all_pos, dim=0)
    xs = torch.cat(all_xs, dim=0)

    # Denormalize
    pred_ys = train_dataset.denorm_y(pred_ys)
    pred_ys = pred_ys.cpu().numpy()
    true_ys = train_dataset.denorm_y(true_ys)
    true_ys = true_ys.cpu().numpy()
    pos = pos.cpu().numpy()
    xs = xs.cpu().numpy()

    # Get InSitu Data
    grid_info_store = GridInfoStore(resolution=RESOLUTION_36KM)
    rows, cols = pos2grid_index(pos, REF_GRID_36KM_PATH)
    insitu_dataset = InsituDataset(resolution=RESOLUTION_36KM)
    insitus, insitu_masks = insitu_dataset.get_data_by_date_row_col(all_dates, rows.flatten(), cols.flatten())

    # Load Bias Corrector (already trained)
    rf_corrector = BiasCorrector()
    pred_ys = rf_corrector.predict(
        pred_ys=pred_ys,
        aux_feats=xs
    )
    pred_ys = np.clip(pred_ys, SM_MIN, SM_MAX)

    # Evaluate corrected predictions only
    evaluator = Evaluator(min_site_num=2, min_date_num=2)

    print("\n" + "=" * 60)
    print("Evaluation by Date: Corrected Predictions vs InSitu Data")
    print("=" * 60)

    df_result_date = evaluator.evaluate_by_date(pred_ys, insitus, insitu_masks, all_dates, true_ys=true_ys)
    dst_file_path = os.path.join(RESULT_DIR_PATH, "Evaluation_By_Date_Test.csv")
    df_result_date.to_csv(dst_file_path, index=False)

    print("\n" + "=" * 60)
    print("Evaluation by Site: Corrected Predictions vs InSitu Data")
    print("=" * 60)

    df_result_site = evaluator.evaluate_by_site(pred_ys, insitus, insitu_masks, all_dates, rows, cols,
                                                true_ys=true_ys)
    dst_file_path = os.path.join(RESULT_DIR_PATH, "Evaluation_By_Site_Test.csv")
    df_result_site.to_csv(dst_file_path, index=False)

    # Evaluate by Spatial Distribution
    grid_info = grid_info_store.get()
    evaluator.evaluate_by_spatial_distribution(df_result_site, height=grid_info['H'], width=grid_info['W'])


if __name__ == "__main__":
    main()
