#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Diffusers-based Image Downscale Task (DDPM) - 2D Image Input/Output
  @Author Chris
  @Date 2025/11/05
"""

import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.utils.data import Dataset, DataLoader

from constant import ROOT_PATH
from util.ModelHelper import SinusoidalPosEmb, evaluate, calc_masked_mse, PosEmbedding, BaseResBlock

# Dataset setting
INPUT_CHANNEL_NUM = 5
OUTPUT_CHANNEL_NUM = 1

# Diffusion setting
STEP_TOTAL_NUM = 200
BETA_START = 1e-4
BETA_END = 0.02

# Train setting
TOTAL_EPOCH = 20
BATCH_SIZE = 32
LR = 2e-4


class ImageDownscaleDataset(Dataset):
    def __init__(self, sample_total_num, seed=42,
                 missing_prob: float = 0.1,
                 image_height: int = 64,
                 image_width: int = 64,
                 lat_min: float = -90.0,
                 lat_max: float = 90.0,
                 lon_min: float = -180.0,
                 lon_max: float = 180.0):

        rng = np.random.RandomState(seed)
        self.image_height = image_height
        self.image_width = image_width

        # Build X & Y
        self.xs = rng.uniform(-1, 1, size=(sample_total_num, image_height, image_width, INPUT_CHANNEL_NUM)).astype(
            np.float32)  # [B, H, W, IC]
        ys = (
                np.sin(self.xs[:, :, :, 0]) * 0.5
                + 0.3 * self.xs[:, :, :, 1] ** 2
                - 0.2 * self.xs[:, :, :, 2]
                + 0.2 * np.cos(self.xs[:, :, :, 3] * np.pi)
                + 0.1 * self.xs[:, :, :, 0] * self.xs[:, :, :, 4]
                + 0.05 * rng.normal(size=(sample_total_num, image_height, image_width))
        ).astype(np.float32)  # [B, H, W, OC]
        self.ys = ys.reshape(sample_total_num, image_height, image_width, OUTPUT_CHANNEL_NUM)

        # Build spatial position grid
        lats = np.linspace(lat_min, lat_max, image_height)  # [H]
        lons = np.linspace(lon_min, lon_max, image_width)  # [W]
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')  # [H, W]
        self.pos = np.stack([lat_grid, lon_grid], axis=-1).astype(np.float32)  # [H, W, 2]

        # Build mask
        if missing_prob > 0:
            # Bernoulli sampling
            masks = rng.binomial(1, 1.0 - missing_prob,
                                 size=(sample_total_num, image_height, image_width, 1))
        else:
            masks = np.ones((sample_total_num, image_height, image_width, 1))
        self.masks = masks.astype(np.float32)  # [B, H, W, 1]

        # Mask X & Y
        self.xs = self.xs * self.masks  # [B, H, W, IC]
        self.ys = self.ys * self.masks  # [B, H, W, OC]

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        xs = torch.from_numpy(self.xs[idx]).permute(2, 0, 1)  # [H, W, IC] -> [IC, H, W]
        ys = torch.from_numpy(self.ys[idx]).permute(2, 0, 1)  # [H, W, OC] -> [OC, H, W]
        pos = torch.from_numpy(self.pos).permute(2, 0, 1)  # [H, W, 2] -> [2, H, W]
        masks = torch.from_numpy(self.masks[idx]).permute(2, 0, 1)  # [H, W, 1] -> [1, H, W]

        return xs, ys, pos, masks


class ResBlock(BaseResBlock):

    def __init__(self, dim):
        super().__init__(dim)
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
        )


class NoisePredictor(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self, input_channel_num, output_channel_num: int = 1, hidden_dim: int = 128,
                 timestep_emb_dim: int = 128, pos_embed_dim: int = 64,
                 lat_min: float = -90.0, lat_max: float = 90.0,
                 lon_min: float = -180.0, lon_max: float = 180.0):
        super().__init__()
        self.input_channels = input_channel_num
        self.hidden_dim = hidden_dim

        # Input layer
        self.input_layer = nn.Conv2d(output_channel_num + input_channel_num, hidden_dim, kernel_size=3, padding=1)

        # Timestep embedding
        self.timestep_embedder = SinusoidalPosEmb(timestep_emb_dim)
        self.timestep_embedding = nn.Sequential(
            SinusoidalPosEmb(timestep_emb_dim),
            nn.Linear(timestep_emb_dim, timestep_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(timestep_emb_dim * 4, timestep_emb_dim * 4),
        )
        self.timestep_emb2hidden = nn.Conv2d(timestep_emb_dim * 4, hidden_dim, kernel_size=1)

        # Position embedding: from coordinates to hidden dimension
        self.pos_embedding = nn.Sequential(
            PosEmbedding(
                embed_dim=pos_embed_dim,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max
            ),
            nn.Conv2d(pos_embed_dim, hidden_dim, kernel_size=1)
        )

        self.net = nn.Sequential(
            ResBlock(hidden_dim),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        )

    def forward(self, diffused_ys: torch.Tensor, xs: torch.Tensor, timesteps: torch.Tensor,
                pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            diffused_ys: [B, output_channel_num, H, W]
            xs: [B, input_channel_num, H, W]
            timesteps: [B]
            pos: [B, 2, H, W]
        Returns:
            [B, 1, H, W]
        """

        # Concatenate diffused_ys and xs as inputs
        inputs = torch.cat([diffused_ys, xs], dim=1)  # [B, input_channel_num + output_channel_num, H, W]
        # Convolve inputs to [B, hidden_channels, H, W]
        inputs = self.input_layer(inputs)

        # Embed timestep
        embed_timesteps = self.timestep_embedding(timesteps)  # [B, timestep_emb_dim * 4]
        # Insert two singleton dimensions for H and W
        embed_timesteps = embed_timesteps[:, :, None, None]
        # Expand to [B, timestep_emb_dim * 4, H, W]
        B, _, H, W = diffused_ys.shape
        embed_timesteps = embed_timesteps.expand(B, -1, H, W)
        # Convolve timestep embeddings to [B, hidden_dim, H, W]
        embed_timesteps = self.timestep_emb2hidden(embed_timesteps)

        # Embed spatial position
        embed_pos = self.pos_embedding(pos)  # [B, 2, H, W] -> [B, hidden_dim, H, W]

        # Add timestep embeddings & position embeddings
        inputs = inputs + embed_timesteps + embed_pos

        out = self.net(inputs)

        return out


def build_scheduler():
    scheduler = DDPMScheduler(
        num_train_timesteps=STEP_TOTAL_NUM,
        beta_start=BETA_START,
        beta_end=BETA_END,
        beta_schedule="linear",
        prediction_type="epsilon",
        clip_sample=False,
    )

    return scheduler


def train(model: NoisePredictor, scheduler: DDPMScheduler, dataset: ImageDownscaleDataset, device: str):
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=TOTAL_EPOCH, eta_min=1e-6
    )

    model.train()
    for epoch in range(TOTAL_EPOCH):
        total_loss = 0.0
        total_samples = 0
        for batch_xs, batch_ys, batch_pos, batch_masks in data_loader:
            batch_xs = batch_xs.to(device)
            batch_ys = batch_ys.to(device)
            batch_masks = batch_masks.to(device)
            batch_pos = batch_pos.to(device)
            batch_size = batch_xs.shape[0]

            # Sample random timesteps uniformly
            sampled_timesteps = torch.randint(0, scheduler.config['num_train_timesteps'], (batch_size,), device=device)

            # Forward diffuse
            noises = torch.randn_like(batch_ys)
            diffused_ys = scheduler.add_noise(original_samples=batch_ys, noise=noises,
                                              timesteps=sampled_timesteps)

            # Predict noise (pass position coordinates from dataset)
            pred_noise = model.forward(diffused_ys, batch_xs, sampled_timesteps, pos=batch_pos)
            # Calculate masked MSE
            loss = calc_masked_mse(pred_noise, noises, batch_masks)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        current_lr = opt.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{TOTAL_EPOCH}  Avg MSE Loss: {avg_loss:.6f}  LR: {current_lr:.6f}")

        # Update learning rate
        lr_scheduler.step()

    return model


@torch.no_grad()
def reverse_diffuse(model: NoisePredictor, scheduler: DDPMScheduler, xs: torch.Tensor, pos: torch.Tensor,
                    device: str) -> torch.Tensor:
    model.eval()
    B, _, H, W = xs.shape
    ys = torch.randn(B, OUTPUT_CHANNEL_NUM, H, W, device=device)

    # Create inference timesteps
    timestep_total_num = scheduler.config['num_train_timesteps']
    # Set the scheduler to inference mode
    scheduler.set_timesteps(timestep_total_num)

    for timestep in scheduler.timesteps:
        timesteps = torch.full((B,), timestep.item(), device=device, dtype=torch.long)
        # Predict noise
        pred_noises = model.forward(ys, xs, timesteps, pos=pos)
        # One reverse step
        step_out = scheduler.step(model_output=pred_noises, timestep=timestep, sample=ys)
        ys = step_out.prev_sample

    return ys


def predict(model: NoisePredictor, scheduler: DDPMScheduler, dataset: ImageDownscaleDataset, device: str):
    example_num = 5
    data_loader = DataLoader(dataset, batch_size=len(dataset))

    xs, true_ys, pos, _ = next(iter(data_loader))
    xs = xs.to(device)
    true_ys = true_ys.to(device)
    pos = pos.to(device)
    with torch.no_grad():
        pred_ys = reverse_diffuse(model, scheduler, xs, pos, device=device)

    for i in range(min(example_num, xs.shape[0])):
        print(f"  True Y range: [{true_ys[i].min().item():.4f}, {true_ys[i].max().item():.4f}]")
        print(f"  Pred Y range: [{pred_ys[i].min().item():.4f}, {pred_ys[i].max().item():.4f}]")

    return true_ys, pred_ys


def main():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.mps.is_available():
        device = 'mps'
    print(f"Device: {device}")

    scheduler = build_scheduler()
    model_save_path = ROOT_PATH + "/Checkpoint/ImageDownscale/Diffusers"

    lat_min = 33.0
    lat_max = 49.0
    lon_min = -120.0
    lon_max = -103.0

    # Train on 64x64 data
    print("=" * 60)
    print("Training on 64x64 data...")
    print("=" * 60)
    train_dataset = ImageDownscaleDataset(
        sample_total_num=1000,
        missing_prob=0.2,
        image_height=64,
        image_width=64,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max
    )

    model = NoisePredictor(
        input_channel_num=INPUT_CHANNEL_NUM,
        hidden_dim=128,
        timestep_emb_dim=64,
        pos_embed_dim=32,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max
    ).to(device)
    model = train(model, scheduler, train_dataset, device)
    model.save_pretrained(model_save_path)

    # Predict on 256x256
    model = NoisePredictor.from_pretrained(model_save_path).to(device)
    for scale in [128]:
        print("\n" + "=" * 60)
        print(f"Predicting on {scale} * {scale} data...")
        print("=" * 60)
        test_dataset = ImageDownscaleDataset(
            sample_total_num=10,
            seed=626,
            image_height=scale,
            image_width=scale,
            lat_min=lat_min,  # Same boundaries as training
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max
        )
        true_ys, pred_ys = predict(model, scheduler, test_dataset, device)

        # Evaluate
        true_ys = true_ys.view(true_ys.shape[0], -1)
        pred_ys = pred_ys.view(pred_ys.shape[0], -1)

        evaluate(true_ys, pred_ys)


if __name__ == "__main__":
    main()
