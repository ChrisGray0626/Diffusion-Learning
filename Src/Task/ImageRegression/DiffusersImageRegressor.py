#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Diffusers-based Image Regression Task (DDPM) - 2D Image Input/Output
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

from Constant import PROJ_PATH
from Util.ModelHelper import SinusoidalPosEmb, evaluate, calc_mse

# Dataset setting
INPUT_CHANNEL_NUM = 5
OUTPUT_CHANNEL_NUM = 1
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64

# Diffusion setting
STEP_TOTAL_NUM = 200
BETA_START = 1e-4
BETA_END = 0.02

# Train setting
TOTAL_EPOCH = 10
BATCH_SIZE = 32
lr = 2e-4


class ImageRegressionDataset(Dataset):
    def __init__(self, sample_total_num, seed=42,
                 missing_prob: float = 0.1):
        rng = np.random.RandomState(seed)
        self.xs = rng.uniform(-1, 1, size=(sample_total_num, IMAGE_HEIGHT, IMAGE_WIDTH, INPUT_CHANNEL_NUM)).astype(
            np.float32)  # [B, H, W, IC]
        ys = (
                np.sin(self.xs[:, :, :, 0]) * 0.5
                + 0.3 * self.xs[:, :, :, 1] ** 2
                - 0.2 * self.xs[:, :, :, 2]
                + 0.2 * np.cos(self.xs[:, :, :, 3] * np.pi)
                + 0.1 * self.xs[:, :, :, 0] * self.xs[:, :, :, 4]
                + 0.05 * rng.normal(size=(sample_total_num, IMAGE_HEIGHT, IMAGE_WIDTH))
        ).astype(np.float32)  # [B, H, W, OC]
        self.ys = ys.reshape(sample_total_num, IMAGE_HEIGHT, IMAGE_WIDTH, OUTPUT_CHANNEL_NUM)

        # Generate mask
        if missing_prob > 0:
            # Bernoulli sampling
            masks = rng.binomial(1, 1.0 - missing_prob,
                                 size=(sample_total_num, IMAGE_HEIGHT, IMAGE_WIDTH, 1))
        else:
            masks = np.ones((sample_total_num, IMAGE_HEIGHT, IMAGE_WIDTH, 1))
        self.masks = masks.astype(np.float32)  # [B, H, W, 1]

        # Mask X
        self.xs = self.xs * self.masks  # [B, H, W, IC]
        # Mask Y
        self.ys = self.ys * self.masks  # [B, H, W, OC]

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        xs = torch.from_numpy(self.xs[idx]).permute(2, 0, 1)  # [H, W, IC] -> [IC, H, W]
        ys = torch.from_numpy(self.ys[idx]).permute(2, 0, 1)  # [H, W, OC] -> [OC, H, W]
        masks = torch.from_numpy(self.masks[idx]).permute(2, 0, 1)  # [H, W, OC] -> [OC, H, W]

        return xs, ys, masks


class NoisePredictor(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self, input_channel_num, output_channel_num: int = 1, hidden_dim: int = 64,
                 timestep_emb_dim: int = 64):
        super().__init__()
        self.input_channels = input_channel_num
        self.hidden_dim = hidden_dim
        self.timestep_embedder = SinusoidalPosEmb(timestep_emb_dim)

        # Timestep embedding projection
        self.timestep_proj = nn.Sequential(
            nn.Linear(timestep_emb_dim, timestep_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(timestep_emb_dim * 4, timestep_emb_dim * 4),
        )
        # Input convolutional layer
        self.input_conv = nn.Conv2d(output_channel_num + input_channel_num, hidden_dim, kernel_size=3, padding=1)

        # Timestep embedding convolution
        self.timestep_emb_conv = nn.Conv2d(timestep_emb_dim * 4, hidden_dim, kernel_size=1)

        self.net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1)
        )

    def forward(self, diffused_ys: torch.Tensor, xs: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            diffused_ys: [B, output_channel_num, H, W]
            xs: [B, input_channel_num, H, W]
            timesteps: [B]
        Returns:
            [B, 1, H, W]
        """
        # Embed timestep
        embedd_timesteps = self.timestep_embedder(timesteps)  # [B, timestep_emb_dim]
        embedd_timesteps = self.timestep_proj(embedd_timesteps)  # [B, timestep_emb_dim * 4]
        # Expand timestep embedding to spatial dimensions
        B, _, H, W = diffused_ys.shape
        # Insert two singleton dimensions for H and W
        embedd_timesteps = embedd_timesteps[
            :, :, None, None]  # [B, timestep_emb_dim * 4] -> [B, timestep_emb_dim * 4, 1, 1]
        # Expand to [B, timestep_emb_dim * 4, H, W]
        embedd_timesteps = embedd_timesteps.expand(B, -1, H,
                                                   W)  # [B, timestep_emb_dim * 4, 1, 1] -> [B, timestep_emb_dim * 4, H, W]
        # Convolve timestep embeddings
        embedd_timesteps = self.timestep_emb_conv(
            embedd_timesteps)  # [B, timestep_emb_dim * 4, H, W] -> [B, hidden_dim, H, W]

        # Concatenate diffused_ys and xs as inputs
        inputs = torch.cat([diffused_ys, xs], dim=1)  # [B, input_channel_num + output_channel_num, H, W]
        # Convolve inputs
        inputs = self.input_conv(
            inputs)  # [B, input_channel_num + output_channel_num, H, W] -> [B, hidden_channels, H, W]
        # Add timestep embeddings
        inputs = inputs + embedd_timesteps

        return self.net(inputs)


def build_scheduler():
    scheduler = DDPMScheduler(
        num_train_timesteps=STEP_TOTAL_NUM,
        beta_start=BETA_START,
        beta_end=BETA_END,
        beta_schedule="linear",
        prediction_type="epsilon",  # predict noise
        clip_sample=False,
    )

    return scheduler


def train(model: NoisePredictor, scheduler: DDPMScheduler, dataset: ImageRegressionDataset, device: str):
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(TOTAL_EPOCH):
        total_loss = 0.0
        total_samples = 0
        for batch_xs, batch_ys, batch_masks in data_loader:
            batch_xs = batch_xs.to(device)
            batch_ys = batch_ys.to(device)
            batch_masks = batch_masks.to(device)
            batch_size = batch_xs.shape[0]

            # Sample random timesteps uniformly
            sampled_timesteps = torch.randint(0, scheduler.config['num_train_timesteps'], (batch_size,), device=device)

            # Forward diffuse
            noises = torch.randn_like(batch_ys)
            diffused_ys = scheduler.add_noise(original_samples=batch_ys, noise=noises,
                                              timesteps=sampled_timesteps)

            # Predict noise
            pred_noise = model.forward(diffused_ys, batch_xs, sampled_timesteps)
            # Calculate masked MSE
            loss = calc_mse(pred_noise, noises, batch_masks)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        print(f"Epoch {epoch + 1}/{TOTAL_EPOCH}  Avg MSE Loss: {avg_loss:.6f}")

    return model


@torch.no_grad()
def reverse_diffuse(model: NoisePredictor, scheduler: DDPMScheduler, xs: torch.Tensor, device: str) -> torch.Tensor:
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
        pred_noises = model.forward(ys, xs, timesteps)
        # One reverse step
        step_out = scheduler.step(model_output=pred_noises, timestep=timestep, sample=ys)
        ys = step_out.prev_sample

    return ys


def predict(model: NoisePredictor, scheduler: DDPMScheduler, dataset: ImageRegressionDataset, device: str):
    example_num = 5
    data_loader = DataLoader(dataset, batch_size=len(dataset))

    xs, true_ys, _ = next(iter(data_loader))
    xs = xs.to(device)
    true_ys = true_ys.to(device)
    with torch.no_grad():
        pred_ys = reverse_diffuse(model, scheduler, xs, device=device)

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
    model_save_path = PROJ_PATH + "/Checkpoint/ImageRegression/Diffusers"

    # Train
    train_dataset = ImageRegressionDataset(sample_total_num=1000)
    model = NoisePredictor(input_channel_num=INPUT_CHANNEL_NUM, hidden_dim=64, timestep_emb_dim=64).to(device)
    model = train(model, scheduler, train_dataset, device)
    model.save_pretrained(model_save_path)

    # Predict
    model = NoisePredictor.from_pretrained(model_save_path).to(device)
    test_dataset = ImageRegressionDataset(sample_total_num=20, missing_prob=0.0, seed=626)
    true_ys, pred_ys = predict(model, scheduler, test_dataset, device)

    # Evaluate
    true_ys = true_ys.view(true_ys.shape[0], -1)
    pred_ys = pred_ys.view(pred_ys.shape[0], -1)

    evaluate(true_ys, pred_ys)


if __name__ == "__main__":
    main()
