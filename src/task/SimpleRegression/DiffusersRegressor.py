#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Diffusers-based Simple Regression Task (DDPM)
  @Author Chris
  @Date 2025/11/05
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from diffusers import DDPMScheduler
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.utils.data import Dataset, DataLoader

from Constant import PROJ_PATH
from util.ModelHelper import SinusoidalPosEmb, SimpleResBlock, evaluate

# Dataset setting
X_DIM = 5

# Diffusion setting
STEP_TOTAL_NUM = 200
BETA_START = 1e-4
BETA_END = 0.02

# Train setting
TOTAL_EPOCH = 10
BATCH_SIZE = 128
lr = 2e-4


class SimpleRegressionDataset(Dataset):
    def __init__(self, sample_total_num, seed=42):
        rng = np.random.RandomState(seed)
        self.x = rng.uniform(-1, 1, size=(sample_total_num, X_DIM)).astype(np.float32)
        self.y = (
                np.sin(self.x[:, 0])
                + 0.5 * self.x[:, 1] ** 2
                - 0.3 * self.x[:, 2]
                + 0.2 * np.cos(self.x[:, 3] * np.pi)
                + 0.1 * self.x[:, 0] * self.x[:, 4]
                + 0.1 * rng.normal(size=sample_total_num)
        ).astype(np.float32)
        self.y = self.y.reshape(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ModelMixin: A base class for models in diffusers that provides utility functions.
# ConfigMixin: A base class for models in diffusers that provides configuration management.
class NoisePredictor(ModelMixin, ConfigMixin):

    # register_to_config: Register the arguments  to the model's configuration.
    @register_to_config
    def __init__(self, x_dim: int, hidden_dim: int = 256, timestep_emb_dim: int = 64):
        super().__init__()
        self.x_dim = x_dim
        self.hidden = hidden_dim
        self.timestep_embedder = SinusoidalPosEmb(timestep_emb_dim)

        # Positional timestep embedding similar to standard diffusion MLPs
        self.timestep_proj = nn.Sequential(
            nn.Linear(timestep_emb_dim, timestep_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(timestep_emb_dim * 2, timestep_emb_dim),
        )

        in_dim = 1 + x_dim + timestep_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            SimpleResBlock(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, diffused_ys: torch.Tensor, xs: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        embedd_timesteps = self.timestep_embedder(timesteps)
        embedd_timesteps = self.timestep_proj(embedd_timesteps)
        feats = torch.cat([diffused_ys, xs, embedd_timesteps], dim=-1)

        return self.net(feats)


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


def train(model: NoisePredictor, scheduler: DDPMScheduler, dataset: SimpleRegressionDataset, device: str):
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(TOTAL_EPOCH):
        total_loss = 0.0
        total_samples = 0
        for batch_xs, batch_ys in data_loader:
            batch_xs = batch_xs.to(device)
            batch_ys = batch_ys.to(device)
            batch_size = batch_xs.shape[0]

            # Sample random timesteps uniformly
            sampled_timesteps = torch.randint(0, scheduler.config['num_train_timesteps'], (batch_size,), device=device)

            # Forward diffuse
            noises = torch.randn_like(batch_ys)
            diffused_ys = scheduler.add_noise(original_samples=batch_ys, noise=noises,
                                              timesteps=sampled_timesteps)

            # Predict noise
            pred_noise = model.forward(diffused_ys, batch_xs, sampled_timesteps)
            loss = functional.mse_loss(pred_noise, noises)

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
    batch_size = xs.shape[0]
    ys = torch.randn(batch_size, 1, device=device)

    # Create standard decreasing timesteps for sampling
    timesteps = scheduler.timesteps
    # For sampling we need to set the scheduler to inference mode
    scheduler.set_timesteps(len(timesteps))

    # When using set_timesteps, scheduler.timesteps is a long tensor like [T-1, ..., 0]
    for t in scheduler.timesteps:
        t_batch = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
        # Predict noise
        pred_noises = model.forward(ys, xs, t_batch)
        # One reverse step
        step_out = scheduler.step(model_output=pred_noises, timestep=t, sample=ys)
        ys = step_out.prev_sample

    return ys


def predict(model: NoisePredictor, scheduler: DDPMScheduler, dataset: SimpleRegressionDataset, device: str):
    example_num = 20
    data_loader = DataLoader(dataset, batch_size=len(dataset))

    xs, true_ys = next(iter(data_loader))
    xs = xs.to(device)
    true_ys = true_ys.to(device)
    with torch.no_grad():
        pred_ys = reverse_diffuse(model, scheduler, xs, device=device)
    for i in range(min(example_num, xs.shape[0])):
        print(f"X: {xs[i].cpu().numpy()}, Pred Y: {pred_ys[i].item():.4f}, True Y: {true_ys[i].item():.4f}")

    return true_ys, pred_ys


def main():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.mps.is_available():
        device = 'mps'
    print(f"Device: {device}")

    scheduler = build_scheduler()
    model_save_path = PROJ_PATH + "/Checkpoint/SimpleRegression/Diffusers"

    # Train
    train_dataset = SimpleRegressionDataset(sample_total_num=10000)
    model = NoisePredictor(x_dim=X_DIM, hidden_dim=256, timestep_emb_dim=64).to(device)
    model = train(model, scheduler, train_dataset, device)
    model.save_pretrained(model_save_path)

    # Predict
    model = NoisePredictor.from_pretrained(model_save_path).to(device)
    test_dataset = SimpleRegressionDataset(sample_total_num=20, seed=626)
    true_ys, pred_ys = predict(model, scheduler, test_dataset, device)

    # Evaluate
    evaluate(true_ys, pred_ys)


if __name__ == "__main__":
    main()
