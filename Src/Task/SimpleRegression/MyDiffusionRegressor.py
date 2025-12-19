#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Simple Regression Task
  @Author Chris
  @Date 2025/10/15
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader

from Util.ModelHelper import SinusoidalPosEmb, SimpleResBlock, evaluate

# Dataset setting
X_DIM = 5

# Diffusion setting
# 扩散总步数
STEP_TOTAL_NUM = 200
# Beta setting
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
                np.sin(self.x[:, 0])  # x0 取 sin
                + 0.5 * self.x[:, 1] ** 2  # x1 二次项
                - 0.3 * self.x[:, 2]  # x2 线性项
                + 0.2 * np.cos(self.x[:, 3] * np.pi)  # x3 cos 非线性
                + 0.1 * self.x[:, 0] * self.x[:, 4]  # x0*x4 交互项
                + 0.1 * rng.normal(size=sample_total_num)  # 小高斯噪声
        ).astype(np.float32)
        # [n_samples, ] -> [n_samples, 1]
        self.y = self.y.reshape(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class DiffusionParams:
    """
    step_total_num: 扩散总步数
    beta: 噪声方差 [total_step,]
    alpha: 信号保留比例 [total_step,]
    cumprod_alpha: 累积信号保留比例 [total_step,]
    signal_scale: 信号缩放系数 [total_step,]
    noise_scale: 噪声缩放系数 [total_step,]
    """

    def __init__(self, device, step_total_num=200):
        self.step_total_num = step_total_num
        self.beta = torch.linspace(BETA_START, BETA_END, step_total_num).to(device)
        self.alpha = (1.0 - self.beta).to(device)
        self.cumprod_alpha = torch.cumprod(self.alpha, dim=0).to(device)
        self.signal_scale = torch.sqrt(self.cumprod_alpha).to(device)
        self.noise_scale = torch.sqrt(1 - self.cumprod_alpha).to(device)


class NoisePredictor(nn.Module):
    def __init__(self, x_dim, hidden=256, timestep_emb_dim=64):
        super().__init__()
        self.timestep_embedder = SinusoidalPosEmb(timestep_emb_dim)
        self.timestep_proj = nn.Sequential(
            nn.Linear(timestep_emb_dim, timestep_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(timestep_emb_dim * 2, timestep_emb_dim)
        )
        in_dim = 1 + x_dim + timestep_emb_dim  # y is scalar -> 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            SimpleResBlock(hidden),
            nn.Linear(hidden, 1)  # predict noise scalar
        )

    def forward(self, diffused_ys, xs, timesteps):
        # Embed time step
        embedd_timesteps = self.timestep_embedder(timesteps)
        embedd_timesteps = self.timestep_proj(embedd_timesteps)
        # Concatenate inputs
        inputs = torch.cat([diffused_ys, xs, embedd_timesteps], dim=-1)

        return self.net(inputs)


def forward_diffuse(params: DiffusionParams, ys, steps, device, noises=None):
    """
    Args:
        ys: [batch_size, 1]
        steps: [batch_size,]
        noises: [batch_size, 1]
    Return:
        diffused_ys: [batch_size, 1]
    """
    if noises is None:
        noises = torch.randn_like(ys)
    signal_scale = params.signal_scale[steps].unsqueeze(-1)
    noise_scale = params.noise_scale[steps].unsqueeze(-1)
    diffused_ys = signal_scale * ys + noise_scale * noises

    return diffused_ys, noises


@torch.no_grad()
def reverse_diffuse(model, params: DiffusionParams, xs, device):
    batch_size = xs.shape[0]
    ys = torch.randn(batch_size, 1, device=device)

    for i in reversed(range(params.step_total_num)):
        timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
        # Predict noise
        pred_noises = model.forward(ys, xs, timesteps)  # (batch_size,1)
        beta_t = params.beta[timesteps].unsqueeze(-1)
        alpha_t = params.alpha[timesteps].unsqueeze(-1)
        cumprod_alpha_t = params.cumprod_alpha[timesteps].unsqueeze(-1)
        # Calculate posterior mean
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (beta_t / torch.sqrt(1 - cumprod_alpha_t))
        posterior_means = coef1 * (ys - coef2 * pred_noises)
        # Add random noise for non-zero timesteps
        if i > 0:
            # Calculate posterior variance
            prev_cumprod_alpha = params.cumprod_alpha[timesteps - 1].unsqueeze(-1)  # ᾱ_{t-1}
            posterior_var = beta_t * (1.0 - prev_cumprod_alpha) / (1.0 - cumprod_alpha_t)
            posterior_std = torch.sqrt(posterior_var)
            random_noises = torch.randn_like(ys)
            ys = posterior_means + posterior_std * random_noises
        else:
            ys = posterior_means
    return ys


def train(model, params: DiffusionParams, dataset, device):
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(TOTAL_EPOCH):
        total_loss = 0.0
        for batch_xs, batch_ys in data_loader:
            batch_xs = batch_xs.to(device)
            batch_ys = batch_ys.to(device)
            batch_size = batch_xs.shape[0]
            # Sample randomly Time Step
            sampled_timesteps = torch.randint(low=0, high=STEP_TOTAL_NUM, size=(batch_size,), device=device,
                                              dtype=torch.long)
            # Forward diffuse
            batch_diffused_ys, noises = forward_diffuse(params, batch_ys, sampled_timesteps, device)
            # Predict Noise
            pred_noises = model.forward(batch_diffused_ys, batch_xs, sampled_timesteps)
            loss = functional.mse_loss(pred_noises, noises)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch_size
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{TOTAL_EPOCH}  Avg MSE Loss: {avg_loss:.6f}")
    return model


def predict(model, params: DiffusionParams, dataset, device):
    example_num = 20
    data_loader = DataLoader(dataset, batch_size=len(dataset))

    model.eval()
    xs, true_ys = next(iter(data_loader))
    xs = xs.to(device)
    true_ys = true_ys.to(device)
    with torch.no_grad():
        pred_ys = reverse_diffuse(model, params, xs, device=device)
    for i in range(example_num):
        print(f"X: {xs[i].cpu().numpy()}, Pred Y: {pred_ys[i].item():.4f}, True Y: {true_ys[i].item():.4f}")

    return true_ys, pred_ys


def main():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    print(f"Device: {device}")

    params = DiffusionParams(step_total_num=STEP_TOTAL_NUM, device=device)
    train_dataset = SimpleRegressionDataset(sample_total_num=10000)
    model = NoisePredictor(x_dim=X_DIM, hidden=256, timestep_emb_dim=64).to(device)

    model = train(model, params, train_dataset, device)

    test_dataset = SimpleRegressionDataset(sample_total_num=20, seed=626)
    true_ys, pred_ys = predict(model, params, test_dataset, device)

    evaluate(true_ys, pred_ys)


if __name__ == "__main__":
    main()
