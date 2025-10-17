#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2025/10/15
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from util.model_helper import SinusoidalPosEmb

# Dataset setting
X_DIM = 3

# Diffusion setting
# 扩散总步数
STEP_TOTAL_NUM = 200
# Beta setting
BETA_START = 1e-4
BETA_END = 0.02

# Train setting
TOTAL_EPOCH = 30
BATCH_SIZE = 128
lr = 2e-4
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'


class SimpleRegressionDataset(Dataset):
    def __init__(self, sample_total_num, seed=0):
        rng = np.random.RandomState(seed)
        self.x = rng.uniform(-2, 2, size=(sample_total_num, 3)).astype(np.float32)
        # 目标 y 为非线性函数，方便测试（标量输出）
        # 示例：y = sin(x0) + 0.5*x1^2 - 0.3*x2 + small noise
        self.y = (np.sin(self.x[:, 0]) + 0.5 * (self.x[:, 1] ** 2) - 0.3 * self.x[:, 2]
                  + 0.1 * rng.normal(size=sample_total_num)).astype(np.float32)
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
        self.cumprod_alpha = torch.cumprod(self.alpha, dim=0)
        self.signal_scale = torch.sqrt(self.cumprod_alpha).to(device)
        self.noise_scale = torch.sqrt(1 - self.cumprod_alpha).to(device)


class NoisePredictor(nn.Module):
    def __init__(self, x_dim, hidden=256, time_emb_dim=64):
        super().__init__()
        self.step_embedder = SinusoidalPosEmb(time_emb_dim)
        # MLP: concat([noisy_y, x, time_emb]) -> MLP -> predict noise
        in_dim = 1 + x_dim + time_emb_dim  # y is scalar -> 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)  # predict noise scalar
        )

    def forward(self, diffused_ys, xs, steps):
        # Embed time step
        embedd_steps = self.step_embedder(steps)
        # Concatenate inputs
        combined_features = torch.cat([diffused_ys, xs, embedd_steps], dim=-1)

        return self.net(combined_features)


def forward_diffuse(params: DiffusionParams, ys, steps, device, noises=None):
    """
    Args:
        ys: [batch_size, 1]
        steps: [batch_size,]
        noises: [batch_size, 1]
    Return:
        diffused_ys: [batch_size, 1]
    """
    # 给定原始 y0 和 t，产生 y_t = sqrt(alpha_bar_t)*y0 + sqrt(1-alpha_bar_t)*epsilon
    # y0: (B,1), t: (B,) ints in [0,T-1]
    if noises is None:
        noises = torch.randn_like(ys)
    signal_scale = params.signal_scale[steps].unsqueeze(-1).to(device)  # (B,1)
    noise_scale = params.noise_scale[steps].unsqueeze(-1).to(device)
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
            random_noises = torch.randn_like(ys)
            sigma_t = torch.sqrt(beta_t)
            ys = posterior_means + sigma_t * random_noises
        else:
            ys = posterior_means
        # ys = posterior_means
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
            loss = F.mse_loss(pred_noises, noises)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch_size
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{TOTAL_EPOCH}  avg_mse_loss: {avg_loss:.6f}")
    return model, params, dataset


def evaluate(model, params: DiffusionParams, device):
    example_num = 10
    dataset = SimpleRegressionDataset(sample_total_num=example_num)
    data_loader = DataLoader(dataset, batch_size=example_num)

    model.eval()
    xs, ys = next(iter(data_loader))
    xs = xs.to(device)
    ys = ys.to(device)
    with torch.no_grad():
        ys_gen = reverse_diffuse(model, params, xs, device=device)
    for i in range(example_num):
        print(f"条件 x: {xs[i].cpu().numpy()}, 生成 y: {ys_gen[i].item():.4f}, 真实 y: {ys[i].item():.4f}")
    mse = F.mse_loss(ys_gen, ys)
    print(f"Evaluation MSE: {mse.item():.6f}")


def main():
    print(f"Device: {device}")

    params = DiffusionParams(step_total_num=STEP_TOTAL_NUM, device=device)
    dataset = SimpleRegressionDataset(sample_total_num=5000)
    model = NoisePredictor(x_dim=X_DIM, hidden=256, time_emb_dim=64).to(device)

    model, params, dataset = train(model, params, dataset, device)

    evaluate(model, params, device)


if __name__ == "__main__":
    main()
