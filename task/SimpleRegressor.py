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
from torch.xpu import device

from util.model_helper import SinusoidalPosEmb

# -----------------------------
# 1) 合成数据集（演示用）
# -----------------------------
# Dataset setting
X_DIM = 3


class SyntheticRegressionDataset(Dataset):
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


# -----------------------------
# 2) 扩散过程参数（DDPM 经典）
# -----------------------------

# 扩散总步数
STEP_TOTAL_NUM = 200
# Beta setting
BETA_START = 1e-4
BETA_END = 0.02

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'


class DiffusionParams:
    """
    step_total_num: 扩散总步数
    beta: 噪声方差 [total_step,]
    alpha: 信号保留比例 [total_step,]
    cumprod_alpha: 累积信号保留比例 [total_step,]
    signal_scale: 信号缩放系数 [total_step,]
    noise_scale: 噪声缩放系数 [total_step,]
    """

    def __init__(self, step_total_num=200):
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


# -----------------------------
# 5) 训练与采样辅助函数
# -----------------------------


"""
ys: [batch_size, 1]
steps: [batch_size,]
noise: [batch_size, 1]
diffused_ys: [batch_size, 1]
"""


def forward_diffuse(params: DiffusionParams, ys, steps, noises=None):
    # 给定原始 y0 和 t，产生 y_t = sqrt(alpha_bar_t)*y0 + sqrt(1-alpha_bar_t)*epsilon
    # y0: (B,1), t: (B,) ints in [0,T-1]
    if noises is None:
        noises = torch.randn_like(ys)
    signal_scale = params.signal_scale[steps].unsqueeze(-1).to(ys.device)  # (B,1)
    noise_scale = params.noise_scale[steps].unsqueeze(-1).to(ys.device)
    diffused_ys = signal_scale * ys + noise_scale * noises

    return diffused_ys, noises


# TODO
@torch.no_grad()
def p_sample_loop(model, params: DiffusionParams, x_cond, device):
    # x_cond: (B, x_dim)
    B = x_cond.shape[0]
    T = params.step_total_num
    # start from N(0,1) for y_T
    y = torch.randn(B, 1, device=device)
    for t_ in reversed(range(T)):
        t = torch.full((B,), t_, device=device, dtype=torch.long)
        # predict noise
        eps_pred = model(y, x_cond, t)  # (B,1)
        beta_t = params.beta[t].view(-1, 1).to(device)  # (B,1)
        alpha_t = params.alpha[t].view(-1, 1).to(device)
        alpha_bar_t = params.cumprod_alpha[t].view(-1, 1).to(device)
        # posterior mean formula for DDPM (original)
        coef1 = 1.0 / torch.sqrt(alpha_t)
        coef2 = (beta_t / torch.sqrt(1 - alpha_bar_t))
        mean = coef1 * (y - coef2 * eps_pred)
        if t_ > 0:
            z = torch.randn_like(y)
            sigma_t = torch.sqrt(beta_t)
            y = mean + sigma_t * z
        else:
            y = mean  # at t=0 no noise
    return y  # (B,1) generated y


# -----------------------------
# 6) Training loop
# Train setting
TOTAL_EPOCH = 30
BATCH_SIZE = 128
lr = 2e-4


def train():
    params = DiffusionParams(step_total_num=STEP_TOTAL_NUM)
    dataset = SyntheticRegressionDataset(sample_total_num=5000)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = NoisePredictor(x_dim=X_DIM, hidden=256, time_emb_dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(TOTAL_EPOCH):
        total_loss = 0.0
        for batch_xs, batch_ys in loader:
            batch_xs = batch_xs.to(device)
            batch_ys = batch_ys.to(device)
            batch_size = batch_xs.shape[0]
            # Sample randomly Time Step
            sampled_steps = torch.randint(low=0, high=STEP_TOTAL_NUM, size=(batch_size,), device=device,
                                          dtype=torch.long)
            # Diffuse Y
            batch_diffused_ys, noises = forward_diffuse(params, batch_ys, sampled_steps)
            # Predict Noise
            pred_noises = model.forward(batch_diffused_ys, batch_xs, sampled_steps)
            loss = F.mse_loss(pred_noises, noises)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch_size
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{TOTAL_EPOCH}  avg_mse_loss: {avg_loss:.6f}")
    return model, params, dataset


# -----------------------------
# 7) 运行训练并做条件采样示例
# -----------------------------
if __name__ == "__main__":
    device = 'mps' if torch.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    model, params, dataset = train()
    model.eval()

    # 随机挑几个 x 做采样（条件化生成 y）
    n_examples = 10
    rng = np.random.RandomState(123)
    x_test = rng.uniform(-2, 2, size=(n_examples, 3)).astype(np.float32)
    x_test_t = torch.from_numpy(x_test).to(device)

    with torch.no_grad():
        y_gen = p_sample_loop(model, params, x_test_t, device=device)  # (n_examples,1)
        y_gen = y_gen.cpu().numpy().reshape(-1)

    # 也可以计算真实 y 来比较（合成数据生成公式）
    y_true = (np.sin(x_test[:, 0]) + 0.5 * (x_test[:, 1] ** 2) - 0.3 * x_test[:, 2]).reshape(-1)

    print("\n示例条件 x, 生成 y  vs 真值 (no observation of y allowed during sampling):")
    for i in range(n_examples):
        print(f"x={x_test[i]}  ->  y_gen={y_gen[i]:.4f}   y_true={y_true[i]:.4f}")
