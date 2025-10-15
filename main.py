#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description
  @Author Chris
  @Date 2025/10/15
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from tqdm import trange


# -----------------------------
# 1) 合成数据集（演示用）
# -----------------------------
class SyntheticRegressionDataset(Dataset):
    def __init__(self, n_samples=5000, x_dim=3, seed=0):
        rng = np.random.RandomState(seed)
        self.x = rng.uniform(-2, 2, size=(n_samples, x_dim)).astype(np.float32)
        # 目标 y 为非线性函数，方便测试（标量输出）
        # 示例：y = sin(x0) + 0.5*x1^2 - 0.3*x2 + small noise
        self.y = (np.sin(self.x[:, 0]) + 0.5 * (self.x[:, 1] ** 2) - 0.3 * self.x[:, 2]
                  + 0.1 * rng.normal(size=n_samples)).astype(np.float32)
        # [n_samples, ] -> [n_samples, 1]
        self.y = self.y.reshape(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# -----------------------------
# 2) 扩散过程参数（DDPM 经典）
# -----------------------------



class DiffusionParams:

    def __init__(self, device, T=200):
        self.T = T
        self.beta = self.make_beta_schedule(T)
        self.beta = self.beta.to(device)  # shape (T,)
        self.alpha = 1.0 - self.beta
        self.alpha = self.alpha.to(device)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)  # \bar{alpha}_t
        # precompute for convenience
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)

    @staticmethod
    def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
        # 线性 beta schedule
        return torch.linspace(beta_start, beta_end, T)


# -----------------------------
# 3) 时间嵌入（sinusoidal）
# -----------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: (batch,) int64
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:  # pad if odd
            emb = F.pad(emb, (0, 1))
        return emb  # (batch, dim)


# -----------------------------
# 4) 条件化 MLP 模型（预测 noise ε）
#    输入：x (条件), noisy y, t embedding
#    输出：预测 ε_hat (与 y 相同 shape)
# -----------------------------
class ConditionalMLP(nn.Module):
    def __init__(self, x_dim, hidden=256, time_emb_dim=64):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        # MLP: concat([noisy_y, x, time_emb]) -> MLP -> predict noise
        in_dim = 1 + x_dim + time_emb_dim  # y is scalar -> 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)  # predict noise scalar
        )

    def forward(self, noisy_y, x, t):
        # noisy_y: (B,1), x:(B,x_dim), t:(B,) int
        te = self.time_emb(t)  # (B, time_emb_dim)
        inp = torch.cat([noisy_y, x, te], dim=-1)
        return self.net(inp)  # (B,1) predicted noise


# -----------------------------
# 5) 训练与采样辅助函数
# -----------------------------
def q_sample(params: DiffusionParams, y0, t, noise=None):
    # 给定原始 y0 和 t，产生 y_t = sqrt(alpha_bar_t)*y0 + sqrt(1-alpha_bar_t)*epsilon
    # y0: (B,1), t: (B,) ints in [0,T-1]
    if noise is None:
        noise = torch.randn_like(y0)
    a_bar = params.sqrt_alpha_bar[t].view(-1, 1).to(y0.device).to(y0.device)  # (B,1)
    b_bar = params.sqrt_one_minus_alpha_bar[t].view(-1, 1).to(y0.device)
    return a_bar * y0 + b_bar * noise, noise


@torch.no_grad()
def p_sample_loop(model, params: DiffusionParams, x_cond, device):
    # x_cond: (B, x_dim)
    B = x_cond.shape[0]
    T = params.T
    # start from N(0,1) for y_T
    y = torch.randn(B, 1, device=device)
    for t_ in reversed(range(T)):
        t = torch.full((B,), t_, device=device, dtype=torch.long)
        # predict noise
        eps_pred = model(y, x_cond, t)  # (B,1)
        beta_t = params.beta[t].view(-1, 1).to(device)  # (B,1)
        alpha_t = params.alpha[t].view(-1, 1).to(device)
        alpha_bar_t = params.alpha_bar[t].view(-1, 1).to(device)
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
# -----------------------------
def train_demo(device):
    # 超参数
    x_dim = 3
    epochs = 30
    batch_size = 128
    lr = 2e-4
    T = 200

    params = DiffusionParams(T=T, device=device)
    dataset = SyntheticRegressionDataset(n_samples=5000, x_dim=x_dim)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = ConditionalMLP(x_dim=x_dim, hidden=256, time_emb_dim=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            B = xb.shape[0]
            # sample random t for each instance
            t = torch.randint(low=0, high=T, size=(B,), device=device, dtype=torch.long)
            # sample y_t
            y_t, noise = q_sample(params, yb, t)
            # predict noise
            eps_pred = model(y_t, xb, t)
            loss = F.mse_loss(eps_pred, noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * B
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{epochs}  avg_mse_loss: {avg_loss:.6f}")
    return model, params, dataset


# -----------------------------
# 7) 运行训练并做条件采样示例
# -----------------------------
if __name__ == "__main__":
    device = 'mps' if torch.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    model, params, dataset = train_demo(device=device)
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
