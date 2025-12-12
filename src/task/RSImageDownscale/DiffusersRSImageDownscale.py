#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Diffusers-Based Geographic Regression Task
  基于扩散模型的地理回归任务：以记录为单位，经纬度直接作为自变量参与回归
  @Author Chris
  @Date 2025/11/12
"""
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.utils.data import Dataset, DataLoader

from Constant import PROJ_PATH, RANGE
from task.RSImageDownscale.Dataset import RSImageDownscaleDataset
from task.RSImageDownscale.Module import TimeEmbedding, SpatialEmbedding, FiLMResBlock
from util.ModelHelper import SinusoidalPosEmb, EarlyStopping, evaluate

# Dataset setting
INPUT_FEATURE_NUM = 5  # NDVI, LST, Albedo, Precipitation, DEM (输入特征数量)
POS_FEATURE_NUM = 2  # Latitude, Longitude (经纬度特征数量，直接作为自变量)

# Diffusion setting
STEP_TOTAL_NUM = 1000  # 增加扩散步数，更细粒度的噪声调度
BETA_START = 1e-4
BETA_END = 0.02

# Model setting
HIDDEN_DIM = 512  # 增加模型容量，从256增加到512
TIMESTEP_EMB_DIM = 128

# Inference setting
INFERENCE_STEPS = 50  # 推理时使用更少的步数（加速采样）

# Train setting
TOTAL_EPOCH = 50
BATCH_SIZE = 64
LR = 2e-4

# Early stopping setting
PATIENCE = 5
MIN_DELTA = 1e-6


class NoisePredictor(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self, input_feature_num: int = 5, pos_feature_num: int = 2,
                 hidden_dim: int = 512, timestep_emb_dim: int = 128,
                 num_res_blocks: int = 3):
        """
        基于记录的地理回归模型（集成时间嵌入、空间嵌入和FiLM+ResBlock）

        Args:
            input_feature_num: 输入特征数量（如NDVI, LST等，通常为5）
            pos_feature_num: 位置特征数量（经纬度，通常为2）
            hidden_dim: 隐藏层维度
            timestep_emb_dim: 时间步嵌入维度
            num_res_blocks: ResBlock 层数
        """
        super().__init__()
        self.input_feature_num = input_feature_num
        self.hidden_dim = hidden_dim

        # 输入特征层（不包含位置，位置单独嵌入）
        input_dim = input_feature_num + 1
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # 扩散时间步嵌入
        self.timestep_embedding = nn.Sequential(
            SinusoidalPosEmb(timestep_emb_dim),
            nn.Linear(timestep_emb_dim, timestep_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(timestep_emb_dim * 4, timestep_emb_dim * 4),
        )
        self.emb_timestep2hidden = nn.Linear(timestep_emb_dim * 4, hidden_dim)

        # 时间嵌入（DOY + 年份 + 多频率 Fourier）
        self.time_embedding = TimeEmbedding(
            hidden_dim=hidden_dim,
            num_fourier=8
        )

        # 空间嵌入（确定性指数频率，分别处理经度和纬度）
        lon_min, lat_min, lon_max, lat_max = RANGE
        self.spatial_embedding = SpatialEmbedding(
            hidden_dim=hidden_dim,
            num_fourier=6,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max
        )

        # 条件信息融合层（融合时间步、时间、空间嵌入）
        self.condition_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # FiLM + ResBlock 层
        self.res_blocks = nn.ModuleList([
            FiLMResBlock(hidden_dim=hidden_dim)
            for _ in range(num_res_blocks)
        ])

        # 输出层
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, diffused_ys: torch.Tensor, xs: torch.Tensor, timesteps: torch.Tensor,
                pos: torch.Tensor, dates: list) -> torch.Tensor:
        """
        基于记录的地理回归前向传播（集成时间嵌入、空间嵌入和FiLM+ResBlock）

        Args:
            diffused_ys: [B, 1] - 扩散后的输出
            xs: [B, input_feature_num] - 输入特征
            timesteps: [B] - 扩散时间步
            pos: [B, pos_feature_num] - 原始坐标 [lon, lat]
            dates: List[str] - 日期字符串列表，格式为 'YYYYMMDD'
        Returns:
            [B, 1] - 预测的噪声
        """
        # 输入特征（不包含位置）
        inputs = torch.cat([xs, diffused_ys], dim=1)
        x = self.input_layer(inputs)

        # 时间步嵌入
        embed_timesteps = self.timestep_embedding(timesteps)
        embed_timesteps = self.emb_timestep2hidden(embed_timesteps)

        # 时间嵌入（从日期字符串提取特征）
        embed_time = self.time_embedding(dates)

        # 空间嵌入
        embed_spatial = self.spatial_embedding(pos)

        # 融合条件信息（时间步 + 时间 + 空间）
        condition = torch.cat([embed_timesteps, embed_time, embed_spatial], dim=1)
        condition = self.condition_fusion(condition)

        # 通过多个 FiLM + ResBlock 层
        for res_block in self.res_blocks:
            x = res_block(x, condition)

        # 输出层
        out = self.output_layer(x)

        return out


class Trainer:

    def __init__(self, model: NoisePredictor, scheduler: DDPMScheduler,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 device: str,
                 early_stopping: EarlyStopping,
                 total_epoch: int,
                 batch_size: int, lr: float):
        self.model = model
        self.scheduler = scheduler
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.early_stopping = early_stopping
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
            batch_dates = list(batch_dates)  # 转换为列表

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
                pos=batch_pos, dates=batch_dates
            )
            # pred_noise: [B, 1], noises: [B, 1]

            # Calculate MSE loss (不再需要mask，所有记录都是有效的)
            loss = nn.functional.mse_loss(pred_noise, noises)
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
                    xs: torch.Tensor, pos: torch.Tensor, dates: list,
                    device: str, num_inference_steps: int = INFERENCE_STEPS) -> torch.Tensor:
    """
    基于记录的反向扩散过程

    Args:
        model: 噪声预测模型
        scheduler: 扩散调度器
        xs: [B, input_feature_num] - 输入特征
        pos: [B, pos_feature_num] - 标准化后的坐标
        dates: List[str] - 日期字符串列表
        device: 设备
        num_inference_steps: 推理步数
    Returns:
        [B, 1] - 预测的输出
    """
    model.eval()
    B = xs.shape[0]

    ys = torch.randn(B, 1, device=device)
    scheduler.set_timesteps(num_inference_steps)

    for timestep in scheduler.timesteps:
        timesteps = torch.full((B,), timestep.item(), device=device, dtype=torch.long)
        pred_noises = model.forward(ys, xs, timesteps, pos=pos, dates=dates)
        step_out = scheduler.step(model_output=pred_noises, timestep=timestep, sample=ys)
        ys = step_out.prev_sample

    return ys


def predict(model: NoisePredictor, scheduler: DDPMScheduler, dataset: Dataset, device: str):
    """
    基于记录的预测函数
    """
    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)  # type: ignore[arg-type]

    all_xs = []
    all_pos = []
    all_true_ys_normalized = []
    all_date_strings = []

    for xs, ys, pos, date_strs in data_loader:
        all_xs.append(xs)
        all_pos.append(pos)
        all_true_ys_normalized.append(ys)
        all_date_strings.extend(list(date_strs))

    xs = torch.cat(all_xs, dim=0).to(device)
    pos = torch.cat(all_pos, dim=0).to(device)
    true_ys_normalized = torch.cat(all_true_ys_normalized, dim=0).to(device)
    date_strings = all_date_strings

    with torch.no_grad():
        pred_ys_normalized = reverse_diffuse(
            model, scheduler, xs, pos, date_strings, device=device
        )
        pred_ys_normalized = pred_ys_normalized.squeeze(1)  # [N, 1] -> [N]

    # 诊断信息：归一化尺度上的MSE
    mse_normalized = nn.functional.mse_loss(pred_ys_normalized, true_ys_normalized).item()
    print(f"\n诊断信息（归一化尺度）:")
    print(f"  - 预测值范围: [{pred_ys_normalized.min().item():.4f}, {pred_ys_normalized.max().item():.4f}]")
    print(f"  - 真实值范围: [{true_ys_normalized.min().item():.4f}, {true_ys_normalized.max().item():.4f}]")
    print(f"  - MSE (归一化尺度): {mse_normalized:.6f}")

    # 反归一化
    true_ys_np = true_ys_normalized.cpu().numpy()  # [N]
    pred_ys_np = pred_ys_normalized.cpu().numpy()  # [N]

    true_ys_denorm = dataset.denormalize_sm(true_ys_np)  # [N]
    pred_ys_denorm = dataset.denormalize_sm(pred_ys_np)  # [N]

    true_ys = torch.from_numpy(true_ys_denorm).to(device)  # [N]
    pred_ys = torch.from_numpy(pred_ys_denorm).to(device)  # [N]

    # 诊断信息：归一化参数
    if hasattr(dataset, 'sm_mean') and hasattr(dataset, 'sm_std'):
        print(f"  - 归一化参数: mean={dataset.sm_mean:.4f}, std={dataset.sm_std:.4f}")
        print(
            f"  - 归一化尺度MSE ({mse_normalized:.6f}) × std² ({dataset.sm_std ** 2:.6f}) = {mse_normalized * dataset.sm_std ** 2:.6f}")
        print(f"  - 这应该接近原始尺度的MSE\n")

    return true_ys, pred_ys


def main():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.mps.is_available():
        device = 'mps'
    print(f"Device: {device}")

    scheduler = build_scheduler()
    model_save_path = PROJ_PATH + "/Checkpoint/RSImageDownscale/Diffusers"

    print("=" * 60)
    print("Training on 36km data...")
    print("=" * 60)

    full_dataset = RSImageDownscaleDataset()

    print(f"\n数据集信息:")
    print(f"  - 总记录数: {len(full_dataset)}")
    if hasattr(full_dataset, 'sm_mean') and hasattr(full_dataset, 'sm_std'):
        print(f"  - SM归一化参数: mean={full_dataset.sm_mean:.4f}, std={full_dataset.sm_std:.4f}")
    print(f"  - 模型配置: hidden_dim={HIDDEN_DIM}, timestep_emb_dim={TIMESTEP_EMB_DIM}")
    print(f"  - 扩散步数: {STEP_TOTAL_NUM} (训练), {INFERENCE_STEPS} (推理)")
    print()

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    model = NoisePredictor(
        input_feature_num=INPUT_FEATURE_NUM,
        pos_feature_num=POS_FEATURE_NUM,
        hidden_dim=HIDDEN_DIM,
        timestep_emb_dim=TIMESTEP_EMB_DIM,
        num_res_blocks=3,
    ).to(device)

    early_stopping = build_early_stopping()
    trainer = Trainer(model, scheduler, train_dataset, val_dataset, device=device, early_stopping=early_stopping,
                      total_epoch=TOTAL_EPOCH, batch_size=BATCH_SIZE, lr=LR)
    model = trainer.run()
    model.save_pretrained(model_save_path)

    # Predict
    model = NoisePredictor.from_pretrained(model_save_path).to(device)
    print("\n" + "=" * 60)
    print(f"Predicting ...")
    print("=" * 60)
    true_ys, pred_ys = predict(model, scheduler, full_dataset, device)

    # Evaluate (所有记录都是有效的，不需要mask)
    # evaluate函数可以处理展平的张量
    true_ys = true_ys.view(-1, 1)  # [N] -> [N, 1]
    pred_ys = pred_ys.view(-1, 1)  # [N] -> [N, 1]
    evaluate(true_ys, pred_ys)


if __name__ == "__main__":
    main()
