#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Diffusers-Channel Attention-CNN Based Image Downscaling Task
  @Author Chris
  @Date 2025/11/12
"""
import numpy as np
import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.utils.data import Dataset, DataLoader

from Constant import PROJ_PATH
from task.ImageDownscale.Module import ChannelAttention, ResBlock
from task.RSImageDownscale.Dataset import RSImageDownscaleDataset
from util.ModelHelper import SinusoidalPosEmb, calc_masked_mse, PosEmbedding, EarlyStopping, evaluate

# Dataset setting
INPUT_CHANNEL_NUM = 5
OUTPUT_CHANNEL_NUM = 1

# Diffusion setting
STEP_TOTAL_NUM = 200
BETA_START = 1e-4
BETA_END = 0.02

# Train setting
TOTAL_EPOCH = 50
BATCH_SIZE = 1
LR = 2e-4

# Early stopping setting
PATIENCE = 5
MIN_DELTA = 1e-6


class NoisePredictor(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self, input_channel_num, output_channel_num: int = 1, hidden_dim: int = 128,
                 timestep_emb_dim: int = 128, pos_embed_dim: int = 64,
                 lat_min: float = -90.0, lat_max: float = 90.0,
                 lon_min: float = -180.0, lon_max: float = 180.0,
                 channel_attention_reduction: int = 16):
        super().__init__()
        self.input_channel_num = input_channel_num
        self.hidden_dim = hidden_dim

        # Input layer
        self.input_layer = nn.Conv2d(output_channel_num + input_channel_num, hidden_dim, kernel_size=3, padding=1)

        # Timestep embedding
        self.timestep_embedding = nn.Sequential(
            SinusoidalPosEmb(timestep_emb_dim),
            nn.Linear(timestep_emb_dim, timestep_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(timestep_emb_dim * 4, timestep_emb_dim * 4),
        )
        self.emb_timestep2hidden = nn.Conv2d(timestep_emb_dim * 4, hidden_dim, kernel_size=1)

        # Position embedding
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
            ChannelAttention(dim=hidden_dim, reduction=channel_attention_reduction),
            ResBlock(hidden_dim),
            ChannelAttention(dim=hidden_dim, reduction=channel_attention_reduction),
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
        embed_timesteps = self.emb_timestep2hidden(embed_timesteps)

        # Embed spatial position
        embed_pos = self.pos_embedding(pos)  # [B, 2, H, W] -> [B, hidden_dim, H, W]

        # Add timestep embeddings & position embeddings
        inputs = inputs + embed_timesteps + embed_pos

        out = self.net(inputs)

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

        for batch_xs, batch_ys, batch_pos, batch_masks in data_loader:
            batch_xs = batch_xs.to(self.device)
            batch_ys = batch_ys.to(self.device)
            batch_masks = batch_masks.to(self.device)
            batch_pos = batch_pos.to(self.device)

            B = batch_xs.shape[0]

            # Sample random timesteps uniformly
            sampled_timesteps = torch.randint(
                0, self.scheduler.config['num_train_timesteps'],
                (B,), device=self.device, dtype=torch.long
            )

            # Forward diffuse
            noises = torch.randn_like(batch_ys)
            diffused_ys = self.scheduler.add_noise(
                original_samples=batch_ys,
                noise=noises,
                timesteps=sampled_timesteps  # type: ignore[arg-type]
            )

            # Predict noise
            pred_noise = self.model.forward(diffused_ys, batch_xs, sampled_timesteps, pos=batch_pos)

            # Calculate masked MSE
            loss = calc_masked_mse(pred_noise, noises, batch_masks)

            # # Check if loss is valid and requires grad
            # if torch.isnan(loss) or torch.isinf(loss):
            #     print(f"WARNING: Invalid loss detected: {loss.item()}")
            #     print(f"  batch_masks sum: {batch_masks.sum().item()}")
            #     continue
            #
            # if not loss.requires_grad:
            #     print(f"WARNING: Loss does not require grad! Loss value: {loss.item()}")
            #     print(f"  pred_noise requires_grad: {pred_noise.requires_grad}")
            #     print(f"  batch_masks sum: {batch_masks.sum().item()}")
            #     # Try to create a loss with gradient
            #     loss = (pred_noise * 0.0).sum() + loss
            #     if not loss.requires_grad:
            #         print(f"  Still no gradient after fix, skipping batch")
            #         continue

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


def predict(model: NoisePredictor, scheduler: DDPMScheduler, dataset: Dataset, device: str):
    data_loader = DataLoader(dataset, batch_size=len(dataset))  # type: ignore[arg-type]

    xs, true_ys_normalized, pos, masks = next(iter(data_loader))
    xs = xs.to(device)
    true_ys_normalized = true_ys_normalized.to(device)
    pos = pos.to(device)
    masks = masks.to(device)

    with torch.no_grad():
        pred_ys_normalized = reverse_diffuse(model, scheduler, xs, pos, device=device)

    true_ys_np = true_ys_normalized.cpu().numpy()  # [B, 1, H, W]
    pred_ys_np = pred_ys_normalized.cpu().numpy()  # [B, 1, H, W]

    true_ys_denorm = []
    pred_ys_denorm = []
    for i in range(true_ys_np.shape[0]):
        sm_2d_true = true_ys_np[i, 0]  # [H, W]
        sm_2d_pred = pred_ys_np[i, 0]  # [H, W]
        true_ys_denorm.append(dataset.denormalize_sm(sm_2d_true))  # [H, W]
        pred_ys_denorm.append(dataset.denormalize_sm(sm_2d_pred))  # [H, W]

    true_ys = torch.from_numpy(np.array(true_ys_denorm)[:, np.newaxis, :, :]).to(device)  # [B, 1, H, W]
    pred_ys = torch.from_numpy(np.array(pred_ys_denorm)[:, np.newaxis, :, :]).to(device)  # [B, 1, H, W]

    return true_ys, pred_ys, masks


def main():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.mps.is_available():
        device = 'mps'
    print(f"Device: {device}")

    scheduler = build_scheduler()
    model_save_path = PROJ_PATH + "/Checkpoint/RSImageDownscale/Diffusers"

    lat_min = 33.0
    lat_max = 49.0
    lon_min = -120.0
    lon_max = -103.0

    # Train on 64x64 data
    print("=" * 60)
    print("Training on 36km data...")
    print("=" * 60)

    full_dataset = RSImageDownscaleDataset()

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    model = NoisePredictor(
        input_channel_num=INPUT_CHANNEL_NUM,
        hidden_dim=128,
        timestep_emb_dim=64,
        pos_embed_dim=32,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
        channel_attention_reduction=16,
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
    true_ys, pred_ys, masks = predict(model, scheduler, full_dataset, device)

    # Evaluate with masks (only on valid pixels)
    evaluate(true_ys, pred_ys, masks)


if __name__ == "__main__":
    main()
