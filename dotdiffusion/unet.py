# dotdiffusion/unet.py

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int) -> int:
    """
    Choose a reasonable number of groups for GroupNorm.
    """
    for g in [32, 16, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal timestep embedding.
    Input:
        t: [B] integer timesteps
    Output:
        emb: [B, dim]
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim > 1:
            t = t.view(t.shape[0])

        half_dim = self.dim // 2
        device = t.device

        if half_dim == 0:
            return t.float().unsqueeze(-1)

        scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -scale)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)

        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class ResidualBlock(nn.Module):
    """
    A minimal ResNet-style block with timestep conditioning.
    """

    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int) -> None:
        super().__init__()

        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))

        time_term = self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_term

        h = self.conv2(self.act2(self.norm2(h)))

        return h + self.shortcut(x)


class Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """
    A small UNet for DDPM-style noise prediction.

    Args:
        in_channels: image channels, usually 3
        base_channels: width of the network
        time_emb_dim: timestep embedding dimension
        cond_dim: optional conditioning dimension for future text/image conditioning

    Forward:
        x: [B, C, H, W]
        t: [B]
        cond: optional [B, cond_dim]
    Returns:
        predicted noise with same shape as x
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        time_emb_dim: int = 256,
        cond_dim: int | None = None,
    ) -> None:
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.cond_proj = (
            nn.Linear(cond_dim, time_emb_dim) if cond_dim is not None else None
        )

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Down path
        self.down1 = ResidualBlock(base_channels, base_channels, time_emb_dim)
        self.downsample1 = Downsample(base_channels)

        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.downsample2 = Downsample(base_channels * 2)

        # Bottleneck
        self.mid1 = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.mid2 = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)

        # Up path
        self.upsample1 = Upsample(base_channels * 2)
        self.up1 = ResidualBlock(base_channels * 4, base_channels, time_emb_dim)

        self.upsample2 = Upsample(base_channels)
        self.up2 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim)

        self.out_norm = nn.GroupNorm(_num_groups(base_channels), base_channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Timestep embedding
        time_emb = self.time_mlp(t)

        # Optional conditioning: just add it into the time embedding for now
        if cond is not None:
            if self.cond_proj is None:
                raise ValueError(
                    "cond was provided, but this UNet was created with cond_dim=None"
                )
            time_emb = time_emb + self.cond_proj(cond)

        # Initial conv
        x = self.init_conv(x)

        # Down
        h1 = self.down1(x, time_emb)  # [B, base, H, W]
        x = self.downsample1(h1)  # [B, base, H/2, W/2]

        h2 = self.down2(x, time_emb)  # [B, 2*base, H/2, W/2]
        x = self.downsample2(h2)  # [B, 2*base, H/4, W/4]

        # Middle
        x = self.mid1(x, time_emb)
        x = self.mid2(x, time_emb)

        # Up
        x = self.upsample1(x)  # [B, 2*base, H/2, W/2]
        x = torch.cat([x, h2], dim=1)  # [B, 4*base, H/2, W/2]
        x = self.up1(x, time_emb)  # [B, base, H/2, W/2]

        x = self.upsample2(x)  # [B, base, H, W]
        x = torch.cat([x, h1], dim=1)  # [B, 2*base, H, W]
        x = self.up2(x, time_emb)  # [B, base, H, W]

        # Output
        x = self.out_conv(self.out_act(self.out_norm(x)))
        return x


# Optional alias if you prefer a more explicit name
SimpleUNet = UNet
