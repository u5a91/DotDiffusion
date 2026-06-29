# dotdiffusion/diffusion.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """
    Extract values from a 1-D tensor `a` at indices `t`, then reshape so that
    it can be broadcast to match `x_shape`.

    Args:
        a: Tensor of shape [T]
        t: Tensor of shape [B] with timestep indices
        x_shape: shape of image tensor, e.g. [B, C, H, W]

    Returns:
        Tensor of shape [B, 1, 1, 1] (for 4D images)
    """
    b = t.shape[0]
    out = a.gather(0, t)
    return out.view(b, *([1] * (len(x_shape) - 1)))


class GaussianDiffusion(nn.Module):
    """
    Minimal Gaussian DDPM implementation.

    This class is responsible for:
    - diffusion schedule
    - forward noising q(x_t | x_0)
    - training loss for epsilon prediction
    - reverse sampling p(x_{t-1} | x_t)

    The denoising model itself is passed from outside:
        model(x_t, t) -> predicted noise
    """

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()

        if timesteps <= 0:
            raise ValueError("timesteps must be positive")
        if not (0.0 < beta_start < beta_end < 1.0):
            raise ValueError("Require 0 < beta_start < beta_end < 1")

        self.timesteps = timesteps

        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat(
            [torch.tensor([1.0], dtype=torch.float32), alpha_bars[:-1]], dim=0
        )

        # Useful precomputed terms
        sqrt_alpha_bars = torch.sqrt(alpha_bars)
        sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # Posterior variance:
        # q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)

        # Register as buffers so device placement is automatic
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars", sqrt_alpha_bars)
        self.register_buffer("sqrt_one_minus_alpha_bars", sqrt_one_minus_alpha_bars)
        self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas)
        self.register_buffer("posterior_variance", posterior_variance)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sample x_t from q(x_t | x_0):
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t = _extract(self.sqrt_alpha_bars, t, x0.shape)
        sqrt_one_minus_alpha_bar_t = _extract(
            self.sqrt_one_minus_alpha_bars, t, x0.shape
        )

        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

    def training_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        DDPM epsilon-prediction loss:
            E_{x0, t, eps}[ ||eps - eps_theta(x_t, t)||^2 ]

        Args:
            model: denoising model
            x0: clean images, shape [B, C, H, W]
            cond: optional conditioning (unused in unconditional case)

        Returns:
            scalar loss
        """
        batch_size = x0.shape[0]
        device = x0.device

        t = torch.randint(0, self.timesteps, (batch_size,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)

        if cond is None:
            pred_noise = model(xt, t)
        else:
            pred_noise = model(xt, t, cond)

        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        xt: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sample one reverse step:
            x_{t-1} ~ p_theta(x_{t-1} | x_t)

        Using the DDPM mean parameterization:
            mu_theta(x_t, t) =
                1/sqrt(alpha_t) * (
                    x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_theta(x_t, t)
                )
        """
        betas_t = _extract(self.betas, t, xt.shape)
        sqrt_one_minus_alpha_bars_t = _extract(
            self.sqrt_one_minus_alpha_bars, t, xt.shape
        )
        sqrt_recip_alphas_t = _extract(self.sqrt_recip_alphas, t, xt.shape)
        posterior_variance_t = _extract(self.posterior_variance, t, xt.shape)

        if cond is None:
            pred_noise = model(xt, t)
        else:
            pred_noise = model(xt, t, cond)

        model_mean = sqrt_recip_alphas_t * (
            xt - betas_t * pred_noise / sqrt_one_minus_alpha_bars_t
        )

        # If t == 0, return the mean directly (no more noise)
        if torch.all(t == 0):
            return model_mean

        noise = torch.randn_like(xt)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int],
        device: torch.device | str,
        cond: torch.Tensor | None = None,
        return_all_steps: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Run the full reverse process starting from x_T ~ N(0, I).

        Args:
            model: denoising model
            shape: e.g. (B, C, H, W)
            device: torch device
            cond: optional conditioning
            return_all_steps: if True, return list of all intermediate x_t

        Returns:
            Final samples x_0, or a list of intermediate samples
        """
        x = torch.randn(shape, device=device)

        if return_all_steps:
            samples = [x]

        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t, cond=cond)

            if return_all_steps:
                samples.append(x)

        if return_all_steps:
            return samples
        return x