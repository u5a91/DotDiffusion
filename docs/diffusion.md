# Gaussian Diffusion

This document summarizes the minimal DDPM implementation in `dotdiffusion/diffusion.py`.

The `GaussianDiffusion` class implements the diffusion process itself:

- noise schedule
- forward noising process $q(x_t \mid x_0)$
- training loss for epsilon prediction
- reverse sampling process $p_\theta(x_{t-1} \mid x_t)$

The denoising neural network is not defined here.
It is passed from outside and is expected to have the following interface:

$$
\epsilon_\theta(x_t, t) \approx \epsilon
$$

In code:

```python
pred_noise = model(x_t, t)
```

For conditional generation, the model may also accept an additional condition:

```python
pred_noise = model(x_t, t, cond)
```

## Notation

Let $T$ be the number of diffusion timesteps.

The implementation uses a linear beta schedule:

$$
\beta_t
=
\beta_{\text{start}}
+
\frac{t}{T - 1}
(\beta_{\text{end}} - \beta_{\text{start}})
\quad
(t = 0, \ldots, T-1).
$$

Then define:

$$
\alpha_t = 1 - \beta_t
$$

and

$$
\bar{\alpha}_t = \prod_{s=0}^{t} \alpha_s.
$$

The code stores these values as buffers:

```python
self.betas
self.alphas
self.alpha_bars
```

Since they are registered as buffers, they automatically move to the same device as the module when calling `.to(device)`.

## Forward process

The forward process gradually adds Gaussian noise to a clean image $x_0$.

The closed-form distribution is:

$$
q(x_t \mid x_0)
=
\mathcal{N} \left(\sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) I\right).
$$

Equivalently, we can sample $x_t$ as:

$$
x_t
=
\sqrt{\bar{\alpha}_t} x_0
+
\sqrt{1 - \bar{\alpha}_t} \epsilon,
\quad
\epsilon \sim \mathcal{N}(0, I).
$$

This corresponds to:

```python
GaussianDiffusion.q_sample(x0, t, noise)
```

where:

- `x0` has shape `[B, C, H, W]`
- `t` has shape `[B]`
- `noise` has the same shape as `x0`

The helper function `_extract` is used to select timestep-dependent coefficients and reshape them to `[B, 1, 1, 1]`, so they can be broadcast over image tensors.

## Training objective

The implementation uses the standard DDPM epsilon-prediction objective.

A random timestep $t$ is sampled for each image in the batch.
Then noise $\epsilon$ is sampled, and $x_t$ is constructed by the forward process.

The model is trained to predict the added noise:

$$
\mathcal{L}
=
\mathbb{E}_{x_0, t, \epsilon}
\left[\left\|\epsilon - \epsilon_\theta(x_t, t)\right\|_2^2\right].
$$

This corresponds to:

```python
GaussianDiffusion.training_loss(model, x0, cond=None)
```

The expected model interface is:

```python
model(x_t, t)
```

or, for conditional generation:

```python
model(x_t, t, cond)
```

The returned value must have the same shape as `x0`.

## Reverse process

Sampling starts from pure Gaussian noise:

$$
x_{T - 1} \sim \mathcal{N}(0, I).
$$

Then the model repeatedly applies reverse denoising steps:

$$
p_\theta(x_{t-1} \mid x_t).
$$

The implementation uses the DDPM mean parameterization:

$$
\mu_\theta(x_t, t)
=
\frac{1}{\sqrt{\alpha_t}}
\left(
x_t
-
\frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}
\epsilon_\theta(x_t, t)
\right).
$$

Then one reverse step is sampled as:

$$
x_{t-1}
=
\mu_\theta(x_t, t)
+
\sigma_t z,
\quad
z \sim \mathcal{N}(0, I).
$$

The variance is currently set to the posterior variance:

$$
\sigma_t^2
=
\tilde{\beta}_t
=
\beta_t
\frac{1 - \bar{\alpha}_{t-1}}
{1 - \bar{\alpha}_t}.
$$

This corresponds to:

```python
GaussianDiffusion.p_sample(model, xt, t, cond=None)
```

When $t = 0$, the implementation returns the mean directly without adding additional noise.

## Full sampling

The full sampling loop is implemented by:

```python
GaussianDiffusion.sample(
    model,
    shape,
    device,
    cond=None,
    return_all_steps=False,
)
```

Example:

```python
import torch
from dotdiffusion import GaussianDiffusion, UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet(in_channels=3, base_channels=64).to(device)
diffusion = GaussianDiffusion(timesteps=1000).to(device)

samples = diffusion.sample(
    model=model,
    shape=(16, 3, 32, 32),
    device=device,
)
```

The returned tensor has shape:

```text
[B, C, H, W]
```

If `return_all_steps=True`, the method returns a list of intermediate samples from $x_T$ to $x_0$.

## Shape conventions

The current implementation assumes image tensors with shape:

```text
[B, C, H, W]
```

where:

- `B`: batch size
- `C`: number of channels
- `H`: height
- `W`: width

Timesteps have shape:

```text
[B]
```

Each image in a batch can use a different timestep during training.

## Notes

This is a minimal implementation intended for learning and experimentation.

Current limitations:

- only linear beta schedule is implemented
- only epsilon prediction is implemented
- no classifier-free guidance yet
- no DDIM sampling yet
- no learned variance
- no clipping or dynamic thresholding during sampling
- conditional generation is only supported through an optional `cond` argument passed to the model

Future extensions may include:

- cosine noise schedule
- DDIM sampler
- classifier-free guidance
- text conditioning
- image-to-image conditioning
- discrete diffusion for palette-based pixel art

## References

- Jonathan Ho, Ajay Jain, Pieter Abbeel.  
  *Denoising Diffusion Probabilistic Models*. NeurIPS 2020.  
  https://arxiv.org/abs/2006.11239

- Hugging Face Blog.  
  *The Annotated Diffusion Model*.  
  https://huggingface.co/blog/annotated-diffusion
