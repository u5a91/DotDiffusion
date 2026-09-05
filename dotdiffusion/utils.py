# dotdiffusion/utils.py

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
import yaml
from torchvision.utils import save_image


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file and return it as a dictionary.
    """
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError("The config must contain a YAML mapping.")
    return config


def set_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    """
    Choose the appropriate device for training.
    """
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def save_samples(images: torch.Tensor, path: str | Path, nrow: int = 4) -> None:
    """
    Save a batch of images to a file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Clamp images to [-1, 1], scale to [0, 1], and save as a grid on CPU
    images = ((images.clamp(-1.0, 1.0) + 1.0) / 2.0).cpu()
    save_image(images, path, nrow=nrow)
