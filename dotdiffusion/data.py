# dotdiffusion/data.py

from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class ImageFolderDataset(Dataset[torch.Tensor]):
    """
    Load all images below a directory for unconditional DDPM training.
    """

    def __init__(self, root: str | Path, image_size: int) -> None:
        self.root = Path(root)
        self.image_size = image_size

        if not self.root.is_dir():
            raise FileNotFoundError(f"Data directory does not exist: {self.root}")

        self.paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise ValueError(f"No supported images found in: {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image_file:
            image = image_file.convert("RGB")
            image = _center_crop_square(image)
            # Use the average color of each cell as its representative color.
            image = image.resize(
                (self.image_size, self.image_size),
                resample=Image.Resampling.BOX,
            )
            tensor = TF.pil_to_tensor(image).float() / 255.0
            tensor = tensor * 2.0 - 1.0
        return tensor


def _center_crop_square(image: Image.Image) -> Image.Image:
    side = min(image.width, image.height)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))
