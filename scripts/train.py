# scripts/train.py

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dotdiffusion import GaussianDiffusion, UNet
from dotdiffusion.data import ImageFolderDataset
from dotdiffusion.utils import choose_device, load_config, save_samples, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the minimal DotDiffusion model."
    )
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_config(parse_args().config)
    set_seed(config["seed"])
    device = choose_device(config["device"])

    dataset = ImageFolderDataset(config["data_dir"], config["image_size"])
    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=device.type == "cuda",
    )

    model = UNet(
        in_channels=3,
        base_channels=config["base_channels"],
        time_emb_dim=config["time_emb_dim"],
    ).to(device)
    diffusion = GaussianDiffusion(
        timesteps=config["timesteps"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    checkpoint_dir = Path(config["checkpoint_dir"])
    output_dir = Path(config["output_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"device={device} images={len(dataset)} parameters={sum(p.numel() for p in model.parameters()):,}"
    )

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        progress = tqdm(loader, desc=f"epoch {epoch}/{config['epochs']}")
        total_loss = 0.0

        for images in progress:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.training_loss(model, images)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.shape[0]
            progress.set_postfix(loss=f"{loss.item():.4f}")

        mean_loss = total_loss / len(dataset)
        print(f"epoch={epoch} mean_loss={mean_loss:.6f}")

        if epoch % config["save_every"] == 0 or epoch == config["epochs"]:
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
            }
            torch.save(checkpoint, checkpoint_dir / "model_latest.pt")
            torch.save(checkpoint, checkpoint_dir / f"model_epoch_{epoch:04d}.pt")

            model.eval()
            samples = diffusion.sample(
                model,
                shape=(
                    config["sample_count"],
                    3,
                    config["image_size"],
                    config["image_size"],
                ),
                device=device,
            )
            assert isinstance(samples, torch.Tensor)
            save_samples(samples, output_dir / f"epoch_{epoch:04d}.png")


if __name__ == "__main__":
    main()
