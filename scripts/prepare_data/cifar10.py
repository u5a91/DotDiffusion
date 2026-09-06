# scripts/prepare_data/cifar10.py

from pathlib import Path

from torchvision.datasets import CIFAR10


ROOT_DIR = Path("data/cifar10")
OUTPUT_DIR = ROOT_DIR / "train"

TARGET_LABEL = 8    # ship


def main() -> None:
    dataset = CIFAR10(
        root=ROOT_DIR,
        train=True,
        download=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    saved_index = 0
    for image, label in dataset:
        if label != TARGET_LABEL:
            continue
        image.save(OUTPUT_DIR / f"{saved_index:05d}.png")
        saved_index += 1


if __name__ == "__main__":
    main()
