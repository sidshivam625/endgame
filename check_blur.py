"""
check_blur.py  –  Visual sanity-check: are the generated blur levels appropriate?

Saves a side-by-side grid of (Original Clean | Blurred Input) from the
training set so you can eyeball the sigma range before committing to a full run.

Usage
-----
    python check_blur.py
    python check_blur.py --samples 8 --split val
"""

import os
import argparse

import matplotlib
matplotlib.use("Agg")   # safe for Kaggle / headless environments
import matplotlib.pyplot as plt

from config import Config
from dataset import build_dataloaders


def visualize_blur(num_samples: int = 4, split: str = "train"):
    cfg = Config()
    cfg.batch_size = num_samples

    print(f"Loading dataset from: {cfg.celeba_root}")
    try:
        train_loader, val_loader, test_loader = build_dataloaders(cfg)
    except FileNotFoundError as exc:
        print(f"Error loading dataset: {exc}")
        return

    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[split]
    batch  = next(iter(loader))

    images_clean = batch["clean"]
    images_blur  = batch["blurred"]
    attrs        = batch["attr"]

    def to_img(t):
        """(C, H, W) tensor in [-1, 1] → (H, W, C) numpy in [0, 1]."""
        return (t * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
    fig.suptitle(
        f"Blur sanity check  |  σ ∈ [{cfg.blur_sigma_lo}, {cfg.blur_sigma_hi}]  "
        f"kernel={cfg.blur_kernel}",
        fontsize=12, fontweight="bold",
    )
    attr_names = cfg.selected_attrs

    for i in range(num_samples):
        attr_str = " ".join(
            attr_names[j] for j in range(len(attr_names)) if attrs[i, j] > 0.5
        ) or "none"

        # Row 0: original clean
        axes[0, i].imshow(to_img(images_clean[i]))
        axes[0, i].set_title(f"Clean\n{attr_str}", fontsize=7)
        axes[0, i].axis("off")

        # Row 1: blurred input
        axes[1, i].imshow(to_img(images_blur[i]))
        axes[1, i].set_title("Blurred Input", fontsize=7)
        axes[1, i].axis("off")

    os.makedirs(cfg.sample_dir, exist_ok=True)
    out_path = os.path.join(cfg.sample_dir, "blur_check.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nVisualization saved: {out_path}")
    print(
        f"Inspect this image to confirm the blur level (σ={cfg.blur_sigma_lo}–{cfg.blur_sigma_hi}) "
        f"is appropriate for your task.\n"
        f"Adjust blur_sigma_lo / blur_sigma_hi in config.py if needed."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check blur augmentation visually")
    parser.add_argument("--samples", type=int, default=4,     help="Number of samples to show")
    parser.add_argument("--split",   type=str, default="train", choices=["train", "val", "test"])
    args = parser.parse_args()
    visualize_blur(args.samples, args.split)
