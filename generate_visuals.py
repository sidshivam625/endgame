"""
generate_visuals.py  –  Paper-quality visualisation grid from a saved checkpoint.

Layout (one row per sample):
  [Target GT] [Input Blur] [Reconst (src attr)] [Black Hair] [Blond Hair] [Brown Hair] [Male] [Young]

Usage
-----
    python generate_visuals.py --checkpoint /kaggle/working/checkpoints/ckpt_best_psnr.pth
    python generate_visuals.py --checkpoint /path/to/ckpt.pth --samples 8 --split test
"""

import os
import argparse

import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe inside Kaggle / SSH)
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config
from dataset import build_dataloaders
from models import Generator


def denorm(t: torch.Tensor):
    """[-1, 1] → [0, 1] numpy array (H, W, 3)."""
    return ((t * 0.5 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy())


def generate_paper_visuals(
    checkpoint_path: str,
    num_samples: int = 5,
    split: str = "test",
):
    cfg = Config()
    cfg.batch_size = num_samples
    device = cfg.device

    # ── Load data ─────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[split]
    batch  = next(iter(loader))

    image_clean = batch["clean"]
    image_blur  = batch["blurred"]
    label_org   = batch["attr"].to(device)
    image_blur  = image_blur.to(device)

    # ── Load Generator ─────────────────────────────────────────────────────────
    G = Generator(
        image_size=cfg.image_size,
        n_attrs=cfg.n_attrs,
        conv_dim=cfg.g_conv_dim,
        repeat_num=cfg.g_repeat_num,
    ).to(device)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(ckpt["G"])          # key is 'G', not 'G_state_dict'
    G.eval()

    # ── Generate translations ──────────────────────────────────────────────────
    attr_names = cfg.selected_attrs       # e.g. ["Black_Hair", "Blond_Hair", …]

    with torch.no_grad():
        visuals_cpu = [image_clean, image_blur.cpu()]

        # 1. Direct reconstruction (keep source attr, let model sharpen)
        rec = G(image_blur, label_org)
        visuals_cpu.append(rec.cpu())

        # 2. One-attribute flips
        for i in range(cfg.n_attrs):
            label_trg = label_org.clone()
            label_trg[:, i] = 1.0 - label_trg[:, i]    # flip bit i
            fake = G(image_blur, label_trg)
            visuals_cpu.append(fake.cpu())

    # ── Plot ──────────────────────────────────────────────────────────────────
    col_labels = ["GT (clean)", "Input (blur)", "Recon (sharp)"] + [f"Flip: {n}" for n in attr_names]
    n_cols = len(visuals_cpu)
    n_rows = num_samples

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.8, n_rows * 2.8))
    if n_rows == 1:
        axes = [axes]

    for r in range(n_rows):
        for c in range(n_cols):
            ax  = axes[r][c]
            img = denorm(visuals_cpu[c][r])
            ax.imshow(img)
            ax.axis("off")
            if r == 0:
                ax.set_title(col_labels[c], fontsize=9, fontweight="bold")

    os.makedirs(cfg.sample_dir, exist_ok=True)
    out_name = f"paper_comparison_{os.path.splitext(os.path.basename(checkpoint_path))[0]}.png"
    out_path = os.path.join(cfg.sample_dir, out_name)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Paper-ready visualisation saved: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper-quality visuals from a checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--samples",    type=int, default=5,    help="Number of rows (samples)")
    parser.add_argument("--split",      type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    generate_paper_visuals(args.checkpoint, args.samples, args.split)
