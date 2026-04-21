"""
generate_visuals.py  –  Paper-quality visualisation grid from a saved checkpoint.

Layout (one row per sample):
    [Input image] [Recon (src attr)] [Target attr-1] [Target attr-2] ...

Supports both multilabel (CelebA-like) and multiclass (RAF-DB) targets.
"""

import os
import argparse

import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe inside Kaggle / SSH)
import matplotlib.pyplot as plt

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

    image_in    = batch["image"]
    label_org   = batch["attr"].to(device)
    image_in    = image_in.to(device)

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
        visuals_cpu = [image_in.cpu()]

        # 1. Direct reconstruction (keep source attr, let model sharpen)
        rec = G(image_in, label_org)
        visuals_cpu.append(rec.cpu())

        # 2. Target-attribute translations
        for i in range(cfg.n_attrs):
            if getattr(cfg, "attr_mode", "multilabel") == "multiclass":
                label_trg = torch.zeros_like(label_org)
                label_trg[:, i] = 1.0
            else:
                label_trg = label_org.clone()
                label_trg[:, i] = 1.0 - label_trg[:, i]    # flip bit i
            fake = G(image_in, label_trg)
            visuals_cpu.append(fake.cpu())

    # ── Plot ──────────────────────────────────────────────────────────────────
    if getattr(cfg, "attr_mode", "multilabel") == "multiclass":
        attr_cols = [f"Target: {n}" for n in attr_names]
    else:
        attr_cols = [f"Flip: {n}" for n in attr_names]
    col_labels = ["Input", "Recon"] + attr_cols
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
