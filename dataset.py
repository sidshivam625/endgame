"""
dataset.py  –  CelebA loader that produces (blurred_input, clean_target, attributes).

Pipeline per sample
───────────────────
  1. Load JPEG → crop centre 178×178 → resize 128×128
  2. Random horizontal flip
  3. ToTensor + Normalize  →  clean_img  ∈ [-1, 1]
  4. Apply random Gaussian blur to clean_img  →  blurred_img
     (blur_sigma ∈ [blur_sigma_lo, blur_sigma_hi], kernel = blur_kernel)
  5. Read binary attribute vector from CSV

The generator receives blurred_img + target_attr and must produce
a sharp, attribute-conditioned face that is identity-consistent.
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ── Gaussian blur helper ───────────────────────────────────────────────────────

def gaussian_blur(img_tensor: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """
    Apply Gaussian blur to a CHW tensor in [-1, 1].
    Uses torchvision's functional gaussian_blur (available since 0.9).
    """
    # torchvision expects uint8 or float32 in [0, 1] — work in float32 directly
    # Shift to [0, 1] for the operation then shift back
    x = (img_tensor + 1.0) / 2.0                          # [-1,1] → [0,1]
    x = TF.gaussian_blur(x, kernel_size=[kernel_size, kernel_size], sigma=sigma)
    return x * 2.0 - 1.0                                  # back to [-1,1]


# ── Dataset ───────────────────────────────────────────────────────────────────

class CelebABlurDataset(Dataset):
    """
    Returns a dict with keys:
        'blurred'  : (3, H, W) float32  – degraded / blurry input
        'clean'    : (3, H, W) float32  – sharp reference (same image, no blur)
        'attr'     : (n_attrs,) float32 – binary attribute vector  {0, 1}
        'filename' : str
    """

    def __init__(
        self,
        image_dir: str,
        attr_path: str,
        selected_attrs: list,
        image_size: int = 128,
        blur_kernel: int = 21,
        blur_sigma_lo: float = 4.0,
        blur_sigma_hi: float = 8.0,
        split: str = "train",        # "train" | "val" | "test"
        augment: bool = True,
    ):
        super().__init__()
        self.image_dir    = image_dir
        self.selected_attrs = selected_attrs
        self.image_size   = image_size
        self.blur_kernel  = blur_kernel
        self.blur_sigma_lo = blur_sigma_lo
        self.blur_sigma_hi = blur_sigma_hi
        self.augment      = augment

        # ── Load attribute CSV ─────────────────────────────────────────────
        df = pd.read_csv(attr_path)

        # Rename first column to 'filename' if needed (CelebA has 'image_id')
        first_col = df.columns[0]
        if first_col != "filename":
            df = df.rename(columns={first_col: "filename"})

        # Convert {-1, 1} → {0, 1}  (some CelebA CSVs use -1/1)
        for col in selected_attrs:
            if col in df.columns:
                df[col] = ((df[col] + 1) // 2).clip(0, 1).astype(np.float32)

        # Official CelebA split: first 162 770 train, next 19 867 val, rest test
        if split == "train":
            df = df.iloc[:162_770]
        elif split == "val":
            df = df.iloc[162_770:162_770 + 19_867]
        else:
            df = df.iloc[162_770 + 19_867:]

        self.df = df.reset_index(drop=True)

        # ── Base transforms (no blur) ──────────────────────────────────────
        crop_size = 178   # CelebA images are 178×218
        self.base_transform = T.Compose([
            T.CenterCrop(crop_size),
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        fname    = row["filename"]
        img_path = os.path.join(self.image_dir, fname)

        img = Image.open(img_path).convert("RGB")

        # Random horizontal flip (keep consistent between clean & blurred)
        if self.augment and random.random() > 0.5:
            img = TF.hflip(img)

        clean = self.base_transform(img)   # (3, H, W) ∈ [-1, 1]

        # ── Apply random Gaussian blur ─────────────────────────────────────
        sigma   = random.uniform(self.blur_sigma_lo, self.blur_sigma_hi)
        blurred = gaussian_blur(clean.clone(), self.blur_kernel, sigma)

        # ── Attribute vector ───────────────────────────────────────────────
        attr = torch.tensor(
            [row[a] for a in self.selected_attrs], dtype=torch.float32
        )

        return {
            "blurred"  : blurred,
            "clean"    : clean,
            "attr"     : attr,
            "filename" : fname,
        }


# ── Factory function ──────────────────────────────────────────────────────────

def build_dataloaders(cfg, worker_init_fn=None, generator=None) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, and test DataLoaders from a Config object."""

    if not os.path.isdir(cfg.image_dir):
        raise FileNotFoundError(
            f"Image directory not found: {cfg.image_dir}\n"
            "Kaggle recommended flow: attach dataset 'jessicali9530/celeba-dataset'\n"
            "Expected mount root: /kaggle/input/celeba-dataset"
        )
    if not os.path.isfile(cfg.attr_path):
        raise FileNotFoundError(
            f"Attribute file not found: {cfg.attr_path}\n"
            "Expected file: /kaggle/input/celeba-dataset/list_attr_celeba.csv"
        )

    common = dict(
        image_dir     = cfg.image_dir,
        attr_path     = cfg.attr_path,
        selected_attrs= cfg.selected_attrs,
        image_size    = cfg.image_size,
        blur_kernel   = cfg.blur_kernel,
        blur_sigma_lo = cfg.blur_sigma_lo,
        blur_sigma_hi = cfg.blur_sigma_hi,
    )

    train_ds = CelebABlurDataset(**common, split="train", augment=True)
    val_ds   = CelebABlurDataset(**common, split="val",   augment=False)
    test_ds  = CelebABlurDataset(**common, split="test",  augment=False)

    loader_kwargs = {}
    if cfg.num_workers > 0:
        loader_kwargs["persistent_workers"] = cfg.persistent_workers
        loader_kwargs["prefetch_factor"] = cfg.prefetch_factor

    train_loader = DataLoader(
        train_ds,
        batch_size       = cfg.batch_size,
        shuffle          = True,
        num_workers      = cfg.num_workers,
        pin_memory       = cfg.pin_memory,
        drop_last        = True,
        worker_init_fn   = worker_init_fn,   # per-worker RNG seed (reproducibility)
        generator        = generator,        # controls shuffle order determinism
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg.batch_size,
        shuffle     = False,
        num_workers = cfg.num_workers,
        pin_memory  = cfg.pin_memory,
        drop_last   = False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = cfg.batch_size,
        shuffle     = False,
        num_workers = cfg.num_workers,
        pin_memory  = cfg.pin_memory,
        drop_last   = False,
        **loader_kwargs,
    )

    print(f"[Dataset] Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader
