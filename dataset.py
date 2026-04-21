"""
dataset.py  –  RAF-DB loader that produces plain StarGAN samples.

Pipeline per sample
───────────────────
    1. Load aligned RAF-DB JPEG → resize 128×128
    2. Optional horizontal flip (train split)
    3. ToTensor + Normalize  →  image  ∈ [-1, 1]
    4. Read expression label and convert to one-hot target vector
"""

import os
import random
import glob
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


# ── RAF-DB dataset ───────────────────────────────────────────────────────────

class RAFDBDataset(Dataset):
    """
    Returns a dict with keys:
        'image'    : (3, H, W) float32  – model input image
        'clean'    : (3, H, W) float32  – reference image (same tensor for compatibility)
        'attr'     : (n_attrs,) float32 – binary attribute vector  {0, 1}
        'filename' : str
    """

    def __init__(
        self,
        image_dir: str,
        attr_path: str,
        selected_attrs: list,
        image_size: int = 128,
        split: str = "train",        # "train" | "val" | "test"
        augment: bool = True,
        val_split_ratio: float = 0.1,
    ):
        super().__init__()
        self.image_dir    = image_dir
        self.selected_attrs = selected_attrs
        self.image_size   = image_size
        self.augment      = augment

        # ── Load RAF-DB label file ────────────────────────────────────────
        rows = []
        with open(attr_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                fname = parts[0]
                # RAF-DB labels are 1..7; convert to 0..6.
                label_idx = int(parts[1]) - 1
                rows.append((fname, label_idx))

        df = pd.DataFrame(rows, columns=["filename", "label_idx"])

        # RAF-DB partition names are encoded in filename prefixes.
        train_df = df[df["filename"].str.startswith("train")].reset_index(drop=True)
        test_df = df[df["filename"].str.startswith("test")].reset_index(drop=True)

        # Deterministic split: hold out validation from RAF train partition.
        n_train = len(train_df)
        n_val = int(round(n_train * val_split_ratio))
        n_val = max(1, min(n_val, n_train - 1)) if n_train > 1 else 0

        if split == "train":
            df_split = train_df.iloc[: n_train - n_val]
        elif split == "val":
            df_split = train_df.iloc[n_train - n_val :]
        else:
            df_split = test_df

        self.df = df_split.reset_index(drop=True)

        # ── Base transforms ────────────────────────────────────────────────
        self.base_transform = T.Compose([
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

        # Random horizontal flip
        if self.augment and random.random() > 0.5:
            img = TF.hflip(img)

        image = self.base_transform(img)   # (3, H, W) ∈ [-1, 1]

        # ── One-hot expression vector ──────────────────────────────────────
        attr = torch.zeros(len(self.selected_attrs), dtype=torch.float32)
        label_idx = int(row["label_idx"])
        if 0 <= label_idx < len(self.selected_attrs):
            attr[label_idx] = 1.0

        return {
            "image"    : image,
            "clean"    : image.clone(),
            "attr"     : attr,
            "filename" : fname,
        }


# ── Factory function ──────────────────────────────────────────────────────────

def _resolve_rafdb_paths(cfg):
    """Resolve RAF-DB image dir and label file from common layouts."""
    root = cfg.data_root

    image_candidates = [
        os.path.join(root, "Image", "aligned"),
        os.path.join(root, "RAF-DB", "Image", "aligned"),
    ]
    label_candidates = [
        os.path.join(root, "EmoLabel", "list_patition_label.txt"),
        os.path.join(root, "RAF-DB", "EmoLabel", "list_patition_label.txt"),
    ]

    image_dir = next((p for p in image_candidates if os.path.isdir(p)), None)
    attr_path = next((p for p in label_candidates if os.path.isfile(p)), None)

    if image_dir is None:
        raise FileNotFoundError(
            f"RAF-DB image directory not found under: {root}\n"
            "Expected one of:\n"
            f"  - {image_candidates[0]}\n"
            f"  - {image_candidates[1]}"
        )
    if attr_path is None:
        raise FileNotFoundError(
            f"RAF-DB label file not found under: {root}\n"
            "Expected one of:\n"
            f"  - {label_candidates[0]}\n"
            f"  - {label_candidates[1]}"
        )

    # Ensure images are discoverable; catches wrong nested roots.
    if not glob.glob(os.path.join(image_dir, "*.jpg")):
        raise FileNotFoundError(f"No .jpg files found in RAF-DB image directory: {image_dir}")

    return image_dir, attr_path


def build_dataloaders(cfg, worker_init_fn=None, generator=None) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, and test DataLoaders from a Config object."""

    image_dir, attr_path = _resolve_rafdb_paths(cfg)
    cfg.image_dir = image_dir
    cfg.attr_path = attr_path

    common = dict(
        image_dir     = image_dir,
        attr_path     = attr_path,
        selected_attrs= cfg.selected_attrs,
        image_size    = cfg.image_size,
        val_split_ratio = getattr(cfg, "val_split_ratio", 0.1),
    )

    train_ds = RAFDBDataset(**common, split="train", augment=True)
    val_ds   = RAFDBDataset(**common, split="val",   augment=False)
    test_ds  = RAFDBDataset(**common, split="test",  augment=False)

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
