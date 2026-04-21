"""
dataset.py  –  RAF-DB loader that produces plain StarGAN samples.

Pipeline per sample
───────────────────
    1. Load RAF-DB image → resize 128×128
    2. Optional horizontal flip (train split)
    3. ToTensor + Normalize  →  image  ∈ [-1, 1]
    4. Read expression label and convert to one-hot target vector

Supported layouts
─────────────────
1) Official label-file layout:
   root/Image/aligned/*.jpg
   root/EmoLabel/list_patition_label.txt

2) Folder layout (common Kaggle repack):
   root/dataset/train/<class_name>/*
   root/dataset/test/<class_name>/*
"""

import os
import random
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMG_EXTS


def _norm_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


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
        samples: list,
        selected_attrs: list,
        image_size: int = 128,
        augment: bool = True,
    ):
        super().__init__()
        self.selected_attrs = selected_attrs
        self.image_size   = image_size
        self.augment      = augment
        self.samples = samples

        # ── Base transforms ────────────────────────────────────────────────
        self.base_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_idx, fname = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        # Random horizontal flip
        if self.augment and random.random() > 0.5:
            img = TF.hflip(img)

        image = self.base_transform(img)   # (3, H, W) ∈ [-1, 1]

        # ── One-hot expression vector ──────────────────────────────────────
        attr = torch.zeros(len(self.selected_attrs), dtype=torch.float32)
        if 0 <= label_idx < len(self.selected_attrs):
            attr[label_idx] = 1.0

        return {
            "image"    : image,
            "clean"    : image.clone(),
            "attr"     : attr,
            "filename" : fname,
        }


# ── Factory function ──────────────────────────────────────────────────────────


def _load_samples_from_label_file(image_dir: str, attr_path: str) -> tuple[list, list]:
    train_samples = []
    test_samples = []

    with open(attr_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue

            fname = parts[0]
            label_idx = int(parts[1]) - 1  # RAF-DB 1..7 -> 0..6
            full_path = os.path.join(image_dir, fname)
            if not os.path.isfile(full_path):
                continue

            item = (full_path, label_idx, fname)
            if fname.startswith("train"):
                train_samples.append(item)
            elif fname.startswith("test"):
                test_samples.append(item)

    if not train_samples or not test_samples:
        raise FileNotFoundError(
            "Could not build train/test splits from label file. "
            "Expected filenames to start with 'train' and 'test'."
        )
    return train_samples, test_samples


def _load_samples_from_folder_layout(train_dir: str, test_dir: str, selected_attrs: list) -> tuple[list, list]:
    selected_norm = [_norm_name(a) for a in selected_attrs]

    def _class_to_index(class_name: str, fallback_map: dict) -> int | None:
        n = _norm_name(class_name)
        if n in selected_norm:
            return selected_norm.index(n)
        return fallback_map.get(class_name)

    train_classes = [d for d in sorted(os.listdir(train_dir)) if os.path.isdir(os.path.join(train_dir, d))]
    test_classes = [d for d in sorted(os.listdir(test_dir)) if os.path.isdir(os.path.join(test_dir, d))]
    class_names = sorted(set(train_classes) | set(test_classes))

    fallback_map = {}
    unresolved = [c for c in class_names if _norm_name(c) not in selected_norm]
    if unresolved:
        if len(class_names) == len(selected_attrs):
            # Deterministic fallback when class names differ but class count matches.
            fallback_map = {c: i for i, c in enumerate(class_names)}
            print("[Dataset] class-name mismatch with selected_attrs; using sorted fallback mapping:")
            for c, i in fallback_map.items():
                print(f"  - {c} -> {selected_attrs[i]}")
        else:
            raise ValueError(
                "Folder class names do not match selected_attrs and class counts differ.\n"
                f"Found classes: {class_names}\n"
                f"selected_attrs: {selected_attrs}"
            )

    def _collect(split_dir: str, split_prefix: str) -> list:
        samples = []
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            label_idx = _class_to_index(class_name, fallback_map)
            if label_idx is None:
                continue
            for fname in sorted(os.listdir(class_dir)):
                fpath = os.path.join(class_dir, fname)
                if os.path.isfile(fpath) and _is_image_file(fpath):
                    rel_name = f"{split_prefix}/{class_name}/{fname}"
                    samples.append((fpath, label_idx, rel_name))
        return samples

    train_samples = _collect(train_dir, "train")
    test_samples = _collect(test_dir, "test")

    if not train_samples or not test_samples:
        raise FileNotFoundError(
            "No images found under folder layout. Expected class subfolders inside train/ and test/."
        )
    return train_samples, test_samples

def _resolve_rafdb_paths(cfg):
    """Resolve RAF-DB paths from label-file or folder-based layouts."""
    root = cfg.data_root

    image_candidates = [
        os.path.join(root, "Image", "aligned"),
        os.path.join(root, "RAF-DB", "Image", "aligned"),
    ]
    label_candidates = [
        os.path.join(root, "EmoLabel", "list_patition_label.txt"),
        os.path.join(root, "RAF-DB", "EmoLabel", "list_patition_label.txt"),
    ]

    folder_train_candidates = [
        os.path.join(root, "dataset", "train"),
        os.path.join(root, "train"),
    ]
    folder_test_candidates = [
        os.path.join(root, "dataset", "test"),
        os.path.join(root, "test"),
    ]

    image_dir = next((p for p in image_candidates if os.path.isdir(p)), None)
    attr_path = next((p for p in label_candidates if os.path.isfile(p)), None)
    train_dir = next((p for p in folder_train_candidates if os.path.isdir(p)), None)
    test_dir = next((p for p in folder_test_candidates if os.path.isdir(p)), None)

    if image_dir is not None and attr_path is not None:
        # Ensure images are discoverable; catches wrong nested roots.
        if not glob.glob(os.path.join(image_dir, "*.jpg")):
            raise FileNotFoundError(f"No .jpg files found in RAF-DB image directory: {image_dir}")
        return {
            "mode": "label_file",
            "image_dir": image_dir,
            "attr_path": attr_path,
            "train_dir": None,
            "test_dir": None,
        }

    if train_dir is not None and test_dir is not None:
        return {
            "mode": "folder",
            "image_dir": None,
            "attr_path": None,
            "train_dir": train_dir,
            "test_dir": test_dir,
        }

    raise FileNotFoundError(
        f"Could not resolve RAF-DB dataset layout under: {root}\n"
        "Supported layouts:\n"
        "1) Label-file layout:\n"
        f"   - {image_candidates[0]} + {label_candidates[0]}\n"
        f"   - {image_candidates[1]} + {label_candidates[1]}\n"
        "2) Folder layout:\n"
        f"   - {folder_train_candidates[0]} + {folder_test_candidates[0]}\n"
        f"   - {folder_train_candidates[1]} + {folder_test_candidates[1]}"
    )


def build_dataloaders(cfg, worker_init_fn=None, generator=None) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, val, and test DataLoaders from a Config object."""

    paths = _resolve_rafdb_paths(cfg)
    cfg.image_dir = paths.get("image_dir")
    cfg.attr_path = paths.get("attr_path")

    if paths["mode"] == "label_file":
        train_samples, test_samples = _load_samples_from_label_file(
            paths["image_dir"], paths["attr_path"]
        )
    else:
        cfg.image_dir = paths["train_dir"]
        cfg.attr_path = "<folder-layout>"
        train_samples, test_samples = _load_samples_from_folder_layout(
            paths["train_dir"], paths["test_dir"], cfg.selected_attrs
        )

    # Deterministic split: hold out validation from training partition.
    n_train = len(train_samples)
    n_val = int(round(n_train * getattr(cfg, "val_split_ratio", 0.1)))
    n_val = max(1, min(n_val, n_train - 1)) if n_train > 1 else 0

    train_split = train_samples[: n_train - n_val]
    val_split = train_samples[n_train - n_val :]

    train_ds = RAFDBDataset(
        samples=train_split,
        selected_attrs=cfg.selected_attrs,
        image_size=cfg.image_size,
        augment=True,
    )
    val_ds = RAFDBDataset(
        samples=val_split,
        selected_attrs=cfg.selected_attrs,
        image_size=cfg.image_size,
        augment=False,
    )
    test_ds = RAFDBDataset(
        samples=test_samples,
        selected_attrs=cfg.selected_attrs,
        image_size=cfg.image_size,
        augment=False,
    )

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
