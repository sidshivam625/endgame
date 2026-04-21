"""
train.py  –  Entry-point for training StarGAN + Contrastive Identity Loss.

Usage
──────
    python train.py --mode train
    python train.py --mode test --checkpoint /kaggle/working/checkpoints/ckpt_ep10.pth
    python train.py --mode overfit --overfit-steps 300 --overfit-samples 8

Or as a Kaggle notebook, paste the cells below in order.
"""

import argparse
import subprocess
import sys


# ── Step 0: install dependencies (run once) ────────────────────────────────────
# Uncomment in a notebook cell:
#
#   !pip install facenet-pytorch --quiet


def install_deps():
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "facenet-pytorch", "wandb", "lpips", "torchmetrics[image]", "--quiet"]
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test", "overfit"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--overfit-steps", type=int, default=300)
    parser.add_argument("--overfit-samples", type=int, default=8)
    parser.add_argument("--no-live-preview", action="store_true")
    parser.add_argument("--disable-wandb", action="store_true")
    args = parser.parse_args()

    from config import Config
    from trainer import Trainer

    cfg = Config()

    if args.epochs is not None:
        cfg.num_epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.disable_wandb:
        cfg.use_wandb = False
    if args.no_live_preview:
        cfg.live_preview = False
    if args.checkpoint:
        cfg.resume_ckpt = args.checkpoint

    # Optional overrides for quick debugging:
    # cfg.num_epochs = 1
    # cfg.batch_size = 4
    # cfg.log_step   = 10

    trainer = Trainer(cfg)
    if args.mode == "train":
        trainer.train()
    elif args.mode == "test":
        if not args.checkpoint:
            raise ValueError("--checkpoint is required when --mode test")
        # resume_ckpt in config handles loading the state
        trainer.evaluate_test_checkpoint(args.checkpoint)
    else:
        # Overfit mode
        trainer.overfit_sanity(
            n_samples=args.overfit_samples,
            n_steps=args.overfit_steps,
        )


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────────────────────────────────────
# KAGGLE NOTEBOOK  – copy these cells top-to-bottom
# ─────────────────────────────────────────────────────────────────────────────
"""
### Cell 1 – Install dependencies
!pip install -q -r /kaggle/input/stargan-cid/requirements.txt

### Cell 2 – Set up workspace and copy project files
import os
import shutil

CODE_ROOT = "/kaggle/input/stargan-cid"
WORK_ROOT = "/kaggle/working/endgame"
DATA_ROOT = "/kaggle/input/datasets/shuvoalok/raf-db-dataset"

os.environ["STARGAN_WORK_ROOT"] = WORK_ROOT
os.environ["STARGAN_DATA_ROOT"] = DATA_ROOT

os.makedirs(WORK_ROOT, exist_ok=True)

for filename in [
    "config.py",
    "dataset.py",
    "models.py",
    "losses.py",
    "trainer.py",
    "train.py",
    "generate_visuals.py",
    "requirements.txt",
]:
    source_path = os.path.join(CODE_ROOT, filename)
    if os.path.exists(source_path):
        shutil.copy(source_path, os.path.join(WORK_ROOT, filename))

os.chdir(WORK_ROOT)
print("Working directory:", os.getcwd())
print("Dataset root:", DATA_ROOT)

### Cell 3 – Verify GPU and dataset paths
import torch
from config import Config

cfg = Config()
print("GPU:", torch.cuda.get_device_name(0))
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("Data root:", cfg.data_root)
print("Image dir:", cfg.image_dir)
print("Attr file:", cfg.attr_path)
print("Sample dir:", cfg.sample_dir)
print("Save dir:", cfg.save_dir)

### Cell 4 – Optional W&B login
import wandb
# wandb.login(key="YOUR_WANDB_KEY")

### Cell 5 – Quick model sanity check
from models import Generator, Discriminator, count_params

G = Generator(image_size=cfg.image_size, n_attrs=cfg.n_attrs, conv_dim=cfg.g_conv_dim, repeat_num=cfg.g_repeat_num).cuda()
D = Discriminator(image_size=cfg.image_size, n_attrs=cfg.n_attrs, conv_dim=cfg.d_conv_dim, repeat_num=cfg.d_repeat_num).cuda()
count_params(G, "G")
count_params(D, "D")
x = torch.randn(2, 3, cfg.image_size, cfg.image_size).cuda()
a = torch.zeros(2, cfg.n_attrs).cuda()
print("Generator output shape:", G(x, a).shape)

### Cell 6 – Tiny overfit sanity check
from trainer import Trainer

cfg = Config()
cfg.use_wandb = False
cfg.live_preview = False
trainer = Trainer(cfg)
trainer.overfit_sanity(n_samples=8, n_steps=300)

### Cell 7 – Train (save + W&B log samples, no inline preview)
from trainer import Trainer

cfg = Config()
cfg.live_preview = False
cfg.use_wandb = True
cfg.wandb_mode = "online"
trainer = Trainer(cfg)
trainer.train()

### Cell 8 – Inspect checkpoints and latest training outputs
import glob
from IPython.display import Image, display

print("Checkpoints:")
for path in sorted(glob.glob(os.path.join(cfg.save_dir, "*.pth"))):
    print(" -", os.path.basename(path))

print("Latest samples:")
for path in sorted(glob.glob(os.path.join(cfg.sample_dir, "*.png")))[-5:]:
    print(" -", os.path.basename(path))

latest_sample = sorted(glob.glob(os.path.join(cfg.sample_dir, "*.png")))
if latest_sample:
    display(Image(filename=latest_sample[-1]))

### Cell 9 – Evaluate a checkpoint on the test split
from trainer import Trainer

cfg = Config()
trainer = Trainer(cfg)
trainer.evaluate_test_checkpoint(os.path.join(cfg.save_dir, "ckpt_best_psnr.pth"))

### Cell 10 – Evaluate all key checkpoints
from trainer import Trainer

cfg = Config()
trainer = Trainer(cfg)

for ckpt_name in [
    "ckpt_best_psnr.pth",
    "ckpt_best_fid.pth",
    "ckpt_best_lpips.pth",
    "ckpt_final.pth",
]:
    ckpt_path = os.path.join(cfg.save_dir, ckpt_name)
    if os.path.exists(ckpt_path):
        print("\nEvaluating", ckpt_path)
        trainer.evaluate_test_checkpoint(ckpt_path)

### Cell 11 – Generate paper-quality visual comparison from a checkpoint
from generate_visuals import generate_paper_visuals

generate_paper_visuals(
    checkpoint_path=os.path.join(cfg.save_dir, "ckpt_best_psnr.pth"),
    num_samples=5,
    split="test",
)
"""
