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
### Cell 1 – Install
!pip install facenet-pytorch wandb --quiet

### Cell 2 – Verify GPU
import torch
print(torch.cuda.get_device_name(0))
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

### Cell 2b – W&B login (optional)
import wandb
# wandb.login(key="YOUR_WANDB_KEY")

### Cell 3 – Clone / upload project files
# If running from Kaggle dataset, copy files from /kaggle/input/your-code-dataset/
import shutil, os
for f in ["config.py","dataset.py","models.py","losses.py","trainer.py","train.py"]:
    shutil.copy(f"/kaggle/input/stargan-cid/{f}", f"/kaggle/working/{f}")
os.chdir("/kaggle/working")

### Cell 4 – Quick sanity-check (optional)
from models import Generator, Discriminator, count_params
import torch
G = Generator(128, 5).cuda()
D = Discriminator(128, 5).cuda()
count_params(G, "G")
count_params(D, "D")
x = torch.randn(2,3,128,128).cuda()
a = torch.zeros(2,5).cuda()
print(G(x,a).shape)

### Cell 5 – Train
from config  import Config
from trainer import Trainer

cfg = Config()
trainer = Trainer(cfg)
trainer.train()

### Cell 6 – Test from any checkpoint
from config import Config
from trainer import Trainer

cfg = Config()
trainer = Trainer(cfg)
trainer.evaluate_test_checkpoint("/kaggle/working/checkpoints/ckpt_ep10.pth")

### Cell 7 – Visualise final samples
from IPython.display import Image
import os
samples = sorted(os.listdir(cfg.sample_dir))
Image(os.path.join(cfg.sample_dir, samples[-1]))

### Cell 8 – Tiny overfit sanity check (quick debug)
from config import Config
from trainer import Trainer

cfg = Config()
cfg.use_wandb = False
trainer = Trainer(cfg)
trainer.overfit_sanity(n_samples=8, n_steps=300)
"""
