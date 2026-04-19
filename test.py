"""
Standalone test entrypoint for StarGAN blur-to-sharp model.

Examples
--------
Single checkpoint:
    python test.py --checkpoint /kaggle/working/checkpoints/ckpt_ep10.pth

Compare latest 5 checkpoints:
    python test.py --checkpoint-dir /kaggle/working/checkpoints --last-k 5
"""

import argparse
import glob
import os

from config import Config
from trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate StarGAN checkpoints on test split")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pth")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory containing checkpoints (evaluates matching checkpoints)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="ckpt_ep*.pth",
        help="Glob pattern used with --checkpoint-dir",
    )
    parser.add_argument("--last-k", type=int, default=1, help="Evaluate only latest K checkpoints from sorted list")
    parser.add_argument("--batch-size", type=int, default=None, help="Override test batch size")
    parser.add_argument("--max-batches", type=int, default=None, help="Cap number of test batches for faster runs")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable W&B logging during test")
    return parser


def resolve_checkpoints(args) -> list[str]:
    if args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        return [args.checkpoint]

    if args.checkpoint_dir:
        pattern = os.path.join(args.checkpoint_dir, args.pattern)
        ckpts = sorted(glob.glob(pattern))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints matched: {pattern}")
        k = max(args.last_k, 1)
        return ckpts[-k:]

    raise ValueError("Provide either --checkpoint or --checkpoint-dir")


def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = Config()
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.max_batches is not None:
        cfg.test_max_batches = args.max_batches
    if args.disable_wandb:
        cfg.use_wandb = False

    trainer = Trainer(cfg)

    ckpt_paths = resolve_checkpoints(args)
    rows = []
    for ckpt_path in ckpt_paths:
        metrics = trainer.evaluate_test_checkpoint(ckpt_path)
        rows.append(
            (
                os.path.basename(ckpt_path),
                metrics["test/psnr"],
                metrics["test/ssim"],
                metrics["test/id"],
                metrics["test/perc"],
                metrics["test/rec"],
            )
        )

    print("\n=== Test Summary ===")
    print("checkpoint\tPSNR\tSSIM\tID\tPERC\tREC")
    for r in rows:
        print(f"{r[0]}\t{r[1]:.3f}\t{r[2]:.4f}\t{r[3]:.4f}\t{r[4]:.4f}\t{r[5]:.4f}")


if __name__ == "__main__":
    main()
