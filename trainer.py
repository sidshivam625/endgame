"""
trainer.py  –  Training / validation / checkpoint-test loop with W&B logging.

Metrics computed
────────────────
Per-batch pixel metrics  (evaluate_loader  – fast, every epoch):
  PSNR     Peak Signal-to-Noise Ratio              ↑ better
  SSIM     Structural Similarity (windowed)        ↑ better
  LPIPS    Perceptual patch similarity (AlexNet)   ↓ better
  rec      Cycle-reconstruction L1                 ↓ better
  id       FaceNet cosine distance                 ↓ better
  perc     VGG perceptual distance                 ↓ better

Distribution / GAN metrics  (compute_gan_metrics – periodic, heavier):
  FID      Fréchet Inception Distance              ↓ better
  IS       Inception Score   mean ± std            ↑ better
  KID      Kernel Inception Distance               ↓ better
  LPIPS    (also reported here per-distribution)   ↓ better

Best checkpoints saved automatically:
  ckpt_best_psnr.pth   – highest val PSNR
  ckpt_best_fid.pth    – lowest  val FID
  ckpt_best_lpips.pth  – lowest  val LPIPS
"""

import os
import time
import random
import csv

import numpy as np
import torch
import torch.optim as optim
from torch import amp
import torchvision.utils as vutils
from tqdm.auto import tqdm

from config import Config
from models import Generator, Discriminator, count_params
from losses import (
    FaceNetIdentityLoss,
    VGGPerceptualLoss,
    gradient_penalty,
    adv_d_loss,
    adv_g_loss,
    cls_loss_real,
    cls_loss_fake,
    cycle_loss,
)
from dataset import build_dataloaders

try:
    import wandb
except ImportError:
    wandb = None

# ── Optional: torchmetrics for GAN evaluation metrics ─────────────────────────
try:
    from torchmetrics.image import (
        FrechetInceptionDistance,
        InceptionScore,
        KernelInceptionDistance,
    )
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchmetrics.functional.image import structural_similarity_index_measure as tm_ssim
    TORCHMETRICS_OK = True
except ImportError:
    TORCHMETRICS_OK = False
    print(
        "[warn] torchmetrics or lpips not installed — FID/IS/KID/LPIPS unavailable.\n"
        "       Install with:  pip install 'torchmetrics[image]' lpips"
    )


# ─────────────────────────────── Utilities ────────────────────────────────────

def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch RNGs for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    """Per-DataLoader-worker RNG seeding (prevents identical augmentation across workers)."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def configure_runtime(cfg: Config):
    """Enable safe performance features for NVIDIA GPUs (Kaggle-friendly)."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = cfg.enable_tf32
        torch.backends.cudnn.allow_tf32 = cfg.enable_tf32
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def denorm(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1]  →  [0, 1]"""
    return (x + 1.0) / 2.0


def to_float01(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] float → [0, 1] float32, safe for torchmetrics FID/IS/KID."""
    return denorm(x).clamp(0.0, 1.0).float()


def batch_psnr(x_fake: torch.Tensor, x_target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """PSNR (dB) over a batch. Inputs in [-1, 1]."""
    x_f = denorm(x_fake)
    x_t = denorm(x_target)
    mse = torch.mean((x_f - x_t) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / (mse + eps))


def batch_lpips(x_fake, x_target, metric_fn):
    """LPIPS distance between two batches in [-1, 1]."""
    if metric_fn is None:
        return torch.tensor(0.0, device=x_fake.device)
    # torchmetrics lpips expects [0, 1] or [-1, 1] depending on net, 
    # but usually expects standard torch -1..1 for AlexNet internally.
    val = metric_fn(x_fake, x_target)
    return val


def batch_ssim(x_fake: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
    """
    Window-based SSIM (torchmetrics) – matches the paper-standard metric.
    Falls back to a fast global approximation when torchmetrics is absent.
    """
    if TORCHMETRICS_OK:
        # torchmetrics expects [0, 1] float, returns a scalar
        val = tm_ssim(to_float01(x_fake), to_float01(x_target), data_range=1.0)
        return val if not isinstance(val, torch.Tensor) else val.mean()
    return _batch_ssim_global(x_fake, x_target)


def _batch_ssim_global(x_fake: torch.Tensor, x_target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Global (non-windowed) SSIM – fallback only."""
    x = denorm(x_fake)
    y = denorm(x_target)
    xm = x.mean(dim=(1, 2, 3))
    ym = y.mean(dim=(1, 2, 3))
    xv = ((x - xm.view(-1, 1, 1, 1)) ** 2).mean(dim=(1, 2, 3))
    yv = ((y - ym.view(-1, 1, 1, 1)) ** 2).mean(dim=(1, 2, 3))
    cv = ((x - xm.view(-1, 1, 1, 1)) * (y - ym.view(-1, 1, 1, 1))).mean(dim=(1, 2, 3))
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * xm * ym + c1) * (2 * cv + c2)
    den = (xm ** 2 + ym ** 2 + c1) * (xv + yv + c2) + eps
    return (num / den).mean()


def save_sample_grid(
    G: Generator,
    x_blur: torch.Tensor,
    x_clean: torch.Tensor,
    fixed_attrs: list,
    path: str,
    device: torch.device,
    n_show: int = 8,
) -> torch.Tensor:
    """Save a visual grid: [blur | clean | trg_attr₁ | trg_attr₂ | ...]."""
    was_training = G.training
    G.eval()
    with torch.no_grad():
        x_b = x_blur[:n_show].to(device)
        x_c = x_clean[:n_show].to(device)
        imgs = [denorm(x_b), denorm(x_c)]
        for trg_attr in fixed_attrs:
            a = torch.tensor(trg_attr, dtype=torch.float32, device=device)
            a = a.unsqueeze(0).expand(n_show, -1)
            imgs.append(denorm(G(x_b, a)))
        grid = vutils.make_grid(torch.cat(imgs, dim=0), nrow=n_show, padding=2)
        vutils.save_image(grid, path)
    if was_training:
        G.train()
    return grid


def linear_lr_decay(
    optimizer: optim.Optimizer,
    current_step: int,
    total_steps: int,
    decay_start: int,
    base_lr: float,
):
    """Linear LR decay from base_lr → 0 over [decay_start, total_steps]."""
    if current_step >= decay_start:
        ratio = (current_step - decay_start) / max(total_steps - decay_start, 1)
        new_lr = max(base_lr * (1.0 - ratio), 0.0)
        for g in optimizer.param_groups:
            g["lr"] = new_lr


# ─────────────────────────────── Trainer ──────────────────────────────────────

class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_seed(cfg.seed)
        configure_runtime(cfg)

        os.makedirs(cfg.save_dir,   exist_ok=True)
        os.makedirs(cfg.sample_dir, exist_ok=True)

        # ── Models ────────────────────────────────────────────────────────────
        self.G = Generator(
            image_size=cfg.image_size,
            n_attrs=cfg.n_attrs,
            conv_dim=cfg.g_conv_dim,
            repeat_num=cfg.g_repeat_num,
        ).to(cfg.device)
        self.D = Discriminator(
            image_size=cfg.image_size,
            n_attrs=cfg.n_attrs,
            conv_dim=cfg.d_conv_dim,
            repeat_num=cfg.d_repeat_num,
        ).to(cfg.device)

        if cfg.use_channels_last:
            self.G = self.G.to(memory_format=torch.channels_last)
            self.D = self.D.to(memory_format=torch.channels_last)

        # Optional: torch.compile for ~15-25% extra throughput on PyTorch 2.x
        if getattr(cfg, "use_compile", False) and hasattr(torch, "compile"):
            print("[compile] Compiling G and D with torch.compile(mode='reduce-overhead')…")
            self.G = torch.compile(self.G, mode="reduce-overhead")
            self.D = torch.compile(self.D, mode="reduce-overhead")

        count_params(self.G, "Generator")
        count_params(self.D, "Discriminator")

        # ── Frozen auxiliary networks ──────────────────────────────────────────
        print("Loading FaceNet (frozen)…")
        self.face_loss = FaceNetIdentityLoss(
            pretrained=cfg.facenet_pretrained,
            device=cfg.device,
        )
        print("Loading VGG16 perceptual loss (frozen)…")
        self.perc_loss = VGGPerceptualLoss(
            layers=cfg.vgg_layers,
            weights=cfg.vgg_weights,
            device=cfg.device,
        )

        # ── Optimizers + AMP scalers ───────────────────────────────────────────
        self.opt_G = optim.Adam(self.G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
        self.opt_D = optim.Adam(self.D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))
        self.amp_enabled     = cfg.use_amp and cfg.device.type == "cuda"
        self.amp_device_type = "cuda" if cfg.device.type == "cuda" else "cpu"
        self.scaler_G = amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.scaler_D = amp.GradScaler("cuda", enabled=self.amp_enabled)

        # ── Quantitative metric objects ───────────────────────────────────────
        self.metric_lpips = None
        if TORCHMETRICS_OK:
            self.metric_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(cfg.device)

        # ── Data loaders (seeded workers) ─────────────────────────────────────
        _g = torch.Generator()
        _g.manual_seed(cfg.seed)
        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(
            cfg, worker_init_fn=seed_worker, generator=_g
        )
        steps_per_epoch   = len(self.train_loader)
        self.total_d_steps = max((steps_per_epoch * cfg.num_epochs) // cfg.n_critic, 1)
        self.decay_start   = int(self.total_d_steps * cfg.lr_decay_start_ratio)

        # Fixed target attributes for qualitative sample grids
        self.fixed_attrs = [
            [1, 0, 0, 0, 1],   # Black hair · Young
            [0, 1, 0, 0, 1],   # Blond hair · Young
            [0, 0, 1, 1, 1],   # Brown hair · Male · Young
            [1, 0, 0, 1, 0],   # Black hair · Male
        ]

        # ── State ─────────────────────────────────────────────────────────────
        self.start_epoch    = 0
        self.global_step    = 0
        self.best_val_psnr  = -float("inf")   # track best PSNR checkpoint
        self.best_val_fid   =  float("inf")   # track best FID  checkpoint
        self.best_val_lpips =  float("inf")   # track best LPIPS checkpoint

        if cfg.resume_ckpt:
            self._load_checkpoint(cfg.resume_ckpt)

        self.wandb_run = self._init_wandb()

    # ── Device helpers ────────────────────────────────────────────────────────

    def _to_device(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.cfg.device, non_blocking=self.cfg.non_blocking_transfer)
        if self.cfg.use_channels_last and x.ndim == 4:
            x = x.contiguous(memory_format=torch.channels_last)
        return x

    def _autocast(self):
        return amp.autocast(device_type=self.amp_device_type, enabled=self.amp_enabled)

    # ── W&B ───────────────────────────────────────────────────────────────────

    def _init_wandb(self):
        cfg = self.cfg
        if not cfg.use_wandb:
            return None
        if wandb is None:
            print("[warn] use_wandb=True but wandb not installed. Continuing without W&B.")
            return None
        return wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            mode=cfg.wandb_mode,
            config={k: v for k, v in cfg.__class__.__dict__.items() if not k.startswith("__")},
        )

    def _log_wandb(self, payload: dict, step: int):
        if self.wandb_run is not None:
            wandb.log(payload, step=step)

    # ── Checkpoints ───────────────────────────────────────────────────────────

    def _save_checkpoint(self, tag: str) -> str:
        path = os.path.join(self.cfg.save_dir, f"ckpt_{tag}.pth")
        torch.save(
            {
                "epoch":          self.start_epoch,
                "global_step":    self.global_step,
                "G":              self.G.state_dict(),
                "D":              self.D.state_dict(),
                "opt_G":          self.opt_G.state_dict(),
                "opt_D":          self.opt_D.state_dict(),
                # AMP scaler states – avoids instability on resume
                "scaler_G":       self.scaler_G.state_dict(),
                "scaler_D":       self.scaler_D.state_dict(),
                # Best-metric bookkeeping
                "best_val_psnr":  self.best_val_psnr,
                "best_val_fid":   self.best_val_fid,
                "best_val_lpips": self.best_val_lpips,
            },
            path,
        )
        print(f"  [ckpt] saved → {path}")
        return path

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.cfg.device)
        self.G.load_state_dict(ckpt["G"])
        self.D.load_state_dict(ckpt["D"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D.load_state_dict(ckpt["opt_D"])
        if "scaler_G" in ckpt:
            self.scaler_G.load_state_dict(ckpt["scaler_G"])
        if "scaler_D" in ckpt:
            self.scaler_D.load_state_dict(ckpt["scaler_D"])
        self.start_epoch    = ckpt.get("epoch",          0)
        self.global_step    = ckpt.get("global_step",    0)
        self.best_val_psnr  = ckpt.get("best_val_psnr",  -float("inf"))
        self.best_val_fid   = ckpt.get("best_val_fid",    float("inf"))
        self.best_val_lpips = ckpt.get("best_val_lpips",  float("inf"))
        print(f"  [ckpt] resumed from {path} (epoch {self.start_epoch})")

    # ── Attribute sampling ────────────────────────────────────────────────────

    @staticmethod
    def _random_target_attr(attr_src: torch.Tensor, n_attrs: int) -> torch.Tensor:
        """Flip one random attribute per sample; enforce single-hot hair constraint."""
        attr_trg = attr_src.clone()
        B = attr_trg.size(0)
        for i in range(B):
            flip_idx = random.randint(0, n_attrs - 1)
            attr_trg[i, flip_idx] = 1.0 - attr_trg[i, flip_idx]
            # Hair attributes are at indices 0-2 → enforce at most one active
            hair = attr_trg[i, :3]
            if hair.sum() > 1:
                keep = hair.argmax().item()
                attr_trg[i, :3] = 0.0
                attr_trg[i, keep] = 1.0
        return attr_trg

    # ── Train steps ───────────────────────────────────────────────────────────

    def _step_D(self, x_blur, x_clean, attr_src, attr_trg) -> dict:
        cfg = self.cfg
        self.opt_D.zero_grad(set_to_none=True)
        with self._autocast():
            src_real, cls_real = self.D(x_clean)
            with torch.no_grad():
                x_fake = self.G(x_blur, attr_trg)
            src_fake, _ = self.D(x_fake.detach())
            l_adv = adv_d_loss(src_real, src_fake)
            l_cls = cls_loss_real(cls_real, attr_src)
            l_gp  = gradient_penalty(self.D, x_clean, x_fake, cfg.device)
            l_D   = cfg.lambda_adv * l_adv + cfg.lambda_cls * l_cls + cfg.lambda_gp * l_gp

        self.scaler_D.scale(l_D).backward()
        self.scaler_D.step(self.opt_D)
        self.scaler_D.update()
        return {
            "D/adv": float(l_adv.item()),
            "D/cls": float(l_cls.item()),
            "D/gp":  float(l_gp.item()),
            "D/tot": float(l_D.item()),
        }

    def _step_G(self, x_blur, x_clean, attr_src, attr_trg) -> dict:
        cfg = self.cfg
        self.opt_G.zero_grad(set_to_none=True)
        with self._autocast():
            x_fake = self.G(x_blur, attr_trg)
            x_rec  = self.G(x_fake, attr_src)    # cycle back to source attribute
            src_fake, cls_fake = self.D(x_fake)

            l_adv  = adv_g_loss(src_fake)
            l_cls  = cls_loss_fake(cls_fake, attr_trg)
            l_rec  = cycle_loss(x_rec, x_blur)   # cycle target = blurred input (stays sharp via perc)
            l_id   = self.face_loss(x_fake, x_clean)
            l_perc = self.perc_loss(x_fake, x_clean)
            l_G = (
                cfg.lambda_adv  * l_adv
                + cfg.lambda_cls  * l_cls
                + cfg.lambda_rec  * l_rec
                + cfg.lambda_id   * l_id
                + cfg.lambda_perc * l_perc
            )

        self.scaler_G.scale(l_G).backward()
        self.scaler_G.step(self.opt_G)
        self.scaler_G.update()
        return {
            "G/adv":  float(l_adv.item()),
            "G/cls":  float(l_cls.item()),
            "G/rec":  float(l_rec.item()),
            "G/id":   float(l_id.item()),
            "G/perc": float(l_perc.item()),
            "G/tot":  float(l_G.item()),
        }

    # ── Evaluation (pixel-level, fast) ────────────────────────────────────────

    def evaluate_loader(self, loader, split: str, max_batches=None) -> dict:
        """
        Compute per-batch pixel-level metrics: PSNR, SSIM, LPIPS, rec, id, perc.
        Fast enough to run every epoch.
        """
        cfg = self.cfg
        was_training = self.G.training
        self.G.eval()

        # Per-batch LPIPS metric (accumulate average)
        lpips_metric = None
        if TORCHMETRICS_OK:
            lpips_metric = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=False  # normalize=False expects [-1, 1]
            ).to(cfg.device)

        rec_sum = id_sum = perc_sum = psnr_sum = ssim_sum = lpips_sum = 0.0
        n_batches = 0

        iterator = tqdm(loader, desc=f"Eval-{split}", leave=False) if cfg.use_tqdm else loader

        with torch.no_grad():
            for batch_idx, batch in enumerate(iterator):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                x_blur   = self._to_device(batch["blurred"])
                x_clean  = self._to_device(batch["clean"])
                attr_src = batch["attr"].to(cfg.device, non_blocking=cfg.non_blocking_transfer)
                attr_trg = self._random_target_attr(attr_src, cfg.n_attrs)

                with self._autocast():
                    x_fake = self.G(x_blur, attr_trg)
                    x_rec  = self.G(x_fake, attr_src)
                    l_rec  = cycle_loss(x_rec, x_blur)
                    l_id   = self.face_loss(x_fake, x_clean)
                    l_perc = self.perc_loss(x_fake, x_clean)

                psnr_val = batch_psnr(x_fake, x_clean).mean()
                ssim_val = batch_ssim(x_fake, x_clean)

                rec_sum  += float(l_rec.item())
                id_sum   += float(l_id.item())
                perc_sum += float(l_perc.item())
                psnr_sum += float(psnr_val.item())
                ssim_sum += float(ssim_val.item() if isinstance(ssim_val, torch.Tensor) else ssim_val)

                if lpips_metric is not None:
                    xf = x_fake.float().clamp(-1.0, 1.0)
                    xc = x_clean.float().clamp(-1.0, 1.0)
                    lpips_metric.update(xf, xc)
                    lpips_sum += float(lpips_metric.compute().item())
                    lpips_metric.reset()

                n_batches += 1

        if was_training:
            self.G.train()

        d = max(n_batches, 1)
        return {
            f"{split}/rec":   rec_sum  / d,
            f"{split}/id":    id_sum   / d,
            f"{split}/perc":  perc_sum / d,
            f"{split}/psnr":  psnr_sum / d,
            f"{split}/ssim":  ssim_sum / d,
            f"{split}/lpips": lpips_sum / d if lpips_metric is not None else float("nan"),
        }

    # ── GAN distribution metrics (FID / IS / KID / LPIPS) ────────────────────

    def compute_gan_metrics(self, loader, split: str, max_batches: int = None) -> dict:
        """
        Compute distribution-level GAN quality metrics over many generated images.

        FID  (Fréchet Inception Distance)
             Compares InceptionV3 feature statistics of real vs generated images.
             Requires torchmetrics[image]. Lower = better.

        IS   (Inception Score)
             Softmax entropy of InceptionV3 predictions on generated images.
             Captures both quality (sharpness) and diversity. Higher = better.

        KID  (Kernel Inception Distance)
             MMD-based alternative to FID; unbiased and more stable on smaller
             sample sizes. Lower = better.

        LPIPS (Learned Perceptual Image Patch Similarity)
             AlexNet feature distance between generated and paired clean images.
             Correlates better with human perception than SSIM/PSNR. Lower = better.

        Requires torchmetrics[image] ≥ 1.0  (pip install 'torchmetrics[image]').
        Silently returns {} if torchmetrics is unavailable.
        """
        if not TORCHMETRICS_OK:
            print(f"[warn] torchmetrics unavailable — skipping GAN metrics for split={split}")
            return {}

        cfg = self.cfg
        was_training = self.G.training
        self.G.eval()

        # Streaming metric accumulators
        fid       = FrechetInceptionDistance(feature=2048, normalize=True).to(cfg.device)
        inception = InceptionScore(normalize=True).to(cfg.device)
        kid       = KernelInceptionDistance(subset_size=50, normalize=True).to(cfg.device)
        lpips_m   = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=False
        ).to(cfg.device)

        iterator  = tqdm(loader, desc=f"GAN-metrics-{split}", leave=False) if cfg.use_tqdm else loader
        n_batches = 0
        lpips_acc = 0.0

        with torch.no_grad():
            for batch_idx, batch in enumerate(iterator):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                x_blur   = self._to_device(batch["blurred"])
                x_clean  = self._to_device(batch["clean"])
                attr_src = batch["attr"].to(cfg.device, non_blocking=cfg.non_blocking_transfer)
                attr_trg = self._random_target_attr(attr_src, cfg.n_attrs)

                with self._autocast():
                    x_fake = self.G(x_blur, attr_trg)

                # FID/IS/KID use normalize=True → expects float [0, 1]
                real_f = to_float01(x_clean)
                fake_f = to_float01(x_fake)

                fid.update(real_f, real=True)
                fid.update(fake_f, real=False)
                kid.update(real_f, real=True)
                kid.update(fake_f, real=False)
                inception.update(fake_f)

                # LPIPS (per-batch, reset after each)
                xf = x_fake.float().clamp(-1.0, 1.0)
                xc = x_clean.float().clamp(-1.0, 1.0)
                lpips_m.update(xf, xc)
                lpips_acc += float(lpips_m.compute().item())
                lpips_m.reset()

                n_batches += 1

        if was_training:
            self.G.train()

        if n_batches == 0:
            return {}

        fid_val           = fid.compute().item()
        is_mean, is_std   = inception.compute()
        kid_mean, kid_std = kid.compute()
        lpips_val         = lpips_acc / n_batches

        metrics = {
            f"{split}/fid":       fid_val,
            f"{split}/is_mean":   is_mean.item(),
            f"{split}/is_std":    is_std.item(),
            f"{split}/kid_mean":  kid_mean.item(),
            f"{split}/kid_std":   kid_std.item(),
            f"{split}/lpips_gan": lpips_val,
        }
        print(
            f"[{split} GAN-metrics] "
            f"FID={fid_val:.2f} | "
            f"IS={is_mean.item():.3f}±{is_std.item():.3f} | "
            f"KID={kid_mean.item():.4f}±{kid_std.item():.4f} | "
            f"LPIPS={lpips_val:.4f}"
        )
        return metrics

    # ── Full test evaluation ───────────────────────────────────────────────────

    def evaluate_test_checkpoint(self, ckpt_path: str) -> dict:
        cfg = self.cfg
        ckpt = torch.load(ckpt_path, map_location=cfg.device)
        self.G.load_state_dict(ckpt["G"])
        print(f"[test] loaded checkpoint: {ckpt_path}")

        # 1. Per-sample pixel metrics
        test_metrics = self.evaluate_loader(
            self.test_loader, split="test", max_batches=cfg.test_max_batches,
        )

        # 2. Distribution-level GAN metrics
        gan_metrics = self.compute_gan_metrics(
            self.test_loader,
            split="test",
            max_batches=getattr(cfg, "fid_max_batches", 400),
        )
        test_metrics.update(gan_metrics)

        # 3. Print summary
        print(
            "[test] "
            f"PSNR={test_metrics['test/psnr']:.3f} | "
            f"SSIM={test_metrics['test/ssim']:.4f} | "
            f"LPIPS={test_metrics.get('test/lpips', float('nan')):.4f} | "
            f"ID={test_metrics['test/id']:.4f} | "
            f"PERC={test_metrics['test/perc']:.4f}"
        )
        if "test/fid" in test_metrics:
            print(
                f"       FID={test_metrics['test/fid']:.2f} | "
                f"IS={test_metrics['test/is_mean']:.3f}±{test_metrics['test/is_std']:.3f} | "
                f"KID={test_metrics['test/kid_mean']:.4f}±{test_metrics['test/kid_std']:.4f}"
            )

        # 4. CSV logging
        csv_path   = os.path.join(cfg.save_dir, "test_metrics.csv")
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "checkpoint", "psnr", "ssim", "lpips", "id_loss", "perc_loss", "rec_loss",
                    "fid", "is_mean", "is_std", "kid_mean", "kid_std",
                ])
            writer.writerow([
                os.path.basename(ckpt_path),
                test_metrics["test/psnr"],
                test_metrics["test/ssim"],
                test_metrics.get("test/lpips", ""),
                test_metrics["test/id"],
                test_metrics["test/perc"],
                test_metrics["test/rec"],
                test_metrics.get("test/fid",      ""),
                test_metrics.get("test/is_mean",  ""),
                test_metrics.get("test/is_std",   ""),
                test_metrics.get("test/kid_mean", ""),
                test_metrics.get("test/kid_std",  ""),
            ])
        print(f"Results appended to {csv_path}")

        # 5. Cycle-consistency visual
        self.visualize_cycle_consistency(ckpt_path)

        self._log_wandb(test_metrics, step=self.global_step)
        return test_metrics

    # ── Cycle-consistency visualisation ───────────────────────────────────────

    def visualize_cycle_consistency(self, ckpt_path: str):
        """Grid: [Clean | Blur | G(blur→attrB) | G(fake→attrA)] — 4 columns × 4 rows."""
        self.G.eval()
        n_show   = 4
        batch    = next(iter(self.test_loader))
        x_clean  = self._to_device(batch["clean"][:n_show])
        x_blur   = self._to_device(batch["blurred"][:n_show])
        attr_src = batch["attr"][:n_show].to(self.cfg.device, non_blocking=self.cfg.non_blocking_transfer)
        attr_trg = self._random_target_attr(attr_src, self.cfg.n_attrs)

        with torch.no_grad():
            with self._autocast():
                x_fake    = self.G(x_blur, attr_trg)
                x_reconst = self.G(x_fake, attr_src)

        imgs = [denorm(x_clean), denorm(x_blur), denorm(x_fake), denorm(x_reconst)]
        grid = vutils.make_grid(torch.cat(imgs, dim=0), nrow=n_show, padding=2)
        out_path = os.path.join(
            self.cfg.sample_dir, f"cycle_check_{os.path.basename(ckpt_path)}.png"
        )
        vutils.save_image(grid, out_path)
        print(f"Cycle consistency visual saved to: {out_path}")

    # ── Overfit sanity check ───────────────────────────────────────────────────

    def overfit_sanity(self, n_samples: int = 8, n_steps: int = 300, print_every: int = 25) -> dict:
        """
        Overfit on a tiny fixed batch to verify gradient / loss wiring.
        Expected behaviour: REC, ID, PERC decrease; PSNR rises.
        """
        cfg = self.cfg
        print(f"\n[overfit] samples={n_samples}, steps={n_steps}")

        fixed_batch = next(iter(self.train_loader))
        x_blur   = self._to_device(fixed_batch["blurred"][:n_samples])
        x_clean  = self._to_device(fixed_batch["clean"][:n_samples])
        attr_src = fixed_batch["attr"][:n_samples].to(cfg.device, non_blocking=cfg.non_blocking_transfer)
        attr_trg = self._random_target_attr(attr_src, cfg.n_attrs)

        self.G.train()
        self.D.train()

        with torch.no_grad():
            x_f0 = self.G(x_blur, attr_trg)
            x_r0 = self.G(x_f0, attr_src)
            init = {
                "rec":  float(cycle_loss(x_r0, x_blur).item()),
                "id":   float(self.face_loss(x_f0, x_clean).item()),
                "perc": float(self.perc_loss(x_f0, x_clean).item()),
                "psnr": float(batch_psnr(x_f0, x_clean).mean().item()),
            }
        print(f"[overfit:init] REC={init['rec']:.4f} | ID={init['id']:.4f} | PERC={init['perc']:.4f} | PSNR={init['psnr']:.3f}")

        t0        = time.time()
        step_iter = tqdm(range(1, n_steps + 1), desc="Overfit", leave=False) if cfg.use_tqdm else range(1, n_steps + 1)

        for step in step_iter:
            log_D = self._step_D(x_blur, x_clean, attr_src, attr_trg)
            log_G = self._step_G(x_blur, x_clean, attr_src, attr_trg)

            if step % print_every == 0 or step == 1 or step == n_steps:
                with torch.no_grad():
                    xf   = self.G(x_blur, attr_trg)
                    xr   = self.G(xf, attr_src)
                    rec_ = float(cycle_loss(xr, x_blur).item())
                    id_  = float(self.face_loss(xf, x_clean).item())
                    pc_  = float(self.perc_loss(xf, x_clean).item())
                    ps_  = float(batch_psnr(xf, x_clean).mean().item())
                print(
                    f"[overfit] {step:04d}/{n_steps} | D {log_D['D/tot']:.3f} | G {log_G['G/tot']:.3f} "
                    f"| REC {rec_:.4f} | ID {id_:.4f} | PERC {pc_:.4f} | PSNR {ps_:.3f}"
                )
                if cfg.use_tqdm:
                    step_iter.set_postfix({"rec": f"{rec_:.4f}", "psnr": f"{ps_:.2f}"})

            self._log_wandb(
                {"overfit/D_tot": log_D["D/tot"], "overfit/G_tot": log_G["G/tot"],
                 "overfit/D_gp": log_D["D/gp"], "overfit/G_id": log_G["G/id"],
                 "overfit/G_perc": log_G["G/perc"]},
                step=step,
            )

        with torch.no_grad():
            xff = self.G(x_blur, attr_trg)
            xrf = self.G(xff, attr_src)
            final = {
                "rec":  float(cycle_loss(xrf, x_blur).item()),
                "id":   float(self.face_loss(xff, x_clean).item()),
                "perc": float(self.perc_loss(xff, x_clean).item()),
                "psnr": float(batch_psnr(xff, x_clean).mean().item()),
            }
            grid = vutils.make_grid(
                torch.cat([denorm(x_blur), denorm(x_clean), denorm(xff)], dim=0),
                nrow=n_samples, padding=2,
            )
            sample_path = os.path.join(cfg.sample_dir, "overfit_result.png")
            vutils.save_image(grid, sample_path)

        self._log_wandb(
            {"overfit/final_rec": final["rec"], "overfit/final_id": final["id"],
             "overfit/final_perc": final["perc"], "overfit/final_psnr": final["psnr"]},
            step=n_steps,
        )
        if self.wandb_run is not None:
            self._log_wandb({"overfit/grid": wandb.Image(grid)}, step=n_steps)

        self._save_checkpoint("overfit")
        elapsed = (time.time() - t0) / 60.0
        print(f"[overfit] grid saved → {sample_path} | done in {elapsed:.1f} min")
        print(f"[overfit:final] REC={final['rec']:.4f} | ID={final['id']:.4f} | PERC={final['perc']:.4f} | PSNR={final['psnr']:.3f}")
        return {"init": init, "final": final, "sample_path": sample_path}

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self):
        cfg        = self.cfg
        fid_every  = getattr(cfg, "fid_every_epochs", 5)
        fid_batches = getattr(cfg, "fid_max_batches", 400)

        print(f"\n{'='*65}")
        print(f" StarGAN + Contrastive Identity  |  {cfg.num_epochs} epochs")
        print(f" Device: {cfg.device}  |  AMP: {cfg.use_amp}  |  compile: {getattr(cfg,'use_compile',False)}")
        print(f" GAN metrics (FID/IS/KID) every {fid_every} epochs  |  {fid_batches} batches each")
        print(f"{'='*65}\n")

        d_step_idx  = self.global_step
        t0          = time.time()

        fixed_batch = next(iter(self.val_loader))
        fx_blur     = self._to_device(fixed_batch["blurred"])
        fx_clean    = self._to_device(fixed_batch["clean"])

        for epoch in range(self.start_epoch, cfg.num_epochs):
            epoch_loader = (
                tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", leave=False)
                if cfg.use_tqdm else self.train_loader
            )

            for batch in epoch_loader:
                x_blur   = self._to_device(batch["blurred"])
                x_clean  = self._to_device(batch["clean"])
                attr_src = batch["attr"].to(cfg.device, non_blocking=cfg.non_blocking_transfer)
                attr_trg = self._random_target_attr(attr_src, cfg.n_attrs)

                log_D = self._step_D(x_blur, x_clean, attr_src, attr_trg)
                self.global_step += 1
                d_step_idx       += 1

                if self.global_step % cfg.n_critic == 0:
                    log_G = self._step_G(x_blur, x_clean, attr_src, attr_trg)

                    linear_lr_decay(self.opt_G, d_step_idx, self.total_d_steps, self.decay_start, cfg.lr_g)
                    linear_lr_decay(self.opt_D, d_step_idx, self.total_d_steps, self.decay_start, cfg.lr_d)
                    lr = self.opt_G.param_groups[0]["lr"]

                    self._log_wandb({**log_D, **log_G, "train/lr": lr, "train/epoch": epoch + 1}, step=d_step_idx)
                    if cfg.use_tqdm:
                        epoch_loader.set_postfix({
                            "D":  f"{log_D['D/tot']:.3f}",
                            "G":  f"{log_G['G/tot']:.3f}",
                            "id": f"{log_G['G/id']:.3f}",
                        })

                    if d_step_idx % cfg.log_step == 0:
                        elapsed = (time.time() - t0) / 60.0
                        print(
                            f"Ep {epoch+1:02d}/{cfg.num_epochs} step {d_step_idx:6d} "
                            f"| D {log_D['D/tot']:.3f} | G {log_G['G/tot']:.3f} "
                            f"| id {log_G['G/id']:.4f} | perc {log_G['G/perc']:.4f} "
                            f"| rec {log_G['G/rec']:.4f} | lr {lr:.2e} | {elapsed:.1f} min"
                        )

                    if d_step_idx % cfg.sample_step == 0:
                        sp = os.path.join(cfg.sample_dir, f"step_{d_step_idx:06d}.png")
                        grid = save_sample_grid(self.G, fx_blur, fx_clean, self.fixed_attrs, sp, cfg.device)
                        print(f"  [sample] → {sp}")
                        if self.wandb_run is not None:
                            self._log_wandb({"samples/grid": wandb.Image(grid)}, step=d_step_idx)

                    if d_step_idx % cfg.save_step == 0:
                        self._save_checkpoint(f"step{d_step_idx:06d}")

            # ── End-of-epoch ────────────────────────────────────────────────
            self.start_epoch = epoch + 1
            self._save_checkpoint(f"ep{epoch + 1:02d}")

            if (epoch + 1) % cfg.val_every_epochs == 0:
                # ── Pixel metrics (every val epoch) ─────────────────────────
                val_m = self.evaluate_loader(
                    self.val_loader, split="val", max_batches=cfg.val_max_batches,
                )
                self._log_wandb({**val_m, "val/epoch": epoch + 1}, step=d_step_idx)
                print(
                    f"[val] ep {epoch+1:02d} "
                    f"| PSNR  {val_m['val/psnr']:.3f} "
                    f"| SSIM  {val_m['val/ssim']:.4f} "
                    f"| LPIPS {val_m.get('val/lpips', float('nan')):.4f} "
                    f"| ID    {val_m['val/id']:.4f} "
                    f"| PERC  {val_m['val/perc']:.4f}"
                )

                # Best PSNR checkpoint
                if val_m["val/psnr"] > self.best_val_psnr:
                    self.best_val_psnr = val_m["val/psnr"]
                    bp = self._save_checkpoint("best_psnr")
                    print(f"  ★ [best-psnr] PSNR={self.best_val_psnr:.3f} → {bp}")
                    self._log_wandb({"best/val_psnr": self.best_val_psnr}, step=d_step_idx)

                # ── GAN distribution metrics (periodic) ─────────────────────
                run_fid = (epoch + 1) % fid_every == 0 or (epoch + 1) == cfg.num_epochs
                if run_fid:
                    print(f"[GAN metrics] Computing FID / IS / KID — epoch {epoch + 1}…")
                    gan_m = self.compute_gan_metrics(
                        self.val_loader, split="val", max_batches=fid_batches,
                    )
                    self._log_wandb({**gan_m, "val/epoch": epoch + 1}, step=d_step_idx)

                    # Best FID checkpoint (lower = better)
                    if "val/fid" in gan_m and gan_m["val/fid"] < self.best_val_fid:
                        self.best_val_fid = gan_m["val/fid"]
                        bf = self._save_checkpoint("best_fid")
                        print(f"  ★ [best-fid]  FID={self.best_val_fid:.2f} → {bf}")
                        self._log_wandb({"best/val_fid": self.best_val_fid}, step=d_step_idx)

                    # Best LPIPS checkpoint (lower = better)
                    if "val/lpips_gan" in gan_m and gan_m["val/lpips_gan"] < self.best_val_lpips:
                        self.best_val_lpips = gan_m["val/lpips_gan"]
                        bl = self._save_checkpoint("best_lpips")
                        print(f"  ★ [best-lpips] LPIPS={self.best_val_lpips:.4f} → {bl}")
                        self._log_wandb({"best/val_lpips": self.best_val_lpips}, step=d_step_idx)

        total_h    = (time.time() - t0) / 3600.0
        final_ckpt = self._save_checkpoint("final")
        print(f"\nTraining complete in {total_h:.2f} h  |  final checkpoint → {final_ckpt}")

        test_metrics = self.evaluate_test_checkpoint(final_ckpt)
        print(
            "[final test] "
            f"PSNR={test_metrics['test/psnr']:.3f}  "
            f"SSIM={test_metrics['test/ssim']:.4f}  "
            f"LPIPS={test_metrics.get('test/lpips', float('nan')):.4f}  "
            f"FID={test_metrics.get('test/fid', float('nan')):.2f}  "
            f"IS={test_metrics.get('test/is_mean', float('nan')):.3f}  "
            f"KID={test_metrics.get('test/kid_mean', float('nan')):.4f}"
        )

        if self.wandb_run is not None:
            wandb.finish()
