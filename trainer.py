"""
trainer.py  –  Training / validation / checkpoint-test loop with W&B logging.
"""

import os
import time
import random

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


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_runtime(cfg: Config):
    """Enable safe performance features for NVIDIA GPUs (Kaggle friendly)."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = cfg.enable_tf32
        torch.backends.cudnn.allow_tf32 = cfg.enable_tf32
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) / 2.0


def batch_psnr(x_fake: torch.Tensor, x_target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """PSNR over a batch. Inputs are expected in [-1, 1]."""
    x_f = denorm(x_fake)
    x_t = denorm(x_target)
    mse = torch.mean((x_f - x_t) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / (mse + eps))


def batch_ssim_global(x_fake: torch.Tensor, x_target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Global SSIM approximation over a batch; fast enough for epoch validation."""
    x = denorm(x_fake)
    y = denorm(x_target)
    x_mean = torch.mean(x, dim=(1, 2, 3))
    y_mean = torch.mean(y, dim=(1, 2, 3))

    x_var = torch.mean((x - x_mean.view(-1, 1, 1, 1)) ** 2, dim=(1, 2, 3))
    y_var = torch.mean((y - y_mean.view(-1, 1, 1, 1)) ** 2, dim=(1, 2, 3))
    xy_cov = torch.mean(
        (x - x_mean.view(-1, 1, 1, 1)) * (y - y_mean.view(-1, 1, 1, 1)),
        dim=(1, 2, 3),
    )

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    num = (2 * x_mean * y_mean + c1) * (2 * xy_cov + c2)
    den = (x_mean ** 2 + y_mean ** 2 + c1) * (x_var + y_var + c2) + eps
    return num / den


def save_sample_grid(
    G: Generator,
    x_blur: torch.Tensor,
    x_clean: torch.Tensor,
    fixed_attrs: list,
    path: str,
    device: torch.device,
    n_show: int = 8,
) -> torch.Tensor:
    """Save sample grid and return it for optional W&B image logging."""
    was_training = G.training
    G.eval()
    with torch.no_grad():
        x_b = x_blur[:n_show].to(device)
        x_c = x_clean[:n_show].to(device)
        imgs = [denorm(x_b), denorm(x_c)]
        for trg_attr in fixed_attrs:
            a = torch.tensor(trg_attr, dtype=torch.float32, device=device)
            a = a.unsqueeze(0).expand(n_show, -1)
            gen = G(x_b, a)
            imgs.append(denorm(gen))
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
    if current_step >= decay_start:
        ratio = (current_step - decay_start) / max(total_steps - decay_start, 1)
        new_lr = base_lr * (1.0 - ratio)
        for g in optimizer.param_groups:
            g["lr"] = max(new_lr, 0.0)


class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        set_seed(cfg.seed)
        configure_runtime(cfg)

        os.makedirs(cfg.save_dir, exist_ok=True)
        os.makedirs(cfg.sample_dir, exist_ok=True)

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
        count_params(self.G, "Generator")
        count_params(self.D, "Discriminator")

        print("Loading FaceNet (frozen)...")
        self.face_loss = FaceNetIdentityLoss(
            pretrained=cfg.facenet_pretrained,
            device=cfg.device,
        )
        print("Loading VGG16 perceptual loss (frozen)...")
        self.perc_loss = VGGPerceptualLoss(
            layers=cfg.vgg_layers,
            weights=cfg.vgg_weights,
            device=cfg.device,
        )

        self.opt_G = optim.Adam(self.G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
        self.opt_D = optim.Adam(self.D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))
        self.amp_enabled = cfg.use_amp and cfg.device.type == "cuda"
        self.amp_device_type = "cuda" if cfg.device.type == "cuda" else "cpu"
        self.scaler_G = amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.scaler_D = amp.GradScaler("cuda", enabled=self.amp_enabled)

        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(cfg)
        steps_per_epoch = len(self.train_loader)
        self.total_d_steps = max((steps_per_epoch * cfg.num_epochs) // cfg.n_critic, 1)
        self.decay_start = int(self.total_d_steps * cfg.lr_decay_start_ratio)

        self.fixed_attrs = [
            [1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0],
        ][: cfg.n_attrs]

        self.start_epoch = 0
        self.global_step = 0
        if cfg.resume_ckpt:
            self._load_checkpoint(cfg.resume_ckpt)

        self.wandb_run = self._init_wandb()

    def _to_device(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.cfg.device, non_blocking=self.cfg.non_blocking_transfer)
        if self.cfg.use_channels_last and x.ndim == 4:
            x = x.contiguous(memory_format=torch.channels_last)
        return x

    def _autocast(self):
        return amp.autocast(device_type=self.amp_device_type, enabled=self.amp_enabled)

    def _init_wandb(self):
        cfg = self.cfg
        if not cfg.use_wandb:
            return None
        if wandb is None:
            print("[warn] use_wandb=True but wandb is not installed. Continuing without W&B.")
            return None
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            mode=cfg.wandb_mode,
            config={k: v for k, v in cfg.__class__.__dict__.items() if not k.startswith("__")},
        )
        return run

    def _log_wandb(self, payload: dict, step: int):
        if self.wandb_run is not None:
            wandb.log(payload, step=step)

    def _save_checkpoint(self, tag: str):
        path = os.path.join(self.cfg.save_dir, f"ckpt_{tag}.pth")
        torch.save(
            {
                "epoch": self.start_epoch,
                "global_step": self.global_step,
                "G": self.G.state_dict(),
                "D": self.D.state_dict(),
                "opt_G": self.opt_G.state_dict(),
                "opt_D": self.opt_D.state_dict(),
            },
            path,
        )
        print(f"  [ckpt] saved -> {path}")
        return path

    def _load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.cfg.device)
        self.G.load_state_dict(ckpt["G"])
        self.D.load_state_dict(ckpt["D"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D.load_state_dict(ckpt["opt_D"])
        self.start_epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        print(f"  [ckpt] resumed from {path} (epoch {self.start_epoch})")

    @staticmethod
    def _random_target_attr(attr_src: torch.Tensor, n_attrs: int) -> torch.Tensor:
        attr_trg = attr_src.clone()
        B = attr_trg.size(0)
        for i in range(B):
            flip_idx = random.randint(0, n_attrs - 1)
            attr_trg[i, flip_idx] = 1.0 - attr_trg[i, flip_idx]
            hair = attr_trg[i, :3]
            if hair.sum() > 1:
                keep = hair.argmax().item()
                attr_trg[i, :3] = 0.0
                attr_trg[i, keep] = 1.0
        return attr_trg

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
            l_gp = gradient_penalty(self.D, x_clean, x_fake, cfg.device)
            l_D = cfg.lambda_adv * l_adv + cfg.lambda_cls * l_cls + cfg.lambda_gp * l_gp

        self.scaler_D.scale(l_D).backward()
        self.scaler_D.step(self.opt_D)
        self.scaler_D.update()
        return {
            "D/adv": float(l_adv.item()),
            "D/cls": float(l_cls.item()),
            "D/gp": float(l_gp.item()),
            "D/tot": float(l_D.item()),
        }

    def _step_G(self, x_blur, x_clean, attr_src, attr_trg) -> dict:
        cfg = self.cfg
        self.opt_G.zero_grad(set_to_none=True)
        with self._autocast():
            x_fake = self.G(x_blur, attr_trg)
            x_rec = self.G(x_fake, attr_src)
            src_fake, cls_fake = self.D(x_fake)

            l_adv = adv_g_loss(src_fake)
            l_cls = cls_loss_fake(cls_fake, attr_trg)
            l_rec = cycle_loss(x_rec, x_blur)
            l_id = self.face_loss(x_fake, x_clean)
            l_perc = self.perc_loss(x_fake, x_clean)
            l_G = (
                cfg.lambda_adv * l_adv
                + cfg.lambda_cls * l_cls
                + cfg.lambda_rec * l_rec
                + cfg.lambda_id * l_id
                + cfg.lambda_perc * l_perc
            )

        self.scaler_G.scale(l_G).backward()
        self.scaler_G.step(self.opt_G)
        self.scaler_G.update()
        return {
            "G/adv": float(l_adv.item()),
            "G/cls": float(l_cls.item()),
            "G/rec": float(l_rec.item()),
            "G/id": float(l_id.item()),
            "G/perc": float(l_perc.item()),
            "G/tot": float(l_G.item()),
        }

    def evaluate_loader(self, loader, split: str, max_batches=None) -> dict:
        cfg = self.cfg
        was_training = self.G.training
        self.G.eval()

        rec_sum = 0.0
        id_sum = 0.0
        perc_sum = 0.0
        psnr_sum = 0.0
        ssim_sum = 0.0
        n_batches = 0

        iterator = loader
        if cfg.use_tqdm:
            iterator = tqdm(loader, desc=f"Eval-{split}", leave=False)

        with torch.no_grad():
            for batch_idx, batch in enumerate(iterator):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                x_blur = self._to_device(batch["blurred"])
                x_clean = self._to_device(batch["clean"])
                attr_src = batch["attr"].to(cfg.device, non_blocking=cfg.non_blocking_transfer)
                attr_trg = self._random_target_attr(attr_src, cfg.n_attrs)

                with self._autocast():
                    x_fake = self.G(x_blur, attr_trg)
                    x_rec = self.G(x_fake, attr_src)
                    l_rec = cycle_loss(x_rec, x_blur)
                    l_id = self.face_loss(x_fake, x_clean)
                    l_perc = self.perc_loss(x_fake, x_clean)

                psnr_val = batch_psnr(x_fake, x_clean).mean()
                ssim_val = batch_ssim_global(x_fake, x_clean).mean()

                rec_sum += float(l_rec.item())
                id_sum += float(l_id.item())
                perc_sum += float(l_perc.item())
                psnr_sum += float(psnr_val.item())
                ssim_sum += float(ssim_val.item())
                n_batches += 1

        if was_training:
            self.G.train()

        denom = max(n_batches, 1)
        metrics = {
            f"{split}/rec": rec_sum / denom,
            f"{split}/id": id_sum / denom,
            f"{split}/perc": perc_sum / denom,
            f"{split}/psnr": psnr_sum / denom,
            f"{split}/ssim": ssim_sum / denom,
        }
        return metrics

    def evaluate_test_checkpoint(self, ckpt_path: str) -> dict:
        cfg = self.cfg
        ckpt = torch.load(ckpt_path, map_location=cfg.device)
        self.G.load_state_dict(ckpt["G"])
        print(f"[test] loaded checkpoint: {ckpt_path}")
        test_metrics = self.evaluate_loader(
            self.test_loader,
            split="test",
            max_batches=cfg.test_max_batches,
        )
        print(
            "[test] "
            f"PSNR={test_metrics['test/psnr']:.3f} | "
            f"SSIM={test_metrics['test/ssim']:.4f} | "
            f"ID={test_metrics['test/id']:.4f} | "
            f"PERC={test_metrics['test/perc']:.4f}"
        )
        self._log_wandb(test_metrics, step=self.global_step)
        return test_metrics

    def overfit_sanity(self, n_samples: int = 8, n_steps: int = 300, print_every: int = 25) -> dict:
        """
        Overfit on a tiny fixed batch to verify gradients/loss wiring.
        Expected behavior: reconstruction/id/perceptual losses decrease and PSNR rises.
        """
        cfg = self.cfg
        print(f"\n[overfit] running tiny-batch sanity check: samples={n_samples}, steps={n_steps}")

        fixed_batch = next(iter(self.train_loader))
        x_blur = self._to_device(fixed_batch["blurred"][:n_samples])
        x_clean = self._to_device(fixed_batch["clean"][:n_samples])
        attr_src = fixed_batch["attr"][:n_samples].to(cfg.device, non_blocking=cfg.non_blocking_transfer)
        attr_trg = self._random_target_attr(attr_src, cfg.n_attrs)

        self.G.train()
        self.D.train()

        with torch.no_grad():
            x_fake0 = self.G(x_blur, attr_trg)
            x_rec0 = self.G(x_fake0, attr_src)
            init_metrics = {
                "rec": float(cycle_loss(x_rec0, x_blur).item()),
                "id": float(self.face_loss(x_fake0, x_clean).item()),
                "perc": float(self.perc_loss(x_fake0, x_clean).item()),
                "psnr": float(batch_psnr(x_fake0, x_clean).mean().item()),
            }
        print(
            "[overfit:init] "
            f"REC={init_metrics['rec']:.4f} | "
            f"ID={init_metrics['id']:.4f} | "
            f"PERC={init_metrics['perc']:.4f} | "
            f"PSNR={init_metrics['psnr']:.3f}"
        )

        t0 = time.time()
        step_iter = range(1, n_steps + 1)
        if cfg.use_tqdm:
            step_iter = tqdm(step_iter, desc="Overfit", leave=False)
        for step in step_iter:
            log_D = self._step_D(x_blur, x_clean, attr_src, attr_trg)
            log_G = self._step_G(x_blur, x_clean, attr_src, attr_trg)

            if step % print_every == 0 or step == 1 or step == n_steps:
                with torch.no_grad():
                    x_fake = self.G(x_blur, attr_trg)
                    x_rec = self.G(x_fake, attr_src)
                    rec_now = float(cycle_loss(x_rec, x_blur).item())
                    id_now = float(self.face_loss(x_fake, x_clean).item())
                    perc_now = float(self.perc_loss(x_fake, x_clean).item())
                    psnr_now = float(batch_psnr(x_fake, x_clean).mean().item())
                print(
                    f"[overfit] step {step:04d}/{n_steps} "
                    f"| D {log_D['D/tot']:.3f} "
                    f"| G {log_G['G/tot']:.3f} "
                    f"| REC {rec_now:.4f} "
                    f"| ID {id_now:.4f} "
                    f"| PERC {perc_now:.4f} "
                    f"| PSNR {psnr_now:.3f}"
                )
                if cfg.use_tqdm:
                    step_iter.set_postfix({"rec": f"{rec_now:.4f}", "psnr": f"{psnr_now:.2f}"})

            self._log_wandb(
                {
                    "overfit/D_tot": log_D["D/tot"],
                    "overfit/G_tot": log_G["G/tot"],
                    "overfit/D_gp": log_D["D/gp"],
                    "overfit/G_id": log_G["G/id"],
                    "overfit/G_perc": log_G["G/perc"],
                },
                step=step,
            )

        with torch.no_grad():
            x_fakef = self.G(x_blur, attr_trg)
            x_recf = self.G(x_fakef, attr_src)
            final_metrics = {
                "rec": float(cycle_loss(x_recf, x_blur).item()),
                "id": float(self.face_loss(x_fakef, x_clean).item()),
                "perc": float(self.perc_loss(x_fakef, x_clean).item()),
                "psnr": float(batch_psnr(x_fakef, x_clean).mean().item()),
            }

            grid = vutils.make_grid(
                torch.cat([denorm(x_blur), denorm(x_clean), denorm(x_fakef)], dim=0),
                nrow=n_samples,
                padding=2,
            )
            sample_path = os.path.join(cfg.sample_dir, "overfit_result.png")
            vutils.save_image(grid, sample_path)

        self._log_wandb(
            {
                "overfit/final_rec": final_metrics["rec"],
                "overfit/final_id": final_metrics["id"],
                "overfit/final_perc": final_metrics["perc"],
                "overfit/final_psnr": final_metrics["psnr"],
            },
            step=n_steps,
        )
        if self.wandb_run is not None:
            self._log_wandb({"overfit/grid": wandb.Image(grid)}, step=n_steps)

        self._save_checkpoint("overfit")
        elapsed = (time.time() - t0) / 60.0
        print(f"[overfit] sample grid saved -> {sample_path}")
        print(f"[overfit] done in {elapsed:.1f} min")
        print(
            "[overfit:final] "
            f"REC={final_metrics['rec']:.4f} | "
            f"ID={final_metrics['id']:.4f} | "
            f"PERC={final_metrics['perc']:.4f} | "
            f"PSNR={final_metrics['psnr']:.3f}"
        )

        return {
            "init": init_metrics,
            "final": final_metrics,
            "sample_path": sample_path,
        }

    def train(self):
        cfg = self.cfg
        print(f"\n{'='*60}")
        print(f" StarGAN + Contrastive Identity | {cfg.num_epochs} epochs")
        print(f" Device: {cfg.device} | AMP: {cfg.use_amp}")
        print(f"{'='*60}\n")

        d_step_idx = self.global_step
        t0 = time.time()

        fixed_batch = next(iter(self.val_loader))
        fx_blur = self._to_device(fixed_batch["blurred"])
        fx_clean = self._to_device(fixed_batch["clean"])

        for epoch in range(self.start_epoch, cfg.num_epochs):
            epoch_loader = self.train_loader
            if cfg.use_tqdm:
                epoch_loader = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}", leave=False)

            for batch in epoch_loader:
                x_blur = self._to_device(batch["blurred"])
                x_clean = self._to_device(batch["clean"])
                attr_src = batch["attr"].to(cfg.device, non_blocking=cfg.non_blocking_transfer)
                attr_trg = self._random_target_attr(attr_src, cfg.n_attrs)

                log_D = self._step_D(x_blur, x_clean, attr_src, attr_trg)
                self.global_step += 1
                d_step_idx += 1

                if self.global_step % cfg.n_critic == 0:
                    log_G = self._step_G(x_blur, x_clean, attr_src, attr_trg)

                    linear_lr_decay(self.opt_G, d_step_idx, self.total_d_steps, self.decay_start, cfg.lr_g)
                    linear_lr_decay(self.opt_D, d_step_idx, self.total_d_steps, self.decay_start, cfg.lr_d)
                    lr = self.opt_G.param_groups[0]["lr"]

                    train_payload = {
                        **log_D,
                        **log_G,
                        "train/lr": lr,
                        "train/epoch": epoch + 1,
                    }
                    self._log_wandb(train_payload, step=d_step_idx)
                    if cfg.use_tqdm:
                        epoch_loader.set_postfix({
                            "D": f"{log_D['D/tot']:.3f}",
                            "G": f"{log_G['G/tot']:.3f}",
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
                        sample_path = os.path.join(cfg.sample_dir, f"step_{d_step_idx:06d}.png")
                        grid = save_sample_grid(
                            self.G,
                            fx_blur,
                            fx_clean,
                            self.fixed_attrs,
                            sample_path,
                            cfg.device,
                        )
                        print(f"  [sample] -> {sample_path}")
                        if self.wandb_run is not None:
                            self._log_wandb({"samples/grid": wandb.Image(grid)}, step=d_step_idx)

                    if d_step_idx % cfg.save_step == 0:
                        self._save_checkpoint(f"step{d_step_idx:06d}")

            self.start_epoch = epoch + 1
            ep_ckpt = self._save_checkpoint(f"ep{epoch+1:02d}")

            if (epoch + 1) % cfg.val_every_epochs == 0:
                val_metrics = self.evaluate_loader(
                    self.val_loader,
                    split="val",
                    max_batches=cfg.val_max_batches,
                )
                self._log_wandb({**val_metrics, "val/epoch": epoch + 1}, step=d_step_idx)
                print(
                    f"[val] ep {epoch+1:02d} "
                    f"| PSNR {val_metrics['val/psnr']:.3f} "
                    f"| SSIM {val_metrics['val/ssim']:.4f} "
                    f"| ID {val_metrics['val/id']:.4f} "
                    f"| PERC {val_metrics['val/perc']:.4f}"
                )

        total_h = (time.time() - t0) / 3600.0
        print(f"\nTraining complete in {total_h:.2f} h")
        final_ckpt = self._save_checkpoint("final")

        test_metrics = self.evaluate_test_checkpoint(final_ckpt)
        print(
            "[final test] "
            f"PSNR={test_metrics['test/psnr']:.3f}, "
            f"SSIM={test_metrics['test/ssim']:.4f}, "
            f"ID={test_metrics['test/id']:.4f}, "
            f"PERC={test_metrics['test/perc']:.4f}"
        )

        if self.wandb_run is not None:
            wandb.finish()
