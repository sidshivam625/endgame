"""
config.py  –  Central hyperparameter store.
All paths assume the standard Kaggle CelebA dataset layout:
  /kaggle/input/celeba-dataset/
      img_align_celeba/img_align_celeba/  ← JPEG images
      list_attr_celeba.csv                ← attribute labels
"""

import os
import torch


def _default_celeba_root() -> str:
    """Pick a sensible dataset root for Kaggle or local runs."""
    env_root = os.getenv("STARGAN_CELEBA_ROOT")
    if env_root:
        return env_root
    kaggle_candidates = [
        "/kaggle/input/datasets/jessicali9530/celeba-dataset",
        "/kaggle/input/celeba-dataset",
    ]
    for kaggle_root in kaggle_candidates:
        if os.path.isdir(kaggle_root):
            return kaggle_root
    return "./data/celeba-dataset"


def _default_work_root() -> str:
    """Pick output root for checkpoints/samples."""
    env_root = os.getenv("STARGAN_WORK_ROOT")
    if env_root:
        return env_root
    kaggle_work = "/kaggle/working"
    if os.path.isdir(kaggle_work):
        return kaggle_work
    return "./runs"


class Config:
    # ── Dataset ───────────────────────────────────────────────────────────────
    celeba_root  = _default_celeba_root()
    attr_path    = os.path.join(celeba_root, "list_attr_celeba.csv")
    image_dir    = os.path.join(
                       celeba_root,
                       "img_align_celeba",
                       "img_align_celeba",
                   )
    image_size   = 128          # training resolution (H = W)
    dataset_mode = "mounted"    # "mounted" (Kaggle input) or "local"
    # Conditioning attributes used by generator/discriminator.
    # Includes facial-expression/appearance attributes and excludes Brown_Hair/Young.
    selected_attrs = [
        "Black_Hair",
        "Blond_Hair",
        "Male",
        "Smiling",
        "Mouth_Slightly_Open",
        "Narrow_Eyes",  # used as a practical proxy for 'disgust-like' expression
    ]
    n_attrs      = len(selected_attrs)

    # ── Blur augmentation  (degraded input simulation) ────────────────────────
    # Applied on-the-fly in the DataLoader; sigma is sampled uniformly in range
    blur_kernel  = 5      # Gaussian kernel size (must be odd)
    blur_sigma_lo = 0.5      # lower bound of random sigma (slight blur)
    blur_sigma_hi = 2       # upper bound of random sigma (slight blur)

    # ── Generator ─────────────────────────────────────────────────────────────
    g_conv_dim   = 64           # base channel width
    g_repeat_num = 6            # number of residual blocks in bottleneck

    # ── Discriminator (PatchGAN + auxiliary classifier) ───────────────────────
    d_conv_dim   = 64
    d_repeat_num = 6            # strided-conv downsampling steps

    # ── Training schedule ─────────────────────────────────────────────────────
    batch_size   = 32
    num_epochs   = 30
    n_critic     = 5            # discriminator updates per generator update
    lr_g         = 1e-4
    lr_d         = 1e-4
    min_lr       = 1e-6         # LR floor to avoid premature zero-learning regime
    beta1        = 0.5
    beta2        = 0.999

    # Linear LR decay: starts after this fraction of total steps
    lr_decay_start_ratio = 0.5

    # ── Validation / Test controls ───────────────────────────────────────────
    # Run validation at the end of every N epochs.
    val_every_epochs = 1
    # Cap number of batches for quick metric passes (None = full loader).
    # 400 batches × bs=16 = 6 400 val images — meaningful statistics.
    val_max_batches  = 400
    test_max_batches = 200

    # ── Loss weights ──────────────────────────────────────────────────────────
    lambda_adv   = 1.0          # adversarial
    lambda_cls   = 1.0          # attribute classification
    lambda_rec   = 10.0         # cycle-reconstruction (L1)
    lambda_id    = 5.0          # contrastive identity (FaceNet embedding)
    lambda_perc  = 0.5          # perceptual (VGG feature matching)
    lambda_gp    = 10.0         # WGAN gradient penalty

    # Identity-loss warmup: reduces early over-constraint on attribute changes.
    lambda_id_start_ratio = 0.3   # start at 30% of lambda_id
    lambda_id_warmup_epochs = 5   # linearly ramp to full lambda_id

    # ── FaceNet / Identity branch ─────────────────────────────────────────────
    # InceptionResnetV1 pretrained on VGGFace2 (facenet-pytorch)
    facenet_pretrained = "vggface2"

    # ── VGG Perceptual loss layers ─────────────────────────────────────────────
    # relu indices in torchvision.models.vgg16.features
    vgg_layers   = [3, 8, 15]   # relu1_2, relu2_2, relu3_3
    vgg_weights  = [1.0, 1.0, 1.0]

    # ── Infrastructure ────────────────────────────────────────────────────────
    use_amp      = True         # automatic mixed precision (fp16)
    num_workers  = 4
    pin_memory   = True
    persistent_workers = True
    prefetch_factor    = 2

    # Runtime/GPU efficiency toggles (safe defaults for Kaggle T4/P100)
    use_tqdm      = True
    use_channels_last = True
    enable_tf32   = True
    cudnn_benchmark = True
    non_blocking_transfer = True

    work_root    = _default_work_root()
    save_dir     = os.path.join(work_root, "checkpoints")
    sample_dir   = os.path.join(work_root, "samples")

    # Training logging cadence (reduced console/W&B spam)
    log_step     = 2000         # print train losses every N D-steps
    wandb_log_every_steps = 2000  # log train losses/LR to W&B every N D-steps

    # Sample cadence: generate exactly this many sample grids each epoch
    sample_times_per_epoch = 5
    sample_step  = 500          # legacy fallback cadence (kept for compatibility)
    # Keep inline preview off by default to avoid notebook IOPub message overflow.
    # Samples are still saved to disk and logged to W&B.
    live_preview  = False
    save_step    = 2000         # checkpoint every N D-steps

    resume_ckpt  = None         # path to checkpoint to resume from (or None)

    # ── GAN Evaluation Metrics ────────────────────────────────────────────────
    # Requires:  pip install 'torchmetrics[image]'
    # FID  (Fréchet Inception Distance)  ↓ better — compares InceptionV3 feature distributions
    # IS   (Inception Score)             ↑ better — quality + diversity of generated images
    # KID  (Kernel Inception Distance)   ↓ better — unbiased FID alternative, stable on small N
    # LPIPS(Learned Perceptual Patch Sim)↓ better — AlexNet feature distance, perceptual quality
    # Epoch-end GAN metric cadence (speed-friendly, still complete)
    gan_metrics_per_epoch = 0       # disable quarter-epoch GAN metric runs
    fid_every_epochs = 1            # compute FID/IS/KID at end of every epoch
    fid_max_batches  = 400          # full epoch-end/test batches for stable estimate
    quarter_val_max_batches = 64    # kept for optional mid-epoch use
    quarter_fid_max_batches = 64    # kept for optional mid-epoch use

    # ── torch.compile (PyTorch 2.x) ───────────────────────────────────────────
    # Adds ~15–25 % throughput boost with reduce-overhead mode.
    # Set True if you are on PyTorch ≥ 2.0 and can tolerate a ~3 min compile warm-up.
    use_compile = False

    # ── Weights & Biases ─────────────────────────────────────────────────────
    use_wandb     = True
    wandb_project = "stargan-blur-upscale"
    wandb_entity  = None        # set your username/team on Kaggle if needed
    wandb_run_name = None       # auto-generated when None
    wandb_mode    = "online"   # "online" or "offline"

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed         = 42
