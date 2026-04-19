"""
config.py  –  Central hyperparameter store.
All paths assume the standard Kaggle CelebA dataset layout:
  /kaggle/input/celeba-dataset/
      img_align_celeba/img_align_celeba/  ← JPEG images
      list_attr_celeba.csv                ← attribute labels
"""

import os
import torch


class Config:
    # ── Dataset ───────────────────────────────────────────────────────────────
    celeba_root  = "/kaggle/input/celeba-dataset"
    attr_path    = os.path.join(celeba_root, "list_attr_celeba.csv")
    image_dir    = os.path.join(
                       celeba_root,
                       "img_align_celeba",
                       "img_align_celeba",
                   )
    image_size   = 128          # training resolution (H = W)
    # 5 binary CelebA attributes used for conditioning
    selected_attrs = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]
    n_attrs      = len(selected_attrs)

    # ── Blur augmentation  (degraded input simulation) ────────────────────────
    # Applied on-the-fly in the DataLoader; sigma is sampled uniformly in range
    blur_kernel  = 21           # Gaussian kernel size (must be odd)
    blur_sigma_lo = 4.0         # lower bound of random sigma
    blur_sigma_hi = 8.0         # upper bound of random sigma

    # ── Generator ─────────────────────────────────────────────────────────────
    g_conv_dim   = 64           # base channel width
    g_repeat_num = 6            # number of residual blocks in bottleneck

    # ── Discriminator (PatchGAN + auxiliary classifier) ───────────────────────
    d_conv_dim   = 64
    d_repeat_num = 6            # strided-conv downsampling steps

    # ── Training schedule ─────────────────────────────────────────────────────
    batch_size   = 16
    num_epochs   = 30
    n_critic     = 5            # discriminator updates per generator update
    lr_g         = 1e-4
    lr_d         = 1e-4
    beta1        = 0.5
    beta2        = 0.999

    # Linear LR decay: starts after this fraction of total steps
    lr_decay_start_ratio = 0.5

    # ── Validation / Test controls ───────────────────────────────────────────
    # Run validation at the end of every N epochs.
    val_every_epochs = 1
    # Cap number of batches for quick metric passes (None = full loader).
    val_max_batches  = 120
    test_max_batches = 200

    # ── Loss weights ──────────────────────────────────────────────────────────
    lambda_adv   = 1.0          # adversarial
    lambda_cls   = 1.0          # attribute classification
    lambda_rec   = 10.0         # cycle-reconstruction (L1)
    lambda_id    = 5.0          # contrastive identity (FaceNet embedding)
    lambda_perc  = 0.5          # perceptual (VGG feature matching)
    lambda_gp    = 10.0         # WGAN gradient penalty

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

    save_dir     = "/kaggle/working/checkpoints"
    sample_dir   = "/kaggle/working/samples"

    log_step     = 50           # print loss every N D-steps
    sample_step  = 500          # save sample grid every N D-steps
    save_step    = 2000         # checkpoint every N D-steps

    resume_ckpt  = None         # path to checkpoint to resume from (or None)

    # ── Weights & Biases ─────────────────────────────────────────────────────
    use_wandb     = True
    wandb_project = "stargan-blur-upscale"
    wandb_entity  = None        # set your username/team on Kaggle if needed
    wandb_run_name = None       # auto-generated when None
    wandb_mode    = "online"   # "online" or "offline"

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed         = 42
