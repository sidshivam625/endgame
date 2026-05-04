# StarGAN + Contrastive Identity Loss (CID)
### Blur-to-Sharp Attribute Translation with Biological Identity Constraint

---

## Motivation & Novelty

Standard CycleGANs/StarGANs suffer from **identity drift** — the generated
person's facial geometry subtly shifts (nose elongates, eyes migrate).  This
project adds a **Contrastive Identity Branch** using a frozen FaceNet model
to enforce a "biological constraint": the GAN may change hair/age/gender
expression but the *facial recognition embedding* must remain close to the
original.

### What makes this different from vanilla StarGAN?

| Feature | Vanilla StarGAN | This model |
|---------|----------------|------------|
| Input   | Clean image    | **Blurred / degraded** image |
| Output  | Attribute-translated | Attribute-translated **+ sharp** |
| Identity loss | L1 pixel | **FaceNet cosine embedding distance** |
| Sharpness | None | **VGG perceptual loss** |
| Upsampling | Bilinear | **Pixel-Shuffle (sub-pixel conv)** |

---

## Architecture

```
  ┌─────────────────────────────────────────────────────────────┐
  │                        GENERATOR                            │
  │                                                             │
  │  x_blur (3,128,128)   attr_trg (5,) → broadcast (5,128,128)│
  │         └──────────────┘ concat (8,128,128)                 │
  │                   ↓                                         │
  │         Conv7×7 → IN → ReLU   (64 ch)                      │
  │         StrideConv → IN → ReLU (128 ch,  64×64)            │
  │         StrideConv → IN → ReLU (256 ch,  32×32)            │
  │         ×6 ResidualBlocks      (256 ch,  32×32)            │
  │         PixelShuffle↑2  (128 ch, 64×64)   ← SHARP          │
  │         PixelShuffle↑2  ( 64 ch,128×128)  ← UPSCALE        │
  │         Conv7×7 → Tanh  (  3 ch,128×128)                   │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │                      DISCRIMINATOR                          │
  │                                                             │
  │  x (3,128,128)                                              │
  │     ↓ ×6 Conv4×4-stride2-LeakyReLU                         │
  │     ↓ feat (512 ch, 2×2)                                    │
  │     ├── Conv3×3 → src_patch (1, 2×2)    ← WGAN real/fake   │
  │     └── AvgPool → Linear → cls (5,)     ← attribute logits  │
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │              CONTRASTIVE IDENTITY BRANCH (frozen)           │
  │                                                             │
  │  x_clean ──→ FaceNet ──→ e_clean (512-d, L2-norm)          │
  │  x_fake  ──→ FaceNet ──→ e_fake  (512-d, L2-norm)          │
  │                   L_id = mean(1 - cosine_sim(e_clean,e_fake))│
  └─────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────┐
  │               PERCEPTUAL LOSS BRANCH (frozen)               │
  │                                                             │
  │  x_clean ──→ VGG16 relu1_2 / relu2_2 / relu3_3             │
  │  x_fake  ──→ VGG16 relu1_2 / relu2_2 / relu3_3             │
  │                   L_perc = Σ MSE(feat_fake, feat_clean)     │
  └─────────────────────────────────────────────────────────────┘
```

---

## Loss Functions

```
L_D = L_adv_D  +  λ_cls · L_cls_real  +  λ_gp · L_gp

L_G = L_adv_G  +  λ_cls · L_cls_fake  +  λ_rec · L_rec
      + λ_id · L_id  +  λ_perc · L_perc
```

| Symbol | Name | Value | Purpose |
|--------|------|-------|---------|
| λ_cls  | Attribute classification | 1.0 | Enforce correct attributes |
| λ_rec  | Cycle L1 reconstruction | 10.0 | Prevent mode collapse |
| λ_gp   | WGAN gradient penalty | 10.0 | Training stability |
| λ_id   | **Identity contrastive** | **5.0** | **Person stays recognisable** |
| λ_perc | Perceptual (VGG) | 0.5 | Recover sharpness from blur |

---

## Setup & Training

### 1. Kaggle Dataset Required
- **CelebA** dataset: `kaggle datasets download -d jessicali9530/celeba-dataset`
- Expected structure:
  ```
  /kaggle/input/datasets/jessicali9530/celeba-dataset/
      img_align_celeba/img_align_celeba/*.jpg
      list_attr_celeba.csv
  ```

### Dataset mode used in this project
- On Kaggle, this project reads directly from mounted dataset storage:
  - `/kaggle/input/datasets/jessicali9530/celeba-dataset/...`
  - (fallback) `/kaggle/input/celeba-dataset/...`
- It does **not** stream samples over network during training.
- If paths are missing, dataloader creation now fails early with a clear error.

### Path handling when using this repo from GitHub on Kaggle
- All file paths are centralized in `config.py`.
- Auto-detection logic now does this:
  - `celeba_root`: `STARGAN_CELEBA_ROOT` (if set) → else `/kaggle/input/datasets/jessicali9530/celeba-dataset` (if present) → else `/kaggle/input/celeba-dataset` (if present) → else `./data/celeba-dataset`
  - `work_root`: `STARGAN_WORK_ROOT` (if set) → else `/kaggle/working` (if present) → else `./runs`
- Checkpoints and samples are created under:
  - `<work_root>/checkpoints`
  - `<work_root>/samples`

Optional environment overrides:
```bash
export STARGAN_CELEBA_ROOT=/kaggle/input/datasets/jessicali9530/celeba-dataset
export STARGAN_WORK_ROOT=/kaggle/working
```

### If running outside Kaggle
Download once, then train from local disk:
```bash
kaggle datasets download -d jessicali9530/celeba-dataset
unzip celeba-dataset.zip -d /path/to/celeba-dataset
```
Then point `celeba_root` in `config.py` to that local folder.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
python train.py --mode train
```

### 4. Test any saved checkpoint on test split
```bash
python train.py --mode test --checkpoint /kaggle/working/checkpoints/ckpt_ep10.pth
```

Or use the standalone test script:
```bash
python test.py --checkpoint /kaggle/working/checkpoints/ckpt_ep10.pth
python test.py --checkpoint-dir /kaggle/working/checkpoints --last-k 5
```

### 5. W&B logging
- Enable or disable with `use_wandb` in `config.py`.
- Core losses and metrics are logged during training:
  - training: `D/*`, `G/*`, learning rate
  - validation: `val/psnr`, `val/ssim`, `val/id`, `val/perc`, `val/rec`
  - test: `test/psnr`, `test/ssim`, `test/id`, `test/perc`, `test/rec`

### Progress bars
- Training, validation/test evaluation, and overfit sanity mode all use `tqdm` progress bars.
- You can disable bars by setting `use_tqdm = False` in `config.py`.

### Kaggle GPU efficiency features enabled
- AMP mixed precision (`use_amp = True`)
- cuDNN benchmark autotuning (`cudnn_benchmark = True`)
- TF32 matmul on supported GPUs (`enable_tf32 = True`)
- Channels-last tensor format for conv nets (`use_channels_last = True`)
- Optimized dataloading (`pin_memory`, `persistent_workers`, `prefetch_factor`)
- Non-blocking host→GPU copies (`non_blocking_transfer = True`)

### 6. Tiny overfit sanity test (recommended before full training)
Run a quick memorization test on a fixed tiny batch:

```bash
python train.py --mode overfit --overfit-steps 300 --overfit-samples 8 --disable-wandb
```

Expected signs the model pipeline is correct:
- `REC`, `ID`, `PERC` should trend down.
- `PSNR` should trend up.
- A visual file is saved at `/kaggle/working/samples/overfit_result.png`.

If these do not improve at all, there is likely a bug in data flow, losses, or optimizer wiring.
Or use the notebook cells documented in `train.py`.

---

## Estimated Training Time (Kaggle T4 / P100, 16 GB)

| Config | Epochs | Batch | Steps/epoch | Est. time |
|--------|--------|-------|-------------|-----------|
| Default (128px, n_critic=5) | 30 | 16 | ~2030 | ~8–10 h |
| Lighter (64px, n_critic=3) | 40 | 32 | ~2030 | ~5–6 h |

AMP (`use_amp=True`) gives ~1.5–2× speedup — enabled by default.

---

## Memory Budget (16 GB GPU)

| Component | VRAM |
|-----------|------|
| Generator (128px, bs=16) | ~1.2 GB |
| Discriminator | ~0.8 GB |
| FaceNet (frozen, bs=16) | ~0.6 GB |
| VGG16 (frozen, bs=16) | ~0.9 GB |
| Activations + gradients | ~8 GB |
| **Total (AMP)** | **~11–12 GB** |

Fits comfortably on 16 GB with AMP.  If OOM, reduce `batch_size` to 8.

---

## File Overview

```
stargan_cid/
├── config.py    — All hyperparameters in one place
├── dataset.py   — CelebA loader with on-the-fly Gaussian blur
├── models.py    — Generator (pixel-shuffle) + Discriminator (PatchGAN)
├── losses.py    — FaceNet identity, VGG perceptual, WGAN-GP, cycle
├── trainer.py   — Full training loop with AMP, sampling, checkpointing
└── train.py     — Entry-point + Kaggle notebook cell guide
```

---

## Troubleshooting

**`facenet_pytorch` import error**
```bash
pip install facenet-pytorch
```

**OOM on 16 GB GPU**
- Reduce `batch_size` to 8 in `config.py`
- Reduce `g_repeat_num` to 4

**CelebA CSV column names differ**
- The loader handles both `image_id` and `filename` as the first column.
- If attributes are in `{-1, 1}`, they are auto-converted to `{0, 1}`.

**Identity loss stays at ~1.0 for first 5 epochs**
- Normal: the generator is still learning to produce valid faces.
  It drops to ~0.05–0.15 by epoch 10–15.

**Blurred images look too degraded / not enough**
- Adjust `blur_sigma_lo` and `blur_sigma_hi` in `config.py`.
- Defaults (4.0–8.0) produce visibly blurred faces where expressions
  are unrecognisable, which is the intended degradation level.
