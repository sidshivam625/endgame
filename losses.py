"""
losses.py  –  All loss functions for StarGAN + Contrastive Identity.

Losses
──────
1. WGAN-GP   adversarial (Wasserstein + gradient penalty)
2. Auxiliary  attribute classification  (BCE-with-logits)
3. Cycle      L1 reconstruction
4. Identity   FaceNet cosine-embedding distance  ← THE NOVELTY
5. Perceptual VGG16 feature-matching  (sharpness recovery)

Identity Loss Design
─────────────────────
Given:
    x_in     = input image
    x_gen    = G(x_in, target_attr)          generated face
    x_clean  = original sharp image (same identity)

We compare:
    e_clean  = FaceNet(x_clean)   – 512-d L2-normalised embedding
    e_gen    = FaceNet(x_gen)     – same

Loss = 1 − cosine_similarity(e_clean, e_gen)

This acts as a "Biological Constraint": the face-recognition model must
still recognise the generated face as the same person.  Unlike a plain
reconstruction loss (L1/L2), this loss is invariant to attribute changes
(hair colour, age, etc.) because FaceNet embeddings are trained to ignore
those and focus on identity-discriminative geometry.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.transforms.functional import normalize as tv_normalize


# ─────────────────────────────── FaceNet wrapper ──────────────────────────────

class FaceNetIdentityLoss(nn.Module):
    """
    Frozen InceptionResnetV1 (pretrained on VGGFace2) used to compute
    cosine embedding distance between the original face and generated face.

    The model is NEVER updated during training.
    """

    def __init__(self, pretrained: str = "vggface2", device: torch.device = None):
        super().__init__()
        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError:
            raise ImportError(
                "facenet-pytorch is not installed.  Run:\n"
                "  pip install facenet-pytorch\n"
                "inside your Kaggle notebook."
            )

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = InceptionResnetV1(pretrained=pretrained).eval()
        # Freeze all parameters
        for p in model.parameters():
            p.requires_grad_(False)
        self.model = model.to(self.device)

        # FaceNet expects images in ~[0, 1] float, normalised per channel
        # We store the constants and do the conversion in forward()
        self.register_buffer(
            "mean", torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W) ∈ [-1, 1]   →   FaceNet-ready (B, 3, 160, 160)
        FaceNet's InceptionResnetV1 was trained on 160×160 images,
        normalised to mean=0, std=1 with (pixel-0.5)/0.5.
        """
        # [-1,1] → [0,1]
        x = (x + 1.0) / 2.0
        # Resize to 160×160
        x = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
        # Normalise: (x - 0.5) / 0.5 = 2x - 1  →  but facenet expects (x-127.5)/128
        # We approximate with standard (0.5, 0.5) normalisation which is close enough.
        x = (x - self.mean) / self.std
        return x

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised 512-d embedding. No gradient."""
        emb = self.model(self._preprocess(x))       # (B, 512)
        return F.normalize(emb, p=2, dim=1)

    def forward(
        self,
        x_gen:   torch.Tensor,   # generated face
        x_clean: torch.Tensor,   # original sharp face (same identity)
    ) -> torch.Tensor:
        """
        Returns scalar identity loss:
            L_id = mean( 1 − cosine_similarity(e_clean, e_gen) )
        Range: 0 (perfect) … 2 (opposite embeddings, should never happen).
        """
        e_clean = self.embed(x_clean)
        e_gen   = self.embed(x_gen)
        cos_sim = (e_clean * e_gen).sum(dim=1)      # (B,)  ∈ [-1, 1]
        return (1.0 - cos_sim).mean()


# ─────────────────────────────── Perceptual loss ─────────────────────────────

class VGGPerceptualLoss(nn.Module):
    """
    Feature-matching loss on intermediate VGG16 relu activations.
    Captures texture/sharpness information across multiple scales.
    """

    _LAYER_NAMES = {
         3: "relu1_2",
         8: "relu2_2",
        15: "relu3_3",
        22: "relu4_3",
    }

    def __init__(
        self,
        layers: list  = None,      # indices into vgg16.features
        weights: list = None,
        device: torch.device = None,
    ):
        super().__init__()
        layers  = layers  or [3, 8, 15]
        weights = weights or [1.0, 1.0, 1.0]
        assert len(layers) == len(weights)
        self.layer_ids = layers
        self.weights   = weights
        self.device    = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build a trimmed VGG up to the deepest requested layer
        vgg      = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
        max_idx  = max(layers) + 1
        features = list(vgg.features.children())[:max_idx]

        # Slice into sub-networks for each loss layer
        self.slices = nn.ModuleList()
        prev = 0
        for idx in sorted(layers):
            self.slices.append(nn.Sequential(*features[prev:idx + 1]))
            prev = idx + 1

        for p in self.parameters():
            p.requires_grad_(False)

        # ImageNet normalisation (VGG input space)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        )

        self.to(self.device)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """[-1, 1] → ImageNet-normalised [~-2, ~2]."""
        x = (x + 1.0) / 2.0          # → [0, 1]
        return (x - self.mean) / self.std

    def forward(
        self,
        x_gen:    torch.Tensor,   # generated image
        x_target: torch.Tensor,   # reference (clean sharp image)
    ) -> torch.Tensor:
        """
        Returns weighted sum of MSE losses across VGG feature maps.
        """
        x_gen    = self._preprocess(x_gen)
        x_target = self._preprocess(x_target.detach())  # no grad through target

        loss = torch.tensor(0.0, device=x_gen.device)
        feat_g = x_gen
        feat_t = x_target

        for w, slc in zip(self.weights, self.slices):
            feat_g = slc(feat_g)
            feat_t = slc(feat_t)
            loss   = loss + w * F.mse_loss(feat_g, feat_t.detach())

        return loss


# ─────────────────────────────── WGAN-GP losses ───────────────────────────────

def gradient_penalty(
    D: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Two-sided gradient penalty (Gulrajani et al. 2017).
    Penalises ||∇D(x̂)||₂ = 1 for interpolated samples x̂.
    """
    B = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)

    src, _ = D(interp)
    grad = torch.autograd.grad(
        outputs  = src,
        inputs   = interp,
        grad_outputs = torch.ones_like(src),
        create_graph = True,
        retain_graph = True,
    )[0]
    grad_norm = grad.reshape(B, -1).norm(2, dim=1)        # (B,)
    return ((grad_norm - 1.0) ** 2).mean()


def adv_d_loss(src_real, src_fake) -> torch.Tensor:
    """Wasserstein discriminator loss: max E[D(real)] - E[D(fake)]."""
    return -src_real.mean() + src_fake.mean()


def adv_g_loss(src_fake) -> torch.Tensor:
    """Wasserstein generator loss: max E[D(G(x))]."""
    return -src_fake.mean()


# ─────────────────────────────── Attribute cls loss ───────────────────────────

def cls_loss_real(cls_pred: torch.Tensor, attr_real: torch.Tensor) -> torch.Tensor:
    """Attribute classification loss on real images.

    Uses CE for one-hot multiclass targets (e.g., RAF-DB), else BCE for multilabel.
    """
    row_sums = attr_real.sum(dim=1)
    is_onehot_multiclass = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
    if is_onehot_multiclass:
        return F.cross_entropy(cls_pred, attr_real.argmax(dim=1))
    return F.binary_cross_entropy_with_logits(cls_pred, attr_real)


def cls_loss_fake(cls_pred: torch.Tensor, attr_target: torch.Tensor) -> torch.Tensor:
    """Attribute classification loss on generated images.

    Uses CE for one-hot multiclass targets (e.g., RAF-DB), else BCE for multilabel.
    """
    row_sums = attr_target.sum(dim=1)
    is_onehot_multiclass = torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
    if is_onehot_multiclass:
        return F.cross_entropy(cls_pred, attr_target.argmax(dim=1))
    return F.binary_cross_entropy_with_logits(cls_pred, attr_target)


# ─────────────────────────────── Cycle-reconstruction loss ────────────────────

def cycle_loss(
    x_rec: torch.Tensor,
    x_real: torch.Tensor,
) -> torch.Tensor:
    """L1 cycle consistency: G(G(x, target), source) ≈ x."""
    return F.l1_loss(x_rec, x_real)
