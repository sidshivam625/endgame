"""
models.py  –  Generator, Discriminator, and helper blocks.

Generator  (StarGAN-style with pixel-shuffle sharpening)
──────────────────────────────────────────────────────────
  Input : [blurred_img (3) | target_attr broadcast (n_attrs)] → (3+n, H, W)
  Encoder : 2× strided Conv  (H→H/4, channels expand)
  Bottleneck : N ResidualBlocks
  Decoder  : 2× PixelShuffle (H/4→H, sub-pixel for sharp reconstruction)
  Output : tanh → (3, H, W) in [-1, 1]

  PixelShuffle gives significantly sharper outputs than bilinear-upsample
  because it learns separate filters for each high-frequency sub-band.

Discriminator  (PatchGAN + auxiliary attribute classifier)
───────────────────────────────────────────────────────────
  6 strided LeakyReLU layers → two heads:
      • real/fake patch logits  (no sigmoid → WGAN-GP)
      • attribute classification logits  (n_attrs outputs, BCEWithLogits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────── Helpers ──────────────────────────────────────

class ResidualBlock(nn.Module):
    """Pre-activation residual block with Instance Normalization."""

    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
        )

    def forward(self, x):
        return x + self.block(x)


def conv_norm_relu(in_c, out_c, kernel=4, stride=2, padding=1, norm=True):
    """Strided convolution + optional InstanceNorm + ReLU (encoder building block)."""
    layers = [nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_c, affine=True))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class PixelShuffleUp(nn.Module):
    """
    2× spatial upsample via pixel-shuffle (sub-pixel convolution).
    in_c  → in_c * 4 channels  →  pixel-shuffle  →  out_c channels.
    Produces much sharper textures than bilinear or transposed convolution.
    """

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv  = nn.Conv2d(in_c, out_c * 4, 3, 1, 1, bias=False)
        self.ps    = nn.PixelShuffle(2)
        self.norm  = nn.InstanceNorm2d(out_c, affine=True)
        self.act   = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.ps(self.conv(x))))


# ─────────────────────────────── Generator ────────────────────────────────────

class Generator(nn.Module):
    """
    Lightweight StarGAN-style generator with pixel-shuffle decoder.

    Parameters
    ──────────
    image_size   : spatial resolution (H = W)
    n_attrs      : number of conditioning attributes
    conv_dim     : base channel width  (default 64)
    repeat_num   : number of residual blocks  (default 6)
    """

    def __init__(
        self,
        image_size: int   = 128,
        n_attrs: int      = 5,
        conv_dim: int     = 64,
        repeat_num: int   = 6,
    ):
        super().__init__()
        self.image_size = image_size
        self.n_attrs    = n_attrs

        in_channels = 3 + n_attrs   # RGB + broadcasted attributes

        # ── Encoder ───────────────────────────────────────────────────────
        # H → H/2 → H/4,  channels: in_c → 64 → 128 → 256
        self.encoder = nn.Sequential(
            # First conv: large kernel, no stride (preserve detail)
            nn.Conv2d(in_channels, conv_dim, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True),
            nn.ReLU(inplace=True),
            # Down 1:  128 → 64
            conv_norm_relu(conv_dim,     conv_dim * 2),
            # Down 2:   64 → 32
            conv_norm_relu(conv_dim * 2, conv_dim * 4),
        )
        bottleneck_dim = conv_dim * 4   # 256

        # ── Bottleneck residual blocks ─────────────────────────────────────
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(bottleneck_dim) for _ in range(repeat_num)]
        )

        # ── Decoder (pixel-shuffle for high-frequency recovery) ───────────
        # 32 → 64 → 128
        self.decoder = nn.Sequential(
            PixelShuffleUp(bottleneck_dim, conv_dim * 2),  # 256 → 128 ch, 2×
            PixelShuffleUp(conv_dim * 2,   conv_dim),      # 128 →  64 ch, 2×
            # Final projection to RGB
            nn.Conv2d(conv_dim, 3, 7, 1, 3, bias=False),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, attr: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, 3, H, W)      blurred input
        attr : (B, n_attrs)      target attribute vector  ∈ {0, 1}
        """
        # Broadcast attr to spatial dims and concatenate channel-wise
        B, _, H, W = x.shape
        c = attr.view(B, self.n_attrs, 1, 1).expand(B, self.n_attrs, H, W)
        inp = torch.cat([x, c], dim=1)   # (B, 3+n_attrs, H, W)

        feat = self.encoder(inp)
        feat = self.bottleneck(feat)
        return self.decoder(feat)


# ─────────────────────────────── Discriminator ────────────────────────────────

class Discriminator(nn.Module):
    """
    PatchGAN discriminator with auxiliary attribute classifier.

    Outputs
    ───────
    src  : (B, 1, Ph, Pw)   real/fake patch logits  (no activation)
    cls  : (B, n_attrs)     attribute logits         (no sigmoid)
    """

    def __init__(
        self,
        image_size: int = 128,
        n_attrs: int    = 5,
        conv_dim: int   = 64,
        repeat_num: int = 6,
    ):
        super().__init__()
        self.image_size = image_size
        self.n_attrs    = n_attrs

        layers  = []
        in_c    = 3
        out_c   = conv_dim

        for i in range(repeat_num):
            layers += [
                nn.Conv2d(in_c, out_c, 4, 2, 1, bias=True),
                nn.LeakyReLU(0.01, inplace=True),
            ]
            in_c   = out_c
            out_c  = min(out_c * 2, 512)

        self.main = nn.Sequential(*layers)

        # Spatial size after repeat_num stride-2 convolutions
        patch_h = image_size // (2 ** repeat_num)   # e.g. 128//64 = 2
        patch_h = max(patch_h, 2)                   # safety floor

        # Real/fake head (patch-level)
        self.src_head = nn.Conv2d(in_c, 1, 3, 1, 1, bias=False)

        # Attribute classification head (global average → FC)
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_c, n_attrs),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """x : (B, 3, H, W) ∈ [-1, 1]"""
        feat = self.main(x)
        src  = self.src_head(feat)
        cls  = self.cls_head(feat)
        return src, cls


# ─────────────────────────────── Param count ──────────────────────────────────

def count_params(model: nn.Module, label: str = ""):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tag = f"[{label}] " if label else ""
    print(f"{tag}Total params: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M")


if __name__ == "__main__":
    # Quick smoke-test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = Generator(image_size=128, n_attrs=5).to(device)
    D = Discriminator(image_size=128, n_attrs=5).to(device)

    count_params(G, "Generator")
    count_params(D, "Discriminator")

    x = torch.randn(4, 3, 128, 128, device=device)
    a = torch.randint(0, 2, (4, 5), dtype=torch.float32, device=device)

    out = G(x, a)
    print(f"Generator output: {out.shape}")          # (4, 3, 128, 128)

    src, cls = D(out)
    print(f"Discriminator src: {src.shape}, cls: {cls.shape}")
