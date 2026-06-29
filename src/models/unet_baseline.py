"""
Lightweight UNet baseline using segmentation-models-pytorch.

Supports:
  - tooth segmentation (binary)
  - bone-line segmentation (binary, thin structure)
  - keypoint heatmap regression (2 channels: CEJ, apex)
  - multi-task (all heads simultaneously)

Usage:
    from src.models.unet_baseline import build_unet
    model = build_unet(task="tooth_seg")
"""
from __future__ import annotations
import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
    _SMP_AVAILABLE = True
except ImportError:
    _SMP_AVAILABLE = False

# ──────────────────────────────────────────────
# Fallback minimal UNet (no smp required)
# ──────────────────────────────────────────────

class _ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = _ConvBnRelu(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MinimalUNet(nn.Module):
    """Pure-PyTorch UNet — no extra dependencies. Useful for dry runs."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 base_filters: int = 32, dropout: float = 0.3):
        super().__init__()
        f = base_filters
        self.enc1 = _ConvBnRelu(in_channels, f)
        self.enc2 = _ConvBnRelu(f, f * 2)
        self.enc3 = _ConvBnRelu(f * 2, f * 4)
        self.enc4 = _ConvBnRelu(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _ConvBnRelu(f * 8, f * 16)
        self.drop = nn.Dropout2d(p=dropout)
        self.up4 = _UpBlock(f * 16, f * 8, f * 8)
        self.up3 = _UpBlock(f * 8, f * 4, f * 4)
        self.up2 = _UpBlock(f * 4, f * 2, f * 2)
        self.up1 = _UpBlock(f * 2, f, f)
        self.head = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.drop(self.pool(e4)))
        d = self.up4(b, e4)
        d = self.up3(d, e3)
        d = self.up2(d, e2)
        d = self.up1(d, e1)
        return self.head(d)

    def enable_mc_dropout(self):
        """Switch Dropout layers to training mode for MC inference."""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()


# ──────────────────────────────────────────────
# SMP-backed UNet (preferred when available)
# ──────────────────────────────────────────────

class SMPUNet(nn.Module):
    """
    segmentation-models-pytorch UNet with configurable encoder.
    Supports: resnet34, efficientnet-b0, vgg16, etc.
    """

    def __init__(
        self,
        encoder: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 1,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        if not _SMP_AVAILABLE:
            raise ImportError("segmentation-models-pytorch not installed. "
                              "pip install segmentation-models-pytorch")
        self.unet = smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )
        self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x):
        # SMP 0.5+ changed decoder API — use full forward pass then apply dropout
        logits = self.unet(x)
        return self.dropout(logits)

    def enable_mc_dropout(self):
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()


# ──────────────────────────────────────────────
# Multi-task UNet
# ──────────────────────────────────────────────

class MultiTaskUNet(nn.Module):
    """
    Shared encoder with three separate decoder heads:
      1. tooth segmentation
      2. bone-line segmentation
      3. keypoint heatmaps (CEJ + apex = 2 channels)

    Uses MinimalUNet encoders for portability.
    Replace with SMP encoder for better performance.
    """

    def __init__(self, in_channels: int = 1, base_filters: int = 32, dropout: float = 0.3):
        super().__init__()
        f = base_filters
        # Shared encoder
        self.enc1 = _ConvBnRelu(in_channels, f)
        self.enc2 = _ConvBnRelu(f, f * 2)
        self.enc3 = _ConvBnRelu(f * 2, f * 4)
        self.enc4 = _ConvBnRelu(f * 4, f * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _ConvBnRelu(f * 8, f * 16)
        self.drop = nn.Dropout2d(p=dropout)

        # Task-specific decoders
        self.tooth_decoder = self._make_decoder(f)
        self.bone_decoder = self._make_decoder(f)
        self.kp_decoder = self._make_decoder(f)

        # Task heads
        self.tooth_head = nn.Conv2d(f, 2, 1)    # bg + tooth
        self.bone_head = nn.Conv2d(f, 2, 1)     # bg + bone line
        self.kp_head = nn.Conv2d(f, 2, 1)       # CEJ + apex heatmaps

    def _make_decoder(self, f):
        return nn.ModuleList([
            _UpBlock(f * 16, f * 8, f * 8),
            _UpBlock(f * 8, f * 4, f * 4),
            _UpBlock(f * 4, f * 2, f * 2),
            _UpBlock(f * 2, f, f),
        ])

    def _decode(self, decoder, b, skips):
        x = b
        for up, skip in zip(decoder, reversed(skips)):
            x = up(x, skip)
        return x

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.drop(self.bottleneck(self.pool(e4)))
        skips = [e1, e2, e3, e4]

        tooth_feat = self._decode(self.tooth_decoder, b, skips)
        bone_feat = self._decode(self.bone_decoder, b, skips)
        kp_feat = self._decode(self.kp_decoder, b, skips)

        return {
            "tooth_logits": self.tooth_head(tooth_feat),
            "bone_logits": self.bone_head(bone_feat),
            "kp_heatmaps": torch.sigmoid(self.kp_head(kp_feat)),  # (B,2,H,W)
        }

    def enable_mc_dropout(self):
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

def build_unet(
    task: str = "tooth_seg",
    encoder: str = "resnet34",
    in_channels: int = 1,
    dropout: float = 0.3,
    use_smp: bool = True,
) -> nn.Module:
    """
    task: "tooth_seg" | "bone_seg" | "keypoints" | "multitask"
    """
    if task == "multitask":
        return MultiTaskUNet(in_channels=in_channels, dropout=dropout)

    num_classes = 2 if task in ("tooth_seg", "bone_seg") else 2  # 2 kp channels too

    if use_smp and _SMP_AVAILABLE:
        try:
            return SMPUNet(encoder=encoder, in_channels=in_channels,
                           num_classes=num_classes, dropout=dropout)
        except Exception as e:
            print(f"  [WARN] SMP UNet failed ({e}), falling back to MinimalUNet.")

    return MinimalUNet(in_channels=in_channels, num_classes=num_classes, dropout=dropout)


# ──────────────────────────────────────────────
# Quick smoke test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing MinimalUNet...")
    m = build_unet("tooth_seg", use_smp=False)
    x = torch.randn(2, 1, 256, 256)
    out = m(x)
    print(f"  Input: {x.shape}  Output: {out.shape}")

    print("Testing MultiTaskUNet...")
    mt = build_unet("multitask", use_smp=False)
    out_mt = mt(x)
    for k, v in out_mt.items():
        print(f"  {k}: {v.shape}")
    print("All model tests passed.")
