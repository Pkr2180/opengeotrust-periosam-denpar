"""
PerioSAM-Lite: Foundation-model-compatible interface.

Stage 1 (current): UNet backbone with a SAM-compatible interface.
Stage 2 (TODO): Replace encoder with frozen MobileSAM/MedSAM/SAM2 + LoRA adapters.

This file provides:
  - PerioSAMLite: clean interface matching the SAM prompt-based paradigm
  - LoRALayer: lightweight adapter block for future foundation encoder adaptation
  - TODO markers for SAM2/MedSAM integration

DO NOT load large foundation model weights here until baseline is validated.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.unet_baseline import MinimalUNet, build_unet


# ──────────────────────────────────────────────
# LoRA-style adapter (for future use)
# ──────────────────────────────────────────────

class LoRALayer(nn.Module):
    """
    Low-rank adaptation for Conv2d or Linear layers.
    Adds a trainable delta W = B @ A scaled by alpha/r.
    TODO: attach to frozen SAM image encoder attention layers.
    """

    def __init__(self, in_features: int, out_features: int, r: int = 4, alpha: float = 1.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scaling


# ──────────────────────────────────────────────
# Prompt encoder (point/box prompts → embeddings)
# ──────────────────────────────────────────────

class LightweightPromptEncoder(nn.Module):
    """
    Encodes sparse prompts (points, boxes) into spatial embeddings.
    Mimics SAM's prompt encoder interface with minimal overhead.
    """

    def __init__(self, embed_dim: int = 64, img_size: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.point_embed = nn.Embedding(3, embed_dim)   # 0=pad, 1=pos, 2=neg
        self.box_embed = nn.Linear(4, embed_dim)

    def forward_points(self, points: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """points: (B, N, 2) normalised [0,1]; labels: (B, N) ∈ {0,1,2}."""
        return self.point_embed(labels)   # (B, N, embed_dim)

    def forward_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """boxes: (B, M, 4) normalised [x1,y1,x2,y2]."""
        return self.box_embed(boxes)      # (B, M, embed_dim)


# ──────────────────────────────────────────────
# PerioSAM-Lite main model
# ──────────────────────────────────────────────

class PerioSAMLite(nn.Module):
    """
    OpenGeoTrust-PerioSAM architecture (Stage 1).

    Architecture:
      Encoder  : ResNet34-UNet (SMP) or MinimalUNet
      Decoder  : Multi-task heads (tooth, bone, keypoint)
      Prompts  : Optional point/box prompt embedding injected at bottleneck
      Geometry : Downstream computation (see geometry_head.py)
      Uncert.  : MC-Dropout (see uncertainty.py)

    TODO (Stage 2):
      - Replace encoder with frozen MobileSAM image encoder
      - Add LoRA adapters to ViT attention Q/V projections
      - Use SAM's prompt encoder directly
      - Reference: https://github.com/bowang-lab/MedSAM (MedSAM)
      - Reference: https://github.com/ChaoningZhang/MobileSAM
      - Reference: https://github.com/facebookresearch/segment-anything-2
    """

    def __init__(
        self,
        encoder: str = "resnet34",
        in_channels: int = 1,
        dropout: float = 0.3,
        use_prompt_encoder: bool = False,
        img_size: int = 512,
    ):
        super().__init__()
        self.img_size = img_size
        self.use_prompt_encoder = use_prompt_encoder

        # Stage 1: UNet backbone
        self.backbone = build_unet("multitask", encoder=encoder,
                                   in_channels=in_channels, dropout=dropout)

        if use_prompt_encoder:
            self.prompt_encoder = LightweightPromptEncoder(img_size=img_size)

    def forward(
        self,
        image: torch.Tensor,
        point_coords: torch.Tensor | None = None,
        point_labels: torch.Tensor | None = None,
        boxes: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Stage 1: pure image forward
        return self.backbone(image)

    def enable_mc_dropout(self):
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train()

    @staticmethod
    def from_config(cfg: dict) -> "PerioSAMLite":
        return PerioSAMLite(
            encoder=cfg.get("encoder", "resnet34"),
            in_channels=cfg.get("in_channels", 1),
            dropout=cfg.get("dropout_rate", 0.3),
            img_size=cfg.get("img_size", 512),
        )


if __name__ == "__main__":
    model = PerioSAMLite(use_prompt_encoder=False)
    x = torch.randn(2, 1, 256, 256)
    out = model(x)
    for k, v in out.items():
        print(f"  {k}: {v.shape}")
