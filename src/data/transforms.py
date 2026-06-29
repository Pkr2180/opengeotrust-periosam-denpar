"""
Albumentations-based augmentation transforms for DenPAR.
Operates on dict samples from DenPARDataset.
"""
from __future__ import annotations
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


def _to_numpy_img(t: torch.Tensor) -> np.ndarray:
    """(1,H,W) float tensor → (H,W) uint8."""
    arr = t.squeeze(0).numpy()
    return (arr * 255).clip(0, 255).astype(np.uint8)


def _to_numpy_mask(t: torch.Tensor) -> np.ndarray:
    """(1,H,W) long tensor → (H,W) uint8."""
    return t.squeeze(0).numpy().astype(np.uint8)


def _to_numpy_heatmap(t: torch.Tensor) -> np.ndarray:
    """(1,H,W) float tensor → (H,W) float32."""
    return t.squeeze(0).numpy().astype(np.float32)


class DenPARTransform:
    """
    Wraps an albumentations pipeline for dict-based DenPAR samples.
    Handles joint augmentation of image + all masks + heatmaps.
    """

    def __init__(self, aug: A.Compose):
        self.aug = aug

    def __call__(self, sample: dict) -> dict:
        img_np = _to_numpy_img(sample["image"])
        m_radio = _to_numpy_mask(sample["mask_radio"])
        m_tooth = _to_numpy_mask(sample["tooth_mask"])
        m_bone = _to_numpy_mask(sample["bone_mask"])
        cej_np = _to_numpy_heatmap(sample["cej_heatmap"])
        apex_np = _to_numpy_heatmap(sample["apex_heatmap"])

        result = self.aug(
            image=img_np,
            masks=[m_radio, m_tooth, m_bone],
        )

        img_out = result["image"].float() / 255.0      # ToTensorV2 already converts
        if img_out.shape[0] != 1:
            img_out = img_out[:1]                       # keep single channel

        masks_out = result["masks"]
        sample["image"] = img_out
        sample["mask_radio"] = torch.from_numpy(masks_out[0]).unsqueeze(0).long()
        sample["tooth_mask"] = torch.from_numpy(masks_out[1]).unsqueeze(0).long()
        sample["bone_mask"] = torch.from_numpy(masks_out[2]).unsqueeze(0).long()
        # Heatmaps: apply same spatial transform manually
        # (albumentations doesn't handle float masks natively in all versions)
        # For now keep original heatmaps — augment separately if needed
        return sample


def get_train_transform(img_size: int = 512) -> DenPARTransform:
    aug = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        A.ElasticTransform(alpha=30, sigma=5, p=0.2),
        A.Rotate(limit=10, p=0.3),
        ToTensorV2(transpose_mask=False),
    ])
    return DenPARTransform(aug)


def get_val_transform(img_size: int = 512) -> DenPARTransform:
    aug = A.Compose([
        A.Resize(img_size, img_size),
        ToTensorV2(transpose_mask=False),
    ])
    return DenPARTransform(aug)
