"""
Monte Carlo dropout uncertainty estimation.

Usage:
    from src.models.uncertainty import MCDropoutEstimator
    estimator = MCDropoutEstimator(model, n_samples=20)
    results = estimator.predict(batch_image)
    # results["mean"], results["variance"], results["entropy"], results["needs_review"]
"""
from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np


class MCDropoutEstimator:
    """
    Wraps any model to perform Monte Carlo dropout inference.
    The model must have a `enable_mc_dropout()` method (or Dropout layers).
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 20,
        entropy_threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.model = model
        self.n_samples = n_samples
        self.entropy_threshold = entropy_threshold
        self.device = device

    @torch.no_grad()
    def predict(self, image: torch.Tensor, task_key: str = "tooth_logits") -> dict:
        """
        image: (1,1,H,W) or (B,1,H,W)
        task_key: which output key to use for multitask models
        Returns dict with mean, variance, entropy, confidence, needs_review.
        """
        self.model.eval()
        if hasattr(self.model, "enable_mc_dropout"):
            self.model.enable_mc_dropout()

        image = image.to(self.device)
        samples = []

        for _ in range(self.n_samples):
            out = self.model(image)
            if isinstance(out, dict):
                logits = out[task_key]
            else:
                logits = out
            probs = torch.softmax(logits, dim=1)   # (B, C, H, W)
            samples.append(probs.cpu())

        samples_tensor = torch.stack(samples, dim=0)  # (S, B, C, H, W)
        mean_prob = samples_tensor.mean(dim=0)         # (B, C, H, W)
        var_prob = samples_tensor.var(dim=0)           # (B, C, H, W)

        # Predictive entropy: -Σ p * log(p)
        eps = 1e-8
        entropy = -(mean_prob * torch.log(mean_prob + eps)).sum(dim=1, keepdim=True)  # (B,1,H,W)

        # Pixel-wise variance (sum over classes)
        variance = var_prob.sum(dim=1, keepdim=True)  # (B,1,H,W)

        # Image-level confidence: 1 - mean entropy normalised
        max_entropy = np.log(mean_prob.shape[1])
        norm_entropy = entropy / (max_entropy + eps)
        confidence = 1.0 - norm_entropy.mean(dim=(1, 2, 3))  # (B,)

        # Needs clinician review flag
        needs_review = (norm_entropy > self.entropy_threshold).float()  # (B,1,H,W)

        return {
            "mean": mean_prob,
            "variance": variance,
            "entropy": entropy,
            "norm_entropy": norm_entropy,
            "confidence": confidence,
            "needs_review": needs_review,
            "samples": samples_tensor,
        }


def temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling for calibration."""
    return logits / temperature


class TemperatureScaler(nn.Module):
    """
    Learned temperature scaling for post-hoc calibration.
    Train on validation set only, after main model is frozen.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.1)

    def fit(self, logits: torch.Tensor, targets: torch.Tensor,
            lr: float = 0.01, max_iter: int = 50) -> float:
        """Fit temperature on validation logits/targets. Returns final temperature."""
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def eval_step():
            optimizer.zero_grad()
            loss = criterion(self(logits), targets)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        return float(self.temperature.item())
