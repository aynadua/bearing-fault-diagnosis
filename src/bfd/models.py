"""1D CNN over a vibration record.

Deliberately the same architecture for the raw-waveform and order-tracked
experiments. If the two score differently, the cause is the representation,
not the network -- which is the only way the comparison means anything.
"""

from __future__ import annotations

import torch
from torch import nn


class VibrationCNN(nn.Module):
    """Five strided convolution blocks, global pooling, linear head."""

    def __init__(self, n_classes: int = 4, width: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        w = width

        def block(c_in: int, c_out: int, kernel: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2, bias=False),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(4),
            )

        self.features = nn.Sequential(
            block(1, w, 15),
            block(w, w, 9),
            block(w, 2 * w, 5),
            block(2 * w, 2 * w, 3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * w, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, 1, length)`` vibration to class logits."""
        z = self.features(x)
        z = self.pool(z).flatten(1)
        return self.head(z)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
