"""Neural network architectures for MNIST federated learning experiments."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

Architecture = Literal["cnn", "mlp"]


class MNISTConvNet(nn.Module):
    """Two-block convolutional network for MNIST (28x28 greyscale, 10 classes).

    Architecture::

        Conv(1→32, 3x3, pad=1) → ReLU → MaxPool(2)   # → 32 x 14 x 14
        Conv(32→64, 3x3, pad=1) → ReLU → MaxPool(2)  # → 64 x 7 x 7
        Flatten → Linear(3136→256) → ReLU → Dropout(0.3)
        Linear(256→10)

    Parameter count: ~863 K
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class MNISTMlp(nn.Module):
    """Three-layer fully-connected MLP for MNIST (28x28 greyscale, 10 classes).

    Architecture::

        Flatten → Linear(784→256) → ReLU → Dropout(0.2)
        Linear(256→128) → ReLU → Dropout(0.2)
        Linear(128→10)

    Lighter than the CNN; useful for fast iteration and CPU-only environments.
    Parameter count: ~235 K
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(architecture: Architecture = "cnn") -> nn.Module:
    """Instantiate a fresh, randomly-initialised model for MNIST.

    Args:
        architecture: ``"cnn"`` for :class:`MNISTConvNet` (default) or
            ``"mlp"`` for :class:`MNISTMlp`.

    Returns:
        Uninitialised (random weights) model instance.

    Raises:
        ValueError: If ``architecture`` is not a known option.
    """
    if architecture == "cnn":
        return MNISTConvNet()
    if architecture == "mlp":
        return MNISTMlp()
    raise ValueError(
        f"Unknown architecture '{architecture}'. Choose 'cnn' or 'mlp'."
    )
