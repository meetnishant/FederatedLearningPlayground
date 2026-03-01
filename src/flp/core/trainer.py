"""Local training logic for federated learning clients."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class LocalTrainer:
    """Handles local model training on a single client's dataset.

    Args:
        model: The neural network model to train.
        device: The torch device to use for training.
        lr: Learning rate for the optimizer.
        momentum: Momentum for SGD optimizer.
        weight_decay: L2 regularization coefficient.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ) -> None:
        self.model = model
        self.device = device
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()

    def train(self, loader: DataLoader, epochs: int) -> dict[str, float]:
        """Train the model for a given number of local epochs.

        Args:
            loader: DataLoader for the client's local dataset.
            epochs: Number of local training epochs.

        Returns:
            Dictionary with ``loss`` (final epoch avg loss) and ``samples`` count.
        """
        self.model.train()
        self.model.to(self.device)

        total_loss = 0.0
        total_samples = 0

        for _ in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)
                epoch_samples += inputs.size(0)
            total_loss = epoch_loss
            total_samples = epoch_samples

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return {"loss": avg_loss, "samples": total_samples}

    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        """Evaluate the model on a given dataset.

        Args:
            loader: DataLoader to evaluate on.

        Returns:
            Dictionary with ``accuracy`` and ``loss``.
        """
        self.model.eval()
        self.model.to(self.device)

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += inputs.size(0)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total if total > 0 else 0.0
        return {"accuracy": accuracy, "loss": avg_loss}
