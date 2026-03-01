"""Local training and evaluation logic for a single federated learning client."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainResult:
    """Outcome of one local training run (possibly multiple epochs).

    Attributes:
        loss: Mean cross-entropy loss averaged over all samples in the
            final epoch.  Reflects the model's training state *after*
            the last SGD update.
        total_samples: Total number of samples seen across all epochs.
        epochs: Number of epochs completed.
    """

    loss: float
    total_samples: int
    epochs: int


@dataclass(frozen=True)
class EvalResult:
    """Outcome of evaluating a model on a dataset split.

    Attributes:
        accuracy: Fraction of correctly classified samples in [0, 1].
        loss: Mean cross-entropy loss.
        total_samples: Number of samples evaluated.
    """

    accuracy: float
    loss: float
    total_samples: int


class LocalTrainer:
    """Encapsulates the local SGD training loop for a single client.

    Keeps the model, optimiser, and loss function together so that
    ``FLClient`` only needs to call :meth:`train` / :meth:`evaluate`
    without managing PyTorch internals directly.

    Args:
        model: The neural network to train (modified in-place).
        device: Torch device used for all tensor operations.
        lr: SGD learning rate.
        momentum: SGD momentum coefficient.
        weight_decay: L2 regularisation coefficient.
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
        self.criterion = nn.CrossEntropyLoss()
        self._lr = lr
        self._momentum = momentum
        self._weight_decay = weight_decay
        self.optimizer = self._make_optimizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, loader: DataLoader, epochs: int) -> TrainResult:  # type: ignore[type-arg]
        """Run ``epochs`` passes of SGD over ``loader``.

        Args:
            loader: DataLoader wrapping the client's local dataset partition.
            epochs: Number of full passes over the local data.

        Returns:
            :class:`TrainResult` with the loss from the *final* epoch and
            the cumulative sample count across all epochs.
        """
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}.")

        self.model.train()
        self.model.to(self.device)

        cumulative_samples = 0
        final_epoch_loss = 0.0
        final_epoch_samples = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0

            for inputs, targets in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(inputs)
                loss = self.criterion(logits, targets)
                loss.backward()
                self.optimizer.step()

                batch_n = inputs.size(0)
                epoch_loss += loss.item() * batch_n
                epoch_samples += batch_n

            cumulative_samples += epoch_samples
            # Track final epoch separately for the returned loss value.
            if epoch == epochs - 1:
                final_epoch_loss = epoch_loss
                final_epoch_samples = epoch_samples

        avg_loss = (
            final_epoch_loss / final_epoch_samples
            if final_epoch_samples > 0
            else 0.0
        )
        return TrainResult(
            loss=avg_loss,
            total_samples=cumulative_samples,
            epochs=epochs,
        )

    def evaluate(self, loader: DataLoader) -> EvalResult:  # type: ignore[type-arg]
        """Evaluate the current model weights on ``loader`` without gradient tracking.

        Args:
            loader: DataLoader to evaluate on (can be local or global test set).

        Returns:
            :class:`EvalResult` with accuracy, mean loss, and sample count.
        """
        self.model.eval()
        self.model.to(self.device)

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

                batch_n = inputs.size(0)
                total_loss += loss.item() * batch_n
                correct += (logits.argmax(dim=1) == targets).sum().item()
                total += batch_n

        return EvalResult(
            accuracy=correct / total if total > 0 else 0.0,
            loss=total_loss / total if total > 0 else 0.0,
            total_samples=total,
        )

    def reset_optimizer(self) -> None:
        """Re-initialise the optimiser (called after receiving global weights)."""
        self.optimizer = self._make_optimizer()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_optimizer(self) -> torch.optim.SGD:
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self._lr,
            momentum=self._momentum,
            weight_decay=self._weight_decay,
        )
