"""Federated learning client implementation."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from flp.core.trainer import LocalTrainer

if TYPE_CHECKING:
    from flp.experiments.config_loader import ClientConfig


class FLClient:
    """A simulated federated learning client.

    Each client holds a local dataset partition and trains a local copy
    of the global model for a fixed number of epochs per round.

    Args:
        client_id: Unique identifier for this client.
        dataset: The full dataset; ``indices`` selects this client's partition.
        indices: Data indices assigned to this client.
        model: The global model architecture (will be copied locally).
        config: Client-level training configuration.
        device: Torch device for local training.
    """

    def __init__(
        self,
        client_id: int,
        dataset: torch.utils.data.Dataset,  # type: ignore[type-arg]
        indices: list[int],
        model: nn.Module,
        config: ClientConfig,
        device: torch.device,
    ) -> None:
        self.client_id = client_id
        self.device = device
        self.config = config
        self.num_samples = len(indices)

        subset = Subset(dataset, indices)
        self.loader = DataLoader(
            subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        self.model: nn.Module = copy.deepcopy(model)
        self._trainer = LocalTrainer(
            self.model,
            device=device,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    def set_global_weights(self, global_state_dict: dict[str, torch.Tensor]) -> None:
        """Overwrite local model weights with the received global weights.

        Args:
            global_state_dict: State dict from the global model.
        """
        self.model.load_state_dict(copy.deepcopy(global_state_dict))
        self._trainer.model = self.model
        self._trainer.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.lr,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

    def train(self) -> dict[str, object]:
        """Run local training and return the updated model state dict and stats.

        Returns:
            Dictionary containing:
            - ``state_dict``: updated model weights
            - ``num_samples``: number of local training samples
            - ``loss``: average training loss
            - ``client_id``: this client's ID
        """
        stats = self._trainer.train(self.loader, epochs=self.config.local_epochs)
        return {
            "client_id": self.client_id,
            "state_dict": copy.deepcopy(self.model.state_dict()),
            "num_samples": stats["samples"],
            "loss": stats["loss"],
        }

    def evaluate(self, loader: DataLoader | None = None) -> dict[str, float]:
        """Evaluate the local model.

        Args:
            loader: Optional external loader; uses local data if None.

        Returns:
            Dictionary with ``accuracy`` and ``loss``.
        """
        eval_loader = loader if loader is not None else self.loader
        return self._trainer.evaluate(eval_loader)
