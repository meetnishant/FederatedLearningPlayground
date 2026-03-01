"""Federated learning client: local data, local model, local training."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from flp.core.trainer import EvalResult, LocalTrainer, TrainResult

if TYPE_CHECKING:
    from flp.experiments.config_loader import ClientConfig


@dataclass
class ClientUpdate:
    """Everything the server needs from one client after a local training round.

    Attributes:
        client_id: Unique client identifier.
        state_dict: Model weights after local training (CPU tensors).
        num_samples: Number of samples used for training this round.
        train_result: Detailed training statistics from :class:`~flp.core.trainer.LocalTrainer`.
    """

    client_id: int
    state_dict: dict[str, torch.Tensor]
    num_samples: int
    train_result: TrainResult


class FLClient:
    """A simulated federated learning client.

    Responsibilities:

    - Holds a fixed local partition of the training dataset.
    - Maintains a local copy of the global model.
    - On each round: receives global weights → runs local SGD → returns update.

    Reproducibility is guaranteed by seeding the DataLoader worker function
    with ``seed ^ client_id``, so every client shuffles its data differently
    but deterministically across runs with the same seed.

    Args:
        client_id: Unique non-negative integer identifying this client.
        dataset: Full training dataset; this client trains only on ``indices``.
        indices: Dataset indices assigned to this client's partition.
        model: Global model architecture.  Deep-copied internally; the
            caller's instance is never mutated.
        config: Per-client training hyperparameters.
        device: Torch device for local training.
        seed: Base random seed for reproducible DataLoader shuffling.
    """

    def __init__(
        self,
        client_id: int,
        dataset: Dataset,  # type: ignore[type-arg]
        indices: list[int],
        model: nn.Module,
        config: ClientConfig,
        device: torch.device,
        seed: int = 42,
    ) -> None:
        self.client_id = client_id
        self.device = device
        self.config = config
        self.num_samples = len(indices)

        if self.num_samples == 0:
            raise ValueError(
                f"Client {client_id} was assigned zero data samples. "
                "Check your partitioning strategy and num_clients setting."
            )

        # Deterministic per-client DataLoader: each client gets a unique seed
        # derived from the global seed XOR'd with its ID, so shuffles differ
        # between clients but are reproducible across runs.
        client_seed = seed ^ client_id
        generator = torch.Generator()
        generator.manual_seed(client_seed)

        def _worker_init(worker_id: int) -> None:
            import numpy as np
            import random
            worker_seed = client_seed + worker_id
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        self.loader: DataLoader = DataLoader(  # type: ignore[type-arg]
            Subset(dataset, indices),
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
            generator=generator,
            worker_init_fn=_worker_init,
        )

        # Local model is always a deep copy of the global model so that
        # clients start each round from the same global state.
        self.model: nn.Module = copy.deepcopy(model)
        self._trainer = LocalTrainer(
            model=self.model,
            device=device,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    # ------------------------------------------------------------------
    # Round interface
    # ------------------------------------------------------------------

    def set_global_weights(self, global_state: dict[str, torch.Tensor]) -> None:
        """Overwrite local model weights with the server's global state dict.

        Also resets the SGD optimiser so momentum buffers from the previous
        round do not contaminate the new round.

        Args:
            global_state: State dict from the server's current global model.
        """
        self.model.load_state_dict(copy.deepcopy(global_state))
        # Reset optimiser: momentum accumulated in previous rounds is stale
        # after receiving new global weights.
        self._trainer.reset_optimizer()

    def train(self) -> ClientUpdate:
        """Run local SGD and return the updated weights and training stats.

        This is the only method the server calls during each round.

        Returns:
            :class:`ClientUpdate` containing the updated state dict (on CPU),
            sample count, and detailed training statistics.
        """
        result: TrainResult = self._trainer.train(
            self.loader,
            epochs=self.config.local_epochs,
        )
        # Move state dict to CPU before sending to server to avoid GPU memory
        # accumulation when many clients share a single GPU.
        state_dict = {
            k: v.cpu() for k, v in self.model.state_dict().items()
        }
        return ClientUpdate(
            client_id=self.client_id,
            state_dict=state_dict,
            num_samples=result.total_samples // self.config.local_epochs,
            train_result=result,
        )

    def evaluate(self, loader: DataLoader | None = None) -> EvalResult:  # type: ignore[type-arg]
        """Evaluate the local model on ``loader`` (or on local data if None).

        Args:
            loader: DataLoader to evaluate on.  Defaults to this client's
                own data partition.

        Returns:
            :class:`~flp.core.trainer.EvalResult` with accuracy and loss.
        """
        return self._trainer.evaluate(loader if loader is not None else self.loader)
