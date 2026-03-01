"""Core federated learning engine.

Public API::

    from flp.core.models import build_model, MNISTConvNet, MNISTMlp
    from flp.core.trainer import LocalTrainer, TrainResult, EvalResult
    from flp.core.client import FLClient, ClientUpdate
    from flp.core.aggregator import FedAvgAggregator, AggregationResult
    from flp.core.server import FLServer, RoundSummary
"""

from flp.core.aggregator import AggregationResult, FedAvgAggregator
from flp.core.client import ClientUpdate, FLClient
from flp.core.models import MNISTConvNet, MNISTMlp, build_model
from flp.core.server import FLServer, RoundSummary
from flp.core.trainer import EvalResult, LocalTrainer, TrainResult

__all__ = [
    "build_model",
    "MNISTConvNet",
    "MNISTMlp",
    "LocalTrainer",
    "TrainResult",
    "EvalResult",
    "FLClient",
    "ClientUpdate",
    "FedAvgAggregator",
    "AggregationResult",
    "FLServer",
    "RoundSummary",
]
