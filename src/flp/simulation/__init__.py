"""Simulation layer: data heterogeneity, client dropout, and communication delays.

Public API::

    from flp.simulation.partitioning import DataPartitioner, PartitionStats
    from flp.simulation.dropout import DropoutSimulator, DropoutResult, DropoutMetrics
    from flp.simulation.delay import DelaySimulator
"""

from flp.simulation.dropout import DropoutMetrics, DropoutResult, DropoutSimulator
from flp.simulation.partitioning import DataPartitioner, PartitionStats

__all__ = [
    "DataPartitioner",
    "PartitionStats",
    "DropoutSimulator",
    "DropoutResult",
    "DropoutMetrics",
]
