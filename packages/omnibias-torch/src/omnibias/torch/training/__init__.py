# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Training-loop helpers for omnibias.

Currently exports:

- :class:`KGrowthScheduler`: plateau-triggered K-growth callback for
  :class:`GrowableOperatorMultiBiasUnit` modules.
"""

from omnibias.torch.training.k_scheduler import (
    AnchorProvider,
    GrowthEvent,
    KGrowthScheduler,
)

__all__ = ["AnchorProvider", "GrowthEvent", "KGrowthScheduler"]
