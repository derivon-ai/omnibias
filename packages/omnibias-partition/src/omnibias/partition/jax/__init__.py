# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""JAX weight twin for :mod:`omnibias.partition` (needs the ``jax`` extra)."""

from __future__ import annotations

from omnibias.partition.jax.weights import (
    combine,
    partition_weights,
    partition_weights_arrays,
)

__all__ = ["combine", "partition_weights", "partition_weights_arrays"]
