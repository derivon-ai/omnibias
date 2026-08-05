# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""JAX weight twin for :mod:`omnibias.partition` (needs the ``jax`` extra)."""

from __future__ import annotations

from omnibias.partition.jax.weights import (
    combine,
    partition_weights,
    partition_weights_arrays,
)

__all__ = ["combine", "partition_weights", "partition_weights_arrays"]
