# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Keras 3 weight twin for :mod:`omnibias.partition` (needs the ``keras`` extra)."""

from __future__ import annotations

from omnibias.partition.keras.weights import (
    combine,
    partition_weights,
    partition_weights_arrays,
    prod_last_axis,
)

__all__ = ["combine", "partition_weights", "partition_weights_arrays", "prod_last_axis"]
