# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""JAX discontinuity-capturing PINN field on omnibias-partition (needs the ``jax`` extra)."""

from __future__ import annotations

from importlib.util import find_spec

if find_spec("omnibias.partition") is None:  # the optional ``partition`` extra is not installed
    raise ImportError(
        "omnibias.pinn.partition requires the optional 'omnibias-partition' package. "
        "Install it with:  pip install 'omnibias-pinn[partition]'"
    )

from omnibias.pinn.partition.jax.field import (  # noqa: E402
    PartitionedField,
    build_partitioned_field,
)

__all__ = ["PartitionedField", "build_partitioned_field"]
