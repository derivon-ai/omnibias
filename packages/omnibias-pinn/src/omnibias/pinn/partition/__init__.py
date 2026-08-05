# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""omnibias.pinn.partition -- discontinuity-capturing PINNs on omnibias-partition.

A single smooth activation network cannot represent a kink / shock / phase boundary; a
**soft partition of unity** of smooth sub-solutions can. This submodule (a bridge on the
:mod:`omnibias.partition` keystone) provides :class:`~omnibias.pinn.partition.torch.PartitionedField`,
a genuine PINN field ``u(x) = sum_l w_l(x) u_l(x)`` that plugs into the existing ops and
develops a genuine interface between regions as the gate sharpness ``beta -> inf``.

Honesty: the blended field's derivatives use the **autodiff product rule** (the closed-form
``sigma``-tower does not cover products of sigmoids); the *soft->hard partition gap* is the
sound, certified quantity (via :func:`omnibias.partition.certify_partition_gap`). The
``beta -> inf`` hardening is the feasibility / temperature sense of "collapse", distinct
from the **founding bias collapse** (the multi-bias ``delta -> 0`` limit to the closed-form
derivative ``sigma^(K-1)``; see ``docs/theory.md``).

Both backends ship a bit-identical twin: :mod:`omnibias.pinn.partition.torch` and
:mod:`omnibias.pinn.partition.jax`. The top-level names below resolve to the torch twin
for backward compatibility; import ``omnibias.pinn.partition.jax`` explicitly for JAX.

Maturity: alpha (a bridge on the alpha ``omnibias-partition`` keystone), inside the beta
``omnibias-pinn`` package. Needs the ``torch`` (or ``jax``) extra plus ``[partition]``.
"""

from __future__ import annotations

__all__ = ["PartitionedField", "build_partitioned_field"]


def __getattr__(name: str) -> object:  # pragma: no cover - thin lazy backend shim
    if name in __all__:
        from omnibias.pinn.partition import torch as _torch

        return getattr(_torch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
