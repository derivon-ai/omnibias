# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Information-theory metadata + scalar primitives (pure Python).

Backend-agnostic helpers that sit between omnibias's probability layer
(:mod:`omnibias.core.probability`) and the differentiable information operators
in ``omnibias.{torch,jax}.information``:

* :func:`binary_entropy` -- the Bernoulli entropy ``H(p)`` in nats (or a chosen
  logarithm base), the scalar reference for the batched backend ``entropy``.
* :func:`is_log_partition_activation` / :func:`has_cumulant_tower` --
  exponential-family metadata. An activation tagged with a GLM ``noise_model`` is
  (interpreted as) a log-partition ``A(theta)``; its closed-form derivative tower
  is then the *cumulant tower* ``kappa_k = A^(k)(theta)`` (mean ``= A'``,
  variance ``= Fisher information ``= A''``, ...), evaluated exactly by the
  backend ``exponential_family_cumulants`` from a single closed-form call.

No tensor backend is imported, matching the ``omnibias.core`` zero-dependency
contract.
"""

from __future__ import annotations

import math
from typing import Any

from omnibias.core.spec import ActivationSpec


def binary_entropy(p: float, *, base: float | None = None) -> float:
    r"""Bernoulli entropy ``H(p) = -p ln p - (1-p) ln(1-p)`` (nats by default).

    With ``base`` given (e.g. ``2`` for bits, ``10`` for bans) the result is
    divided by ``ln(base)``. ``p`` must lie in ``[0, 1]``; the endpoints return
    ``0`` exactly.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0, 1], got {p}")
    if p == 0.0 or p == 1.0:
        return 0.0
    h = -(p * math.log(p) + (1.0 - p) * math.log1p(-p))
    if base is not None:
        if base <= 1.0:
            raise ValueError(f"base must be > 1, got {base}")
        h /= math.log(base)
    return h


def is_log_partition_activation(spec: ActivationSpec[Any]) -> bool:
    """``True`` when ``spec`` is tagged as a GLM log-partition (``noise_model``).

    The recorded ``noise_model`` (e.g. ``"bernoulli"`` for ``softplus``) means
    ``sigma`` arises as the log-partition of an exponential family, so its
    derivative tower is the family's cumulant tower.
    """
    return spec.noise_model not in ("", "none")


def has_cumulant_tower(spec: ActivationSpec[Any]) -> bool:
    """``True`` when ``spec`` exposes a derivative fastpath (closed-form cumulants).

    A log-partition activation needs its ``fastpath`` to yield ``A^(k)`` for the
    cumulants; activations without one cannot form the exact tower.
    """
    return spec.fastpath is not None


__all__ = [
    "binary_entropy",
    "has_cumulant_tower",
    "is_log_partition_activation",
]
