# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Sound certificate for a soft partition of unity.

:func:`certify_partition_gap` bundles the certified **soft->hard collapse** guarantees a
partition can *earn* (not just assert) as ``beta -> inf``:

* a sound per-sample **L1 membership gap** bound ``sum_l |w_soft_l - w_hard_l|`` (via the
  outward-rounded total-variation argument in :mod:`omnibias.partition._core.verified`),
  aggregated to its max / mean over a sample set, with the measured value for the
  ``is_sound`` self-check;
* the closed-form ``log(n_regions)/beta`` Gibbs-to-Dirac scale as a reference.

Honesty (the discrete consumers' yes-if framing): the bound is a genuine sound enclosure of
the collapse gap, never an exact-optimality claim, and the gap is never asserted zero.

Terminology: the ``beta -> inf`` gate hardening is the feasibility / temperature sense of
"collapse", distinct from the **founding bias collapse** (the multi-bias ``delta -> 0``
limit to ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from omnibias.partition._core.params import PartitionParams
from omnibias.partition._core.verified import gibbs_gap_bound, weight_rounding_gap


@dataclass(frozen=True)
class PartitionGapCertificate:
    r"""A certified bound on the soft->hard partition-of-unity gap as ``beta -> inf``."""

    beta: float
    n_regions: int
    max_gap: float
    mean_gap: float
    measured_max: float
    gibbs_scale: float

    @property
    def is_sound(self) -> bool:
        r"""The certified bound must dominate the actually-measured L1 gap (self-check)."""
        return self.max_gap >= self.measured_max - 1e-9

    @property
    def certified(self) -> bool:
        r"""Alias of :attr:`is_sound` (a yes-if verdict: the enclosure holds)."""
        return self.is_sound


def certify_partition_gap(
    params: PartitionParams,
    X: np.ndarray,
    *,
    beta: float | None = None,
) -> PartitionGapCertificate:
    r"""Certified soft->hard membership gap on the samples ``X``.

    Returns a sound per-sample bound on the L1 distance between the soft partition-of-unity
    weights and their crisp ``beta -> inf`` one-hot limit, aggregated to its max / mean over
    ``X`` (with the measured gap for :attr:`PartitionGapCertificate.is_sound`), plus the
    ``log(n_regions)/beta`` Gibbs scale.
    """
    b = float(params.config.beta_final if beta is None else beta)
    bound, measured = weight_rounding_gap(params, X, b)
    return PartitionGapCertificate(
        beta=b,
        n_regions=params.n_regions,
        max_gap=float(bound.max()),
        mean_gap=float(bound.mean()),
        measured_max=float(measured.max()),
        gibbs_scale=gibbs_gap_bound(params.n_regions, b),
    )


__all__ = ["PartitionGapCertificate", "certify_partition_gap"]
