# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Per-region model registry and the one ``combine`` engine every bridge calls.

:class:`RegionModels` attaches a callable ``m_l : X -> out`` to each of the partition's
``2**depth`` regions and blends their outputs by the soft partition-of-unity weights:

.. math:: F(x) = \sum_l w_l(x)\, m_l(x).

This single ``sum_l w_l * out_l`` blend is the shared engine behind every downstream bridge
(the discontinuity PINN's ``u = sum_l w_l u_l``, the atlas's ``g = sum_l w_l G_l``, the
piecewise-symbolic mixture, and the decision layer's relaxed choice). The numpy
:func:`combine_outputs` is the reference; the torch / jax weight twins reproduce the same
blend with their own arrays.

Terminology: the weights harden as ``beta -> inf`` -- the feasibility / temperature sense of
"collapse", distinct from the **founding bias collapse** (the multi-bias ``delta -> 0``
limit to the closed-form derivative ``sigma^(K-1)``; see ``docs/theory.md``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from omnibias.partition._core.params import FloatArray, PartitionParams
from omnibias.partition._core.weights import partition_weights

RegionModel = Callable[[FloatArray], FloatArray]


def combine_outputs(weights: FloatArray, region_outputs: FloatArray) -> FloatArray:
    r"""Blend per-region outputs by partition weights: ``F[n, k] = sum_l w[n, l] out[n, l, k]``.

    Parameters
    ----------
    weights:
        Partition-of-unity weights ``(n, n_regions)`` (rows non-negative, summing to one).
    region_outputs:
        Per-region outputs ``(n, n_regions, k)`` (or ``(n, n_regions)`` for scalar outputs).
    """
    w = np.asarray(weights, dtype=np.float64)
    outs = np.asarray(region_outputs, dtype=np.float64)
    if outs.ndim == 2:  # scalar per region -> (n, n_regions)
        return np.einsum("nl,nl->n", w, outs)
    return np.einsum("nl,nlk->nk", w, outs)


@dataclass
class RegionModels:
    r"""A partition plus one callable per region; :meth:`combine` blends them by the POU."""

    params: PartitionParams
    models: Sequence[RegionModel]

    def __post_init__(self) -> None:
        if len(self.models) != self.params.n_regions:
            raise ValueError(
                f"expected {self.params.n_regions} region models (2**depth), "
                f"got {len(self.models)}"
            )

    def region_outputs(self, X: FloatArray) -> FloatArray:
        r"""Stack each region model's output -> ``(n, n_regions, k)``."""
        Xv = np.asarray(X, dtype=np.float64)
        outs = []
        for m in self.models:
            o = np.asarray(m(Xv), dtype=np.float64)
            outs.append(o.reshape(o.shape[0], -1))
        return np.stack(outs, axis=1)  # (n, L, k)

    def combine(self, X: FloatArray, beta: float | None = None) -> FloatArray:
        r"""Evaluate ``F(x) = sum_l w_l(x) m_l(x)`` at gate sharpness ``beta``."""
        b = float(self.params.config.beta_final if beta is None else beta)
        Xv = np.asarray(X, dtype=np.float64)
        w = partition_weights(self.params, Xv, b)  # (n, L)
        outs = self.region_outputs(Xv)  # (n, L, k)
        out = combine_outputs(w, outs)
        return out[:, 0] if (out.ndim == 2 and out.shape[1] == 1) else out


__all__ = ["RegionModel", "RegionModels", "combine_outputs"]
