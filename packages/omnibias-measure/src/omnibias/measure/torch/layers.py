# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Trainable ``nn.Module`` wrappers for the measure-integral primitives.

Each layer pins a :class:`~omnibias.measure._core.measure.Measure`'s nodes as a
buffer and exposes its weights (and, for the layer-cake, the softness ``beta``)
as optional learnable parameters, so an integral against a measure becomes a
trainable component of a network:

* :class:`LebesgueIntegral` -- ``int f dmu`` as a readout / pooling layer.
* :class:`ExpectationLayer` -- a (self-normalized) importance-sampling
  expectation with a learnable proposal reweighting.
* :class:`LayerCakeIntegral` -- the distribution-function integral with a
  learnable level-set softness ``beta`` (positive by construction via
  ``beta = exp(log_beta)``).

The ``forward`` of each accepts either a callable ``f`` (evaluated at the pinned
nodes -- e.g. a sub-network) or a precomputed value tensor at the nodes.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from omnibias.measure._core.measure import Measure
from omnibias.measure.torch import ops
from torch import Tensor, nn

FOrValues = Tensor | Callable[[Tensor], Tensor]


def _as_values(f: FOrValues, nodes: Tensor) -> Tensor:
    return f(nodes) if callable(f) else f


class _MeasureLayer(nn.Module):
    """Shared node/weight bookkeeping for the measure layers."""

    nodes: Tensor
    weight: Tensor

    def __init__(
        self,
        measure: Measure,
        *,
        learnable_weights: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        dt = dtype if dtype is not None else torch.get_default_dtype()
        self.register_buffer("nodes", torch.as_tensor(measure.nodes, dtype=dt))
        w0 = torch.as_tensor(measure.weights, dtype=dt)
        if learnable_weights:
            self.weight = nn.Parameter(w0)
        else:
            self.register_buffer("weight", w0)
        self.measure_name = measure.name

    def extra_repr(self) -> str:
        return f"n_nodes={self.nodes.shape[0]}, dim={self.nodes.shape[1]}"


class LebesgueIntegral(_MeasureLayer):
    """``int f dmu`` as a trainable readout layer (optionally learnable weights)."""

    def forward(self, f: FOrValues) -> Tensor:
        vals = _as_values(f, self.nodes)
        out: Tensor = ops.lebesgue_integral(
            lambda _n: vals, nodes=self.nodes, weights=self.weight
        )
        return out


class ExpectationLayer(_MeasureLayer):
    r"""(Self-normalized) importance-sampling expectation ``E_p[f]``.

    The pinned measure is the proposal ``q``; ``forward`` takes ``f`` and a
    ``log_weight`` callable / tensor giving ``log(p/q)`` at the nodes.
    """

    def __init__(
        self,
        measure: Measure,
        *,
        self_normalized: bool = True,
        learnable_weights: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(measure, learnable_weights=learnable_weights, dtype=dtype)
        self.self_normalized = self_normalized

    def forward(self, f: FOrValues, log_weight: FOrValues) -> Tensor:
        vals = _as_values(f, self.nodes)
        lw = _as_values(log_weight, self.nodes).reshape(-1)
        out: Tensor = ops.importance_expectation(
            lambda _n: vals,
            log_weight=lambda _n: lw,
            self_normalized=self.self_normalized,
            nodes=self.nodes,
            weights=self.weight,
        )
        return out


class LayerCakeIntegral(_MeasureLayer):
    r"""Distribution-function integral with a learnable level-set softness ``beta``.

    ``beta = exp(log_beta)`` stays positive; set ``learnable_beta=False`` to pin
    it. ``forward(f)`` returns the (signed) layer-cake integral of ``f``.
    """

    log_beta: Tensor

    def __init__(
        self,
        measure: Measure,
        *,
        beta: float = 50.0,
        num_t: int = 256,
        signed: bool = True,
        learnable_beta: bool = True,
        learnable_weights: bool = False,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(measure, learnable_weights=learnable_weights, dtype=dtype)
        if not beta > 0.0:
            raise ValueError(f"beta must be > 0, got {beta}")
        dt = dtype if dtype is not None else torch.get_default_dtype()
        lb = torch.tensor(math.log(beta), dtype=dt)
        if learnable_beta:
            self.log_beta = nn.Parameter(lb)
        else:
            self.register_buffer("log_beta", lb)
        self.num_t = int(num_t)
        self.signed = bool(signed)

    @property
    def beta(self) -> Tensor:
        return torch.exp(self.log_beta)

    def forward(self, f: FOrValues, *, t_max: float | None = None) -> Tensor:
        vals = _as_values(f, self.nodes).reshape(-1)
        out: Tensor = ops.layer_cake_integral(
            lambda _n: vals,
            nodes=self.nodes,
            weights=self.weight,
            beta=self.beta,
            num_t=self.num_t,
            signed=self.signed,
            t_max=t_max,
        )
        return out


__all__ = [
    "ExpectationLayer",
    "LayerCakeIntegral",
    "LebesgueIntegral",
]
