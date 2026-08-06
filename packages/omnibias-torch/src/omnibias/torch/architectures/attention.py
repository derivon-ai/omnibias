# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Non-local attention as a closed-form block inside the jet chain (torch).

Every architecture in :mod:`omnibias.torch.architectures.pinn` is *local* in the
same narrow sense: the value at ``x`` is a chain of elementwise activations
applied to affine maps of ``x``. This module adds the first **non-local** block
that keeps the closed-form tower intact.

.. math::

    u(x) = W_o\,\Big[\underbrace{\mathrm{softmax}\big(\beta\, q(x) K^\top\big) V}
    _{\text{global mixture over the memory}} + q(x)\Big] + b_o,
    \qquad q(x) = \mathrm{MLP}(x).

The softmax couples every memory slot to every other one through a shared
denominator, so ``u(x)`` depends on the *whole* memory at once -- the value at a
point is no longer a function of a purely pointwise feature stack.

What is new here, and what is not
---------------------------------
:mod:`omnibias.hopfield` already carries the closed-form log-sum-exp Jacobian and
Hessian of this same block, but **with respect to the scores**. A PDE residual
needs ``d/dx``. :func:`omnibias.torch.jet_mv.jet_attention` supplies exactly that
missing coordinate story by pushing the query *jet* through the block, and this
module is the trainable network that uses it: one
:func:`~omnibias.torch.jet_mv.mlp_jet_mv` call for the encoder, one attention
block, one affine readout, and every mixed partial ``D^alpha u(x)`` falls out of
that single jet. There is no ``torch.autograd.grad`` in the differential
operator, at any order.

The block is exact because softmax factors into primitives that are each exact:
``exp`` (a registered activation with the tower ``exp^(k) = exp``), a linear sum,
a reciprocal (tower ``(-1)^k k! u^{-(k+1)}``), and a jet product. That is a
genuine extension of the reachable class -- ``compose_jet_mv`` alone handles only
elementwise maps.

Because :class:`AttentionJetMLP` is a
:class:`~omnibias.torch.architectures.pinn._JetMLPCore`, it inherits the exact
readouts (``jet`` / ``gradient`` / ``hessian`` / ``partials``) and drops into the
omnibias PINN field substrate via :mod:`omnibias.pinn.torch.fields.attention`.
"""

from __future__ import annotations

import math

from omnibias.torch.activations.registry import ActivationSpec, get_activation
from omnibias.torch.architectures.pinn import _JetMLPCore
from omnibias.torch.jet import affine_jet
from omnibias.torch.jet_mv import jet_attention, mlp_jet_mv

import torch
import torch.nn as nn
from torch import Tensor

LayerSpec = tuple[Tensor, Tensor | None, "ActivationSpec[Tensor] | None"]


class AttentionJetMLP(_JetMLPCore):
    r"""Deep encoder + one non-local attention block, with exact input derivatives.

    ``u(x) = W_o [ softmax(beta q(x) K^T) V (+ q(x)) ] + b_o`` where ``q`` is an
    ordinary MLP encoder. The memory ``(K, V)`` is a set of trainable slots that
    do **not** depend on ``x``, which is what keeps the block closed form: the
    only ``x``-dependence enters through the query, and the jet of the query is
    exact.

    Interpretation for PINNs: the softmax is a learned soft partition of the
    input domain, and ``V`` holds one local model per region -- a differentiable,
    globally-coupled relative of a domain decomposition, and the reason this
    field type is worth having next to the local ones. Note the temperature
    collapse this shares with the rest of the repo: as ``beta -> inf`` the
    mixture hardens into a crisp nearest-slot assignment (the *feasibility*
    sense of collapse, not the founding ``delta -> 0`` one).

    Parameters
    ----------
    in_dim, out_dim:
        Input coordinate count and output component count.
    hidden:
        Encoder width. Also the query / key width ``d_key``.
    depth:
        Number of hidden (activated) encoder layers ``>= 1``.
    base:
        Encoder activation; must have a closed-form derivative fast path.
    memory:
        Number of memory slots ``n``.
    value_dim:
        Width ``d_val`` of the value slots; defaults to ``hidden``. Must equal
        ``hidden`` when ``residual`` is set.
    beta:
        Initial inverse temperature ``beta > 0``. Larger is sharper.
    learnable_temperature:
        Train ``beta`` alongside the weights. It enters the jet as a scale on the
        keys, so the derivatives stay closed form while it moves.
    residual:
        Add the query to the attention output (the usual transformer skip). Keeps
        a *local* path alongside the non-local one, which trains far better.
    seed:
        Optional seed for the memory draw; ``None`` leaves the global RNG alone.
    dtype:
        Parameter dtype, defaulting to the framework default.
    """

    beta_raw: Tensor

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int = 1,
        depth: int = 2,
        base: str | ActivationSpec[Tensor] = "tanh",
        *,
        memory: int = 16,
        value_dim: int | None = None,
        beta: float = 1.0,
        learnable_temperature: bool = False,
        residual: bool = True,
        seed: int | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1, got {in_dim}")
        if hidden < 1:
            raise ValueError(f"hidden must be >= 1, got {hidden}")
        if out_dim < 1:
            raise ValueError(f"out_dim must be >= 1, got {out_dim}")
        if depth < 1:
            raise ValueError(f"depth (number of hidden layers) must be >= 1, got {depth}")
        if memory < 1:
            raise ValueError(f"memory must be >= 1, got {memory}")
        if beta <= 0.0:
            raise ValueError(f"beta must be > 0, got {beta}")
        d_val = hidden if value_dim is None else int(value_dim)
        if d_val < 1:
            raise ValueError(f"value_dim must be >= 1, got {d_val}")
        if residual and d_val != hidden:
            raise ValueError(
                f"residual needs value_dim == hidden, got {d_val} != {hidden}; "
                "pass residual=False for an asymmetric block"
            )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.depth = depth
        self.memory = memory
        self.value_dim = d_val
        self.residual = bool(residual)
        self.base = base if isinstance(base, ActivationSpec) else get_activation(base)

        if seed is not None:
            torch.manual_seed(seed)
        linears: list[nn.Linear] = []
        prev = in_dim
        for _ in range(depth):
            linears.append(nn.Linear(prev, hidden, dtype=dtype))
            prev = hidden
        self.encoder = nn.ModuleList(linears)
        # Keys scaled by 1/sqrt(d_key) so the initial scores are O(1) and the
        # softmax starts near-uniform rather than saturated.
        self.keys = nn.Parameter(
            torch.randn(memory, hidden, dtype=dtype) / math.sqrt(hidden)
        )
        self.values = nn.Parameter(torch.randn(memory, d_val, dtype=dtype))
        self.readout = nn.Linear(d_val, out_dim, dtype=dtype)
        beta_t = torch.tensor(float(beta), dtype=dtype or torch.get_default_dtype())
        if learnable_temperature:
            self.beta_raw = nn.Parameter(beta_t)
        else:
            self.register_buffer("beta_raw", beta_t)

    # -- the graph is encoder -> attention -> readout, not one chain ------------- #

    def _layer_specs(self) -> list[LayerSpec]:
        raise NotImplementedError(
            "AttentionJetMLP is an encoder + attention block + readout, not a "
            "single layer chain; use _encoder_specs()."
        )

    def _encoder_specs(self) -> list[LayerSpec]:
        """``(W, b, spec)`` chain of the query encoder ``q(x)`` (all layers activated)."""
        specs: list[LayerSpec] = []
        for lin in self.encoder:
            assert isinstance(lin, nn.Linear)
            specs.append((lin.weight, lin.bias, self.base))
        return specs

    @property
    def beta(self) -> Tensor:
        """Current inverse temperature."""
        return self.beta_raw

    def _check_fastpath(self, max_order: int) -> None:
        """Reject an encoder activation without a closed-form tower of ``max_order``.

        The attention block itself needs no check: ``exp`` and the reciprocal
        tower are closed form at every order by construction.
        """
        spec = self.base
        if spec.fastpath is None:
            raise ValueError(
                f"{type(self).__name__} requires activations with a closed-form "
                f"derivative kernel; activation {spec.name!r} has none."
            )
        try:
            spec.fastpath(torch.zeros(1), max_order)
        except NotImplementedError as e:
            raise ValueError(
                f"Activation {spec.name!r} fast-path does not support order "
                f"{max_order}: {e}"
            ) from None

    def _point_jet(self, xi: Tensor, order: int) -> Tensor:
        """Single-point jet of the whole block, shape ``(M, out_dim)``."""
        q_jet = mlp_jet_mv(xi, self._encoder_specs(), order)
        h_jet = jet_attention(
            q_jet, self.keys, self.values, self.in_dim, order, beta=self.beta
        )
        if self.residual:
            h_jet = h_jet + q_jet
        return affine_jet(h_jet, self.readout.weight, self.readout.bias)

    def query(self, x: Tensor) -> Tensor:
        """Encoder output ``q(x)``, shape ``(..., hidden)``."""
        h = x
        for w, b, spec in self._encoder_specs():
            h = h @ w.t()
            if b is not None:
                h = h + b
            assert spec is not None
            h = spec.forward(h)
        return h

    def attention_weights(self, x: Tensor) -> Tensor:
        """Softmax weights over the memory slots, shape ``(..., memory)``.

        The diagnostic that makes the block readable: each row is a partition of
        unity over the memory, so it says which slots the model is consulting at
        ``x``. :func:`omnibias.torch.jet_mv.jet_softmax` differentiates exactly
        this quantity with respect to ``x``.
        """
        scores = self.query(x) @ (self.keys * self.beta).t()
        return torch.softmax(scores, dim=-1)

    def value(self, x: Tensor) -> Tensor:
        """Plain forward value ``u(x)``, shape ``(..., out_dim)`` (no jet needed)."""
        q = self.query(x)
        h = self.attention_weights(x) @ self.values
        if self.residual:
            h = h + q
        out: Tensor = self.readout(h)
        return out

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, depth={self.depth}, "
            f"memory={self.memory}, value_dim={self.value_dim}, "
            f"beta={float(self.beta_raw.detach()):g}, residual={self.residual}"
        )


__all__ = [
    "AttentionJetMLP",
]
