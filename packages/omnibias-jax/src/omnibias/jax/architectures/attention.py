# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Non-local attention as a closed-form block inside the jet chain (JAX).

Bit-identical twin of :mod:`omnibias.torch.architectures.attention`; see that
module for the exposition. The network is

.. math::

    u(x) = W_o\,\Big[\mathrm{softmax}\big(\beta\, q(x) K^\top\big) V + q(x)\Big]
    + b_o, \qquad q(x) = \mathrm{MLP}(x),

with a trainable memory ``(K, V)`` that does not depend on ``x``. The softmax
couples all memory slots through a shared denominator, so this is the first
genuinely *non-local* block on the substrate -- and
:func:`omnibias.jax.jet_mv.jet_attention` keeps every mixed partial ``D^alpha u``
exactly closed form, which is the piece :mod:`omnibias.hopfield` does not supply
(it differentiates the log-sum-exp core with respect to the *scores*, not the
coordinates).

:class:`AttentionJetMLP` is a frozen dataclass registered as a JAX pytree, so the
encoder weights, the memory, the readout and the inverse temperature are ordinary
leaves that ``jax.grad`` / ``jax.jit`` see.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from omnibias.jax.activations import JaxActivationSpec, get_activation
from omnibias.jax.architectures.pinn import (
    _check_layer_fastpaths,
    _partials_from_jet,
)
from omnibias.jax.jet import affine_jet
from omnibias.jax.jet_mv import (
    jet_attention,
    jet_gradient,
    jet_hessian,
    mlp_jet_mv,
)

import jax
import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:  # pragma: no cover
    LayerSpec = tuple[Array, Array | None, JaxActivationSpec | None]


@dataclass(frozen=True)
class AttentionJetMLP:
    r"""Deep encoder + one non-local attention block, with exact input derivatives.

    JAX twin of :class:`omnibias.torch.architectures.attention.AttentionJetMLP`.
    Build one with :func:`make_attention_jet_mlp`.

    Attributes
    ----------
    weights, biases:
        Encoder chain; every encoder layer is activated by ``spec``.
    keys, values:
        Memory of shape ``(n, d_key)`` and ``(n, d_val)``.
    readout_weight, readout_bias:
        Affine readout of shape ``(out_dim, d_val)`` and ``(out_dim,)``.
    beta:
        Inverse temperature (a leaf, so it trains like any other parameter).
    residual:
        Whether the query is added to the attention output.
    """

    weights: tuple[Array, ...]
    biases: tuple[Array, ...]
    keys: Array
    values: Array
    readout_weight: Array
    readout_bias: Array
    beta: Array
    spec: JaxActivationSpec
    in_dim: int
    out_dim: int
    residual: bool = True

    @property
    def depth(self) -> int:
        """Number of hidden (activated) encoder layers."""
        return len(self.weights)

    @property
    def memory(self) -> int:
        """Number of memory slots."""
        return int(self.keys.shape[0])

    @property
    def value_dim(self) -> int:
        """Width of the value slots."""
        return int(self.values.shape[-1])

    def _encoder_specs(self) -> list[LayerSpec]:
        """``(W, b, spec)`` chain of the query encoder ``q(x)`` (all layers activated)."""
        return [
            (w, b, self.spec)
            for w, b in zip(self.weights, self.biases, strict=True)
        ]

    def _check_fastpath(self, max_order: int) -> None:
        """Reject an encoder activation without a closed-form tower of ``max_order``.

        The attention block itself needs no check: ``exp`` and the reciprocal
        tower are closed form at every order by construction.
        """
        _check_layer_fastpaths(self._encoder_specs(), max_order)

    def _point_jet(self, xi: Array, order: int) -> Array:
        """Single-point jet of the whole block, shape ``(M, out_dim)``."""
        q_jet = mlp_jet_mv(xi, self._encoder_specs(), order)
        h_jet = jet_attention(
            q_jet, self.keys, self.values, self.in_dim, order, beta=self.beta
        )
        if self.residual:
            h_jet = h_jet + q_jet
        return affine_jet(h_jet, self.readout_weight, self.readout_bias)

    def query(self, x: Array) -> Array:
        """Encoder output ``q(x)``, shape ``(..., hidden)``."""
        h = x
        for w, b, spec in self._encoder_specs():
            h = h @ w.T
            if b is not None:
                h = h + b
            assert spec is not None
            h = spec.forward(h)
        out: Array = h
        return out

    def attention_weights(self, x: Array) -> Array:
        """Softmax weights over the memory slots, shape ``(..., memory)``."""
        scores = self.query(x) @ (self.keys * self.beta).T
        out: Array = jax.nn.softmax(scores, axis=-1)
        return out

    def value(self, x: Array) -> Array:
        """Plain forward value ``u(x)``, shape ``(..., out_dim)`` (no jet needed)."""
        q = self.query(x)
        h = self.attention_weights(x) @ self.values
        if self.residual:
            h = h + q
        out: Array = h @ self.readout_weight.T + self.readout_bias
        return out

    def __call__(self, x: Array) -> Array:
        return self.value(x)

    def jet(self, x: Array, order: int) -> Array:
        """Batched multivariate jet, shape ``(B, M, out_dim)``."""
        out: Array = jax.vmap(lambda xi: self._point_jet(xi, order))(x)
        return out

    def gradient(self, x: Array) -> Array:
        """Exact input gradient ``d u / d x_i``, shape ``(B, in_dim, out_dim)``."""
        dim = self.in_dim
        out: Array = jax.vmap(
            lambda xi: jet_gradient(self._point_jet(xi, 1), dim, 1)
        )(x)
        return out

    def hessian(self, x: Array) -> Array:
        """Exact input Hessian, shape ``(B, in_dim, in_dim, out_dim)``."""
        dim = self.in_dim
        out: Array = jax.vmap(
            lambda xi: jet_hessian(self._point_jet(xi, 2), dim, 2)
        )(x)
        return out

    def partials(self, x: Array, order: int) -> dict[tuple[int, ...], Array]:
        """All raw partials ``{alpha: D^alpha u(x)}`` to total ``order`` (``(B, out_dim)``)."""
        return _partials_from_jet(self.jet(x, order), self.in_dim, order)


_AttentionAux = tuple[JaxActivationSpec, int, int, bool, int]


def _attention_flatten(
    net: AttentionJetMLP,
) -> tuple[tuple[Any, ...], _AttentionAux]:
    children = (
        *net.weights,
        *net.biases,
        net.keys,
        net.values,
        net.readout_weight,
        net.readout_bias,
        net.beta,
    )
    return children, (
        net.spec,
        net.in_dim,
        net.out_dim,
        net.residual,
        len(net.weights),
    )


def _attention_unflatten(
    aux: _AttentionAux, leaves: tuple[Any, ...]
) -> AttentionJetMLP:
    spec, in_dim, out_dim, residual, n = aux
    flat = tuple(leaves)
    return AttentionJetMLP(
        weights=flat[:n],
        biases=flat[n : 2 * n],
        keys=flat[2 * n],
        values=flat[2 * n + 1],
        readout_weight=flat[2 * n + 2],
        readout_bias=flat[2 * n + 3],
        beta=flat[2 * n + 4],
        spec=spec,
        in_dim=in_dim,
        out_dim=out_dim,
        residual=residual,
    )


jax.tree_util.register_pytree_node(
    AttentionJetMLP, _attention_flatten, _attention_unflatten
)


def make_attention_jet_mlp(
    in_dim: int,
    hidden: int,
    out_dim: int = 1,
    depth: int = 2,
    base: str | JaxActivationSpec = "tanh",
    *,
    memory: int = 16,
    value_dim: int | None = None,
    beta: float = 1.0,
    residual: bool = True,
    seed: int = 0,
    weight_init_scale: float = 1.0,
    dtype: Any = jnp.float64,
) -> AttentionJetMLP:
    """Build a randomly-initialised :class:`AttentionJetMLP`.

    Weights follow :func:`omnibias.jax.architectures.pinn.make_jet_mlp` (scaled
    normals, zero biases); the keys are scaled by ``1/sqrt(d_key)`` so the initial
    scores are ``O(1)`` and the softmax starts near-uniform rather than saturated.
    """
    if in_dim < 1 or hidden < 1 or out_dim < 1:
        raise ValueError("in_dim, hidden and out_dim must all be >= 1")
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
    spec = get_activation(base)
    key = jax.random.PRNGKey(seed)
    weights: list[Array] = []
    biases: list[Array] = []
    prev = in_dim
    for _ in range(depth):
        key, wk = jax.random.split(key)
        scale = weight_init_scale / math.sqrt(prev)
        weights.append(jax.random.normal(wk, (hidden, prev), dtype=dtype) * scale)
        biases.append(jnp.zeros((hidden,), dtype=dtype))
        prev = hidden
    key, kk, vk, rk = jax.random.split(key, 4)
    keys = jax.random.normal(kk, (memory, hidden), dtype=dtype) / math.sqrt(hidden)
    values = jax.random.normal(vk, (memory, d_val), dtype=dtype)
    readout_weight = jax.random.normal(rk, (out_dim, d_val), dtype=dtype) * (
        weight_init_scale / math.sqrt(d_val)
    )
    return AttentionJetMLP(
        weights=tuple(weights),
        biases=tuple(biases),
        keys=keys,
        values=values,
        readout_weight=readout_weight,
        readout_bias=jnp.zeros((out_dim,), dtype=dtype),
        beta=jnp.asarray(float(beta), dtype=dtype),
        spec=spec,
        in_dim=in_dim,
        out_dim=out_dim,
        residual=bool(residual),
    )


__all__ = [
    "AttentionJetMLP",
    "make_attention_jet_mlp",
]
