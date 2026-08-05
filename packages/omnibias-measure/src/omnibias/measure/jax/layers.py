# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Functional / equinox-style (jax) measure-integral layers.

Each layer is a registered pytree whose array leaves (``nodes``, ``weight``,
and, for the layer-cake, ``log_beta``) are differentiable, so ``jax.grad`` /
``optax`` treat the layer like any other parameter container -- the equinox
convention. Static configuration (``num_t``, ``signed``, ...) rides in the
pytree's aux data. Mirrors :mod:`omnibias.measure.torch.layers`.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax.numpy as jnp
from jax import Array
from jax.tree_util import register_pytree_node_class
from omnibias.measure._core.measure import Measure
from omnibias.measure.jax import ops

FOrValues = Array | Callable[[Array], Array]


def _as_values(f: FOrValues, nodes: Array) -> Array:
    return f(nodes) if callable(f) else f


@register_pytree_node_class
class LebesgueIntegral:
    """``int f dmu`` as a differentiable pytree layer (nodes + weights leaves)."""

    def __init__(self, nodes: Array, weight: Array, *, name: str = "measure") -> None:
        self.nodes = nodes
        self.weight = weight
        self.name = name

    @classmethod
    def from_measure(cls, measure: Measure) -> LebesgueIntegral:
        return cls(jnp.asarray(measure.nodes), jnp.asarray(measure.weights), name=measure.name)

    def __call__(self, f: FOrValues) -> Array:
        vals = _as_values(f, self.nodes)
        out: Array = ops.lebesgue_integral(lambda _n: vals, nodes=self.nodes, weights=self.weight)
        return out

    def tree_flatten(self) -> tuple[tuple[Array, Array], str]:
        return (self.nodes, self.weight), self.name

    @classmethod
    def tree_unflatten(cls, aux: str, children: tuple[Array, Array]) -> LebesgueIntegral:
        nodes, weight = children
        return cls(nodes, weight, name=aux)


@register_pytree_node_class
class ExpectationLayer:
    """(Self-normalized) importance-sampling expectation as a pytree layer."""

    def __init__(
        self, nodes: Array, weight: Array, *, self_normalized: bool = True, name: str = "measure"
    ) -> None:
        self.nodes = nodes
        self.weight = weight
        self.self_normalized = self_normalized
        self.name = name

    @classmethod
    def from_measure(cls, measure: Measure, *, self_normalized: bool = True) -> ExpectationLayer:
        return cls(
            jnp.asarray(measure.nodes),
            jnp.asarray(measure.weights),
            self_normalized=self_normalized,
            name=measure.name,
        )

    def __call__(self, f: FOrValues, log_weight: FOrValues) -> Array:
        vals = _as_values(f, self.nodes)
        lw = _as_values(log_weight, self.nodes).reshape(-1)
        out: Array = ops.importance_expectation(
            lambda _n: vals,
            log_weight=lambda _n: lw,
            self_normalized=self.self_normalized,
            nodes=self.nodes,
            weights=self.weight,
        )
        return out

    def tree_flatten(self) -> tuple[tuple[Array, Array], tuple[bool, str]]:
        return (self.nodes, self.weight), (self.self_normalized, self.name)

    @classmethod
    def tree_unflatten(
        cls, aux: tuple[bool, str], children: tuple[Array, Array]
    ) -> ExpectationLayer:
        nodes, weight = children
        self_normalized, name = aux
        return cls(nodes, weight, self_normalized=self_normalized, name=name)


@register_pytree_node_class
class LayerCakeIntegral:
    r"""Distribution-function integral layer with a learnable ``beta = exp(log_beta)``."""

    def __init__(
        self,
        nodes: Array,
        weight: Array,
        log_beta: Array,
        *,
        num_t: int = 256,
        signed: bool = True,
        name: str = "measure",
    ) -> None:
        self.nodes = nodes
        self.weight = weight
        self.log_beta = log_beta
        self.num_t = int(num_t)
        self.signed = bool(signed)
        self.name = name

    @classmethod
    def from_measure(
        cls, measure: Measure, *, beta: float = 50.0, num_t: int = 256, signed: bool = True
    ) -> LayerCakeIntegral:
        if not beta > 0.0:
            raise ValueError(f"beta must be > 0, got {beta}")
        return cls(
            jnp.asarray(measure.nodes),
            jnp.asarray(measure.weights),
            jnp.asarray(math.log(beta)),
            num_t=num_t,
            signed=signed,
            name=measure.name,
        )

    @property
    def beta(self) -> Array:
        return jnp.exp(self.log_beta)

    def __call__(self, f: FOrValues, *, t_max: float | None = None) -> Array:
        vals = _as_values(f, self.nodes).reshape(-1)
        out: Array = ops.layer_cake_integral(
            lambda _n: vals,
            nodes=self.nodes,
            weights=self.weight,
            beta=self.beta,
            num_t=self.num_t,
            signed=self.signed,
            t_max=t_max,
        )
        return out

    def tree_flatten(self) -> tuple[tuple[Array, Array, Array], tuple[int, bool, str]]:
        return (self.nodes, self.weight, self.log_beta), (self.num_t, self.signed, self.name)

    @classmethod
    def tree_unflatten(
        cls, aux: tuple[int, bool, str], children: tuple[Array, Array, Array]
    ) -> LayerCakeIntegral:
        nodes, weight, log_beta = children
        num_t, signed, name = aux
        return cls(nodes, weight, log_beta, num_t=num_t, signed=signed, name=name)


__all__ = [
    "ExpectationLayer",
    "LayerCakeIntegral",
    "LebesgueIntegral",
]
