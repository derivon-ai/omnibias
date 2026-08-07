# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Non-local attention PINN field with closed-form coordinate derivatives (JAX twin).

Bit-identical twin of :mod:`omnibias.pinn.torch.fields.attention`; see that module
for the exposition. The field routes the coordinates through a softmax mixture
over a trainable memory,

.. math::

    u(x) = W_o\,\Big[\mathrm{softmax}\big(\beta\,q(x) K^\top\big) V + q(x)\Big]
    + b_o,

which couples every output to every memory slot through the shared denominator --
the first genuinely *non-local* field on the substrate. Its coordinate
derivatives stay exactly closed form through
:func:`omnibias.jax.jet_mv.jet_attention`, which is the piece
:mod:`omnibias.hopfield` does not supply (it differentiates the log-sum-exp core
with respect to the *scores*).

As everywhere in the JAX backend the field is a ``dataclass(frozen=True)`` pytree
rather than an :class:`torch.nn.Module`, so the encoder weights, the memory and
the inverse temperature are ordinary leaves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.jax.activations import JaxActivationSpec
from omnibias.jax.architectures.attention import (
    AttentionJetMLP,
    make_attention_jet_mlp,
)
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax.fields.base import FieldBase
from omnibias.pinn.jax.fields.jet_mlp import _JetFieldOps, validate_jet_field


@dataclass(frozen=True)
class AttentionVectorField(_JetFieldOps, FieldBase):
    r"""PINN field whose value at ``x`` is a non-local mixture over a trainable memory.

    JAX twin of :class:`omnibias.pinn.torch.fields.AttentionVectorField`. Build one
    with :func:`make_attention_vector_field`.
    """

    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    net: AttentionJetMLP
    jet_order: int = 2

    @property
    def memory(self) -> int:
        """Number of memory slots."""
        return int(self.net.memory)

    @property
    def beta(self) -> Array:
        """Current inverse temperature of the softmax mixture."""
        return self.net.beta

    def attention_weights(self, coords: Array) -> Array:
        """Per-point partition of unity over the memory, shape ``(B, memory)``."""
        return self.net.attention_weights(coords)

    def __repr__(self) -> str:
        return (
            f"AttentionVectorField(axes={self.coordinate_spec.axes}, "
            f"components={self.components.names}, memory={self.memory}, "
            f"jet_order={self.jet_order})"
        )


# -- pytree registration --------------------------------------------------------- #
#
# Same shape as :mod:`omnibias.pinn.jax.fields.jet_mlp`: the wrapped architecture is
# itself a registered pytree and is the single leaf, so JAX recurses into its
# encoder weights, memory and temperature.

_Aux = tuple[CoordinateSpec, ComponentSpec, int]


def _flatten(field: AttentionVectorField) -> tuple[tuple[Any], _Aux]:
    return (field.net,), (field.coordinate_spec, field.components, field.jet_order)


def _unflatten(aux: _Aux, leaves: tuple[Any]) -> AttentionVectorField:
    coordinate_spec, components, jet_order = aux
    obj = AttentionVectorField.__new__(AttentionVectorField)
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    object.__setattr__(obj, "net", leaves[0])
    object.__setattr__(obj, "jet_order", jet_order)
    return obj


jax.tree_util.register_pytree_node(AttentionVectorField, _flatten, _unflatten)


def make_attention_vector_field(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    hidden: int = 64,
    depth: int = 2,
    base: str | JaxActivationSpec = "tanh",
    memory: int = 16,
    value_dim: int | None = None,
    beta: float = 1.0,
    residual: bool = True,
    jet_order: int = 2,
    seed: int = 0,
    weight_init_scale: float = 1.0,
    dtype: Any = jnp.float64,
) -> AttentionVectorField:
    """Initialise a fresh :class:`AttentionVectorField` with random parameters."""
    net = make_attention_jet_mlp(
        coordinate_spec.ndim,
        hidden,
        out_dim=components.n_components,
        depth=depth,
        base=base,
        memory=memory,
        value_dim=value_dim,
        beta=beta,
        residual=residual,
        seed=seed,
        weight_init_scale=weight_init_scale,
        dtype=dtype,
    )
    validate_jet_field(net, coordinate_spec, components, jet_order)
    return AttentionVectorField(
        coordinate_spec=coordinate_spec,
        components=components,
        net=net,
        jet_order=jet_order,
    )


__all__ = [
    "AttentionVectorField",
    "make_attention_vector_field",
]

# Marker read by the omnibias-fields backend ops to select the exact multivariate
# jet path (avoids a fields -> pinn import cycle).
AttentionVectorField._omnibias_dispatch = "jet_mlp"
AttentionVectorField._omnibias_readout_independent = True
