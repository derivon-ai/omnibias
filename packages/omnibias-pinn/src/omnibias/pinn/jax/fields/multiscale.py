# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Multi-scale PINN fields: adaptive slopes and MscaleDNN band mixtures (JAX twin).

Bit-identical twin of :mod:`omnibias.pinn.torch.fields.multiscale`; see that
module for the exposition. As everywhere in the JAX backend the fields are
``dataclass(frozen=True)`` pytrees rather than ``nn.Module`` subclasses, so the
trainable slopes and band weights are ordinary leaves that ``jax.grad`` /
``jax.jit`` traverse.

Both fields carry the ``jet_mlp`` dispatch tag inherited from
:mod:`omnibias.pinn.jax.fields.jet_mlp`, so they reach the whole field-operator
surface with no new wiring.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.jax.activations import JaxActivationSpec
from omnibias.jax.architectures.multiscale import (
    AdaptiveJetMLP,
    MscaleMLP,
    make_adaptive_jet_mlp,
    make_mscale_mlp,
)
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax.fields.base import FieldBase
from omnibias.pinn.jax.fields.jet_mlp import _JetFieldOps, validate_jet_field


@dataclass(frozen=True)
class AdaptiveJetMLPVectorField(_JetFieldOps, FieldBase):
    r"""Deep PINN field with a trainable activation frequency per hidden layer (JAX).

    JAX twin of :class:`omnibias.pinn.torch.fields.AdaptiveJetMLPVectorField`. Build
    one with :func:`make_adaptive_jet_mlp_vector_field`.
    """

    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    net: AdaptiveJetMLP
    jet_order: int = 2

    def slopes(self) -> tuple[Array, ...]:
        """Current effective slope ``n a`` of each hidden layer."""
        return self.net.slopes()

    def __repr__(self) -> str:
        return (
            f"AdaptiveJetMLPVectorField(axes={self.coordinate_spec.axes}, "
            f"components={self.components.names}, depth={self.net.depth}, "
            f"jet_order={self.jet_order})"
        )


@dataclass(frozen=True)
class MscaleVectorField(_JetFieldOps, FieldBase):
    r"""MscaleDNN band-mixture PINN field ``u(x) = sum_j f_j(alpha_j x)`` (JAX).

    JAX twin of :class:`omnibias.pinn.torch.fields.MscaleVectorField`. Build one
    with :func:`make_mscale_vector_field`.
    """

    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    net: MscaleMLP
    jet_order: int = 2

    @property
    def scales(self) -> tuple[float, ...]:
        """The band factors ``alpha_j`` of the mixture."""
        return self.net.scales

    def __repr__(self) -> str:
        return (
            f"MscaleVectorField(axes={self.coordinate_spec.axes}, "
            f"components={self.components.names}, scales={self.scales}, "
            f"jet_order={self.jet_order})"
        )


# -- pytree registration --------------------------------------------------------- #
#
# Same shape as :mod:`omnibias.pinn.jax.fields.jet_mlp`: the wrapped architecture is
# itself a registered pytree and is the single leaf, so JAX recurses into its
# weights and (for the adaptive field) its slopes.

_Aux = tuple[CoordinateSpec, ComponentSpec, int]


def _flatten(
    field: AdaptiveJetMLPVectorField | MscaleVectorField,
) -> tuple[tuple[Any], _Aux]:
    return (field.net,), (field.coordinate_spec, field.components, field.jet_order)


def _populate(obj: object, aux: _Aux, leaves: tuple[Any]) -> None:
    coordinate_spec, components, jet_order = aux
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    object.__setattr__(obj, "net", leaves[0])
    object.__setattr__(obj, "jet_order", jet_order)


def _adaptive_unflatten(aux: _Aux, leaves: tuple[Any]) -> AdaptiveJetMLPVectorField:
    obj = AdaptiveJetMLPVectorField.__new__(AdaptiveJetMLPVectorField)
    _populate(obj, aux, leaves)
    return obj


def _mscale_unflatten(aux: _Aux, leaves: tuple[Any]) -> MscaleVectorField:
    obj = MscaleVectorField.__new__(MscaleVectorField)
    _populate(obj, aux, leaves)
    return obj


jax.tree_util.register_pytree_node(
    AdaptiveJetMLPVectorField, _flatten, _adaptive_unflatten
)
jax.tree_util.register_pytree_node(MscaleVectorField, _flatten, _mscale_unflatten)


# -- builders that handle parameter init ----------------------------------------- #


def make_adaptive_jet_mlp_vector_field(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    hidden: int = 64,
    depth: int = 3,
    base: str | JaxActivationSpec = "tanh",
    slope_scale: float = 1.0,
    granularity: str = "layer",
    scale_init: float = 1.0,
    jet_order: int = 2,
    seed: int = 0,
    weight_init_scale: float = 1.0,
    dtype: Any = jnp.float64,
) -> AdaptiveJetMLPVectorField:
    """Initialise a fresh :class:`AdaptiveJetMLPVectorField` with random parameters."""
    net = make_adaptive_jet_mlp(
        coordinate_spec.ndim,
        hidden,
        out_dim=components.n_components,
        depth=depth,
        base=base,
        slope_scale=slope_scale,
        granularity=granularity,
        scale_init=scale_init,
        seed=seed,
        weight_init_scale=weight_init_scale,
        dtype=dtype,
    )
    validate_jet_field(net, coordinate_spec, components, jet_order)
    return AdaptiveJetMLPVectorField(
        coordinate_spec=coordinate_spec,
        components=components,
        net=net,
        jet_order=jet_order,
    )


def make_mscale_vector_field(
    *,
    coordinate_spec: CoordinateSpec,
    components: ComponentSpec,
    hidden: int = 64,
    depth: int = 3,
    base: str | JaxActivationSpec = "tanh",
    scales: Sequence[float] = (1.0, 2.0, 4.0, 8.0),
    jet_order: int = 2,
    seed: int = 0,
    weight_init_scale: float = 1.0,
    dtype: Any = jnp.float64,
) -> MscaleVectorField:
    """Initialise a fresh :class:`MscaleVectorField` with random parameters.

    ``hidden`` is the *total* width, split evenly across the bands. Read the band
    scales off a measured spectrum with
    :func:`~omnibias.pinn._core.multiscale.suggest_frequency_bands` rather than
    guessing them.
    """
    net = make_mscale_mlp(
        coordinate_spec.ndim,
        hidden,
        out_dim=components.n_components,
        depth=depth,
        base=base,
        scales=scales,
        seed=seed,
        weight_init_scale=weight_init_scale,
        dtype=dtype,
    )
    validate_jet_field(net, coordinate_spec, components, jet_order)
    return MscaleVectorField(
        coordinate_spec=coordinate_spec,
        components=components,
        net=net,
        jet_order=jet_order,
    )


__all__ = [
    "AdaptiveJetMLPVectorField",
    "MscaleVectorField",
    "make_adaptive_jet_mlp_vector_field",
    "make_mscale_vector_field",
]

# Marker read by the omnibias-fields backend ops to select the exact multivariate
# jet path (avoids a fields -> pinn import cycle).
AdaptiveJetMLPVectorField._omnibias_dispatch = "jet_mlp"
MscaleVectorField._omnibias_dispatch = "jet_mlp"
AdaptiveJetMLPVectorField._omnibias_readout_independent = True
MscaleVectorField._omnibias_readout_independent = True
