# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Distance-constrained PINN field for curved boundaries (JAX twin).

JAX cages are pytree nodes constructed via factories; this module wraps
:func:`~omnibias.pinn.jax.cage.make_hard_boundary_field` and exposes the same
surface as the torch :class:`~omnibias.pinn.domain.torch.field.DistanceConstrainedField`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from omnibias.core.multi_index import num_multi_indices
from omnibias.jax.jet_mv import jet_multiply
from omnibias.pinn.domain._core.boundary import (
    BCMode,
    NonSmoothBoundaryError,
    assert_smooth_for_normal_bc,
)
from omnibias.pinn.domain._core.sdf import SDF, Box, Halfspace, RCompose, Sphere
from omnibias.pinn.domain.jax.boundary_jax import boundary_factor_jet_at
from omnibias.pinn.domain.jax.sdf_jax import (
    DistanceFn,
    from_primitive,
    from_sdf,
    normalize_distance,
)
from omnibias.pinn.jax.cage.conservation import (
    HardBoundaryField,
    make_hard_boundary_field,
)
from omnibias.pinn.jax.fields.base import FieldBase


def _wrap_bc_factor(
    phi_fn: DistanceFn,
    *,
    mode: BCMode,
    robin_alpha: float,
    robin_beta: float,
) -> DistanceFn:
    if mode == "dirichlet":
        return phi_fn

    def _fn(coords: Array) -> Array:
        phi = phi_fn(coords)
        if mode == "neumann":
            return phi * phi
        return robin_alpha * phi + robin_beta * phi * phi

    return _fn


@dataclass
class DistanceConstrainedField:
    """Hard BC cage with SDF metadata and jet helpers (JAX)."""

    cage: HardBoundaryField
    sdf: SDF | None
    normalize: bool
    bc_mode: BCMode
    robin_alpha: float
    robin_beta: float

    @property
    def base(self) -> FieldBase:
        return self.cage.base

    @property
    def distance_fn(self) -> DistanceFn:
        return self.cage.distance_fn

    def evaluate(self, coords: Array):
        return self.cage(coords)

    def __call__(self, coords: Array):
        return self.evaluate(coords)

    def phi(self, coords: Array) -> Array:
        if self.bc_mode in ("neumann", "robin") and self.sdf is not None:
            assert_smooth_for_normal_bc(self.sdf, jnp.asarray(coords))
        return self.distance_fn(coords)

    def product_jet_at(
        self,
        x0: Array,
        nn_jet: Array,
        *,
        order: int,
        phi_jet: Array | None = None,
    ) -> Array:
        if jnp.ndim(x0) != 1:
            raise ValueError(f"x0 must be 1-D, got shape {tuple(x0.shape)}")
        dim = int(x0.shape[0])
        m = num_multi_indices(dim, order)
        if nn_jet.shape[0] != m:
            raise ValueError(
                f"nn_jet leading dim {nn_jet.shape[0]} != M={m} for "
                f"dim={dim}, order={order}"
            )
        if phi_jet is None:
            if self.sdf is None:
                raise ValueError("product_jet_at requires sdf or an explicit phi_jet")
            if self.bc_mode in ("neumann", "robin"):
                assert_smooth_for_normal_bc(
                    self.sdf, jnp.asarray(x0).reshape(1, -1)
                )
            phi_jet = boundary_factor_jet_at(
                self.sdf,
                x0,
                order=order,
                mode=self.bc_mode,
                normalize=self.normalize,
                robin_alpha=self.robin_alpha,
                robin_beta=self.robin_beta,
            )
        return jet_multiply(phi_jet, nn_jet, dim, order)


def build_distance_constrained_field(
    base: FieldBase,
    sdf: SDF | None = None,
    *,
    distance_fn: DistanceFn | None = None,
    boundary_value_fn: Callable[[Array], dict[str, Array]] | None = None,
    bounded_names: Sequence[str] | None = None,
    passthrough_names: tuple[str, ...] = (),
    groups: dict[str, tuple[str, ...]] | None = None,
    max_derivative_order: int = 4,
    normalize: bool = True,
    bc_mode: BCMode = "dirichlet",
    robin_alpha: float = 1.0,
    robin_beta: float = 0.0,
) -> DistanceConstrainedField:
    """Build a hard-BC cage ``u = g + phi * NN`` from an SDF / ADF (JAX)."""
    if bc_mode in ("neumann", "robin") and isinstance(sdf, RCompose):
        raise NonSmoothBoundaryError(
            "Neumann / Robin BCs on RCompose domains require smooth "
            "primitives without junctions; use Dirichlet or a single primitive"
        )
    if distance_fn is None:
        if sdf is None:
            raise ValueError("provide distance_fn or sdf")
        if not isinstance(sdf, (Sphere, Box, Halfspace)):
            phi_fn = from_sdf(sdf)
        else:
            phi_fn = from_primitive(sdf)
        if normalize:
            phi_fn = normalize_distance(phi_fn)
        distance_fn = _wrap_bc_factor(
            phi_fn,
            mode=bc_mode,
            robin_alpha=robin_alpha,
            robin_beta=robin_beta,
        )
    elif normalize and sdf is not None and isinstance(sdf, (Sphere, Box, Halfspace)):
        distance_fn = _wrap_bc_factor(
            normalize_distance(distance_fn),
            mode=bc_mode,
            robin_alpha=robin_alpha,
            robin_beta=robin_beta,
        )
    cage = make_hard_boundary_field(
        base=base,
        distance_fn=distance_fn,
        boundary_value_fn=boundary_value_fn,
        bounded_names=tuple(bounded_names) if bounded_names is not None else None,
        passthrough_names=passthrough_names,
        groups=groups,
        max_derivative_order=max_derivative_order,
    )
    return DistanceConstrainedField(
        cage=cage,
        sdf=sdf,
        normalize=normalize,
        bc_mode=bc_mode,
        robin_alpha=robin_alpha,
        robin_beta=robin_beta,
    )


__all__ = [
    "DistanceConstrainedField",
    "build_distance_constrained_field",
]
