# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Global-integral conservation cage for the JAX backend.

JAX twin of :mod:`omnibias.pinn.torch.cage.integral`; see that module for the
mathematics. A conserved integral of a density homogeneous of degree ``p`` is
enforced by the single global rescaling
``lambda = (total / I)^(1/p)``, which leaves every derivative exact because
``lambda`` has no ``x`` dependence.

One deliberate difference from the torch twin. The torch cage inspects the
measured integral and raises when no real rescaling exists (a zero integral, or
a negative one at even degree). Under ``jit`` that value is a tracer and cannot
be inspected, so this backend computes the scale unconditionally and a
degenerate field surfaces as ``inf`` / ``nan`` rather than an exception. Check
it yourself with :meth:`IntegralConservationField.integral` outside the trace if
that matters -- silently branching on a traced value is not something this
module will fake.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.jax.cage.incompressible import _CageFieldBase
from omnibias.pinn.jax.fields.base import FieldBase, _import_jax_ops

if TYPE_CHECKING:  # pragma: no cover
    from omnibias.fields._core.quadrature import QuadratureSpec


@dataclass(frozen=True)
class IntegralConservationField(_CageFieldBase):
    """JAX global-integral conservation cage; build it with the factory below."""

    base: FieldBase
    velocity_names: tuple[str, ...]
    passthrough_names: tuple[str, ...]
    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    quadrature_nodes: Array
    quadrature_weights: Array
    total: float
    degree: int

    def _raw_integral(self) -> Array:
        """``sum_q w_q sum_c u_c(x_q)^degree`` on the *un*-scaled base field."""
        quad_state = self.base.evaluate(self.quadrature_nodes)
        density = None
        for name in self.velocity_names:
            term = quad_state.ops.value(quad_state, name) ** self.degree
            density = term if density is None else density + term
        assert density is not None
        return jnp.sum(self.quadrature_weights * density)

    def _scale(self) -> Array:
        """``lambda`` solving ``lambda^degree * I == total``, sign-preserving."""
        ratio = self.total / self._raw_integral()
        return jnp.sign(ratio) * jnp.abs(ratio) ** (1.0 / self.degree)

    def integral(self, state: FieldState) -> Array:
        """The conserved integral of the *caged* field, recomputed not restated."""
        scale = state.extra["_conservation_scale"]
        return self._raw_integral() * scale**self.degree

    def evaluate(self, coords: Array) -> FieldState[Array]:
        coords = jnp.asarray(coords)
        if coords.ndim != 2:
            raise ValueError(
                f"coords must be 2D (B, D), got shape {tuple(coords.shape)}"
            )
        if coords.shape[-1] != self.coordinate_spec.ndim:
            raise ValueError(
                f"coords last dim {coords.shape[-1]} != coordinate_spec.ndim "
                f"{self.coordinate_spec.ndim}"
            )
        inner_state = self.base.evaluate(coords)
        return FieldState(
            coords=coords,
            field=self,
            components=self.components,
            coordinate_spec=self.coordinate_spec,
            ops=_import_jax_ops(),
            sigma_cache=inner_state.sigma_cache,
            extra={
                "_cage_inner_state": inner_state,
                "_conservation_scale": self._scale(),
            },
        )

    def value_component(self, state: FieldState, name: str) -> Array:
        inner = state.extra["_cage_inner_state"]
        return self._scaled(state, name, inner.ops.value(inner, name))

    def derivative(
        self, state: FieldState, name: str, *, axis: int, order: int = 1
    ) -> Array:
        if order == 0:
            return self.value_component(state, name)
        inner = state.extra["_cage_inner_state"]
        d = inner.ops.derivative(inner, name, axis=axis, order=order)
        return self._scaled(state, name, d)

    def mixed_partial(
        self,
        state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Array:
        inner = state.extra["_cage_inner_state"]
        d = inner.ops.mixed_partial(inner, name, axes, orders)
        return self._scaled(state, name, d)

    def _scaled(self, state: FieldState, name: str, value: Array) -> Array:
        if name in self.velocity_names:
            scale: Array = state.extra["_conservation_scale"]
            return value * scale
        if name in self.passthrough_names:
            return value
        raise KeyError(
            f"{name!r} is neither a conserved component {self.velocity_names!r} "
            f"nor a passthrough {self.passthrough_names!r}"
        )


def make_integral_conservation_field(
    *,
    base: FieldBase,
    rule: QuadratureSpec,
    conserved: tuple[str, ...],
    total: float = 1.0,
    degree: int = 2,
) -> IntegralConservationField:
    """Build an :class:`IntegralConservationField`.

    See :class:`omnibias.pinn.torch.cage.IntegralConservationField` for the
    argument semantics; this factory validates identically.
    """
    if not conserved:
        raise ValueError("conserved must name at least one component")
    for name in conserved:
        if not base.components.is_component(name):
            raise ValueError(
                f"conserved component {name!r} not in base components "
                f"{base.components.names!r}"
            )
    if len(set(conserved)) != len(conserved):
        raise ValueError(f"conserved names must be unique, got {conserved!r}")
    if degree < 1:
        raise ValueError(f"degree must be >= 1, got {degree}")
    if total <= 0.0:
        raise ValueError(f"total must be > 0, got {total}")
    if rule.dim != base.coordinate_spec.ndim:
        raise ValueError(
            f"quadrature dim {rule.dim} != coordinate_spec.ndim "
            f"{base.coordinate_spec.ndim}"
        )
    passthrough = tuple(n for n in base.components.names if n not in conserved)
    return IntegralConservationField(
        base=base,
        velocity_names=tuple(conserved),
        passthrough_names=passthrough,
        coordinate_spec=base.coordinate_spec,
        components=ComponentSpec(
            tuple(conserved) + passthrough, groups={"conserved": tuple(conserved)}
        ),
        quadrature_nodes=jnp.asarray(rule.nodes),
        quadrature_weights=jnp.asarray(rule.weights),
        total=float(total),
        degree=int(degree),
    )


def _icf_flatten(
    f: IntegralConservationField,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    return (f.base, f.quadrature_nodes, f.quadrature_weights), (
        f.velocity_names,
        f.passthrough_names,
        f.coordinate_spec,
        f.components,
        f.total,
        f.degree,
    )


def _icf_unflatten(
    aux: tuple[object, ...], leaves: tuple[object, ...]
) -> IntegralConservationField:
    base, nodes, weights = leaves
    (velocity_names, passthrough_names, coordinate_spec, components, total, degree) = (
        aux
    )
    obj = IntegralConservationField.__new__(IntegralConservationField)
    object.__setattr__(obj, "base", base)
    object.__setattr__(obj, "quadrature_nodes", nodes)
    object.__setattr__(obj, "quadrature_weights", weights)
    object.__setattr__(obj, "velocity_names", velocity_names)
    object.__setattr__(obj, "passthrough_names", passthrough_names)
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    object.__setattr__(obj, "total", total)
    object.__setattr__(obj, "degree", degree)
    return obj


jax.tree_util.register_pytree_node(
    IntegralConservationField, _icf_flatten, _icf_unflatten
)


__all__ = [
    "IntegralConservationField",
    "make_integral_conservation_field",
]
