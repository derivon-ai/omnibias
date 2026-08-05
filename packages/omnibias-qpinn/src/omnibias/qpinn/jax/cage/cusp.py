# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Hard Kato electron-nucleus cusp cage (JAX backend).

A single-particle trial wavefunction ``psi_base`` learned by a smooth neural
field cannot represent the derivative discontinuity that the Coulomb
singularity forces on the *exact* eigenfunction. Kato's theorem fixes the
spherically-averaged logarithmic slope at each nucleus,

.. math::

   \left.\frac{\partial \langle\psi\rangle}{\partial s_a}\right|_{s_a=0}
       = -Z_a\,\psi(R_a),
   \qquad s_a = \lVert r - R_a\rVert,

for a nucleus of charge ``Z_a``. This cage enforces that condition *by
construction* by multiplying the wavefunction by the strictly-positive
cusp factor

.. math::

   C(r) = \exp\!\Bigl(\sum_a u_a(s_a)\Bigr),
   \qquad
   u_a(s) = -\frac{Z_a\,s}{1 + b_a\,s},
   \qquad
   u_a'(0) = -Z_a,

so ``log C`` contributes exactly ``-Z_a`` to the radial slope at nucleus
``a`` while the smooth base contributes zero on the spherical average. The
denominator rate ``b_a`` localises the correction: ``u_a`` saturates to
``-Z_a/b_a`` as ``s -> inf``, and ``b_a = 0`` recovers the bare Slater factor
``exp(-Z_a s)`` (the exact hydrogenic 1s shape).

Closed-form derivatives
-----------------------
The caged value / gradient / Laplacian are assembled by the **exact Leibniz
product rule** ``(C b)' = C' b + C b'`` etc., combining the closed-form
derivatives of ``C`` (elementary rational + ``exp``) with the base field's
closed-form ``omnibias`` derivatives. No autodiff, no finite differences. The
cage supplies value, first, and (same-axis or mixed) second derivatives -- the
full set the Schrodinger local energy needs; third and higher orders raise
:class:`NotImplementedError` (they are not required for the kinetic operator).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.jax.cage.incompressible import _CageFieldBase
from omnibias.pinn.jax.fields.base import FieldBase, _import_jax_ops

_EPS2 = 1e-30


@dataclass(frozen=True)
class NuclearCuspField(_CageFieldBase):
    r"""Multiplicative electron-nucleus cusp cage ``psi = C(r) psi_base(r)`` (JAX).

    See the module docstring for the cusp factor ``C`` and the Kato condition
    it enforces. The wavefunction components in ``velocity_names`` are each
    multiplied by the same real, positive factor ``C``; ``passthrough_names``
    are returned untouched.
    """

    base: FieldBase
    nuclei: Array
    charges: Array
    rates: Array
    velocity_names: tuple[str, ...]
    passthrough_names: tuple[str, ...]
    coordinate_spec: CoordinateSpec
    components: ComponentSpec
    spatial_axis_indices: tuple[int, ...]

    def _coord_to_spatial(self, axis: int) -> int | None:
        """Spatial index of a coordinate axis, or ``None`` if non-spatial."""
        for k, a in enumerate(self.spatial_axis_indices):
            if a == axis:
                return k
        return None

    def _cusp_derivs(self, coords: Array) -> tuple[Array, Array, Array]:
        r"""Closed-form ``(C, grad C, Hess C)`` on the spatial coordinates.

        Returns ``C`` ``(B,)``, ``grad C`` ``(B, n_s)`` and ``Hess C``
        ``(B, n_s, n_s)`` where ``n_s = len(spatial_axis_indices)``.
        """
        idx = list(self.spatial_axis_indices)
        r_s = coords[:, idx]  # (B, n_s)
        diff = r_s[:, None, :] - self.nuclei[None, :, :]  # (B, n_nuc, n_s)
        s = jnp.sqrt(jnp.sum(diff * diff, axis=-1) + _EPS2)  # (B, n_nuc)
        shat = diff / s[..., None]  # (B, n_nuc, n_s)
        z = self.charges[None, :]  # (1, n_nuc)
        denom = 1.0 + self.rates[None, :] * s  # (B, n_nuc)
        up = -z / (denom * denom)  # u'(s)   -> -Z at s=0
        upp = 2.0 * z * self.rates[None, :] / (denom * denom * denom)  # u''(s)

        # grad g = sum_a u'(s_a) shat_a
        grad_g = jnp.sum(up[..., None] * shat, axis=1)  # (B, n_s)
        # Hess g = sum_a [ u''(s_a) shat shat^T + (u'(s_a)/s_a)(I - shat shat^T) ]
        n_s = len(idx)
        eye = jnp.eye(n_s, dtype=coords.dtype)
        outer = shat[..., :, None] * shat[..., None, :]  # (B, n_nuc, n_s, n_s)
        radial = (up / s)[..., None, None] * (eye[None, None, :, :] - outer)
        hess_g = jnp.sum(upp[..., None, None] * outer + radial, axis=1)  # (B,n_s,n_s)

        g = jnp.sum(-z * s / denom, axis=1)  # (B,)  log C
        c = jnp.exp(g)  # (B,)
        grad_c = c[:, None] * grad_g  # (B, n_s)
        hess_c = c[:, None, None] * (
            grad_g[:, :, None] * grad_g[:, None, :] + hess_g
        )  # (B, n_s, n_s)
        return c, grad_c, hess_c

    def evaluate(self, coords: Array) -> FieldState[Array]:
        coords = jnp.asarray(coords)
        if coords.ndim != 2:
            raise ValueError(
                f"coords must be 2D (B, D), got shape {tuple(coords.shape)}"
            )
        if coords.shape[-1] != self.coordinate_spec.ndim:
            raise ValueError(
                f"coords last dim {coords.shape[-1]} != "
                f"coordinate_spec.ndim {self.coordinate_spec.ndim}"
            )
        inner_state = self.base.evaluate(coords)
        c, grad_c, hess_c = self._cusp_derivs(coords)
        return FieldState(
            coords=coords,
            field=self,
            components=self.components,
            coordinate_spec=self.coordinate_spec,
            ops=_import_jax_ops(),
            sigma_cache=inner_state.sigma_cache,
            extra={
                "_cage_inner_state": inner_state,
                "_cusp_C": c,
                "_cusp_gradC": grad_c,
                "_cusp_hessC": hess_c,
            },
        )

    def value_component(self, state: FieldState, name: str) -> Array:
        inner = state.extra["_cage_inner_state"]
        v = inner.ops.value(inner, name)
        if name in self.velocity_names:
            return state.extra["_cusp_C"] * v
        if name in self.passthrough_names:
            return v
        raise KeyError(
            f"{name!r} is neither a caged wavefunction component "
            f"{self.velocity_names!r} nor a passthrough "
            f"{self.passthrough_names!r}"
        )

    def derivative(
        self, state: FieldState, name: str, *, axis: int, order: int = 1,
    ) -> Array:
        if order == 0:
            return self.value_component(state, name)
        inner = state.extra["_cage_inner_state"]
        if name in self.passthrough_names:
            return inner.ops.derivative(inner, name, axis=axis, order=order)
        if name not in self.velocity_names:
            raise KeyError(name)

        c = state.extra["_cusp_C"]
        k = self._coord_to_spatial(axis)
        b0 = inner.ops.value(inner, name)
        b1 = inner.ops.derivative(inner, name, axis=axis, order=1)
        c1 = state.extra["_cusp_gradC"][:, k] if k is not None else 0.0
        if order == 1:
            return c1 * b0 + c * b1
        if order == 2:
            b2 = inner.ops.derivative(inner, name, axis=axis, order=2)
            c2 = state.extra["_cusp_hessC"][:, k, k] if k is not None else 0.0
            return c2 * b0 + 2.0 * c1 * b1 + c * b2
        raise NotImplementedError(
            "NuclearCuspField supplies closed-form derivatives up to order 2 "
            f"(the Schrodinger kinetic operator); got order={order}."
        )

    def mixed_partial(
        self,
        state: FieldState,
        name: str,
        axes: tuple[int, ...],
        orders: tuple[int, ...],
    ) -> Array:
        inner = state.extra["_cage_inner_state"]
        if name in self.passthrough_names:
            return inner.ops.mixed_partial(inner, name, axes, orders)
        if name not in self.velocity_names:
            raise KeyError(name)

        folded: dict[int, int] = {}
        for a, o in zip(axes, orders, strict=False):
            folded[a] = folded.get(a, 0) + int(o)
        total = sum(folded.values())
        if total == 0:
            return self.value_component(state, name)
        if total == 1:
            (a,) = folded.keys()
            return self.derivative(state, name, axis=a, order=1)
        if total == 2:
            if len(folded) == 1:
                (a,) = folded.keys()
                return self.derivative(state, name, axis=a, order=2)
            (ai, aj) = folded.keys()
            c = state.extra["_cusp_C"]
            grad_c = state.extra["_cusp_gradC"]
            hess_c = state.extra["_cusp_hessC"]
            ki = self._coord_to_spatial(ai)
            kj = self._coord_to_spatial(aj)
            b0 = inner.ops.value(inner, name)
            bi = inner.ops.derivative(inner, name, axis=ai, order=1)
            bj = inner.ops.derivative(inner, name, axis=aj, order=1)
            bij = inner.ops.mixed_partial(inner, name, (ai, aj), (1, 1))
            ci = grad_c[:, ki] if ki is not None else 0.0
            cj = grad_c[:, kj] if kj is not None else 0.0
            cij = hess_c[:, ki, kj] if (ki is not None and kj is not None) else 0.0
            return cij * b0 + ci * bj + cj * bi + c * bij
        raise NotImplementedError(
            "NuclearCuspField supplies closed-form mixed partials up to total "
            f"order 2; got total order {total}."
        )


def make_nuclear_cusp_field(
    *,
    base: FieldBase,
    nuclei: Array,
    charges: Array,
    rates: Array | float = 0.0,
    psi_group: str = "psi",
) -> NuclearCuspField:
    r"""Build a :class:`NuclearCuspField` (JAX backend).

    Parameters
    ----------
    base
        Base :class:`FieldBase` carrying a wavefunction group with one or two
        real components (real, or ``(re, im)``).
    nuclei
        Nuclear positions, shape ``(n_nuc, n_spatial)`` in the base field's
        spatial coordinate frame.
    charges
        Nuclear charges ``Z_a``, shape ``(n_nuc,)``.
    rates
        Pade denominator rates ``b_a``; scalar (broadcast) or ``(n_nuc,)``.
        ``0`` gives the exact hydrogenic Slater factor ``exp(-Z_a s)``.
    psi_group
        Wavefunction group name. Default ``"psi"``.
    """
    if not base.components.is_group(psi_group):
        raise ValueError(
            f"base does not have a wavefunction group {psi_group!r}; "
            "build it with omnibias.qpinn.make_psi_components"
        )
    members = base.components.group_members(psi_group)
    n_spatial = base.coordinate_spec.n_spatial
    nuclei = jnp.asarray(nuclei)
    charges = jnp.asarray(charges)
    if nuclei.ndim != 2 or nuclei.shape[-1] != n_spatial:
        raise ValueError(
            f"nuclei must be (n_nuc, {n_spatial}); got {tuple(nuclei.shape)}"
        )
    if charges.ndim != 1 or charges.shape[0] != nuclei.shape[0]:
        raise ValueError(
            f"charges must be (n_nuc,) = ({nuclei.shape[0]},); "
            f"got {tuple(charges.shape)}"
        )
    rates_arr = (
        jnp.full((nuclei.shape[0],), float(rates), dtype=nuclei.dtype)
        if jnp.ndim(rates) == 0
        else jnp.asarray(rates)
    )
    if rates_arr.shape != (nuclei.shape[0],):
        raise ValueError(
            f"rates must be scalar or (n_nuc,); got {tuple(rates_arr.shape)}"
        )
    passthrough = tuple(n for n in base.components.names if n not in members)
    return NuclearCuspField(
        base=base,
        nuclei=nuclei,
        charges=charges,
        rates=rates_arr,
        velocity_names=tuple(members),
        passthrough_names=passthrough,
        coordinate_spec=base.coordinate_spec,
        components=base.components,
        spatial_axis_indices=tuple(
            base.coordinate_spec.axis_index(a)
            for a in base.coordinate_spec.spatial_axes
        ),
    )


def nuclear_cusp_slope(
    field: NuclearCuspField, atom: int = 0
) -> Array:
    r"""The realised cusp slope ``u_a'(0) = -Z_a`` for nucleus ``atom``."""
    out: Array = -jnp.asarray(field.charges)[atom]
    return out


def _cusp_cage_flatten(f: NuclearCuspField):
    leaves = (f.base, f.nuclei, f.charges, f.rates)
    aux = (
        f.velocity_names,
        f.passthrough_names,
        f.coordinate_spec,
        f.components,
        f.spatial_axis_indices,
    )
    return leaves, aux


def _cusp_cage_unflatten(aux, leaves):
    base, nuclei, charges, rates = leaves
    (
        velocity_names,
        passthrough_names,
        coordinate_spec,
        components,
        spatial_axis_indices,
    ) = aux
    obj = NuclearCuspField.__new__(NuclearCuspField)
    object.__setattr__(obj, "base", base)
    object.__setattr__(obj, "nuclei", nuclei)
    object.__setattr__(obj, "charges", charges)
    object.__setattr__(obj, "rates", rates)
    object.__setattr__(obj, "velocity_names", velocity_names)
    object.__setattr__(obj, "passthrough_names", passthrough_names)
    object.__setattr__(obj, "coordinate_spec", coordinate_spec)
    object.__setattr__(obj, "components", components)
    object.__setattr__(obj, "spatial_axis_indices", spatial_axis_indices)
    return obj


jax.tree_util.register_pytree_node(
    NuclearCuspField, _cusp_cage_flatten, _cusp_cage_unflatten,
)


__all__ = [
    "NuclearCuspField",
    "make_nuclear_cusp_field",
    "nuclear_cusp_slope",
]
