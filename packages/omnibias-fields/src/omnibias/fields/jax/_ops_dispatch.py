# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX ops dispatch -- thin entry points the views forward into."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from jax import Array
    from omnibias.fields._core.state import FieldState


def value(state: FieldState, name: str) -> Array:
    from omnibias.fields.jax.ops.basic import value as _value
    return _value(state, name)


def derivative(
    state: FieldState, name: str, *, axis: int | str, order: int = 1,
) -> Array:
    from omnibias.fields.jax.ops.basic import derivative as _derivative
    return _derivative(state, name, axis=axis, order=order)


def gradient(
    state: FieldState, name: str, *, axes: tuple[int | str, ...] | None = None,
) -> Array:
    from omnibias.fields.jax.ops.basic import gradient as _gradient
    return _gradient(state, name, axes=axes)


def divergence(state: FieldState, names: tuple[str, ...]) -> Array:
    from omnibias.fields.jax.ops.basic import divergence as _divergence
    return _divergence(state, names)


def laplacian(
    state: FieldState, name: str, *, axes: tuple[int | str, ...] | None = None,
) -> Array:
    from omnibias.fields.jax.ops.basic import laplacian as _laplacian
    return _laplacian(state, name, axes=axes)


def stack_components(state: FieldState, names: tuple[str, ...]) -> Array:
    from omnibias.fields.jax.ops.basic import stack_components as _stack
    return _stack(state, names)


def vector_derivative(
    state: FieldState, names: tuple[str, ...], *, axis: int | str, order: int = 1,
) -> Array:
    from omnibias.fields.jax.ops.basic import vector_derivative as _vd
    return _vd(state, names, axis=axis, order=order)


def mixed_partial(
    state: FieldState,
    name: str,
    axes: tuple[int | str, ...],
    orders: tuple[int, ...],
) -> Array:
    from omnibias.fields.jax.ops.basic import mixed_partial as _mp
    return _mp(state, name, axes, orders)


def biharmonic(state: FieldState, name: str) -> Array:
    from omnibias.fields.jax.ops.high_order import biharmonic as _b
    return _b(state, name)


def polylaplacian(state: FieldState, name: str, *, k: int) -> Array:
    from omnibias.fields.jax.ops.high_order import polylaplacian as _pl
    return _pl(state, name, k=k)


def hessian(
    state: FieldState,
    name: str,
    *,
    axes: tuple[int | str, ...] | None = None,
) -> Array:
    from omnibias.fields.jax.ops.high_order import hessian as _h
    return _h(state, name, axes=axes)


def spatial_hessian(state: FieldState, name: str) -> Array:
    from omnibias.fields.jax.ops.high_order import spatial_hessian as _sh
    return _sh(state, name)


def gradient_of_derivative(
    state: FieldState, name: str, *, axis: int | str,
) -> Array:
    from omnibias.fields.jax.ops.high_order import gradient_of_derivative as _god
    return _god(state, name, axis=axis)


def jacobian(state: FieldState, names: tuple[str, ...]) -> Array:
    from omnibias.fields.jax.ops.high_order import jacobian as _j
    return _j(state, names)


def vector_hessian(
    state: FieldState,
    names: tuple[str, ...],
    *,
    axes: tuple[int | str, ...] | None = None,
) -> Array:
    from omnibias.fields.jax.ops.high_order import vector_hessian as _vh
    return _vh(state, names, axes=axes)


def vector_laplacian(state: FieldState, names: tuple[str, ...]) -> Array:
    from omnibias.fields.jax.ops.high_order import vector_laplacian as _vl
    return _vl(state, names)


def vector_biharmonic(state: FieldState, names: tuple[str, ...]) -> Array:
    from omnibias.fields.jax.ops.high_order import vector_biharmonic as _vb
    return _vb(state, names)


def vector_polylaplacian(state: FieldState, names: tuple[str, ...], *, k: int) -> Array:
    from omnibias.fields.jax.ops.high_order import vector_polylaplacian as _vp
    return _vp(state, names, k=k)


def advection(
    state: FieldState,
    *,
    velocity: tuple[str, ...],
    target: tuple[str, ...] | None = None,
    scalar: str | None = None,
) -> Array:
    from omnibias.fields.jax.ops.nonlinear import advection as _a
    return _a(state, velocity=velocity, target=target, scalar=scalar)


def material_derivative(
    state: FieldState,
    *,
    velocity: tuple[str, ...],
    scalar: str | None = None,
) -> Array:
    from omnibias.fields.jax.ops.nonlinear import material_derivative as _md
    return _md(state, velocity=velocity, scalar=scalar)


def p_laplacian(
    state: FieldState, name: str, *, p: float, eps: float = 1e-8,
) -> Array:
    from omnibias.fields.jax.ops.nonlinear import p_laplacian as _pl
    return _pl(state, name, p=p, eps=eps)


def directional_derivative(
    state: FieldState, name: str, *, direction: Array,
) -> Array:
    from omnibias.fields.jax.ops.nonlinear import directional_derivative as _dd
    return _dd(state, name, direction=direction)


def curl(state: FieldState, names: tuple[str, ...]) -> Array:
    from omnibias.fields.jax.ops.vector import curl as _c
    return _c(state, names)


def vorticity(state: FieldState, names: tuple[str, ...]) -> Array:
    return curl(state, names)


def strain_rate(state: FieldState, names: tuple[str, ...]) -> Array:
    from omnibias.fields.jax.ops.vector import strain_rate as _s
    return _s(state, names)


def deformation_gradient(state: FieldState, names: tuple[str, ...]) -> Array:
    from omnibias.fields.jax.ops.vector import deformation_gradient as _d
    return _d(state, names)


def spatial_jacobian(state: FieldState, names: tuple[str, ...]) -> Array:
    from omnibias.fields.jax.ops.vector import spatial_jacobian as _sj
    return _sj(state, names)


def integrate(state: FieldState, name: str, *, rule: Any) -> Array:
    from omnibias.fields.jax.ops.integral import integrate as _i
    return _i(state, name, rule=rule)


def line_integral(
    state: FieldState, name: str, curve: Any, *, rule: Any,
) -> Array:
    from omnibias.fields.jax.ops.integral import line_integral as _li
    return _li(state, name, curve, rule=rule)


def inner_product(
    state: FieldState, name_a: str, name_b: str, *, rule: Any, weight: str | None = None,
) -> Array:
    from omnibias.fields.jax.ops.norms import inner_product as _ip
    return _ip(state, name_a, name_b, rule=rule, weight=weight)


def l2_norm(state: FieldState, name: str, *, rule: Any) -> Array:
    from omnibias.fields.jax.ops.norms import l2_norm as _l2
    return _l2(state, name, rule=rule)


def sobolev_norm(
    state: FieldState, name: str, *, rule: Any, k: int = 1,
    weights: tuple[float, ...] | None = None,
) -> Array:
    from omnibias.fields.jax.ops.norms import sobolev_norm as _sn
    return _sn(state, name, rule=rule, k=k, weights=weights)


def tensor_divergence(state: FieldState, sigma_names: Any) -> Array:
    from omnibias.fields.jax.ops.tensor import tensor_divergence as _td
    return _td(state, sigma_names)


def dz(
    state: FieldState, re_name: str, im_name: str, *,
    real_axis: int | str = "x", imag_axis: int | str = "y",
) -> tuple[Array, Array]:
    from omnibias.fields.jax.ops.complex import dz as _dz
    return _dz(state, re_name, im_name, real_axis=real_axis, imag_axis=imag_axis)


def dzbar(
    state: FieldState, re_name: str, im_name: str, *,
    real_axis: int | str = "x", imag_axis: int | str = "y",
) -> tuple[Array, Array]:
    from omnibias.fields.jax.ops.complex import dzbar as _dzbar
    return _dzbar(state, re_name, im_name, real_axis=real_axis, imag_axis=imag_axis)


# ---------------- vector-calculus identities (vector.py) -------


def gradient_of_divergence(state: FieldState, names: tuple[str, ...]) -> Array:
    from omnibias.fields.jax.ops.vector import gradient_of_divergence as _gd
    return _gd(state, names)


def curl_of_curl(state: FieldState, names: tuple[str, ...]) -> Array:
    from omnibias.fields.jax.ops.vector import curl_of_curl as _cc
    return _cc(state, names)


def rate_of_rotation_tensor(state: FieldState, names: tuple[str, ...]) -> Array:
    from omnibias.fields.jax.ops.vector import rate_of_rotation_tensor as _rr
    return _rr(state, names)


def div(state: FieldState, names: tuple[str, ...]) -> Array:
    """Alias for :func:`divergence`."""
    return divergence(state, names)


def rot(state: FieldState, names: tuple[str, ...]) -> Array:
    """Alias for :func:`curl`."""
    return curl(state, names)


# ---------------- conservation / flux / wave (conservation.py) -


def grad_squared_norm(state: FieldState, name: str) -> Array:
    from omnibias.fields.jax.ops.conservation import grad_squared_norm as _g
    return _g(state, name)


def gradient_of_composition(state: FieldState, name: str, fprime: Any) -> Array:
    from omnibias.fields.jax.ops.conservation import gradient_of_composition as _gc
    return _gc(state, name, fprime)


def laplacian_of_composition(
    state: FieldState, name: str, fprime: Any, fsecond: Any,
) -> Array:
    from omnibias.fields.jax.ops.conservation import laplacian_of_composition as _lc
    return _lc(state, name, fprime, fsecond)


def diffusive_flux(
    state: FieldState, name: str, *, diffusivity: float | str = 1.0,
) -> Array:
    from omnibias.fields.jax.ops.conservation import diffusive_flux as _df
    return _df(state, name, diffusivity=diffusivity)


def flux_divergence(state: FieldState, names: tuple[str, ...]) -> Array:
    from omnibias.fields.jax.ops.conservation import flux_divergence as _fd
    return _fd(state, names)


def variable_coefficient_diffusion(
    state: FieldState, name: str, *, diffusivity: float | str = 1.0,
) -> Array:
    from omnibias.fields.jax.ops.conservation import variable_coefficient_diffusion as _vd
    return _vd(state, name, diffusivity=diffusivity)


def conservation_residual(
    state: FieldState, *, density: str, flux: tuple[str, ...],
    source: str | float | None = None,
) -> Array:
    from omnibias.fields.jax.ops.conservation import conservation_residual as _cr
    return _cr(state, density=density, flux=flux, source=source)


def advection_diffusion_residual(
    state: FieldState, *, scalar: str, velocity: tuple[str, ...],
    diffusivity: float | str = 1.0, source: str | float | None = None,
) -> Array:
    from omnibias.fields.jax.ops.conservation import advection_diffusion_residual as _ad
    return _ad(
        state, scalar=scalar, velocity=velocity,
        diffusivity=diffusivity, source=source,
    )


def dalembertian(
    state: FieldState, name: str, *, c: float = 1.0, signature: str = "mostly_plus",
) -> Array:
    from omnibias.fields.jax.ops.conservation import dalembertian as _db
    return _db(state, name, c=c, signature=signature)


def wave_operator(
    state: FieldState, name: str, *, c: float = 1.0, signature: str = "mostly_plus",
) -> Array:
    from omnibias.fields.jax.ops.conservation import wave_operator as _wo
    return _wo(state, name, c=c, signature=signature)


def skew_symmetric_advection(
    state: FieldState, *, velocity: tuple[str, ...], scalar: str | None = None,
) -> Array:
    from omnibias.fields.jax.ops.nonlinear import skew_symmetric_advection as _ssa
    return _ssa(state, velocity=velocity, scalar=scalar)


def tensor_double_dot(a: Array, b: Array) -> Array:
    from omnibias.fields.jax.ops.tensor import tensor_double_dot as _tdd
    return _tdd(a, b)


# ---------------- continuum mechanics / fluids (mechanics.py) --


















# ---------------- chemistry / transport (chemistry.py) --------














# ---------------- electromagnetism (electromagnetism.py) ------




















# ---------------- magnetohydrodynamics (mhd.py) ----------------
















# ---------------- kinetic theory (kinetic.py) ------------------
















def list_ops() -> tuple[str, ...]:
    return tuple(sorted(
        n for n, fn in globals().items()
        if callable(fn) and not n.startswith("_") and n not in ("list_ops",)
        and getattr(fn, "__module__", None) == __name__
    ))


__all__ = [
    "advection",
    "advection_diffusion_residual",
    "biharmonic",
    "conservation_residual",
    "curl",
    "curl_of_curl",
    "dalembertian",
    "deformation_gradient",
    "derivative",
    "diffusive_flux",
    "directional_derivative",
    "div",
    "divergence",
    "dz",
    "dzbar",
    "flux_divergence",
    "grad_squared_norm",
    "gradient",
    "gradient_of_composition",
    "gradient_of_derivative",
    "gradient_of_divergence",
    "hessian",
    "inner_product",
    "integrate",
    "jacobian",
    "l2_norm",
    "laplacian",
    "laplacian_of_composition",
    "line_integral",
    "list_ops",
    "material_derivative",
    "mixed_partial",
    "p_laplacian",
    "polylaplacian",
    "rate_of_rotation_tensor",
    "rot",
    "skew_symmetric_advection",
    "sobolev_norm",
    "spatial_hessian",
    "spatial_jacobian",
    "stack_components",
    "strain_rate",
    "tensor_divergence",
    "tensor_double_dot",
    "value",
    "variable_coefficient_diffusion",
    "vector_biharmonic",
    "vector_derivative",
    "vector_hessian",
    "vector_laplacian",
    "vector_polylaplacian",
    "vorticity",
    "wave_operator",
]
