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


def velocity_from_streamfunction(state: FieldState, psi: str) -> Array:
    from omnibias.fields.jax.ops.mechanics import velocity_from_streamfunction as _v
    return _v(state, psi)


def vorticity_from_streamfunction(state: FieldState, psi: str) -> Array:
    from omnibias.fields.jax.ops.mechanics import vorticity_from_streamfunction as _w
    return _w(state, psi)


def newtonian_stress(
    state: FieldState, velocity: tuple[str, ...], *,
    viscosity: float = 1.0, pressure: str | None = None,
) -> Array:
    from omnibias.fields.jax.ops.mechanics import newtonian_stress as _ns
    return _ns(state, velocity, viscosity=viscosity, pressure=pressure)


def linear_elastic_stress(
    state: FieldState, displacement: tuple[str, ...], *,
    lam: float = 1.0, mu: float = 1.0,
) -> Array:
    from omnibias.fields.jax.ops.mechanics import linear_elastic_stress as _le
    return _le(state, displacement, lam=lam, mu=mu)


def viscous_dissipation(
    state: FieldState, velocity: tuple[str, ...], *, viscosity: float = 1.0,
) -> Array:
    from omnibias.fields.jax.ops.mechanics import viscous_dissipation as _vd
    return _vd(state, velocity, viscosity=viscosity)


def stress_divergence(state: FieldState, sigma_names: Any) -> Array:
    from omnibias.fields.jax.ops.mechanics import stress_divergence as _sd
    return _sd(state, sigma_names)


def stokes_residual(
    state: FieldState, *, velocity: tuple[str, ...], pressure: str,
    viscosity: float = 1.0, body_force: Any = None,
) -> Array:
    from omnibias.fields.jax.ops.mechanics import stokes_residual as _sr
    return _sr(
        state, velocity=velocity, pressure=pressure,
        viscosity=viscosity, body_force=body_force,
    )


def navier_cauchy_residual(
    state: FieldState, *, displacement: tuple[str, ...],
    lam: float = 1.0, mu: float = 1.0, body_force: Any = None,
) -> Array:
    from omnibias.fields.jax.ops.mechanics import navier_cauchy_residual as _nc
    return _nc(
        state, displacement=displacement, lam=lam, mu=mu, body_force=body_force,
    )


# ---------------- chemistry / transport (chemistry.py) --------


def fickian_flux(state: FieldState, name: str, *, diffusivity: float | str = 1.0) -> Array:
    from omnibias.fields.jax.ops.chemistry import fickian_flux as _ff
    return _ff(state, name, diffusivity=diffusivity)


def nernst_planck_flux(
    state: FieldState, concentration: str, potential: str, *,
    diffusivity: float | str = 1.0, valence: float = 1.0,
    mobility: float = 1.0, faraday: float = 1.0,
) -> Array:
    from omnibias.fields.jax.ops.chemistry import nernst_planck_flux as _np
    return _np(
        state, concentration, potential, diffusivity=diffusivity,
        valence=valence, mobility=mobility, faraday=faraday,
    )


def darcy_flux(
    state: FieldState, pressure: str, *,
    permeability: float | str = 1.0, viscosity: float = 1.0,
) -> Array:
    from omnibias.fields.jax.ops.chemistry import darcy_flux as _df
    return _df(state, pressure, permeability=permeability, viscosity=viscosity)


def nernst_planck_residual(
    state: FieldState, *, concentration: str, potential: str,
    diffusivity: float | str = 1.0, valence: float = 1.0,
    mobility: float = 1.0, faraday: float = 1.0,
    source: str | float | None = None,
) -> Array:
    from omnibias.fields.jax.ops.chemistry import nernst_planck_residual as _npr
    return _npr(
        state, concentration=concentration, potential=potential,
        diffusivity=diffusivity, valence=valence, mobility=mobility,
        faraday=faraday, source=source,
    )


def reaction_diffusion_residual(
    state: FieldState, *, scalar: str, diffusivity: float | str = 1.0,
    reaction: Any = None, source: str | float | None = None,
) -> Array:
    from omnibias.fields.jax.ops.chemistry import reaction_diffusion_residual as _rd
    return _rd(
        state, scalar=scalar, diffusivity=diffusivity, reaction=reaction, source=source,
    )


def poisson_residual(
    state: FieldState, potential: str, *,
    source: str | float | None = None, permittivity: float | str = 1.0,
) -> Array:
    from omnibias.fields.jax.ops.chemistry import poisson_residual as _pr
    return _pr(state, potential, source=source, permittivity=permittivity)


# ---------------- electromagnetism (electromagnetism.py) ------


def faraday_residual(
    state: FieldState, *, electric: tuple[str, ...], magnetic: tuple[str, ...],
) -> Array:
    from omnibias.fields.jax.ops.electromagnetism import faraday_residual as _f
    return _f(state, electric=electric, magnetic=magnetic)


def ampere_residual(
    state: FieldState, *, electric: tuple[str, ...], magnetic: tuple[str, ...],
    current: tuple[str, ...] | None = None,
) -> Array:
    from omnibias.fields.jax.ops.electromagnetism import ampere_residual as _a
    return _a(state, electric=electric, magnetic=magnetic, current=current)


def gauss_residual(
    state: FieldState, *, electric: tuple[str, ...], charge: str | float | None = None,
) -> Array:
    from omnibias.fields.jax.ops.electromagnetism import gauss_residual as _g
    return _g(state, electric=electric, charge=charge)


def gauss_magnetic_residual(state: FieldState, *, magnetic: tuple[str, ...]) -> Array:
    from omnibias.fields.jax.ops.electromagnetism import gauss_magnetic_residual as _gm
    return _gm(state, magnetic=magnetic)


def poynting_vector(
    state: FieldState, *, electric: tuple[str, ...], magnetic: tuple[str, ...],
) -> Array:
    from omnibias.fields.jax.ops.electromagnetism import poynting_vector as _p
    return _p(state, electric=electric, magnetic=magnetic)


def magnetic_field_from_potential(state: FieldState, *, potential: tuple[str, ...]) -> Array:
    from omnibias.fields.jax.ops.electromagnetism import magnetic_field_from_potential as _m
    return _m(state, potential=potential)


def electric_field_from_potentials(
    state: FieldState, *, scalar_potential: str,
    vector_potential: tuple[str, ...] | None = None,
) -> Array:
    from omnibias.fields.jax.ops.electromagnetism import electric_field_from_potentials as _e
    return _e(state, scalar_potential=scalar_potential, vector_potential=vector_potential)


def lorenz_gauge_residual(
    state: FieldState, *, scalar_potential: str, vector_potential: tuple[str, ...],
) -> Array:
    from omnibias.fields.jax.ops.electromagnetism import lorenz_gauge_residual as _l
    return _l(state, scalar_potential=scalar_potential, vector_potential=vector_potential)


def vector_dalembertian(
    state: FieldState, names: tuple[str, ...], *,
    c: float = 1.0, signature: str = "mostly_plus",
) -> Array:
    from omnibias.fields.jax.ops.electromagnetism import vector_dalembertian as _vd
    return _vd(state, names, c=c, signature=signature)


# ---------------- magnetohydrodynamics (mhd.py) ----------------


def current_density(state: FieldState, *, magnetic: tuple[str, ...]) -> Array:
    from omnibias.fields.jax.ops.mhd import current_density as _cd
    return _cd(state, magnetic=magnetic)


def lorentz_force(
    state: FieldState, *, magnetic: tuple[str, ...],
    current: tuple[str, ...] | None = None,
) -> Array:
    from omnibias.fields.jax.ops.mhd import lorentz_force as _lf
    return _lf(state, magnetic=magnetic, current=current)


def magnetic_pressure(state: FieldState, *, magnetic: tuple[str, ...]) -> Array:
    from omnibias.fields.jax.ops.mhd import magnetic_pressure as _mp
    return _mp(state, magnetic=magnetic)


def maxwell_stress_tensor(
    state: FieldState, *, magnetic: tuple[str, ...],
    electric: tuple[str, ...] | None = None,
) -> Array:
    from omnibias.fields.jax.ops.mhd import maxwell_stress_tensor as _mst
    return _mst(state, magnetic=magnetic, electric=electric)


def magnetic_divergence(state: FieldState, *, magnetic: tuple[str, ...]) -> Array:
    from omnibias.fields.jax.ops.mhd import magnetic_divergence as _md
    return _md(state, magnetic=magnetic)


def induction_residual(
    state: FieldState, *, velocity: tuple[str, ...], magnetic: tuple[str, ...],
    resistivity: float = 0.0,
) -> Array:
    from omnibias.fields.jax.ops.mhd import induction_residual as _ir
    return _ir(state, velocity=velocity, magnetic=magnetic, resistivity=resistivity)


def ideal_mhd_momentum_residual(
    state: FieldState, *, velocity: tuple[str, ...], magnetic: tuple[str, ...],
    pressure: str, density: float = 1.0, viscosity: float = 0.0,
    current: tuple[str, ...] | None = None,
    forcing: tuple[str, ...] | Array | None = None,
) -> Array:
    from omnibias.fields.jax.ops.mhd import ideal_mhd_momentum_residual as _mm
    return _mm(
        state, velocity=velocity, magnetic=magnetic, pressure=pressure,
        density=density, viscosity=viscosity, current=current, forcing=forcing,
    )


# ---------------- kinetic theory (kinetic.py) ------------------


def vlasov_residual(
    state: FieldState, name: str, *,
    position_axes: tuple[str, ...], velocity_axes: tuple[str, ...],
    force: tuple[str, ...] | Array | None = None, mass: float = 1.0,
) -> Array:
    from omnibias.fields.jax.ops.kinetic import vlasov_residual as _vr
    return _vr(
        state, name, position_axes=position_axes, velocity_axes=velocity_axes,
        force=force, mass=mass,
    )


def bgk_collision(
    state: FieldState, name: str, *, equilibrium: str | Array, tau: float,
) -> Array:
    from omnibias.fields.jax.ops.kinetic import bgk_collision as _bc
    return _bc(state, name, equilibrium=equilibrium, tau=tau)


def bgk_vlasov_residual(
    state: FieldState, name: str, *,
    position_axes: tuple[str, ...], velocity_axes: tuple[str, ...],
    equilibrium: str | Array, tau: float,
    force: tuple[str, ...] | Array | None = None, mass: float = 1.0,
) -> Array:
    from omnibias.fields.jax.ops.kinetic import bgk_vlasov_residual as _bvr
    return _bvr(
        state, name, position_axes=position_axes, velocity_axes=velocity_axes,
        equilibrium=equilibrium, tau=tau, force=force, mass=mass,
    )


def maxwellian(
    state: FieldState, *, velocity_axes: tuple[str, ...],
    density: float | Array = 1.0,
    bulk_velocity: tuple[float, ...] | Array | None = None,
    temperature: float | Array = 1.0, mass: float = 1.0,
) -> Array:
    from omnibias.fields.jax.ops.kinetic import maxwellian as _mx
    return _mx(
        state, velocity_axes=velocity_axes, density=density,
        bulk_velocity=bulk_velocity, temperature=temperature, mass=mass,
    )


def number_density(state: FieldState, name: str, *, rule: Any) -> Array:
    from omnibias.fields.jax.ops.kinetic import number_density as _nd
    return _nd(state, name, rule=rule)


def momentum_density(
    state: FieldState, name: str, *, rule: Any, velocity_axes: tuple[str, ...],
) -> Array:
    from omnibias.fields.jax.ops.kinetic import momentum_density as _md
    return _md(state, name, rule=rule, velocity_axes=velocity_axes)


def kinetic_energy_density(
    state: FieldState, name: str, *, rule: Any, velocity_axes: tuple[str, ...],
) -> Array:
    from omnibias.fields.jax.ops.kinetic import kinetic_energy_density as _ked
    return _ked(state, name, rule=rule, velocity_axes=velocity_axes)


def list_ops() -> tuple[str, ...]:
    return tuple(sorted(
        n for n, fn in globals().items()
        if callable(fn) and not n.startswith("_") and n not in ("list_ops",)
        and getattr(fn, "__module__", None) == __name__
    ))


__all__ = [
    "advection",
    "advection_diffusion_residual",
    "ampere_residual",
    "bgk_collision",
    "bgk_vlasov_residual",
    "biharmonic",
    "conservation_residual",
    "curl",
    "curl_of_curl",
    "current_density",
    "dalembertian",
    "darcy_flux",
    "deformation_gradient",
    "derivative",
    "diffusive_flux",
    "directional_derivative",
    "div",
    "divergence",
    "dz",
    "dzbar",
    "electric_field_from_potentials",
    "faraday_residual",
    "fickian_flux",
    "flux_divergence",
    "gauss_magnetic_residual",
    "gauss_residual",
    "grad_squared_norm",
    "gradient",
    "gradient_of_composition",
    "gradient_of_derivative",
    "gradient_of_divergence",
    "hessian",
    "ideal_mhd_momentum_residual",
    "induction_residual",
    "inner_product",
    "integrate",
    "jacobian",
    "kinetic_energy_density",
    "l2_norm",
    "laplacian",
    "laplacian_of_composition",
    "line_integral",
    "linear_elastic_stress",
    "list_ops",
    "lorentz_force",
    "lorenz_gauge_residual",
    "magnetic_divergence",
    "magnetic_field_from_potential",
    "magnetic_pressure",
    "material_derivative",
    "maxwell_stress_tensor",
    "maxwellian",
    "mixed_partial",
    "momentum_density",
    "navier_cauchy_residual",
    "nernst_planck_flux",
    "nernst_planck_residual",
    "newtonian_stress",
    "number_density",
    "p_laplacian",
    "poisson_residual",
    "polylaplacian",
    "poynting_vector",
    "rate_of_rotation_tensor",
    "reaction_diffusion_residual",
    "rot",
    "skew_symmetric_advection",
    "sobolev_norm",
    "spatial_hessian",
    "spatial_jacobian",
    "stack_components",
    "stokes_residual",
    "strain_rate",
    "stress_divergence",
    "tensor_divergence",
    "tensor_double_dot",
    "value",
    "variable_coefficient_diffusion",
    "vector_biharmonic",
    "vector_dalembertian",
    "vector_derivative",
    "vector_hessian",
    "vector_laplacian",
    "vector_polylaplacian",
    "velocity_from_streamfunction",
    "viscous_dissipation",
    "vlasov_residual",
    "vorticity",
    "vorticity_from_streamfunction",
    "wave_operator",
]
