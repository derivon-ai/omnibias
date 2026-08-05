# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Pure-Python catalog of the field operator surface.

Every operator dispatched through ``state.ops`` is described here once by an
:class:`OperatorInfo` record carrying its ``domain`` (physics / maths area),
a one-line ``formula`` signature, and a ``closed_form`` honesty flag. The
catalog is the single source of truth for :func:`list_operators` (discovery)
and for the grouped ``docs/operators.md`` reference; ``tests/test_catalog.py``
asserts it stays in lock-step with the actual torch / jax dispatch surface, so
adding an operator without cataloguing it (or vice versa) fails CI.

The module is pure Python (no torch / jax / numpy import), matching the rest of
``omnibias.fields._core``. ``closed_form`` is ``True`` for every operator here:
the field surface is built entirely from the closed-form sigma tower and exact
compositions of it. The flag exists so that non-local / approximate operators
contributed by downstream packages (e.g. fractional calculus) can be catalogued
honestly alongside these.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Ordered domain tags, coarse maths/physics areas, used to group the catalog.
DOMAINS: tuple[str, ...] = (
    "calculus",
    "vector_calculus",
    "conservation",
    "fluids",
    "mechanics",
    "chemistry",
    "electromagnetism",
    "magnetohydrodynamics",
    "kinetic",
    "complex",
    "integral",
)


@dataclass(frozen=True)
class OperatorInfo:
    """Metadata for one dispatchable field operator.

    Parameters
    ----------
    name
        The ``state.ops.<name>`` entry point (also the DSL / ``ops`` module name).
    domain
        One of :data:`DOMAINS`.
    formula
        A compact ASCII signature of what the operator computes.
    closed_form
        ``True`` when the operator is the closed-form sigma tower or an exact
        composition of it (no autodiff, no grid approximation).
    backends
        Backends providing a bit-identical implementation.
    """

    name: str
    domain: str
    formula: str
    closed_form: bool = True
    backends: tuple[str, ...] = ("torch", "jax")


def _info(name: str, domain: str, formula: str) -> OperatorInfo:
    return OperatorInfo(name=name, domain=domain, formula=formula)


_CATALOG: tuple[OperatorInfo, ...] = (
    # --- calculus: primitive differential operators / accessors ---------
    _info("value", "calculus", "u"),
    _info("stack_components", "calculus", "(u_1, ..., u_C) -> (B, C)"),
    _info("derivative", "calculus", "d^k u / dx_a^k"),
    _info("mixed_partial", "calculus", "d^|orders| u / prod dx_a^{o_a}"),
    _info("gradient", "calculus", "grad u"),
    _info("divergence", "calculus", "div u"),
    _info("laplacian", "calculus", "Delta u"),
    _info("biharmonic", "calculus", "Delta^2 u"),
    _info("polylaplacian", "calculus", "Delta^k u"),
    _info("hessian", "calculus", "Hess u (all axes)"),
    _info("spatial_hessian", "calculus", "Hess u (spatial axes)"),
    _info("jacobian", "calculus", "J_ij = d u_i / dx_j (all axes)"),
    _info("spatial_jacobian", "calculus", "J_ij = d u_i / dx_j (spatial)"),
    _info("vector_derivative", "calculus", "d^k u_i / dx_a^k"),
    _info("vector_laplacian", "calculus", "(Delta u_1, ..., Delta u_C)"),
    _info("vector_biharmonic", "calculus", "(Delta^2 u_1, ..., Delta^2 u_C)"),
    _info("vector_hessian", "calculus", "stacked component Hessians"),
    _info("vector_polylaplacian", "calculus", "(Delta^k u_1, ..., Delta^k u_C)"),
    _info("gradient_of_derivative", "calculus", "grad(d u / dx_a)"),
    _info("directional_derivative", "calculus", "grad u . n"),
    _info("div", "calculus", "div u (alias of divergence)"),
    # --- vector calculus -----------------------------------------------
    _info("curl", "vector_calculus", "curl u (rot)"),
    _info("rot", "vector_calculus", "curl u (alias)"),
    _info("vorticity", "vector_calculus", "omega = curl u"),
    _info("curl_of_curl", "vector_calculus", "curl(curl u)"),
    _info("gradient_of_divergence", "vector_calculus", "grad(div u)"),
    _info("grad_squared_norm", "vector_calculus", "|grad u|^2"),
    _info("deformation_gradient", "vector_calculus", "F = grad u"),
    _info("strain_rate", "vector_calculus", "eps = 0.5 (J + J^T)"),
    _info("rate_of_rotation_tensor", "vector_calculus", "W = 0.5 (J - J^T)"),
    _info("p_laplacian", "vector_calculus", "div(|grad u|^{p-2} grad u)"),
    # --- conservation / flux / wave ------------------------------------
    _info("diffusive_flux", "conservation", "F = -D grad u"),
    _info("flux_divergence", "conservation", "div F"),
    _info("variable_coefficient_diffusion", "conservation", "div(D grad u)"),
    _info("conservation_residual", "conservation", "d_t rho + div F - s"),
    _info("advection_diffusion_residual", "conservation", "d_t c + (u.grad)c - div(D grad c) - s"),
    _info("gradient_of_composition", "conservation", "grad f(u) = f'(u) grad u"),
    _info("laplacian_of_composition", "conservation", "Delta f(u) = f'(u) Delta u + f''(u)|grad u|^2"),
    _info("dalembertian", "conservation", "box u = Delta u - c^-2 d_tt u"),
    _info("wave_operator", "conservation", "box u (alias of dalembertian)"),
    # --- fluids / continuum kinematics ---------------------------------
    _info("advection", "fluids", "(u.grad) u"),
    _info("material_derivative", "fluids", "D/Dt = d_t + (u.grad)"),
    _info("skew_symmetric_advection", "fluids", "0.5[(u.grad)c + div(u c)]"),
    _info("velocity_from_streamfunction", "fluids", "(d_y psi, -d_x psi)"),
    _info("vorticity_from_streamfunction", "fluids", "omega = -Delta psi"),
    # --- continuum mechanics -------------------------------------------
    _info("newtonian_stress", "mechanics", "sigma = -p I + 2 mu eps"),
    _info("linear_elastic_stress", "mechanics", "sigma = lam tr(eps) I + 2 mu eps"),
    _info("viscous_dissipation", "mechanics", "Phi = 2 mu eps:eps"),
    _info("stokes_residual", "mechanics", "mu Delta u - grad p + f"),
    _info("navier_cauchy_residual", "mechanics", "(lam+mu) grad(div u) + mu Delta u + f"),
    _info("stress_divergence", "mechanics", "div sigma"),
    _info("tensor_divergence", "mechanics", "(div T)_i = d_j T_ij"),
    _info("tensor_double_dot", "mechanics", "A:B = A_ij B_ij"),
    # --- chemistry / transport -----------------------------------------
    _info("fickian_flux", "chemistry", "J = -D grad c"),
    _info("nernst_planck_flux", "chemistry", "J = -D grad c - z mu F c grad phi"),
    _info("nernst_planck_residual", "chemistry", "d_t c + div J_NP - s"),
    _info("darcy_flux", "chemistry", "q = -(k/mu) grad p"),
    _info("reaction_diffusion_residual", "chemistry", "d_t c - div(D grad c) - R(c) - s"),
    _info("poisson_residual", "chemistry", "div(eps grad phi) + rho"),
    # --- electromagnetism (3-D Maxwell, natural units) -----------------
    _info("faraday_residual", "electromagnetism", "d_t B + curl E"),
    _info("ampere_residual", "electromagnetism", "d_t E - curl B + J"),
    _info("gauss_residual", "electromagnetism", "div E - rho"),
    _info("gauss_magnetic_residual", "electromagnetism", "div B"),
    _info("poynting_vector", "electromagnetism", "S = E x B"),
    _info("magnetic_field_from_potential", "electromagnetism", "B = curl A"),
    _info("electric_field_from_potentials", "electromagnetism", "E = -grad phi - d_t A"),
    _info("lorenz_gauge_residual", "electromagnetism", "d_t phi + div A"),
    _info("vector_dalembertian", "electromagnetism", "(box u_1, ..., box u_C)"),
    # --- magnetohydrodynamics (Alfven units, mu_0 = rho_0 = 1) ---------
    _info("current_density", "magnetohydrodynamics", "J = curl B"),
    _info("lorentz_force", "magnetohydrodynamics", "J x B"),
    _info("magnetic_pressure", "magnetohydrodynamics", "p_B = |B|^2 / 2"),
    _info("maxwell_stress_tensor", "magnetohydrodynamics",
          "T_ij = E_i E_j + B_i B_j - 0.5 d_ij (|E|^2 + |B|^2)"),
    _info("magnetic_divergence", "magnetohydrodynamics", "div B (solenoidal constraint)"),
    _info("induction_residual", "magnetohydrodynamics", "d_t B - curl(u x B) - eta Delta B"),
    _info("ideal_mhd_momentum_residual", "magnetohydrodynamics",
          "rho(d_t u + (u.grad)u) + grad p - J x B - nu Delta u - f"),
    # --- kinetic theory (phase-space transport, natural units) ---------
    _info("vlasov_residual", "kinetic", "d_t f + v.grad_x f + (F/m).grad_v f"),
    _info("bgk_collision", "kinetic", "-(f - f_eq)/tau"),
    _info("bgk_vlasov_residual", "kinetic", "L f + (f - f_eq)/tau"),
    _info("maxwellian", "kinetic", "f_eq = n (m/2piT)^{d/2} exp(-m|v-u|^2/2T)"),
    _info("number_density", "kinetic", "n = int f dv"),
    _info("momentum_density", "kinetic", "int v f dv"),
    _info("kinetic_energy_density", "kinetic", "int 0.5 |v|^2 f dv"),
    # --- complex / Wirtinger -------------------------------------------
    _info("dz", "complex", "d/dz = 0.5(d_x - i d_y)"),
    _info("dzbar", "complex", "d/dzbar = 0.5(d_x + i d_y)"),
    # --- integral / inner products / norms -----------------------------
    _info("integrate", "integral", "definite integral over the domain"),
    _info("line_integral", "integral", "int_C grad u . dr = u(end) - u(start)"),
    _info("inner_product", "integral", "<u, v> over the domain"),
    _info("l2_norm", "integral", "||u||_{L2}"),
    _info("sobolev_norm", "integral", "||u||_{H^k}"),
)

#: ``name -> OperatorInfo`` lookup table.
_BY_NAME: dict[str, OperatorInfo] = {op.name: op for op in _CATALOG}


def list_operators(domain: str | None = None) -> tuple[OperatorInfo, ...]:
    """Return catalogued operators, optionally filtered by ``domain``.

    Parameters
    ----------
    domain
        One of :data:`DOMAINS`. ``None`` (default) returns every operator,
        ordered by domain then name.

    Examples
    --------
    >>> from omnibias.fields import list_operators
    >>> [op.name for op in list_operators(domain="electromagnetism")][:3]
    ['ampere_residual', 'electric_field_from_potentials', 'faraday_residual']
    """
    if domain is not None and domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; choose from {DOMAINS!r}")
    ops = _CATALOG if domain is None else tuple(op for op in _CATALOG if op.domain == domain)
    return tuple(sorted(ops, key=lambda op: (DOMAINS.index(op.domain), op.name)))


def get_operator(name: str) -> OperatorInfo:
    """Return the :class:`OperatorInfo` for ``name`` (raises ``KeyError`` if absent)."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"no catalogued operator named {name!r}") from None


def operator_names() -> frozenset[str]:
    """The set of all catalogued operator names."""
    return frozenset(_BY_NAME)


__all__ = [
    "DOMAINS",
    "OperatorInfo",
    "get_operator",
    "list_operators",
    "operator_names",
]
