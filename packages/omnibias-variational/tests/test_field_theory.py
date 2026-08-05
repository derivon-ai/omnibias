# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Classical field theory on the Klein-Gordon field.

Density ``L = 1/2((d_t phi)^2 - (d_x phi)^2) - 1/2 m^2 phi^2`` on axes ``(x, t)``.
The generic field Euler-Lagrange operator must reproduce the Klein-Gordon /
d'Alembertian operator ``phi_tt - phi_xx + m^2 phi`` (checked off the dispersion
relation, so both sides are non-zero and equal), and vanish on a plane wave that
satisfies ``w^2 = k^2 + m^2``. The stress-energy ``T^t_t`` is the energy density
``1/2 phi_t^2 + 1/2 phi_x^2 + 1/2 m^2 phi^2``. All float64, torch/jax parity.
"""

from __future__ import annotations

import numpy as np
import torch
from _traj import jax_planewave_state, to_np, torch_planewave_state
from omnibias.fields._core.quadrature import gauss_legendre
from omnibias.fields.torch.ops.basic import derivative, value
from omnibias.fields.torch.ops.integral import quadrature_nodes
from omnibias.variational import LagrangianDensity
from omnibias.variational.jax import ops as jv
from omnibias.variational.torch import ops as tv

K = 0.9
M = 0.7
XT = np.array(
    [[0.1, 0.0], [0.5, 0.3], [-0.4, 0.8], [1.2, -0.6], [0.7, 1.1]],
    dtype=np.float64,
)


def _kg_density(m):  # type: ignore[no-untyped-def]
    def fn(phi, dphi, x):  # type: ignore[no-untyped-def]
        phi_x = dphi[..., 0, 0]
        phi_t = dphi[..., 0, 1]
        return 0.5 * (phi_t**2 - phi_x**2) - 0.5 * m**2 * (phi[..., 0] ** 2)

    return LagrangianDensity(fn, fields=("phi",))


def test_field_el_matches_klein_gordon_operator() -> None:
    # Off the dispersion relation: both sides are non-zero and must agree.
    omega = 1.9  # != sqrt(K^2 + M^2)
    state = torch_planewave_state(K, omega, XT)
    res = to_np(tv.field_euler_lagrange_residual(state, _kg_density(M)))[:, 0]
    phi_tt = to_np(derivative(state, "phi", axis="t", order=2))
    phi_xx = to_np(derivative(state, "phi", axis="x", order=2))
    phi = to_np(value(state, "phi"))
    manual = phi_tt - phi_xx + M**2 * phi
    assert np.max(np.abs(manual)) > 1e-3  # genuinely off-solution
    assert np.allclose(res, manual, atol=1e-10)


def test_field_el_zero_on_dispersion() -> None:
    omega = float(np.sqrt(K**2 + M**2))
    state = torch_planewave_state(K, omega, XT)
    res = to_np(tv.field_euler_lagrange_residual(state, _kg_density(M)))
    assert np.allclose(res, 0.0, atol=1e-10)


def test_stress_energy_t00_is_energy_density() -> None:
    omega = 1.9
    state = torch_planewave_state(K, omega, XT)
    T = to_np(tv.stress_energy_tensor(state, _kg_density(M)))  # (B, 2, 2), axes (x, t)
    phi_t = to_np(derivative(state, "phi", axis="t", order=1))
    phi_x = to_np(derivative(state, "phi", axis="x", order=1))
    phi = to_np(value(state, "phi"))
    energy_density = 0.5 * phi_t**2 + 0.5 * phi_x**2 + 0.5 * M**2 * phi**2
    assert np.allclose(T[:, 1, 1], energy_density, atol=1e-10)


def test_action_density_matches_manual_quadrature() -> None:
    rule = gauss_legendre([(-1.0, 1.0), (0.0, 2.0)], (10, 12))  # (x, t) box
    nodes = quadrature_nodes(rule, like=torch.zeros(1, dtype=torch.float64))
    state = torch_planewave_state(K, 1.9, to_np(nodes))
    dens = _kg_density(M)
    S = to_np(tv.action_density(state, dens, rule=rule))
    manual = to_np(tv.integrate_values(tv.density_values(state, dens), rule=rule))
    assert np.allclose(S, manual, rtol=1e-13, atol=1e-13)


def test_field_theory_cross_backend() -> None:
    omega = 1.9
    dens = _kg_density(M)
    ts = torch_planewave_state(K, omega, XT)
    js = jax_planewave_state(K, omega, XT)
    assert np.allclose(
        to_np(tv.field_euler_lagrange_residual(ts, dens)),
        to_np(jv.field_euler_lagrange_residual(js, dens)),
        rtol=1e-12, atol=1e-12,
    )
    assert np.allclose(
        to_np(tv.stress_energy_tensor(ts, dens)),
        to_np(jv.stress_energy_tensor(js, dens)),
        rtol=1e-12, atol=1e-12,
    )
