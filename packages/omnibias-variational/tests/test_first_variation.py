# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""First (Gateaux) variation -- particle and field.

- Finite-difference cross-check: ``first_variation ~ (S[q+h eta] - S[q-h eta])/(2h)``
  for an anharmonic Lagrangian (so ``S`` is genuinely non-quadratic in the step).
- Weak == strong: for a boundary-vanishing ``eta`` the exact weak pairing equals
  ``integral (delta S/delta q) . eta dt`` (no boundary term).
- Hamilton's principle: ``delta S = 0`` on a solution for every admissible
  (endpoint-vanishing) ``eta`` -- particle and Klein-Gordon field.
- torch/jax parity.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from _traj import (
    AnalyticTrajField,
    jax_planewave_state,
    jax_separable_state,
    to_np,
    torch_planewave_state,
    torch_separable_state,
)
from omnibias.fields._core.quadrature import gauss_legendre
from omnibias.fields.torch.ops.basic import stack_components
from omnibias.fields.torch.ops.integral import quadrature_nodes
from omnibias.variational import Lagrangian, LagrangianDensity
from omnibias.variational.jax import ops as jv
from omnibias.variational.torch import ops as tv

A, B = 0.0, 2.0  # trajectory time domain [0, 2]
WQ = 1.3         # q(t) = cos(WQ t)
K, M = 0.9, 0.7  # Klein-Gordon plane-wave / mass


# ---- trajectory spec factories (xp-generic so torch/jax share them) --------
def _q_specs(xp):  # type: ignore[no-untyped-def]
    return {
        "q": (
            lambda t: xp.cos(WQ * t),
            lambda t: -WQ * xp.sin(WQ * t),
            lambda t: -(WQ**2) * xp.cos(WQ * t),
        )
    }


def _eta_general(xp):  # type: ignore[no-untyped-def]
    b, k = 0.37, 0.9  # does NOT vanish at the endpoints
    return {
        "q": (
            lambda t: b * xp.sin(k * t),
            lambda t: b * k * xp.cos(k * t),
            lambda t: -b * k**2 * xp.sin(k * t),
        )
    }


def _eta_bv(n):  # type: ignore[no-untyped-def]
    # sin(n pi t / 2): vanishes at t = 0 and t = 2 for integer n.
    c = n * math.pi / (B - A)

    def fac(xp):  # type: ignore[no-untyped-def]
        return {
            "q": (
                lambda t: xp.sin(c * t),
                lambda t: c * xp.cos(c * t),
                lambda t: -(c**2) * xp.sin(c * t),
            )
        }

    return fac


def _combine(sa, sb, h):  # type: ignore[no-untyped-def]
    out = {}
    for name in sa:
        out[name] = tuple(
            (lambda t, fa=fa, fb=fb: fa(t) + h * fb(t))
            for fa, fb in zip(sa[name], sb[name], strict=True)
        )
    return out


def _mk_torch_traj(specs_fac, t):  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch

    field = AnalyticTrajField(torch, _ops_dispatch, specs_fac(torch))
    return field(torch.as_tensor(t[:, None], dtype=torch.float64))


def _mk_jax_traj(specs_fac, t):  # type: ignore[no-untyped-def]
    import jax.numpy as jnp
    from omnibias.fields.jax import _ops_dispatch

    field = AnalyticTrajField(jnp, _ops_dispatch, specs_fac(jnp))
    return field(jnp.asarray(t[:, None], dtype=jnp.float64))


def _anharmonic(dof=("q",)):  # type: ignore[no-untyped-def]
    return Lagrangian(
        lambda q, qd, t: 0.5 * (qd**2).sum(-1) - 0.25 * (q**4).sum(-1), dof=dof
    )


def _sho(w, dof=("q",)):  # type: ignore[no-untyped-def]
    return Lagrangian(
        lambda q, qd, t: 0.5 * (qd**2).sum(-1) - 0.5 * w**2 * (q**2).sum(-1), dof=dof
    )


def _traj_rule_nodes(n=24):  # type: ignore[no-untyped-def]
    rule = gauss_legendre([(A, B)], n)
    nodes = quadrature_nodes(rule, like=torch.zeros(1, dtype=torch.float64))[:, 0].numpy()
    return rule, nodes


def test_first_variation_matches_finite_difference() -> None:
    rule, nodes = _traj_rule_nodes()
    lag = _anharmonic()
    q = _mk_torch_traj(_q_specs, nodes)
    eta = _mk_torch_traj(_eta_general, nodes)
    fv = float(to_np(tv.first_variation(q, lag, eta, rule=rule)))

    h = 1e-4
    qp = _mk_torch_traj(lambda xp: _combine(_q_specs(xp), _eta_general(xp), h), nodes)
    qm = _mk_torch_traj(lambda xp: _combine(_q_specs(xp), _eta_general(xp), -h), nodes)
    sp = float(to_np(tv.action(qp, lag, rule=rule)))
    sm = float(to_np(tv.action(qm, lag, rule=rule)))
    fd = (sp - sm) / (2 * h)

    assert abs(fv) > 1e-3  # a genuine, non-trivial variation
    assert abs(fv - fd) < 1e-6


def test_weak_equals_strong_for_boundary_vanishing_eta() -> None:
    rule, nodes = _traj_rule_nodes()
    lag = _sho(0.8)  # q = cos(1.3 t) is OFF this solution -> delta S/delta q != 0
    q = _mk_torch_traj(_q_specs, nodes)
    eta = _mk_torch_traj(_eta_bv(1), nodes)
    fv = to_np(tv.first_variation(q, lag, eta, rule=rule))

    fd_op = tv.functional_derivative(q, lag)  # (B, n_dof)
    eta_vals = stack_components(eta, ("q",))  # (B, n_dof)
    pairing = (fd_op * eta_vals).sum(-1)
    strong = to_np(tv.integrate_values(pairing, rule=rule))
    assert np.max(np.abs(to_np(fd_op))) > 1e-2  # non-trivial functional derivative
    assert np.allclose(fv, strong, atol=1e-9)


def test_hamilton_principle_delta_S_zero_on_solution() -> None:
    rule, nodes = _traj_rule_nodes()
    lag = _sho(WQ)  # now q = cos(WQ t) IS the solution
    q = _mk_torch_traj(_q_specs, nodes)
    for n in (1, 2, 3):  # a family of endpoint-vanishing variations
        eta = _mk_torch_traj(_eta_bv(n), nodes)
        fv = float(to_np(tv.first_variation(q, lag, eta, rule=rule)))
        assert abs(fv) < 1e-9


def test_first_variation_cross_backend() -> None:
    rule, nodes = _traj_rule_nodes()
    lag = _anharmonic()
    tq, teta = _mk_torch_traj(_q_specs, nodes), _mk_torch_traj(_eta_general, nodes)
    jq, jeta = _mk_jax_traj(_q_specs, nodes), _mk_jax_traj(_eta_general, nodes)
    t = float(to_np(tv.first_variation(tq, lag, teta, rule=rule)))
    j = float(to_np(jv.first_variation(jq, lag, jeta, rule=rule)))
    assert abs(t - j) < 1e-12


# ---- field (Klein-Gordon) --------------------------------------------------
_BOX = [(-1.0, 1.0), (0.0, 2.0)]  # (x, t)


def _kg_density(m):  # type: ignore[no-untyped-def]
    def fn(phi, dphi, x):  # type: ignore[no-untyped-def]
        phi_x = dphi[..., 0, 0]
        phi_t = dphi[..., 0, 1]
        return 0.5 * (phi_t**2 - phi_x**2) - 0.5 * m**2 * (phi[..., 0] ** 2)

    return LagrangianDensity(fn, fields=("phi",))


def _fx_bv(xp):  # type: ignore[no-untyped-def]
    c = math.pi / 2.0  # sin(pi (x + 1)/2): zero at x = -1 and x = 1
    return (lambda x: xp.sin(c * (x + 1.0)), lambda x: c * xp.cos(c * (x + 1.0)))


def _ft_bv(xp):  # type: ignore[no-untyped-def]
    c = math.pi / 2.0  # sin(pi t / 2): zero at t = 0 and t = 2
    return (lambda t: xp.sin(c * t), lambda t: c * xp.cos(c * t))


def _field_rule_nodes():  # type: ignore[no-untyped-def]
    rule = gauss_legendre(_BOX, (10, 12))
    nodes = quadrature_nodes(rule, like=torch.zeros(1, dtype=torch.float64))
    return rule, nodes


def test_field_weak_equals_strong_for_boundary_vanishing_eta() -> None:
    rule, nodes = _field_rule_nodes()
    dens = _kg_density(M)
    phi = torch_planewave_state(K, 1.9, to_np(nodes))  # off dispersion
    eta = torch_separable_state(_fx_bv, _ft_bv, to_np(nodes))
    fv = to_np(tv.first_variation_density(phi, dens, eta, rule=rule))

    ffd = tv.field_functional_derivative(phi, dens)  # (B, nf)
    eta_vals = stack_components(eta, ("phi",))        # (B, nf)
    pairing = (ffd * eta_vals).sum(-1)
    strong = to_np(tv.integrate_values(pairing, rule=rule))
    assert np.max(np.abs(to_np(ffd))) > 1e-2
    assert np.allclose(fv, strong, atol=1e-9)


def test_field_hamilton_principle_on_dispersion() -> None:
    rule, nodes = _field_rule_nodes()
    dens = _kg_density(M)
    omega = float(np.sqrt(K**2 + M**2))  # on shell -> delta S / delta phi = 0
    phi = torch_planewave_state(K, omega, to_np(nodes))
    eta = torch_separable_state(_fx_bv, _ft_bv, to_np(nodes))
    fv = float(to_np(tv.first_variation_density(phi, dens, eta, rule=rule)))
    assert abs(fv) < 1e-8


def test_field_first_variation_cross_backend() -> None:
    rule, nodes = _field_rule_nodes()
    dens = _kg_density(M)
    tphi = torch_planewave_state(K, 1.9, to_np(nodes))
    teta = torch_separable_state(_fx_bv, _ft_bv, to_np(nodes))
    jphi = jax_planewave_state(K, 1.9, to_np(nodes))
    jeta = jax_separable_state(_fx_bv, _ft_bv, to_np(nodes))
    t = float(to_np(tv.first_variation_density(tphi, dens, teta, rule=rule)))
    j = float(to_np(jv.first_variation_density(jphi, dens, jeta, rule=rule)))
    assert abs(t - j) < 1e-12
