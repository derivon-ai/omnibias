# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Local nonintegrability: the covariant-derivative commutator = curvature action.

Validates the Ricci identity ``[D_mu, D_nu] phi = g f^{abc} F_{mu nu}^b phi^c`` in
the closed-form register: it is the Jacobi identity, so the defect vanishes to
machine precision; the ``FieldState`` (closed-form sigma-tower) path matches the
analytic arrays; and torch <-> jax agree bit-identically.
"""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.geometry.gauge._core.connection import (
    connection_component_names,
    gauge_connection_spec,
)
from omnibias.geometry.gauge._core.lie_algebra import su
from test_fieldstate_path import AnalyticConnectionField

DIM = 4


def _sym_ddphi(rng: np.random.Generator, B: int, d: int, n: int) -> np.ndarray:
    """Random second partials symmetric in ``(mu, nu)`` (mixed partials commute)."""
    raw = rng.normal(size=(B, d, d, n))
    return 0.5 * (raw + raw.transpose(0, 2, 1, 3))


@pytest.mark.parametrize("group_n", [2, 3])
def test_ricci_identity_defect_vanishes(backend, group_n: int) -> None:
    alg = su(group_n)
    n = alg.dim
    B = 6
    rng = np.random.default_rng(40 + group_n)
    phi = rng.normal(size=(B, n))
    dphi = rng.normal(size=(B, DIM, n))
    ddphi = _sym_ddphi(rng, B, DIM, n)
    a = rng.normal(size=(B, DIM, n))
    da = rng.normal(size=(B, DIM, DIM, n))
    g = 0.73

    defect = backend.ops.curvature_commutator_defect_from_arrays(
        backend.asarray(phi), backend.asarray(dphi), backend.asarray(ddphi),
        backend.asarray(a), backend.asarray(da), algebra=alg, coupling=g,
    )
    commutator = backend.ops.covariant_derivative_commutator_from_arrays(
        backend.asarray(phi), backend.asarray(dphi), backend.asarray(ddphi),
        backend.asarray(a), backend.asarray(da), algebra=alg, coupling=g,
    )
    # The identity holds exactly (Jacobi), yet the commutator itself is nonzero.
    assert np.abs(backend.tonumpy(defect)).max() < 1e-10
    assert np.abs(backend.tonumpy(commutator)).max() > 1e-2


def test_commutator_is_antisymmetric(backend) -> None:
    alg = su(2)
    n = alg.dim
    B = 4
    rng = np.random.default_rng(2)
    phi = rng.normal(size=(B, n))
    dphi = rng.normal(size=(B, DIM, n))
    ddphi = _sym_ddphi(rng, B, DIM, n)
    a = rng.normal(size=(B, DIM, n))
    da = rng.normal(size=(B, DIM, DIM, n))
    comm = backend.tonumpy(
        backend.ops.covariant_derivative_commutator_from_arrays(
            backend.asarray(phi), backend.asarray(dphi), backend.asarray(ddphi),
            backend.asarray(a), backend.asarray(da), algebra=alg, coupling=0.5,
        )
    )
    np.testing.assert_allclose(comm, -comm.transpose(0, 2, 1, 3), atol=1e-12)


def _analytic_comp(
    field: AnalyticConnectionField, name: str, coords: np.ndarray, deriv_axes: tuple[int, ...] = ()
) -> np.ndarray:
    """Analytic value / partial of a separable component (numpy reference)."""
    orders = {ax: deriv_axes.count(ax) for ax in set(deriv_axes)}
    acc = np.ones(coords.shape[0])
    for d, ax in enumerate(field._axes[name]):
        o = orders.get(d, 0)
        acc = acc * (ax.deriv(np, coords[:, d], o) if o > 0 else ax.value(np, coords[:, d]))
    return acc


def test_fieldstate_path_matches_analytic_arrays(backend) -> None:
    alg = su(2)
    coupling = 0.8
    conn = gauge_connection_spec(alg, coupling=coupling, spacetime_dim=DIM)
    conn_names = connection_component_names(conn)
    n = alg.dim
    phi_names = tuple(f"phi_{a}" for a in range(n))
    if backend.name == "torch":
        from omnibias.fields.torch import _ops_dispatch
    else:
        from omnibias.fields.jax import _ops_dispatch
    field = AnalyticConnectionField(tuple(conn_names) + phi_names, _ops_dispatch)

    rng = np.random.default_rng(11)
    coords = rng.uniform(-1.0, 1.0, size=(12, DIM))
    state = field(backend.asarray(coords))

    a = np.stack(
        [np.stack([_analytic_comp(field, conn_names[mu * n + b], coords) for b in range(n)], -1)
         for mu in range(DIM)],
        axis=1,
    )
    da = np.stack(
        [np.stack([np.stack(
            [_analytic_comp(field, conn_names[nu * n + b], coords, (rho,)) for b in range(n)], -1)
            for nu in range(DIM)], axis=1)
         for rho in range(DIM)],
        axis=1,
    )
    phi = np.stack([_analytic_comp(field, phi_names[b], coords) for b in range(n)], -1)
    dphi = np.stack(
        [np.stack([_analytic_comp(field, phi_names[b], coords, (mu,)) for b in range(n)], -1)
         for mu in range(DIM)],
        axis=1,
    )
    ddphi = np.stack(
        [np.stack([np.stack(
            [_analytic_comp(field, phi_names[b], coords, (mu, nu)) for b in range(n)], -1)
            for nu in range(DIM)], axis=1)
         for mu in range(DIM)],
        axis=1,
    )

    comm_arrays = backend.ops.covariant_derivative_commutator_from_arrays(
        backend.asarray(phi), backend.asarray(dphi), backend.asarray(ddphi),
        backend.asarray(a), backend.asarray(da), algebra=alg, coupling=coupling,
    )
    comm_state = backend.ops.covariant_derivative_commutator(state, conn, phi_names)
    np.testing.assert_allclose(
        backend.tonumpy(comm_state), backend.tonumpy(comm_arrays), rtol=1e-8, atol=1e-9
    )


def test_commutator_cross_backend() -> None:
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import omnibias.geometry.gauge.jax.ops as jops
    import omnibias.geometry.gauge.torch.ops as tops

    torch.set_default_dtype(torch.float64)
    alg = su(3)
    n = alg.dim
    B = 6
    rng = np.random.default_rng(7)
    phi = rng.normal(size=(B, n))
    dphi = rng.normal(size=(B, DIM, n))
    ddphi = _sym_ddphi(rng, B, DIM, n)
    a = rng.normal(size=(B, DIM, n))
    da = rng.normal(size=(B, DIM, DIM, n))
    g = 0.6

    t = tops.covariant_derivative_commutator_from_arrays(
        torch.as_tensor(phi), torch.as_tensor(dphi), torch.as_tensor(ddphi),
        torch.as_tensor(a), torch.as_tensor(da), algebra=alg, coupling=g,
    )
    j = jops.covariant_derivative_commutator_from_arrays(
        jnp.asarray(phi), jnp.asarray(dphi), jnp.asarray(ddphi),
        jnp.asarray(a), jnp.asarray(da), algebra=alg, coupling=g,
    )
    np.testing.assert_allclose(t.detach().numpy(), np.asarray(j), rtol=1e-9, atol=1e-11)
