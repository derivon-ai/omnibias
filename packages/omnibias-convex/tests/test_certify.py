# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Verified optimality certificate: the interval enclosure must contain f*."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.convex import Certificate, CertificationError, certify_qp_optimum

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
from omnibias.convex.jax import solve_qp  # noqa: E402


def _arr(x: object) -> jnp.ndarray:
    return jnp.asarray(np.asarray(x, dtype=np.float64))


def test_certificate_encloses_projection_optimum() -> None:
    # min 1/2||x - p||^2 s.t. -x <= 0  =>  x* = relu(p), f* = 1/2 ||p_-||^2.
    p = np.array([1.5, -2.0, 0.7, -0.1])
    n = 4
    Q = np.eye(n)
    c = -p
    A = -np.eye(n)
    b = np.zeros(n)
    sol = solve_qp(_arr(Q), _arr(c), _arr(A), _arr(b))

    cert = certify_qp_optimum(Q, c, A, b, np.asarray(sol.x), np.asarray(sol.dual))
    x_star = np.maximum(p, 0.0)
    f_star = 0.5 * float(np.sum((x_star - p) ** 2)) - 0.5 * float(np.sum(p**2))
    # f0 = 1/2 x^T x + c^T x ; with c = -p this equals 1/2||x-p||^2 - 1/2||p||^2.
    assert isinstance(cert, Certificate)
    assert cert.primal_feasible
    assert cert.enclosure.lo <= f_star <= cert.enclosure.hi
    assert cert.gap >= 0.0


def test_certificate_encloses_random_qp_optimum() -> None:
    rng = np.random.default_rng(7)
    n = 3
    M = rng.standard_normal((n, n))
    Q = M @ M.T + 2.0 * n * np.eye(n)
    c = rng.standard_normal(n)
    A = np.vstack([rng.standard_normal((2, n)), np.eye(n), -np.eye(n)])
    b = np.concatenate([rng.uniform(0.6, 1.4, size=2), 2.5 * np.ones(2 * n)])
    sol = solve_qp(_arr(Q), _arr(c), _arr(A), _arr(b))

    x = np.asarray(sol.x)
    f_at_x = 0.5 * float(x @ Q @ x) + float(c @ x)
    cert = certify_qp_optimum(Q, c, A, b, x, np.asarray(sol.dual))
    assert cert.enclosure.lo <= f_at_x <= cert.enclosure.hi + 1e-9
    assert cert.enclosure.width >= 0.0


def test_certificate_gap_shrinks_with_tighter_solve() -> None:
    from omnibias.convex import BarrierOptions

    rng = np.random.default_rng(8)
    n = 3
    M = rng.standard_normal((n, n))
    Q = M @ M.T + 2.0 * n * np.eye(n)
    c = rng.standard_normal(n)
    A = np.vstack([np.eye(n), -np.eye(n)])
    b = 2.0 * np.ones(2 * n)

    loose = solve_qp(_arr(Q), _arr(c), _arr(A), _arr(b), options=BarrierOptions(tol=1e-2))
    tight = solve_qp(_arr(Q), _arr(c), _arr(A), _arr(b), options=BarrierOptions(tol=1e-10))
    cert_loose = certify_qp_optimum(Q, c, A, b, np.asarray(loose.x), np.asarray(loose.dual))
    cert_tight = certify_qp_optimum(Q, c, A, b, np.asarray(tight.x), np.asarray(tight.dual))
    assert cert_tight.gap <= cert_loose.gap


def test_certify_raises_when_x_infeasible() -> None:
    n = 2
    Q = np.eye(n)
    c = np.zeros(n)
    A = np.eye(n)
    b = np.ones(n)
    x_bad = np.array([2.0, 2.0])  # violates x <= 1
    with pytest.raises(CertificationError):
        certify_qp_optimum(Q, c, A, b, x_bad, np.zeros(n))
