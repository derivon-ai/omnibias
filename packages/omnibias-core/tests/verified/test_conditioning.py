# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified conditioning -- the rigorous register of the ``eps -> 0`` collapse.

Soundness oracle is high-precision ``mpmath`` symmetric eigensolves and linear
solves (no numpy, matching the core test convention). Every certified enclosure
is checked to contain the true value on a **deterministic grid** *and* a
**random sample** of SPD / rank-deficient matrices; the damping selection and the
regularization-error bound are checked sound against the oracle; the sealed
certificate round-trips and carries the Lean-dischargeable ``lambda_min > 0``
pivot data.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import pytest
from omnibias.core.proof.certificate import (
    decode_interval,
    schema_errors_v1,
    verify_certificate_digest,
)
from omnibias.core.verified.conditioning import (
    certified_condition_number,
    certified_damping,
    certified_max_eigenvalue,
    certified_min_eigenvalue,
    certified_regularization_error,
    conditioning_certificate,
    positive_definite_pivots_certificate,
)

mp = pytest.importorskip("mpmath")
mp.mp.dps = 50

Matrix = Sequence[Sequence[float]]


# --------------------------------------------------------------------------- #
# mpmath oracles (high precision, dependency-free).
# --------------------------------------------------------------------------- #
def _eigvals_sym(rows: Matrix) -> list[float]:
    a = mp.matrix([[mp.mpf(repr(x)) for x in row] for row in rows])
    return sorted(float(e) for e in mp.eigsy(a, eigvals_only=True))


def _true_cond(rows: Matrix) -> float:
    ev = _eigvals_sym(rows)
    return ev[-1] / ev[0]


def _solve(rows: Matrix, b: Sequence[float]) -> list[float]:
    a = mp.matrix([[mp.mpf(repr(x)) for x in row] for row in rows])
    bb = mp.matrix([mp.mpf(repr(x)) for x in b])
    return [float(xi) for xi in mp.lu_solve(a, bb)]


def _pinv_solve(rows: Matrix, b: Sequence[float], tol: float = 1e-9) -> tuple[list[float], float]:
    """Min-norm ``A^+ b`` via a symmetric eigendecomposition + the smallest nonzero eig."""
    a = mp.matrix([[mp.mpf(repr(x)) for x in row] for row in rows])
    evals, evecs = mp.eigsy(a)
    bb = mp.matrix([mp.mpf(repr(x)) for x in b])
    n = a.rows
    x = mp.zeros(n, 1)
    smallest_nonzero = math.inf
    for i in range(n):
        lam = evals[i]
        if abs(lam) > tol:
            vi = evecs[:, i]
            x += ((vi.T * bb)[0] / lam) * vi
            smallest_nonzero = min(smallest_nonzero, float(abs(lam)))
    return [float(xi) for xi in x], smallest_nonzero


def _add_diag(rows: Matrix, eps: float) -> list[list[float]]:
    n = len(rows)
    return [[rows[i][j] + (eps if i == j else 0.0) for j in range(n)] for i in range(n)]


def _l2(v: Sequence[float]) -> float:
    return math.sqrt(math.fsum(x * x for x in v))


# --------------------------------------------------------------------------- #
# Pure-python random SPD / rank-deficient builders.
# --------------------------------------------------------------------------- #
def _gram(m_rows: Sequence[Sequence[float]]) -> list[list[float]]:
    """``M^T M`` (SPD, rank ``= rank(M)``) from an ``m x n`` matrix given by rows."""
    m, n = len(m_rows), len(m_rows[0])
    return [[math.fsum(m_rows[k][i] * m_rows[k][j] for k in range(m)) for j in range(n)] for i in range(n)]


def _rand_spd(rng: random.Random, n: int) -> list[list[float]]:
    rows = [[rng.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(n + 2)]
    a = _gram(rows)
    return _add_diag(a, 0.1)  # keep it strictly PD


def _rand_low_rank(rng: random.Random, n: int, r: int) -> list[list[float]]:
    rows = [[rng.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(r)]  # rank <= r < n
    return _gram(rows)


# Deterministic grid of SPD matrices with genuinely different spectra / sizes.
def _tridiag_laplacian(n: int) -> list[list[float]]:
    a = [[0.0] * n for _ in range(n)]
    for i in range(n):
        a[i][i] = 2.0
        if i + 1 < n:
            a[i][i + 1] = a[i + 1][i] = -1.0
    return a


_GRID: list[list[list[float]]] = [
    [[3.0]],
    [[2.0, 0.0], [0.0, 5.0]],
    [[2.0, 1.0], [1.0, 2.0]],
    [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
    _tridiag_laplacian(5),
    [[10.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.01]],  # ill-conditioned
]


# --------------------------------------------------------------------------- #
# Extreme-eigenvalue enclosures: grid + random.
# --------------------------------------------------------------------------- #
def test_extreme_eigenvalues_enclose_grid() -> None:
    for a in _GRID:
        ev = _eigvals_sym(a)
        lo = certified_min_eigenvalue(a)
        hi = certified_max_eigenvalue(a)
        assert lo.lo <= ev[0] <= lo.hi
        assert hi.lo <= ev[-1] <= hi.hi
        # tight relative to the spectrum
        assert lo.width <= 1e-6 * max(1.0, abs(ev[0]))
        assert hi.width <= 1e-6 * max(1.0, abs(ev[-1]))


def test_extreme_eigenvalues_enclose_random() -> None:
    rng = random.Random(20260729)
    for _ in range(20):
        n = rng.randint(2, 5)
        a = _rand_spd(rng, n)
        ev = _eigvals_sym(a)
        assert certified_min_eigenvalue(a).lo <= ev[0]
        assert certified_min_eigenvalue(a).hi >= ev[0]
        assert certified_max_eigenvalue(a).lo <= ev[-1] <= certified_max_eigenvalue(a).hi


def test_condition_number_encloses_grid_and_random() -> None:
    for a in _GRID:
        k = certified_condition_number(a)
        true_k = _true_cond(a)
        assert k.lo <= true_k <= k.hi
    rng = random.Random(11)
    for _ in range(15):
        a = _rand_spd(rng, rng.randint(2, 5))
        k = certified_condition_number(a)
        assert k.lo <= _true_cond(a) <= k.hi


def test_condition_number_is_infinite_when_rank_deficient() -> None:
    rng = random.Random(7)
    for _ in range(10):
        n = rng.randint(3, 5)
        a = _rand_low_rank(rng, n, n - 1)
        k = certified_condition_number(a)
        assert k.hi == math.inf
        assert k.lo >= 1.0


# --------------------------------------------------------------------------- #
# Certified damping: the achieved condition number is provably below target.
# --------------------------------------------------------------------------- #
def test_certified_damping_achieves_target_random() -> None:
    rng = random.Random(2718)
    nontrivial = 0
    for _ in range(20):
        n = rng.randint(2, 5)
        a = _rand_spd(rng, n)
        target = rng.choice([2.0, 5.0, 20.0])
        eps = certified_damping(a, target_condition=target)
        assert eps >= 0.0 and math.isfinite(eps)
        assert _true_cond(_add_diag(a, eps)) <= target + 1e-9
        if eps > 0.0:
            nontrivial += 1
    assert nontrivial >= 3  # the eps>0 branch is genuinely exercised


def test_certified_damping_zero_when_already_conditioned() -> None:
    a = [[2.0, 0.0], [0.0, 3.0]]  # kappa = 1.5
    assert certified_damping(a, target_condition=10.0) == 0.0


def test_certified_damping_rank_deficient_matrix() -> None:
    rng = random.Random(99)
    for _ in range(8):
        n = rng.randint(3, 5)
        a = _rand_low_rank(rng, n, n - 1)  # singular
        eps = certified_damping(a, target_condition=10.0)
        assert eps > 0.0 and math.isfinite(eps)
        assert _true_cond(_add_diag(a, eps)) <= 10.0 + 1e-9


def test_certified_damping_rejects_target_le_one() -> None:
    with pytest.raises(ValueError, match="target_condition"):
        certified_damping([[1.0, 0.0], [0.0, 1.0]], target_condition=1.0)


# --------------------------------------------------------------------------- #
# Certified regularization error: sound upper bound on ||x_eps - A^+ b||.
# --------------------------------------------------------------------------- #
def test_regularization_error_sound_spd_random() -> None:
    rng = random.Random(31415)
    for _ in range(20):
        n = rng.randint(2, 5)
        a = _rand_spd(rng, n)
        b = [rng.uniform(-2.0, 2.0) for _ in range(n)]
        for eps in (1e-1, 1e-2, 1e-4):
            bound = certified_regularization_error(a, b, eps)
            xe = _solve(_add_diag(a, eps), b)
            x0 = _solve(a, b)  # full rank: min-norm == exact solve
            true_err = _l2([xe[i] - x0[i] for i in range(n)])
            assert bound.lo == 0.0
            assert bound.hi >= true_err - 1e-15


def test_regularization_error_requires_positive_eigenvalue() -> None:
    a = _rand_low_rank(random.Random(5), 4, 3)  # singular
    with pytest.raises(ValueError, match="smallest"):
        certified_regularization_error(a, [1.0, 1.0, 1.0, 1.0], 1e-2)


def test_regularization_error_range_consistent_rank_deficient() -> None:
    """With a range-consistent RHS and the smallest nonzero eig, the bound is sound."""
    rng = random.Random(1234)
    for _ in range(10):
        n = rng.randint(3, 5)
        a = _rand_low_rank(rng, n, n - 1)  # rank n-1, singular
        y = [rng.uniform(-1.0, 1.0) for _ in range(n)]
        b = [math.fsum(a[i][j] * y[j] for j in range(n)) for i in range(n)]  # b in range(A)
        eps = 1e-3
        x0, smallest_nonzero = _pinv_solve(a, b)
        xe = _solve(_add_diag(a, eps), b)
        true_err = _l2([xe[i] - x0[i] for i in range(n)])
        # a valid *lower* bound on the smallest nonzero eigenvalue (range scope)
        min_eig = smallest_nonzero * (1.0 - 1e-6)
        bound = certified_regularization_error(a, b, eps, min_eig=min_eig)
        assert bound.hi >= true_err - 1e-15


# --------------------------------------------------------------------------- #
# Sealed certificate + Lean-dischargeable lambda_min > 0 data.
# --------------------------------------------------------------------------- #
def test_conditioning_certificate_seals_and_verifies() -> None:
    a = _GRID[3]
    cert = conditioning_certificate(a, target_condition=10.0, eps=1e-3)
    assert schema_errors_v1(cert) == []
    assert verify_certificate_digest(cert)
    payload = cert["payload"]
    assert payload["type"] == "conditioning"
    assert payload["positive_definite"] is True
    # kappa enclosure round-trips and contains the truth
    k = decode_interval(payload["condition_number"])
    assert k.lo <= _true_cond(a) <= k.hi
    # the recorded damping actually achieves the target (oracle-checked)
    assert _true_cond(_add_diag(a, payload["certified_damping"])) <= 10.0 + 1e-9
    # pivots present and all strictly positive: the lambda_min > 0 obligation
    pivots = [decode_interval(p) for p in payload["pivots"]]
    assert all(p.lo > 0.0 for p in pivots)


def test_conditioning_certificate_rank_deficient_is_honest() -> None:
    a = _rand_low_rank(random.Random(3), 4, 3)
    cert = conditioning_certificate(a)
    assert schema_errors_v1(cert) == []
    assert cert["payload"]["positive_definite"] is False
    assert decode_interval(cert["payload"]["condition_number"]).hi == math.inf


def test_positive_definite_pivots_certificate() -> None:
    pd = positive_definite_pivots_certificate(_GRID[2])
    assert schema_errors_v1(pd) == []
    assert verify_certificate_digest(pd)
    with pytest.raises(ValueError, match="positive definite"):
        positive_definite_pivots_certificate(_rand_low_rank(random.Random(1), 4, 3))


def test_certificate_is_tamper_evident() -> None:
    cert = conditioning_certificate(_GRID[4])
    cert["payload"]["positive_definite"] = False  # flip a claim
    assert not verify_certificate_digest(cert)
