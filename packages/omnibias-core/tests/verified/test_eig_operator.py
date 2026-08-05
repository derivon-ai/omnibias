# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified bottom-of-spectrum lower bounds: Temple / Lehmann-Maehly-Goerisch.

Soundness oracle is high-precision ``mpmath`` symmetric eigensolves; every
certified bound is checked to enclose / under-estimate the true eigenvalue.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import pytest
from omnibias.core.verified.eig_operator import (
    certified_spectral_gap,
    count_eigenvalues_below,
    generalized_eigenvalue_enclosure,
    interval_ldlt_inertia,
    is_positive_definite,
    lehmann_maehly_lower_bounds,
    operator_comparison_bounds,
    ritz_upper_bound,
    temple_lower_bound_vector,
)

mp = pytest.importorskip("mpmath")
mp.mp.dps = 50


# --------------------------------------------------------------------------- #
# mpmath oracles (pure high-precision, no numpy/scipy).
# --------------------------------------------------------------------------- #
def _eigvals_sym(rows: Sequence[Sequence[float]]) -> list[float]:
    a = mp.matrix([[mp.mpf(repr(x)) for x in row] for row in rows])
    evals = mp.eigsy(a, eigvals_only=True)
    return sorted(float(e) for e in evals)


def _ground_eigvec(rows: Sequence[Sequence[float]]) -> list[float]:
    a = mp.matrix([[mp.mpf(repr(x)) for x in row] for row in rows])
    evals, evecs = mp.eigsy(a)
    order = sorted(range(len(evals)), key=lambda i: float(evals[i]))
    col = order[0]
    return [float(evecs[i, col]) for i in range(a.rows)]


def _gen_eigvals(a_rows: Sequence[Sequence[float]], m_rows: Sequence[Sequence[float]]) -> list[float]:
    a = mp.matrix([[mp.mpf(repr(x)) for x in row] for row in a_rows])
    m = mp.matrix([[mp.mpf(repr(x)) for x in row] for row in m_rows])
    lmat = mp.cholesky(m)
    linv = mp.inverse(lmat)
    c = linv * a * linv.T
    c = (c + c.T) / 2
    evals = mp.eigsy(c, eigvals_only=True)
    return sorted(float(e) for e in evals)


# --------------------------------------------------------------------------- #
# Float linear-algebra test helpers.
# --------------------------------------------------------------------------- #
def _transpose(a: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[a[i][j] for i in range(len(a))] for j in range(len(a[0]))]


def _matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    n, k, m = len(a), len(b), len(b[0])
    return [[math.fsum(a[i][p] * b[p][j] for p in range(k)) for j in range(m)] for i in range(n)]


def _schrodinger(n: int, potential: Callable[[float], float]) -> tuple[list[list[float]], float]:
    """Dirichlet finite-difference ``-u'' + V u`` on ``[0, 1]`` (``n`` interior nodes)."""
    h = 1.0 / (n + 1)
    inv = 1.0 / h**2
    a = [[0.0] * n for _ in range(n)]
    for i in range(n):
        a[i][i] = 2.0 * inv + potential((i + 1) * h)
        if i + 1 < n:
            a[i][i + 1] = a[i + 1][i] = -inv
    return a, h


def _laplacian_modes(n: int) -> tuple[list[float], list[list[float]]]:
    """Exact eigenvalues / eigenvectors (as columns) of the FD Laplacian ``-u''``."""
    h = 1.0 / (n + 1)
    eigs = [(2.0 / h**2) * (1.0 - math.cos(k * math.pi / (n + 1))) for k in range(1, n + 1)]
    vecs = [[math.sin(k * i * math.pi / (n + 1)) for k in range(1, n + 1)] for i in range(1, n + 1)]
    return eigs, vecs


# --------------------------------------------------------------------------- #
# interval LDL^T inertia.
# --------------------------------------------------------------------------- #
def test_inertia_matches_oracle_signs() -> None:
    mats = [
        [[2.0, 1.0], [1.0, 2.0]],  # both positive
        [[1.0, 2.0], [2.0, 1.0]],  # one negative (eigs 3, -1)
        [[-3.0, 0.5], [0.5, -2.0]],  # both negative
        [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
    ]
    for m in mats:
        inertia = interval_ldlt_inertia(m)
        assert inertia is not None
        evals = _eigvals_sym(m)
        assert inertia.negative == sum(1 for e in evals if e < 0)
        assert inertia.positive == sum(1 for e in evals if e > 0)


def test_inertia_returns_none_when_singular() -> None:
    # eigenvalue exactly 0 -> a pivot straddles 0 -> cannot certify the sign.
    assert interval_ldlt_inertia([[1.0, 1.0], [1.0, 1.0]]) is None


def test_is_positive_definite() -> None:
    assert is_positive_definite([[2.0, 1.0], [1.0, 2.0]])
    assert not is_positive_definite([[1.0, 2.0], [2.0, 1.0]])


# --------------------------------------------------------------------------- #
# Generalized eigenvalue counting + enclosure.
# --------------------------------------------------------------------------- #
def test_count_eigenvalues_below_standard() -> None:
    a = [[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]]
    ident = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    evals = _eigvals_sym(a)
    for t in (-1.0, evals[0] + 1e-6, 0.5 * (evals[0] + evals[1]), evals[2] + 1.0):
        expected = sum(1 for e in evals if e < t)
        assert count_eigenvalues_below(a, ident, t) == expected


def test_generalized_eigenvalue_enclosure_standard() -> None:
    a = [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]
    ident = [[1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    evals = _eigvals_sym(a)
    for idx in (1, 2, 3):
        enc = generalized_eigenvalue_enclosure(a, ident, idx)
        assert enc.lo <= evals[idx - 1] <= enc.hi
        assert enc.width < 1e-6


def test_generalized_eigenvalue_enclosure_pencil() -> None:
    a = [[3.0, 0.5, 0.1], [0.5, 2.0, 0.3], [0.1, 0.3, 1.0]]
    m = [[2.0, 0.2, 0.0], [0.2, 1.5, 0.1], [0.0, 0.1, 1.0]]
    assert is_positive_definite(m)
    taus = _gen_eigvals(a, m)
    for idx in (1, 2, 3):
        enc = generalized_eigenvalue_enclosure(a, m, idx)
        assert enc.lo <= taus[idx - 1] <= enc.hi


def test_generalized_eigenvalue_enclosure_rejects_indefinite_m() -> None:
    a = [[1.0, 0.0], [0.0, 1.0]]
    m = [[1.0, 2.0], [2.0, 1.0]]  # indefinite
    with pytest.raises(ValueError, match="positive definite"):
        generalized_eigenvalue_enclosure(a, m, 1)


# --------------------------------------------------------------------------- #
# Temple's inequality.
# --------------------------------------------------------------------------- #
def test_temple_recovers_lambda1_tightly() -> None:
    a = [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]
    evals = _eigvals_sym(a)
    lam1, lam2 = evals[0], evals[1]
    # Approximate ground eigenvector (smooth sine mode); rho strictly below lambda_2.
    u = [math.sin(i * math.pi / 4) for i in range(1, 4)]
    rho = 0.5 * (lam1 + lam2)
    bound = temple_lower_bound_vector(a, u, rho)
    assert bound.lo <= lam1 + 1e-12  # rigorous lower bound
    assert lam1 - bound.lo < 1e-2  # and tight


def test_temple_soundness_random() -> None:
    a = [[5.0, 1.0, 0.5], [1.0, 4.0, 0.7], [0.5, 0.7, 3.0]]
    evals = _eigvals_sym(a)
    lam1, lam2 = evals[0], evals[1]
    rho = 0.5 * (lam1 + lam2)
    ground = _ground_eigvec(a)
    checked = 0
    for seed in range(40):
        # Perturb the true ground eigenvector so the Rayleigh quotient stays near
        # (but, for small noise, below) rho.
        noise = 0.15 * math.sin(2.0 + seed)
        u = [ground[j] + noise * math.cos(seed + j) for j in range(3)]
        # Temple only valid when the Rayleigh quotient is below rho; otherwise it
        # raises.  Whenever it returns, the bound must under-estimate lambda_1.
        try:
            bound = temple_lower_bound_vector(a, u, rho)
        except ValueError:
            continue
        assert bound.lo <= lam1 + 1e-9
        checked += 1
    assert checked >= 5  # the harness actually exercised the bound


def test_temple_requires_rayleigh_below_rho() -> None:
    a = [[1.0, 0.0], [0.0, 5.0]]
    # vector biased to the large eigenvalue -> Rayleigh quotient above rho.
    with pytest.raises(ValueError, match="strictly below rho"):
        temple_lower_bound_vector(a, [0.1, 1.0], rho=2.0)


# --------------------------------------------------------------------------- #
# Lehmann-Maehly-Goerisch.
# --------------------------------------------------------------------------- #
def _gram_matrices(
    a_full: Sequence[Sequence[float]], w_cols: Sequence[Sequence[float]]
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """``A0=W^T W``, ``A1=W^T A W``, ``A2=(AW)^T (AW)`` for trial columns ``w``."""
    w = [list(col) for col in w_cols]  # w[j] is the j-th column (length n)
    n = len(a_full)
    m = len(w)
    wmat = [[w[j][i] for j in range(m)] for i in range(n)]  # n x m
    aw = _matmul(a_full, wmat)  # n x m
    a0 = _matmul(_transpose(wmat), wmat)
    a1 = _matmul(_transpose(wmat), aw)
    a2 = _matmul(_transpose(aw), aw)
    return a0, a1, a2


def test_lehmann_exact_on_full_space() -> None:
    # Full trial space (W = I): A0=I, A1=A, A2=A^2 -> Lehmann recovers eigenvalues.
    a = [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]
    evals = _eigvals_sym(a)
    cols = [[1.0 if i == j else 0.0 for i in range(3)] for j in range(3)]
    a0, a1, a2 = _gram_matrices(a, cols)
    rho = evals[2] + 1.0  # above all 3 eigenvalues -> N = 3 below rho
    cert = lehmann_maehly_lower_bounds(a0, a1, a2, rho, n_below=3)
    assert cert.m_positive_definite and cert.inertia_certified
    assert cert.negatives == 3
    by_index = {b.index: b.lower_bound for b in cert.bounds}
    for k in (1, 2, 3):
        assert by_index[k] <= evals[k - 1] + 1e-9
        assert abs(by_index[k] - evals[k - 1]) < 1e-6


def test_lehmann_subspace_schrodinger_lower_bounds() -> None:
    # 1D Schrodinger with a smooth well; 2D trial space of *non-eigen* Laplacian
    # modes.  Lehmann must give rigorous (and tight) lower bounds on lambda_1,2.
    n = 40

    def potential(x: float) -> float:
        return 30.0 * (x - 0.5) ** 2

    a, _ = _schrodinger(n, potential)
    evals = _eigvals_sym(a)
    _, modes = _laplacian_modes(n)
    trial = [modes[0], modes[1]]  # first two free-Laplacian sine modes
    a0, a1, a2 = _gram_matrices(a, trial)
    rho = 0.5 * (evals[1] + evals[2])  # strictly between lambda_2 and lambda_3
    assert sum(1 for e in evals if e <= rho) == 2  # exactly N = 2 below rho
    cert = lehmann_maehly_lower_bounds(a0, a1, a2, rho, n_below=2)
    assert cert.m_positive_definite and cert.inertia_certified
    by_index = {b.index: b.lower_bound for b in cert.bounds}
    # Rigorous: both lower bounds under-estimate the true eigenvalues.
    assert by_index[1] <= evals[0] + 1e-9
    assert by_index[2] <= evals[1] + 1e-9
    # Tight: a 2-mode subspace already resolves the bottom pair to a few percent.
    assert (evals[1] - by_index[2]) / evals[1] < 0.05


def test_lehmann_reports_failure_without_crashing() -> None:
    # rho far below the spectrum -> no negative pencil eigenvalues -> empty bounds.
    a = [[2.0, 0.0], [0.0, 3.0]]
    cols = [[1.0, 0.0], [0.0, 1.0]]
    a0, a1, a2 = _gram_matrices(a, cols)
    cert = lehmann_maehly_lower_bounds(a0, a1, a2, rho=-10.0, n_below=0)
    assert cert.negatives == 0
    assert cert.bounds == ()


# --------------------------------------------------------------------------- #
# Operator comparison (Weyl) + certified gap.
# --------------------------------------------------------------------------- #
def test_operator_comparison_brackets_contain_truth() -> None:
    n = 30

    def potential(x: float) -> float:
        return 12.0 * math.sin(math.pi * x)  # 0 <= V <= 12

    a, _ = _schrodinger(n, potential)
    base, _ = _laplacian_modes(n)
    evals = _eigvals_sym(a)
    brackets = operator_comparison_bounds(base, perturbation_lower=0.0, perturbation_upper=12.0)
    for k in range(n):
        assert brackets[k].lo <= evals[k] <= brackets[k].hi


def test_certified_spectral_gap_schrodinger() -> None:
    n = 40

    def potential(x: float) -> float:
        return 25.0 * (x - 0.5) ** 2

    a, _ = _schrodinger(n, potential)
    evals = _eigvals_sym(a)
    _, modes = _laplacian_modes(n)
    trial = [modes[0], modes[1]]
    a0, a1, a2 = _gram_matrices(a, trial)
    rho = 0.5 * (evals[1] + evals[2])
    # Rigorous upper bound on lambda_1 from the ground-mode Rayleigh quotient.
    lam1_up = ritz_upper_bound(a, modes[0]).hi
    assert lam1_up >= evals[0] - 1e-9
    cert = certified_spectral_gap(a0, a1, a2, rho, lam1_up)
    assert cert.certified
    true_gap = evals[1] - evals[0]
    assert 0.0 < cert.gap_lower <= true_gap + 1e-9
    assert cert.gap_lower > 0.5 * true_gap  # meaningfully tight


# --------------------------------------------------------------------------- #
# Property-based soundness (hypothesis).
# --------------------------------------------------------------------------- #
def test_property_inertia_matches_oracle() -> None:
    hypothesis = pytest.importorskip("hypothesis")
    st = pytest.importorskip("hypothesis.strategies")

    @hypothesis.given(
        entries=st.lists(
            st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=3,
        ),
        diag=st.lists(
            st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=2,
        ),
    )
    @hypothesis.settings(max_examples=200, deadline=None)
    def check(entries: list[float], diag: list[float]) -> None:
        a = [[diag[0], entries[0]], [entries[0], diag[1]]]
        inertia = interval_ldlt_inertia(a)
        if inertia is None:
            return  # near-singular: sign not certifiable, which is the honest answer
        evals = _eigvals_sym(a)
        if any(abs(e) < 1e-6 for e in evals):
            return
        assert inertia.negative == sum(1 for e in evals if e < 0)

    check()
