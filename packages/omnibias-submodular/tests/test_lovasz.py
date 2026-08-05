# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Lovasz extension (convex closure) + exact P-class submodular minimization."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.submodular import (
    Coverage,
    GraphCut,
    lovasz_extension,
    min_norm_point,
    submodular_minimize,
)
from omnibias.submodular.functions import SubmodularFunction, is_monotone_submodular


class _CutPlusModular(SubmodularFunction):
    """``f(S) = cut_W(S) + a . 1_S`` -- submodular (cut) + modular, a nontrivial minimizer.

    A cut is submodular and a modular term is both super- and submodular, so the sum is
    submodular; mixed-sign ``a`` pushes the minimizer off the empty set. Only ``value`` is
    needed by the minimizer / extension, so the closed-form ``multilinear`` is left unbuilt.
    """

    def __init__(self, weights: np.ndarray, linear: np.ndarray) -> None:
        w = 0.5 * (np.asarray(weights, dtype=float) + np.asarray(weights, dtype=float).T)
        np.fill_diagonal(w, 0.0)
        self._w = w
        self._a = np.asarray(linear, dtype=float).reshape(-1)

    @property
    def n(self) -> int:
        return int(self._w.shape[0])

    def value(self, x: object) -> float | np.ndarray:
        xv = np.asarray(x, dtype=float)
        if xv.ndim == 1:
            cut = float(xv @ self._w @ (1.0 - xv))
            return cut + float(self._a @ xv)
        cut_b = np.sum((xv @ self._w) * (1.0 - xv), axis=1)
        return np.asarray(cut_b + xv @ self._a, dtype=float)

    def multilinear(self, p: object) -> float | np.ndarray:  # pragma: no cover - unused
        raise NotImplementedError("greedy/minimization-path test function")

    def to_polynomial(self):  # type: ignore[no-untyped-def]  # pragma: no cover - unused
        raise NotImplementedError("greedy/minimization-path test function")


def _coverage(seed: int, n: int = 7, m: int = 9) -> Coverage:
    rng = np.random.default_rng(seed)
    return Coverage((rng.random((m, n)) < 0.4).astype(float), rng.random(m) + 0.2)


def _graphcut(seed: int, n: int = 7) -> GraphCut:
    rng = np.random.default_rng(seed)
    w = rng.random((n, n))
    return GraphCut(w + w.T)


def _cut_plus_modular(seed: int, n: int = 8) -> _CutPlusModular:
    rng = np.random.default_rng(seed)
    w = rng.random((n, n))
    a = rng.normal(size=n) * float(n) * 0.25  # mixed sign, comparable scale to the cut
    return _CutPlusModular(w, a)


def _brute_min(function: SubmodularFunction) -> tuple[float, tuple[int, ...]]:
    n = function.n
    best_val = np.inf
    best_set: tuple[int, ...] = ()
    for mask in range(1 << n):
        x = np.array([(mask >> i) & 1 for i in range(n)], dtype=float)
        v = float(function.value(x))
        if v < best_val:
            best_val = v
            best_set = tuple(int(t) for t in x)
    return best_val, best_set


def _threshold_integral(function: SubmodularFunction, p: np.ndarray) -> float:
    r"""Independent Lovasz value via the exact threshold integral ``int_0^1 f({i:p_i>=t}) dt``.

    Weights *set values* by the gaps between sorted coordinates -- algebraically equal to the
    telescoped-marginal formula in :func:`lovasz_extension`, but a distinct code path.
    """
    n = function.n
    order = np.argsort(-p, kind="stable")  # descending
    ps = p[order]
    x = np.zeros(n, dtype=float)
    total = float(function.value(x)) * (1.0 - float(ps[0]))  # theta in (p_(1), 1]
    for k in range(n):
        x[order[k]] = 1.0
        width = float(ps[k] - (ps[k + 1] if k + 1 < n else 0.0))
        total += float(function.value(x)) * width
    return total


# --- Lovasz extension -------------------------------------------------------------------


@pytest.mark.parametrize("maker", [_coverage, _graphcut, _cut_plus_modular])
def test_lovasz_agrees_with_f_on_vertices(maker) -> None:  # type: ignore[no-untyped-def]
    f = maker(0)
    n = f.n
    rng = np.random.default_rng(1)
    for _ in range(24):
        x = rng.integers(0, 2, size=n).astype(float)
        assert lovasz_extension(f, x) == pytest.approx(float(f.value(x)), abs=1e-9)


@pytest.mark.parametrize("maker", [_coverage, _graphcut, _cut_plus_modular])
def test_lovasz_matches_threshold_integral(maker) -> None:  # type: ignore[no-untyped-def]
    f = maker(2)
    n = f.n
    rng = np.random.default_rng(3)
    for _ in range(24):
        p = rng.random(n)
        assert lovasz_extension(f, p) == pytest.approx(_threshold_integral(f, p), abs=1e-9)


@pytest.mark.parametrize("maker", [_coverage, _graphcut, _cut_plus_modular])
def test_lovasz_is_convex_for_submodular(maker) -> None:  # type: ignore[no-untyped-def]
    f = maker(4)
    n = f.n
    rng = np.random.default_rng(5)
    for _ in range(40):
        p = rng.random(n)
        q = rng.random(n)
        mid = 0.5 * (p + q)
        lhs = lovasz_extension(f, mid)
        rhs = 0.5 * (lovasz_extension(f, p) + lovasz_extension(f, q))
        assert lhs <= rhs + 1e-9


def test_lovasz_extension_wrong_length_raises() -> None:
    f = _coverage(0)
    with pytest.raises(ValueError):
        lovasz_extension(f, np.zeros(f.n + 1))


# --- exact minimization (P-class) -------------------------------------------------------


@pytest.mark.parametrize("seed", range(8))
def test_submodular_minimize_matches_brute_force(seed: int) -> None:
    f = _cut_plus_modular(seed, n=8)
    result = submodular_minimize(f)
    brute_val, _ = _brute_min(f)
    assert result.value == pytest.approx(brute_val, abs=1e-7)
    # the returned set really attains the reported value
    x = np.array(result.selection, dtype=float)
    assert float(f.value(x)) == pytest.approx(result.value, abs=1e-9)


@pytest.mark.parametrize("seed", range(6))
def test_min_norm_point_fujishige_identity(seed: int) -> None:
    # Fujishige: min_S f(S) = f(empty) + sum_i min(x*_i, 0) for the base-polytope min-norm x*.
    f = _cut_plus_modular(seed, n=8)
    x_star = min_norm_point(f)
    f_empty = float(f.value(np.zeros(f.n)))
    recovered = f_empty + float(np.sum(np.minimum(x_star, 0.0)))
    brute_val, _ = _brute_min(f)
    assert recovered == pytest.approx(brute_val, abs=1e-6)
    # x* lives in the base polytope: x*(V) = f(V) - f(empty)
    f_full = float(f.value(np.ones(f.n)))
    assert float(np.sum(x_star)) == pytest.approx(f_full - f_empty, abs=1e-7)


def test_minimizer_is_nontrivial_when_the_min_is_interior() -> None:
    # Deterministic instance whose optimum is provably interior: three elements are strongly
    # pulled in (a_i = -5) and three strongly pushed out (a_i = +5) with only a weak cut
    # coupling, so the minimizer is exactly {0, 1, 2} -- neither empty nor full.
    n = 6
    w = np.full((n, n), 0.1)
    np.fill_diagonal(w, 0.0)
    a = np.array([-5.0, -5.0, -5.0, 5.0, 5.0, 5.0])
    f = _CutPlusModular(w, a)
    result = submodular_minimize(f)
    assert result.support == (0, 1, 2)
    assert 0 < len(result.support) < f.n
    brute_val, _ = _brute_min(f)
    assert result.value == pytest.approx(brute_val, abs=1e-9)
    assert result.value < float(f.value(np.zeros(f.n))) - 1e-9


def test_cut_plus_modular_is_submodular() -> None:
    f = _cut_plus_modular(0)
    _, submodular = is_monotone_submodular(f, samples=128)
    assert submodular
