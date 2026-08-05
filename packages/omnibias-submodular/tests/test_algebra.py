# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Submodular function algebra: Sum / Scaled (closed-form composed) + Saturated (greedy-path)."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.submodular import (
    Coverage,
    FacilityLocation,
    LogDeterminant,
    Saturated,
    Scaled,
    Sum,
    UniformMatroid,
    brute_force_max,
    certify_submodular_gap,
    lazy_greedy,
)
from omnibias.submodular.functions import is_monotone_submodular
from omnibias.submodular.problem import SubmodularProblem


def _coverage(seed: int, n: int = 6, m: int = 8) -> Coverage:
    r = np.random.default_rng(seed)
    return Coverage((r.random((m, n)) < 0.4).astype(float), r.random(m) + 0.2)


def _facility(seed: int, n: int = 6, m: int = 5) -> FacilityLocation:
    r = np.random.default_rng(seed)
    return FacilityLocation(r.random((m, n)), r.random(m) + 0.2)


def _logdet(seed: int, n: int = 6) -> LogDeterminant:
    r = np.random.default_rng(seed)
    a = r.random((n, n))
    return LogDeterminant(a @ a.T + 0.5 * np.eye(n))


# --- Sum --------------------------------------------------------------------------------


def test_sum_is_monotone_submodular() -> None:
    f = Sum((_coverage(0), _facility(1)))
    monotone, submodular = is_monotone_submodular(f, samples=128)
    assert monotone and submodular


def test_sum_multilinear_composes_closed_form() -> None:
    parts = (_coverage(0), _facility(1), _coverage(2))
    f = Sum(parts)
    rng = np.random.default_rng(3)
    for _ in range(24):
        p = rng.random(f.n)
        expected = sum(float(part.multilinear(p)) for part in parts)
        assert float(f.multilinear(p)) == pytest.approx(expected, abs=1e-12)
    # agrees with value on the cube
    for _ in range(16):
        x = rng.integers(0, 2, size=f.n).astype(float)
        assert float(f.value(x)) == pytest.approx(float(f.multilinear(x)), abs=1e-12)


def test_sum_multilinear_batch_matches_pointwise() -> None:
    f = Sum((_coverage(0), _coverage(1)))
    rng = np.random.default_rng(4)
    batch = rng.random((5, f.n))
    out = np.asarray(f.multilinear(batch), dtype=float)
    assert out.shape == (5,)
    for i, row in enumerate(batch):
        assert out[i] == pytest.approx(float(f.multilinear(row)), abs=1e-12)


def test_sum_with_greedy_path_part_values_but_no_closed_form_extension() -> None:
    parts = (_coverage(0), _logdet(1))
    f = Sum(parts)
    # value still composes on the cube even though a part is greedy-path
    x = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    expected = sum(float(part.value(x)) for part in parts)
    assert float(f.value(x)) == pytest.approx(expected, abs=1e-12)
    with pytest.raises(NotImplementedError):
        f.multilinear(np.full(f.n, 0.5))


def test_sum_rejects_mismatched_or_empty_parts() -> None:
    with pytest.raises(ValueError):
        Sum(())
    with pytest.raises(ValueError):
        Sum((_coverage(0, n=6), _coverage(0, n=5)))


# --- Scaled -----------------------------------------------------------------------------


def test_scaled_scales_value_and_extension() -> None:
    base = _coverage(0)
    f = Scaled(base, 2.5)
    rng = np.random.default_rng(5)
    for _ in range(24):
        p = rng.random(f.n)
        assert float(f.multilinear(p)) == pytest.approx(2.5 * float(base.multilinear(p)), abs=1e-12)
        x = rng.integers(0, 2, size=f.n).astype(float)
        assert float(f.value(x)) == pytest.approx(2.5 * float(base.value(x)), abs=1e-12)
    monotone, submodular = is_monotone_submodular(f, samples=64)
    assert monotone and submodular


def test_scaled_rejects_negative_scale() -> None:
    with pytest.raises(ValueError):
        Scaled(_coverage(0), -1.0)


# --- Saturated --------------------------------------------------------------------------


def test_saturated_is_monotone_submodular_and_capped() -> None:
    base = _coverage(0)
    cap = 0.5 * float(base.value(np.ones(base.n)))
    f = Saturated(base, cap)
    monotone, submodular = is_monotone_submodular(f, samples=128)
    assert monotone and submodular
    rng = np.random.default_rng(6)
    for _ in range(16):
        x = rng.integers(0, 2, size=f.n).astype(float)
        assert float(f.value(x)) == pytest.approx(min(float(base.value(x)), cap), abs=1e-12)
        assert float(f.value(x)) <= cap + 1e-12


def test_saturated_is_greedy_path() -> None:
    f = Saturated(_coverage(0), 1.0)
    with pytest.raises(NotImplementedError):
        f.multilinear(np.full(f.n, 0.5))
    # but it is fully maximizable/certifiable through the greedy path
    prob = SubmodularProblem(f, UniformMatroid(f.n, 3))
    sel, val = lazy_greedy(f, prob.matroid)
    _, opt = brute_force_max(f, prob.matroid)
    cert = certify_submodular_gap(prob, sel)
    assert cert.value <= opt + 1e-9
    assert opt <= cert.upper_bound + 1e-9


def test_saturated_batch_value_matches_pointwise() -> None:
    f = Saturated(_coverage(0), 0.7)
    rng = np.random.default_rng(9)
    batch = rng.integers(0, 2, size=(5, f.n)).astype(float)
    out = np.asarray(f.value(batch), dtype=float)
    assert out.shape == (5,)
    for i, row in enumerate(batch):
        assert out[i] == pytest.approx(float(f.value(row)), abs=1e-12)
