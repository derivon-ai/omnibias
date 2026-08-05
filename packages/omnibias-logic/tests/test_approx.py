# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""The statistical layer hits its (fixed-seed) targets and stays type-quarantined."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.logic import CountCertificate, exact_model_count, model_count
from omnibias.logic.approx import ApproxCount, ConformalCounter, approx_model_count


def _random_cnf(rng: np.random.Generator, lo: int = 4, hi: int = 6) -> object:
    n = int(rng.integers(lo, hi + 1))
    clauses = []
    for _ in range(int(rng.integers(1, 4))):
        k = int(rng.integers(1, 3))
        variables = rng.choice(np.arange(1, n + 1), size=min(k, n), replace=False)
        signs = rng.choice([-1, 1], size=len(variables))
        clauses.append([int(s * v) for s, v in zip(signs, variables, strict=True)])
    return model_count(clauses, n_vars=n)


def test_hashing_estimate_brackets_exact_with_high_probability() -> None:
    # a moderate instance with many models; (epsilon, delta) = (0.8, 0.2).
    rng = np.random.default_rng(0)
    n = 9
    clauses = [
        [int(s * v) for s, v in zip(rng.choice([-1, 1], 2), rng.choice(np.arange(1, n + 1), 2, replace=False), strict=True)]
        for _ in range(6)
    ]
    mc = model_count(clauses, n_vars=n)
    exact = exact_model_count(mc)
    estimates = []
    covered = 0
    for seed in range(10):
        approx = approx_model_count(mc, epsilon=0.8, delta=0.2, seed=seed)
        estimates.append(approx.estimate)
        covered += approx.contains(exact)
    assert covered >= 7  # delta = 0.2 -> expect >= 8/10; 7 is a generous, non-flaky floor
    assert 0.5 * exact <= float(np.median(estimates)) <= 2.0 * exact


def test_hashing_returns_a_non_sound_approxcount() -> None:
    approx = approx_model_count(model_count([[1, 2]], n_vars=4), seed=1)
    assert isinstance(approx, ApproxCount)
    assert approx.worst_case_sound is False
    assert not isinstance(approx, CountCertificate)
    assert "NOT" in approx.disclaimer


def test_hashing_rejects_weighted() -> None:
    weighted = model_count([[1, 2]], weights=np.array([[1.0, 2.0], [1.0, 1.0]]), n_vars=2)
    with pytest.raises(ValueError, match="unweighted"):
        approx_model_count(weighted)


def test_conformal_coverage_meets_target_on_held_out_split() -> None:
    alpha = 0.2
    rng = np.random.default_rng(2024)
    problems = [_random_cnf(rng) for _ in range(180)]
    truths = [exact_model_count(p) for p in problems]
    counter = ConformalCounter(alpha=alpha, seed=0, samples=2000)
    counter.fit(problems[:110], truths[:110])
    covered = sum(counter.predict(p).contains(t) for p, t in zip(problems[110:], truths[110:], strict=True))
    coverage = covered / len(problems[110:])
    assert coverage >= 1.0 - alpha - 0.1  # marginal guarantee + generous finite-sample slack


def test_conformal_returns_a_non_sound_approxcount() -> None:
    rng = np.random.default_rng(5)
    problems = [_random_cnf(rng) for _ in range(20)]
    truths = [exact_model_count(p) for p in problems]
    counter = ConformalCounter(alpha=0.1, seed=1, samples=800).fit(problems, truths)
    out = counter.predict(problems[0])
    assert isinstance(out, ApproxCount)
    assert out.worst_case_sound is False
    assert out.confidence == pytest.approx(0.9)


def test_approxcount_cannot_be_constructed_sound() -> None:
    ApproxCount(estimate=1.0, interval=(0.0, 2.0), method="x")  # default worst_case_sound=False
    with pytest.raises(ValueError, match="never be worst_case_sound"):
        ApproxCount(estimate=1.0, interval=(0.0, 2.0), method="x", worst_case_sound=True)


def test_approxcount_is_structurally_distinct_from_count_certificate() -> None:
    approx_fields = set(ApproxCount.__dataclass_fields__)
    cert_fields = set(CountCertificate.__dataclass_fields__)
    assert approx_fields != cert_fields
    assert "worst_case_sound" in approx_fields and "worst_case_sound" not in cert_fields
    assert "estimate" in approx_fields and "estimate" not in cert_fields


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        ConformalCounter().predict(model_count([[1]], n_vars=1))
