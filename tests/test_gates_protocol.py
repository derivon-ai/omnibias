# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Self-tests for the theory-program gate helpers in ``benchmarks/_gates.py``.

06-01 G1: each helper must fail on a deliberately bad input and pass on a
good one. Lives under ``tests/`` (not ``packages/omnibias-core/tests/``)
because ``_gates.py`` imports numpy at module scope and the core CI job is
numpy-free — a core-tests path would skip in every CI job. ``benchmarks/`` is
on ``[tool.pytest.ini_options].pythonpath``, so the bare ``_gates`` import
resolves without a hand-rolled ``sys.path`` insert.
"""

from __future__ import annotations

import numpy as np
import pytest
from _gates import (
    require_all_seeds,
    require_capture_rate,
    require_rel_error,
    require_scaling_exponent,
    require_within_stderr,
)


def test_require_scaling_exponent_passes() -> None:
    x = np.logspace(-3, 0, 13)
    y = 1.5 * x**2.0
    verdict = require_scaling_exponent(
        x, y, expected=2.0, tol=0.02, min_decades=3.0, name="ok"
    )
    assert verdict["passed"] is True
    assert abs(verdict["fitted_exponent"] - 2.0) < 1e-10


def test_require_scaling_exponent_wrong_exponent_raises() -> None:
    x = np.logspace(-3, 0, 13)
    y = x**3.0
    with pytest.raises(AssertionError, match="fitted exponent"):
        require_scaling_exponent(
            x, y, expected=2.0, tol=0.02, min_decades=3.0, name="bad_exp"
        )


def test_require_scaling_exponent_too_few_decades_raises() -> None:
    x = np.array([1.0, 2.0, 3.0])
    y = x**2.0
    with pytest.raises(AssertionError, match="decades"):
        require_scaling_exponent(
            x, y, expected=2.0, tol=0.02, min_decades=3.0, name="narrow"
        )


def test_require_scaling_exponent_nonpositive_raises() -> None:
    x = np.array([1.0, 10.0, 100.0, 1000.0])
    y = np.array([1.0, 4.0, -9.0, 16.0])
    with pytest.raises(AssertionError, match="strictly positive"):
        require_scaling_exponent(
            x, y, expected=2.0, tol=0.5, min_decades=3.0, name="neg"
        )


def test_require_scaling_exponent_two_points_raises_on_decades() -> None:
    """Two points spanning < 3 decades must fail the decades check."""
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 4.0])
    with pytest.raises(AssertionError, match="decades"):
        require_scaling_exponent(
            x, y, expected=2.0, tol=0.02, min_decades=3.0, name="two_pt"
        )


def test_require_rel_error_passes() -> None:
    verdict = require_rel_error(1.000001, 1.0, max_rel=1e-5, name="ok")
    assert verdict["passed"] is True


def test_require_rel_error_raises() -> None:
    with pytest.raises(AssertionError, match="rel_error"):
        require_rel_error(1.1, 1.0, max_rel=1e-3, name="far")


def test_require_within_stderr_passes() -> None:
    verdict = require_within_stderr(1.0, 1.001, 0.001, max_sigmas=3.0, name="ok")
    assert verdict["passed"] is True
    assert verdict["sigmas"] == pytest.approx(1.0)


def test_require_within_stderr_raises() -> None:
    with pytest.raises(AssertionError, match="max_sigmas"):
        require_within_stderr(1.0, 2.0, 0.1, max_sigmas=3.0, name="far")


def test_require_capture_rate_passes() -> None:
    verdict = require_capture_rate(128, 128, min_rate=1.0, name="ok")
    assert verdict["passed"] is True
    assert verdict["rate"] == 1.0


def test_require_capture_rate_raises_invalid_experiment() -> None:
    with pytest.raises(RuntimeError, match="INVALID EXPERIMENT"):
        require_capture_rate(127, 128, min_rate=1.0, name="miss")


def test_require_all_seeds_passes() -> None:
    rows = [{"seed": i, "fitted_exponent": 0.5 + 0.01 * (i - 2)} for i in range(5)]
    verdict = require_all_seeds(
        rows, key="fitted_exponent", expected=0.5, tol=0.1, name="ok"
    )
    assert verdict["passed"] is True
    assert verdict["n_seeds"] == 5
    assert verdict["worst_deviation"] <= 0.1


def test_require_all_seeds_worst_seed_raises() -> None:
    rows = [{"seed": i, "fitted_exponent": 0.5} for i in range(5)]
    rows[3]["fitted_exponent"] = 0.75  # |0.75 - 0.5| = 0.25 > 0.1
    with pytest.raises(AssertionError, match="exceeds tol"):
        require_all_seeds(
            rows, key="fitted_exponent", expected=0.5, tol=0.1, name="bad"
        )


def test_require_all_seeds_refuses_fewer_than_five() -> None:
    rows = [{"seed": i, "v": 1.0} for i in range(4)]
    with pytest.raises(ValueError, match="min_seeds"):
        require_all_seeds(rows, key="v", expected=1.0, tol=0.01, name="few")


def test_require_all_seeds_min_seeds_override() -> None:
    rows = [{"seed": 0, "v": 1.0}, {"seed": 1, "v": 1.001}]
    verdict = require_all_seeds(
        rows, key="v", expected=1.0, tol=0.01, name="ok", min_seeds=2
    )
    assert verdict["passed"] is True
    assert verdict["n_seeds"] == 2


def test_require_all_seeds_direction_min_passes() -> None:
    rows = [{"seed": i, "margin": 0.12 + 0.01 * i} for i in range(5)]
    verdict = require_all_seeds(
        rows,
        key="margin",
        expected=0.10,
        tol=0.0,
        direction="min",
        name="g1",
    )
    assert verdict["passed"] is True


def test_require_all_seeds_direction_min_raises() -> None:
    rows = [{"seed": i, "margin": 0.12} for i in range(5)]
    rows[1]["margin"] = 0.05
    with pytest.raises(AssertionError, match="direction=min"):
        require_all_seeds(
            rows,
            key="margin",
            expected=0.10,
            tol=0.0,
            direction="min",
            name="g1_fail",
        )
