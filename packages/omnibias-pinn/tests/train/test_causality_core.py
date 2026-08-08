# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for backend-free causality diagnostics and trivial-solution guards."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.pinn.train import (
    causality_index,
    report_causality,
    trivial_solution_guard,
    unlocked_fraction,
)


def test_causality_index_zero_for_nondecreasing():
    assert causality_index([1.0, 2.0, 3.0, 4.0]) == 0.0


def test_causality_index_one_for_strictly_decreasing():
    assert causality_index([4.0, 3.0, 2.0, 1.0]) == 1.0


def test_causality_index_partial_discordance():
    # One inversion out of three pairs: (0,2) is inverted among (0,1),(0,2),(1,2).
    idx = causality_index([1.0, 3.0, 2.0])
    assert idx == pytest.approx(1.0 / 3.0)


def test_causality_index_single_bin_is_zero():
    assert causality_index([1.5]) == 0.0
    assert causality_index([]) == 0.0


def test_causality_index_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        causality_index([1.0, -0.1])


def test_unlocked_fraction_is_min_weight():
    assert unlocked_fraction([1.0, 0.5, 0.1]) == pytest.approx(0.1)


def test_unlocked_fraction_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        unlocked_fraction([])


def test_report_causality_bundles_fields():
    rep = report_causality([1.0, 2.0, 3.0], [1.0, 0.5, 0.25])
    assert rep.n_bins == 3
    assert rep.causality_index == 0.0
    assert rep.unlocked_fraction == pytest.approx(0.25)
    assert rep.mean_per_bin == (1.0, 2.0, 3.0)


def test_trivial_solution_guard_flags_collapse():
    ic = np.sin(np.linspace(0, 2 * np.pi, 64))
    trivial = 1e-8 * np.ones_like(ic)
    v = trivial_solution_guard(trivial, ic, ratio_threshold=1e-3, mode="energy")
    assert v.is_trivial
    assert v.ratio < 1e-3


def test_trivial_solution_guard_passes_healthy_solution():
    ic = np.sin(np.linspace(0, 2 * np.pi, 64))
    sol = 0.8 * ic
    v = trivial_solution_guard(sol, ic, ratio_threshold=1e-3, mode="energy")
    assert not v.is_trivial
    assert v.ratio == pytest.approx(0.64, rel=1e-6)


def test_trivial_solution_guard_variance_mode_catches_constant():
    ic = np.sin(np.linspace(0, 2 * np.pi, 64))
    constant = np.full_like(ic, 0.5)
    v = trivial_solution_guard(constant, ic, ratio_threshold=1e-3, mode="variance")
    assert v.is_trivial
