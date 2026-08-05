# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Certified finite-time Lyapunov-exponent brackets against analytic oracles."""

from __future__ import annotations

import math

import pytest
from omnibias.dynamics import (
    certified_lyapunov_exponent,
    harmonic_oscillator,
    linear_system,
)

TWO_PI = 2.0 * math.pi


def test_contracting_linear_leading_exponent() -> None:
    # y' = diag(-1, -2) y: sigma_max(M(T)) = e^{-T}, so the leading exponent is -1.
    f, j = linear_system([[-1.0, 0.0], [0.0, -2.0]])
    lb = certified_lyapunov_exponent(f, j, [1.0, 1.0], time=1.0, n_steps=200)
    assert lb.lower <= -1.0 <= lb.upper
    assert lb.width < 1e-6


def test_expanding_linear_leading_exponent() -> None:
    # y' = diag(1, 0.5) y: leading finite-time exponent is +1.
    f, j = linear_system([[1.0, 0.0], [0.0, 0.5]])
    lb = certified_lyapunov_exponent(f, j, [1.0, 1.0], time=1.0, n_steps=200)
    assert lb.lower <= 1.0 <= lb.upper


def test_rotation_has_zero_exponent() -> None:
    # A pure rotation neither grows nor shrinks: sigma_max = 1, exponent 0.
    f, j = harmonic_oscillator(1.0)
    lb = certified_lyapunov_exponent(f, j, [1.0, 0.0], time=TWO_PI, n_steps=400)
    assert lb.contains(0.0)
    assert lb.width < 1e-6


def test_bracket_is_ordered() -> None:
    f, j = linear_system([[-0.3, 0.1], [0.0, -0.7]])
    lb = certified_lyapunov_exponent(f, j, [1.0, 1.0], time=2.0, n_steps=200)
    assert lb.lower <= lb.upper


def test_probe_direction_changes_lower_bound() -> None:
    # Probing the slow direction gives a (still sound) lower bound below the fast one.
    f, j = linear_system([[1.0, 0.0], [0.0, 0.5]])
    fast = certified_lyapunov_exponent(f, j, [1.0, 1.0], time=1.0, probe=[1.0, 0.0])
    slow = certified_lyapunov_exponent(f, j, [1.0, 1.0], time=1.0, probe=[0.0, 1.0])
    assert slow.lower <= fast.lower <= fast.upper


def test_invalid_arguments_raise() -> None:
    f, j = harmonic_oscillator(1.0)
    with pytest.raises(ValueError):
        certified_lyapunov_exponent(f, j, [1.0, 0.0], time=0.0)
    with pytest.raises(ValueError):
        certified_lyapunov_exponent(f, j, [1.0, 0.0], time=1.0, probe=[1.0, 0.0, 0.0])
