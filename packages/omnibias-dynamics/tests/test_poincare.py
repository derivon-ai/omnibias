# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Rigorous Poincare-section return map: soundness + direction handling."""

from __future__ import annotations

import math

import pytest
from _dynamics_helpers import harmonic_float, rk4
from omnibias.dynamics import (
    PoincareSection,
    harmonic_oscillator,
    poincare_map,
)

TWO_PI = 2.0 * math.pi


def _contains(enclosure: tuple, point: tuple[float, ...]) -> bool:
    return all(enclosure[i].contains(point[i]) for i in range(len(point)))


def test_harmonic_upward_return_is_fixed_point() -> None:
    # Section y = 0, upward crossings. Starting at the upward-crossing point
    # (-1, 0), the next upward crossing returns to (-1, 0) after one full period.
    f, j = harmonic_oscillator(1.0)
    sec = PoincareSection(normal=(0.0, 1.0), offset=0.0, direction=1)
    cr = poincare_map(f, j, sec, [-1.0, 0.0], TWO_PI / 400, max_steps=600)
    assert cr.crossed
    assert _contains(cr.enclosure, (-1.0, 0.0))
    assert cr.time_bracket[0] <= TWO_PI <= cr.time_bracket[1]


def test_first_upward_crossing_location() -> None:
    # From (1, 0) the first upward crossing of y = 0 is at (-1, 0), time pi.
    f, j = harmonic_oscillator(1.0)
    sec = PoincareSection(normal=(0.0, 1.0), offset=0.0, direction=1)
    cr = poincare_map(f, j, sec, [1.0, 0.0], TWO_PI / 400, max_steps=600)
    assert cr.crossed
    assert _contains(cr.enclosure, (-1.0, 0.0))
    assert cr.time_bracket[0] <= math.pi <= cr.time_bracket[1]


def test_direction_selects_opposite_crossing() -> None:
    # Downward crossings of y = 0: from (-1, 0) the first is at (1, 0), time pi.
    f, j = harmonic_oscillator(1.0)
    sec = PoincareSection(normal=(0.0, 1.0), offset=0.0, direction=-1)
    cr = poincare_map(f, j, sec, [-1.0, 0.0], TWO_PI / 400, max_steps=600)
    assert cr.crossed
    assert _contains(cr.enclosure, (1.0, 0.0))


def test_enclosure_contains_float_crossing() -> None:
    # The rigorous crossing enclosure must contain a float-integrated crossing.
    f, j = harmonic_oscillator(1.0)
    sec = PoincareSection(normal=(0.0, 1.0), offset=0.0, direction=1)
    cr = poincare_map(f, j, sec, [1.0, 0.0], TWO_PI / 400, max_steps=600)
    # Float crossing: the trajectory (cos t, -sin t) crosses y=0 upward at t=pi.
    truth = rk4(harmonic_float(1.0), [1.0, 0.0], 0.0, math.pi, 20000)
    assert cr.crossed
    assert _contains(cr.enclosure, (truth[0], 0.0))


def test_no_crossing_reported_when_section_unreachable() -> None:
    # The unit circle never reaches y = 5.
    f, j = harmonic_oscillator(1.0)
    sec = PoincareSection(normal=(0.0, 1.0), offset=5.0, direction=0)
    cr = poincare_map(f, j, sec, [1.0, 0.0], TWO_PI / 100, max_steps=300)
    assert not cr.crossed


def test_invalid_step_raises() -> None:
    f, j = harmonic_oscillator(1.0)
    sec = PoincareSection(normal=(0.0, 1.0), offset=0.0)
    with pytest.raises(ValueError):
        poincare_map(f, j, sec, [1.0, 0.0], 0.0)
