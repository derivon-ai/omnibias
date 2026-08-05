# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Rigorous periodic-orbit existence via the Krawczyk / radii-polynomial test."""

from __future__ import annotations

import math

import pytest
from omnibias.dynamics import (
    hopf_normal_form,
    prove_periodic_orbit,
    radial_logistic,
)

TWO_PI = 2.0 * math.pi


def test_hopf_limit_cycle_exists() -> None:
    # The scalar radial map of the Hopf form has an isolated fixed point r = 1
    # (the limit cycle radius); a true periodic orbit is proved to exist there.
    f, j = radial_logistic(1.0)
    cert = prove_periodic_orbit(f, j, [1.0], TWO_PI, n_steps=200)
    assert cert.exists
    assert cert.enclosure is not None
    lo, hi = cert.enclosure[0]
    assert lo <= 1.0 <= hi
    assert cert.krawczyk is not None and cert.krawczyk.kappa < 1.0


def test_certificate_payload_is_present() -> None:
    f, j = radial_logistic(1.0)
    cert = prove_periodic_orbit(f, j, [1.0], TWO_PI, n_steps=200)
    assert cert.krawczyk is not None
    payload = cert.krawczyk.certificate["payload"]
    assert payload["type"] == "krawczyk"


def test_limit_cycle_radius_scales_with_mu() -> None:
    # r' = 4 r - r^3 has its limit cycle at r = 2.
    f, j = radial_logistic(4.0)
    cert = prove_periodic_orbit(f, j, [2.0], TWO_PI, n_steps=200)
    assert cert.exists
    assert cert.enclosure is not None
    lo, hi = cert.enclosure[0]
    assert lo <= 2.0 <= hi


def test_declines_away_from_orbit() -> None:
    # r = 0.5 is not a fixed point of the radial flow: no orbit is certified there.
    f, j = radial_logistic(1.0)
    cert = prove_periodic_orbit(f, j, [0.5], TWO_PI, n_steps=200)
    assert not cert.exists
    assert cert.enclosure is None


def test_unreduced_autonomous_map_is_non_isolated() -> None:
    # In 2-D the time-T map has the trivial multiplier 1 (phase direction), so the
    # un-reduced fixed point is non-isolated and the test must decline gracefully.
    f, j = hopf_normal_form(1.0)
    cert = prove_periodic_orbit(f, j, [1.0, 0.0], TWO_PI, n_steps=80, radii=(1e-3, 1e-2))
    assert not cert.exists


def test_invalid_period_raises() -> None:
    f, j = radial_logistic(1.0)
    with pytest.raises(ValueError):
        prove_periodic_orbit(f, j, [1.0], 0.0)
