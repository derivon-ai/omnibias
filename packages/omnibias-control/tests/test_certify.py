# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Recoverable-set certificate: interval-vs-float agreement, soundness, witnesses."""

from __future__ import annotations

import numpy as np
from omnibias.control import certify_disc_recoverable, disc_obstacle_margin

GAINS = (2.0, 2.0)
AMAX = 2.5
R = 1.0
CENTER = (0.0, 0.0)


def _phi_np(p, v, g=None):
    """Closed-form recoverability margin (float), matching disc_obstacle_margin."""
    d = np.asarray(p) - np.asarray(CENTER)
    g_mat = np.eye(2) if g is None else np.asarray(g, dtype=float)
    b = float(d @ d) - R * R
    dv = float(d @ np.asarray(v))
    v2 = float(np.asarray(v) @ np.asarray(v))
    u = 2.0 * (d @ g_mat)
    return 2 * v2 + 2 * (GAINS[0] + GAINS[1]) * dv + GAINS[0] * GAINS[1] * b + AMAX * np.sum(np.abs(u))


def _sample_min_phi(pr, vmax, g=None, n=6):
    """Minimum of phi over a dense grid of the box (an upper bound on min phi)."""
    from omnibias.core.verified import Interval  # noqa: F401  (import parity)

    axes = [np.linspace(lo, hi, n) for (lo, hi) in pr]
    vs = np.linspace(-vmax, vmax, n)
    best = np.inf
    for px in axes[0]:
        for py in axes[1]:
            for vx in vs:
                for vy in vs:
                    best = min(best, _phi_np((px, py), (vx, vy), g=g))
    return best


def test_interval_margin_matches_float():
    from omnibias.core.verified import Interval

    phi, _grad = disc_obstacle_margin(CENTER, R, GAINS, AMAX)
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = rng.uniform(-2, 2, 2)
        v = rng.uniform(-2, 2, 2)
        box = [Interval.point(p[0]), Interval.point(p[1]),
               Interval.point(v[0]), Interval.point(v[1])]
        iv = phi(box)
        exact = _phi_np(p, v)
        assert abs(iv.lo - exact) < 1e-9 and abs(iv.hi - exact) < 1e-9


def test_safe_box_is_certified():
    pr = [(-1.5, 1.5), (-2.5, -1.5)]
    cert = certify_disc_recoverable(CENTER, R, GAINS, AMAX, pr, 0.5, tol=1e-2)
    assert cert is not None
    assert cert.certified and cert.f_lower >= 0.0
    assert cert.f_lower <= cert.f_upper
    # soundness: the rigorous lower bound never exceeds a true (sampled) value.
    assert cert.f_lower <= _sample_min_phi(pr, 0.5) + 1e-9


def test_unsafe_box_has_rigorous_witness():
    # close under the disc, fast: some states cannot brake in time (non-recoverable).
    pr = [(-0.4, 0.4), (-1.15, -1.02)]
    vmax = 3.0
    sampled = _sample_min_phi(pr, vmax)
    assert sampled < 0.0                      # the box truly contains unsafe states
    cert = certify_disc_recoverable(CENTER, R, GAINS, AMAX, pr, vmax, tol=1e-2)
    assert cert is not None
    assert not cert.certified                 # cannot be proven fully recoverable
    assert cert.f_lower <= sampled + 1e-9     # sound
    assert cert.f_upper < 0.0                 # rigorous witness of a non-recoverable state


def test_model_relative_coupled_g():
    """A coupled model matrix ``g = M^{-1}B`` runs and stays sound / certifiable at low speed."""
    M = np.array([[1.0, 0.6], [0.6, 1.0]])
    g = np.linalg.inv(M)                       # B = I
    pr = [(-1.5, 1.5), (-2.5, -1.5)]
    cert = certify_disc_recoverable(CENTER, R, GAINS, AMAX, pr, 0.5, g=g, tol=1e-2)
    assert cert is not None and cert.certified
    assert cert.f_lower <= _sample_min_phi(pr, 0.5, g=g) + 1e-9
