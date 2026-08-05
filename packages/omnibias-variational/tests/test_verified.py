# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Rigorous (interval) least-action enclosures -- soundness tests.

Following the omnibias verified rule, every enclosure is checked to contain
**both** a dense deterministic grid and a random sample of true values:

- the action enclosure of the harmonic oscillator contains the exact analytic
  action ``S = -1/4 A^2 w (sin 2 w t1 - sin 2 w t0)``;
- the Euler-Lagrange and energy box-enclosures contain the true residual /
  energy at a grid + random points of the phase-space box;
- the sealed certificate round-trips and is tamper-evident.
Pure Python (no torch / jax).
"""

from __future__ import annotations

import math

import numpy as np
from omnibias.core.verified import Interval, cos_iv
from omnibias.variational.verified import (
    action_certificate,
    action_enclosure,
    energy_enclosure,
    euler_lagrange_enclosure,
)

A = 1.0
W = 2.0  # exact float, so w^2 = 4.0 is exact


def _sho_integrand_midpoints(a, b, n):
    """Interval enclosures of L(t) = -1/2 A^2 w^2 cos(2 w t) at the n midpoints."""
    h = (b - a) / n
    coeff = -0.5 * A**2 * W**2
    out = []
    for i in range(n):
        ti = a + (i + 0.5) * h
        out.append(cos_iv(Interval.point(2.0 * W * ti)) * coeff)
    return out


def _true_sho_action(a, b):
    return -0.25 * A**2 * W * (math.sin(2 * W * b) - math.sin(2 * W * a))


def test_action_enclosure_contains_analytic_action() -> None:
    a, b, n = 0.0, 1.3, 40
    vals = _sho_integrand_midpoints(a, b, n)
    # |L''| = |2 A^2 w^4 cos(2 w t)| <= 2 A^2 w^4.
    m2 = 2.0 * A**2 * W**4
    encl = action_enclosure(vals, a, b, second_derivative=Interval(-m2, m2), rule="midpoint")
    assert encl.contains(_true_sho_action(a, b))


def test_action_enclosure_coarser_is_wider_but_sound() -> None:
    a, b = 0.0, 1.3
    m2 = 2.0 * A**2 * W**4
    true = _true_sho_action(a, b)
    coarse = action_enclosure(
        _sho_integrand_midpoints(a, b, 10), a, b, second_derivative=Interval(-m2, m2),
    )
    fine = action_enclosure(
        _sho_integrand_midpoints(a, b, 80), a, b, second_derivative=Interval(-m2, m2),
    )
    assert coarse.contains(true)
    assert fine.contains(true)
    assert fine.width < coarse.width  # refinement tightens


def test_euler_lagrange_box_enclosure_is_sound() -> None:
    # SHO: EL(q, qddot) = qddot + w^2 q over a box.
    q_lo, q_hi = 0.3, 1.1
    a_lo, a_hi = -2.0, 1.0
    encl = euler_lagrange_enclosure(
        Interval(a_lo, a_hi), Interval(q_lo, q_hi) * (W**2), mass=1,
    )
    rng = np.random.default_rng(0)
    qs = np.concatenate([np.linspace(q_lo, q_hi, 17), rng.uniform(q_lo, q_hi, 50)])
    accs = np.concatenate([np.linspace(a_lo, a_hi, 17), rng.uniform(a_lo, a_hi, 50)])
    for q in qs:
        for acc in accs:
            assert encl.contains(acc + W**2 * q)


def test_energy_box_enclosure_is_sound() -> None:
    # SHO: E(q, qdot) = 1/2 qdot^2 + 1/2 w^2 q^2 over a box.
    q_lo, q_hi = 0.3, 1.1
    v_lo, v_hi = -0.5, 0.7
    pot = Interval(q_lo, q_hi).pow_int(2) * (0.5 * W**2)
    encl = energy_enclosure(Interval(v_lo, v_hi), pot, mass=1)
    rng = np.random.default_rng(1)
    qs = np.concatenate([np.linspace(q_lo, q_hi, 17), rng.uniform(q_lo, q_hi, 50)])
    vs = np.concatenate([np.linspace(v_lo, v_hi, 17), rng.uniform(v_lo, v_hi, 50)])
    for q in qs:
        for v in vs:
            assert encl.contains(0.5 * v**2 + 0.5 * W**2 * q**2)


def test_action_certificate_roundtrips_and_is_tamper_evident() -> None:
    from omnibias.core.proof.certificate import verify_certificate_digest

    iv = Interval(1.0, 2.0)
    cert = action_certificate(iv, scope="local_box", meta={"system": "sho"})
    assert verify_certificate_digest(cert)
    assert cert["meta"]["scope"] == "local_box"
    assert cert["meta"]["system"] == "sho"
    tampered = dict(cert)
    tampered["payload"] = {"type": "interval", "interval": {"lo": "0x0p+0", "hi": "0x1p+10"}}
    assert not verify_certificate_digest(tampered)
