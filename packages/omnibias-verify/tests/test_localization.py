# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Theory 03-08: certified scan localization, gates G1–G6."""

from __future__ import annotations

import numpy as np
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.verified.interval import Interval
from omnibias.verify.localization import (
    Inconclusive,
    ScanResponse,
    branch_and_bound_peak,
    certify_multiple_peaks,
    certify_peak,
    honesty_payload,
    seal,
)


def test_g1_soundness_grid_and_random() -> None:
    grid = np.linspace(-0.8, 0.8, 41)
    rng = np.random.default_rng(0)
    random = rng.uniform(-0.8, 0.8, size=64)
    alphas = rng.uniform(3.0, 8.0, size=64)
    for tau_star in grid:
        resp = ScanResponse.sech2_peak(float(tau_star), alpha=5.0)
        box = Interval(float(tau_star) - 0.10, float(tau_star) + 0.10)
        got = certify_peak(resp, box=box, max_iter=8)
        assert not isinstance(got, Inconclusive), (tau_star, got)
        assert got.offset_enclosure.contains(float(tau_star))
        xs = np.linspace(box.lo, box.hi, 51)
        vals = [resp.deriv(Interval.point(float(x)), 0).mid for x in xs]
        assert got.offset_enclosure.contains(float(xs[int(np.argmax(vals))]))
    for tau_star, alpha in zip(random, alphas, strict=True):
        resp = ScanResponse.sech2_peak(float(tau_star), alpha=float(alpha))
        half = 0.50 / float(alpha)
        box = Interval(float(tau_star) - half, float(tau_star) + half)
        got = certify_peak(resp, box=box, max_iter=8)
        assert not isinstance(got, Inconclusive), (tau_star, got)
        assert got.offset_enclosure.contains(float(tau_star))
        assert got.unique_in == box
        assert got.scope == "local_box"


def test_g2_uniqueness_and_two_peaks() -> None:
    one = ScanResponse.sech2_peak(-0.3, alpha=5.0)
    box = Interval(-0.40, -0.20)
    cert = certify_peak(one, box=box)
    assert not isinstance(cert, Inconclusive)
    assert cert.second_derivative_sign == "negative"
    two = ScanResponse.two_peaks(-0.3, 0.3, alpha=6.0)
    wide = Interval(-0.6, 0.6)
    got = certify_peak(two, box=wide)
    assert isinstance(got, Inconclusive)
    found, _exhaustive = certify_multiple_peaks(two, box=wide, max_peaks=2, min_width=0.05)
    assert len(found) >= 1
    for item in found:
        assert item.unique_in.width < wide.width


def test_g3_krawczyk_tighter_than_bab() -> None:
    resp = ScanResponse.sech2_peak(-0.3, alpha=5.0)
    box = Interval(-0.40, -0.20)
    cert = certify_peak(resp, box=box, max_iter=4)
    assert not isinstance(cert, Inconclusive)
    assert cert.route == "krawczyk"
    bab = branch_and_bound_peak(resp, box, n_iters=4)
    assert cert.offset_enclosure.width * 100.0 <= bab.width


def test_g4_quadratic_contraction() -> None:
    resp = ScanResponse.sech2_peak(-0.3, alpha=5.0)
    box = Interval(-0.40, -0.20)
    cert = certify_peak(resp, box=box, max_iter=4)
    assert not isinstance(cert, Inconclusive)
    widths = [w for w in cert.widths if w > 0.0]
    assert len(widths) >= 4
    for prev, nxt in zip(widths[1:-1], widths[2:], strict=True):
        assert nxt <= 40.0 * prev * prev


def test_g5_degenerate_is_inconclusive() -> None:
    box = Interval(-0.2, 0.2)
    for resp in (ScanResponse.constant(1.0), ScanResponse.flat_max()):
        got = certify_peak(resp, box=box)
        assert isinstance(got, Inconclusive)
        assert got.reason


def test_g6_seal_and_tamper() -> None:
    resp = ScanResponse.sech2_peak(-0.3, alpha=5.0)
    cert = certify_peak(resp, box=Interval(-0.40, -0.20))
    assert not isinstance(cert, Inconclusive)
    sealed = seal(cert)
    assert verify_certificate_digest(sealed)
    tampered = dict(sealed)
    payload = dict(tampered["payload"])
    payload["offset"] = [0.0, 1.0]
    tampered["payload"] = payload
    assert verify_certificate_digest(tampered) is False
    assert honesty_payload()["theorem_prover_verified"] is False
    assert honesty_payload()["scope"] == "local_box"


def test_worked_physical_map() -> None:
    resp = ScanResponse.sech2_peak(-0.3, alpha=5.0, direction=1.0, bias=0.0)
    cert = certify_peak(resp, box=Interval(-0.4, -0.2))
    assert not isinstance(cert, Inconclusive)
    assert cert.physical_enclosure.contains(0.3)
    assert cert.offset_enclosure.contains(-0.3)
