# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 03-10: jet-Padé singularity tracking, gates G1–G6."""

from __future__ import annotations

from math import factorial
from pathlib import Path

import pytest
from omnibias.core.verified.interval import Interval
from omnibias.difference.singularity import (
    DISCLAIMER,
    FROISSART_REL,
    SingularityTrack,
    agreement,
    certified_singularity_annulus,
    domb_sykes,
    honesty_payload,
    pade_estimate,
    pade_singularities,
    remainder_on_safe_disc,
    track_singularity,
)


def _geom(rho: float, n: int) -> tuple[float, ...]:
    return tuple(rho**k for k in range(n + 1))


def _branch_half(xs: float, n: int) -> tuple[float, ...]:
    # [x^k] (1 - x/xs)^{-1/2} = xs^{-k} * binom(2k, k) / 4^k
    out = [1.0]
    for k in range(1, n + 1):
        out.append(out[-1] * (2 * k - 1) / (2 * k) / xs)
    return tuple(out)


def test_worked_pole() -> None:
    c = _geom(10.0 / 3.0, 4)
    ds = domb_sykes(c, drop_first=1)
    assert not ds.failed
    assert ds.location is not None
    assert abs(ds.location.real - 0.3) < 1e-9
    assert abs((ds.exponent or 0.0) - 1.0) < 1e-9
    pd = pade_estimate(c, numer_deg=0, denom_deg=1)
    assert pd.location is not None
    assert abs(pd.location.real - 0.3) < 1e-9


def test_g1_known_recovery() -> None:
    c = _geom(10.0 / 3.0, 8)
    ds = domb_sykes(c)
    assert ds.location is not None
    assert abs(ds.location.real - 0.3) / 0.3 <= 1e-6
    br = _branch_half(0.3, 8)
    ds_b = domb_sykes(br, drop_first=2)
    assert not ds_b.failed and ds_b.location is not None
    assert abs(ds_b.location.real - 0.3) / 0.3 <= 1e-6
    assert abs((ds_b.exponent or 0.0) - 0.5) <= 1e-5
    ess = tuple(1.0 / factorial(k) for k in range(12))
    assert domb_sykes(ess).failed
    two = tuple((10.0 / 3.0) ** k + (10.0 / 3.1) ** k for k in range(10))
    hard = domb_sykes(two)
    assert hard.failed or (hard.location is not None and abs(hard.location.real - 0.3) / 0.3 > 1e-3)


def test_g2_annulus_sound() -> None:
    for rho in (2.0, 3.0, 4.0, 5.0):
        xs = 1.0 / rho
        c = [Interval.from_value(rho**k) for k in range(8)]
        enc = certified_singularity_annulus(c, tail_bound=Interval.point(1.0), tail_ratio=rho)
        assert enc.lo <= xs <= enc.hi


def test_g4_agreement_and_hard_case() -> None:
    c = _geom(10.0 / 3.0, 6)
    gap = agreement(domb_sykes(c), pade_estimate(c, numer_deg=0, denom_deg=1))
    assert gap <= 1e-8
    two = tuple((10.0 / 3.0) ** k + ((-10.0 / 3.0) ** k) for k in range(8))
    ds = domb_sykes(two)
    pd = pade_estimate(two, numer_deg=1, denom_deg=2)
    assert agreement(ds, pd) == pytest.approx(float("inf")) or ds.failed or (
        pd.location is not None and ds.location is not None and agreement(ds, pd) > 0.05
    )


def test_g5_tracking() -> None:
    times = (0.0, 0.2, 0.4, 0.6)
    rows = []
    for t in times:
        zs = 0.2 + 0.1j * (1.0 - t)
        rows.append(tuple(zs ** (-(k + 1)) for k in range(8)))
    track = track_singularity(rows, times, method="domb_sykes")
    assert track.disclaimer == DISCLAIMER
    assert track.blowup_fit is not None
    assert abs(track.blowup_fit.t_c - 1.0) / 1.0 <= 0.01
    assert track.blowup_fit.t_c_lo <= 1.0 <= track.blowup_fit.t_c_hi


def test_g6_disclaimer_serialized() -> None:
    track = SingularityTrack((0.0,), (0.3 + 0.0j,), (1.0,), None)
    payload = track.to_payload()
    assert payload["disclaimer"] == DISCLAIMER
    assert honesty_payload()["blowup_proof"] is False
    assert honesty_payload()["ns_regularity"] is False


def test_froissart_threshold_justified() -> None:
    assert FROISSART_REL == 1e-8
    poles = pade_singularities(_geom(2.0, 4), numer_deg=0, denom_deg=1)
    assert len(poles) == 1
    assert abs(poles[0] - 0.5) < 1e-9


def test_remainder_reuses_pade() -> None:
    rem = remainder_on_safe_disc(_geom(2.0, 6), numer_deg=0, denom_deg=1, radius=0.2, tail_bound=1.0, tail_ratio=2.0)
    assert rem.lo <= 0.0 <= rem.hi
    assert rem.mag < 1.0


def test_terminology() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "omnibias" / "difference"
    for rel in ("_core/singularity.py", "singularity.py"):
        text = (root / rel).read_text(encoding="utf-8")
        assert "founding bias collapse" in text
        assert "delta -> 0" in text
        assert "beta -> inf" in text
        assert "feasibility" in text
        assert "do not conflate" in text.lower()
        assert "not a proof of blow-up" in text
