# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""The honestly non-tight optimality-gap certificate and its graceful degradation."""

from __future__ import annotations

import sys

import numpy as np
import pytest
from omnibias.nphard import brute_force_min, certify_gap, decode, qap, schedule
from omnibias.nphard._core.qap import qap_decode


def _small_qap(seed: int) -> object:
    rng = np.random.default_rng(seed)
    flow = rng.integers(0, 6, size=(3, 3)).astype(float)
    dist = rng.integers(0, 6, size=(3, 3)).astype(float)
    flow = (flow + flow.T) / 2.0
    dist = (dist + dist.T) / 2.0
    np.fill_diagonal(flow, 0.0)
    np.fill_diagonal(dist, 0.0)
    return qap(flow, dist)


def test_spectral_sandwich_brackets_the_brute_force_optimum() -> None:
    """lower_bound <= brute_force optimum <= decoded energy, on tiny QAP + scheduling."""
    problems = [_small_qap(s) for s in range(4)]
    problems += [schedule(np.random.default_rng(s).integers(1, 12, size=6).astype(float), 2)
                 for s in range(4)]
    for prob in problems:
        x_opt, e_opt = brute_force_min(prob)
        x_dec, e_dec = decode(prob)
        cert = certify_gap(prob, x_dec, kind="spectral")
        assert cert.method == "spectral"
        assert cert.lower_bound <= e_opt + 1e-6  # lower bound never exceeds the true optimum
        assert e_opt <= cert.energy + 1e-6  # decoded energy is an upper bound
        assert cert.is_sound
        assert cert.absolute_gap >= -1e-9


def test_sos_sandwich_is_sound_certified_and_sealed() -> None:
    """The SOS/Lasserre bound is sound, sealed, and at least as tight as spectral (QAP dim3)."""
    pytest.importorskip("omnibias.sos")
    prob = _small_qap(0)
    _, e_opt = brute_force_min(prob)
    x_dec, _ = qap_decode(prob)
    sos = certify_gap(prob, x_dec, kind="sos", level=1, bisection_steps=16)
    spectral = certify_gap(prob, x_dec, kind="spectral")
    assert sos.method == "sos"
    assert sos.certified is True
    assert sos.sealed is not None
    assert sos.lower_bound <= e_opt + 1e-6
    assert sos.is_sound
    assert sos.lower_bound >= spectral.lower_bound - 1e-6  # SOS at least as tight


def test_gap_is_honestly_non_tight_never_asserted_zero() -> None:
    """NP-hard: the certified gap is generally non-zero; there is no exactness field."""
    prob = _small_qap(1)
    x_dec, _ = decode(prob)
    cert = certify_gap(prob, x_dec, kind="spectral")
    # gap-shaped container: reports a (generally positive) gap, no is_optimal / is_exact
    assert cert.relative_gap >= 0.0
    for banned in ("is_optimal", "is_exact", "optimal", "zero_gap"):
        assert not hasattr(cert, banned)


def test_sos_falls_back_to_spectral_without_sos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "omnibias.sos", None)
    prob = _small_qap(2)
    _, e_opt = brute_force_min(prob)
    x_dec, _ = decode(prob)
    cert = certify_gap(prob, x_dec, kind="sos")
    assert cert.method == "spectral"  # degraded, not crashed
    assert cert.sealed is None
    assert cert.lower_bound <= e_opt + 1e-6
    assert cert.is_sound


def test_spectral_degrades_to_float_without_convex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "omnibias.convex", None)
    prob = _small_qap(3)
    _, e_opt = brute_force_min(prob)
    x_dec, _ = decode(prob)
    cert = certify_gap(prob, x_dec, kind="spectral")
    assert cert.certified is False  # not interval-sealed
    assert cert.lower_bound <= e_opt + 1e-6  # but still a valid bound
    assert cert.is_sound


def test_certify_rejects_a_non_binary_point() -> None:
    prob = _small_qap(0)
    with pytest.raises(ValueError, match="binary"):
        certify_gap(prob, np.full(prob.n, 0.5), kind="spectral")
