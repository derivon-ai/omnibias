# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Verified stochastic layer: rigorous Fokker-Planck / Ito operator residuals.

Validated on the Ornstein-Uhlenbeck process ``dX = -theta X dt + sigma dW`` (the same
closed form as ``omnibias.score``'s ``test_sde.py``): its stationary density
``p_inf(x) = exp(-x^2 / (2V))`` with ``V = sigma^2 / (2 theta)`` makes the Fokker-Planck
adjoint vanish, ``L* p_inf = 0``. The density is reproduced *exactly* by a one-unit
``gaussian`` MLP (``g(z) = exp(-z^2/2)`` with ``w = 1/sqrt(V)``), so the certified jet is
exact and the residual enclosure must contain the true (zero) residual and tighten with
subdivision. Soundness is checked against the analytic closed form on a dense grid plus a
random in-box sample (repo rule); the sealed certificate is digest / tamper checked.
"""

from __future__ import annotations

import math
import random

import pytest
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.verified.interval import Interval
from omnibias.verify import (
    certified_fokker_planck_residual,
    certified_ito_generator_residual,
    certify_fokker_planck_residual,
    certify_ito_generator_residual,
    replay_stochastic_residual_certificate,
    stochastic_residual_schema_errors,
)

THETA = 0.8
A = 0.5  # sigma^2
V = A / (2.0 * THETA)  # stationary variance
W = 1.0 / math.sqrt(V)  # gaussian-unit weight so exp(-(Wx)^2/2) = exp(-x^2/(2V))


def _ou_density_layers() -> list:
    """A one-unit ``gaussian`` MLP equal to the OU stationary density ``exp(-x^2/(2V))``."""
    return [([[W]], [0.0], "gaussian"), ([[1.0]], [0.0], None)]


def _drift() -> list:
    """OU drift ``b(x) = -theta x`` as a per-axis interval coefficient."""
    return [lambda box: Interval.point(-THETA) * box[0]]


def _gauss_derivs(x: float) -> tuple[float, float, float]:
    """The OU stationary density and its first two derivatives at ``x``."""
    p = math.exp(-x * x / (2.0 * V))
    return p, -x / V * p, (x * x / V**2 - 1.0 / V) * p


def _true_fokker_planck(x: float, a: float) -> float:
    """Analytic ``L* p(x) = -((div b) p + b p') + 1/2 a p''`` for drift ``-theta x``."""
    p, pp, ppp = _gauss_derivs(x)
    return -((-THETA) * p + (-THETA * x) * pp) + 0.5 * a * ppp


def _true_generator(x: float, a: float) -> float:
    """Analytic ``L p(x) = b p' + 1/2 a p''`` for drift ``-theta x``."""
    _p, pp, ppp = _gauss_derivs(x)
    return (-THETA * x) * pp + 0.5 * a * ppp


def _grid_and_random(lo: float, hi: float, n_grid: int, seed: int) -> list[float]:
    step = (hi - lo) / (n_grid - 1)
    pts = [lo + k * step for k in range(n_grid)]
    rng = random.Random(seed)
    pts += [rng.uniform(lo, hi) for _ in range(40)]
    return pts


# --- Fokker-Planck adjoint --------------------------------------------------


def test_ou_stationary_fokker_planck_encloses_zero_and_tightens() -> None:
    """The stationary-density FP residual contains 0 and its sup-norm shrinks with splits."""
    layers, dom = _ou_density_layers(), [(-1.5, 1.5)]
    mags = []
    for s in (1, 4, 16, 64):
        r = certified_fokker_planck_residual(
            layers, dom, drift=_drift(), diffusion=[[A]], drift_divergence=-THETA, splits=s
        )
        assert r.lo <= 0.0 <= r.hi  # sound: encloses the true (zero) stationary residual
        mags.append(r.mag)
    assert all(mags[i + 1] < mags[i] for i in range(len(mags) - 1))  # subdivision tightens
    assert mags[-1] < 0.2  # detects near-stationarity


def test_fokker_planck_residual_is_sound_dense_and_random() -> None:
    """A mismatched diffusion gives a nonzero residual; the enclosure contains the truth."""
    layers, dom = _ou_density_layers(), [(-1.5, 1.5)]
    a_mismatch = 1.3  # != 2 theta V, so L* p != 0
    r = certified_fokker_planck_residual(
        layers, dom, drift=_drift(), diffusion=[[a_mismatch]], drift_divergence=-THETA, splits=64
    )
    truth = [_true_fokker_planck(x, a_mismatch) for x in _grid_and_random(-1.5, 1.5, 201, 1)]
    assert all(r.lo - 1e-12 <= v <= r.hi + 1e-12 for v in truth)  # soundness
    assert max(abs(v) for v in truth) > 0.3  # genuinely non-stationary (not a trivial 0-check)


def _sum_gaussian_2d_layers() -> list:
    """A 2-D net ``p(x, y) = exp(-x^2/(2V)) + exp(-y^2/(2V))`` (diagonal gaussian + sum readout)."""
    return [([[W, 0.0], [0.0, W]], [0.0, 0.0], "gaussian"), ([[1.0, 1.0]], [0.0], None)]


def _true_fokker_planck_2d(x: float, y: float, a: list[list[float]]) -> float:
    """Analytic 2-D FP residual for the sum-of-gaussians density (mixed partial is 0)."""
    gx, gxp, gxpp = _gauss_derivs(x)
    gy, gyp, gypp = _gauss_derivs(y)
    p = gx + gy
    px, py = gxp, gyp
    pxx, pyy, pxy = gxpp, gypp, 0.0
    div_b = -2.0 * THETA
    transport = div_b * p + (-THETA * x) * px + (-THETA * y) * py
    quad = a[0][0] * pxx + a[0][1] * pxy + a[1][0] * pxy + a[1][1] * pyy
    return -transport + 0.5 * quad


def test_fokker_planck_multi_axis_is_sound() -> None:
    """A genuine 2-D density: the enclosure contains the true vector-drift / matrix-diffusion FP residual."""
    layers, dom = _sum_gaussian_2d_layers(), [(-1.0, 1.0), (-1.0, 1.0)]
    drift = [lambda box: Interval.point(-THETA) * box[0], lambda box: Interval.point(-THETA) * box[1]]
    a = [[A, 0.2], [0.2, A]]  # off-diagonal exercises the mixed-partial plumbing
    r = certified_fokker_planck_residual(
        layers, dom, drift=drift, diffusion=a, drift_divergence=-2.0 * THETA, splits=[12, 12]
    )
    rng = random.Random(4)
    truth = [
        _true_fokker_planck_2d(px, py, a)
        for px in [-1.0 + k * 2.0 / 20 for k in range(21)]
        for py in [-1.0 + k * 2.0 / 20 for k in range(21)]
    ]
    truth += [_true_fokker_planck_2d(rng.uniform(-1, 1), rng.uniform(-1, 1), a) for _ in range(60)]
    assert all(r.lo - 1e-12 <= v <= r.hi + 1e-12 for v in truth)


# --- Ito generator ----------------------------------------------------------


def test_ito_generator_matches_closed_form_dense_and_random() -> None:
    """The generator residual encloses the analytic ``L p`` on the OU density."""
    layers, dom = _ou_density_layers(), [(-1.5, 1.5)]
    r = certified_ito_generator_residual(
        layers, dom, drift=_drift(), diffusion=[[A]], splits=64
    )
    truth = [_true_generator(x, A) for x in _grid_and_random(-1.5, 1.5, 201, 2)]
    assert all(r.lo - 1e-12 <= v <= r.hi + 1e-12 for v in truth)


def test_ito_generator_reaction_and_source_terms_are_sound() -> None:
    """``L f + c f - g`` with a reaction and source stays a sound enclosure."""
    layers, dom = _ou_density_layers(), [(-1.0, 1.0)]
    lam = 0.5
    r = certified_ito_generator_residual(
        layers, dom, drift=_drift(), diffusion=[[A]], reaction=-lam, source=0.0, splits=48
    )
    truth = []
    for x in _grid_and_random(-1.0, 1.0, 151, 3):
        p, _pp, _ppp = _gauss_derivs(x)
        truth.append(_true_generator(x, A) - lam * p)
    assert all(r.lo - 1e-12 <= v <= r.hi + 1e-12 for v in truth)


# --- Sealed certificate -----------------------------------------------------


def test_fokker_planck_certificate_seals_and_replays() -> None:
    cert = certify_fokker_planck_residual(
        _ou_density_layers(), [(-1.2, 1.2)], drift=_drift(), diffusion=[[A]],
        drift_divergence=-THETA, splits=32, max_residual=1.0,
        provenance={"process": "ornstein_uhlenbeck", "theta": THETA, "sigma_sq": A},
    )
    assert cert.operator == "fokker_planck"
    assert cert.residual_sup >= 0.0
    assert cert.verified
    assert stochastic_residual_schema_errors(cert.certificate) == []
    assert replay_stochastic_residual_certificate(cert.certificate)
    payload = cert.certificate["payload"]
    assert payload["diffusion_constant"] == [[A]]
    assert payload["spatial_dim"] == 1
    fin = payload["finite_obligation"]
    assert fin["type"] == "residual_sup_le_threshold"
    assert fin["margin"][0] == pytest.approx(1.0 - cert.residual_sup, abs=1e-9)


def test_ito_generator_certificate_operator_and_honesty() -> None:
    cert = certify_ito_generator_residual(
        _ou_density_layers(), [(-1.0, 1.0)], drift=_drift(), diffusion=[[A]], splits=16
    )
    assert cert.operator == "ito_generator"
    assert cert.verified
    honesty = cert.certificate["honesty"]
    assert honesty["unproven_claim"] is False
    assert "theorem_prover_verified" not in honesty
    assert honesty["interval_verified"] is True


def test_tampered_certificate_is_rejected() -> None:
    cert = certify_fokker_planck_residual(
        _ou_density_layers(), [(-1.0, 1.0)], drift=_drift(), diffusion=[[A]],
        drift_divergence=-THETA, splits=16,
    )
    assert cert.verified
    tampered = dict(cert.certificate)
    payload = dict(tampered["payload"])
    payload["residual_sup"] = 0.0  # forge a tighter bound
    tampered["payload"] = payload
    assert not verify_certificate_digest(tampered)  # digest breaks
    assert not replay_stochastic_residual_certificate(tampered)


def test_input_validation() -> None:
    layers, dom = _ou_density_layers(), [(-1.0, 1.0)]
    with pytest.raises(ValueError):  # drift wrong length
        certified_fokker_planck_residual(
            layers, dom, drift=[lambda b: b[0], lambda b: b[0]], diffusion=[[A]], drift_divergence=-THETA
        )
    with pytest.raises(ValueError):  # diffusion wrong shape
        certified_fokker_planck_residual(
            layers, dom, drift=_drift(), diffusion=[[A, A]], drift_divergence=-THETA
        )
    with pytest.raises(ValueError):  # negative threshold
        certify_fokker_planck_residual(
            layers, dom, drift=_drift(), diffusion=[[A]], drift_divergence=-THETA, max_residual=-1.0
        )
