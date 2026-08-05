# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified argmax / measure-mode collapse (``omnibias.struct.select``).

Soundness of the three closed-form sub-claims (value gap ``log(N)/beta``, mode-mass
concentration, ``L^inf`` argmax stability) over a dense deterministic grid *and* a random
``(logits, beta)`` sample across many seeds; exact Gibbs moments (mean / covariance /
directional cumulants) vs numpy references; torch <-> jax bit-identical parity (``< 1e-9`` on a
well-determined instance, held across seeds); and the seal / verify digest roundtrip.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from omnibias.struct import (
    SelectionCertificate,
    argmax_stability_margin,
    beta_for_confidence,
    certify_argmax,
    mass_concentration_bound,
    seal_selection_certificate,
)

RTOL, ATOL = 1e-9, 1e-11
PARITY_TOL = 1e-9

# A dense-ish deterministic grid: sizes, temperatures, and a few logit shapes.
GRID_N = (2, 3, 5, 8)
GRID_BETA = (0.25, 1.0, 4.0, 16.0, 64.0)


def _ref_softmax(a: np.ndarray, beta: float) -> np.ndarray:
    s = beta * a
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)


def _ref_lse(a: np.ndarray, beta: float) -> float:
    s = beta * a
    m = s.max()
    return float((m + np.log(np.exp(s - m).sum())) / beta)


def _logit_shapes(n: int, rng: np.random.Generator) -> list[np.ndarray]:
    """A few structurally distinct logit vectors of length ``n``."""
    return [
        np.linspace(-1.0, 1.0, n),  # graded, unique argmax
        np.concatenate([[3.0], np.zeros(n - 1)]),  # a dominant spike
        rng.standard_normal(n),  # generic
    ]


# --------------------------------------------------------------------------- #
# Certificate soundness (numpy core): grid + random sample across seeds.
# --------------------------------------------------------------------------- #
def _assert_sound(cert: SelectionCertificate, logits: np.ndarray, beta: float) -> None:
    n = logits.size
    # Value-gap sandwich vs brute force: max <= lse_beta <= max + log(N)/beta.
    hard = float(np.max(logits))
    soft = _ref_lse(logits, beta)
    assert cert.hard_value == pytest.approx(hard, rel=RTOL, abs=ATOL)
    assert cert.soft_value == pytest.approx(soft, rel=RTOL, abs=ATOL)
    assert cert.gap_bound == pytest.approx(math.log(n) / beta, rel=RTOL, abs=ATOL)
    assert soft >= hard - ATOL  # lse_beta >= max
    assert soft <= hard + cert.gap_bound + ATOL  # within the certified gap
    # Mass concentration vs the *exact* p_max.
    p = _ref_softmax(logits, beta)
    p_max = float(np.max(p))
    assert cert.p_max == pytest.approx(p_max, rel=RTOL, abs=ATOL)
    assert cert.p_max_lower <= p_max + ATOL  # the closed-form bound is a genuine LOWER bound
    assert cert.mass_concentration_holds
    # The whole certificate self-check.
    assert cert.is_sound
    assert cert.certified


def test_certificate_sound_on_grid() -> None:
    rng = np.random.default_rng(0)
    for n in GRID_N:
        for logits in _logit_shapes(n, rng):
            for beta in GRID_BETA:
                cert = certify_argmax(logits, beta)
                _assert_sound(cert, logits, beta)


def test_certificate_sound_on_random_sample() -> None:
    for seed in range(20):  # >= K seeds
        rng = np.random.default_rng(1000 + seed)
        n = int(rng.integers(2, 11))
        scale = float(rng.uniform(0.2, 4.0))
        logits = scale * rng.standard_normal(n)
        beta = float(rng.uniform(0.3, 50.0))
        cert = certify_argmax(logits, beta)
        _assert_sound(cert, logits, beta)


def test_argmax_and_margin_match_logits() -> None:
    rng = np.random.default_rng(7)
    for _ in range(30):
        n = int(rng.integers(2, 9))
        logits = rng.standard_normal(n)
        cert = certify_argmax(logits, 3.0)
        assert cert.argmax == int(np.argmax(logits))
        order = np.sort(logits)[::-1]
        assert cert.margin == pytest.approx(order[0] - order[1], rel=RTOL, abs=ATOL)
        assert cert.margin == pytest.approx(argmax_stability_margin(logits), rel=RTOL, abs=ATOL)


# --------------------------------------------------------------------------- #
# Argmax stability over an L^inf ball: stable iff margin > 2 eps.
# --------------------------------------------------------------------------- #
def test_argmax_stability_threshold() -> None:
    logits = np.array([2.0, 1.0, 0.5, 0.0])  # margin = 1.0
    m = argmax_stability_margin(logits)
    assert m == pytest.approx(1.0)
    stable = certify_argmax(logits, 5.0, eps=0.4 * m)  # 2*eps = 0.8 m < m -> stable
    unstable = certify_argmax(logits, 5.0, eps=0.6 * m)  # 2*eps = 1.2 m > m -> not stable
    assert stable.argmax_stable is True
    assert stable.robust_radius == pytest.approx(0.5)  # m / 2
    assert unstable.argmax_stable is False
    # eps not queried -> None (no claim made), still sound.
    assert certify_argmax(logits, 5.0).argmax_stable is None


# --------------------------------------------------------------------------- #
# Closed-form bounds: mass_concentration_bound & beta_for_confidence.
# --------------------------------------------------------------------------- #
def test_mass_concentration_is_a_lower_bound() -> None:
    rng = np.random.default_rng(3)
    for _ in range(50):
        n = int(rng.integers(2, 12))
        logits = rng.standard_normal(n)
        beta = float(rng.uniform(0.5, 30.0))
        m = argmax_stability_margin(logits)
        bound = mass_concentration_bound(m, n, beta)
        p_max = float(np.max(_ref_softmax(logits, beta)))
        assert 0.0 < bound <= p_max + ATOL


def test_beta_for_confidence_roundtrip() -> None:
    for n, m, target in [(3, 2.0, 0.99), (10, 0.5, 0.9), (2, 1.0, 0.999)]:
        b = beta_for_confidence(m, n, target)
        assert mass_concentration_bound(m, n, b) == pytest.approx(target, rel=1e-9, abs=1e-12)
        # A hair below the required beta undershoots the target.
        assert mass_concentration_bound(m, n, 0.99 * b) < target


def test_beta_to_infinity_collapse() -> None:
    logits = np.array([1.5, 0.3, -0.2, -1.0])
    hard = float(np.max(logits))
    prev_gap, prev_lower = math.inf, -1.0
    for beta in (1.0, 4.0, 16.0, 64.0, 256.0, 1024.0):
        cert = certify_argmax(logits, beta)
        assert cert.gap_bound <= prev_gap + ATOL  # gap shrinks monotonically
        assert cert.p_max_lower >= prev_lower - ATOL  # mass concentrates monotonically
        prev_gap, prev_lower = cert.gap_bound, cert.p_max_lower
    assert cert.gap_bound < 1e-2  # -> 0
    assert cert.p_max_lower > 0.999  # -> 1
    assert cert.soft_value == pytest.approx(hard, abs=1e-2)  # -> max


def test_singleton_is_trivially_certified() -> None:
    cert = certify_argmax([2.0], beta=3.0)
    assert cert.num_choices == 1
    assert cert.gap_bound == 0.0
    assert cert.p_max == 1.0
    assert cert.p_max_lower == 1.0
    assert math.isinf(cert.margin)
    assert cert.is_sound and cert.certified


def test_certify_argmax_input_validation() -> None:
    with pytest.raises(ValueError):
        certify_argmax([], beta=1.0)
    with pytest.raises(ValueError):
        certify_argmax([1.0, 2.0], beta=0.0)
    with pytest.raises(ValueError):
        certify_argmax([1.0, 2.0], beta=1.0, eps=-0.1)


# --------------------------------------------------------------------------- #
# Seal / verify digest roundtrip + tamper detection.
# --------------------------------------------------------------------------- #
def test_seal_verify_roundtrip_and_tamper() -> None:
    cert = certify_argmax([3.0, 1.0, 0.5], beta=6.0, eps=0.2)
    sealed = seal_selection_certificate(cert, meta={"note": "unit-test"})
    # Import here: seal() warms omnibias.core.verified, breaking the core proof/verified cycle.
    from omnibias.core.proof.certificate import verify_certificate_digest

    assert verify_certificate_digest(sealed)
    assert sealed["claim"].startswith("softmax(")
    assert sealed["payload"]["type"] == "certified_argmax"
    assert sealed["payload"]["argmax"] == cert.argmax
    assert sealed["payload"]["argmax_stable"] is True
    assert sealed["honesty"]["unproven_claim"] is False
    assert sealed["meta"]["note"] == "unit-test"
    # Tampering with any bound invalidates the digest.
    tampered = {**sealed, "payload": {**sealed["payload"], "p_max_lower": 0.0}}
    assert not verify_certificate_digest(tampered)


# --------------------------------------------------------------------------- #
# Gibbs moments vs numpy references (torch), incl. directional cumulants.
# --------------------------------------------------------------------------- #
def _ref_cumulants(p: np.ndarray, v: np.ndarray, order: int) -> list[float]:
    """Cumulants ``kappa_1..kappa_order`` (order <= 4) of ``X = v_I``, ``I ~ p``, from moments."""
    mean = float(p @ v)
    xc = v - mean
    mu = {k: float(p @ (xc**k)) for k in (2, 3, 4)}
    out = [mean]
    if order >= 2:
        out.append(mu[2])
    if order >= 3:
        out.append(mu[3])
    if order >= 4:
        out.append(mu[4] - 3.0 * mu[2] ** 2)
    return out[:order]


def test_gibbs_moments_torch_vs_reference() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch import (
        gibbs_covariance,
        gibbs_cumulants_directional,
        gibbs_mean,
        soft_argmax,
        soft_max_value,
    )

    rng = np.random.default_rng(11)
    s = rng.standard_normal(6)
    v = rng.standard_normal(6)
    beta = 2.3
    st, vt = torch.tensor(s, dtype=torch.float64), torch.tensor(v, dtype=torch.float64)
    p = _ref_softmax(s, beta)
    assert np.allclose(gibbs_mean(st, beta).numpy(), p, rtol=RTOL, atol=ATOL)
    assert np.allclose(soft_argmax(st, beta).numpy(), p, rtol=RTOL, atol=ATOL)
    assert soft_max_value(st, beta).item() == pytest.approx(_ref_lse(s, beta), rel=RTOL, abs=ATOL)
    cov = gibbs_covariance(st, beta).numpy()
    assert np.allclose(cov, np.diag(p) - np.outer(p, p), rtol=RTOL, atol=ATOL)
    kappa = gibbs_cumulants_directional(st, vt, beta, order=4).numpy()
    assert np.allclose(kappa, _ref_cumulants(p, v, 4), rtol=1e-7, atol=1e-9)
    assert kappa[0] == pytest.approx(float(p @ v), rel=RTOL, abs=ATOL)  # mean
    assert kappa[1] == pytest.approx(float(v @ cov @ v), rel=RTOL, abs=ATOL)  # variance


def test_gibbs_cumulants_order_validation() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch import gibbs_cumulants_directional

    with pytest.raises(ValueError):
        gibbs_cumulants_directional(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 1.0]), 1.0, order=0)


# --------------------------------------------------------------------------- #
# torch <-> jax bit-identical parity (< 1e-9), well-determined instance + seeds.
# --------------------------------------------------------------------------- #
def test_select_torch_jax_parity() -> None:
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.struct.jax import select as js
    from omnibias.struct.torch import select as ts

    for seed in range(6):
        rng = np.random.default_rng(500 + seed)
        # Well-determined instance: a clear margin so the parity tol is not masking a tie.
        s = np.sort(rng.standard_normal(7))[::-1] + np.linspace(0.0, 0.3, 7)[::-1]
        v = rng.standard_normal(7)
        beta = float(rng.uniform(1.0, 12.0))
        st, vt = torch.tensor(s, dtype=torch.float64), torch.tensor(v, dtype=torch.float64)
        sj, vj = jnp.asarray(s), jnp.asarray(v)

        def close(a, b) -> bool:  # noqa: ANN001
            return bool(np.max(np.abs(np.asarray(a) - np.asarray(b))) < PARITY_TOL)

        assert close(ts.soft_max_value(st, beta).numpy(), js.soft_max_value(sj, beta))
        assert close(ts.soft_argmax(st, beta).numpy(), js.soft_argmax(sj, beta))
        assert close(ts.gibbs_mean(st, beta).numpy(), js.gibbs_mean(sj, beta))
        assert close(ts.gibbs_covariance(st, beta).numpy(), js.gibbs_covariance(sj, beta))
        kt = ts.gibbs_cumulants_directional(st, vt, beta, order=4).numpy()
        kj = np.asarray(js.gibbs_cumulants_directional(sj, vj, beta, order=4))
        assert close(kt, kj)
        soft_t, cert_t = ts.certified_argmax(st, beta, eps=0.05)
        soft_j, cert_j = js.certified_argmax(sj, beta, eps=0.05)
        assert close(soft_t.numpy(), soft_j)
        assert cert_t.argmax == cert_j.argmax
        assert cert_t.p_max_lower == pytest.approx(cert_j.p_max_lower, rel=RTOL, abs=ATOL)


def test_certified_argmax_matches_core_and_rejects_batched() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.struct.torch import select as ts

    s = np.array([2.0, 0.5, -0.3, 1.1])
    soft, cert = ts.certified_argmax(torch.tensor(s, dtype=torch.float64), beta=4.0, eps=0.1)
    assert float(soft.sum()) == pytest.approx(1.0, abs=1e-12)
    ref = certify_argmax(s, 4.0, eps=0.1)
    assert cert.argmax == ref.argmax
    assert cert.p_max == pytest.approx(ref.p_max, rel=RTOL, abs=ATOL)
    with pytest.raises(ValueError):
        ts.certified_argmax(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), beta=1.0)
