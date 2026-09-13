# SPDX-License-Identifier: Apache-2.0
"""Independent Gaussian-image/character diagnostics and certificate attacks.

High-precision reference sums are numerical diagnostics. The production
enclosures are licensed by the analytic infinite-tail proof, not these samples.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as Q
from functools import cache
from random import Random
from typing import Any

import mpmath as mp  # type: ignore[import-untyped]
import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.stochastic import small_time as api


def _mp(value: Q | str | int) -> Any:
    q = Q(value)
    return mp.mpf(q.numerator) / q.denominator


def _contains(bounds: list[str], value: Any) -> None:
    assert _mp(bounds[0]) <= value <= _mp(bounds[1]), (bounds, mp.nstr(value, 40))


def _images(tq: Q, aq: Q) -> dict[str, Any]:
    """Differentiate unregularized Gaussian images directly, without API charts."""
    t, u = _mp(tq), mp.pi * _mp(aq)
    rows = []
    for ell in range(-5, 6):
        v = u - 2 * mp.pi * ell
        e = mp.exp(-v * v / t)
        rows.append((
            v * e,
            (1 - 2 * v**2 / t) * e,
            (4 * v**3 / t**2 - 6 * v / t) * e,
            (-8 * v**4 / t**3 + 24 * v**2 / t**2 - 6 / t) * e,
            v**3 / t**2 * e,
            (3 * v**2 / t**2 - 2 * v**4 / t**3) * e,
        ))
    n, n1, n2, n3, nt, n1t = [mp.fsum(row[j] for row in rows) for j in range(6)]
    pref = 2 * mp.sqrt(mp.pi) * mp.exp(t / 4) * t ** (-mp.mpf(3) / 2)
    if aq in (0, 1):
        kernel = pref * n1 / mp.cos(u)
        radial = mp.mpf(0)
        laplacian = (n3 / n1 + 1) / 4
        time_score = mp.mpf(1) / 4 - mp.mpf(3) / (2 * t) + n1t / n1
    else:
        kernel = pref * n / mp.sin(u)
        radial = n1 / n - mp.cot(u)
        laplacian = (
            n2 / n - (n1 / n)**2 + 1 / mp.sin(u)**2 + 2 * mp.cot(u) * radial
        ) / 4
        time_score = mp.mpf(1) / 4 - mp.mpf(3) / (2 * t) + nt / n
    return {
        "log_kernel_enclosure": mp.log(kernel),
        "scaled_kernel_enclosure": kernel / pref * mp.exp(u**2 / t),
        "radial_log_derivative_enclosure": radial,
        "group_log_laplacian_enclosure": laplacian,
        "time_log_derivative_enclosure": time_score,
        "group_squared_log_gradient_enclosure": radial**2 / 4,
    }


def _points() -> list[tuple[Q, Q]]:
    points = [(t, Q(j, 8)) for t in (Q(5, 64), Q(1, 64), Q(1, 4096)) for j in range(9)]
    points += [
        (t, 1 - t * c)
        for t in (Q(1, 64), Q(1, 4096))
        for c in (Q(1, 16), Q(1, 4), Q(1, 2), Q(1), Q(4))
    ]
    points += [(Q(1, 64), Q(1, 2**100)), (Q(1, 64), 1 - Q(1, 2**100))]
    rng = Random(518726)
    points += [(Q(rng.randint(1, 64), 1024), Q(rng.randint(0, 10000), 10000)) for _ in range(12)]
    return points


@pytest.mark.parametrize("time,angle", _points())
def test_all_enclosures_independent_image_grid_and_seeded_samples(time: Q, angle: Q) -> None:
    with mp.workdps(160):
        report = api.su2_small_time_heat_kernel(time, angle)
        arithmetic = report["witness"]["arithmetic"]
        oracle = _images(time, angle)
        for name, value in oracle.items():
            _contains(arithmetic[name], value)
        _contains(arithmetic["uniform_time_score_enclosure"], oracle["time_log_derivative_enclosure"])
        _contains(arithmetic["heat_equation_identity_enclosure"], 0)
        assert report["all_image_tail_verified"]


@pytest.mark.parametrize("time,angle", [(Q(5, 64), Q(0)), (Q(5, 64), Q(1)), (Q(5, 64), Q(3, 7)), (Q(1, 64), Q(1))])
def test_character_spectrum_independently_matches_poisson_normalization(time: Q, angle: Q) -> None:
    # The cut-locus character sum loses hundreds of decimal digits; use 430
    # digits and 650 terms here. No such cancellation enters production.
    with mp.workdps(430):
        t, u = _mp(time), mp.pi * _mp(angle)
        values, derivatives = [], []
        for n in range(650):
            casimir = mp.mpf(n * (n + 2)) / 4
            character = ((-1)**n if angle == 1 else 1) * (n + 1) if angle in (0, 1) else mp.sin((n + 1) * u) / mp.sin(u)
            value = (n + 1) * mp.exp(-t * casimir) * character
            values.append(value)
            derivatives.append(-casimir * value)
        kernel = mp.fsum(values)
        row = api.su2_small_time_heat_kernel(time, angle)["witness"]["arithmetic"]
        _contains(row["log_kernel_enclosure"], mp.log(kernel))
        _contains(row["time_log_derivative_enclosure"], mp.fsum(derivatives) / kernel)


@pytest.mark.parametrize("time", [Q(5, 64), Q(1, 64), Q(1, 4096), Q(1, 10**8)])
def test_uniform_score_exact_arithmetic_and_scope(time: Q) -> None:
    report = api.su2_small_time_score_bounds(time)
    a = report["witness"]["arithmetic"]
    assert Q(a["score_lower_constant"]) == -3 / time
    assert Q(a["score_upper_constant"]) == -Q(3, 2) / time + Q(1, 2)
    assert Q(a["score_lower_angle_squared_coefficient"]) == 1 / time**2
    assert api.replay_small_time_certificate(report["certificate"])
    for name in ("actual_ks_vacuum_verified", "wilson_conditional_law_identified", "infinite_volume_claim", "continuum_claim", "yang_mills_mass_gap_claim", "theorem_prover_verified", "mathlib_verified"):
        assert report[name] is False


def test_cut_locus_time_squared_growth_and_underflow_safe_logarithm() -> None:
    t = Q(1, 4096)
    row = api.su2_small_time_heat_kernel(t, 1)["witness"]["arithmetic"]
    assert Q(row["log_kernel_enclosure"][1]) < -40000
    assert Q(row["scaled_kernel_enclosure"][0]) > 0
    assert row["radial_log_derivative_enclosure"] == ["0", "0"]
    assert Q(row["group_log_laplacian_enclosure"][0]) * t**2 > 9
    # A globally O(1/t) logarithmic-Laplacian claim would miss this actual value.
    assert Q(row["group_log_laplacian_enclosure"][0]) > 10000 / t


@pytest.mark.parametrize("terms", [16, 24, 48])
def test_finite_even_series_tail_independent_reference(terms: int) -> None:
    t, angle = Q(1, 64), 1 - Q(1, 256)
    with mp.workdps(100):
        row = api.su2_small_time_heat_kernel(t, angle, series_terms=terms)["witness"]["arithmetic"]
        for name, value in _images(t, angle).items():
            _contains(row[name], value)


@pytest.mark.parametrize("time", [True, False, 0, -1, Q(-1, 64), Q(6, 64), 0.01, "1/64"])
def test_time_domain_refuses_inexact_or_unproved_values(time: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        api.su2_small_time_score_bounds(time)
    with pytest.raises((ValueError, TypeError)):
        api.su2_small_time_heat_kernel(time, 0)


@pytest.mark.parametrize("angle", [True, False, Q(-1, 10), Q(11, 10), 0.5, "1/2"])
def test_angle_requires_exact_pi_fraction(angle: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        api.su2_small_time_heat_kernel(Q(1, 64), angle)


@pytest.mark.parametrize("terms", [True, 15, 129, 32.0, "32"])
def test_finite_series_resource_guard(terms: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        api.su2_small_time_heat_kernel(Q(1, 64), 0, series_terms=terms)


@cache
def _certificate(point: bool) -> dict[str, Any]:
    builder = api.su2_small_time_heat_kernel(Q(1, 64), Q(7, 8)) if point else api.su2_small_time_score_bounds(Q(1, 64))
    return dict(builder["certificate"])


@pytest.mark.parametrize("point", [False, True])
@pytest.mark.parametrize("attack", ["scope", "source", "input", "arithmetic", "type", "honesty_type"])
def test_rehashed_attacks_do_not_earn_analytic_claim(point: bool, attack: str) -> None:
    cert = deepcopy(_certificate(point))
    witness = cert["payload"]["witness"]
    if attack == "scope":
        cert["honesty"]["continuum_claim"] = True
    elif attack == "source":
        if point:
            upstream = witness["uniform_source_certificate"]
            upstream["payload"]["witness"]["arithmetic"]["relative_image_remainder_upper"] = "0"
            witness["uniform_source_certificate"] = seal_certificate(upstream)
        else:
            witness["angle_quantifier"] = "all complex angles"
    elif attack == "input":
        witness["inputs"]["time"] = "1/32"
    elif attack == "arithmetic":
        key = "log_kernel_enclosure" if point else "score_upper_constant"
        witness["arithmetic"][key] = ["0", "0"] if point else "0"
    elif attack == "type":
        cert["payload"]["type"] = "not_a_heat_kernel"
    else:
        cert["honesty"]["all_image_tail_verified"] = 1
    assert not api.replay_small_time_certificate(seal_certificate(cert))


@pytest.mark.parametrize("value", [None, [], (), "certificate", 1, True])
def test_replay_nonmapping_guard(value: Any) -> None:
    assert api.replay_small_time_certificate(value) is False


def test_failed_upstream_replay_prevents_point_certificate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "replay_small_time_certificate", lambda _certificate: False)
    with pytest.raises(ValueError, match="source failed replay"):
        api.su2_small_time_heat_kernel(Q(1, 64), 0)


@pytest.mark.parametrize("point", [False, True])
def test_full_canonical_replay(point: bool) -> None:
    assert api.replay_small_time_certificate(_certificate(point))
