# SPDX-License-Identifier: Apache-2.0
"""Independent character, Haar, Weyl and high-precision enclosure checks."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from fractions import Fraction as Q
from functools import cache, lru_cache
from itertools import permutations
from math import comb, factorial, prod
from random import Random
from typing import Any

import mpmath as mp  # type: ignore[import-untyped]
import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.stochastic.heat_kernel import (
    normalized_product_tv_bound,
    su2_heat_kernel_bridge,
    su2_heat_kernel_enclosure,
    su2_wilson_heat_kernel_comparison,
    su3_heat_kernel_torus_enclosure,
)
from omnibias.geometry.gauge.stochastic.heat_kernel import (
    replay_heat_kernel_certificate as replay,
)


def _mp(q: Q | str | int) -> Any:
    f = Q(q)
    return mp.mpf(f.numerator) / f.denominator


def _contains(bounds: list[str], value: Any) -> None:
    assert _mp(bounds[0]) <= value <= _mp(bounds[1])


@cache
def _polynomial(n: int, derivative: int) -> tuple[int, ...]:
    coefficients = [0] * (n - derivative + 1)
    for k in range((n - derivative) // 2 + 1):
        degree = n - 2 * k
        coefficients[2 * k] = (
            (-1) ** k
            * comb(n - k, k)
            * 2**degree
            * factorial(degree)
            // factorial(degree - derivative)
        )
    return tuple(coefficients)


def _su2(t: Q, x: Q, derivative: int = 0, terms: int = 90) -> Any:
    # Explicit integer polynomial coefficients independently differentiate
    # U_n. They also avoid mpmath hypergeometric convergence failures at
    # exact polynomial roots; this is an oracle correction, not extra padding.
    tt, xx = _mp(t), _mp(x)
    return mp.fsum(
        (n + 1) * mp.exp(-tt * n * (n + 2) / 4) * mp.polyval(_polynomial(n, derivative), xx)
        for n in range(derivative, terms)
    )


@pytest.mark.parametrize("derivative", [0, 1, 2])
def test_su2_full_series_deterministic_grid_and_seeded_samples(derivative: int) -> None:
    rng = Random(71934 + derivative)
    points = [Q(j, 12) for j in range(-12, 13)]
    points += [Q(rng.randint(-1000, 1000), 1000) for _ in range(16)]
    with mp.workdps(85):
        for x in points:
            t = Q(rng.randint(10, 50), 25)
            report = su2_heat_kernel_enclosure(t, x, cutoff=24, derivative_order=derivative)
            a = report["witness"]["arithmetic"]
            assert report["status"] == "PASS"
            _contains(a["value_enclosure"], _su2(t, x, derivative))
            n = 25
            actual_tail = mp.fsum(
                (j + 1) * mp.exp(-_mp(t) * j * (j + 2) / 4) * sum(_polynomial(j, derivative))
                for j in range(n, 90)
            )
            assert actual_tail <= _mp(a["tail_upper"])


Poly = dict[tuple[int, int], Q]


def _add(a: Poly, b: Poly, scale: Q = Q(1)) -> Poly:
    out: defaultdict[tuple[int, int], Q] = defaultdict(Q, a)
    for index, value in b.items():
        out[index] += scale * value
    return {index: value for index, value in out.items() if value}


def _mul(a: Poly, b: Poly) -> Poly:
    out: defaultdict[tuple[int, int], Q] = defaultdict(Q)
    for (i, j), value in a.items():
        for (k, ell), other in b.items():
            out[i + k, j + ell] += value * other
    return dict(out)


def _characters(linear: Poly, order: int) -> list[Poly]:
    chars: list[Poly] = [{(0, 0): Q(1)}, {k: 2 * v for k, v in linear.items()}]
    for _ in range(2, order + 1):
        twice = {k: 2 * v for k, v in _mul(linear, chars[-1]).items()}
        chars.append(_add(twice, chars[-2], Q(-1)))
    return chars


def _haar(poly: Poly) -> Q:
    total = Q(0)
    for (a, b), coefficient in poly.items():
        if a % 2 == b % 2 == 0:
            numerator = prod(range(1, a, 2)) * prod(range(1, b, 2))
            denominator = prod(range(4, a + b + 4, 2))
            total += coefficient * Q(numerator, denominator)
    return total


def test_exact_haar_character_convolution_in_noncentral_boundary() -> None:
    # S3 moments integrate the FULL group, not only conjugacy classes.
    # B=(3/5,4/5,0,0), so scalar(X^-1B)=(3X0+4X1)/5.
    left = _characters({(1, 0): Q(1)}, 7)
    right = _characters({(1, 0): Q(3, 5), (0, 1): Q(4, 5)}, 7)
    for n in range(8):
        for m in range(8):
            expected = (
                sum(c * Q(3, 5) ** a for (a, _), c in left[n].items()) / (n + 1) if n == m else 0
            )
            assert _haar(_mul(left[n], right[m])) == expected
    # A central-only integral would not give this two-coordinate identity.
    assert _haar(_mul(left[1], right[1])) == Q(3, 5)


def _unit(rng: Random) -> list[Q]:
    v = [Q(rng.randint(-20, 20), 13) for _ in range(3)]
    square = sum((z * z for z in v), Q(0))
    return [(1 - square) / (1 + square), *[2 * z / (1 + square) for z in v]]


def test_bridge_actual_density_grid_random_center_and_replay() -> None:
    rng = Random(20123)
    grid = [[Q(1), Q(0), Q(0), Q(0)], [Q(-1), Q(0), Q(0), Q(0)], [Q(0), Q(1), Q(0), Q(0)]]
    pairs = [(x, b) for x in grid for b in grid] + [(_unit(rng), _unit(rng)) for _ in range(15)]
    with mp.workdps(85):
        for x, b in pairs:
            t = Q(1, 5)
            report = su2_heat_kernel_bridge(t, x, b)
            dot = sum((a * c for a, c in zip(x, b, strict=True)), Q(0))
            expected = _su2(t, x[0]) * _su2(4 * t, dot) / _su2(5 * t, b[0])
            assert report["status"] == "PASS"
            _contains(report["witness"]["arithmetic"]["density_enclosure"], expected)
            assert replay(report["certificate"])


def test_bridge_conjugation_covariance_is_not_a_deterministic_root() -> None:
    # A simultaneous rational rotation of vector coordinates preserves all
    # three heat-kernel class arguments. This includes B=-I.
    x = [Q(3, 5), Q(4, 5), Q(0), Q(0)]
    b = [Q(0), Q(3, 5), Q(4, 5), Q(0)]

    def rotate(q: list[Q]) -> list[Q]:
        return [q[0], -q[2], q[1], q[3]]

    a = su2_heat_kernel_bridge(Q(1, 5), x, b)["witness"]["arithmetic"]
    c = su2_heat_kernel_bridge(Q(1, 5), rotate(x), rotate(b))["witness"]["arithmetic"]
    assert a == c


def _su3_weyl(t: Q, angles: list[Q], cutoff: int = 42) -> Any:
    # Alternating Weyl determinants are independent of Jacobi-Trudi/complete
    # symmetric functions used in production. Random angles avoid collisions.
    a, b = map(_mp, angles)
    eigenvalues = [mp.exp(1j * a), mp.exp(1j * b), mp.exp(-1j * (a + b))]
    powers = [[z**j for j in range(cutoff + 3)] for z in eigenvalues]
    perms = list(permutations(range(3)))
    signs = [(-1) ** sum(p[i] > p[j] for i in range(3) for j in range(i + 1, 3)) for p in perms]

    def alternant(exponents: tuple[int, int, int]) -> Any:
        return mp.fsum(
            sign * prod(powers[i][exponents[p[i]]] for i in range(3))
            for p, sign in zip(perms, signs, strict=True)
        )

    denominator = alternant((2, 1, 0))
    terms = []
    for p in range(cutoff + 1):
        for q in range(cutoff + 1 - p):
            dim = (p + 1) * (q + 1) * (p + q + 2) // 2
            character = alternant((p + q + 2, q + 1, 0)) / denominator
            energy = Q(p * p + q * q + p * q + 3 * p + 3 * q, 3)
            terms.append(dim * mp.exp(-_mp(t * energy)) * character.real)
    return mp.fsum(terms)


def test_su3_full_representation_grid_and_seeded_weyl_oracle() -> None:
    rng = Random(4556)
    points = [[Q(a, 7), Q(b, 11)] for a, b in [(1, 2), (1, -3), (2, 3), (-3, 4), (5, -2)]]
    points += [[Q(rng.randint(1, 8), 13), Q(rng.randint(-8, -1), 17)] for _ in range(7)]
    with mp.workdps(80):
        for angles in points:
            t = Q(rng.randint(4, 20), 10)
            report = su3_heat_kernel_torus_enclosure(t, angles, cutoff=16)
            assert report["status"] == "PASS"
            _contains(report["witness"]["arithmetic"]["value_enclosure"], _su3_weyl(t, angles))
        identity = su3_heat_kernel_torus_enclosure(1, [0, 0])
        expected = mp.fsum(
            ((p + 1) * (q + 1) * (p + q + 2) / 2) ** 2
            * mp.exp(-_mp(Q(p * p + q * q + p * q + 3 * p + 3 * q, 3)))
            for p in range(42)
            for q in range(42 - p)
        )
        _contains(identity["witness"]["arithmetic"]["value_enclosure"], expected)
        assert replay(identity["certificate"])


def test_insufficient_cutoff_and_antipodal_cancellation_are_inconclusive() -> None:
    reports = [
        su2_heat_kernel_enclosure(Q(1, 100), 1, cutoff=0),
        su3_heat_kernel_torus_enclosure(Q(1, 100), [0, 0], cutoff=0),
        su2_heat_kernel_bridge(Q(1, 100), [1, 0, 0, 0], [-1, 0, 0, 0], cutoff=64),
    ]
    for report in reports:
        assert report["status"] == "INCONCLUSIVE"
        assert not report["finite_gate_verified"]
        assert replay(report["certificate"])


def test_normalized_product_tv_bound_against_exact_finite_measures() -> None:
    rng = Random(14709)
    for _ in range(25):
        base = [Q(rng.randint(1, 10)) for _ in range(5)]
        factors = [[Q(rng.randint(1, 10), 10) for _ in base] for _ in range(4)]
        bounds = [[min(f), max(f)] for f in factors]
        report = normalized_product_tv_bound(bounds)
        weights = [prod(f[i] for f in factors) for i in range(len(base))]
        z1, z2 = sum(base), sum(b * w for b, w in zip(base, weights, strict=True))
        tv = sum(abs(b / z1 - b * w / z2) for b, w in zip(base, weights, strict=True)) / 2
        assert tv <= Q(report["witness"]["arithmetic"]["total_variation_upper"])
        assert not report["pointwise_factor_bounds_verified"]
        assert not report["actual_measure_distance_verified"]
        assert replay(report["certificate"])
    assert normalized_product_tv_bound([])["witness"]["arithmetic"]["total_variation_upper"] == "0"
    assert (
        normalized_product_tv_bound([[2, 2], [3, 3]])["witness"]["arithmetic"][
            "total_variation_upper"
        ]
        == "0"
    )


@pytest.mark.parametrize("steps", [8, 16, 32])
def test_actual_wilson_convolution_error_grid_random_and_all_spin_tail(steps: int) -> None:
    rng = Random(6739 + steps)
    report = su2_wilson_heat_kernel_comparison(4, steps, face_count=4)
    a, w = report["witness"]["arithmetic"], report["witness"]
    assert report["status"] == "PASS"
    assert report["wilson_to_heat_kernel_error_verified"]
    assert report["finite_coarse_product_comparison_verified"]
    assert not report["multidimensional_wilson_refinement_verified"]
    assert replay(report["certificate"])
    points = [Q(j, 12) for j in range(-12, 13)] + [
        Q(rng.randint(-1000, 1000), 1000) for _ in range(16)
    ]
    with mp.workdps(85):
        beta = mp.mpf(steps) / 2
        denominator = mp.besseli(1, beta)
        multipliers = [(mp.besseli(n + 1, beta) / denominator) ** steps for n in range(100)]
        actual_tail = mp.fsum((n + 1) ** 2 * multipliers[n] for n in range(33, 100))
        assert actual_tail <= _mp(w["wilson_tail"]["tail_upper"])
        for x in points:
            density = mp.fsum(
                (n + 1) * multipliers[n] * mp.polyval(_polynomial(n, 0), _mp(x)) for n in range(100)
            )
            heat = _su2(Q(4), x)
            assert abs(density - heat) <= _mp(a["uniform_density_error_upper"])
            assert abs(density / heat - 1) <= _mp(a["uniform_relative_error_upper"])
    assert Q(a["finite_face_total_variation_upper"]) < 1


def test_wilson_multiplier_normalization_via_independent_haar_integral() -> None:
    report = su2_wilson_heat_kernel_comparison(4, 8, cutoff=12)
    with mp.workdps(60):
        denominator = mp.quad(
            lambda theta: mp.exp(4 * mp.cos(theta)) * mp.sin(theta) ** 2, [0, mp.pi]
        )
        for row in report["witness"]["retained_coefficients"][:5]:
            n = row["doubled_spin"]
            numerator = mp.quad(
                lambda theta, n=n: (
                    mp.exp(4 * mp.cos(theta)) * mp.sin((n + 1) * theta) * mp.sin(theta)
                ),
                [0, mp.pi],
            )
            _contains(row["wilson_multiplier"], numerator / ((n + 1) * denominator))


def test_wilson_comparison_separates_absolute_from_relative_and_refinement_claims() -> None:
    relative_fail = su2_wilson_heat_kernel_comparison(1, 1, cutoff=24)
    assert relative_fail["status"] == "PASS"
    assert relative_fail["actual_single_group_distance_verified"]
    assert not relative_fail["finite_coarse_product_comparison_verified"]
    assert relative_fail["witness"]["arithmetic"]["finite_face_total_variation_upper"] is None
    tail_fail = su2_wilson_heat_kernel_comparison(1, 20, cutoff=0)
    assert tail_fail["status"] == "INCONCLUSIVE"
    assert not tail_fail["wilson_to_heat_kernel_error_verified"]
    assert replay(tail_fail["certificate"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"convolution_steps": True},
        {"convolution_steps": 0},
        {"convolution_steps": 1.0},
        {"convolution_steps": 1025},
        {"convolution_steps": 1, "face_count": 0},
        {"convolution_steps": 1, "face_count": True},
    ],
)
def test_wilson_comparison_exact_integer_guards(kwargs: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        su2_wilson_heat_kernel_comparison(4, **kwargs)
    with pytest.raises(ValueError, match="Bessel"):
        su2_wilson_heat_kernel_comparison(Q(1, 1000), 1)


def test_wilson_source_and_nested_tv_tamper_rejected() -> None:
    certificate = su2_wilson_heat_kernel_comparison(4, 8, face_count=4)["certificate"]
    for part in ["coefficient", "tail", "nested", "scope"]:
        cert = deepcopy(certificate)
        w = cert["payload"]["witness"]
        if part == "coefficient":
            w["retained_coefficients"][0]["wilson_multiplier"] = ["0", "0"]
        elif part == "tail":
            w["wilson_tail"]["tail_upper"] = "0"
        elif part == "nested":
            w["product_implication_certificate"]["honesty"]["pointwise_factor_bounds_verified"] = (
                True
            )
            w["product_implication_certificate"] = seal_certificate(
                w["product_implication_certificate"]
            )
        else:
            cert["honesty"]["multidimensional_wilson_refinement_verified"] = True
        assert not replay(seal_certificate(cert))


@pytest.mark.parametrize("value", [True, False, 0.5, "1/2", None])
def test_exact_scalar_inputs_refuse_nonrational_data(value: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        su2_heat_kernel_enclosure(value, 0)
    with pytest.raises((TypeError, ValueError)):
        su2_heat_kernel_enclosure(1, value)
    with pytest.raises((TypeError, ValueError)):
        normalized_product_tv_bound([[value, 1]])


@pytest.mark.parametrize("cutoff", [True, 2.0, -1, 257])
def test_invalid_cutoffs(cutoff: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        su2_heat_kernel_enclosure(1, 0, cutoff=cutoff)


def test_invalid_domains() -> None:
    for time in [0, -1]:
        with pytest.raises(ValueError):
            su2_heat_kernel_enclosure(time, 0)
    for x in [Q(-101, 100), Q(101, 100)]:
        with pytest.raises(ValueError):
            su2_heat_kernel_enclosure(1, x)
    for derivative in [-1, 3, True]:
        with pytest.raises((ValueError, TypeError)):
            su2_heat_kernel_enclosure(1, 0, derivative_order=derivative)
    for components in [[1, 0], [1, 1, 0, 0], [True, 0, 0, 0]]:
        with pytest.raises((ValueError, TypeError)):
            su2_heat_kernel_bridge(1, components, [1, 0, 0, 0])
    for factors in [[[0, 1]], [[2, 1]], [[1]], [[-1, 1]]]:
        with pytest.raises(ValueError):
            normalized_product_tv_bound(factors)


@pytest.mark.parametrize("value", [None, [], (), "certificate", 1, True])
def test_replay_nonmapping_guard(value: Any) -> None:
    assert not replay(value)


@pytest.mark.parametrize("mutation", ["tail", "scope", "input", "normalization", "honesty"])
def test_rehashed_certificate_tampering_rejected(mutation: str) -> None:
    cert = deepcopy(su2_heat_kernel_bridge(Q(1, 5), [1, 0, 0, 0], [-1, 0, 0, 0])["certificate"])
    w = cert["payload"]["witness"]
    if mutation == "tail":
        w["left_kernel"]["tail_upper"] = "0"
    elif mutation == "scope":
        cert["honesty"]["actual_ks_vacuum_verified"] = True
    elif mutation == "input":
        w["inputs"]["cutoff"] = True
    elif mutation == "normalization":
        w["normalization"] = "untruncated Wilson conditional density"
    else:
        cert["honesty"]["exact_bridge_normalization_verified"] = 1
    assert not replay(seal_certificate(cert))
