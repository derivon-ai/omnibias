# SPDX-License-Identifier: Apache-2.0
"""Independent radial, form-domain, all-tail and canonical replay regressions."""

from __future__ import annotations

import math
import random
from copy import deepcopy
from fractions import Fraction as Q
from itertools import pairwise
from typing import Any

import numpy as np
import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer.weak_plaquette import (
    _sturm_count,
)
from omnibias.geometry.gauge.transfer.weak_plaquette import (
    replay_su2_weak_plaquette_gap_certificate as replay,
)
from omnibias.geometry.gauge.transfer.weak_plaquette import su2_weak_plaquette_gap as certify
from scipy.linalg import eigh_tridiagonal  # type: ignore[import-untyped]
from scipy.special import mathieu_b  # type: ignore[import-untyped]


@pytest.mark.parametrize("n", range(9))
def test_independent_chebyshev_radial_operator_keeps_four_electric_edges(n: int) -> None:
    # U_n(x/2) has this integer-coefficient closed form.
    coefficients = {n - 2 * k: (-1) ** k * math.comb(n - k, k) for k in range(n // 2 + 1)}
    actual: dict[int, int] = {}
    for power, value in coefficients.items():
        actual[power] = actual.get(power, 0) + (power * (power - 1) + 3 * power) * value
        if power >= 2:
            actual[power - 2] = actual.get(power - 2, 0) - 4 * power * (power - 1) * value
    actual = {p: c for p, c in actual.items() if c}
    expected = {p: n * (n + 2) * c for p, c in coefficients.items() if n * (n + 2) * c}
    assert actual == expected


def test_wkb_eikonal_and_local_energy_exactly_include_antipodal_failure() -> None:
    # y=sqrt(2+x) rationalizes every derivative and avoids evaluating a
    # guessed floating square root. Include a grid and seeded extra points.
    rng = random.Random(73719)
    ys = [Q(k, 20) for k in range(1, 41)]
    ys += [Q(rng.randint(1, 2000), 1000) for _ in range(30)]
    for y in ys:
        x = y * y - 2
        first, second = -2 / y, 1 / y**3
        assert (4 - x * x) * first**2 == 4 * (2 - x)
        local = ((4 - x * x) * second - 3 * x * first) / 2
        assert local == (5 * y * y - 8) / (2 * y)
        assert 3 - local == (2 - y) * (5 * y + 4) / (2 * y) >= 0
    # The cusp is admissible for a Rayleigh estimate, but it absolutely
    # cannot supply a bounded residual-oscillation comparison.
    y = Q(1, 10**8)
    assert (5 * y * y - 8) / (2 * y) < -(10**8)


@pytest.mark.parametrize(
    "coupling", [Q(1, 10**12), Q(1, 100), Q(1), Q(4, 3), Q(7, 4), Q(2), Q(10**12)]
)
def test_all_coupling_analytic_bound_never_uses_a_strong_coupling_gate(coupling: Q) -> None:
    result = certify(coupling)
    assert result["status"] == "PASS"
    assert Q(result["gap_lower"]) >= Q(26, 33)
    assert result["all_positive_couplings_gap_lower"] == "26/33"
    assert result["actual_plaquette_gap_verified"]
    assert result["all_irreducible_characters_included"]
    assert not result["full_unconstrained_rotor_claim"]
    assert not result["full_omitted_spin_comparison_verified"]
    assert replay(result["certificate"])


def test_middle_interval_factorization_proves_uniform_bound_without_sampling() -> None:
    # Coefficients after multiplying the claimed identity by6*kappa agree
    # exactly: 6k*(49/11-k/2-4/k-26/33)=-3k^2+22k-24.
    expanded = (Q(-24), 6 * (Q(49, 11) - Q(26, 33)), Q(-3))
    factored = (Q(-4) * 6, Q(3) * 6 + 4, Q(-3))
    assert expanded == factored
    assert certify(Q(4, 3))["gap_lower"] == "26/33"


def test_rational_pi_upper_bound_has_an_exact_positive_integrand_identity() -> None:
    quotient = {0: 4, 2: -4, 4: 5, 5: -4, 6: 1}
    product: dict[int, int] = {0: -4}
    for power, coefficient in quotient.items():
        for shift in (0, 2):
            product[power + shift] = product.get(power + shift, 0) + coefficient
    assert {p: c for p, c in product.items() if c} == {
        4: 1,
        5: -4,
        6: 6,
        7: -4,
        8: 1,
    }
    # Integrating the remainder term gives -4*atan(1)=-pi; the quotient
    # integrates to22/7. The residual integrand is x^4*(1-x)^4/(1+x^2)>0.
    assert sum((Q(c, p + 1) for p, c in quotient.items()), Q(0)) == Q(22, 7)


@pytest.mark.parametrize(
    "size, energy, count", [(2, -1, 0), (2, 0, 1), (2, 1, 1), (3, 0, 1), (3, 2, 3)]
)
def test_sturm_zeros_count_strict_eigenvalue_order_exactly(
    size: int, energy: int, count: int
) -> None:
    # Size2 spectrum {-1,1}; size3 spectrum {-sqrt2,0,sqrt2}. Both endpoint
    # eigenvalues and zero intermediate leading minors are exercised.
    assert _sturm_count((0,) * size, 1, 1, Q(energy)) == count


def test_interface_square_identity_and_full_tail_electric_floor_are_exact() -> None:
    rng = random.Random(64241)
    for coupling in (Q(1, 100), Q(1), Q(7, 2)):
        c = 2 / coupling
        for cutoff in (1, 3, 7):
            for _ in range(5):
                vector = [Q(rng.randint(-7, 7), rng.randint(1, 5)) for _ in range(cutoff + 6)]
                electric = sum(
                    (coupling * n * (n + 2) * x * x / 2 for n, x in enumerate(vector)), Q(0)
                )
                matrix_form = electric + 2 * c * sum((x * x for x in vector), Q(0))
                matrix_form -= 2 * c * sum((a * b for a, b in pairwise(vector)), Q(0))
                differences = [*vector[1:], Q(0)]
                square_form = electric + c * vector[0] ** 2
                square_form += c * sum(
                    ((b - a) ** 2 for a, b in zip(vector, differences, strict=True)), Q(0)
                )
                assert matrix_form == square_form
                lower_retained = sum(
                    (coupling * n * (n + 2) * vector[n] ** 2 / 2 for n in range(cutoff + 1)), Q(0)
                )
                lower_retained += c * vector[0] ** 2
                lower_retained += c * sum(
                    ((vector[n + 1] - vector[n]) ** 2 for n in range(cutoff)), Q(0)
                )
                tail_floor = coupling * (cutoff + 1) * (cutoff + 3) / 2
                comparison = lower_retained + tail_floor * sum(
                    (x * x for x in vector[cutoff + 1 :]), Q(0)
                )
                bridge = c * (vector[cutoff + 1] - vector[cutoff]) ** 2
                assert matrix_form - comparison >= bridge >= 0


@pytest.mark.parametrize(
    "coupling, cutoff", [(Q(1, 10), 32), (Q(1, 100), 80), (Q(1), 16), (Q(10), 8)]
)
def test_all_spin_brackets_contain_independent_mathieu_and_larger_jacobi_diagnostics(
    coupling: Q, cutoff: int
) -> None:
    result = certify(coupling, cutoff=cutoff, bisection_steps=28)
    witness = result["witness"]["jacobi_brackets"]
    k = float(coupling)
    n = np.arange(cutoff + 81, dtype=float)
    values = eigh_tridiagonal(
        k * n * (n + 2) / 2 + 4 / k, np.full(len(n) - 1, -2 / k), select="i", select_range=(0, 1)
    )[0]
    for index, (field, approximate) in enumerate(
        zip(
            ("actual_ground_energy_enclosure", "actual_first_excited_energy_enclosure"),
            values,
            strict=True,
        )
    ):
        lo, hi = map(Q, witness[field])
        assert float(lo) <= approximate <= float(hi)
        if coupling >= Q(1, 10):
            # Large-q Mathieu routines can return the wrong characteristic
            # branch (observed already at q=160000). Restrict this optional
            # floating diagnostic; the independent larger Jacobi check above
            # and the exact certificate still cover the smaller coupling.
            special = k * (float(mathieu_b(2 * index + 2, 16 / (k * k))) - 4) / 8 + 4 / k
            assert float(lo) <= special <= float(hi)
    assert Q(result["gap_lower"]) <= Q(result["gap_upper"])
    assert result["full_omitted_spin_comparison_verified"]
    assert replay(result["certificate"])


def test_small_cutoff_stays_a_sound_wide_enclosure_and_keeps_universal_gap() -> None:
    result = certify(Q(1, 100), cutoff=1)
    tail = result["witness"]["jacobi_brackets"]["omitted_electric_floor"]
    assert Q(tail) == Q(1, 25)
    assert Q(result["gap_lower"]) >= Q(26, 33)
    assert Q(result["gap_upper"]) > 100
    assert replay(result["certificate"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kappa": True},
        {"kappa": 1.0},
        {"kappa": "1"},
        {"kappa": 0},
        {"kappa": -1},
        {"kappa": 1, "cutoff": True},
        {"kappa": 1, "cutoff": 2.0},
        {"kappa": 1, "cutoff": 0},
        {"kappa": 1, "bisection_steps": True},
        {"kappa": 1, "bisection_steps": 1.0},
        {"kappa": 1, "bisection_steps": 0},
    ],
)
def test_only_exact_positive_parameters_are_accepted(kwargs: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        certify(**kwargs)


@pytest.mark.parametrize(
    "field, replacement",
    [
        ("gap_lower", "100"),
        ("wkb_local_energy", "bounded everywhere"),
        ("normalization", "one electric edge"),
        ("hilbert_space", "full L2 SU2"),
        ("all_couplings_proof", "sampled coupling grid"),
    ],
)
def test_resealed_witness_tampering_cannot_change_analytic_scope(
    field: str, replacement: str
) -> None:
    changed = deepcopy(certify(Q(1, 10), cutoff=16)["certificate"])
    changed["payload"]["witness"][field] = replacement
    assert not replay(seal_certificate(changed))


@pytest.mark.parametrize(
    "field", ["omitted_electric_floor", "actual_gap_enclosure", "neumann_compression"]
)
def test_resealed_tail_or_compression_tampering_is_rejected(field: str) -> None:
    changed = deepcopy(certify(Q(1, 10), cutoff=16)["certificate"])
    changed["payload"]["witness"]["jacobi_brackets"][field] = "invented"
    assert not replay(seal_certificate(changed))


@pytest.mark.parametrize(
    "field", ["continuum_claim", "volume_uniform_claim", "yang_mills_mass_gap_claim"]
)
def test_parent_flags_cannot_be_earned_by_resealing(field: str) -> None:
    changed = deepcopy(certify(1)["certificate"])
    changed["honesty"][field] = True
    assert not replay(seal_certificate(changed))


@pytest.mark.parametrize("value", [None, [], (), "certificate", 1, True])
def test_replay_malformed_top_level_returns_false(value: Any) -> None:
    assert not replay(value)
