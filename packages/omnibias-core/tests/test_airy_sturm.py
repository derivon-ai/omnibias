# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent analytic transfer diagnostics and exact Sturm-source attacks."""

from copy import deepcopy
from fractions import Fraction as Q
from random import Random
from typing import Any

import mpmath as mp
import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.core.verified.airy_sturm import (
    _Interval,
    _shoot,
    _transfer,
    airy_half_line_lower_bounds,
    replay_airy_sturm_certificate,
)


@pytest.fixture(scope="module")
def source() -> dict[str, Any]:
    return airy_half_line_lower_bounds()


def _mp(q: Q | str) -> Any:
    q = Q(q)
    return mp.mpf(q.numerator) / q.denominator


def _inside(value: Any, interval: list[str]) -> bool:
    return bool(_mp(interval[0]) <= value <= _mp(interval[1]))


def _reseal(certificate: dict[str, Any]) -> dict[str, Any]:
    body = deepcopy(certificate)
    body.pop("digest", None)
    return seal_certificate(body)


def test_full_finite_counts_and_half_line_source(source: dict[str, Any]) -> None:
    assert source["status"] == "PASS"
    assert source["half_line_eigenvalue_lower_bounds_verified_in_written_analysis"]
    assert [row["zero_cells"] for row in source["shooting"]] == [[], [56]]
    assert [row["dirichlet_neumann_count_below_energy"] for row in source["shooting"]] == [0, 1]
    assert all(len(row["cells"]) == 192 for row in source["shooting"])
    assert replay_airy_sturm_certificate(source["certificate"])


def test_all_endpoint_intervals_against_independent_hyperbolic_transfer(
    source: dict[str, Any],
) -> None:
    # High-precision values are independent diagnostics; the certificate
    # itself uses rational series and proved tails, never these evaluations.
    with mp.workdps(110):
        h = mp.mpf(1) / 32
        for shot in source["shooting"]:
            u, p = mp.mpf(0), mp.mpf(1)
            for row in shot["cells"]:
                q = _mp(row["q"])
                if q > 0:
                    w = mp.sqrt(q)
                    c, s = mp.cosh(w * h), mp.sinh(w * h) / w
                elif q < 0:
                    w = mp.sqrt(-q)
                    c, s = mp.cos(w * h), mp.sin(w * h) / w
                else:
                    c, s = mp.mpf(1), h
                assert _inside(c, row["C"])
                assert _inside(s, row["S"])
                u, p = c * u + s * p, q * s * u + c * p
                assert _inside(u, row["u_right"])
                assert _inside(p, row["derivative_right"])


def test_transfer_deterministic_grid_and_seeded_rationals() -> None:
    rng = Random(17293)
    samples = [(Q(q, 2), Q(1, d)) for q in range(-8, 13) for d in (16, 32, 64)]
    samples += [(Q(rng.randrange(-400, 601), 100), Q(1, rng.randrange(16, 65))) for _ in range(24)]
    with mp.workdps(110):
        for q, h in samples:
            c, s, ec, es = _transfer(q, h, 6, 128)
            qm, hm = _mp(q), _mp(h)
            if q:
                root = mp.sqrt(abs(qm))
                c_true = mp.cos(root * hm) if q < 0 else mp.cosh(root * hm)
                s_true = (mp.sin(root * hm) if q < 0 else mp.sinh(root * hm)) / root
            else:
                c_true, s_true = mp.mpf(1), hm
            assert _inside(c_true, c.serialized())
            assert _inside(s_true, s.serialized())
            assert ec >= 0 and es >= 0


@pytest.mark.parametrize("value", [Q(1, 3), Q(-1, 3), Q(2**100 + 1, 3), Q(1, 2**250)])
@pytest.mark.parametrize("bits", [16, 64, 128])
def test_dyadic_rounding_is_exact_even_outside_binary64(value: Q, bits: int) -> None:
    rounded = _Interval(value, value).rounded(bits)
    assert rounded.lo <= value <= rounded.hi
    assert rounded.hi - rounded.lo <= Q(1, 2**bits)
    assert (rounded.lo * 2**bits).denominator == 1
    assert (rounded.hi * 2**bits).denominator == 1


def test_zero_frequency_transfer_is_exact() -> None:
    c, s, ec, es = _transfer(Q(0), Q(1, 32), 6, 128)
    assert c == _Interval(Q(1), Q(1))
    assert s == _Interval(Q(1, 32), Q(1, 32))
    assert ec == es == 0


def test_airy_and_bessel_free_independent_zero_diagnostics(source: dict[str, Any]) -> None:
    with mp.workdps(80):
        assert -mp.airyaizero(1) > _mp(source["shooting"][0]["energy"])
        assert -mp.airyaizero(2) > _mp(source["shooting"][1]["energy"])


def test_threshold_count_is_computed_not_assigned() -> None:
    failed = _shoot(Q(5), 0, 192, 6, 128)
    assert failed["finite_sturm_count_verified"]
    assert failed["dirichlet_neumann_count_below_energy"] == 2
    assert not failed["target_half_line_lower_bound_verified"]


def test_coarse_staircase_does_not_promote_inadequate_count() -> None:
    coarse = airy_half_line_lower_bounds(cells=12, series_order=1, rounding_bits=32)
    assert coarse["status"] == "INCONCLUSIVE"
    assert not coarse["half_line_eigenvalue_lower_bounds_verified_in_written_analysis"]
    assert replay_airy_sturm_certificate(coarse["certificate"])


def test_refined_rational_mesh_also_passes() -> None:
    refined = airy_half_line_lower_bounds(cells=384, series_order=5, rounding_bits=96)
    assert refined["status"] == "PASS"
    assert [row["dirichlet_neumann_count_below_energy"] for row in refined["shooting"]] == [0, 1]


@pytest.mark.parametrize("key", ["cells", "series_order", "rounding_bits"])
@pytest.mark.parametrize("value", [True, False, 192.0, "192", Q(192)])
def test_strict_integer_inputs(key: str, value: Any) -> None:
    with pytest.raises(TypeError):
        airy_half_line_lower_bounds(**{key: value})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cells": 11},
        {"cells": 4097},
        {"series_order": 0},
        {"series_order": 33},
        {"rounding_bits": 15},
        {"rounding_bits": 1025},
    ],
)
def test_resource_and_remainder_domains(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        airy_half_line_lower_bounds(**kwargs)


@pytest.mark.parametrize(
    "case", ["node", "derivative", "tail", "potential", "step", "exterior", "scope", "input_bool"]
)
def test_resealed_attack_is_rejected(source: dict[str, Any], case: str) -> None:
    forged = deepcopy(source["certificate"])
    payload = forged["payload"]
    if case == "node":
        payload["shooting"][1]["zero_cells"] = []
    elif case == "derivative":
        payload["shooting"][1]["cells"][56]["derivative_right"] = ["0", "0"]
    elif case == "tail":
        payload["shooting"][0]["cells"][0]["C_remainder_upper"] = "0"
    elif case == "potential":
        payload["shooting"][0]["cells"][1]["potential_lower"] = "1"
    elif case == "step":
        payload["arithmetic"]["step"] = "1/31"
    elif case == "exterior":
        payload["shooting"][0]["neumann_exterior_potential_floor"] = "7"
    elif case == "scope":
        payload["yang_mills_mass_gap_claim"] = True
    else:
        payload["inputs"]["series_order"] = True
    forged = _reseal(forged)
    assert verify_certificate_digest(forged)
    assert not replay_airy_sturm_certificate(forged)


@pytest.mark.parametrize("bad", [None, [], (), "certificate", 0, True, {}])
def test_malformed_certificates_return_false(bad: Any) -> None:
    assert not replay_airy_sturm_certificate(bad)


def test_certificate_has_no_floating_leaves(source: dict[str, Any]) -> None:
    def visit(value: Any) -> None:
        assert not isinstance(value, float)
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(source)
    assert source["certificate"]["meta"]["transcend_backend"] == "not_used"
    assert not source["theorem_prover_verified"]
    assert not source["mathlib_verified"]
