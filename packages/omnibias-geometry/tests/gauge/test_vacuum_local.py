# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Regression and independent containment checks for actual-vacuum bounds."""
import random
from copy import deepcopy
from fractions import Fraction

import mpmath
import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer.vacuum_local import (
    replay_su2_vacuum_local_certificate,
    su2_vacuum_local_bounds,
)

SQUARE = [(0, 1), (1, 2), (2, 3), (3, 0)]


def square(**kwargs):
    return su2_vacuum_local_bounds(4, SQUARE, plaquettes=[(1, 2, 3, 4)], **kwargs)


def test_haar_vacuum_exact_local_constant_and_full_gap():
    report = su2_vacuum_local_bounds(2, [(0, 1)], kappa=7)
    w = report["witness"]
    assert w["singleton_comparison_floor_enclosures"] == [["3/4", "3/4"]]
    assert w["log_vacuum_gradient_upper"] == ["0"]
    assert Fraction(w["finite_graph_gap_lower"]) == Fraction(21, 8)
    assert report["finite_graph_gap_verified"]
    assert replay_su2_vacuum_local_certificate(report["certificate"])


def test_local_pass_does_not_promote_failed_global_gate():
    report = square(kappa=1)
    assert report["status"] == "PASS"
    assert not report["finite_graph_gap_verified"]
    assert Fraction(report["witness"]["dense_influence_q_upper"]) > 1
    assert report["witness"]["finite_graph_gap_lower"] == "0"
    for name in ("theorem_prover_verified", "mathlib_verified", "uniform_in_a_claim",
                 "infinite_volume_claim", "volume_uniform_neutral_gap_claim",
                 "yang_mills_claim", "continuum_claim"):
        assert report[name] is False


def test_weighted_oriented_cycles_and_block_sum():
    report = su2_vacuum_local_bounds(
        4, SQUARE, kappa=8, plaquettes=[(-4, -3, -2, -1)],
        magnetic_weights=[Fraction(3, 2)], blocks=[(1, 3)])
    w = report["witness"]
    assert w["weighted_incidence"] == ["3/2"] * 4
    assert w["block_incidence"] == ["3"]
    assert w["log_vacuum_gradient_upper"] == ["3/16"] * 4
    assert Fraction(w["block_comparison_floor_enclosures"][0][1]) < Fraction(
        w["singleton_comparison_floor_enclosures"][0][0])


def test_independent_high_precision_dense_and_random_containment():
    # These enclosures contain the explicit analytic comparison floor, not
    # an estimated conditional eigenvalue. The true formula uses mp.pi.
    rng = random.Random(419)
    parameters = [(Fraction(i, 4), Fraction(j, 4))
                  for i in range(1, 41) for j in range(1, 5)]
    parameters += [(Fraction(rng.randrange(1, 1000), 31),
                    Fraction(rng.randrange(1, 100), 29)) for _ in range(80)]
    with mpmath.workdps(90):
        for kappa, degree in parameters:
            w = square(kappa=kappa, magnetic_weights=[degree])["witness"]
            pair = list(map(Fraction, w["singleton_comparison_floor_enclosures"][0]))
            k = mpmath.mpf(kappa.numerator) / kappa.denominator
            d = mpmath.mpf(degree.numerator) / degree.denominator
            true = mpmath.mpf(3) / 4 * mpmath.exp(-32 * mpmath.pi * d / k**2)
            lo = mpmath.mpf(pair[0].numerator) / pair[0].denominator
            hi = mpmath.mpf(pair[1].numerator) / pair[1].denominator
            assert lo <= true <= hi


def test_underflow_is_inconclusive_and_replays():
    report = square(kappa=Fraction(1, 10))
    assert report["status"] == "INCONCLUSIVE"
    assert not report["local_comparison_floors_positive"]
    assert not report["finite_graph_gap_verified"]
    assert replay_su2_vacuum_local_certificate(report["certificate"])


def test_actual_dense_gate_has_correct_independent_scale():
    report = square(kappa=32)
    w = report["witness"]
    assert report["finite_graph_gap_verified"]
    q = Fraction(w["dense_influence_q_upper"])
    assert 0.147 < float(q) < 0.148  # 4*pi*(4/32**2)*3
    gamma = Fraction(w["singleton_comparison_floor_enclosures"][0][0])
    assert Fraction(w["finite_graph_gap_lower"]) == 16 * gamma * (1 - q)


@pytest.mark.parametrize("field,value", [
    ("finite_graph_gap_lower", "1000000"),
    ("weighted_incidence", ["0"] * 4),
    ("comparison_floor_is_actual_gap_enclosure", True),
    ("volume_uniform_global_factorization_verified", True),
    ("continuum_claim", True), ("yang_mills_claim", True),
])
def test_rehashed_false_witness_is_refused(field, value):
    certificate = deepcopy(square(kappa=16)["certificate"])
    certificate["payload"]["witness"][field] = value
    certificate.pop("digest")
    assert not replay_su2_vacuum_local_certificate(seal_certificate(certificate))


@pytest.mark.parametrize("kwargs", [
    {"kappa": True}, {"kappa": 1.0}, {"kappa": 0},
    {"blocks": [()]}, {"blocks": [(1, 1)]}, {"blocks": [(0,)]},
    {"blocks": [(5,)]}, {"blocks": [(True,)]},
    {"magnetic_weights": [-1]}, {"magnetic_weights": []},
])
def test_invalid_parameters_are_refused(kwargs):
    with pytest.raises((ValueError, TypeError)):
        square(**kwargs)


def test_repeated_edge_cycle_is_not_licensed_by_gradient_proof():
    with pytest.raises(ValueError, match="distinct"):
        su2_vacuum_local_bounds(4, SQUARE, plaquettes=[(1, 2, 3, 4, 1, 2, 3, 4)])


@pytest.mark.parametrize("value", [None, [], {}, {"payload": []}])
def test_malformed_replay_returns_false(value):
    assert not replay_su2_vacuum_local_certificate(value)
