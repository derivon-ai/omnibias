# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Production trace enclosures, exact weak-block balance and tamper refusal."""

from copy import deepcopy
from fractions import Fraction as Q
from typing import Any

import mpmath as mp  # type: ignore[import-untyped]
import numpy as np
import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer.nonabelian_small_field import (
    replay_su2_nonabelian_small_field_certificate as replay,
)
from omnibias.geometry.gauge.transfer.nonabelian_small_field import (
    su2_plaquette_trace_jet as trace_jet,
)
from omnibias.geometry.gauge.transfer.nonabelian_small_field import (
    su2_wilson_block_small_field as block,
)
from omnibias.geometry.gauge.transfer.nonabelian_small_field import (
    su2_wilson_small_field_budget as budget,
)


def _mp(x: Q) -> Any:
    return mp.mpf(x.numerator)/x.denominator


def _trace(vectors: list[list[Q]], t: Q) -> Any:
    """Independent complex Pauli matrices and trigonometric exponential."""
    product = mp.eye(2)
    for a in vectors:
        x, y, z = [_mp(v) for v in a]
        r = mp.sqrt(x*x+y*y+z*z)
        if r:
            matrix = mp.matrix([[z, x-1j*y], [x+1j*y, -z]])
            product = product*(mp.cos(_mp(t)*r/2)*mp.eye(2)
                               + 1j*mp.sin(_mp(t)*r/2)/r*matrix)
    return mp.re(product[0, 0]+product[1, 1])


def test_ordered_cubic_coefficient_and_quartic_commutator() -> None:
    row = trace_jet([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
    assert row["witness"]["arithmetic"]["coefficients"] == ["2", "0", "-3/4", "1/4"]
    opposite = trace_jet([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 0]])
    assert opposite["witness"]["arithmetic"]["coefficients"][-1] == "-1/4"
    loop = trace_jet([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], degree=4)
    assert loop["witness"]["arithmetic"]["coefficients"] == ["2", "0", "0", "0", "-1/4"]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("degree", [0, 3, 8, 15])
def test_real_noncommuting_enclosure_on_grid_and_random_sample(seed: int, degree: int) -> None:
    rng = np.random.default_rng(seed)
    a = [[Q(int(x), 10) for x in row] for row in rng.integers(-5, 6, size=(4, 3))]
    row = trace_jet(a, degree=degree, parameter_radius=Q(3, 2))
    assert replay(row["certificate"])
    arithmetic = row["witness"]["arithmetic"]
    coefficients = [Q(x) for x in arithmetic["coefficients"]]
    error = Q(arithmetic["uniform_remainder_upper"])
    times = [Q(i, 20) for i in range(-30, 31)]
    times.extend(Q(int(i), 1000) for i in rng.integers(-1500, 1501, size=30))
    with mp.workdps(80):
        for t in times:
            polynomial = sum((c*t**n for n, c in enumerate(coefficients)), Q(0))
            assert abs(_trace(a, t)-_mp(polynomial)) <= _mp(error)+mp.mpf("1e-70")
        value = _trace(a, Q(3, 2))
        assert _mp(Q(arithmetic["endpoint_trace_lower"])) <= value
        assert value <= _mp(Q(arithmetic["endpoint_trace_upper"]))


def test_checked_euclidean_norms_tighten_l1_majorant() -> None:
    a = [[Q(3, 5), Q(4, 5), Q(0)]]*4
    loose = trace_jet(a)
    tight = trace_jet(a, norm_bounds=[1]*4)
    assert Q(tight["witness"]["arithmetic"]["uniform_remainder_upper"]) == Q(4, 3)
    assert Q(loose["witness"]["arithmetic"]["uniform_remainder_upper"]) > Q(4, 3)
    with pytest.raises(ValueError):
        trace_jet(a, norm_bounds=[Q(9, 10)]*4)


@pytest.mark.parametrize("b", [1, 2, 7])
@pytest.mark.parametrize("t", [Q(1, 16), Q(1, 4096), Q(1, 2**24)])
def test_exact_actual_vacuum_fixed_block_weak_scaling(b: int, t: Q) -> None:
    row = block(t**5, t**2, block_side=b)
    assert replay(row["certificate"])
    arithmetic = row["witness"]["arithmetic"]
    c = Q(8712, 49)*b**4*(b+1)
    assert Q(arithmetic["uncapped_weak_probability_upper"]) == c*t
    assert Q(arithmetic["bad_block_probability_upper"]) == min(1, c*t)
    assert Q(arithmetic["good_localized_vacuum_norm_squared_lower"]) == max(0, 1-2*c*t)
    cost_coefficient = Q(arithmetic["weak_localization_cost_coefficient"])
    assert Q(arithmetic["weak_localization_cost_upper"]) == cost_coefficient*t**2
    assert Q(arithmetic["total_vacuum_localization_form_cost_upper"]) <= cost_coefficient*t**2
    local = row["witness"]["local_chart_certificate"]["payload"]["witness"]["arithmetic"]
    assert Q(local["magnetic_cubic_absolute_upper_per_plaquette"]) == 2*t
    assert Q(local["magnetic_remainder_after_cubic_upper_per_plaquette"]) == Q(8, 3)*t**3
    assert Q(local["electric_derivative_relative_error_upper"]) == (1-t**4/24)**-2-1
    assert row["actual_block_patch_probability_bound_verified"]
    assert not row["physical_gauge_fixed_electric_metric_verified"]
    assert not row["conditional_large_field_bound_verified"]
    assert row["actual_gauge_invariant_vacuum_localization_verified"]
    assert row["all_state_localization_operator_bound_verified"]
    threshold = Q(arithmetic["plaquette_threshold"])
    assert Q(arithmetic["universal_single_block_ims_error_upper"]) == Q(3872, 49)*t**5/threshold
    assert Q(arithmetic["bad_support_local_magnetic_potential_lower"]) == threshold/t**5


@pytest.mark.parametrize("eps", [Q(0), Q(1, 100), Q(1), Q(2)])
def test_electric_constant_and_commutator_lower(eps: Q) -> None:
    row = budget(Q(3, 7), eps)
    assert replay(row["certificate"])
    arithmetic = row["witness"]["arithmetic"]
    assert arithmetic["electric_constant_per_edge"] == "-3/56"
    lower = Q(arithmetic["orthogonal_commutator_action_lower"])
    with mp.workdps(80):
        assert _mp(lower) <= 4*mp.sin(_mp(eps)/2)**4
    assert (lower > 0) == (eps > 0)


@pytest.mark.parametrize("bad", [0.5, True, "1", None])
def test_exact_input_contract(bad: Any) -> None:
    with pytest.raises(TypeError):
        budget(bad, 1)
    with pytest.raises(TypeError):
        block(1, 1, block_side=bad)
    with pytest.raises(TypeError):
        trace_jet([[bad, 0, 0]]*4)


@pytest.mark.parametrize("k,e", [(0, 1), (-1, 1), (1, -1), (1, 3)])
def test_invalid_chart_refused(k: int, e: int) -> None:
    with pytest.raises(ValueError):
        budget(k, e)


def test_zero_chart_cannot_be_probabilistic_block() -> None:
    with pytest.raises(ValueError):
        block(1, 0)
    with pytest.raises(ValueError):
        block(1, 1, block_side=0)
    assert budget(1, 0)["status"] == "PASS"


@pytest.mark.parametrize("flag", ["continuum_claim", "yang_mills_mass_gap_claim",
                                 "physical_gauge_fixed_electric_metric_verified",
                                 "conditional_large_field_bound_verified", "theorem_prover_verified"])
def test_rehashed_promotions_rejected(flag: str) -> None:
    for row in (trace_jet([[0, 0, 0]]*4), budget(1, 1), block(1, 1)):
        cert = deepcopy(row["certificate"])
        cert["honesty"][flag] = True
        assert not replay(seal_certificate(cert))


def test_nested_actual_vacuum_certificate_and_claim_tamper_refused() -> None:
    cert = deepcopy(block(Q(1, 100), 1)["certificate"])
    cert["payload"]["witness"]["actual_vacuum_moment_certificate"]["claim"] = "invented stronger claim"
    assert not replay(seal_certificate(cert))
    cert = deepcopy(trace_jet([[1, 0, 0]]*4)["certificate"])
    cert["payload"]["witness"]["arithmetic"]["coefficients"][2] = "0"
    assert not replay(seal_certificate(cert))


@pytest.mark.parametrize("bad", [None, [], {}, {"payload": {"type": "wrong"}}])
def test_malformed_replay_refused(bad: Any) -> None:
    assert not replay(bad)
