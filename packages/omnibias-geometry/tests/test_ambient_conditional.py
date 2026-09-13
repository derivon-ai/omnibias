# SPDX-License-Identifier: Apache-2.0
"""Exact local counts, positive-kernel comparison and symbolic conditional bounds."""

from collections.abc import Callable, Sequence
from copy import deepcopy
from fractions import Fraction as Q
from json import dumps, loads
from random import Random
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer import ambient_conditional as ambient

_WORDS = [[1, 2, -3, -4], [-1, 5, 6, -7], [8, 9, -10, -1], [1, -11, 12, 13]]


def _case(kappa: int | Q | str = Q(1, 64), **kwargs: Any) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "block_link_ids": [1],
        "plaquette_words": _WORDS,
        "n_links": 13,
    }
    inputs.update(kwargs)
    return ambient.su2_ambient_conditional_gap(kappa, **inputs)


def test_exact_default_grid_witness_and_scope() -> None:
    row = _case()
    a = row["arithmetic"]
    assert row["status"] == "PASS"
    assert (a["b"], a["p"]) == (1, 4)
    assert a["selected_heat_time_ratio"] == "4/7"
    assert a["selected_heat_time"] == "1/112"
    assert a["physical_semigroup_time"] == "8/7"
    assert a["negative_exponent"] == "63744/7"
    assert a["quantum_gap_prefactor"] == "3/4096"
    assert a["poincare_gap_prefactor"] == "3/32"
    assert a["dyadic_extra_exponent"] == 13660
    assert a["dyadic_total_exponent"] == 13663
    assert row["factored_dyadic_quantum_gap_lower"]["prefactor"] == "3/512"
    assert row["touched_plaquette_indices"] == [0, 1, 2, 3]
    assert row["certificate"]["payload"] == {k: v for k, v in row.items() if k != "certificate"}
    for flag in (
        "actual_positive_vacuum_identified_in_written_analysis",
        "actual_conditional_gap_verified_in_written_analysis",
        "ambient_exterior_uniformity_verified_in_written_analysis",
        "all_positive_couplings_covered_in_written_analysis",
        "all_scalar_block_modes_covered_in_written_analysis",
        "ambient_bound_depends_only_on_block_and_touching_count",
    ):
        assert row[flag] is True
    for flag in (
        "graph_cycle_geometry_verified",
        "local_gauge_invariance_verified",
        "physical_gauge_sector_application_verified",
        "frozen_bare_hamiltonian_gap_verified",
        "global_hamiltonian_gap_verified",
        "bulk_uniform_gap_verified",
        "continuum_claim",
        "yang_mills_claim",
        "yang_mills_mass_gap_claim",
        "source_gap_used_as_premise",
        "spin_truncation_used",
        "analytic_proof_formally_verified",
        "theorem_prover_verified",
        "mathlib_verified",
    ):
        assert row[flag] is False
    assert ambient.replay_su2_ambient_conditional_certificate(row["certificate"])


@pytest.mark.parametrize("kappa", [1, "2/9", Q(700), Q(1, 2**512)])
def test_all_couplings_stay_positive_without_materializing_exponential(
    kappa: int | Q | str,
) -> None:
    row = _case(kappa)
    a = row["arithmetic"]
    assert row["status"] == "PASS"
    assert Q(a["quantum_gap_prefactor"]) > 0
    assert Q(a["negative_exponent"]) > 0
    n = a["dyadic_extra_exponent"]
    target = 3 * Q(a["negative_exponent"]) / 2
    assert n - 1 < target <= n
    assert a["dyadic_total_exponent"] == 3 * a["b"] + n
    assert len(dumps(row)) < 30_000
    assert loads(dumps(row)) == row
    assert ambient.replay_su2_ambient_conditional_certificate(row["certificate"])


@pytest.mark.parametrize("time", [Q(1, 128), "1/17", 9])
def test_explicit_heat_time_and_units(time: int | Q | str) -> None:
    row = _case("1/64", heat_time=time)
    a = row["arithmetic"]
    kappa, s = Q(1, 64), Q(time)
    assert a["selected_heat_time"] == str(s)
    assert Q(a["negative_exponent"]) == 128 * s / kappa**2 + Q(1936, 49) / s
    assert Q(a["physical_semigroup_time"]) == 2 * s / kappa
    assert Q(a["quantum_gap_prefactor"]) == kappa * Q(a["poincare_gap_prefactor"]) / 2
    assert ambient.replay_su2_ambient_conditional_certificate(row["certificate"])


def test_rational_grid_minimum_random_local_counts() -> None:
    rng = Random(5061)
    grid = [Q(1, 4), Q(1, 2), Q(4, 7), Q(1), Q(2), Q(4)]
    for _ in range(50):
        b, p = rng.randrange(1, 12), rng.randrange(1, 30)
        kappa = Q(rng.randrange(1, 100), rng.randrange(1, 100))
        words = [[1, b + 1, -(b + 2), -(b + 3)]] * p
        row = _case(
            kappa, block_link_ids=list(range(1, b + 1)), plaquette_words=words, n_links=b + 3
        )
        values = [(32 * q * p + Q(1936 * b, 49) / q) / kappa for q in grid]
        a = row["arithmetic"]
        assert Q(a["negative_exponent"]) == min(values)
        assert Q(a["selected_heat_time_ratio"]) == grid[values.index(min(values))]


@pytest.mark.parametrize("time", [None, Q(1, 100), "99"])
def test_no_touching_plaquettes_exact_haar_bypass(time: int | Q | str | None) -> None:
    row = _case(
        7, block_link_ids=[6, 5], plaquette_words=[[1, -2, 3, -4]], n_links=6, heat_time=time
    )
    a = row["arithmetic"]
    assert a["p"] == 0 and a["b"] == 2
    assert a["selected_heat_time"] is None
    assert a["exact_haar_quantum_gap"] == "21/8"
    assert a["quantum_gap_prefactor"] == "21/8"
    assert a["poincare_gap_prefactor"] == "3/4"
    assert a["negative_exponent"] == "0"
    assert a["dyadic_total_exponent"] == 0
    assert row["exact_haar_conditional_gap_verified_in_written_analysis"]
    assert ambient.replay_su2_ambient_conditional_certificate(row["certificate"])
    empty = _case(7, block_link_ids=[1], plaquette_words=[], n_links=1)
    assert empty["quantum_gap_lower"] == row["quantum_gap_lower"]


def test_exterior_growth_changes_no_local_bound_and_multiplicity_is_not_deduplicated() -> None:
    initial = _case()
    extended = _case(n_links=1000, plaquette_words=_WORDS + [[21, 22, -23, -24]] * 17)
    assert initial["arithmetic"] == extended["arithmetic"]
    assert initial["quantum_gap_lower"] == extended["quantum_gap_lower"]
    assert (
        initial["factored_dyadic_quantum_gap_lower"]
        == extended["factored_dyadic_quantum_gap_lower"]
    )
    repeated = _case(plaquette_words=_WORDS + [_WORDS[0]])
    assert repeated["arithmetic"]["p"] == 5
    assert Q(repeated["arithmetic"]["negative_exponent"]) > Q(
        initial["arithmetic"]["negative_exponent"]
    )


def test_signed_repeated_words_and_block_permutation() -> None:
    words = [[-1, 2, 1, -2], [3, -3, 3, -3], [4, 5, -6, -7]]
    a = _case(block_link_ids=[2, 1], plaquette_words=words)
    b = _case(block_link_ids=[1, 2], plaquette_words=words)
    assert a == b
    assert a["arithmetic"]["p"] == 1
    assert not a["graph_cycle_geometry_verified"]


@pytest.mark.parametrize("value", [True, False, 0, -1, Q(-1, 9), 0.5, "nan", "0", "1/0", None])
def test_bad_couplings(value: object) -> None:
    call: Callable[..., object] = _case
    with pytest.raises((TypeError, ValueError)):
        call(value)


@pytest.mark.parametrize("value", [True, False, 0, -1, 0.2, "nan", "0", "1/0"])
def test_bad_heat_times_even_when_the_haar_branch_would_ignore_them(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _case(heat_time=value, plaquette_words=[])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_links": True},
        {"n_links": 0},
        {"n_links": 1.5},
        {"block_link_ids": []},
        {"block_link_ids": [1, 1]},
        {"block_link_ids": [-1]},
        {"block_link_ids": [0]},
        {"block_link_ids": [14]},
        {"block_link_ids": [True]},
        {"block_link_ids": "1"},
        {"plaquette_words": "1234"},
        {"plaquette_words": [[1, 2, 3]]},
        {"plaquette_words": [[1, 2, 3, 4, 5]]},
        {"plaquette_words": [[0, 2, 3, 4]]},
        {"plaquette_words": [[1, 2, 3, -14]]},
        {"plaquette_words": [[True, 2, 3, 4]]},
        {"plaquette_words": ["1234"]},
    ],
)
def test_graph_input_guards(kwargs: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        _case(**kwargs)


@pytest.mark.parametrize(
    "damage", ["count", "time", "exponent", "dyadic", "scope", "status", "words", "meta"]
)
def test_rehashed_forgery_rejected(damage: str) -> None:
    cert = deepcopy(_case()["certificate"])
    payload = cert["payload"]
    if damage == "count":
        payload["arithmetic"]["p"] = 0
    elif damage == "time":
        payload["arithmetic"]["selected_heat_time"] = "1"
    elif damage == "exponent":
        payload["quantum_gap_lower"]["negative_exponent"] = "0"
    elif damage == "dyadic":
        payload["factored_dyadic_quantum_gap_lower"]["negative_integer_exponent"] -= 1
    elif damage == "scope":
        payload["global_hamiltonian_gap_verified"] = True
    elif damage == "status":
        payload["status"] = "INCONCLUSIVE"
    elif damage == "words":
        payload["inputs"]["plaquette_words"].pop()
    else:
        cert["meta"]["proof_register"] = "full operator formally proved"
    forged = seal_certificate(cert)
    assert verify_certificate_digest(forged)
    assert not ambient.replay_su2_ambient_conditional_certificate(forged)


@pytest.mark.parametrize("value", [{}, {"payload": {}}, None, [], "bad"])
def test_malformed_replay(value: object) -> None:
    call: Callable[..., bool] = ambient.replay_su2_ambient_conditional_certificate
    assert not call(value)


def _dot(a: Sequence[Q], b: Sequence[Q]) -> Q:
    return sum((x * y for x, y in zip(a, b, strict=True)), Q(0))


def test_common_positive_fiber_cancels_in_kernel_ratio_exactly() -> None:
    """Finite positive-kernel algebra behind FK; not a numerical FK proof."""
    rng = Random(532)
    for _ in range(100):
        kernel = [[Q(rng.randrange(1, 30), 31) for _ in range(5)] for _ in range(3)]
        common_fiber = [Q(rng.randrange(1, 100), 17) for _ in range(5)]
        killing_lower = Q(rng.randrange(1, 20), 20)
        for x in range(3):
            for y in range(3):
                ratio = max(kernel[x][z] / kernel[y][z] for z in range(5))
                upper_x = _dot(kernel[x], common_fiber)
                lower_y = killing_lower * _dot(kernel[y], common_fiber)
                assert upper_x <= ratio * lower_y / killing_lower


def test_weighted_variance_comparison_independent_finite_grid() -> None:
    """The exact min/max-density transport is tested without a gap oracle."""
    rng = Random(342)
    for _ in range(100):
        raw = [Q(rng.randrange(1, 20)) for _ in range(5)]
        density = [5 * p / sum(raw) for p in raw]
        f = [Q(rng.randrange(-10, 11), 3) for _ in range(5)]
        haar_mean = sum(f) / 5
        weighted_mean = _dot(f, density) / 5
        var_haar = sum((v - haar_mean) ** 2 for v in f) / 5
        var_weighted = sum(density[i] * (f[i] - weighted_mean) ** 2 for i in range(5)) / 5
        assert var_weighted <= max(density) * var_haar
        assert var_weighted * min(density) / max(density) <= min(density) * var_haar
