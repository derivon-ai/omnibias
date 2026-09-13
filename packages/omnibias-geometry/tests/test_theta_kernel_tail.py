# SPDX-License-Identifier: Apache-2.0
"""Actual-kernel source gates, independent barrier derivatives and tail geometry."""

from collections.abc import Sequence
from copy import deepcopy
from fractions import Fraction as Q
from importlib import import_module
from random import Random
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer import theta_kernel_tail as module
from omnibias.geometry.gauge.transfer import theta_weak_blocks as source

Quat = tuple[Q, Q, Q, Q]
Word = tuple[tuple[int, bool], ...]
_ID: Quat = (Q(1), Q(0), Q(0), Q(0))
_X: Word = ((5, True), (0, True), (4, False), (2, False))
_Y: Word = ((5, True), (1, False), (6, False), (3, True))


def _mul(a: Quat, b: Quat) -> Quat:
    return (
        a[0] * b[0] - sum((a[i] * b[i] for i in range(1, 4)), Q(0)),
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] + a[2] * b[0] + a[3] * b[1] - a[1] * b[3],
        a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1],
    )


def _inv(a: Quat) -> Quat:
    return a[0], -a[1], -a[2], -a[3]


def _stereo(v: Sequence[Q]) -> Quat:
    s = sum((x * x for x in v), Q(0))
    return (1 - s) / (1 + s), 2 * v[0] / (1 + s), 2 * v[1] / (1 + s), 2 * v[2] / (1 + s)


def _word(links: Sequence[Quat], word: Word, edge: int = -1, axis: int = 0) -> Quat:
    if edge >= 0 and all(i != edge for i, _ in word):
        return Q(0), Q(0), Q(0), Q(0)
    result = _ID
    generator = (Q(0), Q(axis == 0, 2), Q(axis == 1, 2), Q(axis == 2, 2))
    for i, inverse in word:
        factor = _mul(links[i], generator) if i == edge else links[i]
        result = _mul(result, _inv(factor) if inverse else factor)
    return result


@pytest.fixture(scope="module")
def result() -> dict[str, Any]:
    return module.su2_theta_normalized_kernel_tail(Q(1, 2**40), 400_000, target=Q(1, 1024))


def test_actual_nonempty_tail_target_and_canonical_source(result: dict[str, Any]) -> None:
    assert result["status"] == result["tail_target_status"] == "PASS"
    assert not result["tail_region_has_zero_haar_measure"]
    assert result["arithmetic"]["ground_energy_upper"] == "6"
    assert result["arithmetic"]["squared_hs_tail_exponent"] == "-65240/9"
    assert result["arithmetic"]["dyadic_upper_negative_integer_exponent"] == 7248
    assert result["arithmetic"]["target_required_dyadic_exponent"] == 35
    assert all(result["arithmetic"]["gates"].values())
    assert result["certificate"]["payload"] == {
        k: v for k, v in result.items() if k != "certificate"
    }
    assert module.replay_su2_theta_kernel_tail_certificate(result["certificate"])
    assert source.replay_su2_theta_weak_block_certificate(
        result["theta_ground_energy_source_certificate"]
    )
    for flag in (
        "actual_normalized_kernel_tail_verified_in_written_analysis",
        "uniform_rescaled_hs_tail_verified_in_written_analysis",
        "radial_marginal_kernel_tail_verified_in_written_analysis",
        "all_reduced_modes_included",
    ):
        assert result[flag] is True
    for flag in (
        "source_gap_used_as_premise",
        "gaussian_kernel_comparison_verified",
        "maximal_correlation_upper_verified",
        "ambient_exterior_uniformity_verified",
        "uniform_in_volume_claim",
        "all_scale_refinement_claim",
        "continuum_claim",
        "yang_mills_claim",
        "yang_mills_mass_gap_claim",
        "analytic_proof_formally_verified",
        "mathlib_verified",
        "theorem_prover_verified",
    ):
        assert result[flag] is False


def test_theorem_and_target_decisions_are_distinct() -> None:
    row = module.su2_theta_normalized_kernel_tail(Q(1, 64), 1, target=Q(1, 1024))
    assert row["status"] == "PASS" and row["tail_target_status"] == "INCONCLUSIVE"
    assert row["actual_normalized_kernel_tail_verified_in_written_analysis"]
    assert not row["tail_target_verified"]
    assert row["factored_dyadic_squared_hs_tail_upper"]["negative_integer_exponent"] < 0
    absent = module.su2_theta_normalized_kernel_tail(Q(1, 64), 1)
    assert absent["tail_target_status"] == "NOT_REQUESTED"
    outside = module.su2_theta_normalized_kernel_tail(Q(1, 32), 400000, target=1)
    assert outside["status"] == outside["tail_target_status"] == "INCONCLUSIVE"
    assert outside["squared_hs_tail_upper"] is None
    assert not outside["actual_normalized_kernel_tail_verified_in_written_analysis"]


def test_compact_endpoint_tail_is_exactly_zero() -> None:
    for radius in (1024, 1025):
        row = module.su2_theta_normalized_kernel_tail(Q(1, 64), radius, target=Q(1, 2**4000))
        assert row["tail_region_has_zero_haar_measure"]
        assert row["effective_squared_hs_tail_upper"] == {"exact_rational": "0"}
        assert row["tail_target_status"] == "PASS"
    row = module.su2_theta_normalized_kernel_tail(Q(1, 64), 1023)
    assert not row["tail_region_has_zero_haar_measure"]


def test_extreme_rationals_never_expand_the_dyadic_power() -> None:
    row = module.su2_theta_normalized_kernel_tail(Q(1, 2**4096), 2**4096, target=Q(1, 2**4096))
    assert row["status"] == row["tail_target_status"] == "PASS"
    assert not row["tail_region_has_zero_haar_measure"]
    n = row["factored_dyadic_squared_hs_tail_upper"]["negative_integer_exponent"]
    assert n.bit_length() > 4000
    assert module.replay_su2_theta_kernel_tail_certificate(row["certificate"])


@pytest.mark.parametrize("field", ("kappa", "radius", "target"))
@pytest.mark.parametrize("bad", (True, False, 0.25, "1/64", 0, -1))
def test_exact_input_guards(field: str, bad: Any) -> None:
    kwargs: dict[str, Any] = {"kappa": Q(1, 64), "radius": 400000, "target": Q(1, 1024)}
    kwargs[field] = bad
    with pytest.raises((TypeError, ValueError)):
        module.su2_theta_normalized_kernel_tail(**kwargs)


@pytest.mark.parametrize("bad", (None, True, 1, [], {}, "certificate"))
def test_malformed_replayer_refuses(bad: Any) -> None:
    assert not module.replay_su2_theta_kernel_tail_certificate(bad)


@pytest.mark.parametrize(
    "damage", ("source", "exponent", "dyadic", "target", "radius", "scope", "ground")
)
def test_resealed_attacks_are_rejected(result: dict[str, Any], damage: str) -> None:
    cert = deepcopy(result["certificate"])
    p = cert["payload"]
    if damage == "source":
        child = p["theta_ground_energy_source_certificate"]
        child["payload"]["arithmetic"]["ground_energy_upper"] = "0"
        p["theta_ground_energy_source_certificate"] = seal_certificate(child)
    elif damage == "exponent":
        p["squared_hs_tail_upper"]["exponent"] = "-999999"
    elif damage == "dyadic":
        p["factored_dyadic_squared_hs_tail_upper"]["negative_integer_exponent"] += 1
    elif damage == "target":
        p["tail_target_status"] = "INCONCLUSIVE"
    elif damage == "radius":
        p["inputs"]["radius"] = "399999"
    elif damage == "scope":
        p["maximal_correlation_upper_verified"] = True
    else:
        p["arithmetic"]["ground_energy_upper"] = "0"
    assert not module.replay_su2_theta_kernel_tail_certificate(seal_certificate(cert))


def test_source_failure_cannot_earn_the_actual_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(source, "replay_su2_theta_weak_block_certificate", lambda cert: False)
    row = module.su2_theta_normalized_kernel_tail(Q(1, 64), 400000, target=1)
    assert row["status"] == row["tail_target_status"] == "INCONCLUSIVE"
    assert not row["actual_normalized_kernel_tail_verified_in_written_analysis"]


def test_only_sealed_ground_energy_is_consumed(monkeypatch: pytest.MonkeyPatch) -> None:
    original = source.su2_theta_weak_block_gaps

    def changed(kappa: Any) -> dict[str, Any]:
        row = deepcopy(original(kappa))
        row["arithmetic"] = deepcopy(row["arithmetic"])
        row["arithmetic"]["ground_energy_upper"] = "999"
        row["arithmetic"]["full_reduced_gap_lower"] = "-999"
        return row

    monkeypatch.setattr(source, "su2_theta_weak_block_gaps", changed)
    row = module.su2_theta_normalized_kernel_tail(Q(1, 64), 1)
    assert row["status"] == "PASS"
    assert row["arithmetic"]["ground_energy_upper"] == "6"
    assert not row["source_gap_used_as_premise"]


def test_independent_original_twenty_one_generator_rows() -> None:
    rng = Random(1192)
    for _ in range(16):
        links = [_stereo([Q(rng.randrange(-5, 6), 9) for _ in range(3)]) for _ in range(7)]
        x, y = _word(links, _X), _word(links, _Y)
        s = 4 - 2 * x[0] - 2 * y[0]
        raw = sum(
            (
                (-2 * _word(links, _X, e, a)[0] - 2 * _word(links, _Y, e, a)[0]) ** 2
                for e in range(7)
                for a in range(3)
            ),
            Q(0),
        )
        reduced = 4 * sum((x[i] ** 2 + y[i] ** 2 for i in range(1, 4)), Q(0))
        reduced += 2 * sum((x[i] * y[i] for i in range(1, 4)), Q(0))
        assert raw == reduced <= 5 * s
        # Every face contains four distinct links; each axis square contributes -1/4.
        delta_s = 4 * 3 * Q(1, 4) * 2 * (x[0] + y[0])
        assert delta_s == 12 - 3 * s


def test_independent_symbolic_radial_casimir_and_smoothing() -> None:
    sp = import_module("sympy")
    q, c, eps = sp.symbols("q c eps", positive=True)
    f = 8 * (1 - sp.sqrt((1 + q) / 2))
    smooth = 8 * (sp.sqrt(1 + eps**2) - sp.sqrt((1 + q) / 2 + eps**2))

    def casimir(expr: Any) -> Any:
        return -(1 - q * q) * sp.diff(expr, q, 2) / 4 + 3 * q * sp.diff(expr, q) / 4

    def gamma(expr: Any) -> Any:
        return (1 - q * q) * sp.diff(expr, q) ** 2 / 4

    assert sp.simplify(gamma(f) - (2 - 2 * q)) == 0
    assert sp.simplify(casimir(f).subs(q, 2 * c * c - 1) - (1 / c - sp.Rational(5, 2) * c)) == 0
    z = (1 + q) / 2 + eps**2
    assert (
        sp.simplify(
            casimir(smooth) + (1 - q * q) / (8 * z ** sp.Rational(3, 2)) + 3 * q / (2 * sp.sqrt(z))
        )
        == 0
    )
    assert sp.simplify(gamma(smooth) - (2 - 2 * q) * ((1 + q) / 2) / z) == 0


def test_cutoff_bochner_arithmetic_and_gaussian_haar_constants() -> None:
    sp = import_module("sympy")
    s, k, b, g = sp.symbols("s k b g", positive=True)
    zeta = 1 - s / (b * k)
    eta = zeta * zeta
    delta = sp.diff(eta, s) * (12 - 3 * s) + sp.diff(eta, s, 2) * g
    gamma = sp.diff(eta, s) ** 2 * g
    assert (
        sp.simplify(
            -delta + 2 * gamma / eta - (2 * zeta * (12 - 3 * s) / (b * k) + 6 * g / (b * b * k * k))
        )
        == 0
    )
    assert Q(649, 2048) < Q(64, 21)
    assert Q(8192 * 22, 21) < 10000
    assert Q(2, 5) * 256 - Q(414, 5) > 0
    assert Q(36, 5) - Q(4, 25) * 256 < 0
    assert Q(7, 11) * Q(25, 36) * Q(1, 3) * Q(1, 27) == Q(175, 32076) > Q(1, 200)
    assert 40000 * Q(9801, 392) ** 2 < 26000000


def test_exact_dyadic_comparison_all_signs() -> None:
    rng = Random(5181)
    for _ in range(150):
        v = Q(rng.randrange(1, 2**100), rng.randrange(1, 2**100))
        n = module._ceil_log_two(v)
        power = Q(2**n) if n >= 0 else Q(1, 2 ** (-n))
        assert power / 2 < v <= power
    for exponent in (Q(-1), Q(-3, 2), Q(0), Q(1, 3), Q(7, 2)):
        n = module._dyadic_exponent(exponent)
        if exponent <= 0:
            assert n <= -exponent < n + 1
        else:
            assert n < 0 and exponent <= Q(-n, 2)


def test_radial_coarsening_contracts_restricted_chi_square() -> None:
    # Four fine states per coordinate, grouped into two radial classes.
    rng = Random(2015)
    weights = [[Q(rng.randrange(1, 30)) for _ in range(4)] for _ in range(4)]
    z = sum((sum(row, Q(0)) for row in weights), Q(0))
    p = [[w / z for w in row] for row in weights]
    px = [sum(row, Q(0)) for row in p]
    py = [sum((p[i][j] for i in range(4)), Q(0)) for j in range(4)]
    # Radially measurable tail: at least one coordinate in group1.
    full = sum(
        (
            p[i][j] ** 2 / (px[i] * py[j])
            for i in range(4)
            for j in range(4)
            if i // 2 + j // 2 >= 1
        ),
        Q(0),
    )
    coarse = Q(0)
    for a in range(2):
        for b in range(2):
            if a + b < 1:
                continue
            joint = sum(
                (p[i][j] for i in range(2 * a, 2 * a + 2) for j in range(2 * b, 2 * b + 2)), Q(0)
            )
            ax = sum(px[2 * a : 2 * a + 2], Q(0))
            by = sum(py[2 * b : 2 * b + 2], Q(0))
            coarse += joint * joint / (ax * by)
    assert coarse <= full


def test_small_density_error_alone_does_not_bound_normalized_kernel() -> None:
    # Strictly positive joint laws compared with their own product marginals.
    # TV -> 0 implies square-root densities converge in L2, yet the centered
    # normalized kernels retain almost unit Hilbert--Schmidt norm.
    for epsilon in (Q(1, 100), Q(1, 1000), Q(1, 10000)):
        joint = ((1 - epsilon - 2 * epsilon**2, epsilon**2), (epsilon**2, epsilon))
        marginal = (1 - epsilon - epsilon**2, epsilon + epsilon**2)
        tv = (
            sum(
                (abs(joint[i][j] - marginal[i] * marginal[j]) for i in range(2) for j in range(2)),
                Q(0),
            )
            / 2
        )
        kernel_square = sum(
            (
                (joint[i][j] - marginal[i] * marginal[j]) ** 2 / (marginal[i] * marginal[j])
                for i in range(2)
                for j in range(2)
            ),
            Q(0),
        )
        assert 0 < tv < 2 * epsilon
        assert kernel_square > Q(9, 10)
