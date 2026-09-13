# SPDX-License-Identifier: Apache-2.0
"""Independent operator algebra and exact certificate boundary regressions."""

from copy import deepcopy
from fractions import Fraction as Q
from math import comb
from typing import Any

import pytest
import sympy as sp  # type: ignore[import-untyped]
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer import vmf_poincare as module


def test_barta_ratios_from_differentiation_and_radial_intertwining() -> None:
    theta, h = sp.symbols("theta h", real=True)
    drift = 2 * sp.cot(theta) - h * sp.sin(theta)
    for exponent, coefficient in ((sp.Rational(1, 8), 7), (sp.Rational(3, 8), 15)):
        logarithmic_derivative = sp.cot(theta) / 2 + exponent * h * sp.sin(theta)
        ratio = (
            -sp.diff(logarithmic_derivative, theta)
            - logarithmic_derivative**2
            - drift * logarithmic_derivative
            + 2 / sp.sin(theta) ** 2
        )
        if coefficient == 15:
            ratio += h * sp.cos(theta)
        expected = (
            sp.Rational(5, 4)
            + sp.Rational(5, 4) / sp.sin(theta) ** 2
            + sp.Rational(coefficient, 64) * h**2 * sp.sin(theta) ** 2
        )
        assert sp.trigsimp(ratio - expected) == 0
    f = sp.Function("f")(theta)
    a0f = -sp.diff(f, theta, 2) - drift * sp.diff(f, theta)
    g = sp.diff(f, theta)
    partner_g = (
        -sp.diff(g, theta, 2)
        - drift * sp.diff(g, theta)
        + (2 / sp.sin(theta) ** 2 + h * sp.cos(theta)) * g
    )
    assert sp.trigsimp(sp.diff(a0f, theta) - partner_g) == 0


@pytest.mark.parametrize("h", (Q(1, 1000), Q(1, 3), Q(1), Q(4), Q(100), Q(10**8)))
def test_all_sector_exact_pointwise_barta_budget(h: Q) -> None:
    for x in (Q(-999, 1000), Q(-9, 10), Q(-1, 3), Q(0), Q(2, 3), Q(999, 1000)):
        sine_squared = 1 - x**2
        angular = Q(5, 4) + Q(5, 4) / sine_squared + Q(7, 64) * h**2 * sine_squared
        radial = Q(5, 4) + Q(5, 4) / sine_squared + Q(15, 64) * h**2 * sine_squared
        assert angular > Q(5, 4) + h / 2
        assert radial > Q(5, 4) + h
        for ell in (1, 2, 3, 17, 100):
            higher_sector = angular + (ell * (ell + 1) - 2) / sine_squared
            assert higher_sector >= angular
    row = module.su2_vmf_poincare_bound(h)
    assert row["status"] == "PASS"
    assert row["arithmetic"]["su2_linear_poincare_gap_lower"] == str(h / 8)
    assert Q(row["arithmetic"]["su2_additive_poincare_gap_lower"]) == Q(5, 16) + h / 8
    assert module.replay_su2_vmf_poincare_certificate(row["certificate"])


def test_completed_squares_are_polynomial_identities() -> None:
    z = sp.symbols("z")
    assert (
        sp.expand(7 * (z - sp.Rational(16, 7)) ** 2 + sp.Rational(304, 7)) == 7 * z**2 - 32 * z + 80
    )
    assert (
        sp.expand(15 * (z - sp.Rational(32, 15)) ** 2 + sp.Rational(176, 15))
        == 15 * z**2 - 64 * z + 80
    )


@pytest.mark.parametrize("n", (1, 2, 4, 8))
def test_actual_path_matrix_walk_enclosures_support_reference_margin(n: int) -> None:
    """Exact convergent-series enclosure, independent of any floating root."""
    order = 12
    adjacency = [[int(abs(i - j) == 1) for j in range(n)] for i in range(n)]
    power = [[int(i == j) for j in range(n)] for i in range(n)]
    partial = [[Q(int(i == j), 2) for j in range(n)] for i in range(n)]
    for k in range(1, order + 1):
        power = [
            [sum(power[i][z] * adjacency[z][j] for z in range(n)) for j in range(n)]
            for i in range(n)
        ]
        coefficient = Q(comb(2 * k, k), 2 * 16**k)
        for i in range(n):
            for j in range(n):
                partial[i][j] += coefficient * power[i][j]
    # Coefficients of (1-z)^(-1/2) are <=1 and row norm A/4<=1/2.
    row_tail = Q(1, 2 ** (order + 1))
    for i in range(n):
        assert partial[i][i] >= Q(1, 2)
        row_upper = sum(partial[i]) + row_tail
        assert row_upper < Q(17, 24)
        assert 3 * partial[i][i] - 2 * row_upper > Q(1, 12)
        assert all(value >= 0 for value in partial[i])


def _quaternion(parameters: tuple[Q, Q, Q]) -> tuple[Q, Q, Q, Q]:
    norm = sum((x * x for x in parameters), Q(0))
    return (
        (1 - norm) / (1 + norm),
        2 * parameters[0] / (1 + norm),
        2 * parameters[1] / (1 + norm),
        2 * parameters[2] / (1 + norm),
    )


@pytest.mark.parametrize(
    "parameters", ((Q(0), Q(0), Q(0)), (Q(1), Q(2), Q(-1)), (Q(1, 3), Q(-2, 7), Q(4, 5)))
)
def test_quaternion_differential_metric_normalization(parameters: tuple[Q, Q, Q]) -> None:
    q0, x, y, z = _quaternion(parameters)
    # Columns are derivatives of exp(i*t*sigma_a/2)*U at t=0.
    derivative = [
        [-x / 2, -y / 2, -z / 2],
        [q0 / 2, z / 2, -y / 2],
        [-z / 2, q0 / 2, x / 2],
        [y / 2, -x / 2, q0 / 2],
    ]
    for i in range(3):
        for j in range(3):
            full_gram = sum((row[i] * row[j] for row in derivative), Q(0))
            vector_gram = sum((row[i] * row[j] for row in derivative[1:]), Q(0))
            assert full_gram == Q(int(i == j), 4)
            # Missing scalar-coordinate outer product is PSD; ||dX||<=1/2.
            assert Q(int(i == j), 4) - vector_gram == derivative[0][i] * derivative[0][j]


def test_reference_conditional_field_identity_including_trace_factor() -> None:
    kappa = Q(3, 7)
    # The identity holds for arbitrary symmetric coefficients; this test is
    # not a replacement for the specific path inverse square root.
    b = [
        [Q(1, 2) if i == j else Q(1, 20) if abs(i - j) == 1 else Q(0) for j in range(4)]
        for i in range(4)
    ]
    points = [_quaternion((Q(i, 5), Q(i - 1, 7), Q(2 - i, 11))) for i in range(4)]

    def phase(qs: list[tuple[Q, Q, Q, Q]]) -> Q:
        diagonal = sum((2 * b[i][i] * (2 - 2 * qs[i][0]) for i in range(4)), Q(0))
        mixed = sum(
            (
                4 * b[i][j] * sum((qs[i][a] * qs[j][a] for a in range(1, 4)), Q(0))
                for i in range(4)
                for j in range(i + 1, 4)
            ),
            Q(0),
        )
        return diagonal + mixed

    for i in range(4):
        changed = points.copy()
        changed[i] = _quaternion((Q(2), Q(-1), Q(3)))
        field = [8 * b[i][i] / kappa] + [
            -8 * sum((b[i][j] * points[j][a] for j in range(4) if j != i), Q(0)) / kappa
            for a in range(1, 4)
        ]
        exponent_difference = -2 * (phase(changed) - phase(points)) / kappa
        assert exponent_difference == sum(
            (field[a] * (changed[i][a] - points[i][a]) for a in range(4)), Q(0)
        )
        assert sum((entry**2 for entry in field), Q(0)) >= (8 * b[i][i] / kappa) ** 2


@pytest.mark.parametrize(
    "length,kappa", ((1, Q(1)), (2, Q(1, 64)), (16, Q(1, 2**50)), (10**6, Q(2**50)))
)
def test_reference_source_binding_exact_units_and_scope(length: int, kappa: Q) -> None:
    row = module.su2_strip_reference_poincare(length, kappa)
    assert row["status"] == "PASS"
    assert row["finite_gate_verified"]
    assert row["source_certificate_replayed"]
    assert row["source_certificate"]["payload"]["inputs"]["h"] == str(4 / kappa)
    assert module.replay_su2_vmf_poincare_certificate(row["certificate"])
    a = row["arithmetic"]
    assert Q(a["cycle_product_poincare_gap_lower"]) == 1 / (12 * kappa)
    assert Q(a["original_electric_coefficient"]) * Q(a["cycle_product_poincare_gap_lower"]) == Q(
        1, 24
    )
    assert row["reference_has_nonzero_pair_interactions"] == (length > 1)
    assert row["reference_constants_uniform_in_length"]
    assert row["reference_rooted_original_electric_gap_verified_in_written_analysis"]
    assert not row["unrestricted_original_link_scalar_gap_verified"]
    assert not row["actual_nonlinear_vacuum_verified"]
    assert not row["actual_wilson_vacuum_verified"]
    assert not row["mathlib_verified"]
    assert not row["theorem_prover_verified"]
    assert {k: v for k, v in row.items() if k != "certificate"} == row["certificate"]["payload"]


@pytest.mark.parametrize("kind", ("scalar", "reference"))
@pytest.mark.parametrize(
    "field", ("input", "arithmetic", "scope", "honesty", "meta", "claim", "status", "type", "gates")
)
def test_resealed_certificate_forgery_refused(kind: str, field: str) -> None:
    row = (
        module.su2_vmf_poincare_bound(Q(4))
        if kind == "scalar"
        else module.su2_strip_reference_poincare(4, Q(1, 8))
    )
    cert = deepcopy(row["certificate"])
    payload = cert["payload"]
    if field == "input":
        payload["inputs"]["h" if kind == "scalar" else "kappa"] = "5"
    elif field == "arithmetic":
        payload["arithmetic"][next(iter(payload["arithmetic"]))] = "999"
    elif field == "scope":
        payload["actual_wilson_vacuum_verified"] = True
    elif field == "honesty":
        cert["honesty"]["analytic_proof_formally_verified"] = True
    elif field == "meta":
        cert["meta"]["transcend_backend"] = "forged"
    elif field == "claim":
        cert["claim"] = "Clay Yang-Mills gap"
    elif field == "status":
        payload["status"] = "INCONCLUSIVE"
    elif field == "type":
        payload["type"] = "unknown"
    else:
        payload["gates"][next(iter(payload["gates"]))] = False
    cert = seal_certificate(cert)
    assert verify_certificate_digest(cert)
    assert not module.replay_su2_vmf_poincare_certificate(cert)


def test_nested_source_resealed_forgery_and_parameter_swap_refused() -> None:
    original = module.su2_strip_reference_poincare(8, Q(1, 64))["certificate"]
    for change in ("scope", "swap", "normalization"):
        cert = deepcopy(original)
        nested = cert["payload"]["source_certificate"]
        if change == "scope":
            nested["payload"]["actual_wilson_vacuum_verified"] = True
        elif change == "normalization":
            nested["payload"]["normalization"]["su2_metric_radius"] = "1"
        else:
            nested = module.su2_vmf_poincare_bound(1)["certificate"]
        cert["payload"]["source_certificate"] = seal_certificate(nested)
        cert = seal_certificate(cert)
        assert verify_certificate_digest(cert)
        assert not module.replay_su2_vmf_poincare_certificate(cert)


def test_reference_refuses_failed_scalar_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "replay_su2_vmf_poincare_certificate", lambda _: False)
    row = module.su2_strip_reference_poincare(4, Q(1, 16))
    assert row["status"] == "INCONCLUSIVE"
    assert not row["finite_gate_verified"]
    assert not row["reference_cycle_product_poincare_verified_in_written_analysis"]
    assert not row["reference_rooted_original_electric_gap_verified_in_written_analysis"]
    assert not row["constructed_compact_reference_verified_in_written_analysis"]


@pytest.mark.parametrize("bad", (True, 1.0, "1", None, [], complex(1), float("nan")))
def test_strict_field_and_coupling_types(bad: Any) -> None:
    with pytest.raises(TypeError):
        module.su2_vmf_poincare_bound(bad)
    with pytest.raises(TypeError):
        module.su2_strip_reference_poincare(4, bad)


@pytest.mark.parametrize("bad", (0, -1, Q(-1, 10)))
def test_nonpositive_field_and_coupling_refused(bad: int | Q) -> None:
    with pytest.raises(ValueError):
        module.su2_vmf_poincare_bound(bad)
    with pytest.raises(ValueError):
        module.su2_strip_reference_poincare(4, bad)


@pytest.mark.parametrize("bad", (True, 1.0, Q(1), "1", None))
def test_length_types(bad: Any) -> None:
    with pytest.raises(TypeError):
        module.su2_strip_reference_poincare(bad, 1)


@pytest.mark.parametrize("bad", (0, -1))
def test_length_range(bad: int) -> None:
    with pytest.raises(ValueError):
        module.su2_strip_reference_poincare(bad, 1)


@pytest.mark.parametrize("bad", (None, [], True, {}, {"payload": []}))
def test_bad_certificate_container(bad: Any) -> None:
    assert not module.replay_su2_vmf_poincare_certificate(bad)


def test_no_mutable_cached_source_and_canonical_rational_spelling() -> None:
    row = module.su2_strip_reference_poincare(4, 1)
    row["source_certificate"]["payload"]["arithmetic"]["su2_linear_poincare_gap_lower"] = "999"
    new = module.su2_strip_reference_poincare(4, 1)
    assert module.replay_su2_vmf_poincare_certificate(new["certificate"])
    cert = deepcopy(new["certificate"])
    cert["payload"]["inputs"]["kappa"] = "2/2"
    assert not module.replay_su2_vmf_poincare_certificate(seal_certificate(cert))


def test_transfer_public_exports() -> None:
    from omnibias.geometry.gauge import transfer

    for name in module.__all__:
        assert name in transfer.__all__
        assert getattr(transfer, name) is getattr(module, name)
