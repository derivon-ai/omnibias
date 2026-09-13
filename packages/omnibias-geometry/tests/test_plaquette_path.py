# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent angular/Friedrichs algebra, original-link geometry and replay tests."""

from copy import deepcopy
from fractions import Fraction as Q
from importlib import import_module
from random import Random
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer import plaquette_path as path
from omnibias.geometry.gauge.transfer import weak_plaquette as radial

sp = import_module("sympy")  # Required independent test oracle; never skipped.
Quaternion = tuple[Q, Q, Q, Q]


@pytest.fixture(scope="module")
def source() -> dict[str, Any]:
    return path.su2_plaquette_path_conditional_gap(Q(1, 64), 3)


def test_radial_unitary_derived_from_the_full_sphere_laplacian() -> None:
    theta = sp.Symbol("theta", positive=True)
    ell = sp.Symbol("ell", integer=True, nonnegative=True)
    u = sp.Function("u")(theta)
    f = u / sp.sin(theta)
    transformed = sp.sin(theta) * (
        -sp.diff(f, theta, 2)
        - 2 * sp.cot(theta) * sp.diff(f, theta)
        + ell * (ell + 1) * f / sp.sin(theta) ** 2
    )
    expected = -sp.diff(u, theta, 2) + (ell * (ell + 1) / sp.sin(theta) ** 2 - 1) * u
    assert sp.simplify(transformed - expected) == 0
    # No angular cutoff: this polynomial factor is nonnegative for every l>=1.
    assert sp.expand(ell * (ell + 1) - 2 - (ell - 1) * (ell + 2)) == 0


def test_half_line_angular_oscillator_floor_from_positive_eigenfunction() -> None:
    r, a, b = sp.symbols("r a b", positive=True)
    ell = sp.Symbol("ell", integer=True, nonnegative=True)
    u = r ** (ell + 1) * sp.exp(-sp.sqrt(b / a) * r**2 / 2)
    local_energy = sp.simplify(
        (-a * sp.diff(u, r, 2) + a * ell * (ell + 1) * u / r**2 + b * r**2 * u) / u
    )
    assert sp.simplify(local_energy - (2 * ell + 3) * sp.sqrt(a * b)) == 0
    kappa = sp.Symbol("kappa", positive=True)
    normalized = local_energy.subs({a: kappa / 2, b: 8 / (kappa * sp.pi**2)}) - kappa / 2
    assert sp.simplify(normalized.subs(ell, 1) - 10 / sp.pi + kappa / 2) == 0


def test_pi_majorant_has_an_independent_positive_integral_witness() -> None:
    t = sp.Symbol("t", real=True)
    numerator = t**4 * (1 - t) ** 4
    quotient, remainder = sp.div(numerator, 1 + t**2)
    assert remainder == -4
    assert sp.integrate(quotient, (t, 0, 1)) == sp.Rational(22, 7)
    # The remaining integral is -4 arctan(1)=-pi; its original integrand
    # is strictly positive on (0,1), so the upper bound is not a decimal fit.
    assert numerator.subs(t, sp.Rational(1, 2)) > 0


def test_all_coupling_crossing_is_an_exact_two_branch_argument() -> None:
    k = sp.Symbol("k", positive=True)
    crossing = sp.solve(sp.Eq(k, sp.Rational(2, 11) - k / 2), k)
    assert crossing == [sp.Rational(4, 33)]
    # Express each branch's excess by its distance to the crossover.
    assert (
        sp.simplify(sp.Rational(2, 11) - k / 2 - sp.Rational(4, 33)) == (sp.Rational(4, 33) - k) / 2
    )
    assert sp.Rational(26, 33) > sp.Rational(4, 33)


def _multiply(a: Quaternion, b: Quaternion) -> Quaternion:
    # SU2 convention U=q0 I+i q.sigma has a MINUS vector cross product.
    return (
        a[0] * b[0] - sum((a[i] * b[i] for i in range(1, 4)), Q(0)),
        a[0] * b[1] + b[0] * a[1] - a[2] * b[3] + a[3] * b[2],
        a[0] * b[2] + b[0] * a[2] - a[3] * b[1] + a[1] * b[3],
        a[0] * b[3] + b[0] * a[3] - a[1] * b[2] + a[2] * b[1],
    )


def _product(links: list[Quaternion]) -> Quaternion:
    value = (Q(1), Q(0), Q(0), Q(0))
    for link in links:
        value = _multiply(value, link)
    return value


def _unit(a: Q, b: Q, c: Q) -> Quaternion:
    norm = a * a + b * b + c * c
    return ((1 - norm) / (1 + norm), 2 * a / (1 + norm), 2 * b / (1 + norm), 2 * c / (1 + norm))


def _gradient(q: Quaternion) -> Quaternion:
    # A noncentral polynomial, so the check includes boundary-flux functions.
    # f(q)=q0^2+q1*q2+q3^3.
    return (2 * q[0], q[2], q[1], 3 * q[3] ** 2)


@pytest.mark.parametrize("count", [1, 2, 3])
def test_original_link_derivatives_and_internal_gauss_with_exact_quaternions(count: int) -> None:
    rng = Random(8219 + count)
    packs = [
        [_unit(Q(i, 3), Q(j, 5), Q(i - j, 7)) for i in range(1, count + 1)] for j in (-2, 0, 3)
    ]
    packs += [
        [_unit(*(Q(rng.randint(-3, 3), rng.randint(1, 5)) for _ in range(3))) for _ in range(count)]
        for _ in range(9)
    ]
    for links in packs:
        for frozen in (_unit(Q(0), Q(0), Q(0)), _unit(Q(1, 2), Q(-2, 3), Q(1, 5))):
            holonomy = _product([*links, frozen])
            grad = _gradient(holonomy)
            tangent_norm = (
                sum((entry**2 for entry in grad), Q(0))
                - sum((a * b for a, b in zip(grad, holonomy, strict=True)), Q(0)) ** 2
            ) / 4
            energy = Q(0)
            for index in range(count):
                for axis in range(1, 4):
                    generator: Quaternion = (
                        Q(0),
                        Q(axis == 1, 2),
                        Q(axis == 2, 2),
                        Q(axis == 3, 2),
                    )
                    differentiated = [*links]
                    differentiated[index] = _multiply(links[index], generator)
                    tangent = _product([*differentiated, frozen])
                    energy += sum((a * b for a, b in zip(grad, tangent, strict=True)), Q(0)) ** 2
            assert energy == count * tangent_norm
        if count > 1:
            h = _unit(Q(1, 3), Q(-2, 5), Q(3, 7))
            inverse = (h[0], -h[1], -h[2], -h[3])
            changed = [*links]
            changed[0] = _multiply(changed[0], inverse)
            changed[1] = _multiply(h, changed[1])
            assert _product(changed) == _product(links)


def test_boundary_flux_function_is_not_discarded_by_a_class_projection() -> None:
    u = _unit(Q(1, 2), Q(0), Q(0))
    h = (Q(0), Q(0), Q(1), Q(0))
    conjugated = _multiply(_multiply(h, u), (h[0], -h[1], -h[2], -h[3]))
    assert conjugated[0] == u[0]
    assert conjugated[1] == -u[1] != u[1]


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("coupling", [Q(1, 2**4000), Q(1, 64), Q(4, 33), Q(2), Q(2**4000)])
def test_canonical_all_coupling_and_length_outputs(coupling: Q, count: int) -> None:
    result = path.su2_plaquette_path_conditional_gap(coupling, count)
    assert result["status"] == "PASS"
    arithmetic = result["arithmetic"]
    assert Q(arithmetic["full_rotor_gap_lower"]) == Q(4, 33)
    assert Q(arithmetic["conditional_quantum_gap_lower"]) == Q(count, 33)
    assert Q(arithmetic["conditional_poincare_lower"]) * coupling / 2 == Q(count, 33)
    assert Q(arithmetic["nonradial_combined_gap_lower"]) >= Q(4, 33)
    assert result["geometry"]["internal_gauss_vertices"] == list(range(1, count))
    assert path.replay_su2_plaquette_path_conditional_certificate(result["certificate"])
    assert result["certificate"]["payload"] == {
        k: v for k, v in result.items() if k != "certificate"
    }


def test_integer_coupling_is_canonical_and_radial_source_is_not_full_rotor(
    source: dict[str, Any],
) -> None:
    assert path.su2_plaquette_path_conditional_gap(2) == path.su2_plaquette_path_conditional_gap(
        Q(2)
    )
    nested = source["radial_source_certificate"]
    assert radial.replay_su2_weak_plaquette_gap_certificate(nested)
    assert nested["honesty"]["full_unconstrained_rotor_claim"] is False
    assert nested["payload"]["witness"]["inputs"]["cutoff"] is None
    assert source["actual_full_rotor_gap_verified_in_written_analysis"] is True


@pytest.mark.parametrize("coupling", [True, False, 0.25, "1/4", None])
def test_exact_coupling_type_guards(coupling: Any) -> None:
    with pytest.raises(TypeError):
        path.su2_plaquette_path_conditional_gap(coupling)


@pytest.mark.parametrize("coupling", [0, -1, Q(-1, 100)])
def test_positive_coupling_guard(coupling: int | Q) -> None:
    with pytest.raises(ValueError):
        path.su2_plaquette_path_conditional_gap(coupling)


@pytest.mark.parametrize("count", [True, False, 1.0, "1", Q(1), None])
def test_exact_length_type_guards(count: Any) -> None:
    with pytest.raises(TypeError):
        path.su2_plaquette_path_conditional_gap(1, count)


@pytest.mark.parametrize("count", [-1, 0, 4, 100])
def test_proper_path_length_guard(count: int) -> None:
    with pytest.raises(ValueError):
        path.su2_plaquette_path_conditional_gap(1, count)


@pytest.mark.parametrize(
    "case",
    [
        "gap",
        "poincare",
        "angular",
        "geometry",
        "endpoint",
        "ambient",
        "formal",
        "source",
        "source_honesty",
        "raw_kappa",
        "raw_count",
        "noncanonical_kappa",
    ],
)
def test_resealed_tampering_is_rejected(source: dict[str, Any], case: str) -> None:
    forged = deepcopy(source["certificate"])
    payload = forged["payload"]
    if case in {"gap", "poincare", "angular"}:
        key = {
            "gap": "conditional_quantum_gap_lower",
            "poincare": "conditional_poincare_lower",
            "angular": "nonradial_combined_gap_lower",
        }[case]
        payload["arithmetic"][key] = "100"
    elif case == "geometry":
        payload["geometry"]["block_oriented_edges"] = [1, 3]
    elif case == "endpoint":
        payload["geometry"]["internal_gauss_vertices"] = [0, 1, 2, 3]
    elif case in {"ambient", "formal"}:
        payload[
            "embedded_ambient_block_claim" if case == "ambient" else "theorem_prover_verified"
        ] = True
    elif case in {"source", "source_honesty"}:
        nested = payload["radial_source_certificate"]
        if case == "source":
            nested["payload"]["witness"]["arithmetic"]["ground_energy_upper"] = "0"
        else:
            nested["honesty"]["actual_plaquette_gap_verified"] = False
        nested.pop("digest")
        payload["radial_source_certificate"] = seal_certificate(nested)
    elif case == "raw_kappa":
        payload["inputs"]["kappa"] = True
    elif case == "raw_count":
        payload["inputs"]["block_edges"] = 3.0
    else:
        payload["inputs"]["kappa"] = "2/128"
    forged.pop("digest")
    forged = seal_certificate(forged)
    assert verify_certificate_digest(forged)
    assert not path.replay_su2_plaquette_path_conditional_certificate(forged)


def test_source_replay_failure_cannot_earn_an_actual_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(radial, "replay_su2_weak_plaquette_gap_certificate", lambda _: False)
    failed = path.su2_plaquette_path_conditional_gap(Q(1, 64))
    assert failed["status"] == "INCONCLUSIVE"
    assert not failed["actual_full_rotor_gap_verified_in_written_analysis"]
    assert not failed["actual_path_conditional_gap_verified_in_written_analysis"]
    assert failed["arithmetic"]["conditional_quantum_gap_lower"] is None


def test_detached_source_report_cannot_bypass_canonical_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = radial.su2_weak_plaquette_gap(Q(1, 64))
    source["witness"] = deepcopy(source["witness"])
    source["witness"]["arithmetic"]["ground_energy_upper"] = "0"
    monkeypatch.setattr(radial, "su2_weak_plaquette_gap", lambda _: source)
    failed = path.su2_plaquette_path_conditional_gap(Q(1, 64))
    assert failed["status"] == "INCONCLUSIVE"
    assert not failed["radial_source_replay_verified"]


def test_scope_is_not_promoted(source: dict[str, Any]) -> None:
    for flag in (
        "spin_truncation_used",
        "endpoint_gauss_or_center_even_restriction_used",
        "frozen_bare_groundstate_substituted",
        "embedded_ambient_block_claim",
        "volume_uniform_claim",
        "all_scale_refinement_claim",
        "continuum_claim",
        "yang_mills_claim",
        "yang_mills_mass_gap_claim",
        "analytic_proof_formally_verified",
        "theorem_prover_verified",
        "mathlib_verified",
    ):
        assert source[flag] is False


@pytest.mark.parametrize("bad", [None, [], (), "certificate", 0, True, {}])
def test_malformed_replay_guard(bad: Any) -> None:
    assert not path.replay_su2_plaquette_path_conditional_certificate(bad)
