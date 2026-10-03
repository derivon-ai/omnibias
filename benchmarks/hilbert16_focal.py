#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Focal/Bautin engine, derived singular return maps, and collar membership.

Seven gates (GF1-GF7) replay the exact-Q machinery added on top of
``omnibias.holonomic._core.groebner``: Buchberger with a cofactor witness
(GF1), the homological-equation focal-value engine (GF2), the Bautin ideal
basis with its always-false stabilization flag (GF3), Poincare-Dulac
resonant normal forms at a hyperbolic saddle (GF4), the first-order derived
Dulac corner expansion (GF5), sound collar-membership agreement including a
genuine unique-cycle proof (GF6), and the wiring of every certificate into
``GraphicTarget``/``certify_graphic_cyclicity`` (GF7). None of this proves
physical return-map membership, graphic-wide finite cyclicity, or Hilbert 16;
see ``omnibias.dynamics.hilbert16_ledger`` for the machine-checked obligation
ledger, whose derived parent flags stay false on every ledger this repository
ships.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from fractions import Fraction as Q
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.realization.polynomial import SparsePolynomial as P  # noqa: E402
from omnibias.core.verified.interval import Interval  # noqa: E402
from omnibias.dynamics.bautin import (  # noqa: E402
    bautin_basis,
    bautin_quadratic_family,
    center_membership,
    hamiltonian_center_example,
    nonzero_focus_example,
)
from omnibias.dynamics.compactify import PlanarPolynomialField  # noqa: E402
from omnibias.dynamics.dulac import DulacExpansion  # noqa: E402
from omnibias.dynamics.focal import (  # noqa: E402
    certify_focal_values,
    focal_order,
    lyapunov_quantities,
    verify_focal_values,
)
from omnibias.dynamics.graphic import (  # noqa: E402
    certify_graphic_cyclicity,
    named_rational_hyperbolic_graphic,
    verify_graphic_cyclicity,
)
from omnibias.dynamics.membership import (  # noqa: E402
    certify_collar_membership,
    verify_collar_membership,
)
from omnibias.dynamics.return_maps import StoppedEventResult  # noqa: E402
from omnibias.dynamics.saddle_normal_form import (  # noqa: E402
    DiagonalSaddleField,
    certify_resonant_normal_form,
    derive_return_map,
    diagonalize_saddle,
    dulac_corner_expansion,
    resonant_normal_form,
    verify_resonant_normal_form,
)
from omnibias.holonomic._core.groebner import (  # noqa: E402
    GroebnerBudget,
    GroebnerBudgetExceeded,
    ideal_member,
    reduced_groebner_basis,
    verify_ideal_membership,
)
from omnibias.holonomic._core.poly_n import PolyN  # noqa: E402


def _gf1_groebner() -> dict[str, Any]:
    # The classic unit-circle-union-both-axes example: (x^2+y^2-1, x*y).
    x, y = PolyN.var(2, 0), PolyN.var(2, 1)
    generators = [x * x + y * y - PolyN.const(2, 1), x * y]
    basis = reduced_groebner_basis(generators, "degrevlex")
    membership = ideal_member(x * x * y - x * y, generators, "degrevlex")
    replay_ok = verify_ideal_membership(x * x * y - x * y, generators, membership)
    budget_refused = False
    try:
        reduced_groebner_basis(
            generators, "degrevlex", budget=GroebnerBudget(max_pairs=1, max_polynomials=1)
        )
    except GroebnerBudgetExceeded:
        budget_refused = True
    ok = len(basis) >= 1 and membership.is_member and replay_ok and budget_refused
    return {
        "name": "gf1_groebner_engine",
        "passed": bool(ok),
        "basis_length": len(basis),
        "cofactor_witness_replayed": replay_ok,
        "budget_refused_loudly": budget_refused,
        "detail": "Buchberger reduced basis + cofactor-witness replay + hard budget refusal",
    }


def _gf2_focal_values() -> dict[str, Any]:
    x, y = (P.variable(2, axis) for axis in range(2))
    p_pert, q_pert = nonzero_focus_example()
    p = -y + P(2, dict(p_pert.terms))
    q = x + P(2, dict(q_pert.terms))
    field = PlanarPolynomialField(p, q, 3)
    certificate = certify_focal_values(field, order=4)
    order = focal_order(certificate.quantities)
    v1 = dict(certificate.quantities.quantities)[4]
    round_trip = verify_focal_values(certificate)

    h_p, h_q = hamiltonian_center_example()
    center_field = PlanarPolynomialField(-y + P(2, dict(h_p.terms)), x + P(2, dict(h_q.terms)), 3)
    center_cert = certify_focal_values(center_field, order=6)
    all_zero = all(v.is_zero() for _, v in center_cert.quantities.quantities)

    tampered = replace(certificate, source_digest="tampered")
    tamper_rejected = not verify_focal_values(tampered)

    ok = order == 4 and v1.terms.get((), None) == Q(3, 4) and round_trip and all_zero and tamper_rejected
    return {
        "name": "gf2_focal_values",
        "passed": bool(ok),
        "focal_order": order,
        "v1": str(v1.terms.get((), None)),
        "hamiltonian_center_all_zero": all_zero,
        "detail": "homological-equation focal values vs. a Hamiltonian structural zero",
    }


def _gf3_bautin() -> dict[str, Any]:
    p, q = bautin_quadratic_family()
    quantities = lyapunov_quantities(p, q, order=8)
    basis = bautin_basis(quantities)
    membership = center_membership(quantities.quantities[0][1], basis.basis, radical=False)
    ok = (
        basis.basis_length == 3
        and basis.cyclicity_bound == 2
        and basis.bautin_ideal_stabilization_proved is False
        and membership.is_member
    )
    return {
        "name": "gf3_bautin_ideal",
        "passed": bool(ok),
        "basis_length": basis.basis_length,
        "cyclicity_bound": basis.cyclicity_bound,
        "bautin_ideal_stabilization_proved": basis.bautin_ideal_stabilization_proved,
        "detail": "Bautin's quadratic family: reduced Groebner basis length 3 (literature replay in tests)",
    }


def _gf4_resonant_normal_form() -> dict[str, Any]:
    target = named_rational_hyperbolic_graphic()
    diag = diagonalize_saddle(target.field, target.saddle)
    normal_form = resonant_normal_form(diag, order=3)
    certificate = certify_resonant_normal_form(target.field, target.saddle, order=3)
    round_trip = verify_resonant_normal_form(certificate)
    tampered = replace(certificate, source_digest="tampered")
    tamper_rejected = not verify_resonant_normal_form(tampered)
    ok = normal_form.resonant_monomials == () and round_trip and tamper_rejected
    return {
        "name": "gf4_resonant_normal_form",
        "passed": bool(ok),
        "resonant_monomial_count": len(normal_form.resonant_monomials),
        "detail": "exact Poincare-Dulac normalization at a rational hyperbolic saddle",
    }


def _gf5_derived_corner() -> dict[str, Any]:
    xi, eta = PolyN.var(2, 0), PolyN.var(2, 1)
    c = Q(1, 3)
    p_star = c * xi * xi * eta
    diag = DiagonalSaddleField(
        Q(-1), Q(1), p_star, PolyN.zero(2), ((Q(1), Q(0)), (Q(0), Q(1))), (Q(0), Q(0))
    )
    normal_form = resonant_normal_form(diag, order=3)
    corner = dulac_corner_expansion(normal_form, Q(1), order=2)
    coefficients = {(t.exponent.lo, t.log_power): t.coefficient.lo for t in corner.expansion.terms}
    # y = x/(1+c*x*log(x)) = x - c*x^2*log(x) + O((x*log(x))^2), exact to first order.
    matches_closed_form = coefficients.get((Q(1), 0)) == Q(1) and coefficients.get((Q(2), 1)) == -c

    regular_arc = StoppedEventResult(
        request=None,  # type: ignore[arg-type]
        source_fingerprint="benchmark-regular-arc",
        status="certified",
        reason="synthetic fixture for composition gate",
        slabs=(),
        return_jacobian=((Interval.point(2.0),),),
    )
    composed = derive_return_map(corner, regular_arc)
    scaled_ok = all(
        abs(float(s.coefficient.lo) - 2.0 * float(o.coefficient.lo)) < 1e-9
        and abs(float(s.coefficient.hi) - 2.0 * float(o.coefficient.hi)) < 1e-9
        for o, s in zip(corner.expansion.terms, composed.expansion.terms, strict=True)
    )
    ok = corner.residual_verified and matches_closed_form and scaled_ok
    return {
        "name": "gf5_derived_corner_expansion",
        "passed": bool(ok),
        "matches_bernoulli_closed_form": matches_closed_form,
        "regular_arc_composition_scaled": scaled_ok,
        "detail": "first-order log-correction vs. the hand-solvable Bernoulli case",
    }


def _gf6_collar_membership() -> dict[str, Any]:
    target = named_rational_hyperbolic_graphic()
    diag = diagonalize_saddle(target.field, target.saddle)
    normal_form = resonant_normal_form(diag, order=3)
    derived = dulac_corner_expansion(normal_form, Q(2), order=2)

    blocked = certify_collar_membership(
        target, derived, delta=Q(1, 1000), delta0=Q(1, 100), grid_points=4
    )
    matching_map = DulacExpansion.create(((2, 0, 1),), truncation_order=2, remainder_bound=Q(0))
    proved_target = replace(target, return_map=matching_map)
    proved = certify_collar_membership(
        proved_target, derived, delta=Q(1, 1000), delta0=Q(1, 100), grid_points=4
    )

    cycle_map = DulacExpansion.create(
        ((1, 0, Q(9, 10)), (2, 0, 20)), truncation_order=2, remainder_bound=Q(0)
    )
    cycle_target = replace(target, return_map=cycle_map)
    cycle_derived = replace(derived, expansion=cycle_map)
    cycle_cert = certify_collar_membership(
        cycle_target, cycle_derived, delta=Q(3, 1000), delta0=Q(7, 1000), grid_points=4
    )

    ok = (
        blocked.status == "BLOCKED"
        and verify_collar_membership(blocked)
        and proved.status == "PROVED_COLLAR"
        and verify_collar_membership(proved)
        and cycle_cert.status == "PROVED_COLLAR"
        and cycle_cert.unique_cycle.proved_unique_cycle
        and cycle_cert.unique_cycle.enclosure is not None
        and cycle_cert.unique_cycle.enclosure[0] <= 0.005 <= cycle_cert.unique_cycle.enclosure[1]
    )
    return {
        "name": "gf6_collar_membership",
        "passed": bool(ok),
        "blocked_status": blocked.status,
        "proved_status": proved.status,
        "unique_cycle_enclosure": cycle_cert.unique_cycle.enclosure,
        "detail": "sound declared-vs-derived agreement + a genuine unique-cycle proof",
    }


def _gf7_wiring() -> dict[str, Any]:
    target = named_rational_hyperbolic_graphic()
    nf_cert = certify_resonant_normal_form(target.field, target.saddle, order=3)
    diag = diagonalize_saddle(target.field, target.saddle)
    normal_form = resonant_normal_form(diag, order=3)
    derived = dulac_corner_expansion(normal_form, Q(2), order=2)
    p, q = bautin_quadratic_family()
    basis = bautin_basis(lyapunov_quantities(p, q, order=6))

    x, y = (P.variable(2, axis) for axis in range(2))
    p_pert, q_pert = nonzero_focus_example()
    focal_field = PlanarPolynomialField(-y + P(2, dict(p_pert.terms)), x + P(2, dict(q_pert.terms)), 3)
    focal_cert = certify_focal_values(focal_field, order=4)

    wired = replace(
        target,
        focal=focal_cert,
        bautin=basis,
        resonant_normal_form=nf_cert,
        derived_corner=derived,
        collar_status="PROVED_COLLAR",
    )
    certificate = certify_graphic_cyclicity(wired)
    honesty = certificate.seal["honesty"]
    round_trip = verify_graphic_cyclicity(certificate)
    ok = (
        round_trip
        and honesty["focal_values_computed"] is True
        and honesty["bautin_ideal_stabilization_proved"] is False
        and honesty["collar_return_membership_proved"] is True
        and honesty["physical_return_membership_proved"] is False
        and honesty["graphic_finite_cyclicity_proved"] is False
        and honesty["full_hilbert16_solved"] is False
    )
    return {
        "name": "gf7_graphic_wiring",
        "passed": bool(ok),
        "honesty": {
            k: honesty[k]
            for k in (
                "focal_values_computed",
                "bautin_ideal_stabilization_proved",
                "collar_return_membership_proved",
                "physical_return_membership_proved",
                "graphic_finite_cyclicity_proved",
                "full_hilbert16_solved",
            )
        },
        "detail": "focal/bautin/normal-form/corner/collar certificates wired into GraphicTarget",
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "hilbert16_focal.json" if full else "hilbert16_focal_smoke.json"
    t0 = time.perf_counter()
    entries = [
        _gf1_groebner(),
        _gf2_focal_values(),
        _gf3_bautin(),
        _gf4_resonant_normal_form(),
        _gf5_derived_corner(),
        _gf6_collar_membership(),
        _gf7_wiring(),
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.hilbert16_focal.v1",
            config={"family": "hilbert16_focal", "full": full},
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
        "honesty": {
            "physical_return_membership_proved": False,
            "graphic_finite_cyclicity_proved": False,
            "drr_case_closed": False,
            "full_hilbert16_solved": False,
        },
        "disclaimer": (
            "Exact-Q Groebner/focal/Bautin/normal-form/corner/collar machinery "
            "replayed on declared fields and named graphics. Not physical "
            "return-map membership, graphic-wide finite cyclicity, or Hilbert 16 "
            "-- see omnibias.dynamics.hilbert16_ledger / benchmarks/hilbert16_ledger.py."
        ),
    }
    path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
