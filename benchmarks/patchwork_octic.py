#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Symbolic/exact search for the selected 22-oval Hilbert-XVI octic scheme.

The differentiable CSP and annealed relaxations are proposers.  Exact projective
T-curve topology and rational lower-hull inequalities are the acceptance gate.
A finite miss is ``search_incomplete`` and never an obstruction.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict, is_dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.proof.certificate import verify_certificate_digest  # noqa: E402
from omnibias.core.realization.polynomial import SparsePolynomial  # noqa: E402
from omnibias.geometry.algebraic import (  # noqa: E402
    HomogeneousPlaneCurve,
    PolygonalAnnulus,
    RationalPolygon,
    certify_curve,
    find_smoothness_witness,
    replay_curve_certificate,
)
from omnibias.geometry.patchwork import (  # noqa: E402
    SignDistribution,
    patchwork_curve,
    staircase_triangulation,
)
from omnibias.geometry.patchwork_height_lp import (  # noqa: E402
    certify_regular_heights,
    lower_hull_inequalities,
    propose_farkas_infeasibility,
    verify_rational_infeasibility,
    verify_regular_height_certificate,
)
from omnibias.geometry.patchwork_search import (  # noqa: E402
    run_patchwork_discovery,
    triangulation_bank,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _safe(value: Any) -> Any:
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if is_dataclass(value) and not isinstance(value, type):
        return _safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_safe(item) for item in value]
    return value


def _square_polygon(a: int, b: int, radius: Fraction) -> RationalPolygon:
    return RationalPolygon((
        (a - radius, b - radius),
        (a + radius, b - radius),
        (a + radius, b + radius),
        (a - radius, b + radius),
    ))


def _gp1() -> dict[str, Any]:
    triangulation = staircase_triangulation(4)
    signs = SignDistribution.create(
        4,
        {
            point: (point[0] * point[1]) % 2
            for point in triangulation.vertices
        },
    )
    result = patchwork_curve(triangulation, signs)
    passed = (
        len(triangulation.vertices) == 15
        and len(triangulation.triangles) == 16
        and len(triangulation.edges) == 30
        and result.component_count == 4
        and result.rooted_tree == ((), (), (), ())
    )
    return {
        "name": "gp1_exact_patchwork_substrate",
        "passed": passed,
        "quartic_components": result.component_count,
        "quartic_region_tree": _safe(result.rooted_tree),
        "triangles": len(triangulation.triangles),
        "edges": len(triangulation.edges),
    }


def _gp2() -> dict[str, Any]:
    triangulation = staircase_triangulation(8)
    heights = {
        point: point[0] ** 2 + point[1] ** 2 + point[0] * point[1]
        for point in triangulation.vertices
    }
    regular = certify_regular_heights(triangulation, heights)
    points = ((4, 0), (0, 4), (0, 0), (2, 1), (1, 2), (1, 1))
    nonregular = lower_hull_inequalities(
        points,
        (
            (points[0], points[1], points[3]),
            (points[1], points[2], points[4]),
            (points[0], points[2], points[5]),
            (points[1], points[3], points[4]),
            (points[2], points[4], points[5]),
            (points[0], points[3], points[5]),
            (points[3], points[4], points[5]),
        ),
    )
    farkas = propose_farkas_infeasibility(nonregular)
    passed = (
        verify_regular_height_certificate(regular)
        and min(regular.feasibility.residuals) >= 0
        and verify_rational_infeasibility(nonregular, farkas)
        and verify_certificate_digest(farkas.seal)
    )
    return {
        "name": "gp2_exact_height_and_farkas",
        "passed": passed,
        "height_constraints": len(regular.feasibility.residuals),
        "minimum_residual": _safe(min(regular.feasibility.residuals)),
        "height_digest": regular.feasibility.seal["digest"],
        "farkas_multipliers": _safe(farkas.multipliers),
        "farkas_contradiction": _safe(farkas.contradiction),
        "nonregular_configuration_points": len(points),
        "scope": "Farkas exclusion is for the supplied six-point nonregular triangulation only.",
    }


def _gp3() -> dict[str, Any]:
    x, y, z = tuple(SparsePolynomial.variable(3, i) for i in range(3))
    px = (x**2 - z**2) * (x**2 - 9 * z**2)
    py = (y**2 - z**2) * (y**2 - 9 * z**2)
    curve = HomogeneousPlaneCurve(px**2 + py**2 - Fraction(1, 16) * z**8)
    witness = find_smoothness_witness(curve, max_multiplier_degree=12)
    if witness is None:
        return {"name": "gp3_polygonal_annuli", "passed": False, "reason": "no witness"}
    annuli = [
        PolygonalAnnulus(
            _square_polygon(i, j, Fraction(1, 1024)),
            _square_polygon(i, j, Fraction(1, 32)),
        )
        for i in (-3, -1, 1, 3)
        for j in (-3, -1, 1, 3)
    ]
    certificate = certify_curve(curve, witness, annuli)
    passed = (
        certificate.component_lower_bound == 16
        and certificate.harnack_upper_bound == 22
        and not certificate.complete_real_scheme
        and replay_curve_certificate(curve, certificate)
    )
    return {
        "name": "gp3_polygonal_annuli",
        "passed": passed,
        "component_lower_bound": certificate.component_lower_bound,
        "harnack_upper_bound": certificate.harnack_upper_bound,
        "complete_real_scheme": certificate.complete_real_scheme,
        "finite_obligation_digest": certificate.formal_seal["digest"],
    }


def search_box_identity(
    *,
    full: bool,
    budget: int | None,
    seeds: int | None,
    seed_start: int,
) -> dict[str, Any]:
    """Name the finite box ``run`` will search, before any candidate is scored.

    ``--full`` selects flip depth 3 and triangulation limit 128. Smoke keeps
    one staircase triangulation. Seed ids are ``seed_start`` plus a count, so
    a later box can avoid seeds already recorded.
    """
    from omnibias.geometry.part_a_target import PARTA_SELECTED_TARGET

    if type(seed_start) is not int or seed_start < 0:
        raise ValueError("seed_start must be a nonnegative integer")
    actual_budget = budget if budget is not None else (4096 if full else 8)
    actual_seeds = seeds if seeds is not None else (16 if full else 1)
    if type(actual_budget) is not int or actual_budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    if type(actual_seeds) is not int or actual_seeds < 1:
        raise ValueError("seeds must be a positive integer")
    flip_depth = 3 if full else 0
    triangulation_limit = 128 if full else 1
    seed_ids = list(range(seed_start, seed_start + actual_seeds))
    return {
        "seed_ids": seed_ids,
        "budget_per_seed": actual_budget,
        "flip_depth": flip_depth,
        "triangulation_limit": triangulation_limit,
        "searched_scheme": PARTA_SELECTED_TARGET.scheme,
        "searched_tree": PARTA_SELECTED_TARGET.tree,
    }


def _gp4(
    *,
    full: bool,
    budget: int | None,
    seeds: int | None,
    seed_start: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    identity = search_box_identity(
        full=full,
        budget=budget,
        seeds=seeds,
        seed_start=seed_start,
    )
    actual_budget = int(identity["budget_per_seed"])
    flip_depth = int(identity["flip_depth"])
    triangulation_limit = int(identity["triangulation_limit"])
    seed_ids = [int(seed) for seed in identity["seed_ids"]]
    bank = triangulation_bank(
        8,
        flip_depth=flip_depth,
        limit=triangulation_limit,
    )
    runs: list[dict[str, Any]] = []
    hit = False
    for seed in seed_ids:
        from omnibias.geometry.part_a_target import PARTA_SELECTED_TARGET

        report = run_patchwork_discovery(
            degree=8,
            budget=actual_budget,
            flip_depth=flip_depth,
            triangulation_limit=triangulation_limit,
            seed=seed,
            target=PARTA_SELECTED_TARGET.tree,
            triangulations=bank,
        )
        result = report["result"]
        characterization = result.characterization
        runs.append({
            "seed": seed,
            "triangulation_count": report["triangulation_count"],
            "flip_depth": flip_depth,
            "triangulation_limit": triangulation_limit,
            "obligation": result.statement.obligation,
            "family_complete": (
                False if characterization is None else characterization.family_complete
            ),
            "seed_report": _safe(report["seed"]),
            "status": result.status,
            "evaluated": result.evaluated,
            "search_incomplete": result.search_incomplete,
            "detail": result.detail,
            "hit": None if result.check is None else _safe(result.check.payload),
        })
        if result.status == "PROVED":
            hit = True
            break
    honest = all(
        run["status"] == "PROVED"
        or (run["status"] == "BLOCKED" and run["search_incomplete"])
        for run in runs
    )
    return {
        "name": "gp4_symbolic_search",
        "passed": honest,
        "target_hit": hit,
        "budget_per_seed": actual_budget,
        "seeds": len(runs),
        "seed_ids": seed_ids[: len(runs)],
        "flip_depth": flip_depth,
        "triangulation_limit": triangulation_limit,
        "triangulation_count": len(bank),
        "evaluated": sum(int(run["evaluated"]) for run in runs),
        "searched_scheme": identity["searched_scheme"],
        "searched_tree": _safe(identity["searched_tree"]),
        "search_incomplete": not hit,
        "scope": (
            "A hit is an exact regular T-curve witness. A miss does not exclude "
            "the target scheme or its T-curve realization."
        ),
    }, runs


def _gp5(gp4: dict[str, Any]) -> dict[str, Any]:
    from omnibias.geometry.algebraic import PolygonalAnnulus
    from omnibias.geometry.algebraic_route2 import search_route2_coefficients, square_polygon
    from omnibias.geometry.part_a_target import PARTA_SELECTED_TARGET

    target_hit = bool(gp4["target_hit"])
    annuli = [
        PolygonalAnnulus(
            square_polygon(i, j, Fraction(1, 1024)),
            square_polygon(i, j, Fraction(1, 32)),
        )
        for i in (-3, -1, 1, 3)
        for j in (-3, -1, 1, 3)
    ]
    route2 = search_route2_coefficients(annuli, [1] * len(annuli))
    return {
        "name": "gp5_direct_octic_acceptance",
        "passed": False,
        "target_patchwork_found": target_hit,
        "route2_candidate_found": route2.candidate_found,
        "route2_curve_certificate_passed": route2.curve_certificate_passed,
        "explicit_rational_polynomial": None,
        "complete_real_scheme": False,
        "target_tree": _safe(PARTA_SELECTED_TARGET.tree),
        "selected_target": PARTA_SELECTED_TARGET.scheme,
        "status": "awaiting_target_candidate" if not target_hit else "awaiting_direct_annuli",
        "scope": (
            "No Viro asymptotic is substituted for direct coefficient-level "
            "complex smoothness and 22 whole-boundary oval certificates."
        ),
    }


def run(
    *,
    full: bool = False,
    budget: int | None = None,
    seeds: int | None = None,
    seed_start: int = 0,
) -> dict[str, Any]:
    from omnibias.geometry.part_a_target import PARTA_SELECTED_TARGET

    started = time.perf_counter()
    gp1, gp2, gp3 = _gp1(), _gp2(), _gp3()
    gp4, search_runs = _gp4(
        full=full,
        budget=budget,
        seeds=seeds,
        seed_start=seed_start,
    )
    gp5 = _gp5(gp4)
    entries = [gp1, gp2, gp3, gp4, gp5]
    return {
        **provenance(
            schema="omnibias.benchmarks.patchwork_octic.v1",
            config={
                "full": full,
                "budget": budget,
                "seeds": seeds,
                "seed_start": seed_start,
                "flip_depth": gp4["flip_depth"],
                "triangulation_limit": gp4["triangulation_limit"],
                "seed_ids": gp4["seed_ids"],
                "searched_scheme": PARTA_SELECTED_TARGET.scheme,
            },
        ),
        "gates": dict(gates_block(entries)),
        "search_runs": search_runs,
        "target": "<4 + 1<2 + 1<14>>>",
        "searched_scheme": PARTA_SELECTED_TARGET.scheme,
        "searched_tree": _safe(PARTA_SELECTED_TARGET.tree),
        "target_hit": gp4["target_hit"],
        "full_hilbert16_solved": False,
        "disclaimer": (
            "Finite differentiable proposal plus exact-Q patchwork verification. "
            "A finite miss is search_incomplete. No H(n), limit-cycle, or full "
            "Hilbert-XVI claim."
        ),
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run(
        full=args.full,
        budget=args.budget,
        seeds=args.seeds,
        seed_start=args.seed_start,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            __import__("json").dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        path = args.output
    elif args.full:
        destination = SCRATCH / "patchwork_octic"
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "patchwork_octic.json"
        path.write_text(
            __import__("json").dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        path = write_json("patchwork_octic_smoke.json", payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
