#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Systematic finite-family Route-2 search for the wide/deep octic target."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.geometry.algebraic_route2 import OcticSymmetry
from omnibias.geometry.algebraic_route2_search import Route2Candidate, Route2FiniteFamily


def _load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    return raw if isinstance(raw, dict) else {}


def _candidate_payload(candidate: Route2Candidate) -> dict[str, object]:
    return candidate.as_dict()


def _candidate_from_payload(raw: object) -> Route2Candidate | None:
    if not isinstance(raw, dict):
        return None
    coordinates = raw.get("coordinates")
    depth = raw.get("subdivision_depth")
    anchor = raw.get("anchor", 0)
    symmetry = raw.get("symmetry", "d4")
    if (
        not isinstance(coordinates, list)
        or not all(type(value) is int for value in coordinates)
        or type(depth) is not int
        or type(anchor) is not int
        or symmetry not in ("d4", "klein", "central", "full")
    ):
        return None
    return Route2Candidate(
        tuple(coordinates),
        depth,
        anchor,
        cast(OcticSymmetry, symmetry),
    )


def _write_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def _run_best_first_tier(
    family: Route2FiniteFamily,
    *,
    budget: int,
    checkpoint: Path,
    state: dict[str, Any],
    seed_all_anchors: bool,
    stagnation_window: int,
    stagnation_tolerance: float,
) -> dict[str, Any]:
    tiers_state = state.setdefault("tiers", {})
    saved = tiers_state.setdefault(family.name, {})
    family.load_score_checkpoint(saved.get("scores", []))
    frontier = {
        candidate
        for candidate in (
            _candidate_from_payload(raw) for raw in saved.get("frontier", [])
        )
        if candidate is not None
    }
    seen = {
        candidate
        for candidate in (_candidate_from_payload(raw) for raw in saved.get("seen", []))
        if candidate is not None
    }
    evaluated = int(saved.get("evaluated", 0))
    last_payload = saved.get("last_payload")
    last_candidate = saved.get("last_candidate")
    history = list(saved.get("history", []))
    ancestry = list(saved.get("ancestry", []))
    if not frontier and evaluated == 0:
        origin = family.origin()
        seeds = (
            tuple(
                Route2Candidate(
                    origin.coordinates,
                    origin.subdivision_depth,
                    anchor,
                    origin.symmetry,
                )
                for anchor in range(family.anchor_count)
            )
            if seed_all_anchors
            else (origin,)
        )
        frontier.update(seeds)
        seen.update(seeds)
        ancestry.extend(
            {"parent": None, "child": _candidate_payload(seed), "relation": "seed"}
            for seed in seeds
        )
    hit = bool(saved.get("hit", False))
    witness = saved.get("witness")
    while frontier and evaluated < budget and not hit:
        candidate = max(
            frontier,
            key=lambda item: (
                family.score(item),
                item.coordinates,
                item.subdivision_depth,
                item.anchor,
            ),
        )
        frontier.remove(candidate)
        checked = family.check(candidate)
        evaluated += 1
        last_candidate = _candidate_payload(candidate)
        last_payload = checked.payload if checked is not None else None
        history.append(
            {
                "candidate": last_candidate,
                "payload": last_payload,
            }
        )
        if checked is not None and checked.ok:
            hit = True
            witness = {
                "candidate": _candidate_payload(candidate),
                "payload": checked.payload,
            }
        else:
            for neighbor in family.neighbors(candidate):
                if neighbor not in seen:
                    seen.add(neighbor)
                    frontier.add(neighbor)
                    relation = (
                        "bernstein_depth_relaxation"
                        if neighbor.subdivision_depth != candidate.subdivision_depth
                        else "projective_or_geometry_neighbor"
                    )
                    ancestry.append(
                        {
                            "parent": _candidate_payload(candidate),
                            "child": _candidate_payload(neighbor),
                            "relation": relation,
                        }
                    )
        saved.update(
            {
                "evaluated": evaluated,
                "hit": hit,
                "witness": witness,
                "last_payload": last_payload,
                "last_candidate": last_candidate,
                "frontier": [
                    _candidate_payload(item)
                    for item in sorted(
                        frontier,
                        key=lambda item: (
                            item.subdivision_depth,
                            item.coordinates,
                            item.anchor,
                        ),
                    )
                ],
                "seen": [
                    _candidate_payload(item)
                    for item in sorted(
                        seen,
                        key=lambda item: (
                            item.subdivision_depth,
                            item.coordinates,
                            item.anchor,
                        ),
                    )
                ],
                "scores": family.score_checkpoint(),
                "top_scores": family.top_scores(),
                "history": history,
                "ancestry": ancestry,
            }
        )
        _write_checkpoint(checkpoint, state)
    exhausted = not frontier and evaluated >= family.cardinality()
    recent_margins = [
        float(payload["margin_float"])
        for row in history[-stagnation_window:]
        if isinstance((payload := row.get("payload")), dict)
        and isinstance(payload.get("margin_float"), int | float)
    ]
    stagnant = (
        len(recent_margins) == stagnation_window
        and max(recent_margins) - min(recent_margins) <= stagnation_tolerance
    )
    return {
        "kind": "finite_discovery",
        "status": "PROVED" if hit else "BLOCKED",
        "family": family.name,
        "symmetry": family.symmetry,
        "basis_dimension": family.anchor_count // 2,
        "lattice": family.lattice_spec(),
        "budget": budget,
        "evaluated": evaluated,
        "cardinality": family.cardinality(),
        "frontier_size": len(frontier),
        "seen": len(seen),
        "search_incomplete": not hit and not exhausted,
        "exhausted": exhausted,
        "stagnant": stagnant,
        "witness": witness,
        "last_payload": last_payload,
        "last_candidate": last_candidate,
        "score_checkpoint_count": len(family.score_checkpoint()),
        "top_scores": family.top_scores(),
        "history": history,
        "ancestry": ancestry,
    }


def _search_tiers(
    *,
    tiers: tuple[OcticSymmetry, ...],
    budget: int,
    max_depth: int,
    certify_infeasible: bool,
    checkpoint: Path,
    fine: bool,
    origin_candidate: Route2Candidate | None,
    stagnation_window: int,
    stagnation_tolerance: float,
) -> dict[str, Any]:
    state = _load_checkpoint(checkpoint)
    state.setdefault("schema", "omnibias.route2.frontier.v1")
    results: list[dict[str, Any]] = []
    hit = False
    for symmetry in tiers:
        family = Route2FiniteFamily(
            symmetry=symmetry,
            max_subdivision_depth=max_depth,
            certify_infeasible=certify_infeasible,
            fine=fine,
            origin_candidate=origin_candidate,
        )
        tier_budget = min(budget, family.cardinality())
        payload = _run_best_first_tier(
            family,
            budget=tier_budget,
            checkpoint=checkpoint,
            state=state,
            seed_all_anchors=certify_infeasible and origin_candidate is None,
            stagnation_window=stagnation_window,
            stagnation_tolerance=stagnation_tolerance,
        )
        payload["max_subdivision_depth"] = max_depth
        results.append(payload)
        if payload["status"] == "PROVED":
            hit = True
            break
        if not payload["exhausted"] and not payload["stagnant"]:
            break
    _write_checkpoint(checkpoint, state)
    checkpoint_digest = hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "hit": hit,
        "tiers": results,
        "checkpoint": str(checkpoint),
        "checkpoint_digest": checkpoint_digest,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--artifact")
    parser.add_argument("--fine", action="store_true")
    parser.add_argument("--start-coordinates")
    parser.add_argument("--start-depth", type=int, default=0)
    parser.add_argument("--start-anchor", type=int, default=0)
    parser.add_argument("--stagnation-window", type=int, default=8)
    parser.add_argument("--stagnation-tolerance", type=float, default=1e-8)
    parser.add_argument(
        "--tiers",
        nargs="+",
        choices=("d4", "klein", "central", "full"),
    )
    args = parser.parse_args()
    if args.stagnation_window < 1 or args.stagnation_tolerance < 0:
        parser.error("stagnation window must be positive and tolerance nonnegative")
    tiers: tuple[OcticSymmetry, ...] = (
        tuple(args.tiers)  # type: ignore[assignment]
        if args.tiers
        else (("d4", "klein", "central", "full") if args.full else ("d4",))
    )
    budget = args.budget if args.budget is not None else (64 if args.full else 1)
    max_depth = args.max_depth if args.max_depth is not None else (2 if args.full else 0)
    scratch = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
    checkpoint = args.checkpoint or scratch / "hilbert16_route2_frontier.json"
    origin_candidate = None
    if args.start_coordinates:
        origin_candidate = Route2Candidate(
            tuple(int(value) for value in args.start_coordinates.split(",")),
            args.start_depth,
            args.start_anchor,
            tiers[0],
        )
    started = time.time()
    payload = _search_tiers(
        tiers=tiers,
        budget=budget,
        max_depth=max_depth,
        certify_infeasible=args.full,
        checkpoint=checkpoint,
        fine=args.fine,
        origin_candidate=origin_candidate,
        stagnation_window=args.stagnation_window,
        stagnation_tolerance=args.stagnation_tolerance,
    )
    entries = [
        {
            "name": "route2_layout_lp_sweep",
            "passed": payload["hit"],
            "payload": payload,
            "detail": "finite best-first search; only a sealed curve is a hit",
        }
    ]
    report = {
        **provenance(
            schema="omnibias.benchmarks.hilbert16_route2_sweep.v2",
            config={
                "family": "hilbert16_route2_sweep",
                "full": args.full,
                "tiers": list(tiers),
                "budget": budget,
                "max_depth": max_depth,
                "fine": args.fine,
                "origin_candidate": (
                    origin_candidate.as_dict() if origin_candidate is not None else None
                ),
                "stagnation_window": args.stagnation_window,
                "stagnation_tolerance": args.stagnation_tolerance,
            },
        ),
        "benchmark": "hilbert16_route2_sweep",
        "gates": gates_block(entries),
        "full_hilbert16_solved": False,
        "hilbert16_part_a_solved": False,
        "wall_seconds": time.time() - started,
    }
    artifact = args.artifact or (
        "hilbert16_route2_sweep.json"
        if args.full
        else "hilbert16_route2_sweep_smoke.json"
    )
    path = write_json(artifact, report)
    print(f"wrote {path}")
    return 0 if report["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
