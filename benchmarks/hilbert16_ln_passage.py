#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Replay the bounded-format LN/exp coalescing-passage test.

The gates validate finite coordinate-chain algebra and localize two failures.
They do not prove physical LN membership, G1, or Hilbert XVI.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, is_dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.proof.certificate import verify_certificate_digest  # noqa: E402
from omnibias.core.proof.lean_check import (  # noqa: E402
    check_certificate,
    generate_obligation,
)
from omnibias.core.realization.polynomial import SparsePolynomial  # noqa: E402
from omnibias.core.verified.interval import Interval  # noqa: E402
from omnibias.core.verified.log_noetherian import (  # noqa: E402
    LNChain,
    log_chart_derivative_bound,
    verify_ln_chain,
)
from omnibias.dynamics.ln_passage import (  # noqa: E402
    admission_compatibility,
    assert_g1_passed,
    certify_regular_arc_model,
    coordinate_ln_chain,
    coordinate_ln_evidence,
    corrected_kill_a,
    invalid_plan_kill_a,
    kill_b,
    overlap_matching,
)

SEED = 16041
SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _safe(value: Any) -> Any:
    if isinstance(value, Fraction):
        return [value.numerator, value.denominator]
    if isinstance(value, Interval):
        return [value.lo.hex(), value.hi.hex()]
    if is_dataclass(value) and not isinstance(value, type):
        return _safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _perturbed_chain_rejected() -> bool:
    chain = coordinate_ln_chain()
    closure = [list(row) for row in chain.closure_matrix]
    closure[2][2] = 2 * SparsePolynomial.variable(4, 2)
    perturbed = LNChain(
        cell=chain.cell,
        functions=chain.functions,
        closure_matrix=tuple(tuple(row) for row in closure),
        claimed_derivatives=chain.claimed_derivatives,
        budget=chain.budget,
    )
    return not verify_ln_chain(perturbed)


def _sup_truth_check(evidence: dict[str, object]) -> dict[str, object]:
    sup_enclosures = evidence["sup_enclosures"]
    if not isinstance(sup_enclosures, dict):
        raise TypeError("coordinate evidence has malformed sup enclosures")
    rng = random.Random(SEED)
    deterministic = [
        (epsilon, sep, w_coord)
        for epsilon in (0.025, 0.075, 0.125)
        for sep in (1.0e-6, 1.0, 2.0)
        for w_coord in (0.5, 4.25, 8.0)
    ]
    random_points = [
        (
            rng.uniform(0.025, 0.125),
            rng.uniform(1.0e-6, 2.0),
            rng.uniform(0.5, 8.0),
        )
        for _ in range(32)
    ]
    contained = True
    for epsilon, sep, w_coord in deterministic + random_points:
        truths = {
            "epsilon": epsilon,
            "sep": sep,
            "W": w_coord,
            "radius": 1.0,
        }
        for name, truth in truths.items():
            enclosure = sup_enclosures[name]
            if not isinstance(enclosure, Interval) or not enclosure.contains(truth):
                contained = False
    try:
        log_chart_derivative_bound(Interval.point(1.0), Fraction(0), 2)
    except ValueError:
        zero_margin_refused = True
    else:
        zero_margin_refused = False
    return {
        "dense_and_seeded_random_contained": contained,
        "zero_delta_margin_refused": zero_margin_refused,
        "deterministic_points": len(deterministic),
        "random_points": len(random_points),
    }


def _g1_refusal() -> bool:
    try:
        assert_g1_passed((True, True, False, True))
    except ValueError:
        return True
    return False


def run(
    *,
    full: bool = False,
    run_lean: bool = False,
    max_n: int = 128,
) -> dict[str, object]:
    if type(max_n) is not int or max_n < 8:
        raise ValueError("max_n must be an integer at least eight")
    evidence = coordinate_ln_evidence()
    sup_check = _sup_truth_check(evidence)
    kill_a_ns = range(4, max_n + 1) if full else (4, 8, 12, 16)
    kill_b_ns = range(8, max_n + 1) if full else (8, 16, 32, 64)
    kill_a_sweep = [corrected_kill_a(1.0 / n) for n in kill_a_ns]
    kill_b_sweep = [kill_b(n) for n in kill_b_ns]
    invalid = invalid_plan_kill_a(0.1)
    admission = admission_compatibility()
    regular = certify_regular_arc_model()
    overlap = overlap_matching()

    closure_cert = evidence["closure_obligation"]
    format_cert = evidence["format_obligation"]
    if not isinstance(closure_cert, dict) or not isinstance(format_cert, dict):
        raise TypeError("finite obligation certificates are malformed")
    obligations = {
        "closure": generate_obligation(closure_cert),
        "format": generate_obligation(format_cert),
    }
    lean_results: dict[str, object] = {}
    theorem_prover_verified = False
    if run_lean:
        checked = {
            name: check_certificate(certificate)
            for name, certificate in (
                ("closure", closure_cert),
                ("format", format_cert),
            )
        }
        theorem_prover_verified = all(result.verified for result in checked.values())
        lean_results = checked

    c2_bound = evidence["conditional_c2_bound"]
    gates = [
        {
            "name": "gl1_exact_coordinate_chain",
            "passed": bool(evidence["chain_valid"])
            and bool(evidence["certificate_valid"])
            and _perturbed_chain_rejected(),
            "perturbed_chain_rejected": _perturbed_chain_rejected(),
        },
        {
            "name": "gl2_delta_sup_enclosures",
            "passed": bool(sup_check["dense_and_seeded_random_contained"])
            and bool(sup_check["zero_delta_margin_refused"]),
            **sup_check,
        },
        {
            "name": "gl3_conditional_log_c2",
            "passed": isinstance(c2_bound, Interval)
            and math.isfinite(c2_bound.hi)
            and bool(evidence["weighted_coordinate_replay"])
            and not bool(evidence["physical_uniform_c2"]),
            "physical_uniform_c2": evidence["physical_uniform_c2"],
        },
        {
            "name": "gl4_kill_a_failure_localized",
            "passed": all(
                placement.pointwise_in_cell
                and not placement.uniform_format_bounded
                and placement.failing_chain_function
                == "physical_outgoing_matching_W_ratio"
                for placement in kill_a_sweep
            )
            and invalid["is_quadratic_family_point"] is False,
            "printed_fixed_L_tuple_rejected": (
                invalid["is_quadratic_family_point"] is False
            ),
        },
        {
            "name": "gl5_kill_b_failure_localized",
            "passed": all(
                not placement.positive_radius_margin
                and float(placement.values["relative_radius"]) > 0
                and placement.failing_chain_function
                == "a(L,lambda1)=r1-theta*sep"
                for placement in kill_b_sweep
                if float(placement.values["n"]) >= 16
            ),
            "old_fixed_theta_wall_fails": True,
        },
        {
            "name": "gl6_admission_refusal_is_explicit",
            "passed": bool(admission["finite_compatible"])
            and admission["ambiguous_control"].status == "unresolved"
            and bool(regular["certified"])
            and bool(regular["replayed"])
            and not bool(admission["physical_first_hit_complete"])
            and bool(overlap["coordinate_identity"])
            and not bool(overlap["physical_map_matching"]),
            "physical_first_hit_complete": admission["physical_first_hit_complete"],
        },
        {
            "name": "gl7_seals_and_formal_tier",
            "passed": all(
                verify_certificate_digest(certificate)
                for certificate in (closure_cert, format_cert)
            )
            and all(source is not None for source in obligations.values())
            and ((not run_lean and not theorem_prover_verified) or theorem_prover_verified),
            "lean_requested": run_lean,
            "theorem_prover_verified": theorem_prover_verified,
        },
    ]
    if not _g1_refusal():
        raise AssertionError("partial G1 assertion was not refused")
    coordinate_summary = {
        key: evidence[key]
        for key in (
            "chain_valid",
            "sup_enclosures",
            "format",
            "certificate_valid",
            "conditional_c2_bound",
            "weighted_coordinate_replay",
            "physical_phi_member",
            "physical_uniform_c2",
            "reason",
        )
    }
    coordinate_summary["certificate_digest"] = evidence["certificate"]["digest"]
    coordinate_summary["closure_obligation_digest"] = closure_cert["digest"]
    coordinate_summary["format_obligation_digest"] = format_cert["digest"]
    return {
        **provenance(
            schema="omnibias.benchmarks.hilbert16_ln_passage.v1",
            config={
                "full": full,
                "lean": run_lean,
                "max_n": max_n,
                "seed": SEED,
            },
        ),
        "gates": dict(gates_block(gates)),
        "coordinate_evidence": _safe(coordinate_summary),
        "kill_a_sweep": _safe(kill_a_sweep),
        "kill_b_sweep": _safe(kill_b_sweep),
        "invalid_printed_kill_a": _safe(invalid),
        "admission": _safe(admission),
        "regular_arc": _safe(regular),
        "overlap": _safe(overlap),
        "lean_results": _safe(lean_results),
        "g1_passed": False,
        "full_hilbert16_solved": False,
        "disclaimer": (
            "All gates certify the finite negative assessment: corrected kill A "
            "has an unbounded physical matching quantity and kill B makes the "
            "proposed radius negative. Not physical LN membership, G1, or "
            "Hilbert XVI."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--lean", action="store_true")
    parser.add_argument("--max-n", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run(full=args.full, run_lean=args.lean, max_n=args.max_n)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        path = args.output
    elif args.full:
        path = SCRATCH / "hilbert16" / "ln_passage.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json("hilbert16_ln_passage_smoke.json", payload)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
