# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3 trainer: certified I/O step (theory 08-09).

Smoke earns G1 (section-5 affine toy: every accepted step has sealed
Lipschitz ``<= P_max`` and at least one rejected step would violate
the cap), G2 (exploding ReLU enclosure is ``vacuous``, not accepted),
and G3 (``robust_without_enclosure`` stays false). Distinct from 08-04
(root of ``F``). Not a global min and not CCF stretch. Bias collapse
(``delta -> 0``) supplies ``sigma'`` in the Lipschitz / Taylor path.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.verified.interval import Interval
from omnibias.verify import (
    CertifiedStepConfig,
    CertifiedStepForbidden,
    CertifiedStepResult,
    Network,
    ReLULayer,
    affine_layer,
    certified_accept,
    honesty_payload,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
P_MAX = 2.0
TRIALS = (0.5, 1.0, 1.5, 1.8, 2.0, 2.01, 2.5, 3.0)


def _affine(w: float) -> Network:
    return Network([affine_layer([[w]], [0.0])])


def _cfg() -> CertifiedStepConfig:
    return CertifiedStepConfig(
        property="lipschitz",
        p_max=P_MAX,
        cell=[Interval(0.0, 1.0)],
    )


def _run_g1() -> dict[str, Any]:
    cfg = _cfg()
    rows: list[dict[str, Any]] = []
    for w in TRIALS:
        result = certified_accept(_affine(1.5), (w, 0.0), config=cfg)
        rows.append(
            {
                "w": w,
                "accepted": result.accepted,
                "reason": result.reason,
                "bound_hi": result.bound_hi,
                "true_lip": abs(w),
            }
        )
    accepted = [r for r in rows if r["accepted"]]
    rejected = [r for r in rows if not r["accepted"]]
    ok_cap = all(
        r["bound_hi"] is not None and r["bound_hi"] <= P_MAX for r in accepted
    )
    ok_sound = all(r["true_lip"] <= (r["bound_hi"] or 0.0) + 1e-12 for r in accepted)
    has_reject = any(r["reason"] == "violates" and r["true_lip"] > P_MAX for r in rejected)
    passed = bool(accepted) and ok_cap and ok_sound and has_reject
    return {
        "name": "g1_accept_reject",
        "passed": passed,
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
        "p_max": P_MAX,
        "rows": rows,
        "note": (
            "Affine f(x)=w x on [0, 1]. Accepted steps have sealed "
            "Lipschitz <= P_max. At least one rejected trial has |w| > P_max."
        ),
    }


def _run_g2() -> dict[str, Any]:
    exploding = Network(
        [
            affine_layer([[1.0e300]], [0.0]),
            ReLULayer(),
            affine_layer([[1.0e300]], [0.0]),
        ]
    )
    boom = certified_accept(
        exploding,
        config=CertifiedStepConfig(
            property="lipschitz",
            p_max=P_MAX,
            cell=[Interval(-1.0e300, 1.0e300)],
        ),
    )
    empty = certified_accept(
        _affine(1.0),
        config=CertifiedStepConfig(p_max=P_MAX, cell=None),
    )
    passed = (
        boom.accepted is False
        and boom.reason == "vacuous"
        and boom.bound_hi is None
        and empty.accepted is False
        and empty.reason == "vacuous"
    )
    return {
        "name": "g2_vacuous",
        "passed": bool(passed),
        "exploding_reason": boom.reason,
        "empty_cell_reason": empty.reason,
        "note": (
            "Empty or exploding enclosures reject with reason='vacuous'. "
            "That is not a robustness claim."
        ),
    }


def _run_g3() -> dict[str, Any]:
    payload = honesty_payload()
    raised = False
    try:
        CertifiedStepResult(
            accepted=False,
            bound_hi=None,
            reason="vacuous",
            p_max=P_MAX,
            robust_without_enclosure=True,
        )
    except CertifiedStepForbidden:
        raised = True
    passed = payload["robust_without_enclosure"] is False and raised
    return {
        "name": "g3_honesty",
        "passed": bool(passed),
        "honesty": payload,
        "forge_raises": raised,
        "note": (
            "Artifact key robust_without_enclosure is sealed false. "
            "Forging it on CertifiedStepResult raises."
        ),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "certified_step.json" if full else "certified_step_smoke.json"
    t0 = time.perf_counter()
    print("G1 accept/reject...")
    g1 = _run_g1()
    print("G2 vacuous...")
    g2 = _run_g2()
    print("G3 honesty...")
    g3 = _run_g3()
    entries = [g1, g2, g3]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.certified_step.v1",
            config={
                "family": "certified_step",
                "full": full,
                "collapse": "bias_collapse",
                "honesty": honesty_payload(),
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "training" / "certified_step"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
