# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 04-02: conformal slabs.

Smoke checks the worked quantile, type safety, adaptive win,
shift diagnostic, and combination. Full repeats G1 over more
resamples. Conformal intervals are not sealed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from omnibias.core.uncertainty import (
    DISCLAIMER,
    CombinedStatement,
    GuaranteeKind,
    UncertaintyInterval,
    adaptive_vs_fixed,
    combine_enclosure_with_conformal,
    honesty_payload,
    refuse_conformal_seal,
    resample_coverage,
    shift_diagnostic,
    theoretical_coverage,
    worked_example,
)

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    t0 = time.perf_counter()
    ex = worked_example()
    g1 = ex["q"] == 0.88
    for alpha in (0.01, 0.05, 0.1, 0.2):
        for n in (19, 99, 999):
            g1 = g1 and theoretical_coverage(n, alpha) + 1e-15 >= 1.0 - alpha
    trials = 2000 if full else 200
    cover = resample_coverage(19, 0.1, trials=trials, seed=1)
    g1 = g1 and cover >= (0.86 if full else 0.80)
    sound = UncertaintyInterval(0.0, 1.0, GuaranteeKind.SOUND_ENCLOSURE)
    conf = UncertaintyInterval(-1.0, 1.0, GuaranteeKind.CONFORMAL, level=0.9)
    add_raised = False
    try:
        _ = sound + conf
    except TypeError:
        add_raised = True
    seal_raised = False
    try:
        refuse_conformal_seal(conf)
    except ValueError:
        seal_raised = True
    skill = adaptive_vs_fixed(seeds=5, n_cal=800 if not full else 2500, n_test=600 if not full else 2000)
    # Smoke uses fewer points; still require the population contrast.
    g3 = bool(skill["g3_earned"]) or (
        not full
        and float(skill["adaptive_mean_dev"]) < float(skill["fixed_mean_dev"])
        and float(skill["adaptive_width"]) <= float(skill["fixed_width"])
        and float(skill["high_noise_fixed_cover"]) <= 0.82
    )
    shift = shift_diagnostic()
    g4 = shift.shift_detected and (not shift.exchangeability_ok) and shift.average_width > 0.0
    stmt = combine_enclosure_with_conformal(
        UncertaintyInterval(2.31, 2.47, GuaranteeKind.SOUND_ENCLOSURE),
        0.88,
        alpha=0.1,
    )
    g5 = isinstance(stmt, CombinedStatement) and seal_raised
    hon = honesty_payload()
    g2 = add_raised
    g6 = shift.average_width > 0.0
    entries = [
        {"name": "g1_cell", "passed": g1, "q": ex["q"], "resample": cover},
        {"name": "g2_types", "passed": g2, "add_raised": add_raised},
        {
            "name": "g3_adaptive",
            "passed": g3,
            "adaptive_max_dev": skill["adaptive_max_dev"],
            "fixed_max_dev": skill["fixed_max_dev"],
        },
        {
            "name": "g4_shift",
            "passed": g4,
            "shift_detected": shift.shift_detected,
            "coverage": shift.marginal_coverage,
        },
        {"name": "g5_combine", "passed": g5, "combined_lo": stmt.combined_lo},
        {"name": "g6_width", "passed": g6, "average_width": shift.average_width},
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.conformal_slabs.v1",
            config={"family": "conformal_slabs", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "conformal_slabs.json" if full else "conformal_slabs_smoke.json"
    if full:
        dest = SCRATCH / "uncertainty"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
