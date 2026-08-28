# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Theory 10-03 (G5): certified truncation-horizon coverage sweep.

For each random contracting linear system, ``certified_horizon`` derives the
smallest window ``h`` whose enclosed closed-loop monodromy has spectral-
radius upper bound ``<= tol``. Two things are checked over the sweep,
never conflated into one number:

1. **Coverage of the enclosed upper bound.** The realized point-matrix
   decay ``max(|diag|)**h`` must lie in the sound interval
   ``[0, spectral_radius_bound_hi]`` the certificate reports -- this is the
   genuine ``require_enclosure_coverage`` claim (coverage must be exactly
   ``1.0``; a miss means the enclosure was unsound, not merely wide).
2. **Certification rate.** Every contracting system in the declared regime
   must actually receive a certificate (``certified=True``) within
   ``max_horizon`` steps; this is reported separately, never co-gated with
   coverage, so a wide-but-sound certificate cannot be hidden behind a
   high certification rate or vice versa.

This is founding bias collapse (``delta -> 0``), not temperature collapse
(``beta -> inf``, the feasibility sense): the certificate is a sound
enclosure of a fixed finite product. Do not conflate the two.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import (  # type: ignore[import-not-found]  # noqa: E402
    gates_block,
    require_enclosure_coverage,
)
from omnibias.control.horizon import DISCLAIMER, certified_horizon  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _sweep(*, n: int, seed: int, max_horizon: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    enclosures: list[tuple[float, float]] = []
    truths: list[float] = []
    n_certified = 0
    horizons: list[int] = []
    for _ in range(n):
        dim = int(rng.choice((1, 2, 3)))
        diag = rng.uniform(0.2, 0.8, size=dim)
        m = np.diag(diag)
        radius = float(rng.uniform(0.0, 0.02))
        tol = float(rng.uniform(0.02, 0.1))
        result = certified_horizon(
            [m] * max_horizon, radius=radius, tol=tol, max_horizon=max_horizon
        )
        if not result.certified:
            continue
        assert result.horizon is not None
        assert result.spectral_radius_bound_hi is not None
        n_certified += 1
        horizons.append(result.horizon)
        actual_decay = float(np.max(diag) ** result.horizon)
        enclosures.append((0.0, result.spectral_radius_bound_hi))
        truths.append(actual_decay)

    coverage = require_enclosure_coverage(
        enclosures, truths, n_min=min(n, len(enclosures)), name="certified_horizon_coverage"
    )
    return {
        "n_trials": n,
        "n_certified": n_certified,
        "certification_rate": n_certified / n,
        "median_horizon": float(np.median(horizons)) if horizons else float("nan"),
        "coverage": coverage,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    t0 = time.perf_counter()

    n = 1000 if full else 60
    sweep = _sweep(n=n, seed=20260827, max_horizon=60)

    entries = [
        {
            "name": "g5_certification_rate",
            "passed": bool(sweep["certification_rate"] == 1.0),
            "certification_rate": sweep["certification_rate"],
            "n_trials": sweep["n_trials"],
            "n_certified": sweep["n_certified"],
        },
        sweep["coverage"],
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")

    payload = {
        **provenance(
            schema="omnibias.benchmarks.certified_horizon.v1",
            config={"family": "certified_horizon", "full": full, "n_trials": n},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "median_horizon": sweep["median_horizon"],
        "honesty": {
            "n_min_1000_full_run": full,
            "note": "the >= 1000-trial acceptance run requires --full; the "
            "default smoke uses a reduced n for CI speed",
        },
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "certified_horizon.json" if full else "certified_horizon_smoke.json"
    if full:
        dest = SCRATCH / "control" / "certified"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
