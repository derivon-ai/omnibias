# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Theory 10-03 (G6): policy-gradient truncation-bias enclosure coverage sweep.

Random small linear closed-loop systems where the *true* terminal-adjoint
perturbation ``delta_lambda`` is known exactly, so the realized gradient
bias ``||grad_theta J - grad_theta J_h||`` is computable by direct
recursion. ``truncation_bias_bound`` is fed the exact norm of that
perturbation as its ``terminal_error_bound`` input (the case where the
caller's terminal-error model is exact); the certified bound must never
fall below the realized bias, over >= 1000 draws when ``--full``.

Width is reported, never co-gated with coverage (a tight-but-unsound bound
must never look like a pass because it is narrow).

This is founding bias collapse (``delta -> 0``), not temperature collapse
(``beta -> inf``, the feasibility sense); nothing here sharpens a
``beta -> inf`` gate. Do not conflate the two. The bound stays conditional
on the terminal-error input genuinely representing the true perturbation --
true by construction in this synthetic sweep, not asserted in general (see
``omnibias.control.certified.gradient_bias`` module docstring).
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
from omnibias.control.certified.gradient_bias import (  # noqa: E402
    DISCLAIMER,
    truncation_bias_bound,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _sweep(*, n: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    enclosures: list[tuple[float, float]] = []
    truths: list[float] = []
    bounds: list[float] = []
    for _ in range(n):
        dim = int(rng.choice((1, 2, 3)))
        n_params = int(rng.choice((1, 2, 4)))
        action_dim = int(rng.choice((1, 2)))
        horizon = int(rng.integers(1, 7))
        m_seq = [
            0.7 * np.eye(dim) + 0.05 * (rng.random() - 0.5) * np.eye(dim) for _ in range(horizon)
        ]
        dpi_seq = [
            0.1 * (rng.random() + 0.1) * rng.choice((-1.0, 1.0)) * np.ones((action_dim, n_params))
            for _ in range(horizon)
        ]
        b_seq = [
            0.1 * (rng.random() + 0.1) * rng.choice((-1.0, 1.0)) * np.ones((dim, action_dim))
            for _ in range(horizon)
        ]
        err_true = rng.uniform(-1.0, 1.0, size=dim)
        err_bound = float(np.max(np.abs(err_true)))
        report = truncation_bias_bound(dpi_seq, b_seq, m_seq, err_bound)

        running = np.eye(dim)
        realized = np.zeros(n_params)
        for k in range(horizon - 1, -1, -1):
            state_vec = running.T @ err_true
            action_vec = b_seq[k].T @ state_vec
            realized = realized + dpi_seq[k].T @ action_vec
            running = running @ m_seq[k]
        realized_norm = float(np.max(np.abs(realized))) if realized.size else 0.0

        # A tiny numerical-noise margin: both `report.bound` and `realized_norm`
        # are themselves floating-point computations (not exact rationals), so
        # a genuinely-sound bound can still sit within a few ULP of a realized
        # value computed along a different summation order. The pre-existing
        # `gradient_bias_skill` in the module uses the same `1e-9` margin for
        # exactly this reason; this is not a widening of the *certificate*.
        enclosures.append((0.0, report.bound + 1e-9))
        truths.append(realized_norm)
        bounds.append(report.bound)

    coverage = require_enclosure_coverage(
        enclosures, truths, n_min=n, name="gradient_bias_enclosure_coverage"
    )
    return {
        "n_trials": n,
        "median_bound": float(np.median(bounds)),
        "coverage": coverage,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    t0 = time.perf_counter()

    n = 1000 if full else 60
    sweep = _sweep(n=n, seed=20260827)

    entries = [sweep["coverage"]]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")

    payload = {
        **provenance(
            schema="omnibias.benchmarks.gradient_bias_enclosure.v1",
            config={"family": "gradient_bias_enclosure", "full": full, "n_trials": n},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "median_bound": sweep["median_bound"],
        "honesty": {
            "n_min_1000_full_run": full,
            "note": "the >= 1000-draw acceptance run requires --full; the "
            "default smoke uses a reduced n for CI speed. terminal_error_bound "
            "is fed the exact perturbation norm in this synthetic sweep -- the "
            "case where the caller's error model is exact, not a general claim",
        },
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "gradient_bias_enclosure.json" if full else "gradient_bias_enclosure_smoke.json"
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
