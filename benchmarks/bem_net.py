# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: BEM-Net (theory 02-06). Off-surface exact; BC approximated.

G2 disc-accuracy stays smoke/``--full``. Single-layer wall vs ``n_quad``
is reported, not in CI ``all_passed``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

COST_NS = (12, 24, 48)
COST_WARMUP = 1
COST_REPEATS = 3


def _median_seconds(fn: Any, *, warmup: int, repeats: int) -> float:
    for _ in range(int(warmup)):
        fn()
    samples: list[float] = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def _run_cost() -> dict[str, Any]:
    """Named leftover: single-layer wall vs n_quad; G2 disc L2 stays --full."""
    from omnibias.pinn.bem._core import KernelSpec, Surface, single_layer

    kernel = KernelSpec("laplace", dimension=2)
    rows: list[dict[str, Any]] = []
    for n in COST_NS:
        surface = Surface("circle", radius=1.0, n_quad=n)
        dens = [0.1] * n

        def _one(s=surface, d=dens) -> None:
            single_layer((2.0, 0.0), s, d, kernel)

        def _dense(s=surface, d=dens, n_pts=n) -> None:
            for i in range(n_pts):
                th = 2.0 * math.pi * i / n_pts
                single_layer((2.0 * math.cos(th), 2.0 * math.sin(th)), s, d, kernel)

        rows.append(
            {
                "n_quad": int(n),
                "one_point_wall_seconds": float(
                    _median_seconds(_one, warmup=COST_WARMUP, repeats=COST_REPEATS)
                ),
                "n_point_wall_seconds": float(
                    _median_seconds(_dense, warmup=COST_WARMUP, repeats=COST_REPEATS)
                ),
            }
        )
    return {
        "name": "cost_single_layer_vs_n",
        "passed": False,
        "earned": False,
        "reported": True,
        "in_ci_all_passed": False,
        "rows": rows,
        "g2_disc_accuracy": {
            "earned": False,
            "stays_full": True,
            "need": "exterior Dirichlet disc relative L2 <= 1e-8 on a test annulus, skill > 0",
            "reason": (
                "No Dirichlet density solve is wired. Constant-density "
                "single_layer wall vs n_quad is recorded; that is not the "
                "named annulus L2 gate. Dense N-point eval is the O(N^2) "
                "honesty bound (no 2-D FMM)."
            ),
        },
        "note": (
            "single_layer wall vs n_quad (one far point and N exterior "
            "points). G2 disc-accuracy is a Dirichlet L2 study under "
            "$OMNIBIAS_SCRATCH, not CI. Previous g2_disc_accuracy "
            "passed=True / smoke/--full stub with no timing withdrawn. "
            "Off-surface PDE exact; BC approximated. Not in CI all_passed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    from omnibias.pinn.bem._core import (
        KernelSpec,
        Surface,
        half_plane_dtn,
        pde_residual_off_surface,
        poisson_pair_dictionary,
        single_layer,
    )

    surface = Surface("circle", radius=1.0, n_quad=12)
    kernel = KernelSpec("laplace", dimension=2)
    dens = [0.1] * 12
    x = (2.0, 0.0)
    res = pde_residual_off_surface(x, surface, dens, kernel)
    mag = abs(single_layer(x, surface, dens, kernel))
    dictionary, coeffs = poisson_pair_dictionary(scale=1.0)
    dtn = half_plane_dtn(dictionary, coeffs, 0.3)
    y = 0.3
    expect = (1.0 - y * y) / (y * y + 1.0) ** 2
    entries: list[dict[str, Any]] = [
        {
            "name": "g1_off_surface",
            "passed": abs(res) <= 1e-13 * max(mag, 1.0),
            "residual": res,
            "in_ci_all_passed": True,
        },
        {
            "name": "g5_half_plane_dtn",
            "passed": abs(dtn - expect) <= 4e-15 * max(abs(expect), 1.0) * 8,
            "in_ci_all_passed": True,
        },
    ]
    cost = _run_cost()
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.bem_net.v1",
        config={
            "mode": "full" if args.full else "smoke",
            "cost_in_all_passed": False,
            "gates_in_scope": ["g1", "g5"],
        },
    )
    payload["gates"] = gates_block(entries)
    payload["cost"] = cost
    payload["honesty"] = {
        "pde_exact": "off-surface by construction",
        "bc": "approximated",
        "scope": "linear constant-coeff homogeneous",
        "founding_bias_collapse": True,
        "temperature_collapse": False,
        "g2_disc_accuracy_earned": False,
        "g2_stays_full": True,
        "cost_earned": False,
        "cost_reported": True,
        "cost_in_ci_all_passed": False,
        "fmm": False,
    }
    if args.full:
        dest = SCRATCH / "bem" / "bem_net.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('bem_net_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
