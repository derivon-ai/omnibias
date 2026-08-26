# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: BEM-Net (theory 02-06). Off-surface exact; BC approximated.

G2 disc-accuracy is earned: ``circle_dirichlet_density`` hits the
annulus ``L2`` gate. Single-layer wall vs ``n_quad`` is reported. G3
exterior win is leftover-recorded (leftover #30): no volume-PINN loop.
G3 is not in CI ``all_passed``.
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
    """Named leftover: single-layer wall vs n_quad; not a cost win."""
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
        "note": (
            "Single-layer wall vs n_quad (one far point and N exterior "
            "points). G2 disc-accuracy is earned separately via "
            "circle_dirichlet_density. G3 exterior win stays leftover "
            "(no volume PINN). Dense N-point eval is the O(N^2) honesty "
            "bound (no 2-D FMM). Not a cost win."
        ),
    }


def _run_g2() -> dict[str, Any]:
    """Named G2: exterior Dirichlet disc annulus L2 via the Fourier density."""
    from omnibias.pinn.bem._core import (
        KernelSpec,
        Surface,
        annulus_rel_l2,
        circle_dirichlet_density,
        exterior_disc_field,
    )

    n = 48
    surface = Surface("circle", radius=1.0, n_quad=n)
    kernel = KernelSpec("laplace", dimension=2)
    g = [math.cos(2.0 * math.pi * i / n) for i in range(n)]
    phi = circle_dirichlet_density(surface, g)
    rel, skill = annulus_rel_l2(surface, phi, kernel, exterior_disc_field)
    earned = bool(rel <= 1e-8 and skill > 0.0)
    return {
        "name": "g2_disc_accuracy",
        "passed": earned,
        "earned": earned,
        "reported": True,
        "leftover_recorded": False,
        "leftover_id": 24,
        "leftover_tick": 75,
        "stays_full": False,
        "in_ci_all_passed": earned,
        "need": "exterior Dirichlet disc relative L2 <= 1e-8 on a test annulus, skill > 0",
        "rel_l2": float(rel),
        "skill_vs_zero": float(skill),
        "n_quad": n,
        "dirichlet_solver": "circle_dirichlet_density",
        "note": (
            "Leftover #24 earned on tick #75: circle_dirichlet_density "
            "is the single-layer Fourier solve (mode k scaled by 2k), "
            "not a train. Annulus L2 vs exterior_disc_field. In CI "
            "all_passed."
        ),
    }


def _run_g3() -> dict[str, Any]:
    """Named leftover: pack-tree crossover is unearned; volume PINN stays --full."""
    pack_path = Path(__file__).resolve().parent.parent / "docs" / "benchmarks" / "pack_tree_smoke.json"
    pack = __import__("json").loads(pack_path.read_text(encoding="utf-8"))
    pack_g3 = pack["g3"]
    crossover = pack_g3.get("crossover_m")
    hier_over = float(pack_g3["hier_over_dense_at_m_hi"])
    return {
        "name": "g3_exterior_win",
        "passed": False,
        "earned": False,
        "reported": True,
        "in_ci_all_passed": False,
        "need": "100x far-field vs truncated volumetric PINN, five seeds",
        "volume_pinn": False,
        "pack_tree_crossover_m": crossover,
        "pack_tree_hier_over_dense_at_m_hi": hier_over,
        "pack_tree_far_eval_is_per_source_taylor": False,
        "stays_full": True,
        "leftover_recorded": True,
        "leftover_id": 30,
        "leftover_tick": 62,
        "note": (
            "Leftover #30 leftover-recorded: named G3 is a 100x "
            "far-field win versus a truncated volumetric PINN at "
            "matched cost. No density solve or volume-PINN loop is "
            "wired. Pack-tree 02-07 G3 now has a dense crossover "
            f"(hier/dense {hier_over:.2f} at M=3200, "
            f"crossover_m={crossover!r}); that is not a volume-PINN "
            "bake-off. "
            "Previous spec-status smoke/--full line withdrawn. Not in "
            "CI all_passed."
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
    g2 = _run_g2()
    g3 = _run_g3()
    entries.append(
        {
            "name": "g2_disc_accuracy",
            "passed": bool(g2["passed"]),
            "in_ci_all_passed": bool(g2["in_ci_all_passed"]),
            "rel_l2": g2["rel_l2"],
        }
    )
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.bem_net.v1",
        config={
            "mode": "full" if args.full else "smoke",
            "cost_in_all_passed": False,
            "g2_in_all_passed": bool(g2["in_ci_all_passed"]),
            "g3_in_all_passed": False,
            "gates_in_scope": ["g1", "g2", "g5"],
        },
    )
    payload["gates"] = gates_block(entries)
    payload["cost"] = cost
    payload["g2"] = g2
    payload["g3"] = g3
    payload["honesty"] = {
        "pde_exact": "off-surface by construction",
        "bc": "approximated",
        "scope": "linear constant-coeff homogeneous",
        "founding_bias_collapse": True,
        "temperature_collapse": False,
        "g2_disc_accuracy_earned": bool(g2["earned"]),
        "g2_reported": True,
        "g2_leftover_recorded": False,
        "g2_leftover_id": 24,
        "g2_leftover_tick": 75,
        "g2_stays_full": False,
        "g3_exterior_win_earned": False,
        "g3_reported": True,
        "g3_leftover_recorded": True,
        "g3_leftover_id": 30,
        "g3_leftover_tick": 62,
        "g3_in_ci_all_passed": False,
        "g3_volume_pinn": False,
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
