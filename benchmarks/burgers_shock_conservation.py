# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Conservative vs non-conservative Burgers PINN as the viscosity shrinks.

Does building the conservation law into the architecture buy anything
measurable? The two arms share an architecture, a parameter count, a seed and a
collocation budget; only the residual differs. As ``nu`` falls at fixed budget
the viscous layer becomes under-resolved, which is where a finite-volume scheme
earns its keep over a non-conservative one -- so that is where this looks.

The metric is the **shock speed**, which Rankine-Hugoniot pins at
``(u_L + u_R)/2`` exactly. Relative L2 against the exact wave is reported
alongside it, because the two do not have to move together and reporting only
the favourable one would be dishonest.

The physics, the architectures and the estimators are imported from the example
rather than restated, so the published numbers come from the same code a reader
runs. Run::

    uv run python benchmarks/burgers_shock_conservation.py
"""

from __future__ import annotations

import importlib.util
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from _common import REPO_ROOT, provenance, write_json  # noqa: E402

#: Single-threaded on purpose: the sweep is run as a pool of processes, and
#: torch's default intra-op threading would make the workers fight each other.
torch.set_num_threads(1)


def _floats(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get(name, "").strip()
    return tuple(float(v) for v in raw.split(",")) if raw else default


#: Spans resolved (``2e-2``) to badly under-resolved (``2e-3``) at a fixed
#: collocation budget, which is the axis the comparison is about.
VISCOSITIES = _floats(
    "BURGERS_VISCOSITIES", (2.0e-2, 1.2e-2, 8.0e-3, 5.0e-3, 3.0e-3, 2.0e-3)
)
SEEDS = tuple(int(s) for s in _floats("BURGERS_SEEDS", (0, 1, 2, 3, 4)))
OUT_NAME = os.environ.get("BURGERS_OUT", "burgers_shock_conservation.json")


def _load_example() -> Any:
    """Import ``docs/examples/pinn_burgers_shock.py`` as a module."""
    path = REPO_ROOT / "docs" / "examples" / "pinn_burgers_shock.py"
    spec = importlib.util.spec_from_file_location("_burgers_shock_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_case(nu: float, seed: int) -> dict[str, Any]:
    """One ``(nu, seed)`` cell: train both arms, measure both."""
    ex = _load_example()
    torch.set_default_dtype(ex.DTYPE)
    ex.NU = nu
    ex.SEED = seed

    out: dict[str, Any] = {"nu": nu, "seed": seed}
    for arm, build in (("conservative", ex.build_cage_field), ("baseline", ex.build_plain_field)):
        torch.manual_seed(seed)
        field = build()
        started = time.perf_counter()
        final_loss = ex.train(field, seed=seed)
        speed = ex.shock_speed(field)
        out[arm] = {
            "shock_speed": speed,
            "speed_abs_error": abs(speed - ex.C0) if math.isfinite(speed) else None,
            "mass_balance_error": ex.mass_balance_error(field),
            "relative_l2": ex.solution_error(field),
            "final_loss": final_loss,
            "train_seconds": round(time.perf_counter() - started, 2),
        }
        if arm == "conservative":
            # Recorded after training, so it reports the guarantee under the
            # trained (moved, sharpened) seam rather than at initialisation.
            out[arm]["structural_divergence"] = ex.structural_divergence(
                field, ex._sample(seed)[0]
            )
    cage = out["conservative"]["speed_abs_error"]
    base = out["baseline"]["speed_abs_error"]
    out["speed_error_ratio"] = (base / cage) if (cage and base) else None
    return out


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2e}"


def _aggregate(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Median across seeds, per viscosity. Median, not mean: outliers happen."""
    rows: list[dict[str, Any]] = []
    for nu in VISCOSITIES:
        block = [c for c in cells if c["nu"] == nu]
        row: dict[str, Any] = {"nu": nu, "seeds": len(block)}
        for arm in ("conservative", "baseline"):
            errs = [c[arm]["speed_abs_error"] for c in block]
            errs = [e for e in errs if e is not None]
            mass = [c[arm]["mass_balance_error"] for c in block]
            l2s = [c[arm]["relative_l2"] for c in block]
            row[arm] = {
                "speed_abs_error_median": statistics.median(errs) if errs else None,
                "speed_abs_error_max": max(errs) if errs else None,
                "mass_balance_error_median": statistics.median(mass),
                "mass_balance_error_max": max(mass),
                "relative_l2_median": statistics.median(l2s),
                "converged_cells": len(errs),
            }
        cage = row["conservative"]["speed_abs_error_median"]
        base = row["baseline"]["speed_abs_error_median"]
        row["speed_error_ratio_median"] = (base / cage) if (cage and base) else None
        row["mass_error_ratio_median"] = (
            row["baseline"]["mass_balance_error_median"]
            / row["conservative"]["mass_balance_error_median"]
        )
        for metric, key in (("speed", "speed_abs_error"), ("mass", "mass_balance_error")):
            row[f"conservative_wins_{metric}"] = sum(
                1
                for c in block
                if c["conservative"][key] is not None
                and c["baseline"][key] is not None
                and c["conservative"][key] < c["baseline"][key]
            )
        rows.append(row)
    return rows


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(nu, seed) for nu in VISCOSITIES for seed in SEEDS]
    print(f"burgers shock sweep: {len(jobs)} cells x 2 arms")
    started = time.perf_counter()
    with ProcessPoolExecutor() as pool:
        cells = list(pool.map(_run_star, jobs))
    elapsed = time.perf_counter() - started

    rows = _aggregate(cells)
    for row in rows:
        cage = row["conservative"]
        base = row["baseline"]
        ratio = row["speed_error_ratio_median"]
        tail = f"ratio {ratio:.1f}x" if ratio else "ratio n/a"
        print(
            f"  nu={row['nu']:.1e}  speed |err| cage "
            f"{_fmt(cage['speed_abs_error_median'])}"
            f"  base {_fmt(base['speed_abs_error_median'])}  {tail}"
            f"  wins {row['conservative_wins_speed']}/{row['seeds']}"
        )
        print(
            f"            mass |err| cage {cage['mass_balance_error_median']:.2e}"
            f"  base {base['mass_balance_error_median']:.2e}"
            f"  ratio {row['mass_error_ratio_median']:.1f}x"
            f"  wins {row['conservative_wins_mass']}/{row['seeds']}"
        )
        print(
            f"            rel-L2     cage {cage['relative_l2_median']:.2e}"
            f"  base {base['relative_l2_median']:.2e}"
        )

    worst = max(
        r["conservative"]["speed_abs_error_max"]
        for r in rows
        if r["conservative"]["speed_abs_error_max"] is not None
    )
    example = _load_example()
    payload = provenance(
        schema="burgers_shock_conservation/v1",
        config={
            "viscosities": list(VISCOSITIES),
            "seeds": list(SEEDS),
            "collocation_points": example.N_INTERIOR,
            "adam_steps": example.STEPS,
            "exact_shock_speed": example.C0,
            "note": "both arms share architecture, parameter count, seed and budget",
        },
    )
    payload["by_viscosity"] = rows
    payload["cells"] = cells
    payload["worst_conservative_speed_error"] = worst
    payload["elapsed_seconds"] = round(elapsed, 1)
    path = write_json(OUT_NAME, payload)
    print(f"\nworst conservative speed error across all cells: {worst:.2e}")
    print(f"wrote {path} in {elapsed:.0f}s")


def _run_star(job: tuple[float, int]) -> dict[str, Any]:
    return run_case(*job)


if __name__ == "__main__":
    main()
