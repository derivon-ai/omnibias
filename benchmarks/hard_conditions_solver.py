# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hard vs soft boundary / initial conditions on the mesh-free solver.

Absorbing a condition into the architecture makes it exact. That much is
algebra. What is *not* algebra is whether it helps the rest of the solve, so
this measures both halves on three problems with analytic solutions -- Poisson
(elliptic), heat (parabolic) and wave (hyperbolic) -- over several seeds, at an
identical architecture, parameter count and collocation budget. The only
difference between the arms is ``hard_conditions``.

Two numbers per cell, and the second one is the honest one:

* **boundary violation** -- the largest condition residual over the faces and
  the initial slice. The hard arm is exact here by construction; this is the
  falsifier for that claim, not evidence for it.
* **interior relative L2** -- error against the analytic solution on a dense
  interior grid. This is *optimised*, not proven, and it is the number that
  decides whether absorption is worth using.

Run::

    uv run python benchmarks/hard_conditions_solver.py
"""

from __future__ import annotations

import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import omnibias.pinn.solver as pde  # noqa: E402
import omnibias.pinn.solver.torch as pt  # noqa: E402
import torch  # noqa: E402
from _common import provenance, write_json  # noqa: E402
from omnibias.pinn.solver.torch.assemble import condition_residual  # noqa: E402

torch.set_num_threads(1)

SEEDS = tuple(range(int(os.environ.get("HARD_SEEDS", "5"))))
HIDDEN = int(os.environ.get("HARD_HIDDEN", "96"))
OUT_NAME = os.environ.get("HARD_OUT", "hard_conditions_solver.json")

#: Equal budget for both arms. The hard arm solves a *smaller* system (the
#: absorbed rows are gone), which is a saving, not an advantage -- the interior
#: rows it fits are the same rows.
SPEC = pde.CollocationSpec(n_interior=48, n_boundary=16)

ALPHA = 0.25
SPEED = 1.0
T_END = 0.3


# ----------------------------------------------------------------- problems --


def poisson_case() -> tuple[Any, Any]:
    """``u'' = -pi^2 sin(pi x)`` on ``[0, 1]``, ``u(0) = u(1) = 0``."""
    dom = pde.Domain(("x",), ((0.0, 1.0),))

    def source(c):  # noqa: ANN001, ANN202
        xp = pde.array_namespace(c)
        return -(math.pi**2) * xp.sin(math.pi * c[:, 0])

    def exact(pts: np.ndarray) -> np.ndarray:
        return np.sin(math.pi * pts[:, 0])

    return pde.poisson(dom, source=source, boundary=0.0), exact


def heat_case() -> tuple[Any, Any]:
    """``u_t = alpha u_xx``, ``u(x,0) = sin(pi x)``, zero ends."""
    dom = pde.Domain(("t", "x"), ((0.0, T_END), (0.0, 1.0)), time_axis="t")

    def initial(c):  # noqa: ANN001, ANN202
        xp = pde.array_namespace(c)
        return xp.sin(math.pi * c[:, 1])

    def exact(pts: np.ndarray) -> np.ndarray:
        decay = np.exp(-ALPHA * math.pi**2 * pts[:, 0])
        return decay * np.sin(math.pi * pts[:, 1])

    return pde.heat(dom, diffusivity=ALPHA, initial=initial, boundary=0.0), exact


def wave_case() -> tuple[Any, Any]:
    """``u_tt = c^2 u_xx``, ``u(x,0) = sin(pi x)``, ``u_t(x,0) = 0``, zero ends."""
    dom = pde.Domain(("t", "x"), ((0.0, T_END), (0.0, 1.0)), time_axis="t")

    def initial(c):  # noqa: ANN001, ANN202
        xp = pde.array_namespace(c)
        return xp.sin(math.pi * c[:, 1])

    def exact(pts: np.ndarray) -> np.ndarray:
        return np.cos(SPEED * math.pi * pts[:, 0]) * np.sin(math.pi * pts[:, 1])

    sys_ = pde.wave(
        dom, speed=SPEED, initial=initial, initial_velocity=0.0, boundary=0.0
    )
    return sys_, exact


CASES = {"poisson": poisson_case, "heat": heat_case, "wave": wave_case}


# ------------------------------------------------------------------ metrics --


def _interior_grid(system: Any, n: int = 40) -> np.ndarray:
    """A dense grid strictly inside the domain -- the faces are measured apart."""
    axes = [
        np.linspace(lo + 0.02 * (hi - lo), hi - 0.02 * (hi - lo), n)
        for lo, hi in system.domain.bounds
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=-1)


def boundary_violation(sol: Any) -> float:
    """Largest condition residual, assembled *ignoring* what the plan absorbed."""
    rows = condition_residual(sol.field, sol.system, SPEC, None)
    return float(rows.detach().abs().max()) if rows.numel() else 0.0


def interior_error(sol: Any, exact: Any) -> float:
    pts = _interior_grid(sol.system)
    u = sol.evaluate(pts, "u").detach().numpy()
    truth = exact(pts)
    return float(np.linalg.norm(u - truth) / np.linalg.norm(truth))


# --------------------------------------------------------------------- run ---


def run_case(name: str, seed: int) -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    system, exact = CASES[name]()
    out: dict[str, Any] = {"case": name, "seed": seed}
    for arm, mode in (("hard", "auto"), ("soft", "none")):
        started = time.perf_counter()
        sol = pt.solve_least_squares(
            system,
            hidden=HIDDEN,
            weight_init_scale=3.0,
            seed=seed,
            collocation=SPEC,
            hard_conditions=mode,
        )
        out[arm] = {
            "boundary_violation": boundary_violation(sol),
            "interior_relative_l2": interior_error(sol, exact),
            "n_rows": sol.diagnostics.get("n_rows"),
            "n_unknowns": sol.diagnostics.get("n_unknowns"),
            "absorbed": sol.diagnostics.get("hard_absorbed", 0),
            "seconds": round(time.perf_counter() - started, 3),
        }
    return out


def _aggregate(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in CASES:
        block = [c for c in cells if c["case"] == name]
        row: dict[str, Any] = {"case": name, "seeds": len(block)}
        for arm in ("hard", "soft"):
            row[arm] = {
                "boundary_violation_median": statistics.median(
                    c[arm]["boundary_violation"] for c in block
                ),
                "boundary_violation_max": max(
                    c[arm]["boundary_violation"] for c in block
                ),
                "interior_l2_median": statistics.median(
                    c[arm]["interior_relative_l2"] for c in block
                ),
                "interior_l2_max": max(c[arm]["interior_relative_l2"] for c in block),
                "n_rows": block[0][arm]["n_rows"],
            }
        row["hard_wins_interior"] = sum(
            1
            for c in block
            if c["hard"]["interior_relative_l2"] < c["soft"]["interior_relative_l2"]
        )
        row["absorbed"] = block[0]["hard"]["absorbed"]
        rows.append(row)
    return rows


def main() -> None:
    jobs = [(name, seed) for name in CASES for seed in SEEDS]
    print(f"hard vs soft conditions: {len(jobs)} cells x 2 arms (hidden={HIDDEN})")
    started = time.perf_counter()
    cells = [run_case(name, seed) for name, seed in jobs]
    elapsed = time.perf_counter() - started

    rows = _aggregate(cells)
    for row in rows:
        hard, soft = row["hard"], row["soft"]
        print(
            f"  {row['case']:8s} absorbed {row['absorbed']}  "
            f"rows {hard['n_rows']} vs {soft['n_rows']}"
        )
        print(
            f"           boundary |err| hard {hard['boundary_violation_max']:.2e} (max)"
            f"   soft {soft['boundary_violation_median']:.2e}"
        )
        print(
            f"           interior L2    hard {hard['interior_l2_median']:.2e}"
            f"   soft {soft['interior_l2_median']:.2e}"
            f"   hard wins {row['hard_wins_interior']}/{row['seeds']}"
        )

    worst = max(r["hard"]["boundary_violation_max"] for r in rows)
    payload = provenance(
        schema="hard_conditions_solver/v1",
        config={
            "seeds": list(SEEDS),
            "hidden": HIDDEN,
            "n_interior": SPEC.n_interior,
            "n_boundary": SPEC.n_boundary,
            "driver": "solve_least_squares",
            "note": (
                "both arms share architecture, parameter count, seed and collocation "
                "budget; only hard_conditions differs"
            ),
        },
    )
    payload["by_case"] = rows
    payload["cells"] = cells
    payload["worst_hard_boundary_violation"] = worst
    payload["elapsed_seconds"] = round(elapsed, 1)
    path = write_json(OUT_NAME, payload)
    print(f"\nworst hard-arm boundary violation over all cells: {worst:.2e}")
    print(f"wrote {path} in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
