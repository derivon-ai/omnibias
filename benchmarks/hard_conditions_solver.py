# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hard vs soft boundary / initial conditions on the mesh-free solver.

Absorbing a condition into the architecture makes it exact. That much is
algebra. What is *not* algebra is whether it helps the rest of the solve, so
this measures both halves on problems with analytic solutions -- Poisson
(elliptic), heat (parabolic), wave (hyperbolic), a 2-D square whose four faces
are absorbed at once, a gauge-free periodic seam, and a **gauge-pinned**
periodic heat -- over several seeds, at an identical architecture, parameter
count and collocation budget. The only difference between the soft and hard
arms is ``hard_conditions``.

The gauge-pinned periodic heat row also carries a third **spectral** arm
(``basis="spectral"``), where spatial periodicity is free in the Fourier
ansatz rather than spent as relative constraints.

Three numbers per cell, and the last two are the honest ones:

* **boundary violation** -- the largest condition residual over the faces and
  the initial slice. The hard arm is exact here by construction; this is the
  falsifier for that claim, not evidence for it.
* **seam jump beyond contract** (periodic cases only) -- the jump at the first
  derivative order the condition does *not* declare. The violation column scores
  exactly the orders the seam matches, so on a periodic case it is graded against
  the same set the cage enforces and could never falsify smoothness above it.
  This is the column that can: a ``(0, 1, 2)`` seam is machine-zero on the first
  three orders and genuinely discontinuous on the fourth, and saying so is the
  difference between "exact" and "exact at the declared orders".
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
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import omnibias.pinn.solver as pde  # noqa: E402
import omnibias.pinn.solver.torch as pt  # noqa: E402
import torch  # noqa: E402
from _common import provenance, write_json  # noqa: E402
from omnibias.pinn.solver._core.sampling import periodic_axes  # noqa: E402
from omnibias.pinn.solver.torch.assemble import condition_residual, to_tensor  # noqa: E402

torch.set_num_threads(1)

SEEDS = tuple(range(int(os.environ.get("HARD_SEEDS", "5"))))
HIDDEN = int(os.environ.get("HARD_HIDDEN", "96"))
OUT_NAME = os.environ.get("HARD_OUT", "hard_conditions_solver.json")
CASE_FILTER = {
    c.strip()
    for c in os.environ.get("HARD_CASES", "").split(",")
    if c.strip()
}

#: Equal budget for both MLP arms. The hard arm solves a *smaller* system (the
#: absorbed rows are gone), which is a saving, not an advantage -- the interior
#: rows it fits are the same rows.
SPEC = pde.CollocationSpec(n_interior=48, n_boundary=16)

ALPHA = 0.25
SPEED = 1.0
T_END = 0.3
SPECTRAL_K = 8


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


def square_case() -> tuple[Any, Any]:
    """2-D Poisson, ``u = sin(pi x) sin(pi y)``, all four faces zero.

    Two constrained *spatial* axes, so the corner terms the recursion generates
    are load bearing rather than incidental.
    """
    dom = pde.Domain(("x", "y"), ((0.0, 1.0), (0.0, 1.0)))

    def source(c):  # noqa: ANN001, ANN202
        xp = pde.array_namespace(c)
        return -2.0 * (math.pi**2) * xp.sin(math.pi * c[:, 0]) * xp.sin(math.pi * c[:, 1])

    def exact(pts: np.ndarray) -> np.ndarray:
        return np.sin(math.pi * pts[:, 0]) * np.sin(math.pi * pts[:, 1])

    return pde.poisson(dom, source=source, boundary=0.0), exact


def seam_case() -> tuple[Any, Any]:
    """``u'' = -(2 pi)^2 sin(2 pi x)`` on a periodic interval.

    A relative constraint rather than a pointwise one. Note the solution is
    pinned only up to an additive constant, so the interior error is measured
    after removing the mean from both sides -- that gauge freedom is a property
    of the problem, not of either arm.
    """
    dom = pde.Domain(("x",), ((0.0, 1.0),), periodic=True)

    def source(c):  # noqa: ANN001, ANN202
        xp = pde.array_namespace(c)
        return -((2.0 * math.pi) ** 2) * xp.sin(2.0 * math.pi * c[:, 0])

    def exact(pts: np.ndarray) -> np.ndarray:
        return np.sin(2.0 * math.pi * pts[:, 0])

    system = replace(
        pde.poisson(dom, source=source, boundary=0.0),
        boundary=(pde.BoundaryCondition(component="u", kind="periodic", axis="x"),),
    )
    return system, exact


def heat_periodic_case() -> tuple[Any, Any]:
    """Periodic heat with IC ``u(x,0)=sin(2 pi x)`` -- gauge-pinned.

    Exact ``exp(-alpha (2 pi)^2 t) sin(2 pi x)``. Soft's additive freedom buys
    nothing once the initial slice pins the mode; this is the comparison the
    gauge-free seam row cannot make. Also hosts the spectral arm.
    """
    dom = pde.Domain(
        ("t", "x"),
        ((0.0, T_END), (0.0, 1.0)),
        periodic=(False, True),
        time_axis="t",
    )

    def initial(c):  # noqa: ANN001, ANN202
        xp = pde.array_namespace(c)
        return xp.sin(2.0 * math.pi * c[:, 1])

    def exact(pts: np.ndarray) -> np.ndarray:
        decay = np.exp(-ALPHA * (2.0 * math.pi) ** 2 * pts[:, 0])
        return decay * np.sin(2.0 * math.pi * pts[:, 1])

    system = pde.heat(
        dom,
        diffusivity=ALPHA,
        initial=initial,
        periodic_boundary=True,
    )
    return system, exact


CASES = {
    "poisson": poisson_case,
    "heat": heat_case,
    "wave": wave_case,
    "square": square_case,
    "seam": seam_case,
    "heat_periodic": heat_periodic_case,
}

#: Problems whose solution is only determined up to an additive constant.
GAUGE_FREE = frozenset({"seam"})

#: Cases that also run a spectral Fourier arm (needs a time axis).
SPECTRAL_CASES = frozenset({"heat_periodic"})


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


def _derivative_scale(sol: Any, bc: Any, order: int) -> float:
    """Typical size of ``d^order u`` inside the domain, to scale a seam jump.

    A raw jump is unreadable on its own: ``d^3`` of a ``2 pi`` mode is ``(2 pi)^3``
    before anything goes wrong, so the same absolute number means different things
    on different problems.
    """
    axes = periodic_axes(sol.system.domain, bc)
    if not axes:
        return 0.0
    state = sol.field(to_tensor(_interior_grid(sol.system, n=12), sol.field))
    d = state.ops.derivative(state, bc.component, axis=axes[0], order=order)
    return float(d.detach().abs().max())


def seam_jump_beyond_contract(sol: Any) -> dict[str, Any] | None:
    """Seam jump at the first derivative order the condition does *not* declare.

    Built by re-declaring the periodic conditions at that single order and
    reassembling, so the probe travels the same seam-row path as the enforced
    orders rather than a lookalike that could drift from it. Initial conditions
    are dropped so the maximum is the seam and nothing else.
    """
    periodic = [b for b in sol.system.boundary if b.kind == "periodic"]
    if not periodic:
        return None
    beyond = max(max(b.periodic_orders or (0,)) for b in periodic) + 1
    probe = replace(
        sol.system,
        boundary=tuple(replace(b, periodic_orders=(beyond,)) for b in periodic),
        initial=(),
    )
    rows = condition_residual(sol.field, probe, SPEC, None)
    if not rows.numel():
        return None
    jump = float(rows.detach().abs().max())
    scale = _derivative_scale(sol, periodic[0], beyond)
    return {
        "order": beyond,
        "jump": jump,
        "relative": jump / scale if scale > 0.0 else float("inf"),
    }


def interior_error(sol: Any, exact: Any, *, gauge_free: bool = False) -> float:
    pts = _interior_grid(sol.system)
    u = sol.evaluate(pts, "u").detach().numpy()
    truth = exact(pts)
    if gauge_free:
        u, truth = u - u.mean(), truth - truth.mean()
    return float(np.linalg.norm(u - truth) / np.linalg.norm(truth))


def _arm_metrics(sol: Any, exact: Any, *, gauge_free: bool, started: float) -> dict[str, Any]:
    out = {
        "boundary_violation": boundary_violation(sol),
        "interior_relative_l2": interior_error(sol, exact, gauge_free=gauge_free),
        "n_rows": sol.diagnostics.get("n_rows"),
        "n_unknowns": sol.diagnostics.get("n_unknowns"),
        "absorbed": sol.diagnostics.get("hard_absorbed", 0),
        "seconds": round(time.perf_counter() - started, 3),
    }
    beyond = seam_jump_beyond_contract(sol)
    if beyond is not None:
        out["seam_beyond_contract"] = beyond
    return out


# --------------------------------------------------------------------- run ---


def run_case(name: str, seed: int) -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    system, exact = CASES[name]()
    out: dict[str, Any] = {"case": name, "seed": seed}
    gauge_free = name in GAUGE_FREE
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
        out[arm] = _arm_metrics(sol, exact, gauge_free=gauge_free, started=started)

    if name in SPECTRAL_CASES:
        started = time.perf_counter()
        # Fourier spatial modes make periodicity free; ``auto`` still absorbs
        # the initial condition (and will also absorb the seam if the planner
        # certifies it -- redundant algebraically on a spectral base, free
        # numerically). Parameter count is *not* matched to the MLP arms.
        sol = pt.solve_least_squares(
            system,
            hidden=HIDDEN,
            weight_init_scale=3.0,
            seed=seed,
            collocation=SPEC,
            hard_conditions="auto",
            basis="spectral",
            K=SPECTRAL_K,
            L=1.0,
        )
        out["spectral"] = _arm_metrics(
            sol, exact, gauge_free=gauge_free, started=started
        )
        out["spectral"]["basis"] = "spectral"
        out["spectral"]["K"] = SPECTRAL_K
    return out


def _arm_names(block: list[dict[str, Any]]) -> tuple[str, ...]:
    names = ["hard", "soft"]
    if any("spectral" in c for c in block):
        names.append("spectral")
    return tuple(names)


def _aggregate(cells: list[dict[str, Any]], case_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in case_names:
        block = [c for c in cells if c["case"] == name]
        if not block:
            continue
        arms = _arm_names(block)
        row: dict[str, Any] = {"case": name, "seeds": len(block), "arms": list(arms)}
        for arm in arms:
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
            beyond = [
                c[arm]["seam_beyond_contract"]
                for c in block
                if c[arm].get("seam_beyond_contract")
            ]
            if beyond:
                row[arm]["seam_beyond_contract"] = {
                    "order": beyond[0]["order"],
                    "jump_median": statistics.median(b["jump"] for b in beyond),
                    "relative_median": statistics.median(b["relative"] for b in beyond),
                }
            if arm == "spectral":
                row[arm]["K"] = block[0][arm].get("K", SPECTRAL_K)
        row["hard_wins_interior"] = sum(
            1
            for c in block
            if c["hard"]["interior_relative_l2"] < c["soft"]["interior_relative_l2"]
        )
        row["absorbed"] = block[0]["hard"]["absorbed"]
        if "spectral" in arms:
            row["spectral_beats_hard"] = sum(
                1
                for c in block
                if c["spectral"]["interior_relative_l2"]
                < c["hard"]["interior_relative_l2"]
            )
            row["spectral_beats_soft"] = sum(
                1
                for c in block
                if c["spectral"]["interior_relative_l2"]
                < c["soft"]["interior_relative_l2"]
            )
        rows.append(row)
    return rows


def _print_row(row: dict[str, Any]) -> None:
    hard, soft = row["hard"], row["soft"]
    print(
        f"  {row['case']:14s} absorbed {row['absorbed']}  "
        f"rows {hard['n_rows']} vs {soft['n_rows']}"
    )
    print(
        f"           boundary |err| hard {hard['boundary_violation_max']:.2e} (max)"
        f"   soft {soft['boundary_violation_median']:.2e}"
    )
    if "seam_beyond_contract" in hard:
        b = hard["seam_beyond_contract"]
        print(
            f"           seam jump at unmatched order {b['order']}: "
            f"hard {b['jump_median']:.2e} ({b['relative_median']:.1%} of scale)"
        )
    print(
        f"           interior L2    hard {hard['interior_l2_median']:.2e}"
        f"   soft {soft['interior_l2_median']:.2e}"
        f"   hard wins {row['hard_wins_interior']}/{row['seeds']}"
    )
    if "spectral" in row:
        sp = row["spectral"]
        print(
            f"           spectral       boundary {sp['boundary_violation_median']:.2e}"
            f"   interior L2 {sp['interior_l2_median']:.2e}"
            f"   beats hard {row['spectral_beats_hard']}/{row['seeds']}"
            f" soft {row['spectral_beats_soft']}/{row['seeds']}"
        )


def main() -> None:
    case_names = [n for n in CASES if not CASE_FILTER or n in CASE_FILTER]
    jobs = [(name, seed) for name in case_names for seed in SEEDS]
    n_spectral = sum(1 for n, _ in jobs if n in SPECTRAL_CASES)
    print(
        f"hard vs soft conditions: {len(jobs)} cells x 2 MLP arms"
        f"{f' + {n_spectral} spectral' if n_spectral else ''} "
        f"(hidden={HIDDEN})"
    )
    started = time.perf_counter()
    cells = [run_case(name, seed) for name, seed in jobs]
    elapsed = time.perf_counter() - started

    rows = _aggregate(cells, case_names)
    for row in rows:
        _print_row(row)

    worst = max(r["hard"]["boundary_violation_max"] for r in rows)
    payload = provenance(
        schema="hard_conditions_solver/v3",
        config={
            "seeds": list(SEEDS),
            "hidden": HIDDEN,
            "n_interior": SPEC.n_interior,
            "n_boundary": SPEC.n_boundary,
            "driver": "solve_least_squares",
            "cases": case_names,
            "spectral_cases": sorted(SPECTRAL_CASES & set(case_names)),
            "spectral_K": SPECTRAL_K,
            "note": (
                "MLP hard/soft arms share architecture, parameter count, seed and "
                "collocation budget; only hard_conditions differs. The spectral "
                "arm (heat_periodic only) uses basis='spectral' with "
                "hard_conditions='auto' (periodicity free in the Fourier base; "
                "IC absorbed); its parameter count is not matched to the MLP. "
                "seam_beyond_contract probes the first derivative order the "
                "periodic condition does NOT declare -- the boundary_violation "
                "column is graded on the declared orders only and so cannot "
                "falsify smoothness above them."
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
