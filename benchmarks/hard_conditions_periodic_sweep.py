# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Diagnose why the hard-conditions seam row loses on interior L2, and how far
to raise the matched orders.

Sweeps ``periodic_orders`` in ``{(0, 1), (0, 1, 2), (0, 1, 2, 3)}`` by ``hidden``
in ``{48, 96, 192}`` (optional ``384`` via ``SWEEP_HIDDENS``) over a few seeds on
the gauge-free Poisson seam from ``hard_conditions_solver.py``. Soft and hard
arms share everything except ``hard_conditions``.

Decision rule (stated *before* the run; the verdict below is read against it):

* **gap decays like 1/hidden** -- the hard projection spends two (or three)
  degrees of freedom tying the seam; the loss is intrinsic and vanishes with
  width. Keep the default orders ``(0, 1)``.
* **(0, 1, 2) closes the gap while (0, 1) does not** -- the C¹ seam is the
  issue (quadratic switching leaves ``u''`` discontinuous under a second-order
  operator). Change the default ``PERIODIC_ORDERS`` to ``(0, 1, 2)``.
* **neither moves it** -- the gauge freedom in the comparison is the real
  confounder; the Poisson seam case is the wrong benchmark, and a
  gauge-pinned problem (periodic heat) is needed instead.

``(0, 1, 2, 3)`` is swept to answer a *different* question, and its own rule is
fixed here too: it changes the shipped default only if it improves the hard arm
by more than the ``(0,1) -> (0,1,2)`` step did. A smooth manufactured solution
will keep rewarding higher orders forever -- every derivative of a smooth
periodic function matches -- so "better on this problem" is not sufficient
grounds. The default has to survive non-smooth problems too, and the periodic
emit measurement already showed Burgers losing interior accuracy when its seam
is enforced. Diminishing returns therefore mean stop, not continue.

Run::

    uv run python benchmarks/hard_conditions_periodic_sweep.py
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
from omnibias.pinn.solver.torch.assemble import condition_residual  # noqa: E402

torch.set_num_threads(1)

SEEDS = tuple(range(int(os.environ.get("SWEEP_SEEDS", "3"))))
HIDDENS = tuple(
    int(x) for x in os.environ.get("SWEEP_HIDDENS", "48,96,192").split(",") if x.strip()
)
ORDER_SETS: tuple[tuple[int, ...], ...] = ((0, 1), (0, 1, 2), (0, 1, 2, 3))
OUT_NAME = os.environ.get("SWEEP_OUT", "hard_conditions_periodic_sweep.json")
SPEC = pde.CollocationSpec(n_interior=48, n_boundary=16)


def seam_system(orders: tuple[int, ...]) -> Any:
    """Gauge-free periodic Poisson used by the hard-conditions seam row."""
    dom = pde.Domain(("x",), ((0.0, 1.0),), periodic=True)

    def source(c):  # noqa: ANN001, ANN202
        xp = pde.array_namespace(c)
        return -((2.0 * math.pi) ** 2) * xp.sin(2.0 * math.pi * c[:, 0])

    return replace(
        pde.poisson(dom, source=source, boundary=0.0),
        boundary=(
            pde.BoundaryCondition(
                component="u",
                kind="periodic",
                axis="x",
                periodic_orders=orders,
            ),
        ),
    )


def exact(pts: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * math.pi * pts[:, 0])


def _interior_grid(system: Any, n: int = 40) -> np.ndarray:
    axes = [
        np.linspace(lo + 0.02 * (hi - lo), hi - 0.02 * (hi - lo), n)
        for lo, hi in system.domain.bounds
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=-1)


def boundary_violation(sol: Any) -> float:
    rows = condition_residual(sol.field, sol.system, SPEC, None)
    return float(rows.detach().abs().max()) if rows.numel() else 0.0


def interior_error(sol: Any) -> float:
    pts = _interior_grid(sol.system)
    u = sol.evaluate(pts, "u").detach().numpy()
    truth = exact(pts)
    u, truth = u - u.mean(), truth - truth.mean()
    return float(np.linalg.norm(u - truth) / np.linalg.norm(truth))


def run_cell(
    orders: tuple[int, ...], hidden: int, seed: int
) -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    system = seam_system(orders)
    out: dict[str, Any] = {
        "orders": list(orders),
        "hidden": hidden,
        "seed": seed,
    }
    for arm, mode in (("hard", "auto"), ("soft", "none")):
        started = time.perf_counter()
        sol = pt.solve_least_squares(
            system,
            hidden=hidden,
            weight_init_scale=3.0,
            seed=seed,
            collocation=SPEC,
            hard_conditions=mode,
        )
        out[arm] = {
            "boundary_violation": boundary_violation(sol),
            "interior_relative_l2": interior_error(sol),
            "n_rows": sol.diagnostics.get("n_rows"),
            "n_unknowns": sol.diagnostics.get("n_unknowns"),
            "absorbed": sol.diagnostics.get("hard_absorbed", 0),
            "seconds": round(time.perf_counter() - started, 3),
        }
    hard_l2 = out["hard"]["interior_relative_l2"]
    soft_l2 = out["soft"]["interior_relative_l2"]
    out["gap"] = hard_l2 - soft_l2
    out["gap_ratio"] = hard_l2 / soft_l2 if soft_l2 > 0.0 else float("inf")
    return out


def _aggregate(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for orders in ORDER_SETS:
        for hidden in HIDDENS:
            block = [
                c
                for c in cells
                if tuple(c["orders"]) == orders and c["hidden"] == hidden
            ]
            if not block:
                continue
            gaps = [c["gap"] for c in block]
            hard_l2 = [c["hard"]["interior_relative_l2"] for c in block]
            soft_l2 = [c["soft"]["interior_relative_l2"] for c in block]
            rows.append(
                {
                    "orders": list(orders),
                    "hidden": hidden,
                    "seeds": len(block),
                    "hard_interior_l2_median": statistics.median(hard_l2),
                    "soft_interior_l2_median": statistics.median(soft_l2),
                    "gap_median": statistics.median(gaps),
                    "gap_max": max(gaps),
                    "gap_ratio_median": statistics.median(
                        c["gap_ratio"] for c in block
                    ),
                    "hard_wins": sum(1 for c in block if c["gap"] < 0.0),
                    "absorbed": block[0]["hard"]["absorbed"],
                }
            )
    return rows


def _slope_vs_inv_hidden(rows: list[dict[str, Any]], orders: tuple[int, ...]) -> float | None:
    """Log-log slope of |gap| vs 1/hidden; ~1 means gap ~ 1/hidden."""
    pts = [(r["hidden"], abs(r["gap_median"])) for r in rows if tuple(r["orders"]) == orders]
    pts = [(h, g) for h, g in pts if g > 0.0]
    if len(pts) < 2:
        return None
    # Fit log|gap| = a + b * log(1/hidden)  =>  b ~ 1 for 1/hidden decay.
    xs = [math.log(1.0 / h) for h, _ in pts]
    ys = [math.log(g) for _, g in pts]
    xbar, ybar = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys, strict=True))
    den = sum((x - xbar) ** 2 for x in xs)
    return None if den == 0.0 else num / den


def _marginal_gain(
    rows: list[dict[str, Any]], lo: tuple[int, ...], hi: tuple[int, ...]
) -> float | None:
    """Median factor by which the hard arm improves going from ``lo`` to ``hi``.

    Greater than one means the extra matched order helped.
    """
    a = {r["hidden"]: r["hard_interior_l2_median"] for r in rows if tuple(r["orders"]) == lo}
    b = {r["hidden"]: r["hard_interior_l2_median"] for r in rows if tuple(r["orders"]) == hi}
    shared = [h for h in sorted(set(a) & set(b)) if b[h] > 0.0]
    return statistics.median(a[h] / b[h] for h in shared) if shared else None


def decide(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the docstring decision rule to the aggregated medians."""
    by_o01 = [r for r in rows if tuple(r["orders"]) == (0, 1)]
    by_o012 = [r for r in rows if tuple(r["orders"]) == (0, 1, 2)]
    gap01 = {r["hidden"]: r["gap_median"] for r in by_o01}
    gap012 = {r["hidden"]: r["gap_median"] for r in by_o012}
    shared = sorted(set(gap01) & set(gap012))

    slope = _slope_vs_inv_hidden(rows, (0, 1))
    # (0,1,2) "closes" the gap if its median gap is much smaller (or flipped)
    # at the same widths where (0,1) still loses.
    closes = False
    if shared:
        closes = all(
            gap012[h] < 0.25 * gap01[h] or gap012[h] <= 0.0
            for h in shared
            if gap01[h] > 0.0
        ) and any(gap01[h] > 0.0 for h in shared)

    # Decaying like 1/hidden: positive gaps that shrink, slope near 1.
    decays = (
        slope is not None
        and 0.5 <= slope <= 1.5
        and len(by_o01) >= 2
        and by_o01[-1]["gap_median"] < by_o01[0]["gap_median"]
        and by_o01[0]["gap_median"] > 0.0
    )

    if closes:
        verdict = "c1_seam"
        summary = (
            "(0,1,2) closes the hard-soft interior gap that (0,1) leaves open "
            "-- C¹ seam issue; default PERIODIC_ORDERS should become (0,1,2)."
        )
    elif decays:
        verdict = "lost_degrees_of_freedom"
        summary = (
            f"hard-soft gap decays like 1/hidden (log-log slope vs 1/h ≈ "
            f"{slope:.2f}) -- intrinsic cost of the relative constraints."
        )
    else:
        verdict = "gauge"
        summary = (
            "neither 1/hidden decay nor (0,1,2) closing explains the loss -- "
            "gauge freedom confounds the comparison; use a gauge-pinned case."
        )

    # How far to go: compare the marginal gain of each extra matched order.
    gain_012 = _marginal_gain(rows, (0, 1), (0, 1, 2))
    gain_0123 = _marginal_gain(rows, (0, 1, 2), (0, 1, 2, 3))
    raise_further = (
        gain_012 is not None and gain_0123 is not None and gain_0123 > gain_012
    )
    if gain_0123 is None:
        stopping = "(0,1,2,3) not swept; no marginal-gain evidence either way."
    elif raise_further:
        stopping = (
            f"the fourth matched order gains {gain_0123:.2f}x, more than the third's "
            f"{gain_012:.2f}x -- returns are not diminishing, so the default should "
            "be re-examined rather than stopped at (0,1,2)."
        )
    else:
        stopping = (
            f"the fourth matched order gains only {gain_0123:.2f}x against the third's "
            f"{gain_012:.2f}x -- diminishing returns on a smooth manufactured "
            "solution, which is exactly where extra orders always look good. Stop at "
            "(0,1,2); a higher default would over-smooth seams on problems with steep "
            "gradients, which the periodic-emit measurement already showed for Burgers."
        )

    return {
        "verdict": verdict,
        "summary": summary,
        "slope_gap_vs_inv_hidden_orders_01": slope,
        "orders_012_closes_gap": closes,
        "hard_gain_01_to_012": gain_012,
        "hard_gain_012_to_0123": gain_0123,
        "raise_default_past_012": raise_further,
        "stopping_rule": stopping,
        "gap_median_by_orders_hidden": {
            "0,1": {str(h): gap01[h] for h in sorted(gap01)},
            "0,1,2": {str(h): gap012[h] for h in sorted(gap012)},
            "0,1,2,3": {
                str(r["hidden"]): r["gap_median"]
                for r in rows
                if tuple(r["orders"]) == (0, 1, 2, 3)
            },
        },
    }


def main() -> None:
    jobs = [
        (orders, hidden, seed)
        for orders in ORDER_SETS
        for hidden in HIDDENS
        for seed in SEEDS
    ]
    print(
        f"periodic seam sweep: {len(jobs)} cells x 2 arms "
        f"(hiddens={list(HIDDENS)}, seeds={list(SEEDS)})"
    )
    started = time.perf_counter()
    cells = [run_cell(orders, hidden, seed) for orders, hidden, seed in jobs]
    elapsed = time.perf_counter() - started
    rows = _aggregate(cells)
    decision = decide(rows)

    for row in rows:
        print(
            f"  orders={tuple(row['orders'])!s:12s} hidden={row['hidden']:3d}  "
            f"hard L2 {row['hard_interior_l2_median']:.2e}  "
            f"soft L2 {row['soft_interior_l2_median']:.2e}  "
            f"gap {row['gap_median']:+.2e}  "
            f"ratio {row['gap_ratio_median']:.2f}  "
            f"hard wins {row['hard_wins']}/{row['seeds']}"
        )

    print(f"\nVERDICT: {decision['verdict']}")
    print(f"  {decision['summary']}")
    if decision["slope_gap_vs_inv_hidden_orders_01"] is not None:
        print(
            "  slope of log|gap| vs log(1/hidden) for orders (0,1): "
            f"{decision['slope_gap_vs_inv_hidden_orders_01']:.3f}"
        )
    print(f"\nHOW FAR: {decision['stopping_rule']}")

    payload = provenance(
        schema="hard_conditions_periodic_sweep/v2",
        config={
            "seeds": list(SEEDS),
            "hiddens": list(HIDDENS),
            "order_sets": [list(o) for o in ORDER_SETS],
            "n_interior": SPEC.n_interior,
            "n_boundary": SPEC.n_boundary,
            "driver": "solve_least_squares",
            "case": "gauge-free periodic Poisson seam",
            "decision_rule": (
                "gap ~ 1/hidden -> lost DoF; (0,1,2) closes -> C1 seam; "
                "neither -> gauge / wrong comparison"
            ),
            "stopping_rule": (
                "raise the default past (0,1,2) only if the fourth matched order "
                "gains more than the third did; a smooth manufactured solution "
                "rewards extra orders indefinitely, so diminishing returns mean stop"
            ),
        },
    )
    payload["by_cell"] = rows
    payload["cells"] = cells
    payload["decision"] = decision
    payload["elapsed_seconds"] = round(elapsed, 1)
    path = write_json(OUT_NAME, payload)
    print(f"\nwrote {path} in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
