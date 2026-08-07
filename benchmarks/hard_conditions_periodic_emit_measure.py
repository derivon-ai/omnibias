# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Measure Stage 2 ``periodic_boundary`` auto-emit on manufactured solutions.

Compares ``periodic_boundary=False`` (today) vs ``True`` on viscous Burgers and
a two-species reaction-diffusion problem, both on a periodic spatial domain.
Each setting is run hard (``hard_conditions="auto"``) and soft
(``hard_conditions="none"``) across a few seeds.

Two numbers decide the default:

* **interior relative L2** against the manufactured truth
* **seam violation** -- max ``|d^n u(hi) - d^n u(lo)|`` for ``n in (0, 1)``,
  measured on the spatial faces regardless of whether a periodic BC was
  declared (so the off path is still scored honestly)

Decision rule (stated up front): flip the builder default to ``True`` only if
``True`` is strictly better on *both* metrics for *both* problems (median over
seeds, comparing the better of hard/soft for each flag). Otherwise keep the
default ``False`` (opt-in).

Run (CPU, small)::

    uv run python benchmarks/hard_conditions_periodic_emit_measure.py
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
from omnibias.pinn.solver._core.sampling import boundary_points  # noqa: E402
from omnibias.pinn.solver.torch.assemble import to_tensor  # noqa: E402

torch.set_num_threads(1)

SEEDS = tuple(range(int(os.environ.get("EMIT_SEEDS", "3"))))
HIDDEN = int(os.environ.get("EMIT_HIDDEN", "32"))
ADAM = int(os.environ.get("EMIT_ADAM", "60"))
ITERS = int(os.environ.get("EMIT_ITERS", "30"))
OUT_NAME = os.environ.get("EMIT_OUT", "hard_conditions_periodic_emit_measure.json")
SPEC = pde.CollocationSpec(n_interior=24, n_boundary=12)

NU = 0.1
A0 = 1.5
K = 1.0
T_END = 0.25
DU, DV = 0.2, 0.1


# ----------------------------------------------------------------- problems --


def _burgers_exact(pts: np.ndarray) -> np.ndarray:
    """Cole-Hopf periodic solution of viscous Burgers on ``[0, 2 pi]``."""
    t, x = pts[:, 0], pts[:, 1]
    decay = np.exp(-NU * (K**2) * t)
    num = 2.0 * NU * K * np.sin(K * x) * decay
    den = A0 + np.cos(K * x) * decay
    return num / den


def burgers_case(periodic_boundary: bool) -> tuple[Any, Any]:
    dom = pde.Domain(
        ("t", "x"),
        ((0.0, T_END), (0.0, 2.0 * math.pi)),
        periodic=(False, True),
        time_axis="t",
    )

    def initial(c):  # noqa: ANN001, ANN202
        xp = pde.array_namespace(c)
        x = c[:, 1]
        return (2.0 * NU * K * xp.sin(K * x)) / (A0 + xp.cos(K * x))

    system = pde.burgers(
        dom, viscosity=NU, initial=initial, periodic_boundary=periodic_boundary
    )
    return system, _burgers_exact


def _rd_exact(pts: np.ndarray) -> dict[str, np.ndarray]:
    t, x = pts[:, 0], pts[:, 1]
    return {
        "u": np.sin(x) * np.exp(-(DU + 1.0) * t),
        "v": np.cos(x) * np.exp(-(DV + 1.0) * t),
    }


def reaction_diffusion_case(periodic_boundary: bool) -> tuple[Any, Any]:
    dom = pde.Domain(
        ("t", "x"),
        ((0.0, T_END), (0.0, 2.0 * math.pi)),
        periodic=(False, True),
        time_axis="t",
    )

    def initial_u(c):  # noqa: ANN001, ANN202
        xp = pde.array_namespace(c)
        return xp.sin(c[:, 1])

    def initial_v(c):  # noqa: ANN001, ANN202
        xp = pde.array_namespace(c)
        return xp.cos(c[:, 1])

    def reaction(u, v):  # noqa: ANN001, ANN202
        return (-u, -v)

    system = pde.reaction_diffusion(
        dom,
        diffusivities=(DU, DV),
        reaction=reaction,
        initial=(initial_u, initial_v),
        periodic_boundary=periodic_boundary,
    )
    return system, _rd_exact


CASES = {
    "burgers": burgers_case,
    "reaction_diffusion": reaction_diffusion_case,
}


# ------------------------------------------------------------------ metrics --


def _interior_grid(system: Any, n: int = 24) -> np.ndarray:
    axes = [
        np.linspace(lo + 0.02 * (hi - lo), hi - 0.02 * (hi - lo), n)
        for lo, hi in system.domain.bounds
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=-1)


def interior_rel_l2(sol: Any, exact: Any) -> float:
    pts = _interior_grid(sol.system)
    if callable(exact) and sol.system.name == "burgers":
        u = sol.evaluate(pts, "u").detach().numpy()
        truth = exact(pts)
        return float(np.linalg.norm(u - truth) / np.linalg.norm(truth))
    truth = exact(pts)
    errs = []
    for name in sol.system.component_names():
        u = sol.evaluate(pts, name).detach().numpy()
        t = truth[name]
        errs.append(float(np.linalg.norm(u - t) / max(np.linalg.norm(t), 1e-30)))
    return float(max(errs))


def seam_violation(sol: Any) -> float:
    """Max ``|d^n u(hi)-d^n u(lo)|`` over components, orders 0/1, on spatial seams."""
    domain = sol.system.domain
    cs = domain.coordinate_spec
    axes = [a for a in domain.spatial_axes if cs.is_periodic(a)]
    if not axes:
        return 0.0
    worst = 0.0
    for axis in axes:
        pts = boundary_points(domain, SPEC, axis=axis, side="lo")
        if pts.shape[0] == 0:
            continue
        index = cs.axis_index(axis)
        _, hi = domain.bounds[index]
        low = to_tensor(pts, sol.field)
        high = low.clone()
        high[:, index] = hi
        lo_state, hi_state = sol.field(low), sol.field(high)
        for name in sol.system.component_names():
            for order in (0, 1):
                if order == 0:
                    a = lo_state.ops.value(lo_state, name)
                    b = hi_state.ops.value(hi_state, name)
                else:
                    a = lo_state.ops.derivative(lo_state, name, axis=axis, order=1)
                    b = hi_state.ops.derivative(hi_state, name, axis=axis, order=1)
                worst = max(worst, float((a - b).detach().abs().max()))
    return worst


# --------------------------------------------------------------------- run ---


def run_cell(case: str, periodic_boundary: bool, mode: str, seed: int) -> dict[str, Any]:
    torch.set_default_dtype(torch.float64)
    system, exact = CASES[case](periodic_boundary)
    started = time.perf_counter()
    sol = pt.solve_optimize(
        system,
        hidden=HIDDEN,
        weight_init_scale=3.0,
        seed=seed,
        collocation=SPEC,
        adam_iters=ADAM,
        iters=ITERS,
        hard_conditions=mode,
    )
    return {
        "case": case,
        "periodic_boundary": periodic_boundary,
        "hard_conditions": mode,
        "seed": seed,
        "interior_relative_l2": interior_rel_l2(sol, exact),
        "seam_violation": seam_violation(sol),
        "residual_norm": sol.residual_norm,
        "absorbed": sol.diagnostics.get("hard_absorbed", 0),
        "n_boundary": len(system.boundary),
        "seconds": round(time.perf_counter() - started, 3),
    }


def _median(cells: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(c[key] for c in cells))


def _best_arm(
    cells: list[dict[str, Any]], case: str, flag: bool
) -> dict[str, float]:
    """Per flag, take the hard/soft arm with the smaller median interior error."""
    block = [
        c
        for c in cells
        if c["case"] == case and c["periodic_boundary"] is flag
    ]
    summary = {}
    for mode in ("auto", "none"):
        arm = [c for c in block if c["hard_conditions"] == mode]
        summary[mode] = {
            "interior_l2_median": _median(arm, "interior_relative_l2"),
            "seam_violation_median": _median(arm, "seam_violation"),
        }
    # Prefer the arm that wins on interior; report its seam too.
    winner = min(summary, key=lambda m: summary[m]["interior_l2_median"])
    return {
        "arm": winner,
        "interior_l2_median": summary[winner]["interior_l2_median"],
        "seam_violation_median": summary[winner]["seam_violation_median"],
        "by_arm": summary,
    }


def decide(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Flip default only if True is strictly better on both metrics for both cases."""
    per_case: dict[str, Any] = {}
    flip = True
    reasons: list[str] = []
    for case in CASES:
        off = _best_arm(cells, case, False)
        on = _best_arm(cells, case, True)
        better_interior = on["interior_l2_median"] < off["interior_l2_median"]
        better_seam = on["seam_violation_median"] < off["seam_violation_median"]
        wins = better_interior and better_seam
        if not wins:
            flip = False
            reasons.append(
                f"{case}: interior {on['interior_l2_median']:.3e} vs "
                f"{off['interior_l2_median']:.3e}; seam "
                f"{on['seam_violation_median']:.3e} vs "
                f"{off['seam_violation_median']:.3e}"
            )
        per_case[case] = {"off": off, "on": on, "strictly_better": wins}
    return {
        "flip_default_to_true": flip,
        "rule": (
            "flip only if True is strictly better on both interior rel-L2 and "
            "seam violation for both burgers and reaction_diffusion "
            "(best hard/soft arm, median over seeds)"
        ),
        "per_case": per_case,
        "reasons_to_keep_false": reasons,
    }


def main() -> None:
    jobs = [
        (case, flag, mode, seed)
        for case in CASES
        for flag in (False, True)
        for mode in ("none", "auto")
        for seed in SEEDS
    ]
    print(
        f"periodic_boundary emit measure: {len(jobs)} cells "
        f"(hidden={HIDDEN}, adam={ADAM}, iters={ITERS}, seeds={list(SEEDS)})"
    )
    started = time.perf_counter()
    cells = [run_cell(case, flag, mode, seed) for case, flag, mode, seed in jobs]
    elapsed = time.perf_counter() - started

    decision = decide(cells)
    for case, row in decision["per_case"].items():
        off, on = row["off"], row["on"]
        print(f"\n  {case}")
        print(
            f"    off  arm={off['arm']:4s}  interior {off['interior_l2_median']:.3e}"
            f"  seam {off['seam_violation_median']:.3e}"
        )
        print(
            f"    on   arm={on['arm']:4s}  interior {on['interior_l2_median']:.3e}"
            f"  seam {on['seam_violation_median']:.3e}"
            f"  strictly_better={row['strictly_better']}"
        )
    print(
        f"\nflip_default_to_true = {decision['flip_default_to_true']}"
        f"  ({'FLIP' if decision['flip_default_to_true'] else 'keep False / opt-in'})"
    )

    payload = provenance(
        schema="hard_conditions_periodic_emit_measure/v1",
        config={
            "seeds": list(SEEDS),
            "hidden": HIDDEN,
            "adam_iters": ADAM,
            "iters": ITERS,
            "n_interior": SPEC.n_interior,
            "n_boundary": SPEC.n_boundary,
            "driver": "solve_optimize",
            "decision_rule": decision["rule"],
        },
    )
    payload["cells"] = cells
    payload["decision"] = decision
    payload["elapsed_seconds"] = round(elapsed, 1)
    path = write_json(OUT_NAME, payload)
    print(f"wrote {path} in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
