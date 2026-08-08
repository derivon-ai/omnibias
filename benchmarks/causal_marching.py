# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Causal marching vs whole-interval / causal-only / marching-only.

Two problem families share the same arm ladder and equal advertised step
budget. Both use an omnibias field with closed-form residuals, a hard IC
cage, and ``advance_policy="gate"``.

Families
--------
* ``heat`` -- manufactured ``u = sin(pi x) exp(-pi^2 t)`` for ``u_t = u_xx``
  on ``[0,1] x [0,0.5]``. Smooth decaying solution; whole-interval is
  expected to win.
* ``reaction`` -- Krishnapriyan reaction ``u_t = rho u (1 - u)`` at
  ``rho = 12`` on ``[0, 2 pi) x [0, 1]`` with a Gaussian bump IC and the
  logistic exact solution. Whole-interval is the classical causality
  failure; marching is gated to beat it.

Seam metric
-----------
The IC is always supplied as ``ic_fn`` evaluated on the marcher's own
``initial_points()``. Passing a linspace-ordered ``ic_values`` vector is
refused because ``slice_points`` draws space uniformly at random, so a
positional vector would measure a grid-alignment artifact rather than
handoff error.

Modes
-----
* default (smoke): 1 seed, tiny nets / steps — CI wiring gate.
* ``--full``: 5 seeds, larger budget — acceptance artifact.

Gates (absolute, in order)
--------------------------
1. Every arm reports finite metrics and ``skill_score > 0``.
2. Heat: median rel-L2 of the best arm clears a named threshold.
3. Reaction: best marching arm median rel-L2 beats ``whole_interval``.
4. Marching arms use ``advance_policy="gate"`` (exhausted windows stay put).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # noqa: E402
from _gates import gates_block, rel_l2, skill_score  # noqa: E402
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn._core.constrained import HardCondition, dirichlet
from omnibias.pinn._core.marching import TimeWindowSchedule
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.cage import ConstrainedExpressionField
from omnibias.pinn.torch.fields import JetMLPVectorField, OneLayerVectorField
from omnibias.pinn.train.torch import march_solve

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
DTYPE = torch.float64
RHO = 12.0
LENGTH = 2.0 * math.pi


@dataclass(frozen=True)
class ProblemFamily:
    name: str
    t_final: float
    residual_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor]
    build_field: Callable[..., nn.Module]
    exact: Callable[[torch.Tensor], torch.Tensor]
    ic_fn: Callable[[torch.Tensor], torch.Tensor]
    pde: str
    solution: str
    coordinate_order: str
    max_rel_l2_smoke: float
    max_rel_l2_full: float
    ic_mode: str = "hard"
    ic_weight: float = 1.0
    n_march_windows: int = 4
    causal_epsilon: float = 1.0


@dataclass(frozen=True)
class ArmConfig:
    name: str
    n_windows: int
    epsilon: float
    steps_per_window: int
    n_time_bins: int


def _heat_exact(coords: torch.Tensor) -> torch.Tensor:
    """``coords`` columns are ``(t, x)``."""
    return torch.sin(math.pi * coords[:, 1]) * torch.exp(-(math.pi**2) * coords[:, 0])


def _heat_ic(coords: torch.Tensor) -> torch.Tensor:
    return torch.sin(math.pi * coords[:, 1])


def _build_heat_field(*, width: int, seed: int, t_final: float) -> ConstrainedExpressionField:
    torch.manual_seed(seed)
    cs = CoordinateSpec(
        ("t", "x"), domain=((0.0, t_final), (0.0, 1.0)), time_axis="t"
    )
    base = OneLayerVectorField(
        coordinate_spec=cs,
        components=ComponentSpec(("u",)),
        hidden=width,
        base="tanh",
        dtype=DTYPE,
    )
    return ConstrainedExpressionField(
        base=base,
        conditions=[
            HardCondition("u", 1, dirichlet(0.0), 0.0),
            HardCondition("u", 1, dirichlet(1.0), 0.0),
            HardCondition(
                "u", 0, dirichlet(0.0), lambda c: torch.sin(math.pi * c[:, 1])
            ),
        ],
    )


def _heat_residual(fld: nn.Module, coords: torch.Tensor) -> torch.Tensor:
    state = fld(coords)
    u_t = tops.derivative(state, "u", axis="t", order=1)
    u_xx = tops.derivative(state, "u", axis="x", order=2)
    return u_t - u_xx


def _reaction_u0(x: torch.Tensor) -> torch.Tensor:
    """Gaussian bump ``exp(-(x - pi)^2 / (2 (pi/4)^2))``."""
    return torch.exp(-((x - math.pi) ** 2) / (2.0 * (math.pi / 4.0) ** 2))


def _reaction_cs(t_final: float) -> CoordinateSpec:
    # Match the classical Krishnapriyan layout: (x, t).
    return CoordinateSpec(
        ("x", "t"), domain=((0.0, LENGTH), (0.0, t_final)), time_axis="t"
    )


def _build_reaction_field(*, width: int, seed: int, t_final: float) -> nn.Module:
    """Free JetMLP; soft IC is applied by ``march_solve`` (not a hard cage).

    The classical causality failure needs a soft IC weight cliff: with
    ``ic_weight=1`` whole-interval collapses, with ``ic_weight=10`` it
    partially recovers, and marching at the same budget beats it.
    """
    torch.manual_seed(seed)
    return JetMLPVectorField(
        coordinate_spec=_reaction_cs(t_final),
        components=ComponentSpec(("u",)),
        hidden=max(width, 32),
        depth=3,
        jet_order=1,
        dtype=DTYPE,
    )


def _reaction_residual(fld: nn.Module, coords: torch.Tensor) -> torch.Tensor:
    state = fld(coords)
    u = tops.value(state, "u")
    u_t = tops.derivative(state, "u", axis="t", order=1)
    return u_t - RHO * u * (1.0 - u)


def _reaction_exact_xt(coords: torch.Tensor) -> torch.Tensor:
    """Logistic growth; ``coords`` columns are ``(x, t)``."""
    h = _reaction_u0(coords[:, 0])
    growth = torch.exp(RHO * coords[:, 1])
    return h * growth / (h * growth + 1.0 - h)


def _reaction_ic_xt(coords: torch.Tensor) -> torch.Tensor:
    return _reaction_u0(coords[:, 0])


HEAT = ProblemFamily(
    name="heat",
    t_final=0.5,
    residual_fn=_heat_residual,
    build_field=_build_heat_field,
    exact=_heat_exact,
    ic_fn=_heat_ic,
    pde="u_t = u_xx",
    solution="sin(pi x) exp(-pi^2 t)",
    coordinate_order="(t, x)",
    max_rel_l2_smoke=0.5,
    max_rel_l2_full=0.25,
    ic_mode="hard",
    n_march_windows=4,
    causal_epsilon=1.0,
)

REACTION = ProblemFamily(
    name="reaction",
    t_final=1.0,
    residual_fn=_reaction_residual,
    build_field=_build_reaction_field,
    exact=_reaction_exact_xt,
    ic_fn=_reaction_ic_xt,
    pde="u_t = rho u (1 - u), rho=12",
    solution="logistic of Gaussian bump",
    coordinate_order="(x, t)",
    max_rel_l2_smoke=2.0,
    max_rel_l2_full=1.0,
    ic_mode="soft",
    ic_weight=10.0,
    n_march_windows=5,
    causal_epsilon=0.5,
)


def _value_fn(fld: nn.Module, coords: torch.Tensor) -> torch.Tensor:
    return tops.value(fld(coords), "u")


def _eval_metrics(
    field: nn.Module,
    problem: ProblemFamily,
    *,
    cs: CoordinateSpec,
    n: int = 64,
) -> dict[str, float]:
    # Build a dense grid in the field's own coordinate order.
    axes = []
    for i, name in enumerate(cs.axes):
        lo, hi = cs.domain[i]
        if name != cs.time_axis and abs(hi - LENGTH) < 1e-12:
            # Periodic-like spatial domain for the reaction family.
            axes.append(torch.linspace(lo, hi, n + 1, dtype=DTYPE)[:-1])
        else:
            axes.append(torch.linspace(lo, hi, n, dtype=DTYPE))
    mesh = torch.meshgrid(*axes, indexing="ij")
    coords = torch.stack([m.reshape(-1) for m in mesh], dim=-1)
    with torch.no_grad():
        pred = _value_fn(field, coords).detach().cpu().numpy()
    exact = problem.exact(coords).detach().cpu().numpy()
    return {
        "reference_mse": float(np.mean((pred - exact) ** 2)),
        "rel_l2": rel_l2(pred, exact),
        "skill_score": skill_score(pred, exact),
        "zero_predictor_mse": float(np.mean(exact**2)),
    }


def _run_arm(
    arm: ArmConfig,
    problem: ProblemFamily,
    *,
    seed: int,
    width: int,
    n_slice: int,
    per_bin: int,
) -> dict[str, Any]:
    field = problem.build_field(width=width, seed=seed, t_final=problem.t_final)
    if isinstance(field, ConstrainedExpressionField):
        cs = field.base.coordinate_spec
    else:
        cs = field.coordinate_spec  # type: ignore[attr-defined]
    schedule = TimeWindowSchedule(
        0.0,
        problem.t_final,
        n_windows=arm.n_windows,
        n_time_bins=arm.n_time_bins,
        epsilon=arm.epsilon,
        tolerance=1e-3 if problem.name == "heat" else 0.1,
    )

    t0 = time.perf_counter()
    result = march_solve(
        field,
        problem.residual_fn,
        cs,
        schedule,
        steps_per_window=arm.steps_per_window,
        max_steps_per_window=max(arm.steps_per_window * 4, arm.steps_per_window),
        lr=3e-3 if problem.name == "reaction" else 1e-2,
        per_bin=per_bin,
        n_slice=n_slice,
        ic_fn=problem.ic_fn,
        ic_mode=problem.ic_mode,
        ic_weight=problem.ic_weight,
        value_fn=_value_fn,
        seed=seed,
        dtype=DTYPE,
        check_trivial=True,
        trivial_mode="variance",
        advance_policy="gate",
    )
    elapsed = time.perf_counter() - t0
    metrics = _eval_metrics(field, problem, cs=cs)
    seam = [
        float(w.seam_mse) if w.seam_mse is not None else float("nan")
        for w in result.windows
    ]
    total_steps = int(sum(w.steps_run for w in result.windows))
    return {
        "family": problem.name,
        "arm": arm.name,
        "seed": seed,
        "n_windows": len(result.windows),
        "all_converged": result.all_converged,
        "reference_mse": metrics["reference_mse"],
        "rel_l2": metrics["rel_l2"],
        "skill_score": metrics["skill_score"],
        "zero_predictor_mse": metrics["zero_predictor_mse"],
        "mean_seam_mse": float(np.nanmean(seam)) if seam else float("nan"),
        "last_causality_index": result.windows[-1].causality.causality_index,
        "unlocked_fraction": result.windows[-1].causality.unlocked_fraction,
        "trivial": bool(result.trivial.is_trivial) if result.trivial else None,
        "elapsed_seconds": elapsed,
        "total_steps": total_steps,
        "advertised_steps": arm.steps_per_window * arm.n_windows,
        "advance_policy": "gate",
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = f"{row['family']}/{row['arm']}"
        by_key.setdefault(key, []).append(row)
    out: dict[str, Any] = {}
    for name, group in by_key.items():
        ref = np.asarray([g["reference_mse"] for g in group], dtype=float)
        rel = np.asarray([g["rel_l2"] for g in group], dtype=float)
        skill = np.asarray([g["skill_score"] for g in group], dtype=float)
        seam = np.asarray([g["mean_seam_mse"] for g in group], dtype=float)
        times = np.asarray([g["elapsed_seconds"] for g in group], dtype=float)
        steps = np.asarray([g["total_steps"] for g in group], dtype=float)
        trivial = np.asarray([bool(g["trivial"]) for g in group], dtype=float)
        out[name] = {
            "n_seeds": len(group),
            "reference_mse_median": float(np.median(ref)),
            "rel_l2_median": float(np.median(rel)),
            "skill_score_median": float(np.median(skill)),
            "seam_mse_median": float(np.median(seam)),
            "trivial_rate": float(np.mean(trivial)),
            "elapsed_seconds_median": float(np.median(times)),
            "total_steps_median": float(np.median(steps)),
            "total_steps_max": float(np.max(steps)),
            "converged_rate": float(
                np.mean([bool(g["all_converged"]) for g in group])
            ),
        }
    return out


def _arms(problem: ProblemFamily, budget: int, bins: int) -> list[ArmConfig]:
    n_win = problem.n_march_windows
    eps = problem.causal_epsilon
    return [
        ArmConfig("whole_interval", 1, 0.0, budget, bins),
        ArmConfig("causal_only", 1, eps, budget, bins),
        ArmConfig("marching_only", n_win, 0.0, budget // n_win, bins),
        ArmConfig("causal_marching", n_win, eps, budget // n_win, bins),
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="multi-seed acceptance run (default is smoke)",
    )
    args = parser.parse_args(argv)
    smoke = not args.full

    if smoke:
        seeds = [0]
        # Reaction needs a real budget even in smoke to expose causality.
        heat_width, heat_slice, heat_bin, heat_bins, heat_budget = 32, 24, 6, 6, 320
        rxn_width, rxn_slice, rxn_bin, rxn_bins, rxn_budget = 48, 48, 16, 8, 600
    else:
        seeds = list(range(5))
        heat_width, heat_slice, heat_bin, heat_bins, heat_budget = 64, 32, 8, 8, 800
        rxn_width, rxn_slice, rxn_bin, rxn_bins, rxn_budget = 48, 96, 32, 16, 2000

    families = (HEAT, REACTION)
    family_cfg = {
        "heat": (heat_width, heat_slice, heat_bin, heat_bins, heat_budget),
        "reaction": (rxn_width, rxn_slice, rxn_bin, rxn_bins, rxn_budget),
    }

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for problem in families:
        width, n_slice, per_bin, bins, budget = family_cfg[problem.name]
        for arm in _arms(problem, budget, bins):
            for seed in seeds:
                rows.append(
                    _run_arm(
                        arm,
                        problem,
                        seed=seed,
                        width=width,
                        n_slice=n_slice,
                        per_bin=per_bin,
                    )
                )

    summary = _summarize(rows)
    gate_entries: list[dict[str, Any]] = []
    for row in rows:
        skill = float(row["skill_score"])
        passed = bool(skill > 0.0 and np.isfinite(row["rel_l2"]))
        gate_entries.append(
            {
                "name": f"{row['family']}_{row['arm']}_seed{row['seed']}_skill",
                "skill_score": skill,
                "rel_l2": float(row["rel_l2"]),
                "passed": passed,
            }
        )
        if not passed:
            raise AssertionError(
                f"{row['family']}/{row['arm']} seed {row['seed']}: "
                f"skill={skill:.4f} (must beat the zero predictor)"
            )

    # Heat: best-arm absolute threshold.
    heat_keys = [k for k in summary if k.startswith("heat/")]
    best_heat = min(heat_keys, key=lambda k: summary[k]["rel_l2_median"])
    max_rel_heat = HEAT.max_rel_l2_smoke if smoke else HEAT.max_rel_l2_full
    best_heat_rel = float(summary[best_heat]["rel_l2_median"])
    heat_ok = bool(best_heat_rel <= max_rel_heat)
    gate_entries.append(
        {
            "name": "heat_best_arm_rel_l2",
            "best_arm": best_heat,
            "rel_l2_median": best_heat_rel,
            "max_rel_l2": max_rel_heat,
            "passed": heat_ok,
        }
    )
    if not heat_ok:
        raise AssertionError(
            f"heat best arm {best_heat} median rel_l2={best_heat_rel:.4e} "
            f"exceeds gate {max_rel_heat}"
        )

    # Reaction: marching must beat whole_interval (the causality claim).
    whole = summary["reaction/whole_interval"]["rel_l2_median"]
    marching_keys = [
        k
        for k in ("reaction/marching_only", "reaction/causal_marching")
        if k in summary
    ]
    best_march = min(marching_keys, key=lambda k: summary[k]["rel_l2_median"])
    best_march_rel = float(summary[best_march]["rel_l2_median"])
    march_beats = bool(best_march_rel < float(whole))
    gate_entries.append(
        {
            "name": "reaction_marching_beats_whole_interval",
            "best_marching_arm": best_march,
            "marching_rel_l2_median": best_march_rel,
            "whole_interval_rel_l2_median": float(whole),
            "passed": march_beats,
        }
    )
    if not march_beats:
        # Discovery doctrine: state the loss plainly; do not silently drop.
        raise AssertionError(
            f"reaction: best marching arm {best_march} "
            f"rel_l2={best_march_rel:.4e} did not beat "
            f"whole_interval={float(whole):.4e}"
        )

    payload = provenance(
        schema="causal_marching/v4",
        config={
            "mode": "smoke" if smoke else "full",
            "seeds": seeds,
            "advance_policy": "gate",
            "closed_form_residual": True,
            "ic_supply": "ic_fn (evaluated on marcher slice_points)",
            "families": {
                fam.name: {
                    "t_final": fam.t_final,
                    "pde": fam.pde,
                    "solution": fam.solution,
                    "coordinate_order": fam.coordinate_order,
                    "ic_mode": fam.ic_mode,
                    "ic_weight": fam.ic_weight,
                    "n_march_windows": fam.n_march_windows,
                    "causal_epsilon": fam.causal_epsilon,
                    "width": family_cfg[fam.name][0],
                    "n_slice": family_cfg[fam.name][1],
                    "per_bin": family_cfg[fam.name][2],
                    "n_time_bins": family_cfg[fam.name][3],
                    "equal_step_budget": family_cfg[fam.name][4],
                }
                for fam in families
            },
            "note_equal_step_budget": (
                "per-family equal_step_budget; gated arms may exceed it via "
                "max_steps_per_window retries -- see total_steps_* in summary"
            ),
        },
    )
    payload.update(
        {
            "runs": rows,
            "summary": summary,
            "best_arm_heat": best_heat,
            "best_arm_reaction_marching": best_march,
            "gates": gates_block(gate_entries),
            "elapsed_seconds": time.perf_counter() - t0,
        }
    )

    SCRATCH.mkdir(parents=True, exist_ok=True)
    if not smoke:
        scratch_path = SCRATCH / "causal_marching_full.json"
        scratch_path.write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {scratch_path}")

    out_name = "causal_marching_smoke.json" if smoke else "causal_marching.json"
    write_json(out_name, payload)
    print(f"wrote docs/benchmarks/{out_name}")


if __name__ == "__main__":
    main()
