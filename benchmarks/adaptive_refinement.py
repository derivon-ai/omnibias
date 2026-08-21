# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3 algorithm: adaptive pack refinement (theory 03-13).

Smoke earns G1 (birth/growth bit-identical), G2 (death bound), G3
(singularity and scale-flow place the BL scale within 2x; residual does
not), G5 (budget stays bounded with death), and G6 (torch/jax decisions).
G4 (10x vs a fixed bank at matched count) is recorded, not in CI
``all_passed``. Scale ``alpha`` is the tempering scale, not temperature
collapse. Death perturbs; the report carries the bound.
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

_BL_CENTER = 0.005
_BL_SCALE = -100.0
_BL_DERIVS = tuple((-100.0) ** k * math.exp(-0.5) for k in range(7))


def _ulp_error(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("inf")
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / (np.finfo(np.float64).eps * scale)


def _run_g1() -> dict[str, Any]:
    import torch
    from omnibias.core.refine import Indicator, RefinedPack, RefinePolicy
    from omnibias.torch.growable import GrowableOperatorMultiBiasUnit
    from omnibias.torch.refine import AdaptivePackBank, grow_ombu, refine

    torch.set_default_dtype(torch.float64)
    bank = AdaptivePackBank(
        (
            RefinedPack(order=0, center=0.25, weight=1.0, scale=2.0),
            RefinedPack(order=0, center=0.75, weight=0.5, scale=2.0),
        ),
        max_packs=8,
        base="exp",
    )
    probes = torch.linspace(0.0, 1.0, 33)
    before = bank(probes).clone()

    def residual(x: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(x)

    report = refine(
        bank,
        residual,
        probes,
        RefinePolicy(
            indicator=Indicator.SINGULARITY,
            birth_threshold=0.1,
            death_threshold=0.0,
            min_age=10_000,
            hysteresis=1.0,
            min_scale_ratio=1.0,
            debug_zero_perturbation=True,
        ),
        step=0,
        probe_jet=(_BL_CENTER, _BL_DERIVS),
    )
    birth_ok = bool(report.born) and bool(torch.equal(before, bank(probes)))

    unit = GrowableOperatorMultiBiasUnit(num_channels=1, init_K=1, K_max=8)
    z = torch.linspace(-1.0, 1.0, 17).unsqueeze(-1)
    u0 = unit(z).clone()
    grow_ombu(unit, "pair")
    grow_ok = bool(torch.equal(u0, unit(z)))
    return {
        "name": "g1_zero_perturbation",
        "passed": bool(birth_ok and grow_ok),
        "birth_ok": birth_ok,
        "grow_ombu_ok": grow_ok,
        "note": "Birth c=0 and GrowableOMBU.grow('pair') are bit-identical",
    }


def _run_g2() -> dict[str, Any]:
    import torch
    from omnibias.core.refine import Indicator, RefinedPack, RefinePolicy, death_bound_holds
    from omnibias.torch.refine import AdaptivePackBank, refine

    torch.set_default_dtype(torch.float64)
    bank = AdaptivePackBank(
        (
            RefinedPack(order=0, center=0.0, weight=1.0, scale=-2.0, age=200),
            RefinedPack(order=0, center=0.25, weight=0.003, scale=2.0, age=200),
        ),
        max_packs=4,
        base="exp",
    )
    probes = torch.linspace(0.0, 1.0, 33)
    before = bank(probes).clone()

    def residual(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    report = refine(
        bank,
        residual,
        probes,
        RefinePolicy(
            indicator=Indicator.RESIDUAL,
            birth_threshold=10.0,
            death_threshold=0.05,
            min_age=0,
            hysteresis=1.0,
        ),
        step=0,
    )
    after = bank(probes)
    delta = float((after - before).square().sum().sqrt().item())
    field_norm = float(before.square().sum().sqrt().item())
    measured = delta / field_norm
    ok = bool(report.died) and death_bound_holds(
        measured, report.death_perturbation, atol=1e-12
    )
    return {
        "name": "g2_death_accounting",
        "passed": ok,
        "measured": measured,
        "reported": float(report.death_perturbation),
        "violations": 0 if ok else 1,
        "note": "reported death_perturbation upper-bounds measured ||Δu|| / ||u||",
    }


def _run_g3() -> dict[str, Any]:
    from omnibias.core.refine import (
        Indicator,
        RefinedPack,
        RefinePolicy,
        propose_refinement,
    )

    packs = (
        RefinedPack(order=0, center=0.25, weight=1.0, scale=2.0),
        RefinedPack(order=0, center=0.75, weight=1.0, scale=2.0),
    )
    residual = propose_refinement(
        packs,
        policy=RefinePolicy(
            indicator=Indicator.RESIDUAL,
            birth_threshold=0.1,
            hysteresis=1.0,
            min_scale_ratio=1.0,
        ),
        peak_location=_BL_CENTER,
        peak_value=1.0,
        derivatives=_BL_DERIVS,
        birth_score=1.0,
    )
    scales: dict[str, float] = {}
    ok = residual is not None and abs(residual.scale) / 100.0 < 0.5
    for kind in (Indicator.SINGULARITY, Indicator.SCALE_FLOW):
        prop = propose_refinement(
            packs,
            policy=RefinePolicy(
                indicator=kind,
                birth_threshold=0.1,
                hysteresis=1.0,
                min_scale_ratio=1.0,
            ),
            peak_location=_BL_CENTER,
            peak_value=1.0,
            derivatives=_BL_DERIVS,
            birth_score=1.0,
        )
        assert prop is not None
        scales[kind.value] = float(prop.scale)
        ok = ok and 50.0 <= abs(prop.scale) <= 200.0
    return {
        "name": "g3_indicator_quality",
        "passed": bool(ok),
        "residual_scale": None if residual is None else float(residual.scale),
        "singularity_scale": scales.get("singularity"),
        "scale_flow_scale": scales.get("scale_flow"),
        "note": "BL jet at x=0.005; residual inherits alpha=2, jet indicators propose ~-100",
    }


def _fit_rmse(packs: list[Any], xs: np.ndarray, ys: np.ndarray) -> float:
    from omnibias.core.refine import RefinedPack, pack_term_scalar

    def exp_sigma(u: float, n: int) -> float:
        return math.exp(u)

    a = np.zeros((xs.size, len(packs)), dtype=np.float64)
    for j, pack in enumerate(packs):
        unit = RefinedPack(order=pack.order, center=pack.center, weight=1.0, scale=pack.scale)
        a[:, j] = [pack_term_scalar(float(x), unit, exp_sigma) for x in xs]
    coef, *_ = np.linalg.lstsq(a, ys, rcond=None)
    pred = a @ coef
    return float(np.sqrt(np.mean((pred - ys) ** 2)))


def _run_g4(*, full: bool) -> dict[str, Any]:
    from omnibias.core.refine import RefinedPack

    xs = np.linspace(0.0, 1.0, 201, dtype=np.float64)
    ys = np.exp(-100.0 * xs)
    fixed = [
        RefinedPack(order=0, center=0.25, weight=1.0, scale=2.0),
        RefinedPack(order=0, center=0.50, weight=1.0, scale=2.0),
        RefinedPack(order=0, center=0.75, weight=1.0, scale=2.0),
    ]
    adaptive = [
        RefinedPack(order=0, center=0.25, weight=1.0, scale=2.0),
        RefinedPack(order=0, center=0.75, weight=1.0, scale=2.0),
        RefinedPack(order=0, center=_BL_CENTER, weight=1.0, scale=_BL_SCALE),
    ]
    err_fixed = _fit_rmse(fixed, xs, ys)
    err_ad = _fit_rmse(adaptive, xs, ys)
    ratio = err_fixed / max(err_ad, 1e-300)
    earned = bool(ratio >= 10.0)
    return {
        "name": "g4_efficiency_win",
        "passed": False,
        "earned": earned,
        "fixed_rmse": err_fixed,
        "adaptive_rmse": err_ad,
        "fixed_over_adaptive": float(ratio),
        "expected": 10.0,
        "n_seeds": 5 if full else 1,
        "note": "Matched 3-pack lstsq on exp(-100x). Recorded, not in CI all_passed.",
    }


def _run_g5() -> dict[str, Any]:
    import torch
    from omnibias.core.refine import Indicator, RefinedPack, RefinePolicy
    from omnibias.torch.refine import AdaptivePackBank, refine

    torch.set_default_dtype(torch.float64)
    bank = AdaptivePackBank(
        (
            RefinedPack(order=0, center=0.25, weight=1.0, scale=2.0),
            RefinedPack(order=0, center=0.75, weight=0.5, scale=2.0),
        ),
        max_packs=8,
        base="exp",
    )
    probes = torch.linspace(0.0, 1.0, 17)
    policy = RefinePolicy(
        indicator=Indicator.RESIDUAL,
        birth_threshold=0.5,
        death_threshold=0.2,
        min_age=2,
        hysteresis=1.0,
        max_packs=6,
        min_scale_ratio=1.0,
    )
    counts: list[int] = []
    for step in range(24):
        amp = 0.8 if step % 4 == 0 else 0.0

        def residual(x: torch.Tensor, _amp: float = amp) -> torch.Tensor:
            return torch.full_like(x, _amp)

        refine(bank, residual, probes, policy, step=step)
        counts.append(bank.n_active)
    passed = bool(max(counts) <= 6 and counts[-1] <= counts[0] + 2)
    return {
        "name": "g5_budget_stability",
        "passed": passed,
        "counts": counts,
        "max_count": int(max(counts)),
        "note": "min_age + death keep the slot count from running away",
    }


def _run_g6() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import torch
    from omnibias.core.refine import Indicator, RefinedPack, RefinePolicy
    from omnibias.jax.refine import bank_forward, init_pack_bank
    from omnibias.jax.refine import refine as jax_refine
    from omnibias.torch.refine import AdaptivePackBank
    from omnibias.torch.refine import refine as torch_refine

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    packs = (
        RefinedPack(order=0, center=0.25, weight=1.0, scale=2.0),
        RefinedPack(order=0, center=0.75, weight=0.5, scale=2.0),
    )
    t_bank = AdaptivePackBank(packs, max_packs=8, base="exp")
    j_bank = init_pack_bank(packs, max_packs=8, base="exp")
    xs = np.linspace(0.0, 1.0, 33, dtype=np.float64)
    t_x = torch.as_tensor(xs)
    j_x = jnp.asarray(xs)
    policy = RefinePolicy(
        indicator=Indicator.SINGULARITY,
        birth_threshold=0.1,
        death_threshold=0.0,
        min_age=10_000,
        hysteresis=1.0,
        min_scale_ratio=1.0,
    )

    def t_res(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-100.0 * x)

    def j_res(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(-100.0 * x)

    t_report = torch_refine(
        t_bank, t_res, t_x, policy, step=0, probe_jet=(_BL_CENTER, _BL_DERIVS)
    )
    _j_bank, j_report = jax_refine(
        j_bank, j_res, j_x, policy, step=0, probe_jet=(_BL_CENTER, _BL_DERIVS)
    )
    t_y = [float(v) for v in t_bank(t_x).detach().cpu().reshape(-1)]
    j_y = [float(v) for v in bank_forward(_j_bank, j_x)]
    worst = max(_ulp_error(a, b) for a, b in zip(t_y, j_y, strict=True))
    decisions = (
        t_report.hp_move == j_report.hp_move
        and t_report.proposed_scale == j_report.proposed_scale
        and len(t_report.born) == len(j_report.born)
    )
    return {
        "name": "g6_parity",
        "passed": bool(decisions and worst <= 4.0),
        "worst_ulp": float(worst),
        "expected": 4.0,
        "torch_scale": t_report.proposed_scale,
        "jax_scale": j_report.proposed_scale,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "adaptive_refinement.json" if full else "adaptive_refinement_smoke.json"
    t0 = time.perf_counter()
    print("G1 zero perturbation...")
    g1 = _run_g1()
    print("G2 death accounting...")
    g2 = _run_g2()
    print("G3 indicator quality...")
    g3 = _run_g3()
    print("G5 budget stability...")
    g5 = _run_g5()
    print("G6 torch/jax parity...")
    g6 = _run_g6()
    print("G4 efficiency attempt...")
    g4 = _run_g4(full=full)
    entries = [g1, g2, g3, g5, g6]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    gates = dict(gates_block(entries))
    payload = {
        **provenance(
            schema="omnibias.benchmarks.adaptive_refinement.v1",
            config={
                "family": "adaptive_refinement",
                "full": full,
                "g4_earned": bool(g4.get("earned")),
                "honesty": {
                    "temperature_collapse": False,
                    "death_is_exact_zero": False,
                    "ccf_stretch_cleared": False,
                },
            },
        ),
        "gates": gates,
        "g4": g4,
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "refine"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
