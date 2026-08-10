# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""One tick of the DeepMind-style autonomous singularity campaign.

Compatible with Cursor ``/loop``: measure gates, emit diagnosis, advance stage.

Stage order
-----------
0. ``phase0_reproduce_neural`` — DeepMind recipe on neural line CCF until
   dense Wang residual ≤ 1e-13 (stretch). Hardy CAP deferred.
1. Rung-1 / Rung-2 Hardy earn path (Adam forbidden).
2. IPM / Boussinesq, then Phase 5.

Does not forge Clay / Navier–Stokes claims.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT))


def _scratch_dir() -> Path:
    base = Path(os.environ.get("OMNIBIAS_SCRATCH", ROOT / "artifacts"))
    out = base / "deepmind_campaign"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _stage_from_status(
    *,
    stretch_cleared: bool,
    status: dict | None,
) -> str:
    if not stretch_cleared:
        return "phase0_reproduce_neural"
    assert status is not None
    gates = status.get("gates", {})
    if not gates.get("rung1_earned"):
        return "phase1_rung1_residual"
    if not gates.get("rung2_earned"):
        return "phase2_rung2_cap"
    if not status.get("ipm_boussinesq_ready"):
        return "phase4_ipm_boussinesq"
    if not status.get("phase5_partition_earned"):
        return "phase5a_partition"
    if not status.get("phase5_router_earned"):
        return "phase5b_router"
    if not status.get("phase5_obligations_earned"):
        return "phase5c_obligations"
    return "complete"


def run_tick(*, smoke: bool = True, family: str = "1st_unstable") -> dict:
    """Run one campaign tick and write scratch status JSON."""
    t0 = time.perf_counter()
    from reproduce_deepmind_ccf import STRETCH, escalate_loop, run_once  # noqa: E402

    if smoke:
        repro = run_once(smoke=True, multistage_rounds=0)
    else:
        repro = escalate_loop(max_rounds=8, smoke=False)["best"]

    stretch_cleared = bool(repro.get("gates", {}).get("stretch_1e-13_cleared"))
    residual = float(repro["metrics"]["reproduction_dense_max_abs_for_gate"])
    status = None
    adapter_ok = True

    if stretch_cleared:
        from ccf_hardy_rung_acceptance import run_acceptance  # noqa: E402
        from omnibias.pinn.jax.discovery.pipeline import (
            CCFHardyAdapter,
            PipelineConfig,
            run_singularity_pipeline,
        )

        status = run_acceptance(family=family, smoke=smoke)
        pipe = run_singularity_pipeline(
            CCFHardyAdapter(n_scales=3, n_gamma_multiples=2, n_grid=33, steps=8),
            PipelineConfig(seed=0, residual_gate=1e-11),
        )
        adapter_ok = (
            pipe.discovery.get("optimizer") == "martens_grosse_gn"
            and pipe.honesty.get("navier_stokes_proof_claim") is False
        )

    stage = _stage_from_status(stretch_cleared=stretch_cleared, status=status)
    diagnosis = {
        "stage": stage,
        "reproduction_dense_residual": residual,
        "stretch_1e-13_cleared": stretch_cleared,
        "stretch_gate": STRETCH,
        "residual_gap_to_stretch": residual - STRETCH,
        "next_actions": list(repro.get("diagnosis", {}).get("next_actions", [])),
        "optimizer": repro.get("config", {}).get("optimizer"),
        "train_hilbert": repro.get("config", {}).get("train_hilbert"),
        "navier_stokes_proof_claim": False,
    }
    if stretch_cleared and status is not None:
        diagnosis["rung1_earned"] = status["gates"]["rung1_earned"]
        diagnosis["rung2_earned"] = status["gates"]["rung2_earned"]
        diagnosis["dense_residual_hardy"] = status["metrics"][
            "dense_max_abs_vorticity_for_gate"
        ]
        if not status["gates"]["rung1_earned"]:
            diagnosis["next_actions"] = [
                "increase_martens_grosse_budget",
                "enrich_hardy_dictionary",
                "keep_hardy_train_hilbert",
                "never_use_adam_on_earn_path",
                "never_weaken_1e-11_gate",
            ]
        elif not status["gates"]["rung2_earned"]:
            diagnosis["next_actions"] = [
                "tighten_interval_covering",
                "improve_vorticity_jacobian",
                "close_sequence_nk",
            ]
    elif not stretch_cleared:
        diagnosis["next_actions"] = [
            "escalate_reproduce_deepmind_ccf",
            "widen_network_or_mg_steps",
            "try_pv_line_hilbert_if_spectral_stalls",
            "never_weaken_1e-13_stretch",
            *diagnosis["next_actions"],
        ]

    tick = {
        "benchmark": "deepmind_campaign_tick",
        "wall_seconds": time.perf_counter() - t0,
        "stage": stage,
        "reproduction": repro,
        "status": status,
        "diagnosis": diagnosis,
        "adapter_smoke_ok": adapter_ok,
        "honesty": {
            "navier_stokes_proof_claim": False,
            "continuum_claim": False,
            "phase0_before_hardy_cap": True,
            "phase5_blocked_until_rung2": True
            if status is None
            else (not bool(status["gates"]["rung2_earned"])),
        },
        "gates": {
            "stretch_1e-13_cleared": stretch_cleared,
            "rung1_earned": bool(status and status["gates"]["rung1_earned"]),
            "rung2_earned": bool(status and status["gates"]["rung2_earned"]),
            "adapter_smoke_ok": adapter_ok,
            "passed": bool(
                stretch_cleared
                and status
                and status["gates"]["rung1_earned"]
                and status["gates"]["rung2_earned"]
            ),
        },
    }
    out = _scratch_dir() / f"tick_{family}_{'smoke' if smoke else 'full'}.json"
    out.write_text(json.dumps(tick, indent=2, default=str) + "\n", encoding="utf-8")
    tick["artifact"] = str(out)
    return tick


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true")
    p.add_argument("--family", default="1st_unstable")
    args = p.parse_args(argv)
    tick = run_tick(smoke=not args.full, family=args.family)
    print(
        json.dumps(
            {
                "stage": tick["stage"],
                "gates": tick["gates"],
                "residual": tick["diagnosis"]["reproduction_dense_residual"],
                "artifact": tick["artifact"],
            },
            indent=2,
        )
    )
    # Smoke: infrastructure OK if tick wrote and residual is finite.
    ok = bool(
        tick["adapter_smoke_ok"]
        and tick["diagnosis"]["reproduction_dense_residual"] == tick["diagnosis"]["reproduction_dense_residual"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
