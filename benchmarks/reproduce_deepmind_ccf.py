# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""DeepMind-faithful neural CCF reproduction (Wang vorticity, stretch 1e-13).

Primary path: compactified Ω-PINN + ``hardy_corrected_pv`` Hilbert +
Martens–Grosse Gauss–Newton (exact JVP) + optional multistage. Dense residual
is scored on the **neural** profile with matched train Hilbert (not
Hardy-projected for the CAP gate).

Never weakens ``CCF_STRETCH_RESIDUAL_GATE`` (1e-13) or Rung-1 (1e-11).
``navier_stokes_proof_claim`` stays False.

Known stretch blocker (audit): spectral/PV Hilbert alone err at O(1e-1); with
high ``proj_defect_weight`` the neural Ω is pulled into a Hardy span that itself
floors near ~1e-1 under MG. Stretch remains unearned until Hilbert/dictionary
capacity improves — more MG alone does not clear 1e-13.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT))

from _gates import (  # noqa: E402
    CCF_LAMBDA_1ST_UNSTABLE,
    CCF_RESIDUAL_GATE_1ST_UNSTABLE,
    CCF_STRETCH_RESIDUAL_GATE,
    ccf_lambda_digits_gate,
    ccf_residual_gate,
)

STRETCH = float(CCF_STRETCH_RESIDUAL_GATE)
RUNG1 = float(CCF_RESIDUAL_GATE_1ST_UNSTABLE)
LAM = float(CCF_LAMBDA_1ST_UNSTABLE)


def _scratch() -> Path:
    out = Path(os.environ.get("OMNIBIAS_SCRATCH", ROOT / "artifacts")) / "deepmind_campaign"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _next_actions(residual: float, *, hilbert: str, hidden: int, mg_steps: int) -> list[str]:
    actions = [
        "never_weaken_1e-13_stretch",
        "never_forge_navier_stokes",
    ]
    if residual > 1.0:
        actions.extend(["check_gauge_nontrivial", "cold_start_or_explicit_adam_warmup"])
    if residual > 1e-2:
        actions.extend(
            [
                "widen_hardy_dictionary_or_hilbert_fidelity",
                "widen_network_hidden",
                "increase_mg_steps",
                "more_collocation",
                "run_multistage_rounds",
            ]
        )
    if residual > 1e-6 and hilbert == "truncated_line_spectral":
        actions.append("switch_train_hilbert_hardy_corrected_pv")
    if residual > 1e-6 and hilbert in ("pv_line", "truncated_line_spectral"):
        actions.append("increase_hardy_dictionary_for_corrected_hilbert")
    if residual > 1e-2 and hilbert == "hardy_corrected_pv":
        actions.append("hilbert_dictionary_catch22_enrich_dict_or_free_omega_hilbert")
    if residual > STRETCH:
        actions.append(f"orders_to_stretch={math.log10(max(residual, 1e-300) / STRETCH):.2f}")
    actions.append(f"current_hidden={hidden}")
    actions.append(f"current_mg_steps={mg_steps}")
    return actions


def run_once(
    *,
    smoke: bool = True,
    hidden: int | None = None,
    n_grid: int | None = None,
    mg_steps: int | None = None,
    qr_gn_steps: int | None = None,
    adam_warmup_steps: int | None = None,
    depth: int = 1,
    seed: int = 0,
    train_hilbert: str = "hardy_corrected_pv",
    mg_solver: str = "qr",
    multistage_rounds: int = 0,
    y_max: float = 40.0,
    dense_n_val: int | None = None,
    warm_state_path: str | None = None,
    save_state_path: str | None = None,
    use_grad_norm: bool | None = None,
    exp_core: bool | None = None,
    origin_fraction: float | None = None,
    n_fourier: int | None = None,
    fourier_scale: float | None = None,
) -> dict[str, Any]:
    """One reproduction discovery (+ optional multistage)."""
    import torch

    torch.set_default_dtype(torch.float64)
    from omnibias.pinn.torch.discovery import ccf_vorticity_neural as cvn
    from omnibias.pinn.torch.discovery import multistage as ms

    if smoke:
        hidden = 8 if hidden is None else hidden
        n_grid = 33 if n_grid is None else n_grid
        mg_steps = 6 if mg_steps is None else mg_steps
        qr_gn_steps = 2 if qr_gn_steps is None else qr_gn_steps
        # Tiny smoke may use a few Adam steps; stretch gates stay unearned.
        adam_warmup_steps = 5 if adam_warmup_steps is None else adam_warmup_steps
        dense_n_val = 201 if dense_n_val is None else dense_n_val
        mg_solver = "qr"
        y_max = min(float(y_max), 8.0)
    else:
        hidden = 64 if hidden is None else hidden
        n_grid = 257 if n_grid is None else n_grid
        mg_steps = 200 if mg_steps is None else mg_steps
        qr_gn_steps = 40 if qr_gn_steps is None else qr_gn_steps
        # Full path: Adam only when explicitly requested; cold start uses
        # reproduce_deepmind_config default (50) via None → config factory.
        # Escalate passes adam_warmup_steps=0.
        if adam_warmup_steps is None:
            adam_warmup_steps = 50
        dense_n_val = 4001 if dense_n_val is None else dense_n_val

    warm_sd = None
    if warm_state_path:
        wp = Path(warm_state_path)
        if wp.is_file():
            warm_sd = torch.load(wp, map_location="cpu", weights_only=True)

    t0 = time.perf_counter()
    cfg_kw: dict[str, Any] = dict(
        hidden=int(hidden),
        depth=int(depth),
        n_grid=int(n_grid),
        n_adaptive=int(n_grid),
        mg_steps=int(mg_steps),
        qr_gn_steps=int(qr_gn_steps),
        adam_warmup_steps=int(adam_warmup_steps),
        seed=int(seed),
        train_hilbert=train_hilbert,  # type: ignore[arg-type]
        mg_solver=mg_solver,  # type: ignore[arg-type]
        y_max=float(y_max),
        dense_n_val=int(dense_n_val),
        n_scales=4 if smoke else 8,
        n_gamma_multiples=2 if smoke else 4,
        d2_weight=0.0 if smoke else 0.01,
        resample_every=max(2, int(mg_steps) // 4),
    )
    if use_grad_norm is not None:
        cfg_kw["use_grad_norm"] = bool(use_grad_norm)
    if exp_core is not None:
        cfg_kw["exp_core"] = bool(exp_core)
    if origin_fraction is not None:
        cfg_kw["origin_fraction"] = float(origin_fraction)
    if n_fourier is not None:
        cfg_kw["n_fourier"] = int(n_fourier)
    if fourier_scale is not None:
        cfg_kw["fourier_scale"] = float(fourier_scale)
    cfg = cvn.reproduce_deepmind_config(**cfg_kw)
    disc = cvn.run_ccf_vorticity_neural_discovery(cfg, warm_state_dict=warm_sd)
    if save_state_path and "net" in disc.extra:
        Path(save_state_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {k: v.detach().cpu() for k, v in disc.extra["net"].state_dict().items()},
            save_state_path,
        )
    residual = float(disc.diagnostics["reproduction_dense_max_abs_for_gate"])
    omega = np.asarray(disc.omega, dtype=float)
    y = np.asarray(disc.y, dtype=float)
    omy = np.asarray(disc.omega_y, dtype=float)
    ms_info: dict[str, Any] = {"skipped": True}

    if multistage_rounds > 0 and residual < 1.0:
        scales = torch.as_tensor(disc.scales, dtype=torch.float64)
        gammas = torch.as_tensor(disc.gammas, dtype=torch.float64)

        def _residual_fn(phi: np.ndarray) -> np.ndarray:
            yt = torch.as_tensor(y, dtype=torch.float64)
            om = torch.as_tensor(phi, dtype=torch.float64)
            omy_t = torch.gradient(om, spacing=(yt,))[0]
            r, _, _, _ = cvn.vorticity_fields(
                yt,
                om,
                omy_t,
                lam=float(cfg.lam),
                scales=scales,
                gammas=gammas,
                train_hilbert=cfg.train_hilbert,
                hilbert_n_uniform=cfg.hilbert_n_uniform,
            )
            return r.detach().cpu().numpy()

        ms_out = ms.iterate_multistage(
            y,
            omega,
            _residual_fn,
            rounds=int(multistage_rounds),
            cfg=ms.MultiStageConfig(
                steps=20 if smoke else 80,
                hidden=16 if smoke else 32,
                n_fourier=8 if smoke else 16,
                linearized=True,
            ),
            omega_y0=omy,
            # Corr-matching GN proxy (honest label); Adam stage-2 has regressed basins.
            optimizer="gauss_newton",
        )
        omega = np.asarray(ms_out["composed"], dtype=float)
        omy = np.gradient(omega, y)
        # Re-score dense residual via interpolation onto linspace (matched Hilbert).
        y_d = np.linspace(-float(cfg.y_max), float(cfg.y_max), int(cfg.dense_n_val))
        om_d = np.interp(y_d, y, omega)
        # Anti-ghost on pre-rescale profile; Wang score after gauge rescale.
        g_raw = float(np.interp(cfg.gauge_point, y_d, om_d))
        omega_max_raw = float(np.max(np.abs(om_d)))
        if abs(g_raw) > 1e-14:
            om_d = om_d * (float(cfg.gauge_value) / g_raw)
        yt = torch.as_tensor(y_d, dtype=torch.float64)
        om_t = torch.as_tensor(om_d, dtype=torch.float64)
        omy_t = torch.gradient(om_t, spacing=(yt,))[0]
        r_d, defect_t, _, _ = cvn.vorticity_fields(
            yt,
            om_t,
            omy_t,
            lam=float(cfg.lam),
            scales=scales,
            gammas=gammas,
            train_hilbert=cfg.train_hilbert,
            hilbert_n_uniform=cfg.hilbert_n_uniform,
        )
        dense_max = float(torch.max(torch.abs(r_d)).item())
        defect_f = float(defect_t.item())
        score = (
            max(dense_max, defect_f)
            if cfg.train_hilbert in ("hardy_projection", "hardy_corrected_pv")
            else dense_max
        )
        residual = cvn._anti_ghost_residual(
            score,
            omega_gauge_sample=g_raw,
            omega_max_abs=omega_max_raw,
            gauge_value=cfg.gauge_value,
        )
        ms_info = {
            "skipped": False,
            "rounds_run": ms_out["rounds_run"],
            "optimizer": ms_out["optimizer"],
            "max_abs_after": float(ms_out["max_abs_residual_after"]),
            "reproduction_dense_after": residual,
            "projection_defect_after": defect_f,
        }

    lam_gate = ccf_lambda_digits_gate(disc.lam, LAM, name="ccf_lambda_reproduce")
    stretch_gate = ccf_residual_gate(
        residual, STRETCH, name="ccf_stretch_1e-13_reproduce"
    )
    rung1_report = ccf_residual_gate(residual, RUNG1, name="ccf_rung1_1e-11_report")
    wall = time.perf_counter() - t0
    # Drop non-serializable net handle
    extra = {k: v for k, v in disc.extra.items() if k != "net"}
    payload = {
        "benchmark": "reproduce_deepmind_ccf",
        "tier": "cpu_smoke" if smoke else "full",
        "wall_seconds": wall,
        "config": {
            "arm": "reproduce",
            "lam": float(cfg.lam),
            "hidden": int(hidden),
            "depth": int(depth),
            "n_grid": int(n_grid),
            "mg_steps": int(mg_steps),
            "qr_gn_steps": int(qr_gn_steps),
            "adam_warmup_steps": int(adam_warmup_steps),
            "train_hilbert": cfg.train_hilbert,
            "mg_solver": cfg.mg_solver,
            "optimizer": extra.get("optimizer"),
            "multistage_rounds": int(multistage_rounds),
            "y_max": float(cfg.y_max),
            "dense_n_val": int(cfg.dense_n_val),
            "seed": int(seed),
        },
        "metrics": {
            "reproduction_dense_max_abs_for_gate": residual,
            "reproduction_dense_max_abs": float(
                disc.diagnostics.get("reproduction_dense_max_abs", residual)
            ),
            "omega_gauge_sample": float(disc.diagnostics["omega_gauge_sample"]),
            "omega_max_abs": float(disc.diagnostics["omega_max_abs"]),
            "collocation_max_abs": float(disc.diagnostics["max_abs_vorticity_residual"]),
            "orders_to_stretch": float(math.log10(max(residual, 1e-300) / STRETCH)),
        },
        "multistage": ms_info,
        "gates": {
            "lambda_ok": bool(lam_gate["passed"]),
            "stretch_1e-13_cleared": bool(stretch_gate["passed"]),
            "rung1_1e-11_report": bool(rung1_report["passed"]),
            "entries": [lam_gate, stretch_gate, rung1_report],
        },
        "diagnosis": {
            "residual": residual,
            "next_actions": _next_actions(
                residual,
                hilbert=str(cfg.train_hilbert),
                hidden=int(hidden),
                mg_steps=int(mg_steps),
            ),
        },
        "honesty": {
            "navier_stokes_proof_claim": False,
            "continuum_claim": False,
            "arm": "reproduce",
            "hardy_cap_deferred_until_stretch": not bool(stretch_gate["passed"]),
            "metric": "wang_vorticity_dense_neural_matched_hilbert",
        },
    }
    return payload


def escalate_loop(
    *,
    max_rounds: int = 8,
    smoke: bool = False,
) -> dict[str, Any]:
    """Escalate budgets until stretch clears.

    Fixed architecture + warm-start across rounds (continuous Martens–Grosse).
    Never weakens ``1e-13``. No early plateau exit while stretch is open.
    """
    # Canonical warm lineage: warm_net_reproduce.pt is the best known basin;
    # warm_net_ab.pt is the escalate working copy. Refresh ab from reproduce when
    # ab is missing OR when reproduce meta reports a strictly better floor.
    warm_src = _scratch() / "warm_net_reproduce.pt"
    warm_path = str(_scratch() / "warm_net_ab.pt")
    reproduce_meta = _scratch() / "warm_best_residual.json"
    ab_meta = _scratch() / "warm_best_ab.json"

    def _meta_residual(path: Path) -> float:
        if not path.is_file():
            return float("inf")
        try:
            return float(json.loads(path.read_text(encoding="utf-8")).get("residual", "inf"))
        except Exception:
            return float("inf")

    if warm_src.is_file():
        import shutil

        src_floor = _meta_residual(reproduce_meta)
        ab_floor = _meta_residual(ab_meta)
        if (not Path(warm_path).is_file()) or (src_floor < ab_floor):
            shutil.copy2(warm_src, warm_path)
            if src_floor < ab_floor and reproduce_meta.is_file():
                ab_meta.write_text(reproduce_meta.read_text(encoding="utf-8"), encoding="utf-8")
                print(
                    f"[escalate] refreshed ab warm from reproduce "
                    f"(floor {src_floor:.6e} < ab {ab_floor:.6e})",
                    flush=True,
                )
    schedule: list[dict[str, Any]] = [
        {
            "hidden": 48,
            "n_grid": 97,
            "mg_steps": 300,
            "adam_warmup_steps": 0,
            "multistage_rounds": 0,
            "train_hilbert": "hardy_corrected_pv",
            "depth": 2,
            "mg_solver": "qr",
            "dense_n_val": 1601,
            "y_max": 60.0,
            "use_grad_norm": False,
            "exp_core": True,
        },
        {
            "hidden": 48,
            "n_grid": 97,
            "mg_steps": 300,
            "adam_warmup_steps": 0,
            "multistage_rounds": 0,
            "train_hilbert": "hardy_corrected_pv",
            "depth": 2,
            "mg_solver": "qr",
            "dense_n_val": 1601,
            "y_max": 60.0,
            "use_grad_norm": True,
            "exp_core": False,
        },
        {
            "hidden": 48,
            "n_grid": 97,
            "mg_steps": 300,
            "adam_warmup_steps": 0,
            "multistage_rounds": 0,
            "train_hilbert": "hardy_corrected_pv",
            "depth": 2,
            "mg_solver": "qr",
            "dense_n_val": 1601,
            "y_max": 60.0,
            "use_grad_norm": False,
            "exp_core": False,
        },
        {
            "hidden": 48,
            "n_grid": 129,
            "mg_steps": 600,
            "adam_warmup_steps": 0,
            "multistage_rounds": 0,
            "train_hilbert": "hardy_corrected_pv",
            "depth": 2,
            "mg_solver": "qr",
            "dense_n_val": 2001,
            "y_max": 60.0,
            "use_grad_norm": False,
            "exp_core": True,
        },
    ]
    if smoke:
        schedule = [
            {
                "hidden": 8,
                "n_grid": 33,
                "mg_steps": 4,
                "adam_warmup_steps": 2,
                "multistage_rounds": 0,
                "train_hilbert": "hardy_projection",
                "depth": 1,
                "mg_solver": "qr",
                "dense_n_val": 201,
            }
        ]
        max_rounds = 1
        warm_path = str(_scratch() / "warm_net_smoke.pt")

    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_meta_path = _scratch() / "warm_best_ab.json"
    best_floor = float("inf")
    if best_meta_path.is_file():
        try:
            meta = json.loads(best_meta_path.read_text(encoding="utf-8"))
            best_floor = float(meta.get("residual", "inf"))
            print(
                f"[escalate] prior best residual floor={best_floor:.6e} "
                f"(hilbert={meta.get('train_hilbert')})",
                flush=True,
            )
        except Exception:
            best_floor = float("inf")
    for i, sched in enumerate(schedule[: max(1, int(max_rounds))]):
        print(
            f"[escalate] round {i}: hidden={sched.get('hidden')} "
            f"depth={sched.get('depth')} grid={sched.get('n_grid')} "
            f"mg={sched.get('mg_steps')} "
            f"hilbert={sched.get('train_hilbert')} warm={Path(warm_path).is_file()}",
            flush=True,
        )
        cand_path = str(_scratch() / "warm_net_candidate.pt")
        out = run_once(
            smoke=smoke,
            seed=i,
            warm_state_path=warm_path if Path(warm_path).is_file() else None,
            save_state_path=cand_path,
            **sched,
        )
        r = float(out["metrics"]["reproduction_dense_max_abs_for_gate"])
        print(
            f"[escalate] residual={r:.6e} stretch_cleared={out['gates']['stretch_1e-13_cleared']}",
            flush=True,
        )
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        history.append(
            {
                "round": i,
                "residual": r,
                "config": out["config"],
                "stretch_cleared": out["gates"]["stretch_1e-13_cleared"],
            }
        )
        improved_run = best is None or r < float(
            best["metrics"]["reproduction_dense_max_abs_for_gate"]
        )
        # Persist floor across restarts so a pivot cannot clobber a better basin.
        improved_global = r < best_floor
        if improved_run:
            best = out
        if improved_global and Path(cand_path).is_file():
            import shutil

            shutil.copy2(cand_path, warm_path)
            # Keep reproduce lineage in sync with ab when escalate improves.
            shutil.copy2(cand_path, _scratch() / "warm_net_reproduce.pt")
            best_floor = r
            best = out  # summary best_residual must match persisted floor
            meta_payload = (
                json.dumps(
                    {
                        "residual": r,
                        "train_hilbert": sched.get("train_hilbert"),
                        "round": i,
                    },
                    indent=2,
                )
                + "\n"
            )
            best_meta_path.write_text(meta_payload, encoding="utf-8")
            (_scratch() / "warm_best_residual.json").write_text(
                meta_payload, encoding="utf-8"
            )
            print(f"[escalate] warm checkpoint updated -> {warm_path}", flush=True)
        elif not improved_global:
            print(
                "[escalate] residual worsened vs persisted best; keeping prior warm",
                flush=True,
            )
        if out["gates"]["stretch_1e-13_cleared"]:
            break

    assert best is not None
    # Prefer the persisted global floor when this run never beat it (restart case).
    session_best = float(best["metrics"]["reproduction_dense_max_abs_for_gate"])
    reported_best = min(session_best, best_floor) if math.isfinite(best_floor) else session_best
    summary = {
        "benchmark": "reproduce_deepmind_ccf_escalate",
        "stretch_gate": STRETCH,
        "rung1_gate_report": RUNG1,
        "best_residual": reported_best,
        "session_best_residual": session_best,
        "persisted_best_residual": best_floor if math.isfinite(best_floor) else None,
        "stretch_1e-13_cleared": bool(reported_best <= STRETCH),
        "orders_to_stretch": float(
            math.log10(max(reported_best, 1e-300) / STRETCH)
        ),
        "rounds": history,
        "best": best,
        "warm_state_path": warm_path,
        "honesty": {
            "navier_stokes_proof_claim": False,
            "plateau_without_weaken": True,
            "never_stop_until_1e-13_or_budget": True,
            "stretch_unearned": reported_best > STRETCH,
            "known_floor_note": "hilbert_dictionary_catch22_near_1e-1",
        },
    }
    return summary



def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true", help="Full / escalate budget")
    p.add_argument("--escalate", action="store_true", help="Run escalate_loop")
    p.add_argument("--max-rounds", type=int, default=8)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--n-grid", type=int, default=None)
    p.add_argument("--mg-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--train-hilbert",
        default="hardy_corrected_pv",
        choices=[
            "hardy_corrected_pv",
            "hardy_projection",
            "truncated_line_spectral",
            "pv_line",
        ],
    )
    p.add_argument("--multistage-rounds", type=int, default=0)
    p.add_argument(
        "--write-docs",
        action="store_true",
        help="Also write docs/benchmarks/reproduce_deepmind_ccf_smoke.json",
    )
    args = p.parse_args(argv)

    if args.escalate:
        payload = escalate_loop(max_rounds=args.max_rounds, smoke=not args.full)
        name = "reproduce_ccf_escalate.json"
    else:
        payload = run_once(
            smoke=not args.full,
            hidden=args.hidden,
            n_grid=args.n_grid,
            mg_steps=args.mg_steps,
            seed=args.seed,
            train_hilbert=args.train_hilbert,
            multistage_rounds=args.multistage_rounds,
        )
        name = f"reproduce_ccf_{'full' if args.full else 'smoke'}.json"

    out = _scratch() / name
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    if args.write_docs and not args.full and not args.escalate:
        docs = ROOT / "docs" / "benchmarks" / "reproduce_deepmind_ccf_smoke.json"
        docs.parent.mkdir(parents=True, exist_ok=True)
        # Infrastructure-only doc artifact (gates may be unearned).
        docs.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")

    cleared = bool(
        payload.get("gates", {}).get("stretch_1e-13_cleared")
        or payload.get("stretch_1e-13_cleared")
    )
    print(
        json.dumps(
            {
                "artifact": str(out),
                "residual": payload.get("metrics", payload).get(
                    "reproduction_dense_max_abs_for_gate",
                    payload.get("best_residual"),
                ),
                "stretch_1e-13_cleared": cleared,
            },
            indent=2,
        )
    )
    # Smoke exit 0 on infrastructure success; stretch may remain unearned.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
