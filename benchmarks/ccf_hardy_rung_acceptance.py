# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""CCF Hardy Rung-1 / Rung-2 acceptance ladder (CPU smoke / GPU full).

Stage order (DeepMind residual push):

1. Torch compactified neural vorticity PINN (CubicGaussNewton + QR)
2. Multistage residual correction
3. Hardy projection + vorticity mpmath polish (true dictionary; no multi-α collapse)
4. Dense Wang vorticity residual on fixed ``linspace(-40,40,4001)``
5. Anti-ghost (gauge + max|Ω| floors)
6. ``ccf_absolute_gates`` (unchanged thresholds)
7. Vorticity whole-line CAP → ``whole_line_certified`` only if truly closed
8. Stretch metric vs ``1e-13`` (report only; never forges Rung-1)

Writes acceptance JSON under ``docs/benchmarks/`` only when absolute gates are
earned. Scratch status always lands under ``$OMNIBIAS_SCRATCH``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from _gates import (  # noqa: E402
    CCF_LAMBDA_1ST_UNSTABLE,
    CCF_LAMBDA_2ND_UNSTABLE,
    ccf_absolute_gates,
)

from omnibias.pinn.certified.ccf_hardy import (  # noqa: E402
    certified_ccf_hardy_wholeline_blowup_attempt,
    certified_ccf_hardy_wholeline_blowup_attempt_schema_errors,
    refine_ccf_hardy_profile,
)
from omnibias.pinn.certified.navier_stokes import (  # noqa: E402
    default_ccf_collocation_nodes,
)
from omnibias.pinn.jax.discovery import ccf_vorticity, polish_mp  # noqa: E402
from omnibias.pinn.jax.discovery.funnel import (  # noqa: E402
    FunnelState,
    funnel_next_lambda,
    signed_max_residual_near_origin,
)
from omnibias.pinn.jax.equations.ccf_compactified import alpha_from_lambda  # noqa: E402
from omnibias.pinn.torch.discovery import ccf_vorticity_neural as cvn  # noqa: E402
from omnibias.pinn.torch.discovery import multistage as ms  # noqa: E402


def _family_target(family: str) -> float:
    if family == "1st_unstable":
        return CCF_LAMBDA_1ST_UNSTABLE
    if family == "2nd_unstable":
        return CCF_LAMBDA_2ND_UNSTABLE
    raise ValueError(family)


def _collocation_nodes(n_terms: int) -> list[float]:
    n_free = (n_terms - 1) + n_terms  # free coeffs + log-scales; lam frozen
    return [0.12 * (1.48**k) for k in range(n_free + 8)]


def run_acceptance(
    *,
    family: str = "1st_unstable",
    smoke: bool = True,
    n_scales: int | None = None,
    n_gamma_multiples: int | None = None,
    n_grid: int | None = None,
    y_max: float = 40.0,
    cubic_gn_steps: int | None = None,
    qr_gn_steps: int | None = None,
    multistage_steps: int | None = None,
    polish_dps: int = 40,
    polish_iters: int = 25,
    seed: int = 0,
    funnel_updates: int = 0,
    hidden: int | None = None,
) -> dict:
    target = _family_target(family)
    # CPU smoke vs full budgets (GPU / submit path uses --full).
    if smoke:
        n_scales = 4 if n_scales is None else n_scales
        n_gamma_multiples = 2 if n_gamma_multiples is None else n_gamma_multiples
        n_grid = 129 if n_grid is None else n_grid
        cubic_gn_steps = 8 if cubic_gn_steps is None else cubic_gn_steps
        qr_gn_steps = 4 if qr_gn_steps is None else qr_gn_steps
        multistage_steps = 20 if multistage_steps is None else multistage_steps
        hidden = 16 if hidden is None else hidden
        polish_iters = min(polish_iters, 8)
        polish_dps = min(polish_dps, 25)
    else:
        # Serious full budget (not toy); wall-time recorded in metrics.
        n_scales = 12 if n_scales is None else n_scales
        n_gamma_multiples = 4 if n_gamma_multiples is None else n_gamma_multiples
        n_grid = 401 if n_grid is None else n_grid
        cubic_gn_steps = 2000 if cubic_gn_steps is None else cubic_gn_steps
        qr_gn_steps = 200 if qr_gn_steps is None else qr_gn_steps
        multistage_steps = 500 if multistage_steps is None else multistage_steps
        hidden = 64 if hidden is None else hidden

    t0 = time.perf_counter()
    lam = float(target)
    funnel: FunnelState | None = None
    if funnel_updates > 0:
        funnel = FunnelState()

    # Stage A — JAX Hardy-Ω Martens–Grosse GN (earn-path primary; no Adam).
    gn_steps = int(cubic_gn_steps) + int(qr_gn_steps)
    jax_cfg = ccf_vorticity.CCFVorticityDiscoveryConfig(
        n_scales=int(n_scales),
        n_gamma_multiples=int(n_gamma_multiples),
        n_grid=int(n_grid),
        y_max=float(y_max),
        lam=float(lam),
        seed=int(seed),
        gn_steps=max(gn_steps, 8),
    )
    jax_disc = ccf_vorticity.run_ccf_vorticity_discovery(jax_cfg)

    # Stage B — optional neural CubicGN warm-start with Hardy train Hilbert.
    def _discover_neural(lam_fixed: float) -> cvn.CCFVorticityNeuralResult:
        cfg = cvn.CCFVorticityNeuralConfig(
            lam=float(lam_fixed),
            n_grid=int(n_grid),
            y_max=float(y_max),
            hidden=int(hidden),
            n_scales=int(n_scales),
            n_gamma_multiples=int(n_gamma_multiples),
            seed=int(seed),
            cubic_gn_steps=int(cubic_gn_steps),
            qr_gn_steps=int(qr_gn_steps),
            adam_warmup_steps=0,
            use_grad_norm=True,
            exp_core=True,
            d1_weight=0.1,
            d2_weight=0.01 if not smoke else 0.0,
            resample_every=max(5, int(cubic_gn_steps) // 8) if cubic_gn_steps else 10,
            train_hilbert="hardy_projection",
        )
        return cvn.run_ccf_vorticity_neural_discovery(cfg)

    disc = _discover_neural(lam)
    if funnel is not None:
        funnel.record(
            lam,
            signed_max_residual_near_origin(disc.y, disc.residual),
        )
        for _ in range(int(funnel_updates)):
            lam = funnel_next_lambda(funnel)
            disc = _discover_neural(lam)
            funnel.record(
                lam,
                signed_max_residual_near_origin(disc.y, disc.residual),
            )
        lam = float(target)
        disc = _discover_neural(lam)
        jax_cfg = ccf_vorticity.CCFVorticityDiscoveryConfig(
            n_scales=int(n_scales),
            n_gamma_multiples=int(n_gamma_multiples),
            n_grid=int(n_grid),
            y_max=float(y_max),
            lam=float(lam),
            seed=int(seed),
            gn_steps=max(gn_steps, 8),
        )
        jax_disc = ccf_vorticity.run_ccf_vorticity_discovery(jax_cfg)

    # Pick the better of JAX Martens–Grosse vs neural Hardy CubicGN by dense residual.
    jax_dense = float(jax_disc.diagnostics["dense_max_abs_vorticity"])
    neural_dense = float(
        ccf_vorticity.dense_vorticity_residual(
            disc.coeffs, disc.scales, disc.gammas, lam, n_val=801, y_max=y_max
        )["dense_max_abs_vorticity"]
    )
    if jax_dense <= neural_dense:
        coeffs = np.asarray(jax_disc.coeffs, dtype=float)
        scales = np.asarray(jax_disc.scales, dtype=float)
        gammas = np.asarray(jax_disc.alphas, dtype=float)
        y_train = np.asarray(jax_disc.y, dtype=float)
        omega0 = np.asarray(jax_disc.omega, dtype=float)
        omega_y0 = np.gradient(omega0, y_train)
        disc_resid = float(jax_disc.diagnostics["max_abs_vorticity_residual"])
        discovery_source = "jax_martens_grosse"
        proj_defect0 = 0.0
    else:
        coeffs = np.asarray(disc.coeffs, dtype=float)
        scales = np.asarray(disc.scales, dtype=float)
        gammas = np.asarray(disc.gammas, dtype=float)
        y_train = np.asarray(disc.y, dtype=float)
        omega0 = np.asarray(disc.omega, dtype=float)
        omega_y0 = np.asarray(disc.omega_y, dtype=float)
        disc_resid = float(disc.diagnostics["max_abs_vorticity_residual"])
        discovery_source = "torch_cubic_gn_hardy"
        proj_defect0 = float(disc.diagnostics.get("projection_defect", 0.0))

    # Multistage on Hardy-projected Wang residual (train metric = Rung metric).
    sc_t = torch.as_tensor(np.array(scales, dtype=float, copy=True), dtype=torch.float64)
    ga_t = torch.as_tensor(np.array(gammas, dtype=float, copy=True), dtype=torch.float64)

    def _omega_residual(omega_samples: np.ndarray) -> np.ndarray:
        om = np.asarray(omega_samples, dtype=float)
        yt = torch.as_tensor(y_train, dtype=torch.float64)
        ot = torch.as_tensor(om, dtype=torch.float64)
        omy_t = torch.as_tensor(np.gradient(om, y_train), dtype=torch.float64)
        with torch.no_grad():
            r, _, _, _ = cvn.vorticity_fields(
                yt,
                ot,
                omy_t,
                lam=lam,
                scales=sc_t,
                gammas=ga_t,
                train_hilbert="hardy_projection",
                hilbert_n_uniform=None,
            )
        return r.detach().cpu().numpy()

    ms_out = ms.correct_profile(
        y_train,
        omega0,
        _omega_residual,
        cfg=ms.MultiStageConfig(
            steps=int(multistage_steps),
            hidden=24,
            n_fourier=12,
            linearized=True,
        ),
        omega_y0=omega_y0,
    )
    composed = np.asarray(ms_out["composed"], dtype=float)
    g_c = float(np.interp(0.5, y_train, composed))
    if abs(g_c) > 1e-14:
        composed = composed * (0.05 / g_c)
    coeffs_ms, proj_defect, _ = cvn.project_omega_hardy(
        y_train, composed, scales=scales, gammas=gammas
    )
    coeffs_ms = np.asarray(coeffs_ms, dtype=float)
    disc_c = np.asarray(coeffs, dtype=float)
    if (
        float(np.max(np.abs(coeffs_ms))) > 1e3
        or float(ms_out["max_abs_residual_after"])
        >= float(ms_out["max_abs_residual_before"]) * 0.98
    ):
        coeffs = disc_c
        proj_defect = float(proj_defect0)
    else:
        coeffs = coeffs_ms
        # Re-evaluate which source won after multistage.
        discovery_source = discovery_source + "+hardy_multistage"
    # Only refine / polish when discovery is already in the fine basin.
    # Wang residual is not homogeneous under amplitude rescaling, so refine on a
    # coarse projection tends to explode coeffs — skip rather than forge.
    free_scales = disc_resid < 0.05
    polish_ok = disc_resid < 0.01
    refine_ok = disc_resid < 0.05
    nodes = (
        _collocation_nodes(int(scales.size))
        if free_scales
        else list(default_ccf_collocation_nodes(int(scales.size)))
    )

    refined: dict = {
        "coeffs": coeffs.tolist(),
        "scales": scales.tolist(),
        "gammas": gammas.tolist(),
        "lam": lam,
        "residual_max_abs": float("nan"),
    }
    if refine_ok:
        refined = refine_ccf_hardy_profile(
            coeffs=coeffs.tolist(),
            scales=scales.tolist(),
            lam=lam,
            nodes=nodes,
            form="vorticity",
            gammas=gammas.tolist(),
            free_scales=free_scales,
            free_lam=False,
            lam_target=target,
            iters=60 if smoke else 120,
            tol=1e-14,
            velocity_sign=-1.0,
            omega_gauge_point=0.5,
            omega_gauge_value=0.05,
            min_scale=0.5,
            max_scale=8.0,
        )
        coeffs = np.asarray(refined["coeffs"], dtype=float)
        scales = np.asarray(refined["scales"], dtype=float)
        gammas = np.asarray(refined.get("gammas", gammas), dtype=float)
        lam = float(refined["lam"])

    polished: dict = {
        "coeffs": coeffs,
        "scales": scales,
        "lam": lam,
        "gammas": gammas,
        "max_abs_residual_mpmath": float("nan"),
    }
    if polish_ok:
        polished = polish_mp.polish_hardy_ccf(
            coeffs=coeffs,
            scales=scales,
            lam=lam,
            nodes=np.asarray(nodes, dtype=float),
            form="vorticity",
            gammas=gammas,
            dps=polish_dps,
            max_iter=polish_iters,
            free_lam=False,
            velocity_sign=-1.0,
        )
        coeffs = np.asarray(polished["coeffs"], dtype=float)
        scales = np.asarray(polished["scales"], dtype=float)
        lam = float(polished["lam"])
        if "gammas" in polished:
            gammas = np.asarray(polished["gammas"], dtype=float)

    refined2: dict = {
        "coeffs": coeffs.tolist(),
        "scales": scales.tolist(),
        "gammas": gammas.tolist(),
        "lam": lam,
        "residual_max_abs": float(refined.get("residual_max_abs", float("nan"))),
    }
    if polish_ok:
        refined2 = refine_ccf_hardy_profile(
            coeffs=coeffs.tolist(),
            scales=scales.tolist(),
            lam=lam,
            nodes=None,
            form="vorticity",
            gammas=gammas.tolist(),
            free_scales=False,
            free_lam=False,
            lam_target=target,
            iters=40 if smoke else 80,
            tol=1e-14,
            velocity_sign=-1.0,
            omega_gauge_point=0.5,
            omega_gauge_value=0.05,
            min_scale=0.5,
            max_scale=8.0,
        )
        coeffs = np.asarray(refined2["coeffs"], dtype=float)
        scales = np.asarray(refined2["scales"], dtype=float)
        gammas = np.asarray(refined2.get("gammas", gammas), dtype=float)
        lam = float(refined2["lam"])

    dense = ccf_vorticity.dense_vorticity_residual(
        coeffs, scales, gammas, lam, n_val=4001, y_max=y_max
    )
    gauge_ok = abs(float(dense.get("omega_gauge_sample", 0.0)) - 0.05) <= 0.01
    nontrivial = float(dense.get("omega_max_abs", 0.0)) >= 0.02
    residual_for_gate = float(dense["dense_max_abs_vorticity"])
    if not (gauge_ok and nontrivial):
        residual_for_gate = max(residual_for_gate, 1.0)

    stretch_cleared = bool(residual_for_gate <= 1e-13 and gauge_ok and nontrivial)
    absolute = ccf_absolute_gates(
        lam=lam,
        max_abs_residual=residual_for_gate,
        family=family,
        stretch_mp_residual=float(polished.get("max_abs_residual_mpmath", float("nan"))),
    )

    cap_nodes = list(default_ccf_collocation_nodes(int(scales.size)))
    cert = certified_ccf_hardy_wholeline_blowup_attempt(
        coeffs=coeffs.tolist(),
        scales=scales.tolist(),
        lam=lam,
        nodes=cap_nodes,
        form="vorticity",
        gammas=gammas.tolist(),
        residual_gate=1e-11 if family == "1st_unstable" else 1e-6,
        velocity_sign=-1.0,
    )
    schema_errs = certified_ccf_hardy_wholeline_blowup_attempt_schema_errors(cert)
    wall = time.perf_counter() - t0

    return {
        "benchmark": "ccf_hardy_rung_acceptance",
        "family": family,
        "tier": "cpu_smoke" if smoke else "full",
        "config": {
            "n_scales": n_scales,
            "n_gamma_multiples": n_gamma_multiples,
            "n_grid": n_grid,
            "y_max": y_max,
            "cubic_gn_steps": cubic_gn_steps,
            "qr_gn_steps": qr_gn_steps,
            "multistage_steps": multistage_steps,
            "hidden": hidden,
            "target_lam": target,
            "seed": seed,
            "funnel_updates": funnel_updates,
            "hilbert": "hardy_projection_exact",
            "train_hilbert": "hardy_exact_omega",
            "rung_hilbert": "hardy_projection_exact",
            "residual_form": "wang_vorticity",
            "optimizer": "MartensGrosseGN+CubicGaussNewton+QR",
            "gn_solver": "qr",
            "martens_grosse": "exact_jvp",
            "discovery_source": discovery_source,
            "use_grad_norm": True,
            "linearized_msnn": True,
            "train_lam": False,
            "rung_metric_uses_fft": False,
            "multi_alpha_collapse": False,
        },
        "metrics": {
            "lam": lam,
            "collocation_residual_max_abs": float(refined2["residual_max_abs"]),
            "dense_max_abs_vorticity": dense["dense_max_abs_vorticity"],
            "dense_rms_vorticity": dense["dense_rms_vorticity"],
            "dense_max_abs_vorticity_for_gate": residual_for_gate,
            "stretch_1e-13_cleared": stretch_cleared,
            "projection_defect_after_multistage": float(proj_defect),
            "omega_gauge_sample": dense.get("omega_gauge_sample", float("nan")),
            "omega_max_abs": dense.get("omega_max_abs", float("nan")),
            "gauge_ok": bool(gauge_ok),
            "nontrivial_profile": bool(nontrivial),
            "mpmath_max_abs_residual": float(
                polished.get("max_abs_residual_mpmath", float("nan"))
            ),
            "wall_seconds": wall,
            "discovery_train_max_abs_vorticity": float(disc_resid),
            "jax_dense_max_abs_vorticity": float(jax_dense),
            "neural_dense_max_abs_vorticity": float(neural_dense),
            "discovery_source": discovery_source,
            "polish_ran": bool(polish_ok),
            "refine_ran": bool(refine_ok),
            "free_scales_in_refine": bool(free_scales),
            "multistage_max_abs_before": float(ms_out["max_abs_residual_before"]),
            "multistage_max_abs_after": float(ms_out["max_abs_residual_after"]),
            "multistage_sigma": float(ms_out.get("sigma", float("nan"))),
            "smoke_gate_1e-4_cleared": bool(
                residual_for_gate < 1e-4 and gauge_ok and nontrivial
            ),
        },
        "absolute_gates": absolute,
        "rung2": {
            "whole_line_certified": bool(cert["honesty"]["whole_line_certified"]),
            "closure_certified": bool(cert["closure_certified"]),
            "residual_certified_sup": cert["closure_report"]["residual_certified_sup"],
            "schema_ok": schema_errs == [],
            "quantified_gap": cert["closure_report"].get("quantified_gap"),
            "form": "vorticity",
        },
        "profile": {
            "coeffs": coeffs.tolist(),
            "scales": scales.tolist(),
            "gammas": gammas.tolist(),
            "lam": lam,
            "alpha": float(alpha_from_lambda(lam)),
        },
        "honesty": {
            "navier_stokes_proof_claim": False,
            "reproduces_published_lambda": absolute["honesty"][
                "reproduces_published_lambda"
            ],
            "rung1_earned": bool(absolute["earned"]),
            "rung2_earned": bool(cert["honesty"]["whole_line_certified"]),
            "stretch_1e-13_earned": stretch_cleared,
            "residual_metric": "wang_vorticity_dense_linspace",
            "hilbert_metric": "hardy_projection_exact",
            "train_hilbert": "hardy_projection",
            "measured_gap_note": (
                "Train Hilbert matches Rung/CAP (Hardy projection / exact H[Q]=-P). "
                "Earn path prefers JAX Martens–Grosse Hardy-Ω GN vs torch CubicGN; "
                "Adam is forbidden. Gates stay unearned until dense residual clears "
                "published thresholds with anti-ghost floors; stretch 1e-13 never "
                "forges Rung-1. Clay NS remains external."
            ),
        },
        "gates": {
            "rung1_earned": bool(absolute["earned"]),
            "rung2_earned": bool(cert["honesty"]["whole_line_certified"]),
            "stretch_1e-13_earned": stretch_cleared,
            "schema_ok": schema_errs == [],
            "navier_stokes_proof_claim": False,
            "passed": bool(
                absolute["earned"] and cert["honesty"]["whole_line_certified"]
            ),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--family", choices=("1st_unstable", "2nd_unstable"), default="1st_unstable"
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Full budget (prefer $OMNIBIAS_SUBMIT / GPU); default is CPU smoke",
    )
    p.add_argument("--n-scales", type=int, default=None)
    p.add_argument("--n-gamma-multiples", type=int, default=None)
    p.add_argument("--n-grid", type=int, default=None)
    p.add_argument("--y-max", type=float, default=40.0)
    p.add_argument("--cubic-gn-steps", type=int, default=None)
    p.add_argument("--qr-gn-steps", type=int, default=None)
    p.add_argument("--multistage-steps", type=int, default=None)
    p.add_argument("--funnel-updates", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument(
        "--no-submit",
        action="store_true",
        help="Force local run even if $OMNIBIAS_SUBMIT is set",
    )
    args = p.parse_args()

    submit = os.environ.get("OMNIBIAS_SUBMIT", "").strip()
    already = os.environ.get("OMNIBIAS_CCF_FULL_RUNNING", "").strip()
    if args.full and submit and not args.no_submit and not already:
        import shlex
        import subprocess

        cmd = [
            submit,
            sys.executable,
            str(Path(__file__).resolve()),
            "--full",
            "--family",
            args.family,
            "--no-submit",
            "--seed",
            str(args.seed),
        ]
        if args.n_scales is not None:
            cmd += ["--n-scales", str(args.n_scales)]
        if args.n_gamma_multiples is not None:
            cmd += ["--n-gamma-multiples", str(args.n_gamma_multiples)]
        if args.n_grid is not None:
            cmd += ["--n-grid", str(args.n_grid)]
        if args.cubic_gn_steps is not None:
            cmd += ["--cubic-gn-steps", str(args.cubic_gn_steps)]
        if args.qr_gn_steps is not None:
            cmd += ["--qr-gn-steps", str(args.qr_gn_steps)]
        if args.multistage_steps is not None:
            cmd += ["--multistage-steps", str(args.multistage_steps)]
        if args.out is not None:
            cmd += ["--out", str(args.out)]
        env = os.environ.copy()
        env["OMNIBIAS_CCF_FULL_RUNNING"] = "1"
        print("submit:", " ".join(shlex.quote(c) for c in cmd))
        raise SystemExit(subprocess.call(cmd, env=env))

    out = args.out or (
        ROOT / "docs" / "benchmarks" / f"ccf_line_rung1_{args.family}.json"
    )
    payload = run_acceptance(
        family=args.family,
        smoke=not args.full,
        n_scales=args.n_scales,
        n_gamma_multiples=args.n_gamma_multiples,
        n_grid=args.n_grid,
        y_max=args.y_max,
        cubic_gn_steps=args.cubic_gn_steps,
        qr_gn_steps=args.qr_gn_steps,
        multistage_steps=args.multistage_steps,
        funnel_updates=args.funnel_updates,
        seed=args.seed,
    )
    if payload["gates"]["rung1_earned"]:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    scratch = Path(os.environ.get("OMNIBIAS_SCRATCH", str(ROOT / "artifacts")))
    scratch.mkdir(parents=True, exist_ok=True)
    status_path = scratch / f"ccf_rung_status_{args.family}.json"
    status_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"status {status_path}")
    print(
        json.dumps(
            {
                "lam": payload["metrics"]["lam"],
                "dense_max_abs_vorticity": payload["metrics"][
                    "dense_max_abs_vorticity"
                ],
                "smoke_gate_1e-4_cleared": payload["metrics"]["smoke_gate_1e-4_cleared"],
                "stretch_1e-13_cleared": payload["metrics"]["stretch_1e-13_cleared"],
                "rung1_earned": payload["gates"]["rung1_earned"],
                "rung2_earned": payload["gates"]["rung2_earned"],
                "residual_certified_sup": payload["rung2"]["residual_certified_sup"],
                "wall_seconds": payload["metrics"]["wall_seconds"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if payload["gates"]["passed"] else 1)


if __name__ == "__main__":
    main()
