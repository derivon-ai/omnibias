# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Track C (honest): fractional / hyperdissipative Navier-Stokes.

The *honest* engagement with the global-regularity problem. Global regularity of the
**classical** 3D incompressible Navier-Stokes equation (``alpha = 1``) is open,
and this repository is structurally gated never to claim it: ``unproven_claim`` is
``False`` everywhere here, and nothing forges ``theorem_prover_verified``.

What is genuinely provable is the **hyperdissipative** generalisation

.. math::

    \partial_t u + (u\cdot\nabla)u + \nabla p = -\nu (-\Delta)^{\alpha} u + f,
    \qquad \nabla\cdot u = 0 ,

whose energy is critical at ``alpha_c = (n+2)/4 = 5/4`` in ``n = 3``:

- ``alpha >= 5/4``: **global regularity proven** (Lions 1969) -- *external*.
- Tao (2009): global regularity for a *logarithmically supercritical*
  dissipation just below ``5/4`` -- *external*.
- ``1 <= alpha < 5/4``: **open**; ``alpha = 1`` is the classical global-regularity problem.

Stages: (1) certify the model against exact solutions across ``alpha``;
(2) recover the fractional order from data with a learnable exponent;
(3-4) criticality ladder + conditional continuation criteria;
(5) theorem-readiness gates (all report *not closed*);
(6) Tao's logarithmically supercritical regime.

Usage::

    python -m examples.certified_fluid_dynamics.run_fractional_ns --smoke
    python -m examples.certified_fluid_dynamics.run_fractional_ns \
        --out-dir "artifacts/omnibias_runs/fractional_ns"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
from omnibias.fractional.torch.order import LearnableOrder  # noqa: E402
from omnibias.pinn.certified import (  # noqa: E402
    build_axisymmetric_interval_report,
    build_ns_solve_or_falsify_report,
    build_refined_axisymmetric_swirl_candidate_artifact,
    build_regularity_closure_report,
)

from examples.certified_fluid_dynamics.fractional_ns_theory import (  # noqa: E402
    CRITICAL_ALPHA_3D,
    LIONS_CITATION,
    TWO_PI,
    classify_log_supercritical,
    classify_regime,
    exact_decaying_abc,
    exact_decaying_shear,
    frac_laplacian_torch,
    fractional_ns_residual,
    spectral_grad,
    tao_log_supercritical_diagnostic,
)


def _json_default(o: object) -> object:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _emit(event: str, **payload: object) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True, default=_json_default), flush=True)


def _write_json(path: str, obj: object) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_json_default)


# --------------------------------------------------------------------------- #
# Stage 1: certify the fractional model across alpha.                         #
# --------------------------------------------------------------------------- #
def stage1_certify(n: int, alphas: list[float], nu: float, T: float, m: int) -> dict:
    rows: list[dict] = []
    for alpha in alphas:
        u0, p0, u0t, rate0 = exact_decaying_shear(n, 0.0, m=m, nu=nu, alpha=alpha)
        res, div = fractional_ns_residual(u0, p0, u0t, alpha=alpha, nu=nu)

        uT, _, _, _ = exact_decaying_shear(n, T, m=m, nu=nu, alpha=alpha)
        e0 = float(np.sum(u0 * u0))
        eT = float(np.sum(uT * uT))
        rate_measured = -math.log(eT / e0) / (2.0 * T)

        alpha_wrong = alpha + 0.25
        res_wrong, _ = fractional_ns_residual(u0, p0, u0t, alpha=alpha_wrong, nu=nu)

        ua, pa, uat = exact_decaying_abc(n, 0.0, nu=nu)
        res_abc, div_abc = fractional_ns_residual(ua, pa, uat, alpha=alpha, nu=nu)

        row = {
            "alpha": alpha,
            "shear_residual_sup": float(np.max(np.abs(res))),
            "shear_div_sup": float(np.max(np.abs(div))),
            "decay_rate_predicted": rate0,
            "decay_rate_measured": rate_measured,
            "decay_rate_abs_err": abs(rate_measured - rate0),
            "wrong_alpha_residual_sup": float(np.max(np.abs(res_wrong))),
            "abc_residual_sup": float(np.max(np.abs(res_abc))),
            "abc_div_sup": float(np.max(np.abs(div_abc))),
        }
        rows.append(row)
        _emit(
            "stage1_alpha",
            alpha=alpha,
            shear_residual_sup=row["shear_residual_sup"],
            decay_rate_predicted=row["decay_rate_predicted"],
            wrong_alpha_residual_sup=row["wrong_alpha_residual_sup"],
        )
    return {
        "grid_n": n,
        "viscosity": nu,
        "shear_wavenumber": m,
        "max_shear_residual_sup": float(max(r["shear_residual_sup"] for r in rows)),
        "max_abc_residual_sup": float(max(r["abc_residual_sup"] for r in rows)),
        "max_decay_rate_abs_err": float(max(r["decay_rate_abs_err"] for r in rows)),
        "min_wrong_alpha_residual_sup": float(min(r["wrong_alpha_residual_sup"] for r in rows)),
        "per_alpha": rows,
    }


# --------------------------------------------------------------------------- #
# Stage 2: recover the fractional order from data (learnable degree).         #
# --------------------------------------------------------------------------- #
def recover_order(alpha_true: float, *, n: int, steps: int, lr: float, seed: int) -> dict:
    torch.manual_seed(seed)
    x = torch.arange(n, dtype=torch.float64) * (TWO_PI / n)
    signal = sum(torch.sin((m + 1) * x) / (m + 1) for m in range(4))
    target = frac_laplacian_torch(signal, torch.tensor(alpha_true, dtype=torch.float64))
    order = LearnableOrder(init=0.6, lo=0.1, hi=2.0).double()
    opt = torch.optim.Adam(order.parameters(), lr=lr)
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = frac_laplacian_torch(signal, order())
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        opt.step()
    recovered = float(order().item())
    return {
        "alpha_true": alpha_true,
        "alpha_recovered": recovered,
        "abs_err": abs(recovered - alpha_true),
        "final_loss": float(loss.detach()),
    }


def stage2_learnable_order(n: int, steps: int, lr: float, seed: int) -> dict:
    rows = [recover_order(a, n=n, steps=steps, lr=lr, seed=seed) for a in (0.75, 1.0, 1.25, 1.5)]
    for r in rows:
        _emit("stage2_recover", **r)
    return {
        "method": "LearnableOrder + Adam on |k|^{2a} multiplier (differentiable order)",
        "max_abs_err": float(max(r["abs_err"] for r in rows)),
        "recoveries": rows,
    }


# --------------------------------------------------------------------------- #
# Stage 3+4: criticality ladder + conditional continuation criteria.          #
# --------------------------------------------------------------------------- #
def _bkm_and_energy(n: int, alpha: float, nu: float, T: float, m: int, steps: int) -> dict:
    ts = np.linspace(0.0, T, steps)
    bkm_integrand, energies = [], []
    for t in ts:
        u, _, _, _ = exact_decaying_shear(n, float(t), m=m, nu=nu, alpha=alpha)
        vort = np.stack(spectral_grad(u[0]))
        bkm_integrand.append(float(np.max(np.abs(vort))))
        energies.append(float(0.5 * np.sum(u * u)))
    trapz = getattr(np, "trapezoid", np.trapz)
    bkm = float(trapz(bkm_integrand, ts))
    return {
        "alpha": alpha,
        "bkm_vorticity_time_integral": bkm,
        "bkm_finite": bool(math.isfinite(bkm)),
        "energy_initial": energies[0],
        "energy_final": energies[-1],
        "energy_nonincreasing": bool(energies[-1] <= energies[0] + 1e-12),
        "note": "finite BKM / non-increasing energy => this trajectory stays smooth on [0,T] (conditional)",
    }


def stage34_ladder(n: int, alphas: list[float], nu: float, T: float, m: int, steps: int) -> dict:
    ladder = [classify_regime(a) for a in alphas]
    criteria = [_bkm_and_energy(n, a, nu, T, m, steps) for a in alphas]
    for row in ladder:
        _emit("stage3_regime", **{k: row[k] for k in ("alpha", "regime", "global_regularity_status")})
    return {
        "critical_alpha_3d": CRITICAL_ALPHA_3D,
        "scaling": "u_lambda(x,t) = lambda^{2a-1} u(lambda x, lambda^{2a} t); energy critical at 2a-1 = n/2",
        "dimension": 3,
        "proven_threshold_note": (
            f"alpha >= 5/4 => global regularity is an EXTERNAL theorem ({LIONS_CITATION}); "
            "omnibias records the citation and certifies numerical evidence only."
        ),
        "regularity_ladder": ladder,
        "conditional_criteria": criteria,
        "unproven_claim": False,
    }


# --------------------------------------------------------------------------- #
# Stage 5: honest theorem-readiness gates.                                       #
# --------------------------------------------------------------------------- #
def stage5_theorem_readiness(nu: float, m: int, out_dir: str) -> dict:
    regularity = build_regularity_closure_report(
        inequality_name="fractional_ns_alpha_1.00_energy_dissipation",
        coefficients={"nu_times_m_pow_2alpha": float(nu * m**2)},
        continuation_criterion="BKM_or_LPS_type_continuation",
    )
    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=23, n_radial=6, n_axial=7, radial_degree=1, axial_degree=1,
        max_iterations=2, step_size=0.01, viscosity=0.01,
    )
    interval_report = build_axisymmetric_interval_report(artifact)
    roadmap = build_ns_solve_or_falsify_report(interval_report)

    _write_json(os.path.join(out_dir, "regularity_closure_alpha1.json"), regularity)
    _write_json(os.path.join(out_dir, "solve_or_falsify_roadmap.json"), roadmap)

    final_gate = roadmap["phases"]["formal_verification"]["final_claim_gate"]
    result = {
        "classical_regularity_route": {
            "inequality_name": regularity["inequality_name"],
            "formalizable": regularity["formalizable"],
            "all_smooth_finite_energy_data_proof": regularity["obligations"][
                "all_smooth_finite_energy_data_proof"
            ],
            "open_obligations": regularity["open_obligations"],
            "unproven_claim": regularity["unproven_claim"],
        },
        "solve_or_falsify_roadmap": {
            "candidate_type": roadmap["candidate_type"],
            "unproven_claim": roadmap["unproven_claim"],
            "final_claim_gate_unproven_claim": final_gate["unproven_claim"],
            "n_open_obligations": len(roadmap["open_obligations"]),
            "sample_open_obligations": list(roadmap["open_obligations"])[:6],
        },
        "conclusion": (
            "The classical (alpha=1) global-regularity route is NOT closed and the roadmap's "
            "Global-regularity gate stays False. Track C reaches the proven hyperdissipative regime and "
            "certifies numerical evidence; it does not and cannot close the global-regularity problem."
        ),
    }
    _emit(
        "stage5",
        classical_route_formalizable=result["classical_regularity_route"]["formalizable"],
        roadmap_assumptions_gate=result["solve_or_falsify_roadmap"]["final_claim_gate_unproven_claim"],
    )
    return result


# --------------------------------------------------------------------------- #
# Stage 6: Tao's logarithmically supercritical dissipation (research edge).   #
# --------------------------------------------------------------------------- #
def stage6_log_supercritical(betas: list[float]) -> dict:
    rows = []
    for beta in betas:
        cls = classify_log_supercritical(beta)
        diag = tao_log_supercritical_diagnostic(beta)
        row = {**cls, "partial_integral_to_r_max": diag["partial_integral_to_r_max"]}
        rows.append(row)
        _emit(
            "stage6_beta",
            beta=beta,
            divergence_condition_met=cls["divergence_condition_met"],
            global_regularity_status=cls["global_regularity_status"],
        )
    return {
        "family": "logarithmically_supercritical_hyperdissipation",
        "note": (
            "Dissipation |k|^{5/2}/(log(e+|k|^2))^{2 beta} is strictly weaker than critical "
            "|k|^{5/2} yet Tao (2009) proves global regularity iff 4 beta <= 1 (external theorem)."
        ),
        "borderline_beta": 0.25,
        "per_beta": rows,
        "unproven_claim": False,
    }


# --------------------------------------------------------------------------- #
# Entry point.                                                                #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track C: fractional / hyperdissipative Navier-Stokes")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25, 1.5])
    p.add_argument("--betas", type=float, nargs="+", default=[0.125, 0.25, 0.5, 1.0])
    p.add_argument("--nu", type=float, default=0.05)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--m", type=int, default=2)
    p.add_argument("--recover-steps", type=int, default=600)
    p.add_argument("--recover-lr", type=float, default=0.05)
    p.add_argument("--criteria-steps", type=int, default=64)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--out-dir", type=str,
        default=os.path.join(
            os.environ.get("OMNIBIAS_SCRATCH", "artifacts"),
            "omnibias_runs", "fractional_ns",
        ),
    )
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def apply_smoke(args: argparse.Namespace) -> None:
    args.n = 16
    args.alphas = [1.0, 1.25]
    args.betas = [0.25, 0.5]
    args.recover_steps = 150
    args.criteria_steps = 16


def main() -> None:
    args = parse_args()
    if args.smoke:
        apply_smoke(args)
    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()
    _emit("start", config=vars(args), critical_alpha_3d=CRITICAL_ALPHA_3D)

    summary = {
        "unproven_claim": False,
        "note": (
            "Fractional / hyperdissipative Navier-Stokes. Classical alpha=1 global regularity is "
            "the OPEN global-regularity problem and is never claimed. alpha>=5/4 (and Tao's log-supercritical) "
            "regularity are EXTERNAL theorems, recorded as citations, not verified by omnibias."
        ),
        "config": vars(args),
        "stage1_model_certification": stage1_certify(args.n, list(args.alphas), args.nu, args.T, args.m),
        "stage2_learnable_order": stage2_learnable_order(args.n, args.recover_steps, args.recover_lr, args.seed),
        "stage3_4_criticality_ladder": stage34_ladder(
            args.n, list(args.alphas), args.nu, args.T, args.m, args.criteria_steps
        ),
        "stage5_theorem_readiness": stage5_theorem_readiness(args.nu, args.m, args.out_dir),
        "stage6_log_supercritical": stage6_log_supercritical(list(args.betas)),
    }
    summary["elapsed_s"] = round(time.time() - t_start, 2)

    summary_path = os.path.join(args.out_dir, "fractional_ns_summary.json")
    _write_json(summary_path, summary)
    _emit("saved", summary_json=summary_path, elapsed_s=summary["elapsed_s"])


if __name__ == "__main__":
    main()
