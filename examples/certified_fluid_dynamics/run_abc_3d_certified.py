# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Track B: rigorous certified-evidence bridge for 3D incompressible Navier-Stokes.

This turns Track A's *trained numerical* 3D field into **replayable, schema-
validated residual certificates**, and exercises the genuine 3D-reduced interval
machinery. Nothing here claims a global-regularity result: ``unproven_claim`` is ``False``
everywhere, and the honesty gate is shown to *block* any forged claim.

Four honest stages
------------------
A. **Exact baseline** -- ``manufactured_abc_flow`` (steady Beltrami ABC) through
   ``build_ns_cap_bundle`` -> schema check -> independent numpy replay
   (``verify_ns_cap_bundle``). Two independent spectral implementations agree at
   machine precision.

B. **The bridge** -- load the Track A ``VectorPotentialField`` checkpoint, sample
   ``u = curl(A)``, ``p`` and the *closed-form* time derivative ``u_t`` on a
   periodic grid at several time slices, seal each into a CAP bundle, validate
   the schema, and independently replay it. We also re-confirm the sub-1% ABC
   accuracy through the numpy/FFT path (a fully independent check of Track A).

C. **Proof machine** -- adjudicate the shipped 3D ``beltrami_abc_flow`` fixture
   to a full ``Verdict`` (schema / replay / honesty gates) and demonstrate that a
   forged ``unproven_claim`` is BLOCKED.

D. **3D-reduced rigor** -- the axisymmetric-swirl interval-enclosure and blow-up
   closure pipeline, each cross-checked by its numpy replay twin. These are
   genuine finite-dimensional *interval* certificates (``interval_verified``),
   not a continuum theorem.

Usage
-----
Local (CPU, ~a minute)::

    python -m examples.certified_fluid_dynamics.run_abc_3d_certified --smoke

Full run (uses the Track A v2 checkpoint on scratch)::

    python -m examples.certified_fluid_dynamics.run_abc_3d_certified \
        --ckpt-dir "artifacts/omnibias_runs/abc3d_gpu_v2" \
        --out-dir "artifacts/omnibias_runs/abc3d_certified"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# Allow ``import examples...`` when this file is run directly by path (the cluster
# submits scripts by absolute path, not via ``-m``).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
from omnibias.core.proof import Conjecture  # noqa: E402
from omnibias.pinn.certified import (  # noqa: E402
    build_axisymmetric_blowup_closure_report,
    build_axisymmetric_interval_report,
    build_default_machine,
    build_ns_cap_bundle,
    build_refined_axisymmetric_swirl_candidate_artifact,
    manufactured_abc_flow,
    ns_cap_schema_errors,
)
from omnibias.symbolic import (  # noqa: E402
    verify_axisymmetric_interval_report,
    verify_blowup_closure_report,
    verify_ns_cap_bundle,
    verify_refined_axisymmetric_swirl_candidate_artifact,
)

from examples.certified_fluid_dynamics.run_abc_3d_pinn import (  # noqa: E402
    TWO_PI,
    VEL,
    build_field,
)

LENGTHS_3D = (TWO_PI, TWO_PI, TWO_PI)


def _emit(event: str, **payload: object) -> None:
    print(
        json.dumps({"event": event, **payload}, sort_keys=True, default=_json_default),
        flush=True,
    )


def _json_default(o: object) -> object:
    if isinstance(o, np.generic):  # numpy scalars, incl. np.bool_
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _write_json(path: str, obj: object) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_json_default)


# --------------------------------------------------------------------------- #
# Stage A: exact manufactured ABC baseline.                                   #
# --------------------------------------------------------------------------- #
def stage_a_baseline(n: int, nu: float, out_dir: str) -> dict:
    mms = manufactured_abc_flow(n, viscosity=nu)
    bundle = build_ns_cap_bundle(
        mms["velocity"],
        mms["pressure"],
        velocity_t=mms["velocity_t"],
        forcing=mms["forcing"],
        viscosity=mms["viscosity"],
        density=mms["density"],
        lengths=mms["lengths"],
    )
    errs = ns_cap_schema_errors(bundle)
    replay = verify_ns_cap_bundle(bundle)
    _write_json(os.path.join(out_dir, "cap_bundle_manufactured_abc.json"), bundle)
    diag = bundle["residual_diagnostics"]
    result = {
        "n": n,
        "viscosity": nu,
        "schema_ok": errs == [],
        "schema_errors": errs,
        "residual_samples_match": replay["residual_samples_match"],
        "replay_momentum_max_abs_diff": replay["agreement_momentum_max_abs_diff"],
        "replay_continuity_max_abs_diff": replay["agreement_continuity_max_abs_diff"],
        "rms_momentum_residual": diag["rms_momentum_residual"],
        "max_abs_momentum_residual": diag["max_abs_momentum_residual"],
        "max_abs_continuity": diag["max_abs_continuity"],
        "max_abs_pressure_poisson": diag["max_abs_pressure_poisson"],
        "honesty": bundle["honesty"],
    }
    _emit("stage_a", **result)
    return result


# --------------------------------------------------------------------------- #
# Stage B: bridge the trained field into CAP bundles.                         #
# --------------------------------------------------------------------------- #
def _load_trained_field(ckpt_dir: str, device: str):
    with open(os.path.join(ckpt_dir, "metrics.json")) as fh:
        meta = json.load(fh)
    cfg = meta["config"]
    dtype = torch.float64 if cfg["dtype"] == "float64" else torch.float32
    field = build_field(
        int(cfg["K"]), int(cfg["time_hidden"]), int(cfg["time_depth"]), dtype, device
    )
    state = torch.load(
        os.path.join(ckpt_dir, "abc3d_field.pt"), map_location=device, weights_only=True
    )
    field.load_state_dict(state)
    field.eval()
    return field, cfg, dtype


def _grid_coords(n: int, t0: float, dtype: torch.dtype, device: str) -> torch.Tensor:
    ax = torch.arange(n, dtype=dtype, device=device) * (TWO_PI / n)
    gx, gy, gz = torch.meshgrid(ax, ax, ax, indexing="ij")
    t = torch.full((n**3,), float(t0), dtype=dtype, device=device)
    return torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1), t], dim=-1)


@torch.no_grad()
def _sample_trained_grid(field, t_axis: int, n: int, t0: float, dtype, device):
    """Sample ``(velocity, pressure, velocity_t)`` on a periodic grid, closed-form."""
    coords = _grid_coords(n, t0, dtype, device)
    state = field(coords)

    def grid(t: torch.Tensor) -> np.ndarray:
        return t.reshape(n, n, n).detach().cpu().numpy()

    velocity = np.stack([grid(state.ops.value(state, c)) for c in VEL])
    velocity_t = np.stack(
        [grid(state.ops.derivative(state, c, axis=t_axis, order=1)) for c in VEL]
    )
    pressure = grid(state.ops.value(state, "p"))
    return velocity, pressure, velocity_t


def _exact_decaying_abc_grid(n: int, t0: float, nu: float) -> np.ndarray:
    x = TWO_PI * np.arange(n, dtype=float) / n
    xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
    decay = float(np.exp(-nu * t0))
    return decay * np.stack(
        [np.sin(zz) + np.cos(yy), np.sin(xx) + np.cos(zz), np.sin(yy) + np.cos(xx)]
    )


def stage_b_bridge(
    ckpt_dir: str, n: int, times: list[float], device: str, out_dir: str
) -> dict:
    if not os.path.exists(os.path.join(ckpt_dir, "abc3d_field.pt")):
        result = {"available": False, "ckpt_dir": ckpt_dir, "reason": "checkpoint not found"}
        _emit("stage_b", **result)
        return result

    field, cfg, dtype = _load_trained_field(ckpt_dir, device)
    nu = float(cfg["nu"])
    t_axis = field.base.coordinate_spec.axis_index("t")

    slices: list[dict] = []
    for t0 in times:
        velocity, pressure, velocity_t = _sample_trained_grid(
            field, t_axis, n, t0, dtype, device
        )
        bundle = build_ns_cap_bundle(
            velocity,
            pressure,
            velocity_t=velocity_t,
            forcing=None,  # decaying ABC is unforced (f = 0)
            viscosity=nu,
            lengths=LENGTHS_3D,
        )
        errs = ns_cap_schema_errors(bundle)
        replay = verify_ns_cap_bundle(bundle)
        exact = _exact_decaying_abc_grid(n, t0, nu)
        rel_l2 = float(np.linalg.norm(velocity - exact) / np.linalg.norm(exact))
        _write_json(os.path.join(out_dir, f"cap_bundle_trained_t{t0:.2f}.json"), bundle)
        diag = bundle["residual_diagnostics"]
        rec = {
            "time": t0,
            "schema_ok": errs == [],
            "residual_samples_match": replay["residual_samples_match"],
            "replay_momentum_max_abs_diff": replay["agreement_momentum_max_abs_diff"],
            "rms_momentum_residual": diag["rms_momentum_residual"],
            "max_abs_momentum_residual": diag["max_abs_momentum_residual"],
            "max_abs_continuity": diag["max_abs_continuity"],
            "max_abs_pressure_poisson": diag["max_abs_pressure_poisson"],
            "rel_l2_vs_exact_abc": rel_l2,
            "honesty": bundle["honesty"],
        }
        slices.append(rec)
        _emit("stage_b_slice", **rec)

    result = {
        "available": True,
        "ckpt_dir": ckpt_dir,
        "grid_n": n,
        "viscosity": nu,
        "config": {k: cfg[k] for k in ("K", "time_hidden", "time_depth", "dtype")},
        "all_schema_ok": all(s["schema_ok"] for s in slices),
        "all_replay_match": all(s["residual_samples_match"] for s in slices),
        "mean_rel_l2_vs_exact_abc": float(np.mean([s["rel_l2_vs_exact_abc"] for s in slices])),
        "max_rel_l2_vs_exact_abc": float(np.max([s["rel_l2_vs_exact_abc"] for s in slices])),
        "mean_rms_momentum_residual": float(np.mean([s["rms_momentum_residual"] for s in slices])),
        "max_abs_continuity": float(np.max([s["max_abs_continuity"] for s in slices])),
        "slices": slices,
    }
    _emit("stage_b", **{k: v for k, v in result.items() if k != "slices"})
    return result


# --------------------------------------------------------------------------- #
# Stage C: proof-machine verdict + honesty gate.                              #
# --------------------------------------------------------------------------- #
def _verdict_row(verdict) -> dict:
    honesty = {}
    if verdict.certificate is not None:
        honesty = verdict.certificate.get("honesty", {})
    return {
        "status": verdict.status,
        "prover": verdict.prover,
        "schema_ok": verdict.schema_ok,
        "replay_ok": verdict.replay_ok,
        "honesty_ok": verdict.honesty_ok,
        "obligations": list(verdict.obligations),
        "certificate_honesty": honesty,
    }


def stage_c_machine(n: int, nu: float) -> dict:
    machine = build_default_machine()
    # Non-builtin fixtures reach the certificate builder through a ``fixture``
    # descriptor (only taylor-green / kolmogorov are matched by bare ``name``).
    data = {"fixture": {"name": "beltrami_abc_flow", "n": n, "viscosity": nu}}

    honest = machine.evaluate(
        Conjecture("beltrami ABC 3D residual", "navier_stokes_periodic_residual", data)
    )
    forged = machine.evaluate(
        Conjecture(
            "forged unproven ABC 3D",
            "navier_stokes_periodic_residual",
            data,
            claims={"unproven_claim": True},
        )
    )
    result = {
        "fixture": "beltrami_abc_flow",
        "honest_verdict": _verdict_row(honest),
        "forged_claim_verdict": _verdict_row(forged),
        "honesty_gate_blocks_forgery": forged.status == "BLOCKED"
        and forged.honesty_ok is False,
    }
    _emit(
        "stage_c",
        honest_status=result["honest_verdict"]["status"],
        honest_obligations=result["honest_verdict"]["obligations"],
        forged_status=result["forged_claim_verdict"]["status"],
        honesty_gate_blocks_forgery=result["honesty_gate_blocks_forgery"],
    )
    return result


# --------------------------------------------------------------------------- #
# Stage D: axisymmetric-swirl interval / blow-up closure (3D-reduced rigor).  #
# --------------------------------------------------------------------------- #
def stage_d_axisymmetric(out_dir: str) -> dict:
    artifact = build_refined_axisymmetric_swirl_candidate_artifact(
        seed=19,
        n_radial=6,
        n_axial=7,
        radial_degree=1,
        axial_degree=1,
        max_iterations=2,
        step_size=0.01,
        viscosity=0.01,
    )
    artifact_replay = verify_refined_axisymmetric_swirl_candidate_artifact(artifact)

    interval_report = build_axisymmetric_interval_report(artifact)
    interval_replay = verify_axisymmetric_interval_report(interval_report)

    closure = build_axisymmetric_blowup_closure_report(
        interval_report, norm_growth_exponent=0.25, linked_norm_profile=True
    )
    closure_replay = verify_blowup_closure_report(closure)

    _write_json(os.path.join(out_dir, "axisym_refined_artifact.json"), artifact)
    _write_json(os.path.join(out_dir, "axisym_interval_report.json"), interval_report)
    _write_json(os.path.join(out_dir, "axisym_blowup_closure_report.json"), closure)

    result = {
        "artifact_candidate_type": artifact["candidate_type"],
        "artifact_residual_descended": artifact["result"]["residual_descended"],
        "artifact_replay_match": bool(artifact_replay["replay_match"]),
        "interval_verified": interval_report["honesty"]["interval_verified"],
        "interval_report_match": bool(interval_replay["interval_report_match"]),
        "max_interval_violation": interval_replay["max_interval_violation"],
        "interval_stage": interval_replay["stage"],
        "tail_certified": interval_replay["tail_certified"],
        "axis_certified": interval_replay["axis_certified"],
        "continuum_certified": interval_replay["continuum_certified"],
        "upgrade_gate": interval_report["upgrade_gate"],
        "closure_report_match": bool(closure_replay["closure_report_match"]),
        "closure_expected_obligations": closure_replay["expected_obligations"],
        "closure_recomputed_radii_interval": closure_replay["recomputed_radii_interval"],
        "unproven_claim": bool(interval_report["honesty"]["unproven_claim"]),
    }
    _emit(
        "stage_d",
        interval_verified=result["interval_verified"],
        interval_report_match=result["interval_report_match"],
        interval_stage=result["interval_stage"],
        closure_report_match=result["closure_report_match"],
    )
    return result


# --------------------------------------------------------------------------- #
# Entry point.                                                                #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track B: 3D Navier-Stokes certified-evidence bridge")
    p.add_argument(
        "--ckpt-dir",
        type=str,
        default=os.path.join(
            os.environ.get("OMNIBIAS_SCRATCH", "artifacts"),
            "omnibias_runs",
            "abc3d_gpu_v2",
        ),
        help="Track A checkpoint dir (expects abc3d_field.pt + metrics.json)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(
            os.environ.get("OMNIBIAS_SCRATCH", "artifacts"),
            "omnibias_runs",
            "abc3d_certified",
        ),
        help="artifact directory for CAP bundles + reports; override with $OMNIBIAS_SCRATCH",
    )
    p.add_argument("--n", type=int, default=32, help="periodic grid points per axis")
    p.add_argument(
        "--times",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="time slices at which to bridge the trained field",
    )
    p.add_argument("--nu", type=float, default=0.05, help="viscosity for baseline/machine")
    p.add_argument("--smoke", action="store_true", help="tiny config for a quick check")
    return p.parse_args()


def apply_smoke(args: argparse.Namespace) -> None:
    args.n = 16
    args.times = [0.0, 0.5, 1.0]


def main() -> None:
    args = parse_args()
    if args.smoke:
        apply_smoke(args)
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.time()
    _emit("start", device=device, config=vars(args))

    summary = {
        "unproven_claim": False,
        "note": (
            "Rigorous certified evidence for one 3D NS instance: replayable, "
            "schema-validated residual certificates + 3D-reduced interval "
            "certificates. NOT a global-regularity statement."
        ),
        "config": vars(args),
        "device": device,
        "stage_a_manufactured_baseline": stage_a_baseline(args.n, args.nu, args.out_dir),
        "stage_b_trained_bridge": stage_b_bridge(
            args.ckpt_dir, args.n, list(args.times), device, args.out_dir
        ),
        "stage_c_proof_machine": stage_c_machine(args.n, args.nu),
        "stage_d_axisymmetric_interval": stage_d_axisymmetric(args.out_dir),
    }
    summary["elapsed_s"] = round(time.time() - t_start, 2)

    summary_path = os.path.join(args.out_dir, "certified_summary.json")
    _write_json(summary_path, summary)
    _emit("saved", summary_json=summary_path, elapsed_s=summary["elapsed_s"])


if __name__ == "__main__":
    main()
