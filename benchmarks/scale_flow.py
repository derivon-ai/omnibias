# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-5: scale flow and coarse-graining (theory 03-07).

Smoke earns G1 (exact rescaling), G2 (overlap vs quadrature),
G3 (linear Galerkin exactness), G4 (derived schedule vs hand-tuned
FBPINN-like scales on the spectral-bias sinusoid), G5 (V-cycle
5x residual drop), and G6 (exponents refuse without a truncation
order; three-order study recorded). ``alpha`` is a tempering scale,
not a collapse parameter.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.composed_curvature import eval_tanh_derivative
from omnibias.core.scale import (
    FlowSystem,
    ScaleBand,
    ScaledPack,
    coarse_grain_linear,
    eval_gaussian_derivative,
    flow_coefficients,
    honesty_payload,
    overlap,
    report_exponents,
    rescale_pack,
    stiffness_matrix,
)
from omnibias.fields.scale import (
    _jacobi,
    grid_free_vcycle,
    lstsq_readout_mse,
    residual_norm,
    scale_schedule,
)
from spectral_bias_fbpinn import (
    scale_flow_window_scales,  # type: ignore[import-not-found]  # noqa: E402
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _run_g1() -> dict[str, Any]:
    worst_ulp = 0.0
    for n in range(11):
        pack = ScaledPack(order=n, mean=0.0, alpha=1.0, base="tanh")
        for f in (1.0 / 1024.0, 1.0 / 8.0, 8.0, 1024.0):
            back = rescale_pack(rescale_pack(pack, f), 1.0 / f)
            if back.alpha != pack.alpha:
                return {"name": "g1_rescale", "passed": False, "detail": "round-trip failed"}
            got = rescale_pack(pack, f).value(0.2)
            ref = (f**n) * pack.value(f * 0.2)
            ulps = abs(got - ref) / math.ulp(ref if ref != 0.0 else 1.0)
            worst_ulp = max(worst_ulp, ulps)
    worked = rescale_pack(ScaledPack(order=2, mean=0.0, alpha=1.0, base="tanh"), 2.0).value(0.2)
    ref = 4.0 * eval_tanh_derivative(0.4, 2)
    ok = worst_ulp <= 4.0 and abs(worked - ref) <= 4.0 * math.ulp(ref)
    return {"name": "g1_rescale", "passed": ok, "worst_ulp": worst_ulp, "detail": "orders 0..10, three decades"}


def _run_g2() -> dict[str, Any]:
    p = ScaledPack(order=1, mean=0.0, alpha=1.0)
    q = ScaledPack(order=1, mean=0.0, alpha=8.0)
    closed = overlap(p, q, derivative_order=2)
    xs = np.linspace(-12.0, 12.0, 40001)
    dx = float(xs[1] - xs[0])
    qk = np.array(
        [
            q.weight
            * (q.alpha ** (q.order + 2))
            * eval_gaussian_derivative(q.alpha * (float(x) - q.mean), q.order + 2)
            for x in xs
        ],
        dtype=np.float64,
    )
    pv = np.array([p.value(float(x)) for x in xs], dtype=np.float64)
    num = float(np.dot(pv, qk) * dx)
    rel = abs(closed - num) / max(1.0, abs(num))
    return {"name": "g2_overlap", "passed": rel <= 1e-12, "rel": rel, "closed": closed, "numeric": num}


def _run_g3() -> dict[str, Any]:
    packs = (
        ScaledPack(order=0, mean=-0.4, alpha=1.0),
        ScaledPack(order=0, mean=0.4, alpha=1.0),
        ScaledPack(order=0, mean=0.0, alpha=8.0),
    )
    op = coarse_grain_linear(packs, cutoff=1.0, derivative_order=2)
    c = (0.3, -0.7)
    xs = np.linspace(-10.0, 10.0, 30001)
    dx = float(xs[1] - xs[0])
    lu = np.zeros_like(xs)
    for cj, q in zip(c, op.slow, strict=True):
        lu = lu + cj * np.array(
            [
                q.weight
                * (q.alpha ** (q.order + 2))
                * eval_gaussian_derivative(q.alpha * (float(x) - q.mean), q.order + 2)
                for x in xs
            ],
            dtype=np.float64,
        )
    worst = 0.0
    for pred, p in zip(op.apply(c), op.slow, strict=True):
        pv = np.array([p.value(float(x)) for x in xs], dtype=np.float64)
        num = float(np.dot(pv, lu) * dx)
        worst = max(worst, abs(pred - num) / max(1.0, abs(num)))
    return {"name": "g3_linear", "passed": worst <= 1e-12, "worst": worst, "detail": "slow-subspace Galerkin"}


def _run_g4() -> dict[str, Any]:
    freq = 8
    x = np.linspace(0.0, 1.0, 256)
    y = np.sin(2.0 * np.pi * freq * x)
    hand = (1.0, 2.0, 4.0)
    derived = scale_flow_window_scales(freq, n_levels=3)
    scheduled = scale_schedule(
        target_band=lambda t: 2.0 * np.pi * (1.0 + (freq - 1) * t),
        steps=3,
        base="gaussian",
        order=1,
    )
    mse_hand = lstsq_readout_mse(x, y, hand, n_means=16)
    mse_derived = lstsq_readout_mse(x, y, derived, n_means=16)
    verdict = "beats" if mse_derived < mse_hand else "matches"
    ok = mse_derived <= mse_hand * 1.0000001 and derived == scheduled
    return {
        "name": "g4_curriculum",
        "passed": ok,
        "mse_hand": mse_hand,
        "mse_derived": mse_derived,
        "verdict": verdict,
        "derived_alphas": list(derived),
        "detail": "spectral-bias sinusoid f=8; hand-tuned (1,2,4) vs derived",
    }


def _run_g5() -> dict[str, Any]:
    packs = (
        ScaledPack(order=0, mean=-0.8, alpha=0.8),
        ScaledPack(order=0, mean=0.8, alpha=0.8),
        ScaledPack(order=0, mean=-0.4, alpha=2.0),
        ScaledPack(order=0, mean=0.0, alpha=2.0),
        ScaledPack(order=0, mean=0.4, alpha=2.0),
    )
    a = -stiffness_matrix(packs, derivative_order=2)
    rng = np.random.default_rng(0)
    true = rng.normal(size=len(packs))
    rhs = a @ true
    bands = (ScaleBand(0.5, 1.0), ScaleBand(1.5, 2.5))
    u0 = np.zeros(len(packs))
    single = _jacobi(a, rhs, u0.copy(), omega=0.6, sweeps=4)
    cycled = np.asarray(grid_free_vcycle(packs, rhs, bands=bands, u0=u0), dtype=np.float64)
    r0 = float(np.linalg.norm(rhs))
    r_single = residual_norm(packs, rhs, single)
    r_cycle = residual_norm(packs, rhs, cycled)
    ratio = r0 / max(r_cycle, 1e-18)
    return {
        "name": "g5_vcycle",
        "passed": r_cycle <= r0 / 5.0 and r_cycle < r_single,
        "reduction": ratio,
        "r_single": r_single,
        "r_cycle": r_cycle,
        "detail": "grid-free V-cycle vs four Jacobi sweeps",
    }


def _run_g6() -> dict[str, Any]:
    packs = (
        ScaledPack(order=0, mean=0.0, alpha=1.0),
        ScaledPack(order=0, mean=0.0, alpha=4.0),
    )
    system = flow_coefficients(packs, order=2)
    fp = system.fixed_points()[0]
    refused = False
    try:
        report_exponents(SimpleNamespace(coefficients=system.coefficients), fp)
    except ValueError:
        refused = True
    study = [float(FlowSystem(system.coefficients, truncation_order=o).exponents(fp)[0]) for o in (1, 2, 3)]
    exp = report_exponents(system, fp)
    ok = refused and system.truncation_order == 2 and len(study) == 3 and len(exp) == 1
    return {
        "name": "g6_truncation",
        "passed": ok,
        "study": study,
        "truncation_order": system.truncation_order,
        "detail": "refuses without order; three-order study recorded",
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "scale_flow.json" if full else "scale_flow_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(), _run_g5(), _run_g6()]
    for e in entries:
        print(e["name"], "ok" if e["passed"] else "FAIL")
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.scale_flow.v1",
            config={"family": "scale_flow", "full": full, "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "scaleflow"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
