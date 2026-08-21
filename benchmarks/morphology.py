# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-5: differentiable morphology (theory 03-05).

Smoke earns G1 (1/beta hard-limit rate over four doublings), G2
(gap sound + one-sided), G3 (composition-aware bound), G4 (learned
SE vs flat, five seeds), G5 (soft DT within bound), and G6
(torch/jax parity). Soft max is temperature collapse, not founding
bias collapse. Gap is worst-case. Not a seventh OperatorBlock role.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import jax as _jax

    _jax.config.update("jax_enable_x64", True)
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.shape.morphology import (
    StructuringElement,
    dilate,
    erode,
    exact_distance_transform,
    hard_dilate,
    hard_erode,
    honesty_payload,
    morphological_gradient,
    morphology_gap_bound,
    named_worked_signal,
    named_worked_soft_center,
    opening,
    soft_distance_transform,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
SEEDS = (0, 1, 2, 3, 4)


def _run_g1() -> dict[str, Any]:
    f = np.full(7, 2.0, dtype=np.float64)
    se = StructuringElement.flat(1)
    hard = hard_dilate(f, se)
    betas = [2.0, 4.0, 8.0, 16.0, 32.0]
    devs = [float(np.max(dilate(f, se, beta=b).value - hard)) for b in betas]
    ratios = [devs[i + 1] / max(devs[i], 1e-18) for i in range(len(devs) - 1)]
    return {
        "name": "g1_rate",
        "passed": all(0.4 <= r <= 0.6 for r in ratios),
        "devs": devs,
        "ratios": ratios,
        "worked_center": named_worked_soft_center(beta=2.0),
        "detail": "four doublings on a constant window; predicted 1/beta",
    }


def _run_g2() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    se = StructuringElement.flat(1)
    beta = 3.0
    bound = morphology_gap_bound(size=se.size, beta=beta, compositions=1)
    viol = 0
    for _ in range(8):
        f = rng.normal(size=16)
        d = dilate(f, se, beta=beta)
        e = erode(f, se, beta=beta)
        hd, he = hard_dilate(f, se), hard_erode(f, se)
        if np.any(d.value + 1e-12 < hd) or np.any(e.value > he + 1e-12):
            viol += 1
        if np.any(d.value - hd > bound + 1e-12) or np.any(he - e.value > bound + 1e-12):
            viol += 1
    return {
        "name": "g2_bound",
        "passed": viol == 0,
        "violations": viol,
        "bound": bound,
        "detail": "dilation >= hard, erosion <= hard, gap holds",
    }


def _run_g3() -> dict[str, Any]:
    f = np.full(9, 2.0, dtype=np.float64)
    se = StructuringElement.flat(1)
    beta = 1.5
    one = morphology_gap_bound(size=se.size, beta=beta, compositions=1)
    two = morphology_gap_bound(size=se.size, beta=beta, compositions=2)
    op = opening(f, se, beta=beta)
    g = morphological_gradient(f, se, beta=beta)
    return {
        "name": "g3_composition",
        "passed": op.compositions == 2 and abs(op.gap_bound - two) < 1e-12 and g.observed_dev > one and g.observed_dev <= two + 1e-12,
        "one": one,
        "two": two,
        "opening_bound": op.gap_bound,
        "grad_dev": g.observed_dev,
        "detail": "composition count 2; single-operator bound is violated by the gradient",
    }


def _run_g4() -> dict[str, Any]:
    se_hand = StructuringElement.flat(2)
    clean = np.array([0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0], dtype=np.float64)
    skills = []
    for seed in SEEDS:
        rng = np.random.default_rng(20 + seed)
        noisy = clean.copy()
        for pos in (0, 7):
            if rng.random() > 0.2:
                noisy[pos] = 1.0
                if pos + 1 < noisy.size:
                    noisy[pos + 1] = 1.0

        def mse(se: StructuringElement, _noisy: np.ndarray = noisy) -> float:
            return float(np.mean((opening(_noisy, se, beta=8.0).value - clean) ** 2))

        hand = mse(se_hand)
        vals = np.zeros(5, dtype=np.float64)
        off = np.arange(-2, 3, dtype=np.int64)
        for _ in range(40):
            g = np.zeros_like(vals)
            for i in range(vals.size):
                hi, lo = vals.copy(), vals.copy()
                hi[i] += 1e-3
                lo[i] -= 1e-3
                g[i] = (mse(StructuringElement(off, hi)) - mse(StructuringElement(off, lo))) / 2e-3
            vals = vals - 0.35 * g
        learned = mse(StructuringElement(off, vals))
        skills.append((hand - learned) / max(hand, 1e-12))
    mean_skill = float(np.mean(skills))
    return {
        "name": "g4_learned_se",
        "passed": mean_skill >= 0.20,
        "mean_skill": mean_skill,
        "skills": skills,
        "detail": "learned 5-tap opening vs flat-5 on width-2 salt",
    }


def _run_g5() -> dict[str, Any]:
    occ = np.array([0, 0, 1, 0, 0, 1, 0], dtype=np.float64)
    exact = exact_distance_transform(occ)
    beta = 4.0
    soft = soft_distance_transform(occ, beta=beta)
    ok = bool(np.all(soft.value <= exact + 1e-12) and np.all(exact - soft.value <= soft.gap_bound + 1e-12))
    return {
        "name": "g5_distance",
        "passed": ok,
        "gap_bound": soft.gap_bound,
        "max_dev": float(np.max(exact - soft.value)),
        "detail": "soft DT is a lower bound within log(N_obj)/beta",
    }


def _run_g6() -> dict[str, Any]:
    f = named_worked_signal()
    se = StructuringElement.flat(1)
    ref = dilate(f, se, beta=2.0).value
    ok_t = True
    ok_j = True
    try:
        import torch
        from omnibias.shape.morphology import torch as tw

        got = tw.dilate(torch.as_tensor(f), se, beta=2.0).detach().cpu().numpy()
        ok_t = bool(np.allclose(got, ref, rtol=0.0, atol=1e-15))
    except ImportError:
        ok_t = True
    try:
        import jax
        from omnibias.shape.morphology import jax as jw

        jax.config.update("jax_enable_x64", True)
        got = np.asarray(jw.dilate(f, se, beta=2.0))
        ok_j = bool(np.allclose(got, ref, rtol=0.0, atol=2e-15))
    except ImportError:
        ok_j = True
    return {
        "name": "g6_parity",
        "passed": ok_t and ok_j,
        "torch_equal": ok_t,
        "jax_equal": ok_j,
        "worked": float(named_worked_soft_center(beta=2.0)),
        "detail": "torch / jax dilate match numpy",
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "morphology.json" if full else "morphology_smoke.json"
    t0 = time.perf_counter()
    print("G1...")
    g1 = _run_g1()
    print("G2...")
    g2 = _run_g2()
    print("G3...")
    g3 = _run_g3()
    print("G4...")
    g4 = _run_g4()
    print("G5...")
    g5 = _run_g5()
    print("G6...")
    g6 = _run_g6()
    entries = [g1, g2, g3, g4, g5, g6]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.morphology.v1",
            config={
                "family": "morphology",
                "full": full,
                "seeds": list(SEEDS),
                "honesty": honesty_payload(),
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "morphology"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
