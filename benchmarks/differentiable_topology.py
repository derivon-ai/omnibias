# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-5: differentiable topology of arrangements (theory 03-09).

Smoke earns G1 (soft counts at the predicted rate), G2 (gap-bound
soundness on a grid and a random sample), G3 (certified count vs
labelling; ``Inconclusive`` on the spectrum; ``--full`` is 10 000
instances), G4 (1-D Morse persistence vs the same oracle,
bottleneck ``<= 1e-6``), G5 (persistence prior halves topological
error without degrading the pixel metric), and G6 (torch/jax
parity). No differentiable function equals a Betti number.
``beta -> inf`` is temperature collapse, not founding bias
collapse.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.shape.topology import (
    Inconclusive,
    bimodal_saddle_index,
    bottleneck_distance,
    certified_component_count,
    cluster_laplacian,
    connected_components_1d,
    exact_h0_persistence,
    honesty_payload,
    soft_component_count,
    soft_euler_characteristic,
    soft_persistence,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _run_g1() -> dict[str, Any]:
    ev = (0.0, 0.0, 0.0, 0.41, 0.55, 0.72)
    betas = (8.0, 16.0, 32.0, 64.0, 128.0)
    errs = [abs(soft_component_count(ev, epsilon=0.2, beta=b).value - 3.0) for b in betas]
    ok = all(errs[i + 1] <= 0.55 * errs[i] for i in range(len(errs) - 1))
    return {"name": "g1_rate", "passed": ok, "errors": errs, "detail": "four doublings; sigmoid tail"}


def _run_g2() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    violations = 0
    n = 0
    for eps in np.linspace(0.05, 0.9, 17):
        ev = (0.0, 0.0, 0.41, 0.55)
        sc = soft_component_count(ev, epsilon=float(eps), beta=20.0)
        true = sum(1 for lam in ev if lam < eps)
        n += 1
        if abs(sc.value - true) > sc.gap_bound + 1e-12:
            violations += 1
    for _ in range(64):
        ev = tuple(float(v) for v in rng.uniform(0.0, 1.0, size=6))
        eps = float(rng.uniform(0.1, 0.8))
        sc = soft_component_count(ev, epsilon=eps, beta=15.0)
        true = sum(1 for lam in ev if lam < eps)
        n += 1
        if abs(sc.value - true) > sc.gap_bound + 1e-12:
            violations += 1
    eu = soft_euler_characteristic((0.9, 0.1, 0.8), (0, 1, 0))
    n += 1
    if abs(eu.value - 2.0) > eu.gap_bound + 1e-12:
        violations += 1
    return {"name": "g2_gap", "passed": violations == 0, "n": n, "violations": violations}


def _run_g3(*, n_random: int) -> dict[str, Any]:
    rng = np.random.default_rng(1)
    wrong = 0
    n = 0
    for _ in range(int(n_random)):
        k = int(rng.integers(1, 5))
        g = float(rng.uniform(0.3, 0.8))
        lap = cluster_laplacian(k, n_per=2, gap=g)
        got = certified_component_count(lap, epsilon=0.15)
        n += 1
        if got != k:
            wrong += 1
    deg = certified_component_count(cluster_laplacian(1, n_per=2, gap=0.5), epsilon=1.0)
    ok = wrong == 0 and isinstance(deg, Inconclusive)
    return {
        "name": "g3_certified",
        "passed": ok,
        "n": n,
        "wrong": wrong,
        "degenerate": getattr(deg, "reason", None),
        "detail": "never a guessed integer",
    }


def _run_g4() -> dict[str, Any]:
    xs = np.linspace(-2.0, 2.0, 401)
    field = (xs**2 - 1.0) ** 2
    soft = soft_persistence(field, beta=1.0, max_dim=0)
    ref = exact_h0_persistence(1.0 / (1.0 + np.exp(-field)))
    dist = bottleneck_distance(soft, ref)
    return {"name": "g4_persistence", "passed": dist <= 1e-6, "bottleneck": dist}


def _run_g5() -> dict[str, Any]:
    rng = np.random.default_rng(2)
    x = np.linspace(0.0, 1.0, 32)
    topo_plain: list[int] = []
    topo_loss: list[int] = []
    pix_plain: list[float] = []
    pix_loss: list[float] = []
    for _ in range(5):
        c1 = 0.28 + 0.02 * float(rng.normal())
        c2 = 0.72 + 0.02 * float(rng.normal())
        obs = np.exp(-((x - c1) ** 2) / (2 * 0.045**2)) + np.exp(-((x - c2) ** 2) / (2 * 0.045**2))
        mask = obs >= 0.25

        def run(use_topo: bool, image: np.ndarray, support: np.ndarray) -> tuple[float, int]:
            u = image.copy()
            for _step in range(24):
                grad = np.zeros_like(u)
                grad[support] = 2.0 * (u[support] - image[support])
                if use_topo:
                    grad[bimodal_saddle_index(u)] -= 1.0
                u = np.clip(u - 0.15 * grad, 0.0, 1.0)
            pix = float(np.mean((u[support] - image[support]) ** 2))
            return pix, connected_components_1d(u, threshold=0.22)

        mse_p, n_p = run(False, obs, mask)
        mse_t, n_t = run(True, obs, mask)
        pix_plain.append(mse_p)
        pix_loss.append(mse_t)
        topo_plain.append(abs(n_p - 1))
        topo_loss.append(abs(n_t - 1))
    mean_plain = float(np.mean(topo_plain))
    mean_loss = float(np.mean(topo_loss))
    ok = mean_plain > 0.0 and mean_loss <= 0.5 * mean_plain + 1e-12
    ok = ok and float(np.mean(pix_loss)) <= float(np.mean(pix_plain)) + 0.05
    return {
        "name": "g5_prior",
        "passed": ok,
        "topo_plain": mean_plain,
        "topo_loss": mean_loss,
        "pix_plain": float(np.mean(pix_plain)),
        "pix_loss": float(np.mean(pix_loss)),
    }


def _run_g6() -> dict[str, Any]:
    ev = np.array([0.0, 0.0, 0.41, 0.55], dtype=np.float64)
    ref = soft_component_count(ev, epsilon=0.2, beta=50.0).value
    try:
        import torch
        from omnibias.shape.topology import torch as tw

        got_t = float(tw.soft_component_count(torch.tensor(ev, dtype=torch.float64), epsilon=0.2, beta=50.0))
    except ImportError:
        return {"name": "g6_parity", "passed": False, "detail": "torch missing"}
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        from omnibias.shape.topology import jax as jw

        got_j = float(jw.soft_component_count(ev, epsilon=0.2, beta=50.0))
    except ImportError:
        return {"name": "g6_parity", "passed": False, "detail": "jax missing"}
    ok = abs(got_t - ref) <= 1e-15 and abs(got_j - ref) <= 2e-15
    return {"name": "g6_parity", "passed": ok, "torch": got_t, "jax": got_j, "ref": ref}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "differentiable_topology.json" if full else "differentiable_topology_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(n_random=10_000 if full else 256), _run_g4(), _run_g5(), _run_g6()]
    for e in entries:
        print(e["name"], "ok" if e["passed"] else "FAIL")
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.differentiable_topology.v1",
            config={"family": "differentiable_topology", "full": full, "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "topology"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
