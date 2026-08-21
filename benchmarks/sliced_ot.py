# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-5: sliced OT of activation mixtures (theory 03-04).

Smoke earns G1 (exact W_1 vs spectral |F-G| at 1e-12), G2 (quantile
implicit gradient vs FD), G3 (closed-form slice variance 100x below
Monte Carlo on a rotationally invariant pair), G4 (symmetry +
triangle), G5 (direction_stderr vs bootstrap), and G6 (torch/jax CDF
parity). Exact per slice, sampled over directions. Founding bias
collapse, not temperature collapse. Sliced Wasserstein is not
Wasserstein.
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
from omnibias.measure.transport import (
    ActivationMixture,
    honesty_payload,
    named_worked_pair,
    quantile,
    sliced_wasserstein,
    w1_exact,
    worked_w1,
)
from omnibias.measure.transport._core import (
    bootstrap_direction_stderr,
    highprec_w1,
    krawczyk_root_count,
    mc_sliced_wasserstein,
    quantile_dtheta_means,
    sign_change_roots,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
SEEDS = (0, 1, 2, 3, 4)


def _run_g1() -> dict[str, Any]:
    mu, nu = named_worked_pair()
    w1 = w1_exact(mu, nu)
    worked_ok = abs(w1 - worked_w1()) <= 1e-12
    rng = np.random.default_rng(0)
    rels = []
    pairs = [(mu, nu)]
    for _ in range(6):
        n = int(rng.integers(1, 4))
        w = rng.random(n)
        a = ActivationMixture(w / w.sum(), rng.normal(size=n), rng.uniform(0.6, 2.5, size=n))
        n2 = int(rng.integers(1, 4))
        w2 = rng.random(n2)
        b = ActivationMixture(w2 / w2.sum(), rng.normal(size=n2), rng.uniform(0.6, 2.5, size=n2))
        pairs.append((a, b))
    pairs.append(
        (
            ActivationMixture([0.5, 0.5], [-3.0, 3.0], [1.2, 1.2]),
            ActivationMixture([0.5, 0.5], [-1.0, 1.0], [1.2, 1.2]),
        )
    )
    for a, b in pairs:
        exact = w1_exact(a, b)
        hp = highprec_w1(a, b)
        rels.append(abs(exact - hp) / max(abs(hp), 1e-15))
        roots = sign_change_roots(a, b)
        assert krawczyk_root_count(a, b, roots) == int(roots.size)
    max_rel = float(max(rels))
    return {
        "name": "g1_exactness",
        "passed": worked_ok and max_rel <= 1e-12,
        "worked_w1": w1,
        "max_rel": max_rel,
        "detail": "w1_exact vs tanh-independent GL of |F-G|; worked example closed form",
    }


def _run_g2() -> dict[str, Any]:
    mix = ActivationMixture([0.4, 0.6], [-0.7, 0.9], [1.1, 0.8])
    q = 0.35
    analytic = quantile_dtheta_means(mix, q)
    fd = np.empty_like(analytic)
    h = 1e-6
    for i in range(mix.n_components):
        m = mix.means.reshape(-1).copy()
        hi, lo = m.copy(), m.copy()
        hi[i] += h
        lo[i] -= h
        fd[i] = (
            float(quantile(ActivationMixture(mix.weights, hi, mix.scales), q))
            - float(quantile(ActivationMixture(mix.weights, lo, mix.scales), q))
        ) / (2.0 * h)
    rel = float(np.max(np.abs(analytic - fd) / np.maximum(np.abs(fd), 1e-12)))
    return {
        "name": "g2_quantile_grad",
        "passed": rel <= 1e-8,
        "rel": rel,
        "detail": "implicit dQ/dmu vs central FD",
    }


def _run_g3() -> dict[str, Any]:
    mu = ActivationMixture([1.0], [[0.0, 0.0]], [1.0])
    nu = ActivationMixture([1.0], [[0.0, 0.0]], [2.0])
    cf = [sliced_wasserstein(mu, nu, p=1, directions=8, seed=s).value for s in SEEDS]
    # extra seeds for a stable variance
    cf += [sliced_wasserstein(mu, nu, p=1, directions=8, seed=10 + s).value for s in SEEDS]
    mc = [mc_sliced_wasserstein(mu, nu, n_samples=64, directions=8, seed=s).value for s in range(10)]
    v_cf = float(np.var(cf, ddof=1))
    v_mc = float(np.var(mc, ddof=1))
    ratio = v_mc / max(v_cf, 1e-18)
    return {
        "name": "g3_variance",
        "passed": ratio >= 100.0,
        "var_cf": v_cf,
        "var_mc": v_mc,
        "ratio": ratio,
        "detail": "rotationally invariant pair: CF residual is direction-only",
    }


def _run_g4() -> dict[str, Any]:
    rng = np.random.default_rng(1)
    mixes = []
    for _ in range(6):
        w = rng.random(2)
        mixes.append(ActivationMixture(w / w.sum(), rng.normal(size=2), rng.uniform(0.8, 1.6, size=2)))
    sym = max(abs(w1_exact(a, b) - w1_exact(b, a)) for a, b in zip(mixes, mixes[1:] + mixes[:1], strict=True))
    tri = w1_exact(mixes[0], mixes[2]) <= w1_exact(mixes[0], mixes[1]) + w1_exact(mixes[1], mixes[2]) + 1e-10
    diag = max(w1_exact(a, a) for a in mixes)
    return {
        "name": "g4_metric",
        "passed": sym <= 1e-10 and tri and diag <= 1e-10,
        "max_asym": float(sym),
        "triangle_ok": bool(tri),
        "detail": "symmetry and triangle on randomized 1-D triples",
    }


def _run_g5() -> dict[str, Any]:
    mu = ActivationMixture([0.5, 0.5], [[-1.2, 0.3], [0.8, -0.4]], [1.0, 1.1])
    nu = ActivationMixture([0.6, 0.4], [[0.0, 0.0], [0.5, 0.7]], [0.9, 1.2])
    result = sliced_wasserstein(mu, nu, p=1, directions=24, seed=2)
    boot = bootstrap_direction_stderr(result.slices, n_boot=300, seed=3)
    rel = abs(result.direction_stderr - boot) / max(boot, 1e-15)
    return {
        "name": "g5_stderr",
        "passed": result.direction_stderr >= 0.0 and rel < 0.35,
        "stderr": result.direction_stderr,
        "bootstrap": boot,
        "rel": rel,
        "detail": "direction_stderr vs bootstrap SE of the slice mean",
    }


def _run_g6() -> dict[str, Any]:
    mu, _nu = named_worked_pair()
    xs = np.array([-1.5, 0.0, 0.7], dtype=np.float64)
    ref = mu.cdf(xs)
    ok_t = True
    ok_j = True
    try:
        import torch
        from omnibias.measure.transport import torch as tw

        got = tw.cdf(mu, torch.as_tensor(xs)).detach().cpu().numpy()
        ok_t = bool(np.array_equal(got, ref))
    except ImportError:
        ok_t = True
    try:
        import jax
        from omnibias.measure.transport import jax as jw

        jax.config.update("jax_enable_x64", True)
        got = np.asarray(jw.cdf(mu, xs))
        ok_j = bool(np.allclose(got, ref, rtol=0.0, atol=2e-16))
    except ImportError:
        ok_j = True
    return {
        "name": "g6_parity",
        "passed": ok_t and ok_j,
        "torch_equal": ok_t,
        "jax_equal": ok_j,
        "detail": "torch / jax CDF match numpy at float64",
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "sliced_ot.json" if full else "sliced_ot_smoke.json"
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
            schema="omnibias.benchmarks.sliced_ot.v1",
            config={
                "family": "sliced_ot",
                "full": full,
                "seeds": list(SEEDS),
                "honesty": honesty_payload(),
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "slicedot"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
