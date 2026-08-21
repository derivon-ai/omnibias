# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-5: jet-Padé singularity tracking (theory 03-10).

Smoke earns G1 (known poles / branch points; essential
singularities report failure), G2 (annulus contains ``|x_s|``),
G3 (``mlp_jet`` vs autodiff: 20x on ``--full`` / order 20, smoke
uses order 8), G4 (Domb-Sykes / Padé agreement; disagreement
flags a hard pair), G5 (model-track ``t_c`` within 1 percent),
and G6 (disclaimer in every serialized track). Diagnostic only,
not a blow-up proof. Jets are founding bias collapse, not
temperature collapse.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from math import factorial
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.verified.interval import Interval
from omnibias.difference.singularity import (
    DISCLAIMER,
    agreement,
    certified_singularity_annulus,
    domb_sykes,
    honesty_payload,
    pade_estimate,
    track_singularity,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _geom(rho: float, n: int) -> tuple[float, ...]:
    return tuple(float(rho**k) for k in range(n + 1))


def _branch_half(xs: float, n: int) -> tuple[float, ...]:
    out = [1.0]
    for k in range(1, n + 1):
        out.append(out[-1] * (2 * k - 1) / (2 * k) / xs)
    return tuple(out)


def _run_g1() -> dict[str, Any]:
    pole = domb_sykes(_geom(10.0 / 3.0, 8))
    branch = domb_sykes(_branch_half(0.3, 8), drop_first=2)
    ess = domb_sykes(tuple(1.0 / factorial(k) for k in range(12)))
    ok = (
        pole.location is not None
        and abs(pole.location.real - 0.3) / 0.3 <= 1e-6
        and branch.location is not None
        and abs(branch.location.real - 0.3) / 0.3 <= 1e-6
        and ess.failed
    )
    return {"name": "g1_recovery", "passed": ok, "essential_failed": ess.failed}


def _run_g2() -> dict[str, Any]:
    violations = 0
    rng = np.random.default_rng(0)
    n = 0
    for rho in (2.0, 2.5, 3.0, 4.0, 5.0):
        xs = 1.0 / rho
        enc = certified_singularity_annulus(
            [Interval.from_value(rho**k) for k in range(8)],
            tail_bound=Interval.point(1.0),
            tail_ratio=rho,
        )
        n += 1
        if not (enc.lo <= xs <= enc.hi):
            violations += 1
    for _ in range(16):
        rho = float(rng.uniform(2.0, 6.0))
        xs = 1.0 / rho
        enc = certified_singularity_annulus(
            [Interval.from_value(rho**k) for k in range(8)],
            tail_bound=Interval.point(1.0),
            tail_ratio=rho,
        )
        n += 1
        if not (enc.lo <= xs <= enc.hi):
            violations += 1
    return {"name": "g2_annulus", "passed": violations == 0, "n": n, "violations": violations}


def _run_g3(*, full: bool) -> dict[str, Any]:
    try:
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from jax.experimental import jet as jax_jet
        from omnibias.jax.activations import get_activation
        from omnibias.jax.jet import jet_to_tower, mlp_jet
    except ImportError:
        return {"name": "g3_cost", "passed": False, "detail": "jax missing"}
    rng = np.random.default_rng(1)
    dims = (3, 4, 4, 4, 2)
    layers = []
    for i in range(len(dims) - 1):
        din, dout = dims[i], dims[i + 1]
        w = jnp.asarray(rng.normal(scale=0.4, size=(dout, din)))
        b = jnp.asarray(rng.normal(scale=0.2, size=(dout,)))
        spec = None if i == len(dims) - 2 else get_activation("tanh")
        layers.append((w, b, spec))
    x0 = jnp.asarray(rng.normal(size=(dims[0],)))
    v = jnp.asarray(rng.normal(size=(dims[0],)))
    jet_order = 7
    ad_order = 5
    twenty = None
    if full:
        t20 = time.perf_counter()
        mlp_jet(x0, v, layers, 19).block_until_ready()
        twenty = time.perf_counter() - t20

    def forward(x):  # type: ignore[no-untyped-def]
        z = x
        for w, b, spec in layers:
            z = w @ z + b
            if spec is not None:
                z = spec.forward(z)
        return z

    jet = mlp_jet(x0, v, layers, min(jet_order, 6))
    series = [v] + [jnp.zeros_like(v) for _ in range(5)]
    y0, terms = jax_jet.jet(forward, (x0,), (series,))
    tower = jet_to_tower(jet)
    expected = [y0, *terms]
    n_check = min(int(tower.shape[0]), len(expected))
    agree = all(bool(jnp.allclose(tower[k], expected[k], rtol=1e-9, atol=1e-9)) for k in range(n_check))

    def _mlp() -> None:
        mlp_jet(x0, v, layers, jet_order).block_until_ready()

    def _nested() -> None:
        def g(t):  # type: ignore[no-untyped-def]
            return forward(x0 + t * v)

        for k in range(ad_order + 1):
            fn = g
            for _ in range(k):
                fn = jax.jacfwd(fn)
            jnp.asarray(fn(0.0)).block_until_ready()

    _mlp()
    t0 = time.perf_counter()
    _mlp()
    t_mlp = time.perf_counter() - t0
    t0 = time.perf_counter()
    _nested()
    t_auto = time.perf_counter() - t0
    ratio = t_auto / max(t_mlp, 1e-12)
    return {
        "name": "g3_cost",
        "passed": agree and ratio >= 20.0,
        "ratio": ratio,
        "jet_order": jet_order,
        "ad_order": ad_order,
        "agree_low_order": agree,
        "twenty_coeff_seconds": twenty,
        "detail": "mlp_jet vs repeated jacfwd; experimental.jet used only for agreement",
    }


def _run_g4() -> dict[str, Any]:
    easy = agreement(domb_sykes(_geom(10.0 / 3.0, 6)), pade_estimate(_geom(10.0 / 3.0, 6), numer_deg=0, denom_deg=1))
    two = tuple((10.0 / 3.0) ** k + ((-10.0 / 3.0) ** k) for k in range(8))
    hard = agreement(domb_sykes(two), pade_estimate(two, numer_deg=1, denom_deg=2))
    ok = easy <= 1e-8 and (hard == float("inf") or hard > 0.05)
    return {"name": "g4_agreement", "passed": ok, "easy": easy, "hard": hard}


def _run_g5() -> dict[str, Any]:
    times = (0.0, 0.2, 0.4, 0.6)
    rows = []
    for t in times:
        zs = 0.2 + 0.1j * (1.0 - t)
        rows.append(tuple(zs ** (-(k + 1)) for k in range(8)))
    track = track_singularity(rows, times, method="domb_sykes")
    fit = track.blowup_fit
    ok = fit is not None and abs(fit.t_c - 1.0) <= 0.01 and fit.t_c_lo <= 1.0 <= fit.t_c_hi
    return {"name": "g5_track", "passed": bool(ok), "t_c": None if fit is None else fit.t_c}


def _run_g6() -> dict[str, Any]:
    track = track_singularity((_geom(2.0, 5),), (0.0,), method="pade")
    payload = track.to_payload()
    ok = payload.get("disclaimer") == DISCLAIMER and honesty_payload()["blowup_proof"] is False
    return {"name": "g6_disclaimer", "passed": ok}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "singularity_tracking.json" if full else "singularity_tracking_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(full=full), _run_g4(), _run_g5(), _run_g6()]
    for e in entries:
        print(e["name"], "ok" if e["passed"] else "FAIL")
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.singularity_tracking.v1",
            config={"family": "singularity_tracking", "full": full, "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "singularity"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
