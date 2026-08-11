# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-1 primitive: multipack Birkhoff collapse (theory 01-01 G1/G2/G3/G5).

Smoke (default) earns exactness / stability / poisedness honesty gates and
records the measured float64 order ceiling. G4 (two-interface task skill) is
deferred -- ``g4_earned: false``. ``--full`` repeats the ulp sweep on a denser
grid under ``$OMNIBIAS_SCRATCH/multipack/``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _ulp_error(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("inf")
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b), 2.2250738585072014e-308)
    return abs(a - b) / (np.finfo(np.float64).eps * scale)


def _mp_poly(base: str, z: float, n: int) -> float:
    import mpmath
    from omnibias.core.polynomials import sigmoid_polynomial_coeffs, tanh_polynomial_coeffs

    mpmath.mp.dps = 80
    if base == "tanh":
        s = mpmath.tanh(mpmath.mpf(z))
        coeffs = tanh_polynomial_coeffs(n) if n > 0 else None
    else:
        s = mpmath.mpf(1) / (mpmath.mpf(1) + mpmath.exp(-mpmath.mpf(z)))
        coeffs = sigmoid_polynomial_coeffs(n) if n > 0 else None
    if n == 0:
        return float(s)
    assert coeffs is not None
    acc = mpmath.mpf(0)
    for c in reversed(coeffs):
        acc = acc * s + mpmath.mpf(c)
    return float(acc)


def _run_g1(*, full: bool) -> dict[str, Any]:
    import torch
    from omnibias.core.multipack import MultiPackSpec, PackSpec
    from omnibias.torch.multipack import MultiPackUnit

    torch.set_default_dtype(torch.float64)
    n_grid = 161 if full else 81
    grid = np.linspace(-1.0, 1.0, n_grid)
    ceilings: dict[str, int] = {}
    curves: dict[str, list[float]] = {}
    passed = True
    for base in ("sigmoid", "tanh"):
        worsts: list[float] = []
        ceiling = -1
        for n in range(0, 7):
            unit = MultiPackUnit(
                1,
                MultiPackSpec((PackSpec(n, 0.0, 1.0),)),
                base=base,
                learnable_means=False,
                learnable_weights=False,
            )
            worst = 0.0
            for z in grid:
                pred = float(
                    unit(torch.tensor([[float(z)]], dtype=torch.float64)).item()
                )
                worst = max(worst, _ulp_error(pred, _mp_poly(base, float(z), n)))
            worsts.append(worst)
            if worst <= 4.0:
                ceiling = n
        curves[base] = [float(w) for w in worsts]
        ceilings[base] = int(ceiling)
        if ceiling < 1:
            passed = False
    # Worked-example multipack on the same domain.
    spec = MultiPackSpec((PackSpec(1, -0.5, 1.0), PackSpec(2, 0.5, 0.25)))
    unit = MultiPackUnit(
        1, spec, base="tanh", learnable_means=False, learnable_weights=False
    )
    worst_mp = 0.0
    for z in grid:
        pred = float(unit(torch.tensor([[float(z)]], dtype=torch.float64)).item())
        ref = sum(
            p.weight * _mp_poly("tanh", float(z) + p.mean, p.order) for p in spec.packs
        )
        worst_mp = max(worst_mp, _ulp_error(pred, ref))
    if worst_mp > 4.0:
        passed = False
    return {
        "name": "g1_exactness",
        "passed": passed,
        "expected": 4.0,
        "order_ceilings": ceilings,
        "ulp_curves": curves,
        "worked_example_worst_ulp": float(worst_mp),
        "domain": {"z_lo": -1.0, "z_hi": 1.0, "n_grid": n_grid},
        "note": (
            "ceiling = max order with worst_ulp<=4 on the domain; "
            "§11 conditioning recorded rather than threshold softening"
        ),
    }


def _run_g2() -> dict[str, Any]:
    import math as _math

    import torch
    from omnibias.core.multipack import (
        MultiPackSpec,
        PackSpec,
        central_stencil_weights,
    )
    from omnibias.torch.multipack import MultiPackUnit

    torch.set_default_dtype(torch.float64)
    spec = MultiPackSpec((PackSpec(1, -0.5, 1.0), PackSpec(2, 0.5, 0.25)))
    unit = MultiPackUnit(
        1, spec, base="tanh", learnable_means=False, learnable_weights=False
    )
    z0 = 0.0
    closed = float(unit(torch.tensor([[z0]], dtype=torch.float64)).item())
    deltas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    fd_errs = []
    for delta in deltas:
        acc = 0.0
        for p in spec.packs:
            offsets, signs = central_stencil_weights(p.order, delta)
            for off, s in zip(offsets, signs, strict=True):
                acc += p.weight * s * _math.tanh(z0 + p.mean + off)
        fd_errs.append(float(abs(acc - closed)))
    closed2 = float(unit(torch.tensor([[z0]], dtype=torch.float64)).item())
    passed = closed2 == closed and fd_errs[-1] > fd_errs[2]
    return {
        "name": "g2_stability",
        "passed": passed,
        "closed_form": closed,
        "fd_errors": fd_errs,
        "deltas": deltas,
    }


def _run_g3() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import torch
    from omnibias.core.multipack import MultiPackSpec, PackSpec
    from omnibias.jax.multipack import init_multipack, multipack_apply
    from omnibias.torch.multipack import MultiPackUnit

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    spec = MultiPackSpec((PackSpec(1, -0.5, 1.0), PackSpec(2, 0.5, 0.25)))
    unit = MultiPackUnit(
        2, spec, base="tanh", learnable_means=False, learnable_weights=False
    )
    z_np = np.array([[0.0, 0.0], [0.25, -0.25], [0.5, 0.5]], dtype=np.float64)
    torch_out = unit(torch.as_tensor(z_np)).detach().numpy()
    _a, means, weights, orders, mean_index = init_multipack(
        2, spec, base="tanh", share_means=True
    )
    jax_out = np.asarray(
        multipack_apply(
            jnp.asarray(z_np), means, weights, orders, "tanh", mean_index=mean_index
        )
    )
    equal = bool(np.array_equal(torch_out, jax_out))
    return {
        "name": "g3_parity",
        "passed": equal,
        "max_abs_diff": float(np.max(np.abs(torch_out - jax_out))),
    }


def _run_g5() -> dict[str, Any]:
    from omnibias.core.multipack import MultiPackSpec, PackSpec, is_poised, polya_condition

    unpoised = MultiPackSpec((PackSpec(0, 0.0), PackSpec(2, 0.0)))
    poised = MultiPackSpec(
        (
            PackSpec(0, -0.5),
            PackSpec(1, -0.5),
            PackSpec(0, 0.5),
            PackSpec(1, 0.5),
            PackSpec(2, 0.5),
        )
    )
    u = is_poised(unpoised)
    p = is_poised(poised)
    passed = (
        polya_condition(unpoised) is False
        and u is False
        and p is True
    )
    return {
        "name": "g5_poisedness_honesty",
        "passed": passed,
        "unpoised_is_poised": u,
        "poised_is_poised": p,
        "representation_claim_withheld_for_unpoised": True,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = (
        "multipack_birkhoff.json" if full else "multipack_birkhoff_smoke.json"
    )
    config = {
        "family": "multipack_birkhoff",
        "full": full,
        "g4_earned": False,
        "gates_in_scope": ["g1", "g2", "g3", "g5"],
    }
    payload = provenance(schema="multipack-birkhoff-v1", config=config)
    t0 = time.perf_counter()
    print("G1 exactness / order ceiling...")
    g1 = _run_g1(full=full)
    print("G2 FD collapse stability...")
    g2 = _run_g2()
    print("G3 torch/jax parity...")
    g3 = _run_g3()
    print("G5 poisedness honesty...")
    g5 = _run_g5()
    entries = [g1, g2, g3, g5]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    gates = dict(gates_block(entries))
    payload.update(
        {
            "gates": gates,
            "g1": g1,
            "g2": g2,
            "g3": g3,
            "g5": g5,
            "honesty": {
                "claim_rung": 1,
                "bias_collapse": True,
                "temperature_collapse": False,
                "g1_earned": bool(g1["passed"]),
                "g2_earned": bool(g2["passed"]),
                "g3_earned": bool(g3["passed"]),
                "g4_earned": False,
                "g5_earned": bool(g5["passed"]),
                "order_ceilings": g1["order_ceilings"],
                "representation_requires_poisedness": True,
                "licensed_sentence": (
                    "MultiPackUnit matches a high-dps mpmath reference within "
                    "4 ulp up to the recorded float64 order ceiling; the "
                    "closed form is stable as delta->0 while FD breaks; "
                    "torch/jax agree bit-for-bit on the gated grid; unpoised "
                    "supports return is_poised=False and representation "
                    "claims are withheld"
                ),
            },
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }
    )
    out = write_json(artifact, payload)
    print(f"wrote {out} all_passed={gates['all_passed']} ceilings={g1['order_ceilings']}")
    if full:
        scratch = SCRATCH / "multipack"
        scratch.mkdir(parents=True, exist_ok=True)
        (scratch / artifact).write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"copied to {scratch / artifact}")
    if not gates["all_passed"]:
        raise SystemExit(1)
    return dict(payload)


if __name__ == "__main__":
    main()
