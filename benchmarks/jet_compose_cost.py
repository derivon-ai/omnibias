# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Cost + exactness smoke for the directional ``compose_jet`` kernels.

The kernel composes an activation onto a jet from an **arbitrary** derivative
tower, which is truncated power-series composition; it is cubic in the
truncation order and this benchmark does **not** claim ``O(N^2)`` for it. What
it measures is (a) the constant-factor reduction from skipping the products that
vanish because ``v = u - u_0`` has valuation 1, at bit-for-bit identical values,
and (b) the genuinely quadratic Riccati fastpath ``compose_jet_riccati``, which
only exists because ``sigma' = P(sigma)`` closes the chain rule on the
activation itself.

Gates
-----
G1  the reduced kernel reproduces the dense kernel's values exactly (torch and
    jax, eager float64) -- exact equality, not ``allclose``.
G2  the reduced kernel issues at least 3x fewer elementwise multiplies at
    order 16 (counted, not estimated).
G3  the Riccati fastpath matches the tower path to 1e-11 relative, and reaches
    orders where the order-capped ``cot`` fastpath makes the tower path raise.
G4  torch <-> jax agreement of both kernels at 1e-14 relative.

Wall-clock rows are recorded, not gated (they are hardware-dependent).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import enable_x64, median_time_ms, provenance, write_json  # noqa: E402
from _gates import gates_block, require_backend_parity  # noqa: E402

enable_x64()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omnibias.jax.jet import compose_jet as j_compose  # noqa: E402
from omnibias.jax.jet import compose_jet_riccati as j_riccati  # noqa: E402
from omnibias.torch.activations.registry import get_activation  # noqa: E402
from omnibias.torch.jet import (  # noqa: E402
    _path_jet,
    _sigma_tower,
    layer_jet,
)
from omnibias.torch.jet import compose_jet as t_compose  # noqa: E402
from omnibias.torch.jet import compose_jet_riccati as t_riccati  # noqa: E402
from torch.func import jacfwd  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
BASELINE_NAME = "dense shifted-power compose_jet (full (k, n) convolution)"


def dense_compose_jet_torch(u_jet: Any, sigma_tower: Any) -> Any:
    """The dense kernel: every ``(k, n)`` product, including the zero ones."""
    np1 = u_jet.shape[0]
    zero = torch.zeros_like(u_jet[0])
    w = [zero] + [u_jet[j] for j in range(1, np1)]
    p = [torch.ones_like(u_jet[0])] + [zero for _ in range(np1 - 1)]
    result = [sigma_tower[0] * p[0]] + [zero for _ in range(np1 - 1)]
    fact = 1.0
    for k in range(1, np1):
        fact *= k
        new_p = []
        for n in range(np1):
            acc = zero
            for i in range(n + 1):
                acc = acc + p[i] * w[n - i]
            new_p.append(acc)
        p = new_p
        dk = sigma_tower[k] / fact
        for n in range(np1):
            result[n] = result[n] + dk * p[n]
    return torch.stack(result, dim=0)


def dense_compose_jet_jax(u_jet: Any, sigma_tower: Any) -> Any:
    np1 = u_jet.shape[0]
    zero = jnp.zeros_like(u_jet[0])
    w = [zero] + [u_jet[j] for j in range(1, np1)]
    p = [jnp.ones_like(u_jet[0])] + [zero for _ in range(np1 - 1)]
    result = [sigma_tower[0] * p[0]] + [zero for _ in range(np1 - 1)]
    fact = 1.0
    for k in range(1, np1):
        fact *= k
        new_p = []
        for n in range(np1):
            acc = zero
            for i in range(n + 1):
                acc = acc + p[i] * w[n - i]
            new_p.append(acc)
        p = new_p
        dk = sigma_tower[k] / fact
        for n in range(np1):
            result[n] = result[n] + dk * p[n]
    return jnp.stack(result, axis=0)


def _count_multiplies(fn: Any) -> int:
    from torch.utils._python_dispatch import TorchDispatchMode

    class MulCounter(TorchDispatchMode):
        def __init__(self) -> None:
            self.n = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001
            if "mul" in str(func):
                self.n += 1
            return func(*args, **(kwargs or {}))

    with MulCounter() as counter:
        fn()
    return int(counter.n)


def _max_rel(a: Any, b: Any) -> float:
    left = np.asarray(a, dtype=np.float64)
    right = np.asarray(b, dtype=np.float64)
    denom = np.abs(right) + 1e-30
    return float(np.max(np.abs(left - right) / denom))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    t0 = time.perf_counter()
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(0)
    orders = (4, 8, 12, 16, 20) if full else (4, 8, 12, 16)
    width = 4096 if full else 512

    # ---- G1: exact value preservation, both backends -------------------------
    exact_entries = []
    for order in orders:
        u = rng.normal(size=(order + 1, 16))
        tow = rng.normal(size=(order + 1, 16))
        tu, tt = torch.as_tensor(u), torch.as_tensor(tow)
        exact_entries.append(
            require_backend_parity(
                t_compose(tu, tt).numpy(),
                dense_compose_jet_torch(tu, tt).numpy(),
                name=f"g1_torch_exact_order{order}",
            )
        )
        ju, jt = jnp.asarray(u), jnp.asarray(tow)
        exact_entries.append(
            require_backend_parity(
                np.asarray(j_compose(ju, jt)),
                np.asarray(dense_compose_jet_jax(ju, jt)),
                name=f"g1_jax_exact_order{order}",
            )
        )

    # ---- G2: counted multiplies -------------------------------------------
    cost_rows: list[dict[str, Any]] = []
    x0 = torch.as_tensor(rng.normal(size=(8,)))
    v0 = torch.as_tensor(rng.normal(size=(8,)))
    W0 = torch.as_tensor(rng.normal(scale=0.6, size=(16, 8)))
    b0 = torch.as_tensor(rng.normal(scale=0.4, size=(16,)))
    for order in orders:
        u = torch.as_tensor(rng.normal(size=(order + 1, 2)))
        tow = torch.as_tensor(rng.normal(size=(order + 1, 2)))
        dense_mul = _count_multiplies(partial(dense_compose_jet_torch, u, tow))
        lean_mul = _count_multiplies(partial(t_compose, u, tow))
        # Whole-layer counts include building the derivative tower, which the
        # Riccati recurrence never does.
        z_jet0 = _path_jet(x0, v0, order)
        tower_mul = _count_multiplies(
            partial(layer_jet, z_jet0, W0, b0, "tanh", order)
        )
        riccati_mul = _count_multiplies(
            partial(layer_jet, z_jet0, W0, b0, "tanh", order, riccati=True)
        )
        cost_rows.append(
            {
                "order": order,
                "dense_multiplies": dense_mul,
                "reduced_multiplies": lean_mul,
                "ratio": dense_mul / max(lean_mul, 1),
                "layer_tower_multiplies": tower_mul,
                "layer_riccati_multiplies": riccati_mul,
                "layer_ratio": tower_mul / max(riccati_mul, 1),
            }
        )
    ratio16 = next(r["ratio"] for r in cost_rows if r["order"] == 16)
    g2 = {
        "name": "g2_multiply_count_ratio_order16",
        "passed": bool(ratio16 >= 3.0),
        "ratio": float(ratio16),
        "min_ratio": 3.0,
        "note": "constant factor; the kernel stays cubic in the truncation order",
    }

    # ---- G3: Riccati fastpath accuracy + reach ----------------------------
    ric_rows: list[dict[str, Any]] = []
    worst_rel = 0.0
    for name in ("sigmoid", "tanh", "exp"):
        spec = get_activation(name)
        poly = spec.riccati_polynomial
        assert poly is not None
        order = 12 if full else 8
        u = torch.zeros(order + 1, 8)
        u[0] = 0.3 + torch.as_tensor(rng.normal(scale=0.05, size=8))
        u[1] = torch.as_tensor(rng.normal(scale=0.3, size=8))
        u[2] = torch.as_tensor(rng.normal(scale=0.2, size=8))
        ref = t_compose(u, _sigma_tower(spec, u[0], order))
        fast = t_riccati(u, spec.forward(u[0]), poly)
        rel = _max_rel(fast.numpy(), ref.numpy())
        worst_rel = max(worst_rel, rel)
        ric_rows.append({"activation": name, "order": order, "max_rel": rel})
    g3a = {
        "name": "g3_riccati_matches_tower",
        "passed": bool(worst_rel <= 1e-11),
        "max_rel": float(worst_rel),
        "max_rel_allowed": 1e-11,
        "rows": ric_rows,
    }

    # cot caps its fastpath at order 3; the Riccati recurrence does not care.
    spec = get_activation("cot")
    poly_cot = spec.riccati_polynomial
    assert poly_cot is not None
    cap_order = 7
    z0 = torch.tensor([1.0])
    d = torch.tensor([0.35])
    u_cot = torch.zeros(cap_order + 1, 1)
    u_cot[0], u_cot[1] = z0, d
    tower_raised = False
    try:
        _sigma_tower(spec, z0, cap_order)
    except ValueError:
        tower_raised = True
    got = t_riccati(u_cot, spec.forward(z0), poly_cot)
    # Independent oracle: nested forward-mode jacfwd of t -> cot(z0 + t d).
    truth = []
    for k in range(cap_order + 1):
        fn = partial(lambda t, s=spec: s.forward(z0 + t * d))
        for _ in range(k):
            fn = jacfwd(fn)
        truth.append(float(fn(torch.tensor(0.0)).reshape(())) / math.factorial(k))
    cap_rel = _max_rel(got.numpy().reshape(-1), np.asarray(truth))
    g3b = {
        "name": "g3_riccati_passes_fastpath_order_cap",
        "passed": bool(tower_raised and cap_rel <= 1e-9),
        "activation": "cot",
        "order": cap_order,
        "tower_path_raised": tower_raised,
        "max_rel_vs_nested_jacfwd": float(cap_rel),
    }

    # ---- G4: cross-backend agreement --------------------------------------
    order = 9
    u = rng.normal(size=(order + 1, 8))
    tow = rng.normal(size=(order + 1, 8))
    gen_rel = _max_rel(
        t_compose(torch.as_tensor(u), torch.as_tensor(tow)).numpy(),
        np.asarray(j_compose(jnp.asarray(u), jnp.asarray(tow))),
    )
    s0 = np.tanh(u[0])
    ric_rel = _max_rel(
        t_riccati(torch.as_tensor(u), torch.as_tensor(s0), (1.0, 0.0, -1.0)).numpy(),
        np.asarray(j_riccati(jnp.asarray(u), jnp.asarray(s0), (1.0, 0.0, -1.0))),
    )
    g4 = {
        "name": "g4_torch_jax_agreement",
        "passed": bool(gen_rel <= 1e-14 and ric_rel <= 1e-14),
        "compose_jet_max_rel": float(gen_rel),
        "compose_jet_riccati_max_rel": float(ric_rel),
        "max_rel_allowed": 1e-14,
    }

    entries = [*exact_entries, g2, g3a, g3b, g4]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")

    # ---- informational wall clock -----------------------------------------
    timings: list[dict[str, Any]] = []
    for order in orders:
        u = torch.as_tensor(rng.normal(size=(order + 1, width)))
        tow = torch.as_tensor(rng.normal(size=(order + 1, width)))
        dense_ms = median_time_ms(partial(dense_compose_jet_torch, u, tow), repeats=11)
        lean_ms = median_time_ms(partial(t_compose, u, tow), repeats=11)
        D, H = 8, width
        W = torch.as_tensor(rng.normal(scale=0.6, size=(H, D)))
        bias = torch.as_tensor(rng.normal(scale=0.4, size=(H,)))
        z_jet = _path_jet(
            torch.as_tensor(rng.normal(size=(D,))),
            torch.as_tensor(rng.normal(size=(D,))),
            order,
        )
        tower_ms = median_time_ms(
            partial(layer_jet, z_jet, W, bias, "tanh", order), repeats=11
        )
        riccati_ms = median_time_ms(
            partial(layer_jet, z_jet, W, bias, "tanh", order, riccati=True),
            repeats=11,
        )
        timings.append(
            {
                "order": order,
                "trailing_width": width,
                "compose_dense_ms": dense_ms,
                "compose_reduced_ms": lean_ms,
                "compose_speedup": dense_ms / max(lean_ms, 1e-12),
                "layer_tower_ms": tower_ms,
                "layer_riccati_ms": riccati_ms,
                "layer_speedup": tower_ms / max(riccati_ms, 1e-12),
            }
        )
        print(
            f"order {order:2d}: compose dense {dense_ms:7.3f} ms -> reduced "
            f"{lean_ms:7.3f} ms | layer tower {tower_ms:7.3f} ms -> riccati "
            f"{riccati_ms:7.3f} ms"
        )

    payload = {
        **provenance(
            schema="omnibias.benchmarks.jet_compose_cost.v1",
            config={
                "family": "jet_compose_cost",
                "full": full,
                "orders": list(orders),
                "trailing_width": width,
                "claim": (
                    "arbitrary-tower compose_jet is cubic in the truncation order; "
                    "the reduction is a ~3x constant factor at identical values. "
                    "O(N^2) is claimed only for compose_jet_riccati, where "
                    "sigma' = P(sigma) closes the chain rule on the activation."
                ),
                "timing_note": (
                    "wall clock is recorded on a shared commodity CPU and is "
                    "noisy; the deterministic multiply_counts carry the cost "
                    "claim, not the timings"
                ),
            },
        ),
        "baseline": {"name": BASELINE_NAME},
        "gates": dict(gates_block(entries)),
        "multiply_counts": cost_rows,
        "timings_informational": timings,
        "jax_version": jax.__version__,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "jet_compose_cost.json" if full else "jet_compose_cost_smoke.json"
    if full:
        dest = SCRATCH / "jets" / "compose_cost"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
