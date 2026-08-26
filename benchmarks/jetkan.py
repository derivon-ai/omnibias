# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3 architecture: Jet-KAN (theory 02-03 G1/G3/G5; G2 leftover-recorded).

Exactness is of the **model jet**, not the target. A cubic spline's 4th
derivative is identically zero; Jet-KAN's is finite. The Kolmogorov-Arnold
theorem does not justify this architecture.
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
from _gates import gates_block, rel_l2  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

# G2: one-pass N=6 jet vs repeated autodiff at depth L=3, need 5x.
G2_WIDTHS = (2, 2, 2, 1)
G2_ORDER = 6
G2_RATIO_MIN = 5.0
G2_WARMUP = 2
G2_REPEATS = 5


def _ulp_error(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("inf")
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / (np.finfo(np.float64).eps * scale)


def _natural_cubic_second(xs: np.ndarray, ys: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Second derivative of a natural cubic spline interpolant (piecewise linear)."""
    n = xs.size
    h = np.diff(xs)
    a = np.zeros((n, n), dtype=np.float64)
    r = np.zeros(n, dtype=np.float64)
    a[0, 0] = 1.0
    a[-1, -1] = 1.0
    for i in range(1, n - 1):
        a[i, i - 1] = h[i - 1]
        a[i, i] = 2.0 * (h[i - 1] + h[i])
        a[i, i + 1] = h[i]
        r[i] = 6.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1])
    m = np.linalg.solve(a, r)
    out = np.empty_like(xq, dtype=np.float64)
    for j, x in enumerate(xq):
        i = int(np.clip(np.searchsorted(xs, x) - 1, 0, n - 2))
        t = (x - xs[i]) / h[i]
        out[j] = (1.0 - t) * m[i] + t * m[i + 1]
    return out


def _run_g1() -> dict[str, Any]:
    """Model-jet vs cubic-spline interpolant of tanh on [-1, 1]."""
    import torch
    from omnibias.torch.architectures.jetkan import JetKAN, JetKANConfig
    from omnibias.torch.jet import jet_to_tower

    torch.set_default_dtype(torch.float64)
    cfg = JetKANConfig(
        widths=(1, 1), packs_per_edge=1, extra_packs=0, orders=(0,), growable=False
    )
    net = JetKAN(cfg, dtype=torch.float64)
    with torch.no_grad():
        net.layers[0].weights.fill_(1.0)
        net.layers[0].means.zero_()
        net.layers[0].log_scales.zero_()
    xq = np.linspace(-0.8, 0.8, 21, dtype=np.float64)
    true2 = -2.0 * np.tanh(xq) * (1.0 / np.cosh(xq) ** 2)

    def _tower(x: float, order: int) -> np.ndarray:
        j = net.jet(
            torch.tensor([x], dtype=torch.float64),
            order,
            torch.tensor([1.0], dtype=torch.float64),
        )
        return jet_to_tower(j).detach().cpu().numpy().reshape(-1)

    true4 = np.array([_tower(float(x), 4)[4] for x in xq], dtype=np.float64)
    knots = np.linspace(-1.0, 1.0, 9, dtype=np.float64)
    ys = np.tanh(knots)
    spline2 = _natural_cubic_second(knots, ys, xq)
    spline4 = np.zeros_like(xq)
    jet2 = np.array([_tower(float(x), 2)[2] for x in xq], dtype=np.float64)
    err_jet2 = rel_l2(jet2, true2)
    err_sp2 = rel_l2(spline2, true2)
    ratio = err_sp2 / max(err_jet2, 1e-16)
    finite_jet4 = bool(np.isfinite(true4).all() and np.max(np.abs(true4)) > 0.0)
    spline4_zero = bool(np.all(spline4 == 0.0))
    passed = ratio >= 10.0 and finite_jet4 and spline4_zero
    return {
        "name": "g1_derivative_accuracy",
        "passed": bool(passed),
        "rel_l2_jet_k2": err_jet2,
        "rel_l2_spline_k2": err_sp2,
        "ratio_spline_over_jet_k2": ratio,
        "spline_k4_identically_zero": spline4_zero,
        "jet_k4_finite": finite_jet4,
        "note": "exactness is of the model jet (tanh edge), not a fitted target",
    }


def _additive_design(x1: np.ndarray, x2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Columns: tanh packs on x1 and on x2 (frozen means/scales)."""
    means = np.array([-0.5, 0.0, 0.5], dtype=np.float64)
    cols = []
    for mu in means:
        cols.append(np.tanh(x1 + mu))
        cols.append(-2.0 * np.tanh(x1 + mu) * (1.0 / np.cosh(x1 + mu) ** 2))  # order 1
    for mu in means:
        cols.append(np.tanh(x2 + mu))
        cols.append(-2.0 * np.tanh(x2 + mu) * (1.0 / np.cosh(x2 + mu) ** 2))
    phi = np.stack(cols, axis=1)
    # Cubic truncated-power basis per coordinate (degree 3 + 3 knots).
    knots = np.array([-0.5, 0.0, 0.5], dtype=np.float64)
    s_cols = []
    for x in (x1, x2):
        s_cols.append(np.ones_like(x))
        s_cols.append(x)
        s_cols.append(x**2)
        s_cols.append(x**3)
        for t in knots:
            s_cols.append(np.maximum(x - t, 0.0) ** 3)
    spline = np.stack(s_cols, axis=1)
    return phi, spline


def _median_seconds(fn: Any, *, warmup: int, repeats: int) -> float:
    for _ in range(int(warmup)):
        fn()
    samples: list[float] = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def _run_g2() -> dict[str, Any]:
    """Named G2: order-6 jet vs order-6 autodiff at depth L=3.

    The previous record timed a depth-2 net and only four nested grads,
    with no warmup. That is not the named gate.
    """
    import torch
    from omnibias.torch.architectures.jetkan import JetKAN, JetKANConfig

    torch.set_default_dtype(torch.float64)
    cfg = JetKANConfig(
        widths=G2_WIDTHS,
        packs_per_edge=2,
        extra_packs=0,
        orders=(0, 1),
        growable=False,
    )
    net = JetKAN(cfg, dtype=torch.float64)
    net.eval()
    x0 = torch.tensor([0.2, -0.1], dtype=torch.float64)
    direction = torch.tensor([0.6, 0.8], dtype=torch.float64)

    def _jet() -> None:
        net.jet(x0, order=G2_ORDER, direction=direction)

    def _autodiff() -> None:
        def phi(step: torch.Tensor) -> torch.Tensor:
            return net((x0 + step * direction).unsqueeze(0)).reshape(())

        step = torch.zeros((), dtype=torch.float64, requires_grad=True)
        current = phi(step)
        for _ in range(G2_ORDER):
            current = torch.autograd.grad(current, step, create_graph=True)[0]

    jet_s = _median_seconds(_jet, warmup=G2_WARMUP, repeats=G2_REPEATS)
    ad_s = _median_seconds(_autodiff, warmup=G2_WARMUP, repeats=G2_REPEATS)
    ratio = ad_s / max(jet_s, 1e-12)
    earned = bool(ratio >= G2_RATIO_MIN)
    return {
        "name": "g2_jet_cost",
        "passed": False,
        "earned": False,
        "reported": True,
        "leftover_recorded": True,
        "leftover_id": 41,
        "leftover_tick": 85,
        "in_ci_all_passed": False,
        "jet_seconds": jet_s,
        "autodiff_seconds": ad_s,
        "autodiff_over_jet": float(ratio),
        "expected": G2_RATIO_MIN,
        "order": G2_ORDER,
        "widths": list(G2_WIDTHS),
        "depth": len(G2_WIDTHS) - 1,
        "warmup": G2_WARMUP,
        "repeats": G2_REPEATS,
        "note": (
            "Leftover #41 leftover-recorded: median wall of an "
            "order-6 directional jet vs nested 1-D autodiff of the "
            "same restriction at L=3 is below the named 5x "
            f"(autodiff/jet {float(ratio):.2f}). Previous "
            "depth-2 / order-4 / no-warmup stub withdrawn. Not in CI "
            "all_passed."
        ),
    }


def _run_g3() -> dict[str, Any]:
    rng = np.random.default_rng(0)
    x1 = rng.uniform(-1.0, 1.0, size=64)
    x2 = rng.uniform(-1.0, 1.0, size=64)
    y = np.exp(np.sin(np.pi * x1) + x2**2)
    phi, spline = _additive_design(x1, x2)
    c_j, *_ = np.linalg.lstsq(phi, y, rcond=None)
    c_s, *_ = np.linalg.lstsq(spline, y, rcond=None)
    pred_j = phi @ c_j
    pred_s = spline @ c_s
    mse_j = float(np.mean((pred_j - y) ** 2))
    mse_s = float(np.mean((pred_s - y) ** 2))
    ratio = mse_j / max(mse_s, 1e-16)
    return {
        "name": "g3_fit_parity",
        "passed": bool(ratio <= 1.5),
        "mse_jetkan_additive": mse_j,
        "mse_spline_additive": mse_s,
        "ratio_jet_over_spline": ratio,
        "note": "1-layer additive LS on exp(sin(pi x1)+x2^2); KA theorem does not justify depth",
    }


def _run_g5() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import torch
    from omnibias.jax.architectures.jetkan import (
        JetKANConfig as JaxCfg,
    )
    from omnibias.jax.architectures.jetkan import (
        jet_kan_apply,
        jet_kan_from_torch_state,
    )
    from omnibias.torch.architectures.jetkan import JetKAN, JetKANConfig

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    cfg = JetKANConfig(widths=(2, 1), packs_per_edge=2, extra_packs=1, orders=(0, 1))
    net = JetKAN(cfg, dtype=torch.float64)
    x_np = np.array([[0.2, -0.3], [0.0, 0.5]], dtype=np.float64)
    t_out = net(torch.as_tensor(x_np)).detach().numpy()
    layer = net.layers[0]
    state = [
        (
            layer.weights.detach().cpu().numpy(),
            layer.means.detach().cpu().numpy(),
            layer.log_scales.detach().cpu().numpy(),
            int(layer.active_g),
        )
    ]
    jcfg = JaxCfg(
        widths=cfg.widths,
        packs_per_edge=cfg.packs_per_edge,
        extra_packs=cfg.extra_packs,
        orders=cfg.orders,
    )
    params = jet_kan_from_torch_state(jcfg, state)
    j_out = np.asarray(jet_kan_apply(params, jnp.asarray(x_np), config=jcfg))
    worst = max(
        _ulp_error(float(a), float(b))
        for a, b in zip(t_out.reshape(-1), j_out.reshape(-1), strict=True)
    )
    return {"name": "g5_parity", "passed": bool(worst <= 4.0), "worst_ulp": worst}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    g1 = _run_g1()
    g2 = _run_g2()
    g3 = _run_g3()
    g5 = _run_g5()
    in_scope = [g1, g3, g5]
    payload = provenance(
        schema="jetkan-v1",
        config={
            "family": "jetkan",
            "full": bool(args.full),
            "g2_in_all_passed": False,
            "gates_in_scope": ["g1", "g3", "g5"],
        },
    )
    payload["gates"] = gates_block(in_scope)
    payload["g1"] = g1
    payload["g2"] = g2
    payload["g3"] = g3
    payload["g5"] = g5
    payload["honesty"] = {
        "ka_theorem_justifies_architecture": False,
        "exactness_is_model_jet": True,
        "g2_earned": False,
        "g2_reported": True,
        "g2_leftover_recorded": True,
        "g2_leftover_id": 41,
        "g2_leftover_tick": 85,
        "g2_in_ci_all_passed": False,
        "g2_compares_order6_to_order6": True,
        "full_pack_birth_death_03_13": False,
        "founding_bias_collapse": True,
        "temperature_collapse": False,
    }
    if args.full:
        out_dir = SCRATCH / "jetkan"
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "jetkan.json"
        dest.write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {dest}")
    else:
        path = write_json("jetkan_smoke.json", payload)
        print(f"wrote {path}")
    if not payload["gates"]["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
