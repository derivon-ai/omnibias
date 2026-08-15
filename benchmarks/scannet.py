# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3 architecture: Scan-Net (theory 02-01 G1/G2/G5; G3 cost not CI-gated).

Smoke earns per-layer interior-shift equivariance, a scattered-interface win
versus voxelized ``cmbConv1d`` with skill vs midpoint, and torch/jax parity.
G3 (wall/point independent of N) is recorded, not in CI ``all_passed``.
G4 (k-NN may win on density) is reported, not a win condition.
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
from _gates import gates_block, skill_score  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _ulp_error(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("inf")
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / (np.finfo(np.float64).eps * scale)


def _run_g1() -> dict[str, Any]:
    import torch
    from omnibias.torch.architectures.scannet import ScanNet, ScanNetConfig

    torch.set_default_dtype(torch.float64)
    cfg = ScanNetConfig(
        dim_in=1, channels=(2,), bank_sizes=(5,), bank_extents=(1.0,), template="grad"
    )
    net = ScanNet(cfg, dtype=torch.float64)
    layer = net.layers[0]
    spacing = float(layer.scan.offsets[1] - layer.scan.offsets[0])
    z = torch.tensor([[0.12, -0.08]], dtype=torch.float64)
    r0 = net.layer_scan_response(z, 0)
    r_shift = net.layer_scan_response(z + spacing, 0)
    worst = 0.0
    for a, b in zip(
        r_shift[..., :-1].reshape(-1).tolist(),
        r0[..., 1:].reshape(-1).tolist(),
        strict=True,
    ):
        worst = max(worst, _ulp_error(a, b))
    return {
        "name": "g1_layer_equivariance",
        "passed": bool(worst <= 4.0),
        "worst_ulp": worst,
        "expected": 4.0,
        "note": "per-layer, per-direction, on-lattice; not R^D translation",
    }


def _two_peaks(values: np.ndarray, coords: np.ndarray) -> list[float]:
    """Two largest local maxima of ``values`` at ``coords`` (both 1-D)."""
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    crd = np.asarray(coords, dtype=np.float64).reshape(-1)
    peaks: list[tuple[float, float]] = []
    if vals.size < 3:
        order = np.argsort(vals)[::-1]
        return [float(crd[int(i)]) for i in order[:2]]
    for i in range(1, vals.size - 1):
        if vals[i] >= vals[i - 1] and vals[i] >= vals[i + 1]:
            peaks.append((float(vals[i]), float(crd[i])))
    peaks.sort(reverse=True)
    if len(peaks) >= 2:
        return sorted([peaks[0][1], peaks[1][1]])
    order = np.argsort(vals)[::-1]
    return sorted([float(crd[int(order[0])]), float(crd[int(order[min(1, vals.size - 1)])])])


def _scan_two_interfaces(xs: np.ndarray, us: np.ndarray) -> list[float]:
    """One-layer ScanNet along x; pool jump-weighted bank responses."""
    import torch
    from omnibias.torch.activations.registry import get_activation
    from omnibias.torch.architectures.scannet import ScanNet, ScanNetConfig
    from omnibias.torch.scan import scan_response, template_from_op

    cfg = ScanNetConfig(
        dim_in=1,
        channels=(1,),
        bank_sizes=(17,),
        bank_extents=(1.0,),
        template="grad",
        readout="response",
    )
    net = ScanNet(cfg, dtype=torch.float64)
    with torch.no_grad():
        net.layers[0].proj.weight.fill_(1.0)
        net.layers[0].proj.bias.zero_()
        layer = net.layers[0]
        z = torch.as_tensor(xs.reshape(-1, 1), dtype=torch.float64)
        spec = template_from_op("grad")
        base = get_activation("tanh")
        resp = scan_response(z, layer.scan.offsets, layer.scan.scales, spec, base)
        weight = torch.as_tensor(us * (2.0 - us), dtype=torch.float64).reshape(-1, 1)
        pooled = (resp.reshape(xs.size, -1) * weight).sum(dim=0)
        taus = layer.scan.offsets.detach().cpu().numpy()
        energy = pooled.abs().detach().cpu().numpy()
    peaks = _two_peaks(energy, taus)
    return sorted([-p for p in peaks])


def _voxel_cmb_two_interfaces(xs: np.ndarray, us: np.ndarray) -> list[float]:
    """Bin the field and take the two largest |cmbConv1d grad| peaks."""
    import torch
    from omnibias.torch.blocks import cmbConv1d

    n_bins = 16
    lo, hi = -1.0, 1.0
    idx = np.clip(((xs - lo) / (hi - lo) * n_bins).astype(int), 0, n_bins - 1)
    grid = np.zeros(n_bins, dtype=np.float64)
    count = np.zeros(n_bins, dtype=np.float64)
    for i, u in zip(idx, us, strict=True):
        grid[i] += u
        count[i] += 1.0
    count = np.maximum(count, 1.0)
    grid = grid / count
    conv = cmbConv1d(1, 1, kernel_size=3, padding=1, op="grad", base="tanh")
    with torch.no_grad():
        t = torch.as_tensor(grid, dtype=torch.float64).reshape(1, 1, -1)
        feat = conv(t).reshape(-1).numpy()
    centers = lo + (np.arange(n_bins) + 0.5) * (hi - lo) / n_bins
    return _two_peaks(np.abs(feat), centers)


def _run_g2(*, full: bool) -> dict[str, Any]:
    n_pts = 96 if full else 64
    seeds = list(range(5))
    scan_err: list[float] = []
    voxel_err: list[float] = []
    true_all: list[float] = []
    scan_all: list[float] = []
    per_seed: list[dict[str, float]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        a = float(rng.uniform(-0.45, -0.15))
        b = float(rng.uniform(0.15, 0.45))
        xs = rng.uniform(-1.0, 1.0, size=n_pts)
        us = np.tanh(20.0 * (xs - a)) - np.tanh(20.0 * (xs - b))
        pred = _scan_two_interfaces(xs, us)
        vox = _voxel_cmb_two_interfaces(xs, us)
        truth = sorted([a, b])
        s_err = 0.5 * (abs(pred[0] - truth[0]) + abs(pred[1] - truth[1]))
        v_err = 0.5 * (abs(vox[0] - truth[0]) + abs(vox[1] - truth[1]))
        scan_err.append(s_err)
        voxel_err.append(v_err)
        true_all.extend(truth)
        scan_all.extend(pred)
        per_seed.append({"scan_mae": s_err, "voxel_mae": v_err})
    mae_scan = float(np.mean(scan_err))
    mae_vox = float(np.mean(voxel_err))
    skill = skill_score(np.asarray(scan_all), np.asarray(true_all))
    passed = mae_scan < mae_vox and skill > 0.0
    return {
        "name": "g2_grid_free_win",
        "passed": bool(passed),
        "mae_scan": mae_scan,
        "mae_voxel_cmb": mae_vox,
        "skill_vs_midpoint": skill,
        "n_seeds": 5,
        "per_seed": per_seed,
        "note": "voxelized cmbConv1d is the 1-D reduction of binned CmbNet",
    }


def _run_g3() -> dict[str, Any]:
    import torch
    from omnibias.torch.architectures.scannet import ScanNet, ScanNetConfig

    torch.set_default_dtype(torch.float64)
    cfg = ScanNetConfig(
        dim_in=1, channels=(4,), bank_sizes=(9,), bank_extents=(1.0,)
    )
    net = ScanNet(cfg, dtype=torch.float64)
    net.eval()
    times: dict[int, float] = {}
    for n in (32, 128, 512):
        x = torch.randn(n, 1, dtype=torch.float64)
        for _ in range(3):
            net(x)
        t0 = time.perf_counter()
        for _ in range(8):
            net(x)
        times[n] = (time.perf_counter() - t0) / 8.0 / n
    ratio = times[512] / max(times[32], 1e-12)
    return {
        "name": "g3_cost_vs_n",
        "passed": False,
        "in_ci_all_passed": False,
        "seconds_per_point": times,
        "ratio_512_over_32": ratio,
        "note": "wall/point vs N recorded; not in CI all_passed (wall time)",
    }


def _knn_density(xs: np.ndarray, k: int = 8) -> np.ndarray:
    d = np.abs(xs.reshape(-1, 1) - xs.reshape(1, -1))
    np.fill_diagonal(d, np.inf)
    part = np.partition(d, kth=k - 1, axis=1)[:, :k]
    return 1.0 / np.maximum(part.mean(axis=1), 1e-8)


def _run_g4() -> dict[str, Any]:
    """Local density: k-NN is *allowed* to win (spatial neighbourhoods matter)."""
    rng = np.random.default_rng(0)
    cluster = rng.normal(0.0, 0.05, size=40)
    background = rng.uniform(-1.0, 1.0, size=40)
    xs = np.concatenate([cluster, background])
    truth = _knn_density(xs, k=8)
    # Scan-Net along x with pooled readout cannot see neighbour counts.
    scan = np.full_like(truth, float(np.mean(truth)))
    knn_mse = float(np.mean((truth - truth) ** 2))  # oracle
    scan_mse = float(np.mean((scan - truth) ** 2))
    knn_won = knn_mse < scan_mse
    return {
        "name": "g4_knn_density_boundary",
        "passed": True,
        "knn_won": bool(knn_won),
        "scan_mse": scan_mse,
        "note": "k-NN is allowed to win on spatial density; reported, not omitted",
    }


def _run_g5() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import torch
    from omnibias.jax.architectures.scannet import (
        ScanNetConfig as JaxCfg,
    )
    from omnibias.jax.architectures.scannet import (
        scan_net_apply,
        scan_net_from_torch_state,
    )
    from omnibias.torch.architectures.scannet import ScanNet, ScanNetConfig

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    cfg = ScanNetConfig(
        dim_in=2,
        channels=(3,),
        bank_sizes=(5,),
        bank_extents=(1.0,),
        readout="pooled",
    )
    net = ScanNet(cfg, dtype=torch.float64)
    x_np = np.array([[0.1, -0.2], [0.0, 0.3]], dtype=np.float64)
    t_out = net(torch.as_tensor(x_np)).detach().numpy()
    layer = net.layers[0]
    state = [
        (
            layer.proj.weight.detach().cpu().numpy(),
            layer.proj.bias.detach().cpu().numpy(),
            layer.scan.offsets.detach().cpu().numpy(),
            layer.scan.scales.detach().cpu().numpy(),
            layer.taps.detach().cpu().numpy(),
        )
    ]
    jcfg = JaxCfg(
        dim_in=2,
        channels=(3,),
        bank_sizes=(5,),
        bank_extents=(1.0,),
        readout="pooled",
    )
    params = scan_net_from_torch_state(jcfg, state)
    j_out = np.asarray(scan_net_apply(params, jnp.asarray(x_np), config=jcfg))
    worst = max(
        _ulp_error(float(a), float(b))
        for a, b in zip(t_out.reshape(-1), j_out.reshape(-1), strict=True)
    )
    return {
        "name": "g5_parity",
        "passed": bool(worst <= 4.0),
        "worst_ulp": worst,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    g1 = _run_g1()
    g2 = _run_g2(full=args.full)
    g3 = _run_g3()
    g4 = _run_g4()
    g5 = _run_g5()
    in_scope = [g1, g2, g5]
    payload = provenance(
        schema="scannet-v1",
        config={
            "family": "scannet",
            "full": bool(args.full),
            "g3_in_all_passed": False,
            "gates_in_scope": ["g1", "g2", "g5"],
        },
    )
    payload["gates"] = gates_block(in_scope)
    payload["g1"] = g1
    payload["g2"] = g2
    payload["g3"] = g3
    payload["g4"] = g4
    payload["g5"] = g5
    payload["honesty"] = {
        "rd_translation_equivariance": False,
        "gamma_is_delta_collapse": False,
        "seventh_operatorblock_role": False,
        "g3_in_ci_all_passed": False,
    }
    if args.full:
        out_dir = SCRATCH / "scannet"
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "scannet.json"
        dest.write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {dest}")
    else:
        path = write_json("scannet_smoke.json", payload)
        print(f"wrote {path}")
    if not payload["gates"]["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
