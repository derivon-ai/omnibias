# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-1 primitive: bias scan (theory 01-02 G1/G2/G3; G4 attempted).

Smoke (default) earns interior-shift equivariance, 5-seed localization, and
torch/jax parity. The two-interface soft-argmax bias is recorded, not gated
as a win. G4 (point-cloud vs voxelized ``cmbConv1d``) is attempted; if the
scan does not win on MAE at equal-or-lower wall time, ``g4_earned`` stays
false -- the threshold is not moved. ``--full`` writes under
``$OMNIBIAS_SCRATCH/scan/``.
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
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) / (np.finfo(np.float64).eps * scale)


def _run_g1() -> dict[str, Any]:
    import torch
    from omnibias.core.scan import BankSpec
    from omnibias.torch.scan import BiasScan

    torch.set_default_dtype(torch.float64)
    bank = BankSpec.uniform(-1.0, 1.0, 5)
    spacing = bank.spacing
    assert spacing is not None
    scan = BiasScan(
        1,
        bank,
        template="grad",
        base="tanh",
        learnable_offsets=False,
        learnable_scales=False,
        readout="response",
        dtype=torch.float64,
    )
    worst = 0.0
    for z0 in np.linspace(-0.4, 0.4, 9):
        z = torch.tensor([[float(z0)]], dtype=torch.float64)
        r0 = scan(z)
        r_shift = scan(z + spacing)
        left = r_shift[..., :-1].reshape(-1).tolist()
        right = r0[..., 1:].reshape(-1).tolist()
        for a, b in zip(left, right, strict=True):
            worst = max(worst, _ulp_error(float(a), float(b)))
    return {
        "name": "g1_interior_equivariance",
        "passed": bool(worst <= 4.0),
        "expected": 4.0,
        "worst_ulp": float(worst),
        "note": "interior shift R(z+Delta)[..., :-1] vs R(z)[..., 1:]; not a circular wrap",
    }


def _run_g2(*, full: bool) -> dict[str, Any]:
    import torch
    from omnibias.core.scan import BankSpec
    from omnibias.torch.activations.registry import get_activation
    from omnibias.torch.scan import scan_response, soft_argmax_offset, template_from_op

    torch.set_default_dtype(torch.float64)
    bank = BankSpec.uniform(-1.0, 1.0, 9)
    spacing = bank.spacing
    assert spacing is not None
    offsets = torch.tensor(bank.offsets, dtype=torch.float64)
    scales = torch.tensor(bank.scales, dtype=torch.float64)
    spec = template_from_op("grad")
    base = get_activation("tanh")
    n_per_seed = 32 if full else 8
    seeds = list(range(5))
    true_all: list[float] = []
    pred_all: list[float] = []
    per_seed: list[dict[str, float]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        zs = rng.uniform(-0.4, 0.4, size=n_per_seed)
        preds: list[float] = []
        trues: list[float] = []
        for z0 in zs:
            z = torch.tensor([[float(z0)]], dtype=torch.float64)
            resp = scan_response(z, offsets, scales, spec, base).reshape(-1)
            amp = float(resp.abs().max())
            noise = 0.01 * amp * torch.as_tensor(
                rng.standard_normal(resp.numel()), dtype=torch.float64
            )
            noisy = resp + noise
            tau_star = float(soft_argmax_offset(noisy, offsets, gamma=8.0))
            preds.append(tau_star)
            trues.append(-float(z0))
        mae = float(np.mean(np.abs(np.asarray(preds) - np.asarray(trues))))
        mid = 0.5 * (float(bank.offsets[0]) + float(bank.offsets[-1]))
        mse_pred = float(np.mean((np.asarray(preds) - np.asarray(trues)) ** 2))
        mse_mid = float(np.mean((mid - np.asarray(trues)) ** 2))
        skill = 1.0 - mse_pred / mse_mid if mse_mid > 0.0 else 0.0
        per_seed.append({"mae": mae, "skill": skill})
        pred_all.extend(preds)
        true_all.extend(trues)
    mae = float(np.mean(np.abs(np.asarray(pred_all) - np.asarray(true_all))))
    mid = 0.5 * (float(bank.offsets[0]) + float(bank.offsets[-1]))
    mse_pred = float(np.mean((np.asarray(pred_all) - np.asarray(true_all)) ** 2))
    mse_mid = float(np.mean((mid - np.asarray(true_all)) ** 2))
    skill = 1.0 - mse_pred / mse_mid
    passed = mae <= 0.1 * spacing and skill > 0.0
    return {
        "name": "g2_localization",
        "passed": bool(passed),
        "mae": mae,
        "mae_limit": 0.1 * spacing,
        "spacing": float(spacing),
        "skill_vs_midpoint": skill,
        "n_seeds": 5,
        "n_per_seed": n_per_seed,
        "noise_frac": 0.01,
        "per_seed": per_seed,
    }


def _run_g3() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import torch
    from omnibias.core.scan import BankSpec
    from omnibias.jax.scan import bias_scan, init_bias_scan
    from omnibias.torch.scan import BiasScan

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    bank = BankSpec.uniform(-1.0, 1.0, 5)
    unit = BiasScan(
        2,
        bank,
        template="grad",
        base="tanh",
        learnable_offsets=False,
        learnable_scales=False,
        readout="response",
        dtype=torch.float64,
    )
    z_np = np.array([[0.0, 0.0], [0.3, -0.25], [0.5, 0.5]], dtype=np.float64)
    torch_out = unit(torch.as_tensor(z_np)).detach().numpy()
    act, offsets, scales, tmpl, _taps = init_bias_scan(2, bank, template="grad", base="tanh")
    jax_out = np.asarray(
        bias_scan(jnp.asarray(z_np), offsets, scales, tmpl, act, readout="response")
    )
    worst = 0.0
    for a, b in zip(torch_out.reshape(-1), jax_out.reshape(-1), strict=True):
        worst = max(worst, _ulp_error(float(a), float(b)))
    return {
        "name": "g3_parity",
        "passed": bool(worst <= 4.0),
        "worst_ulp": float(worst),
        "max_abs_diff": float(np.max(np.abs(torch_out - jax_out))),
    }


def _two_interface_diagnostic() -> dict[str, Any]:
    import torch
    from omnibias.core.scan import BankSpec
    from omnibias.torch.activations.registry import get_activation
    from omnibias.torch.scan import scan_response, soft_argmax_offset, template_from_op

    torch.set_default_dtype(torch.float64)
    bank = BankSpec((-1.0, -0.5, 0.0, 0.5, 1.0))
    offsets = torch.tensor(bank.offsets, dtype=torch.float64)
    scales = torch.tensor(bank.scales, dtype=torch.float64)
    spec = template_from_op("grad")
    base = get_activation("tanh")
    r1 = scan_response(torch.tensor([[0.4]], dtype=torch.float64), offsets, scales, spec, base)
    r2 = scan_response(torch.tensor([[-0.4]], dtype=torch.float64), offsets, scales, spec, base)
    tau_star = float(soft_argmax_offset((r1 + r2).squeeze(), offsets, gamma=8.0))
    return {
        "tau_star": tau_star,
        "peaks": [-0.4, 0.4],
        "spacing": 0.5,
        "biased_to_zero": bool(abs(tau_star) < 0.15),
        "note": (
            "soft-argmax of two peaks one spacing apart collapses toward 0; "
            "recorded as a visible failure, not gated as a win. "
            "gamma is temperature-collapse sharpness, not delta->0."
        ),
    }


def _run_g4(*, full: bool) -> dict[str, Any]:
    """Point-cloud interface vs voxelized cmbConv1d on a 1-D histogram of w·x."""
    import torch
    from omnibias.core.scan import BankSpec
    from omnibias.torch.activations.registry import get_activation
    from omnibias.torch.blocks import cmbConv1d
    from omnibias.torch.scan import scan_response, soft_argmax_offset, template_from_op

    torch.set_default_dtype(torch.float64)
    n_pts = 64 if full else 24
    n_bins = 32 if full else 16
    rng = np.random.default_rng(1)
    w = np.array([1.0, 0.0], dtype=np.float64)
    true_z = 0.25
    # Cluster along the interface line {x : w·x = true_z}.
    along = rng.normal(true_z, 0.03, size=n_pts)
    perp = rng.uniform(-0.5, 0.5, size=n_pts)
    points = np.stack([along, perp], axis=1)
    z = points @ w

    bank = BankSpec.uniform(-1.0, 1.0, 9)
    offsets = torch.tensor(bank.offsets, dtype=torch.float64)
    scales = torch.tensor(bank.scales, dtype=torch.float64)
    spec = template_from_op("grad")
    base = get_activation("tanh")
    z_t = torch.as_tensor(z, dtype=torch.float64).reshape(-1, 1)

    def _scan_once() -> float:
        resp = scan_response(z_t, offsets, scales, spec, base)
        pooled = resp.mean(dim=0).reshape(-1)
        return float(soft_argmax_offset(pooled, offsets, gamma=8.0))

    t0 = time.perf_counter()
    tau_scan = _scan_once()
    scan_s = time.perf_counter() - t0
    scan_mae = abs(tau_scan + true_z)

    lo, hi = -1.0, 1.0
    hist, edges = np.histogram(z, bins=n_bins, range=(lo, hi))
    x = torch.as_tensor(hist, dtype=torch.float64).view(1, 1, n_bins)
    layer = cmbConv1d(1, 1, kernel_size=3, padding=1, op="identity", base="tanh", bias=False)
    with torch.no_grad():
        layer.conv.weight.zero_()
        layer.conv.weight[0, 0, 1] = 1.0

    def _conv_once() -> float:
        out = layer(x).reshape(-1)
        peak = int(torch.argmax(out).item())
        return float(0.5 * (edges[peak] + edges[peak + 1]))

    t1 = time.perf_counter()
    tau_conv = _conv_once()
    conv_s = time.perf_counter() - t1
    conv_mae = abs(tau_conv - true_z)
    earned = bool(scan_mae < conv_mae and scan_s <= conv_s)
    return {
        "name": "g4_no_grid_win",
        "passed": bool(earned),
        "earned": bool(earned),
        "scan_mae": float(scan_mae),
        "conv_mae": float(conv_mae),
        "scan_wall_seconds": float(scan_s),
        "conv_wall_seconds": float(conv_s),
        "n_points": n_pts,
        "n_bins": n_bins,
        "true_z": true_z,
        "scan_tau_star": tau_scan,
        "conv_peak_z": tau_conv,
        "note": (
            "scan localizes -z of the cluster; voxelized cmbConv1d "
            "(identity 3-tap on the 1-D histogram of w.x) peaks on a bin "
            "center. Thresholds not moved if unearned."
        ),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "bias_scan.json" if full else "bias_scan_smoke.json"
    t0 = time.perf_counter()
    print("G1 interior equivariance...")
    g1 = _run_g1()
    print("G2 5-seed localization...")
    g2 = _run_g2(full=full)
    print("G3 torch/jax parity...")
    g3 = _run_g3()
    print("two-interface diagnostic...")
    two = _two_interface_diagnostic()
    print("G4 no-grid attempt...")
    g4 = _run_g4(full=full)
    entries = [g1, g2, g3]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    gates = dict(gates_block(entries))
    g4_earned = bool(g4["earned"])
    config = {
        "family": "bias_scan",
        "full": full,
        "g4_earned": g4_earned,
        "gates_in_scope": ["g1", "g2", "g3"],
    }
    payload = provenance(schema="bias-scan-v1", config=config)
    payload.update(
        {
            "gates": gates,
            "g1": g1,
            "g2": g2,
            "g3": g3,
            "g4": g4,
            "two_interface": two,
            "honesty": {
                "claim_rung": 1,
                "bias_collapse": True,
                "temperature_collapse": False,
                "gamma_is_not_delta_collapse": True,
                "equivariance_is_interior_shift": True,
                "g1_earned": bool(g1["passed"]),
                "g2_earned": bool(g2["passed"]),
                "g3_earned": bool(g3["passed"]),
                "g4_earned": g4_earned,
                "two_interface_visible_failure": bool(two["biased_to_zero"]),
                "licensed_sentence": (
                    "BiasScan interior-shifts the response to <= 4 ulp on the "
                    "bank lattice; soft-argmax localizes a noisy interface to "
                    "<= 0.1 spacing with skill vs the domain midpoint; "
                    "torch/jax responses agree within 4 ulp; two-interface "
                    "soft-argmax bias is visible; G4 is earned only on a "
                    "measured no-grid MAE win at equal-or-lower wall time"
                ),
            },
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }
    )
    out = write_json(artifact, payload)
    print(
        f"wrote {out} all_passed={gates['all_passed']} "
        f"g4_earned={g4_earned} g2_mae={g2['mae']:.4f}"
    )
    if full:
        scratch = SCRATCH / "scan"
        scratch.mkdir(parents=True, exist_ok=True)
        (scratch / artifact).write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"copied to {scratch / artifact}")
    if not gates["all_passed"]:
        raise SystemExit(1)
    return dict(payload)


if __name__ == "__main__":
    main()
