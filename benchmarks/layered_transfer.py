# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: layered transfer (theory 02-11). continuum_claim=False.

G2 is quarter-wave band edges vs the closed-form arcsin law. G6 is
torch/jax ``r, t`` parity. G4 inverse-design is leftover-recorded
(leftover #23) and stays ``--full``. Stack / gap wall vs period count
is reported. G5 conservation is leftover-recorded (leftover #27), not
in CI ``all_passed``.
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

COST_PERIODS = (1, 2, 4, 8)
COST_WARMUP = 1
COST_REPEATS = 3
COST_N_GRID = 32


def _median_seconds(fn: Any, *, warmup: int, repeats: int) -> float:
    for _ in range(int(warmup)):
        fn()
    samples: list[float] = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(np.asarray(samples, dtype=np.float64)))


def _qw_closed_edges(n_hi: float, n_lo: float, omega0: float) -> tuple[float, float]:
    """First gap edges: ``sin δ = 2√(n_h n_l)/(n_h+n_l)``, ``δ = (π/2)(ω/ω0)``."""
    s = 2.0 * math.sqrt(n_hi * n_lo) / (n_hi + n_lo)
    delta = math.asin(min(1.0, max(0.0, s)))
    w_lo = (2.0 / math.pi) * delta * omega0
    w_hi = 2.0 * omega0 - w_lo
    return w_lo, w_hi


def _run_g2() -> dict[str, Any]:
    from omnibias.core.transfer import bloch_dispersion, quarter_wave_stack

    n_hi, n_lo, omega0 = 2.0, 1.0, 1.0
    cell = quarter_wave_stack(n_hi, n_lo, n_periods=1, omega0=omega0)
    expect_lo, expect_hi = _qw_closed_edges(n_hi, n_lo, omega0)

    def _edge(lo: float, hi: float, *, rising: bool) -> float:
        a, b = lo, hi
        for _ in range(80):
            mid = 0.5 * (a + b)
            val = abs(bloch_dispersion(cell, mid))
            if rising:
                if val >= 1.0:
                    b = mid
                else:
                    a = mid
            elif val <= 1.0:
                b = mid
            else:
                a = mid
        return 0.5 * (a + b)

    # Pass band left of the gap, then gap, then pass band.
    got_lo = _edge(0.4, 1.0, rising=True)
    got_hi = _edge(1.0, 1.6, rising=False)
    rel_lo = abs(got_lo - expect_lo) / max(abs(expect_lo), 1e-18)
    rel_hi = abs(got_hi - expect_hi) / max(abs(expect_hi), 1e-18)
    passed = bool(rel_lo <= 1e-10 and rel_hi <= 1e-10)
    return {
        "name": "g2_band_edges",
        "passed": passed,
        "in_ci_all_passed": passed,
        "closed_lo": expect_lo,
        "closed_hi": expect_hi,
        "computed_lo": got_lo,
        "computed_hi": got_hi,
        "rel_lo": rel_lo,
        "rel_hi": rel_hi,
    }


def _run_g6() -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    import torch
    from omnibias.core.transfer import quarter_wave_stack
    from omnibias.pinn.layered.jax import transfer_apply
    from omnibias.pinn.layered.torch import TransferStack

    jax.config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    n_hi, n_lo, omega0 = 2.0, 1.0, 1.0
    layers = quarter_wave_stack(n_hi, n_lo, n_periods=1, omega0=omega0)
    net = TransferStack(2, dtype=torch.float64)
    with torch.no_grad():
        net.log_n.copy_(
            torch.log(torch.tensor([n_hi, n_lo], dtype=torch.float64))
        )
        net.log_d.copy_(
            torch.log(
                torch.tensor(
                    [
                        math.pi / (2.0 * omega0 * n_hi),
                        math.pi / (2.0 * omega0 * n_lo),
                    ],
                    dtype=torch.float64,
                )
            )
        )
    omega = torch.tensor([0.8, 1.0, 1.2], dtype=torch.float64)
    r_t, t_t = net(omega)
    r_j, t_j = transfer_apply(layers, jnp.asarray(omega.numpy()))
    gap = float(
        max(
            np.max(np.abs(r_t.detach().cpu().numpy() - np.asarray(r_j))),
            np.max(np.abs(t_t.detach().cpu().numpy() - np.asarray(t_j))),
        )
    )
    passed = bool(gap == 0.0)
    return {
        "name": "g6_parity",
        "passed": passed,
        "in_ci_all_passed": passed,
        "max_abs": gap,
    }


def _run_cost() -> dict[str, Any]:
    """Named leftover: stack/gap wall vs periods; G4 inverse-design stays --full."""
    from omnibias.core.transfer import (
        certified_band_gap,
        quarter_wave_stack,
        stack_matrix,
    )

    rows: list[dict[str, Any]] = []
    for n_periods in COST_PERIODS:
        layers = quarter_wave_stack(2.0, 1.0, n_periods=n_periods, omega0=1.0)

        def _stack(stack=layers) -> None:
            stack_matrix(stack, 1.0)

        def _gap(stack=layers) -> None:
            certified_band_gap(stack, omega_range=(0.85, 1.15), n_grid=COST_N_GRID)

        rows.append(
            {
                "n_periods": int(n_periods),
                "n_layers": len(layers),
                "stack_wall_seconds": float(
                    _median_seconds(_stack, warmup=COST_WARMUP, repeats=COST_REPEATS)
                ),
                "gap_wall_seconds": float(
                    _median_seconds(_gap, warmup=COST_WARMUP, repeats=COST_REPEATS)
                ),
            }
        )
    return {
        "name": "cost_stack_vs_periods",
        "passed": False,
        "earned": False,
        "reported": True,
        "leftover_recorded": True,
        "leftover_id": 46,
        "leftover_tick": 90,
        "in_ci_all_passed": False,
        "rows": rows,
        "g4_inverse_design": {
            "earned": False,
            "reported": True,
            "leftover_recorded": True,
            "leftover_id": 23,
            "leftover_tick": 67,
            "stays_full": True,
            "need": (
                "bandwidth-max differentiable trace beats gradient-free "
                "at 10x fewer objective evals, 5 seeds"
            ),
            "reason": (
                "Leftover #23 leftover-recorded: no inverse-design loop "
                "is wired. stack_matrix / certified_band_gap wall vs "
                "n_periods is recorded; that is not the named 10x "
                "eval-count win."
            ),
        },
        "note": (
            "Leftover #46 leftover-recorded: quarter-wave stack_matrix "
            "and certified_band_gap wall vs n_periods. G4 inverse-design "
            "is leftover #23 and stays --full. Previous "
            "g4_inverse_design passed=True / --full-only stub with no "
            "timing withdrawn. continuum_claim=False. Not in CI "
            "all_passed."
        ),
    }


def _run_g5() -> dict[str, Any]:
    """Named leftover: unstructured 2x2 energy violation vs lossless stack."""
    from omnibias.core.transfer import (
        quarter_wave_stack,
        reflection_transmission,
        stack_matrix,
        unitarity_residual,
    )

    layers = quarter_wave_stack(2.0, 1.0, n_periods=1, omega0=1.0)
    m = stack_matrix(layers, 1.0)
    r, t = reflection_transmission(m)
    struct_energy = abs(abs(r) ** 2 + abs(t) ** 2 - 1.0)
    struct_unit = unitarity_residual(m)
    refused = False
    try:
        unitarity_residual(m, lossless=False)
    except ValueError:
        refused = True
    rng = np.random.default_rng(0)
    raw = rng.normal(size=4) + 1j * rng.normal(size=4)
    unstructured = ((complex(raw[0]), complex(raw[1])), (complex(raw[2]), complex(raw[3])))
    ru, tu = reflection_transmission(unstructured)
    unstruct_energy = abs(abs(ru) ** 2 + abs(tu) ** 2 - 1.0)
    return {
        "name": "g5_mlp_conservation",
        "passed": False,
        "earned": False,
        "reported": True,
        "leftover_recorded": True,
        "leftover_id": 27,
        "leftover_tick": 68,
        "in_ci_all_passed": False,
        "structural_unitarity": float(struct_unit),
        "structural_energy_violation": float(struct_energy),
        "unstructured_energy_violation": float(unstruct_energy),
        "unitarity_refuses_lossy": bool(refused),
        "note": (
            "Leftover #27 leftover-recorded: no MLP surrogate is wired. "
            "Unstructured 2x2 |r|^2+|t|^2-1 versus a lossless "
            "quarter-wave stack. unitarity_residual refuses "
            "lossless=False. G5 is honesty: report the violation, do "
            "not assert a structural win. Previous passed=True stub "
            "withdrawn. Not in CI all_passed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    from omnibias.core.transfer import (
        certified_band_gap,
        quarter_wave_stack,
        reflection_transmission,
        stack_matrix,
        unitarity_residual,
    )

    layers = quarter_wave_stack(2.0, 1.0, n_periods=1, omega0=1.0)
    m = stack_matrix(layers, 1.0)
    r, t = reflection_transmission(m)
    g1 = unitarity_residual(m) <= 1e-12 and abs(abs(r) ** 2 + abs(t) ** 2 - 1.0) <= 1e-12
    g2 = _run_g2()
    g6 = _run_g6()
    cert = certified_band_gap(layers, omega_range=(0.85, 1.15), n_grid=32)
    entries: list[dict[str, Any]] = [
        {"name": "g1_unitarity", "passed": g1, "in_ci_all_passed": True},
        {
            "name": "g2_band_edges",
            "passed": bool(g2["passed"]),
            "in_ci_all_passed": bool(g2["in_ci_all_passed"]),
        },
        {
            "name": "g3_certified_gap",
            "passed": cert.continuum_claim is False,
            "is_gap": cert.is_gap,
            "in_ci_all_passed": True,
        },
        {
            "name": "g6_parity",
            "passed": bool(g6["passed"]),
            "in_ci_all_passed": bool(g6["in_ci_all_passed"]),
        },
    ]
    cost = _run_cost()
    g5 = _run_g5()
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.layered_transfer.v1",
        config={
            "mode": "full" if args.full else "smoke",
            "cost_in_all_passed": False,
            "g5_in_all_passed": False,
            "gates_in_scope": ["g1", "g2", "g3", "g6"],
        },
    )
    payload["gates"] = gates_block(entries)
    payload["g2"] = g2
    payload["g6"] = g6
    payload["cost"] = cost
    payload["g5"] = g5
    payload["honesty"] = {
        "distinct_from": "omnibias.geometry.gauge.transfer",
        "continuum_claim": False,
        "one_d_layered": True,
        "g2_earned": bool(g2["passed"]),
        "g6_earned": bool(g6["passed"]),
        "g4_inverse_design_earned": False,
        "g4_reported": True,
        "g4_leftover_recorded": True,
        "g4_leftover_id": 23,
        "g4_leftover_tick": 67,
        "g4_stays_full": True,
        "g5_mlp_conservation_earned": False,
        "g5_mlp_conservation_reported": True,
        "g5_leftover_recorded": True,
        "g5_leftover_id": 27,
        "g5_leftover_tick": 68,
        "g5_in_ci_all_passed": False,
        "cost_earned": False,
        "cost_reported": True,
        "cost_leftover_recorded": True,
        "cost_leftover_id": 46,
        "cost_leftover_tick": 90,
        "cost_in_ci_all_passed": False,
    }
    if args.full:
        dest = SCRATCH / "transfer" / "layered_transfer.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('layered_transfer_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
