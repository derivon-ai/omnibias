# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated primitive: OMBU frames (theory 01-06). G4 denoising is unearned.

G1/G2 are CI-gated. G3 lives in ``test_frames.py``. G4 is recorded and
**not** in CI ``all_passed``: the order-2 bank beats a matched-cost
order-1 Gaussian-derivative bank in MSE, but skill versus the noisy
identity is negative on all five seeds. ``sigma'`` is not admissible;
frames are not orthonormal and not compactly supported.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _g4_denoising() -> dict[str, Any]:
    """Order-2 bank vs matched-cost n=1; skill vs identity. Not CI-gated."""
    import numpy as np
    from omnibias.core.frames import dilated_sigma_n

    xs = np.linspace(-4.0, 4.0, 256)
    trend = 0.3 * xs**2
    bump = np.exp(-((xs - 0.7) ** 2) / (2 * 0.05**2))
    signal = trend + bump
    # Order-2 gaussian bank vs matched-cost gaussian-derivative (n=1, not admissible).
    scales = (0.4, 0.8, 1.6)
    offsets = np.linspace(-3.0, 3.0, 17)
    seeds = (0, 1, 2, 3, 4)

    def recon(order: int, y: np.ndarray) -> np.ndarray:
        cols = []
        for a in scales:
            for b in offsets:
                cols.append(
                    np.array(
                        [dilated_sigma_n("gaussian", float(u - b), order, a) for u in xs]
                    )
                )
        # Polynomials of degree < 2 (annihilated by admissible n=2 atoms).
        cols.append(np.ones_like(xs))
        cols.append(xs)
        design = np.stack(cols, axis=1)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        return design @ coef

    rows: list[dict[str, float]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        y = signal + 0.05 * rng.normal(size=xs.shape)
        pred_n2 = recon(2, y)
        pred_n1 = recon(1, y)
        mse_n2 = float(np.mean((pred_n2 - signal) ** 2))
        mse_n1 = float(np.mean((pred_n1 - signal) ** 2))
        mse_id = float(np.mean((y - signal) ** 2))
        skill = 1.0 - mse_n2 / max(mse_id, 1e-30)
        rows.append(
            {
                "seed": float(seed),
                "mse_order2": mse_n2,
                "mse_order1": mse_n1,
                "skill_vs_identity": skill,
                "beats_n1": float(mse_n2 < mse_n1),
                "skill_positive": float(skill > 0.0),
            }
        )
    beats = sum(1 for r in rows if r["beats_n1"] >= 1.0)
    skill_wins = sum(1 for r in rows if r["skill_positive"] >= 1.0)
    earned = beats == len(rows) and skill_wins == len(rows)
    return {
        "name": "g4_denoising",
        "passed": False,
        "earned": False,
        "reported": True,
        "mse_order2": float(np.median([r["mse_order2"] for r in rows])),
        "mse_order1": float(np.median([r["mse_order1"] for r in rows])),
        "beats_n1": beats == len(rows),
        "beats_n1_wins": int(beats),
        "skill_vs_identity": float(np.median([r["skill_vs_identity"] for r in rows])),
        "skill_positive_wins": int(skill_wins),
        "n_seeds": len(rows),
        "per_seed": rows,
        "in_ci_all_passed": False,
        "note": (
            "order-2 beats matched-cost n=1 in MSE on all five seeds, but "
            "skill versus the noisy identity is negative on all five "
            f"(median {float(np.median([r['skill_vs_identity'] for r in rows])):.3f}). "
            "Named G4 needs both. sigma' is not admissible. Not orthonormal, "
            "not compactly supported. Not in CI all_passed."
        ),
        "both_gates": bool(earned),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    from omnibias.core.frames import FrameSpec, admissibility_constant, littlewood_paley_bounds
    from omnibias.core.spectral_design import hat_sigma_magnitude

    c = admissibility_constant("gaussian", 2)
    g1 = {
        "name": "g1_admissibility",
        "passed": c is not None and abs(c.mid - math.pi) / math.pi <= 1e-12,
        "c": None if c is None else c.mid,
        "n1_is_none": admissibility_constant("gaussian", 1) is None,
        "in_ci_all_passed": True,
    }
    spec = FrameSpec("gaussian", 2, scales=(0.5, 1.0, 2.0), offset_spacing=0.25)
    a, b = littlewood_paley_bounds(spec, grid=512)
    # Tails of a finite scale bank make the LP sum ~0; A=0 there is sound.
    xi_mid = 1.0
    acc_mid = 0.0
    for s in spec.scales:
        mag = (abs(s * xi_mid) ** spec.order) * math.sqrt(s) * hat_sigma_magnitude(
            spec.base, s * xi_mid
        )
        acc_mid += mag * mag
    g2 = {
        "name": "g2_lp_bounds",
        "passed": (
            a.lo >= 0.0
            and b.hi >= a.lo
            and a.lo - 1e-12 <= acc_mid <= b.hi + 1e-12
            and acc_mid > 0.0
        ),
        "A": a.lo,
        "B": b.hi,
        "mid_band": acc_mid,
        "in_ci_all_passed": True,
    }
    g4 = _g4_denoising()
    ci = [g1, g2]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.ombu_frames.v1",
        config={"mode": "full" if args.full else "smoke"},
    )
    payload["gates"] = gates_block(ci)
    payload["g4"] = g4
    payload["honesty"] = {
        "sigma_prime_admissible": False,
        "orthonormal": False,
        "compact_support": False,
        "fast_transform": False,
        "littlewood_paley_completeness": False,
        "g4_earned": False,
        "g4_reported": True,
        "g4_in_ci_all_passed": False,
    }
    if args.full:
        dest = SCRATCH / "frames"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "ombu_frames.json").write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        write_json("ombu_frames_smoke.json", payload)
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
