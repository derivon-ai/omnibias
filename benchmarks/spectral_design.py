# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3 primitive: spectral design (theory 01-07 G1/G2/G4; G3 unearned).

Smoke earns formula peak correctness, a transfer-magnitude check of ``R``
against a DFT of the sampled kernel, and hole detection. G3 (2x fewer steps
vs MscaleMLP on the spectral-bias arm) is recorded unearned and is **not**
in CI ``all_passed``: the named lstsq-level gate is not reached by either
the geometric or a band-planned Mscale in the CI step budget. Does not
mutate ``spectral_bias_fbpinn`` gates. Pack order is a band selector, not
Littlewood-Paley completeness.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _run_g1() -> dict[str, Any]:
    from omnibias.core.spectral_design import locate_peak_numerically, peak_frequency

    worst = 0.0
    n_ok = 0
    n = 0
    for base in ("gaussian", "sech"):
        for order in range(1, 9):
            for alpha in (0.7, 1.0, 2.5):
                pred = peak_frequency(base, order, alpha)
                found = locate_peak_numerically(base, order, alpha)
                rel = abs(pred - found) / pred
                worst = max(worst, rel)
                n += 1
                if rel <= 1e-6:
                    n_ok += 1
    return {
        "name": "g1_peak_frequency",
        "passed": n_ok == n,
        "worst_rel": worst,
        "expected": 1e-6,
        "n": n,
    }


def _spatial_kernel(base: str, order: int, alpha: float, x: np.ndarray) -> np.ndarray:
    from omnibias.core.polynomials import hermite_coeffs, sech_polynomial_coeffs

    u = alpha * x
    if base == "gaussian":
        coeffs = hermite_coeffs(order)
        he = np.zeros_like(u)
        for k, c in enumerate(coeffs):
            he = he + float(c) * (u**k)
        g = np.exp(-0.5 * u * u)
        # g^{(n)} = (-1)^n He_n g
        return ((-1.0) ** order) * he * g
    coeffs = sech_polynomial_coeffs(order)
    t = np.tanh(u)
    q = np.zeros_like(u)
    for k, c in enumerate(coeffs):
        q = q + float(c) * (t**k)
    return q / np.cosh(u)


def _run_g2() -> dict[str, Any]:
    """DFT of sampled ``sigma^{(n)}`` vs ``R_{n,1}`` near the peak (alpha=1)."""
    from omnibias.core.spectral_design import peak_frequency, response_profile

    n_pts = 4096
    half = 40.0
    x = np.linspace(-half, half, n_pts, endpoint=False)
    dx = float(x[1] - x[0])
    xi = 2.0 * math.pi * np.fft.fftfreq(n_pts, d=dx)
    worst = 0.0
    n_ok = 0
    n = 0
    for base in ("gaussian", "sech"):
        for order in (1, 2, 3):
            kernel = _spatial_kernel(base, order, 1.0, x)
            hat = np.fft.fft(kernel) * dx
            mag = np.abs(hat)
            peak = peak_frequency(base, order, 1.0)
            band = (xi > 0.6 * peak) & (xi < 1.4 * peak)
            r = np.array(response_profile(base, order, 1.0, xi.tolist()), dtype=float)
            # Spec R at alpha=1 is |xi|^n |hat_sigma|; DFT of sigma^{(n)}
            # matches that up to the (i xi)^n phase. Compare on the positive band.
            rels = np.abs(mag[band] - r[band]) / np.maximum(r[band], 1e-30)
            rel = float(np.median(rels))
            worst = max(worst, rel)
            n += 1
            if rel <= 0.02:
                n_ok += 1
    return {
        "name": "g2_transfer_magnitude",
        "passed": n_ok == n,
        "worst_median_rel": worst,
        "expected": 0.02,
        "n": n,
        "note": "median relative |DFT - R| on a neighbourhood of xi_peak; alpha=1",
    }


def _run_g4() -> dict[str, Any]:
    from omnibias.core.spectral_design import band_plan_from_peaks, design_band_plan

    dense = design_band_plan("sech", xi_lo=1.0, xi_hi=32.0, channels=4, order=2)
    hole = band_plan_from_peaks(
        "sech", peaks=(2.0, 16.0), order=2, xi_lo=1.0, xi_hi=32.0
    )
    flagged = hole.has_spectral_hole()
    dense_ok = not dense.has_spectral_hole()
    return {
        "name": "g4_hole_detection",
        "passed": bool(flagged and dense_ok),
        "hole_flatness": hole.flatness,
        "dense_flatness": dense.flatness,
        "note": "01-06 wavelet frames stay concept; flatness is a hole diagnostic",
    }


def _run_g3() -> dict[str, Any]:
    """Record the named 2x-steps comparison; do not put it in ``all_passed``.

    The four-gap arm's absolute gate is lstsq-level (``rel L2 <= 1e-5`` on
    smoke). Geometric Mscale and a band-planned Mscale (peaks mapped to
    input stretches) are trained for the smoke step budget. The 2x ratio
    is defined only if both arms hit that gate.
    """
    import math

    import torch
    from omnibias.core.spectral_design import design_band_plan, peak_frequency
    from omnibias.pinn import ComponentSpec, CoordinateSpec
    from omnibias.pinn.torch.fields import MscaleVectorField

    torch.set_default_dtype(torch.float64)
    freq = 8
    n_grid = 64
    hidden = 16
    steps = 60
    lr = 1e-2
    gate = 1e-5
    seeds = (0, 1, 2, 3, 4)
    geometric = (1.0, 2.0, 4.0, 8.0)
    plan = design_band_plan(
        "sech", xi_lo=2.0 * math.pi, xi_hi=2.0 * math.pi * 16.0, channels=4, order=2
    )
    planned = tuple(
        peak_frequency("sech", n, a) / (2.0 * math.pi)
        for n, a in zip(plan.orders, plan.scales, strict=True)
    )
    cs = CoordinateSpec(("x",), domain=((0.0, 1.0),))
    comps = ComponentSpec(("u",))

    def _rel_after(scales: tuple[float, ...], seed: int) -> float:
        torch.manual_seed(seed)
        field = MscaleVectorField(
            coordinate_spec=cs,
            components=comps,
            hidden=hidden,
            depth=1,
            scales=scales,
            dtype=torch.float64,
        )
        x = torch.linspace(0.0, 1.0, n_grid, dtype=torch.float64).unsqueeze(-1)
        target = torch.sin(2.0 * math.pi * freq * x[:, 0])
        opt = torch.optim.Adam(field.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            pred = field.forward_values(x)[:, 0]
            torch.mean((pred - target) ** 2).backward()
            opt.step()
        with torch.no_grad():
            pred = field.forward_values(x)[:, 0]
            return float(
                torch.linalg.vector_norm(pred - target)
                / torch.linalg.vector_norm(target)
            )

    geo_rels = [_rel_after(geometric, seed) for seed in seeds]
    plan_rels = [_rel_after(planned, seed) for seed in seeds]
    geo_hits = sum(1 for r in geo_rels if r <= gate)
    plan_hits = sum(1 for r in plan_rels if r <= gate)
    earned = geo_hits == len(seeds) and plan_hits == len(seeds)
    return {
        "name": "g3_spectral_bias_steps",
        "passed": False,
        "earned": False,
        "in_ci_all_passed": False,
        "gate": gate,
        "steps": steps,
        "n_seeds": len(seeds),
        "geometric_hits": int(geo_hits),
        "planned_hits": int(plan_hits),
        "geometric_median_rel_l2": float(np.median(geo_rels)),
        "planned_median_rel_l2": float(np.median(plan_rels)),
        "both_reached_gate": bool(earned),
        "note": (
            "named 2x-fewer-steps comparison is undefined while neither "
            "geometric nor band-planned Mscale reaches the four-gap lstsq "
            "gate in the CI step budget. Calculator stays diagnostic. "
            "Four-gap gates are not mutated. Not Littlewood-Paley."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    g1 = _run_g1()
    g2 = _run_g2()
    g4 = _run_g4()
    g3 = _run_g3()
    in_scope = [g1, g2, g4]
    payload = provenance(
        schema="spectral-design-v1",
        config={
            "family": "spectral_design",
            "full": bool(args.full),
            "g3_in_all_passed": False,
            "gates_in_scope": ["g1", "g2", "g4"],
        },
    )
    payload["gates"] = gates_block(in_scope)
    payload["g1"] = g1
    payload["g2"] = g2
    payload["g3"] = g3
    payload["g4"] = g4
    payload["honesty"] = {
        "wavelet_frame_claim": False,
        "littlewood_paley_claim": False,
        "g3_in_ci_all_passed": False,
        "collapse": "delta -> 0; pack order is a band selector",
    }
    if args.full:
        out_dir = SCRATCH / "spectral_design"
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / "spectral_design.json"
        dest.write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {dest}")
    else:
        path = write_json("spectral_design_smoke.json", payload)
        print(f"wrote {path}")
    if not payload["gates"]["all_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
