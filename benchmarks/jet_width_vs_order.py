# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified 1-D jet-width sweep by derivative order.

This benchmark measures the width of the interval enclosure returned by
``certified_partials`` for successive derivatives of one fixed, non-degenerate
tanh unit.  It checks every enclosure against a deterministic grid and
independent random points.  Raw widths have a factorial envelope, so the
scaling gate fits ``width / n!`` against derivative order; a near-zero
power-law exponent is the declared finite-network behavior being tested.

This is evidence about certified finite MLP derivative enclosures.  It does
not establish a spectral tail theorem or a continuum PDE certificate.

    python benchmarks/jet_width_vs_order.py
    python benchmarks/jet_width_vs_order.py --full
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
from _gates import (  # type: ignore[import-not-found]  # noqa: E402
    gates_block,
    require_enclosure_coverage,
    require_scaling_exponent,
)
from omnibias.core.polynomials import tanh_polynomial_coeffs  # noqa: E402
from omnibias.core.verified.jet_mv import certified_partials  # noqa: E402
from omnibias.core.verified.transcend import BACKEND_MPMATH, backend_name  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
BOX = (-0.35, 0.35)
WEIGHT = 1.1
BIAS = 0.2
NORMALIZED_WIDTH_EXPONENT = 0.0
NORMALIZED_WIDTH_TOLERANCE = 0.15


def _layers() -> list[tuple[list[list[float]], list[float], str]]:
    return [([[WEIGHT]], [BIAS], "tanh")]


def _tanh_derivative(x: float, order: int) -> float:
    """Independent floating readout of ``d^n tanh(WEIGHT*x + BIAS) / dx^n``."""
    value = math.tanh(WEIGHT * x + BIAS)
    coeffs = tanh_polynomial_coeffs(order)
    polynomial = math.fsum(coefficient * value**degree for degree, coefficient in enumerate(coeffs))
    return float(WEIGHT**order * polynomial)


def _sample_points(*, grid_count: int, random_count: int) -> tuple[list[float], list[float]]:
    """Return deterministic grid and seeded independent random samples."""
    lo, hi = BOX
    grid = [float(x) for x in np.linspace(lo, hi, grid_count)]
    rng = np.random.default_rng(20260827)
    random = [float(x) for x in rng.uniform(lo, hi, size=random_count)]
    return grid, random


def _run(
    *,
    max_order: int,
    grid_count: int,
    random_count: int,
) -> tuple[list[dict[str, float]], dict[str, Any], dict[str, Any]]:
    orders = list(range(1, max_order + 1))
    grid, random = _sample_points(grid_count=grid_count, random_count=random_count)
    all_points = [*grid, *random]
    rows: list[dict[str, float]] = []
    enclosures: list[tuple[float, float]] = []
    truths: list[float] = []
    for order in orders:
        enclosure = certified_partials([BOX], _layers(), order)[(order,)][0]
        width = float(enclosure.width)
        normalized_width = width / float(math.factorial(order))
        rows.append(
            {
                "order": float(order),
                "width": width,
                "factorial_normalized_width": normalized_width,
                "lo": float(enclosure.lo),
                "hi": float(enclosure.hi),
            }
        )
        for point in all_points:
            enclosures.append((float(enclosure.lo), float(enclosure.hi)))
            truths.append(_tanh_derivative(point, order))

    coverage = require_enclosure_coverage(
        enclosures,
        truths,
        n_min=1000,
        name="certified_partials_tanh_derivative_coverage",
    )
    normalized_exponent = require_scaling_exponent(
        orders,
        [row["factorial_normalized_width"] for row in rows],
        expected=NORMALIZED_WIDTH_EXPONENT,
        tol=NORMALIZED_WIDTH_TOLERANCE,
        min_decades=1.0,
        name="factorial_normalized_jet_width_vs_order",
    )
    raw_exponent = float(
        np.polyfit(
            np.log(np.asarray(orders, dtype=float)),
            np.log(np.asarray([row["width"] for row in rows], dtype=float)),
            1,
        )[0]
    )
    return rows, coverage, {
        **normalized_exponent,
        "raw_loglog_exponent": raw_exponent,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--max-order", type=int, default=None)
    args = parser.parse_args(argv)
    full = bool(args.full)
    max_order = int(args.max_order if args.max_order is not None else (40 if full else 24))
    if max_order < 10:
        raise ValueError("max-order must be >= 10 to span one decade")
    grid_count = 129 if full else 65
    random_count = 128 if full else 64
    t0 = time.perf_counter()
    rows, coverage, exponent = _run(
        max_order=max_order,
        grid_count=grid_count,
        random_count=random_count,
    )
    entries = [
        {
            **coverage,
            "name": "certified_partials_coverage",
            "passed": bool(coverage["passed"]),
        },
        {
            **exponent,
            "name": "factorial_normalized_width_exponent",
            "passed": bool(exponent["passed"]),
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
    artifact = "jet_width_vs_order.json" if full else "jet_width_vs_order_smoke.json"
    payload: dict[str, Any] = {
        **provenance(
            schema="omnibias.benchmarks.jet_width_vs_order.v1",
            config={
                "family": "certified_jet_width_vs_order",
                "full": full,
                "box": list(BOX),
                "layers": "one_tanh_unit",
                "weight": WEIGHT,
                "bias": BIAS,
                "max_order": max_order,
                "grid_count": grid_count,
                "random_count": random_count,
                "factorial_normalized_width_exponent": NORMALIZED_WIDTH_EXPONENT,
                "honesty": {
                    "transcendental_backend": backend_name(),
                    "unconditional_transcendentals": backend_name() == BACKEND_MPMATH,
                    "spectral_tail_theorem": False,
                    "continuum_pde_claim": False,
                },
            },
        ),
        "gates": dict(gates_block(entries)),
        "per_order": rows,
        "sample_count_per_order": grid_count + random_count,
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        destination = SCRATCH / "jet_width_vs_order"
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / artifact
        path.write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
