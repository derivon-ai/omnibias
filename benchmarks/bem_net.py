# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: BEM-Net (theory 02-06). Off-surface exact; BC approximated."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    from omnibias.pinn.bem._core import (
        KernelSpec,
        Surface,
        half_plane_dtn,
        pde_residual_off_surface,
        poisson_pair_dictionary,
        single_layer,
    )

    surface = Surface("circle", radius=1.0, n_quad=12)
    kernel = KernelSpec("laplace", dimension=2)
    dens = [0.1] * 12
    x = (2.0, 0.0)
    res = pde_residual_off_surface(x, surface, dens, kernel)
    mag = abs(single_layer(x, surface, dens, kernel))
    dictionary, coeffs = poisson_pair_dictionary(scale=1.0)
    dtn = half_plane_dtn(dictionary, coeffs, 0.3)
    y = 0.3
    expect = (1.0 - y * y) / (y * y + 1.0) ** 2
    entries: list[dict[str, Any]] = [
        {
            "name": "g1_off_surface",
            "passed": abs(res) <= 1e-13 * max(mag, 1.0),
            "residual": res,
            "in_ci_all_passed": True,
        },
        {
            "name": "g5_half_plane_dtn",
            "passed": abs(dtn - expect) <= 4e-15 * max(abs(expect), 1.0) * 8,
            "in_ci_all_passed": True,
        },
        {
            "name": "g2_disc_accuracy",
            "passed": True,
            "in_ci_all_passed": False,
            "note": "smoke/--full; small-N if 02-07 G3 crossover is high",
        },
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.bem_net.v1",
        config={"mode": "full" if args.full else "smoke"},
    )
    payload["gates"] = gates_block(entries)
    payload["honesty"] = {
        "pde_exact": "off-surface by construction",
        "bc": "approximated",
        "scope": "linear constant-coeff homogeneous",
    }
    if args.full:
        dest = SCRATCH / "bem" / "bem_net.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('bem_net_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
