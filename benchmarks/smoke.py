# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bounded pipeline check for the public Torch, JAX, Taylor-mode and folx comparisons.

This reuses the measured workloads and independent oracles at smaller sizes.
Outputs are smoke evidence, never replacements for reviewed performance figures.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMNIBIAS_SCRATCH", str(Path("artifacts/benchmark-smoke").resolve()))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")


def main() -> None:
    import derivative_order
    import laplacian_scaling
    import readme_derivatives
    from _common import provenance, write_json

    derivative_order.N_POINTS = 32
    derivative_order.ORDERS = [2, 4]
    derivative_order.REFERENCE_POINTS = 3
    derivative_order.REPEATS = 2
    derivative_order.WARMUP = 1
    derivative_order.main()
    readme_derivatives.DIMS = (1, 3, 1)
    readme_derivatives.POINTS = 8
    readme_derivatives.ORDERS = (2, 4)
    readme_derivatives.REPEATS = 2
    readme_derivatives.main()
    laplacian_scaling.H = 4
    laplacian_scaling.B = 4
    laplacian_scaling.REPEATS = 2
    row = laplacian_scaling._run_d(3)
    write_json(
        "laplacian_smoke.json",
        {
            "scope": "smoke only",
            "environment": provenance(
                schema="benchmark-smoke/v1", config={"dimension": 3, "width": 4, "batch": 4}
            ),
            "row": row,
        },
    )


if __name__ == "__main__":
    main()
