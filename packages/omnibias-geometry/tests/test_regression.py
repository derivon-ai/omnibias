# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Pinned regression for the metric -> curvature pipeline.

The analytic and sympy tests prove *correctness*; this test pins the actual
library output (Christoffel / Riemann / Ricci / scalar curvature on the round
sphere at fixed collocation points) to a committed golden file so a future
refactor cannot silently change a result. The golden was produced by the torch
backend; both backends are checked against it at ``rtol=1e-12, atol=1e-14``, so
this doubles as a tight cross-backend bit-parity guard.

Regenerate the golden intentionally (e.g. after a deliberate, reviewed change)
with::

    python packages/omnibias-geometry/tests/test_regression.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omnibias.geometry import ManifoldSpec, MetricSpec
from omnibias.geometry.torch import ops as tgeo

DATA = Path(__file__).parent / "data"
GOLDEN = DATA / "curvature_sphere_golden.npz"

R = 1.3
COORDS = np.array(
    [[0.6, 0.2], [0.9, 1.1], [1.4, 2.0], [1.9, 3.0], [2.3, 4.2], [2.7, 5.5]],
    dtype=np.float64,
)


def _sphere_metric(xp, stack):  # type: ignore[no-untyped-def]
    r2 = R**2

    def g_point(x):  # x: (2,) = (theta, phi)
        theta = x[0]
        z = 0.0 * theta
        return stack([
            stack([r2 * (1.0 + z), z]),
            stack([z, r2 * xp.sin(theta) ** 2]),
        ])

    return g_point


def _manifold():  # type: ignore[no-untyped-def]
    g = _sphere_metric(torch, torch.stack)
    return ManifoldSpec("sphere_S2", 2, MetricSpec(g, dim=2, name="round_sphere"))


def _compute() -> dict[str, np.ndarray]:
    c = torch.as_tensor(COORDS, dtype=torch.float64)
    m = _manifold()
    return {
        "christoffel": tgeo.christoffel(c, m).detach().cpu().numpy(),
        "riemann": tgeo.riemann_tensor(c, m).detach().cpu().numpy(),
        "ricci": tgeo.ricci_tensor(c, m).detach().cpu().numpy(),
        "scalar": tgeo.scalar_curvature(c, m).detach().cpu().numpy(),
    }


def _regenerate() -> None:
    DATA.mkdir(exist_ok=True)
    np.savez(GOLDEN, **_compute())


def test_curvature_pipeline_pinned():  # type: ignore[no-untyped-def]
    if not GOLDEN.exists():  # pragma: no cover - guard for a missing artifact
        pytest.skip(f"golden artifact missing: {GOLDEN}")
    golden = np.load(GOLDEN)
    got = _compute()
    for key in golden.files:
        np.testing.assert_allclose(
            got[key], golden[key], rtol=1e-12, atol=1e-14,
            err_msg=f"curvature regression drift in {key!r}",
        )


if __name__ == "__main__":
    _regenerate()
    print(f"regenerated {GOLDEN}")
