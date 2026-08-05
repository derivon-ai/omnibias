# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Cross-consistency: symbolic's NumPy differential geometry vs ``omnibias-geometry``.

``omnibias.symbolic.geometry_discovery`` re-derives the Christoffel / Riemann / Ricci /
scalar-curvature columns in pure NumPy from an explicit metric jet ``(g, dg, ddg)``, while
``omnibias.geometry`` computes the same tensors on a backend via forward-mode autodiff of an
analytic ``g_point``. They deliberately share the index conventions
(``Gamma[..., k, i, j]``, ``R[..., rho, sigma, mu, nu]``), so on the *same* metric they must
agree to float64 tolerance. This guard pins that agreement so the two implementations cannot
drift apart. It skips cleanly if ``omnibias-geometry`` or torch is unavailable.

The shared metric is the 2-D warped product ``ds^2 = dx^2 + f(x)^2 dy^2`` (a surface of
revolution): symbolic gets ``(f, f', f'')`` analytically; geometry gets the closed-form
``g_point`` and differentiates it. A quadratic warp gives *non-constant* curvature, so the
element-wise comparison exercises every Christoffel / Riemann component, not just a constant.
"""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.symbolic.geometry_discovery import (
    christoffel_symbols,
    ricci_tensor,
    riemann_tensor,
    scalar_curvature,
    warped_product_metric_field,
)

# Sample points: x in (0, pi) style range (kept away from f = 0); y is arbitrary because the
# warped-product metric does not depend on y.
_X = np.array([[0.4, 0.0], [0.9, 1.3], [1.4, -0.7], [1.9, 2.1]], dtype=float)


def _warp(t: float) -> float:
    """A quadratic warp f(x) = 1 + x/2 + x^2/5 (strictly positive on the samples)."""
    return 1.0 + 0.5 * t + 0.2 * t * t


def _warp_derivs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = 1.0 + 0.5 * x + 0.2 * x * x
    fp = 0.5 + 0.4 * x
    fpp = np.full_like(x, 0.4)
    return f, fp, fpp


def _geometry_manifold():  # type: ignore[no-untyped-def]
    torch = pytest.importorskip("torch")
    geometry = pytest.importorskip("omnibias.geometry")
    ManifoldSpec = geometry.ManifoldSpec
    MetricSpec = geometry.MetricSpec

    def g_point(x):  # type: ignore[no-untyped-def]
        fx = _warp(x[0])
        zero = x[0] * 0.0
        one = zero + 1.0
        return torch.stack(
            [torch.stack([one, zero]), torch.stack([zero, fx * fx])]
        )

    return ManifoldSpec("warp2d", 2, MetricSpec(g_point, dim=2, name="warped_product"))


def _symbolic_metric():  # type: ignore[no-untyped-def]
    f, fp, fpp = _warp_derivs(_X[:, 0])
    return warped_product_metric_field(_X, f, fp, fpp)


def test_christoffel_matches_geometry_package() -> None:
    torch = pytest.importorskip("torch")
    tgeo = pytest.importorskip("omnibias.geometry.torch").ops
    manifold = _geometry_manifold()
    coords = torch.as_tensor(_X, dtype=torch.float64)

    sym = christoffel_symbols(_symbolic_metric())
    geo = tgeo.christoffel(coords, manifold).detach().cpu().numpy()
    np.testing.assert_allclose(sym, geo, atol=1e-9, rtol=0.0)


def test_riemann_ricci_scalar_match_geometry_package() -> None:
    torch = pytest.importorskip("torch")
    tgeo = pytest.importorskip("omnibias.geometry.torch").ops
    manifold = _geometry_manifold()
    coords = torch.as_tensor(_X, dtype=torch.float64)
    metric = _symbolic_metric()

    np.testing.assert_allclose(
        riemann_tensor(metric),
        tgeo.riemann_tensor(coords, manifold).detach().cpu().numpy(),
        atol=1e-9,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        ricci_tensor(metric),
        tgeo.ricci_tensor(coords, manifold).detach().cpu().numpy(),
        atol=1e-9,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        scalar_curvature(metric),
        tgeo.scalar_curvature(coords, manifold).detach().cpu().numpy(),
        atol=1e-9,
        rtol=0.0,
    )


def test_scalar_curvature_matches_closed_form_gaussian() -> None:
    """Independent anchor: for ds^2=dx^2+f^2 dy^2, R = 2K = -2 f''/f."""
    metric = _symbolic_metric()
    f, _fp, fpp = _warp_derivs(_X[:, 0])
    np.testing.assert_allclose(scalar_curvature(metric), -2.0 * fpp / f, atol=1e-9, rtol=0.0)
