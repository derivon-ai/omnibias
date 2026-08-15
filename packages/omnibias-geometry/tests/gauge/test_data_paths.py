# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Legal gauge data paths: analytic / weak / lattice; refuse path D and 1-D integrals."""

from __future__ import annotations

import itertools
from typing import get_args
from unittest.mock import patch

import numpy as np
import omnibias.geometry.gauge._core.lie_algebra as la
import pytest
from omnibias.geometry.gauge._core.covariant_jet import (
    GaugeCovariantJet,
    global_gauge_transform_connection,
    random_special_unitary,
)
from omnibias.geometry.gauge._core.data_paths import (
    ConnectionSource,
    LatticeLinkField,
    is_scalar_integral_column_name,
    refuse_connection_jet_from_links,
    refuse_lattice_random_feature_jet,
    refuse_scalar_integral_as_ym_weak_form,
)
from omnibias.geometry.gauge._core.instanton import bpst_instanton_arrays
from omnibias.geometry.gauge._core.weak_ym import (
    WEAK_YM_FLOOR,
    evaluate_weak_ym_identity,
    gaussian_adjoint_test_bank,
    weak_yang_mills_residuals,
)
from omnibias.geometry.gauge.lattice._core.kernels import plaquette_trace

SIG_E = (1, 1, 1, 1)
WEAK_VS_POINTWISE_FACTOR = 5.0
FD_NOISE_SIGMA = 4e-2
HIGHPASS_H = 0.2


def _sample_points(n: int = 24, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1.5, 1.5, size=(n * 2, 4))
    pts = pts[np.linalg.norm(pts, axis=1) > 0.4]
    return pts[:n]


def _take(grid: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return grid[idx[:, 0], idx[:, 1], idx[:, 2], idx[:, 3]]


def _central_fd_connection(
    a_grid: np.ndarray, h: float, idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central differences of a 4-D grid of ``A_mu^a`` at integer sites ``idx``."""
    batch = int(idx.shape[0])
    dim, adj = a_grid.shape[-2:]
    a_arr = _take(a_grid, idx)
    da_arr = np.zeros((batch, dim, dim, adj), dtype=np.float64)
    dda_arr = np.zeros((batch, dim, dim, dim, adj), dtype=np.float64)
    for rho in range(dim):
        plus = idx.copy()
        plus[:, rho] += 1
        minus = idx.copy()
        minus[:, rho] -= 1
        da_arr[:, rho] = (_take(a_grid, plus) - _take(a_grid, minus)) / (2.0 * h)
    for sigma in range(dim):
        for rho in range(dim):
            if sigma == rho:
                plus = idx.copy()
                plus[:, sigma] += 1
                minus = idx.copy()
                minus[:, sigma] -= 1
                dda_arr[:, sigma, rho] = (
                    _take(a_grid, plus) - 2.0 * a_arr + _take(a_grid, minus)
                ) / (h * h)
            else:
                pp = idx.copy()
                pp[:, sigma] += 1
                pp[:, rho] += 1
                pm = idx.copy()
                pm[:, sigma] += 1
                pm[:, rho] -= 1
                mp = idx.copy()
                mp[:, sigma] -= 1
                mp[:, rho] += 1
                mm = idx.copy()
                mm[:, sigma] -= 1
                mm[:, rho] -= 1
                dda_arr[:, sigma, rho] = (
                    _take(a_grid, pp)
                    - _take(a_grid, pm)
                    - _take(a_grid, mp)
                    + _take(a_grid, mm)
                ) / (4.0 * h * h)
    return a_arr, da_arr, dda_arr


def _polynomial_grid(
    axes: np.ndarray,
    *,
    adjoint_dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Quadratic ``A`` on a tensor grid, shape ``(*N, 4, n)``."""
    mesh = np.stack(np.meshgrid(*([axes] * 4), indexing="ij"), axis=-1)
    x = mesh.reshape(-1, 4)
    dim = 4
    c0 = 0.2 * rng.normal(size=(dim, adjoint_dim))
    c1 = 0.2 * rng.normal(size=(dim, dim, adjoint_dim))
    c2 = 0.2 * rng.normal(size=(dim, dim, dim, adjoint_dim))
    c2 = 0.5 * (c2 + np.swapaxes(c2, 0, 1))
    a_arr = (
        c0[None]
        + np.einsum("lma,Bl->Bma", c1, x)
        + 0.5 * np.einsum("slma,Bs,Bl->Bma", c2, x, x)
    )
    n = int(axes.shape[0])
    return a_arr.reshape(n, n, n, n, dim, adjoint_dim)


def test_connection_source_tags() -> None:
    assert set(get_args(ConnectionSource)) == {
        "analytic",
        "spectral",
        "lattice_links",
        "random_feature",
        "finite_difference",
    }


def test_path_a_bpst_weak_identity() -> None:
    pts = _sample_points()
    a, da, dda = bpst_instanton_arrays(pts)
    jet = GaugeCovariantJet.from_arrays(
        a, da, dda, algebra=la.su(2), coupling=1.0, signature=SIG_E
    )
    report = evaluate_weak_ym_identity(
        jet, pts, atol=WEAK_YM_FLOOR, n_tests=8, rng=np.random.default_rng(0)
    )
    assert report["passed"] is True
    assert float(report["max_abs"]) < WEAK_YM_FLOOR
    assert report["yang_mills_claim"] is False
    assert report["continuum_claim"] is False
    assert int(report["n_tests"]) == 8


def test_path_b_fd_is_highpass_weak_form_averages() -> None:
    rng = np.random.default_rng(11)
    n_coarse = 5
    h_coarse = HIGHPASS_H
    h_fine = 0.5 * h_coarse
    lo, hi = -0.4, 0.4
    axes_fine = np.linspace(lo, hi, 2 * (n_coarse - 1) + 1)
    a_exact = _polynomial_grid(axes_fine, adjoint_dim=3, rng=rng)
    a_noisy = a_exact + FD_NOISE_SIGMA * rng.normal(size=a_exact.shape)
    a_coarse = a_noisy[::2, ::2, ::2, ::2]
    interior = np.array(
        list(itertools.product((1, 2, 3), repeat=4)), dtype=np.int64
    )
    a_c, da_c, dda_c = _central_fd_connection(a_coarse, h_coarse, interior)
    a_f, da_f, dda_f = _central_fd_connection(a_noisy, h_fine, interior * 2)
    np.testing.assert_allclose(a_c, a_f, atol=1e-15)
    jet_c = GaugeCovariantJet.from_arrays(
        a_c, da_c, dda_c, algebra=la.su(2), coupling=0.8, signature=SIG_E
    )
    jet_f = GaugeCovariantJet.from_arrays(
        a_f, da_f, dda_f, algebra=la.su(2), coupling=0.8, signature=SIG_E
    )
    pointwise_coarse = float(np.max(np.abs(jet_c.ym_eom)))
    pointwise_fine = float(np.max(np.abs(jet_f.ym_eom)))
    assert pointwise_fine > pointwise_coarse
    mesh = np.stack(np.meshgrid(*([axes_fine] * 4), indexing="ij"), axis=-1)
    pts = mesh[tuple((interior * 2).T)]
    bank = gaussian_adjoint_test_bank(
        pts, n_tests=8, algebra=la.su(2), rng=rng
    )
    weak_max = float(np.max(np.abs(weak_yang_mills_residuals(jet_f, bank))))
    assert weak_max * WEAK_VS_POINTWISE_FACTOR <= pointwise_fine


def test_path_c_links_stay_links() -> None:
    links = np.zeros((4, 2, 2, 2, 2, 4), dtype=np.float64)
    links[..., 0] = 1.0
    field = LatticeLinkField(links=links, spacing=0.5)
    assert field.spacing == 0.5
    with pytest.raises(ValueError, match="lattice"):
        refuse_connection_jet_from_links(field)
    trace = np.asarray(plaquette_trace(np, links, 0, 1))
    assert float(np.max(np.abs(trace - 1.0))) < 1e-15


def test_path_d_refuses_before_jet() -> None:
    called = {"from_arrays": False}

    def _boom(*_args: object, **_kwargs: object) -> None:
        called["from_arrays"] = True
        raise AssertionError("from_arrays must not run")

    with patch.object(GaugeCovariantJet, "from_arrays", _boom):
        with pytest.raises(ValueError, match="lattice") as links_exc:
            GaugeCovariantJet.from_lattice_links(np.ones((4, 2, 2, 2, 2, 4)))
        assert "partial" in str(links_exc.value)
        with pytest.raises(ValueError, match="random-feature") as rf_exc:
            GaugeCovariantJet.from_neural_fields(object())
        rf_text = str(rf_exc.value).lower()
        assert "lattice" in rf_text
        assert "interpolat" in rf_text
        with pytest.raises(ValueError, match="path D"):
            refuse_lattice_random_feature_jet(object())
    assert called["from_arrays"] is False


def test_scalar_integral_is_not_ym_weak_form() -> None:
    with pytest.raises(ValueError, match="Fredholm"):
        refuse_scalar_integral_as_ym_weak_form(["I(v)", "fredholm_K"])
    assert is_scalar_integral_column_name("I(v)")
    assert is_scalar_integral_column_name("fredholm_heat")
    assert is_scalar_integral_column_name("volterra_causal")
    assert is_scalar_integral_column_name("running_integral")
    assert not is_scalar_integral_column_name("tr(F^2)")


def test_weak_residuals_gauge_invariant_on_bpst() -> None:
    pts = _sample_points(n=16, seed=3)
    a, da, dda = bpst_instanton_arrays(pts)
    algebra = la.su(2)
    jet = GaugeCovariantJet.from_arrays(
        a, da, dda, algebra=algebra, coupling=1.0, signature=SIG_E
    )
    rng = np.random.default_rng(3)
    bank = gaussian_adjoint_test_bank(
        pts, n_tests=8, algebra=algebra, rng=rng
    )
    r0 = weak_yang_mills_residuals(jet, bank)
    u = random_special_unitary(2, rng)
    a_g, da_g, dda_g = global_gauge_transform_connection(a, da, dda, u, algebra=algebra)
    jet_g = GaugeCovariantJet.from_arrays(
        a_g, da_g, dda_g, algebra=algebra, coupling=1.0, signature=SIG_E
    )
    r1 = weak_yang_mills_residuals(jet_g, bank)
    assert float(np.max(np.abs(r0))) < WEAK_YM_FLOOR
    assert float(np.max(np.abs(r1))) < WEAK_YM_FLOOR
    assert float(np.max(np.abs(r0 - r1))) < WEAK_YM_FLOOR


def test_weak_residual_contraction_unit() -> None:
    a = np.zeros((4, 4, 3), dtype=np.float64)
    da = np.zeros((4, 4, 4, 3), dtype=np.float64)
    dda = np.zeros((4, 4, 4, 4, 3), dtype=np.float64)
    jet = GaugeCovariantJet.from_arrays(
        a, da, dda, algebra=la.su(2), coupling=1.0, signature=SIG_E
    )
    object.__setattr__(jet, "ym_eom", np.zeros_like(jet.ym_eom))
    jet.ym_eom[:, 0, 0] = 2.0
    omega = np.zeros((1, 4, 4, 3), dtype=np.float64)
    omega[0, :, 0, 0] = 1.0
    from omnibias.geometry.gauge._core.weak_ym import AdjointTestBank

    bank = AdjointTestBank(omega=omega, weights=np.full(4, 0.25))
    residual = weak_yang_mills_residuals(jet, bank)
    assert residual.shape == (1,)
    assert float(residual[0]) == pytest.approx(2.0, abs=1e-15)
