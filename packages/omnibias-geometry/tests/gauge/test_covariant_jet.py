# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gauge-covariant jet: F / D F fibers, singlet allowlist, equivariance gate."""

from __future__ import annotations

import numpy as np
import omnibias.geometry.gauge._core.lie_algebra as la
import pytest
from _gauge_helpers import instanton_arrays
from omnibias.geometry.gauge._core import kernels
from omnibias.geometry.gauge._core.covariant_jet import (
    LEGAL_SINGLET_ATOMS,
    SELF_DUAL_ACTION_OVER_TOPOLOGICAL,
    SINGLET_TR_F2,
    SINGLET_TR_F_FTILDE,
    GaugeCovariantJet,
    assert_library_gauge_legal,
    covariant_singlet_columns,
    evaluate_gauge_law_gate,
    gauge_equivariance_defect,
)
from omnibias.geometry.gauge._core.instanton import (
    bpst_instanton_arrays,
    thooft_eta,
)


class _Eq:
    def __init__(self, name: str, coef: float) -> None:
        self.term_names = (name,)
        self.coefficients = np.array([coef])

    def predict(self, design: np.ndarray) -> np.ndarray:
        return design @ self.coefficients

SIG_E = (1, 1, 1, 1)
EOM_FLOOR = 1e-7
SELF_DUAL_FLOOR = 1e-9
BIANCHI_FLOOR = 1e-7


def _sample_points(n: int = 24, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.5, 1.5, size=(n, 4))


def _polynomial_connection(
    points: np.ndarray,
    *,
    adjoint_dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Consistent quadratic ``A(x)`` so ``dA`` / ``ddA`` are true derivatives."""
    x = np.asarray(points, dtype=np.float64)
    batch, dim = x.shape
    c0 = rng.normal(size=(dim, adjoint_dim))
    c1 = rng.normal(size=(dim, dim, adjoint_dim))
    c2 = rng.normal(size=(dim, dim, dim, adjoint_dim))
    c2 = 0.5 * (c2 + np.swapaxes(c2, 0, 1))
    a_arr = (
        c0[None]
        + np.einsum("lma,Bl->Bma", c1, x)
        + 0.5 * np.einsum("slma,Bs,Bl->Bma", c2, x, x)
    )
    da_arr = c1[None] + np.einsum("rlna,Bl->Brna", c2, x)
    dda_arr = np.broadcast_to(c2[None], (batch, dim, dim, dim, adjoint_dim)).copy()
    return a_arr, da_arr, dda_arr


def test_instanton_helper_is_public_alias() -> None:
    assert instanton_arrays is bpst_instanton_arrays
    assert thooft_eta().shape == (3, 4, 4)


def test_bpst_jet_eom_bianchi_self_dual() -> None:
    pts = _sample_points()
    a, da, dda = bpst_instanton_arrays(pts)
    jet = GaugeCovariantJet.from_arrays(
        a, da, dda, algebra=la.su(2), coupling=1.0, signature=SIG_E
    )
    assert float(np.max(np.abs(jet.ym_eom))) < EOM_FLOOR
    assert float(np.max(np.abs(jet.bianchi))) < BIANCHI_FLOOR
    singles = jet.singlets()
    assert float(np.max(singles["|F-*F|^2"])) < SELF_DUAL_FLOOR**2 * 100
    assert float(np.max(singles["|D*F|^2"])) < EOM_FLOOR**2 * 100
    assert float(np.max(singles["|Bianchi|^2"])) < BIANCHI_FLOOR**2 * 100


def test_df_matches_partial_plus_bracket() -> None:
    pts = _sample_points()
    a, da, dda = _polynomial_connection(pts, adjoint_dim=3, rng=np.random.default_rng(2))
    su2 = la.su(2)
    f = su2.structure_constants()
    coupling = 0.7
    jet = GaugeCovariantJet.from_arrays(
        a, da, dda, algebra=su2, coupling=coupling, signature=SIG_E
    )
    dF = kernels.field_strength_partials(np, a, da, dda, f, coupling)
    fld = kernels.field_strength(np, a, da, f, coupling)
    bracket = coupling * np.einsum("pqa,Brp,Bmnq->Brmna", f, a, fld)
    np.testing.assert_allclose(jet.DF, dF + bracket, atol=1e-11)


def test_df_central_difference_rate() -> None:
    rng = np.random.default_rng(4)
    x0 = rng.uniform(-0.4, 0.4, size=(6, 4))
    su2 = la.su(2)
    coupling = 0.9
    a0, da0, dda0 = _polynomial_connection(x0, adjoint_dim=3, rng=rng)
    jet = GaugeCovariantJet.from_arrays(
        a0, da0, dda0, algebra=su2, coupling=coupling, signature=SIG_E
    )
    f = su2.structure_constants()

    # Frozen coefficients: evaluate F at x0 ± h e_rho from the quadratic Taylor shift.
    def _shift(h: float, rho: int) -> np.ndarray:
        eye = np.zeros(4)
        eye[rho] = h
        # A(x+h) = A + h d_rho A + (h^2/2) d_rho d_rho A
        a_h = a0 + h * da0[:, rho] + 0.5 * h * h * dda0[:, rho, rho]
        da_h = da0 + h * dda0[:, rho]
        return kernels.field_strength(np, a_h, da_h, f, coupling)

    def _fd_error(h: float) -> float:
        errs = []
        for rho in range(4):
            dF_fd = (_shift(h, rho) - _shift(-h, rho)) / (2.0 * h)
            bracket = coupling * np.einsum(
                "pqa,Bp,Bmnq->Bmna", f, a0[:, rho], jet.F
            )
            errs.append(np.max(np.abs(dF_fd + bracket - jet.DF[:, rho])))
        return float(max(errs))

    err_h = _fd_error(1e-3)
    err_h2 = _fd_error(5e-4)
    assert err_h2 < 0.4 * err_h
    assert err_h2 < 1e-6


def test_contraction_recovers_ym_operator() -> None:
    pts = _sample_points()
    a, da, dda = _polynomial_connection(pts, adjoint_dim=3, rng=np.random.default_rng(5))
    su2 = la.su(2)
    jet = GaugeCovariantJet.from_arrays(
        a, da, dda, algebra=su2, coupling=0.6, signature=SIG_E
    )
    eta = np.ones(4)
    contracted = np.einsum("m,n,Bmmna->Bna", eta, eta, jet.DF)
    np.testing.assert_allclose(contracted, jet.ym_eom, atol=1e-11)


def test_antisymmetry_of_f_and_df() -> None:
    pts = _sample_points()
    a, da, dda = bpst_instanton_arrays(pts)
    jet = GaugeCovariantJet.from_arrays(
        a, da, dda, algebra=la.su(2), coupling=1.0, signature=SIG_E
    )
    np.testing.assert_allclose(jet.F, -np.swapaxes(jet.F, 1, 2), atol=1e-11)
    np.testing.assert_allclose(jet.DF, -np.swapaxes(jet.DF, 2, 3), atol=1e-11)


def test_abelian_and_zero_coupling_df_is_partial() -> None:
    rng = np.random.default_rng(6)
    pts = _sample_points(8, seed=6)
    a, da, dda = _polynomial_connection(pts, adjoint_dim=3, rng=rng)
    su2 = la.su(2)
    f = su2.structure_constants()
    jet0 = GaugeCovariantJet.from_arrays(
        a, da, dda, algebra=su2, coupling=0.0, signature=SIG_E
    )
    dF = kernels.field_strength_partials(np, a, da, dda, f, 0.0)
    np.testing.assert_allclose(jet0.DF, dF, atol=1e-11)

    u1 = la.u1()
    a1, da1, dda1 = _polynomial_connection(pts, adjoint_dim=1, rng=rng)
    jet_u = GaugeCovariantJet.from_arrays(
        a1, da1, dda1, algebra=u1, coupling=2.5, signature=SIG_E
    )
    dF_u = kernels.field_strength_partials(
        np, a1, da1, dda1, u1.structure_constants(), 2.5
    )
    np.testing.assert_allclose(jet_u.DF, dF_u, atol=1e-11)


@pytest.mark.parametrize("n", [2, 3])
def test_generic_connection_bianchi_vanishes_eom_does_not(n: int) -> None:
    rng = np.random.default_rng(7 + n)
    pts = _sample_points(16, seed=7 + n)
    algebra = la.su(n)
    a, da, dda = _polynomial_connection(pts, adjoint_dim=algebra.dim, rng=rng)
    jet = GaugeCovariantJet.from_arrays(
        a, da, dda, algebra=algebra, coupling=0.8, signature=SIG_E
    )
    assert float(np.max(np.abs(jet.bianchi))) < BIANCHI_FLOOR
    assert float(np.max(np.abs(jet.ym_eom))) > BIANCHI_FLOOR


def test_singlet_allowlist_and_no_connection_leakage() -> None:
    pts = _sample_points(4)
    a, da, dda = bpst_instanton_arrays(pts)
    jet = GaugeCovariantJet.from_arrays(
        a, da, dda, algebra=la.su(2), coupling=1.0, signature=SIG_E
    )
    assert set(jet.singlets()) == LEGAL_SINGLET_ATOMS
    assert set(jet.library_names()) == LEGAL_SINGLET_ATOMS
    assert set(covariant_singlet_columns(jet)) == LEGAL_SINGLET_ATOMS
    fields = set(jet.__dataclass_fields__)
    assert "A" not in fields and "dA" not in fields and "ddA" not in fields
    assert not hasattr(jet, "A")
    assert not hasattr(jet, "dA")
    assert not hasattr(jet, "ddA")


@pytest.mark.parametrize(
    "name",
    ["A_0_0", "u_x", "|A|^2", "F_01_2", "DF_0_12_1"],
)
def test_assert_library_gauge_legal_rejects_coordinate_atoms(name: str) -> None:
    with pytest.raises(ValueError, match="allowlisted"):
        assert_library_gauge_legal([name])


def test_from_arrays_shape_and_algebra_guards() -> None:
    pts = _sample_points(3)
    a, da, dda = bpst_instanton_arrays(pts)
    su2 = la.su(2)
    with pytest.raises(ValueError, match="dA must have shape"):
        GaugeCovariantJet.from_arrays(
            a, da[:, :2], dda, algebra=su2, coupling=1.0, signature=SIG_E
        )
    with pytest.raises(ValueError, match="signature"):
        GaugeCovariantJet.from_arrays(
            a, da, dda, algebra=su2, coupling=1.0, signature=(1, 1, 1)
        )
    with pytest.raises(ValueError, match="adjoint"):
        GaugeCovariantJet.from_arrays(
            a, da, dda, algebra=la.su(3), coupling=1.0, signature=SIG_E
        )


def test_infinitesimal_singlets_quadratic_F_linear() -> None:
    pts = _sample_points(10, seed=8)
    a, da, dda = bpst_instanton_arrays(pts)
    omega = np.random.default_rng(8).normal(size=(pts.shape[0], 3))
    report = gauge_equivariance_defect(
        a,
        da,
        dda,
        omega,
        algebra=la.su(2),
        coupling=1.0,
        signature=SIG_E,
        eps=1e-4,
        rng=np.random.default_rng(8),
    )
    for name in LEGAL_SINGLET_ATOMS:
        full = report["infinitesimal"][name]
        half = report["infinitesimal_half"][name]
        assert full < 5e-7
        if full > 1e-16:
            assert half < 0.4 * full
    assert report["F_change"] > 10.0 * max(report["infinitesimal"].values(), default=0.0)
    if report["F_change_half"] > 1e-16:
        ratio = report["F_change"] / report["F_change_half"]
        assert 1.6 < ratio < 2.6


@pytest.mark.parametrize("n", [2, 3])
def test_global_equivariance_singlets(n: int) -> None:
    rng = np.random.default_rng(9 + n)
    if n == 2:
        pts = _sample_points(12, seed=9)
        a, da, dda = bpst_instanton_arrays(pts)
        algebra = la.su(2)
        coupling = 1.0
    else:
        pts = _sample_points(12, seed=10)
        algebra = la.su(3)
        a, da, dda = _polynomial_connection(pts, adjoint_dim=8, rng=rng)
        coupling = 0.7
    omega = rng.normal(size=(pts.shape[0], algebra.dim))
    report = gauge_equivariance_defect(
        a,
        da,
        dda,
        omega,
        algebra=algebra,
        coupling=coupling,
        signature=SIG_E,
        rng=rng,
    )
    floor = 1e-10 if n == 2 else 1e-8
    for name, defect in report["global"].items():
        assert defect < floor, name


def test_numpy_singlets_match_torch_kernel() -> None:
    torch = pytest.importorskip("torch")
    torch.set_default_dtype(torch.float64)
    import omnibias.geometry.gauge.torch.ops as tops

    pts = _sample_points(8, seed=11)
    a, da, dda = bpst_instanton_arrays(pts)
    su2 = la.su(2)
    jet = GaugeCovariantJet.from_arrays(
        a, da, dda, algebra=su2, coupling=1.0, signature=SIG_E
    )
    F_t = tops.field_strength_from_arrays(
        torch.as_tensor(a), torch.as_tensor(da), algebra=su2, coupling=1.0
    )
    DF_t = tops.covariant_derivative_field_strength_from_arrays(
        torch.as_tensor(a),
        torch.as_tensor(da),
        torch.as_tensor(dda),
        algebra=su2,
        coupling=1.0,
    )
    np.testing.assert_allclose(jet.F, F_t.detach().numpy(), atol=1e-11)
    np.testing.assert_allclose(jet.DF, DF_t.detach().numpy(), atol=1e-11)
    action_t = tops.action_density(F_t, signature=SIG_E).detach().numpy()
    np.testing.assert_allclose(jet.singlets()[SINGLET_TR_F2], action_t, atol=1e-11)


def test_self_dual_normalization_ratio() -> None:
    pts = _sample_points(20, seed=12)
    a, da, dda = bpst_instanton_arrays(pts)
    jet = GaugeCovariantJet.from_arrays(
        a, da, dda, algebra=la.su(2), coupling=1.0, signature=SIG_E
    )
    s = jet.singlets()
    ratio = s[SINGLET_TR_F2] / np.clip(np.abs(s[SINGLET_TR_F_FTILDE]), 1e-30, None)
    np.testing.assert_allclose(ratio, SELF_DUAL_ACTION_OVER_TOPOLOGICAL, rtol=1e-6)


def test_evaluate_gauge_law_gate_on_self_dual_identity() -> None:
    pts = _sample_points(16, seed=13)
    a, da, dda = bpst_instanton_arrays(pts)
    jet = GaugeCovariantJet.from_arrays(
        a, da, dda, algebra=la.su(2), coupling=1.0, signature=SIG_E
    )
    s = jet.singlets()
    equation = _Eq(SINGLET_TR_F_FTILDE, SELF_DUAL_ACTION_OVER_TOPOLOGICAL)
    gate = evaluate_gauge_law_gate(
        equation,
        lhs_name=SINGLET_TR_F2,
        A=a,
        dA=da,
        ddA=dda,
        algebra=la.su(2),
        coupling=1.0,
        signature=SIG_E,
        rng=np.random.default_rng(13),
        atol=1e-10,
    )
    assert gate["passed"] is True
    assert gate["yang_mills_claim"] is False
    assert gate["continuum_claim"] is False
    pred = equation.predict(s[SINGLET_TR_F_FTILDE][:, None])
    assert float(np.max(np.abs(pred - s[SINGLET_TR_F2]))) < 1e-8


def test_evaluate_gauge_law_gate_rejects_illegal_extra() -> None:
    pts = _sample_points(4, seed=14)
    a, da, dda = bpst_instanton_arrays(pts)
    equation = _Eq(SINGLET_TR_F_FTILDE, 1.0)
    with pytest.raises(ValueError, match="allowlisted"):
        evaluate_gauge_law_gate(
            equation,
            lhs_name=SINGLET_TR_F2,
            A=a,
            dA=da,
            ddA=dda,
            algebra=la.su(2),
            coupling=1.0,
            extra_columns={"|A|^2": np.sum(a**2, axis=(1, 2))},
        )
