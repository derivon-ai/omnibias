# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Invariant dictionary: dimension census, Weyl roles, new densities."""

from __future__ import annotations

import numpy as np
import omnibias.geometry.gauge._core.lie_algebra as la
import pytest
from omnibias.geometry.gauge._core import kernels
from omnibias.geometry.gauge._core.covariant_jet import (
    SINGLET_BIANCHI_SQ,
    SINGLET_SELF_DUAL_SQ,
    SINGLET_TR_F2,
    SINGLET_TR_F_FTILDE,
    GaugeCovariantJet,
    global_gauge_transform_connection,
    random_special_unitary,
)
from omnibias.geometry.gauge._core.instanton import bpst_instanton_arrays
from omnibias.geometry.gauge._core.invariants import (
    MAX_SEARCHABLE_DIM6_SU3,
    SINGLET_DF_SQ,
    SINGLET_TR_F3,
    GaugeInvariantDictionary,
    enumerate_gauge_invariants,
    evaluate_named_invariants,
    representation_complexity,
)
from omnibias.geometry.gauge._core.jet_dimension import (
    SU3_4D_COORDINATE_2JET,
    SU3_4D_DF_BIANCHI_REDUCED,
    SU3_4D_DF_RAW,
    SU3_4D_F,
    bianchi_reduced_df_dimension,
    raw_connection_jet_dimension,
    raw_covariant_fiber_dimension,
    refuse_component_fiber_library,
    refuse_coordinate_jet_library,
    refuse_flattened_adjoint_library,
)

SIG_E = (1, 1, 1, 1)


def _polynomial_connection(
    points: np.ndarray,
    *,
    adjoint_dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def test_su3_coordinate_2jet_is_480() -> None:
    assert raw_connection_jet_dimension(4, 8, 2) == SU3_4D_COORDINATE_2JET
    assert SU3_4D_COORDINATE_2JET == 480


def test_su3_covariant_fibers() -> None:
    assert raw_covariant_fiber_dimension(4, 8, 0) == SU3_4D_F == 48
    assert raw_covariant_fiber_dimension(4, 8, 1) == SU3_4D_DF_RAW == 192
    assert bianchi_reduced_df_dimension(4, 8) == SU3_4D_DF_BIANCHI_REDUCED == 160


def test_searchable_dictionary_is_tiny() -> None:
    dim4 = GaugeInvariantDictionary.build(
        mass_dimension=4, max_cov_order=1, algebra=la.su(3)
    )
    assert dim4.legal_names == frozenset({SINGLET_TR_F2, SINGLET_TR_F_FTILDE})
    dim6 = GaugeInvariantDictionary.build(
        mass_dimension=6, max_cov_order=1, algebra=la.su(3)
    )
    assert SINGLET_TR_F3 not in dim6.legal_names
    assert SINGLET_DF_SQ in dim6.legal_names
    assert SINGLET_BIANCHI_SQ not in dim6.legal_names
    assert SINGLET_SELF_DUAL_SQ not in dim6.legal_names
    assert len(dim6.legal_names) <= MAX_SEARCHABLE_DIM6_SU3
    su2_dim6 = GaugeInvariantDictionary.build(
        mass_dimension=6, max_cov_order=1, algebra=la.su(2)
    )
    assert SINGLET_TR_F3 not in su2_dim6.legal_names


def test_bianchi_is_identity_and_self_dual_is_syzygy() -> None:
    atoms = enumerate_gauge_invariants(
        mass_dimension=6, max_cov_order=1, algebra=la.su(3)
    )
    roles = {atom.name: atom.role for atom in atoms}
    assert roles[SINGLET_BIANCHI_SQ] == "identity"
    assert roles[SINGLET_SELF_DUAL_SQ] == "syzygy"
    assert roles[SINGLET_TR_F3] == "syzygy"
    assert roles[SINGLET_TR_F2] == "search"


def test_max_cov_order_two_still_contracted_only() -> None:
    atoms = enumerate_gauge_invariants(
        mass_dimension=6, max_cov_order=2, algebra=la.su(3)
    )
    assert all(atom.cov_order <= 1 for atom in atoms)
    with pytest.raises(ValueError, match="k>=2"):
        refuse_component_fiber_library(cov_order=2)


def test_refuse_coordinate_and_flattened() -> None:
    with pytest.raises(ValueError, match="coordinate 2-jet"):
        refuse_coordinate_jet_library()
    with pytest.raises(ValueError, match="allowlisted"):
        refuse_flattened_adjoint_library(["F_01_2", "DF_0_12_1", "A_0_0"])


def test_df_square_matches_einsum() -> None:
    rng = np.random.default_rng(1)
    pts = rng.uniform(-0.6, 0.6, size=(8, 4))
    a, da, dda = _polynomial_connection(pts, adjoint_dim=3, rng=rng)
    jet = GaugeCovariantJet.from_arrays(
        a, da, dda, algebra=la.su(2), coupling=0.8, signature=SIG_E
    )
    eta = np.asarray(SIG_E, dtype=np.float64)
    expected = np.einsum("r,m,n,Brmna,Brmna->B", eta, eta, eta, jet.DF, jet.DF)
    got = kernels.df_square_density(np, jet.DF, eta)
    np.testing.assert_allclose(got, expected, atol=1e-11)
    cols = evaluate_named_invariants(jet, [SINGLET_DF_SQ])
    np.testing.assert_allclose(cols[SINGLET_DF_SQ], expected, atol=1e-11)


def test_tr_f3_vanishes_on_su2() -> None:
    rng = np.random.default_rng(2)
    pts = rng.uniform(-0.6, 0.6, size=(8, 4))
    a, da, dda = _polynomial_connection(pts, adjoint_dim=3, rng=rng)
    jet = GaugeCovariantJet.from_arrays(
        a, da, dda, algebra=la.su(2), coupling=0.7, signature=SIG_E
    )
    cols = evaluate_named_invariants(jet, [SINGLET_TR_F3])
    assert float(np.max(np.abs(cols[SINGLET_TR_F3]))) < 1e-11


def test_tr_f3_is_4d_syzygy_and_df_sq_invariant_on_su3() -> None:
    rng = np.random.default_rng(3)
    pts = rng.uniform(-0.5, 0.5, size=(10, 4))
    a, da, dda = _polynomial_connection(pts, adjoint_dim=8, rng=rng)
    algebra = la.su(3)
    jet = GaugeCovariantJet.from_arrays(
        a, da, dda, algebra=algebra, coupling=0.6, signature=SIG_E
    )
    base = evaluate_named_invariants(jet, [SINGLET_TR_F3, SINGLET_DF_SQ])
    assert np.all(np.isfinite(base[SINGLET_TR_F3]))
    assert float(np.max(np.abs(base[SINGLET_TR_F3]))) < 1e-11
    assert float(np.max(np.abs(base[SINGLET_DF_SQ]))) > 1e-8
    u = random_special_unitary(3, rng)
    a_g, da_g, dda_g = global_gauge_transform_connection(
        a, da, dda, u, algebra=algebra
    )
    jet_g = GaugeCovariantJet.from_arrays(
        a_g, da_g, dda_g, algebra=algebra, coupling=0.6, signature=SIG_E
    )
    transformed = evaluate_named_invariants(jet_g, [SINGLET_TR_F3, SINGLET_DF_SQ])
    np.testing.assert_allclose(
        transformed[SINGLET_TR_F3], base[SINGLET_TR_F3], atol=1e-8
    )
    np.testing.assert_allclose(
        transformed[SINGLET_DF_SQ], base[SINGLET_DF_SQ], atol=1e-8
    )


def test_new_densities_invariant_on_bpst() -> None:
    rng = np.random.default_rng(4)
    pts = rng.uniform(-1.2, 1.2, size=(12, 4))
    pts = pts[np.linalg.norm(pts, axis=1) > 0.4]
    a, da, dda = bpst_instanton_arrays(pts)
    jet = GaugeCovariantJet.from_arrays(
        a, da, dda, algebra=la.su(2), coupling=1.0, signature=SIG_E
    )
    base = evaluate_named_invariants(jet, [SINGLET_DF_SQ])
    u = random_special_unitary(2, rng)
    a_g, da_g, dda_g = global_gauge_transform_connection(
        a, da, dda, u, algebra=la.su(2)
    )
    jet_g = GaugeCovariantJet.from_arrays(
        a_g, da_g, dda_g, algebra=la.su(2), coupling=1.0, signature=SIG_E
    )
    transformed = evaluate_named_invariants(jet_g, [SINGLET_DF_SQ])
    np.testing.assert_allclose(
        transformed[SINGLET_DF_SQ], base[SINGLET_DF_SQ], atol=1e-10
    )


def test_representation_complexity_penalizes_mass_dimension() -> None:
    atoms = {
        atom.name: atom
        for atom in enumerate_gauge_invariants(
            mass_dimension=6, max_cov_order=1, algebra=la.su(3)
        )
    }
    c4 = representation_complexity([SINGLET_TR_F2], atoms)
    c6 = representation_complexity([SINGLET_DF_SQ], atoms)
    assert c6 > c4
    assert c4 == 5
    assert c6 == 7


def test_jet_singlets_still_the_point1_five() -> None:
    rng = np.random.default_rng(5)
    pts = rng.uniform(-0.8, 0.8, size=(4, 4))
    a, da, dda = _polynomial_connection(pts, adjoint_dim=3, rng=rng)
    jet = GaugeCovariantJet.from_arrays(
        a, da, dda, algebra=la.su(2), coupling=1.0, signature=SIG_E
    )
    from omnibias.geometry.gauge._core.covariant_jet import LEGAL_SINGLET_ATOMS

    assert set(jet.singlets()) == LEGAL_SINGLET_ATOMS
    assert SINGLET_DF_SQ not in jet.singlets()
