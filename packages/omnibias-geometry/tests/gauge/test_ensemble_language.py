# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Path B ensemble language: allowlist, adapters, language-split refuses."""

from __future__ import annotations

import numpy as np
import omnibias.geometry.gauge._core.lie_algebra as la
import pytest
from omnibias.geometry.gauge._core.covariant_jet import GaugeCovariantJet
from omnibias.geometry.gauge._core.data_paths import LatticeLinkField
from omnibias.geometry.gauge._core.ensemble_language import (
    LEGAL_ENSEMBLE_ATOMS,
    assert_library_ensemble_legal,
    ensemble_table_from_link_ensemble,
    ensemble_table_from_mc_dict,
    is_ensemble_atom_name,
    refuse_cert_as_ensemble_atom,
    refuse_jet_as_ensemble_source,
    refuse_loop_table_as_ensemble,
    refuse_single_config_as_ensemble,
)
from omnibias.geometry.gauge._core.instanton import bpst_instanton_arrays
from omnibias.geometry.gauge._core.loop_language import (
    LoopObservableTable,
    identity_numpy_links,
    random_numpy_links,
)

SIG_E = (1, 1, 1, 1)


def _bpst_jet() -> GaugeCovariantJet:
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.2, 1.2, size=(8, 4))
    pts = pts[np.linalg.norm(pts, axis=1) > 0.4][:8]
    a, da, dda = bpst_instanton_arrays(pts)
    return GaugeCovariantJet.from_arrays(
        a, da, dda, algebra=la.su(2), coupling=1.0, signature=SIG_E
    )


def test_legal_ensemble_atoms_exclude_certificates() -> None:
    assert LEGAL_ENSEMBLE_ATOMS == {
        "abs_P",
        "chi_P",
        "log_abs_P",
        "log_abs_t",
        "C_P",
        "log_C_P",
        "r",
        "G_p2",
        "p2",
        "rho",
        "omega",
        "area",
        "perimeter",
        "log_p2",
        "inv_p2",
        "inv_p2_sq",
        "ghost_G",
        "log_ghost_G",
        "T_lat",
        "V_r",
        "t_wilson",
        "creutz_chi",
        "L_lat",
        "F_r",
        "sigma_lat",
        "Lambda_QCD",
        "r0",
        "t0",
        "a_lat",
        "log_p2_over_L2",
        "p2_g025",
        "p2_g05",
        "p2_g075",
        "p2_g1",
        "log_p2_g025",
        "log_p2_g05",
        "log_p2_g075",
        "log_p2_g1",
        "Li2_z",
        "Li3_z",
        "F21_z",
        "glueball_mass",
    }
    assert "gevp" not in LEGAL_ENSEMBLE_ATOMS
    assert "transfer_gap" not in LEGAL_ENSEMBLE_ATOMS
    assert is_ensemble_atom_name("abs_P")
    assert is_ensemble_atom_name("G_p2")
    assert is_ensemble_atom_name("rho")
    assert is_ensemble_atom_name("gevp")


def test_assert_library_refuses_certs() -> None:
    with pytest.raises(ValueError, match="allowlisted"):
        assert_library_ensemble_legal(["abs_P", "gevp"])


def test_one_config_is_not_an_ensemble() -> None:
    field = LatticeLinkField(links=identity_numpy_links((2, 2, 2, 2)))
    with pytest.raises(ValueError, match="single"):
        ensemble_table_from_link_ensemble([field], beta=2.3)
    with pytest.raises(ValueError, match="single"):
        refuse_single_config_as_ensemble(field)


def test_jet_and_loop_table_are_refused() -> None:
    jet = _bpst_jet()
    with pytest.raises(ValueError, match="ensemble"):
        refuse_jet_as_ensemble_source(jet)
    table = LoopObservableTable(values={"plaquette": np.ones(2)}, source="lattice_links")
    with pytest.raises(ValueError, match="LoopObservableTable|holonomy|language"):
        refuse_loop_table_as_ensemble(table)
    with pytest.raises(ValueError, match="certificate"):
        refuse_cert_as_ensemble_atom(["gevp"])


def test_link_ensemble_identity_has_unit_polyakov_and_flat_correlator() -> None:
    fields = [
        LatticeLinkField(links=identity_numpy_links((4, 4, 4, 4))),
        LatticeLinkField(links=identity_numpy_links((4, 4, 4, 4))),
    ]
    table = ensemble_table_from_link_ensemble(fields, beta=2.3, beta_c=2.0)
    assert table.source == "lattice_ensemble"
    assert table.values["abs_P"].item() == pytest.approx(1.0, abs=1e-12)
    assert table.values["chi_P"].item() == pytest.approx(0.0, abs=1e-12)
    assert float(np.max(np.abs(table.values["C_P"]))) < 1e-12


def test_link_ensemble_two_random_configs() -> None:
    rng = np.random.default_rng(2)
    fields = [
        LatticeLinkField(links=random_numpy_links((4, 4, 4, 4), rng)),
        LatticeLinkField(links=random_numpy_links((4, 4, 4, 4), rng)),
    ]
    table = ensemble_table_from_link_ensemble(fields, beta=1.5)
    assert table.values["abs_P"].shape == (1,)
    assert table.values["C_P"].shape[0] == table.values["r"].shape[0]
    assert 0.0 <= float(table.values["abs_P"].item()) <= 1.0 + 1e-12


def test_link_ensemble_records_metadata_and_finite_t() -> None:
    fields = [
        LatticeLinkField(links=identity_numpy_links((4, 4, 4, 2))),
        LatticeLinkField(links=identity_numpy_links((4, 4, 4, 2))),
    ]
    table = ensemble_table_from_link_ensemble(fields, beta=2.3)
    assert table.metadata.n_configs == 2
    assert table.metadata.beta == pytest.approx(2.3)
    assert table.metadata.lattice_shape == (4, 4, 4, 2)
    assert table.values["T_lat"].item() == pytest.approx(0.5)
    assert table.values["L_lat"].item() == pytest.approx(4.0)


def test_mc_dict_adapter_without_live_mc() -> None:
    table = ensemble_table_from_mc_dict(
        {
            "avg_polyakov": -0.25,
            "glueball_correlator": [0.4, 0.2, 0.1],
        }
    )
    assert table.source == "lattice_ensemble"
    assert table.values["abs_P"].item() == pytest.approx(0.25)
    assert table.values["C_P"].tolist() == pytest.approx([0.4, 0.2, 0.1])
    assert table.values["r"].tolist() == pytest.approx([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="no ensemble atoms"):
        ensemble_table_from_mc_dict({"beta": 2.3})
