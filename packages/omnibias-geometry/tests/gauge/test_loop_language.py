# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Loop language: Wilson/Polyakov atoms, W(1,1)=plaquette, language-split refuses."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import omnibias.geometry.gauge._core.lie_algebra as la
import pytest
from omnibias.geometry.gauge._core.covariant_jet import GaugeCovariantJet
from omnibias.geometry.gauge._core.data_paths import LatticeLinkField
from omnibias.geometry.gauge._core.instanton import bpst_instanton_arrays
from omnibias.geometry.gauge._core.loop_language import (
    LEGAL_LOOP_ATOMS,
    LOOP_INVARIANCE_ATOL,
    LOOP_PLAQUETTE,
    LoopObservableTable,
    creutz_ratio_from_wilson,
    evaluate_loop_atoms,
    evaluate_loop_gauge_invariance,
    identity_numpy_links,
    is_green_column_name,
    is_loop_atom_name,
    random_numpy_links,
    refuse_green_as_jet_atom,
    refuse_jet_as_loop_source,
    refuse_loop_as_covariant_jet,
    wilson_plaquette_pairs,
)
from omnibias.geometry.gauge.lattice._core.kernels import plaquette_trace

SIG_E = (1, 1, 1, 1)
W11_PLAQUETTE_ATOL = 1e-12
COLD_START_ATOL = 1e-15


def _bpst_jet() -> GaugeCovariantJet:
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.2, 1.2, size=(8, 4))
    pts = pts[np.linalg.norm(pts, axis=1) > 0.4][:8]
    a, da, dda = bpst_instanton_arrays(pts)
    return GaugeCovariantJet.from_arrays(
        a, da, dda, algebra=la.su(2), coupling=1.0, signature=SIG_E
    )


def test_legal_loop_atoms_exclude_certificates() -> None:
    assert LEGAL_LOOP_ATOMS == {
        "plaquette",
        "W(1,1)",
        "W(2,1)",
        "W(1,2)",
        "W(2,2)",
        "Polyakov",
    }
    assert "gevp" not in LEGAL_LOOP_ATOMS
    assert "transfer_gap" not in LEGAL_LOOP_ATOMS


def test_w11_equals_plaquette_identity_and_random() -> None:
    ident = LatticeLinkField(links=identity_numpy_links((2, 2, 2, 2)))
    w_i, p_i = wilson_plaquette_pairs(ident)
    assert float(np.max(np.abs(w_i - p_i))) < W11_PLAQUETTE_ATOL
    rng = np.random.default_rng(1)
    field = LatticeLinkField(links=random_numpy_links((4, 4, 4, 4), rng))
    w_r, p_r = wilson_plaquette_pairs(field)
    assert float(np.max(np.abs(w_r - p_r))) < W11_PLAQUETTE_ATOL
    links = np.asarray(field.links)
    plane = np.asarray(plaquette_trace(np, links, 0, 3), dtype=np.float64)
    assert float(np.max(np.abs(plane))) <= 1.0 + 1e-12


def test_cold_start_wilson_polyakov_are_one() -> None:
    table = evaluate_loop_atoms(
        LatticeLinkField(links=identity_numpy_links((4, 4, 4, 4)))
    )
    for name in LEGAL_LOOP_ATOMS:
        assert float(np.max(np.abs(table.values[name] - 1.0))) < COLD_START_ATOL


def test_language_split_refuses_before_jet() -> None:
    called = {"from_arrays": False}

    def _boom(*_args: object, **_kwargs: object) -> None:
        called["from_arrays"] = True
        raise AssertionError("from_arrays must not run")

    jet = _bpst_jet()
    with patch.object(GaugeCovariantJet, "from_arrays", _boom):
        with pytest.raises(ValueError, match="holonomy"):
            refuse_jet_as_loop_source(jet)
        with pytest.raises(ValueError, match="language split"):
            refuse_loop_as_covariant_jet(["W(2,2)"])
        with pytest.raises(ValueError, match="lattice"):
            GaugeCovariantJet.from_lattice_links(identity_numpy_links((2, 2, 2, 2)))
    assert called["from_arrays"] is False


def test_green_and_loop_name_guards() -> None:
    assert is_loop_atom_name("W(2,2)")
    assert is_loop_atom_name("Polyakov")
    assert is_loop_atom_name("gevp")
    assert not is_loop_atom_name("tr(F^2)")
    assert is_green_column_name("inverse_laplacian")
    assert is_green_column_name("green(x)")
    assert is_green_column_name("delta^{-1}")
    with pytest.raises(ValueError, match="inverse Laplacian"):
        refuse_green_as_jet_atom(["inverse_laplacian"])


def test_loop_atoms_gauge_invariant() -> None:
    rng = np.random.default_rng(2)
    shape = (3, 3, 3, 3)
    field = LatticeLinkField(links=random_numpy_links(shape, rng))
    raw = rng.normal(size=(*shape, 4))
    g = raw / np.maximum(np.linalg.norm(raw, axis=-1, keepdims=True), 1e-30)
    report = evaluate_loop_gauge_invariance(field, g, atol=LOOP_INVARIANCE_ATOL)
    assert report["passed"] is True
    assert float(report["max_abs"]) < LOOP_INVARIANCE_ATOL
    assert report["yang_mills_claim"] is False
    assert report["continuum_claim"] is False


def test_creutz_on_planted_area_law() -> None:
    sigma = 0.2
    values = {
        f"W({r_ext},{t_ext})": np.full(4, float(np.exp(-sigma * r_ext * t_ext)))
        for r_ext, t_ext in ((1, 1), (2, 1), (1, 2), (2, 2))
    }
    chi = creutz_ratio_from_wilson(LoopObservableTable(values=values))
    assert chi == pytest.approx(sigma, rel=1e-12)
    assert LOOP_PLAQUETTE not in values
    assert "gevp" not in values
