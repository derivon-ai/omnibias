# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wilson-rectangle → Path B table, V(r), Creutz atom."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.geometry.gauge._core.ensemble_language import (
    LEGAL_ENSEMBLE_ATOMS,
    assert_library_ensemble_legal,
    ensemble_table_from_mc_dict,
    static_potential_from_wilson,
    wilson_loops_to_ensemble_table,
)


def _planted_wilson_loops(*, sigma: float = 0.2, kappa: float = 0.05, n_side: int = 5):
    out: dict[str, dict[str, float]] = {}
    for r in range(1, n_side + 1):
        for t in range(1, n_side + 1):
            area = r * t
            peri = 2 * (r + t)
            weight = float(np.exp(-sigma * area - kappa * peri))
            out[f"{r}x{t}"] = {"value": weight, "err": 0.0}
    return out


def test_wilson_loops_to_table_has_area_creutz_and_potential() -> None:
    table = wilson_loops_to_ensemble_table(
        _planted_wilson_loops(),
        lattice_shape=(8, 8, 8, 8),
        n_configs=4,
    )
    assert "area" in table.values
    assert "perimeter" in table.values
    assert "t_wilson" in table.values
    assert "creutz_chi" in table.values
    assert "V_r" in table.values
    assert "F_r" in table.values
    assert "L_lat" in table.values
    assert table.values["L_lat"][0] == pytest.approx(8.0)
    finite = table.values["creutz_chi"][np.isfinite(table.values["creutz_chi"])]
    assert finite.size > 0
    assert float(np.mean(finite)) == pytest.approx(0.2, rel=5e-2)
    assert "gevp" not in LEGAL_ENSEMBLE_ATOMS
    with pytest.raises(ValueError, match="allowlisted"):
        assert_library_ensemble_legal(["creutz_chi", "gevp"])


def test_static_potential_plateau_is_linear() -> None:
    table = wilson_loops_to_ensemble_table(_planted_wilson_loops(sigma=0.25, kappa=0.0))
    unique = {}
    for r, v in zip(table.values["r"], table.values["V_r"], strict=True):
        unique[float(r)] = float(v)
    radii = np.asarray(sorted(unique))
    pots = np.asarray([unique[float(r)] for r in radii])
    slope = float(np.polyfit(radii, pots, 1)[0])
    assert slope == pytest.approx(0.25, rel=1e-1)
    again = static_potential_from_wilson(table)
    assert again.values["V_r"].shape == table.values["V_r"].shape


def test_mc_dict_ingests_wilson_loops() -> None:
    table = ensemble_table_from_mc_dict(
        {
            "wilson_loops": _planted_wilson_loops(),
            "lattice_shape": (6, 6, 6, 6),
            "beta": 2.3,
        }
    )
    assert "log_C_P" in table.values
    assert "creutz_chi" in table.values
    assert table.metadata.beta == pytest.approx(2.3)
