# SPDX-License-Identifier: Apache-2.0
"""H7 polygonal-barrier and Positivstellensatz obstruction audit."""

from __future__ import annotations

import pytest
from omnibias.geometry.part_a_obstruction import (
    audit_part_a_obstruction_route,
    sos_basis_size,
    symmetric_gram_entries,
)


def test_full_octic_sos_size_is_already_large_at_half_degree_two() -> None:
    assert sos_basis_size(45, 0) == 1
    assert sos_basis_size(45, 1) == 46
    assert sos_basis_size(45, 2) == 1081
    assert symmetric_gram_entries(45, 1) == 1081
    assert symmetric_gram_entries(45, 2) == 584821
    with pytest.raises(ValueError, match="positive variable"):
        sos_basis_size(0, 2)


def test_h7_widens_barriers_but_does_not_obstruct_the_scheme() -> None:
    payload = audit_part_a_obstruction_route().to_payload()
    assert payload["schema"] == "hilbert16-part-a-polygon-sos-audit-v1"
    assert payload["target_annulus_count"] == 22
    assert payload["polygonal_layout_valid"] is True
    assert payload["route2_constraint_count"] == 1584
    assert payload["coefficient_dimensions"] == {
        "d4": 9,
        "klein": 15,
        "central": 25,
        "full": 45,
    }
    assert payload["route2_lp_ran"] is False
    assert payload["route2_lp_candidate_found"] is None
    assert payload["toy_positivstellensatz_empty_set_certified"] is True
    assert payload["scheme_basic_closed_encoding_supplied"] is False
    assert payload["symmetry_reduction_complete_for_all_realizations"] is False
    assert payload["octic_scheme_obstructed"] is False
    assert payload["part_a_22_oval_realized"] is False
    assert payload["hilbert16_part_a_solved"] is False
    assert payload["full_hilbert16_solved"] is False
