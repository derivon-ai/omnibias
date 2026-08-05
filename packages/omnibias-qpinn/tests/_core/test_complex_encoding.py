# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the backend-agnostic complex encoding helpers."""

from __future__ import annotations

import math

import pytest
from omnibias.pinn._core.components import ComponentSpec
from omnibias.qpinn import (
    AMU_TO_ME,
    BOHR_TO_ANGSTROM,
    HARTREE_TO_CM,
    HARTREE_TO_EV,
    make_psi_components,
)


class TestMakePsiComponents:
    def test_default_names(self):
        spec = make_psi_components()
        assert spec.names == ("psi_re", "psi_im")
        assert spec.group_members("psi") == ("psi_re", "psi_im")

    def test_custom_name(self):
        spec = make_psi_components(name="phi")
        assert spec.names == ("phi_re", "phi_im")
        assert spec.group_members("phi") == ("phi_re", "phi_im")

    def test_extra_groups_alias(self):
        spec = make_psi_components(
            name="psi",
            extra_groups={"alias": ("psi_re", "psi_im")},
        )
        assert spec.group_members("psi") == ("psi_re", "psi_im")
        assert spec.group_members("alias") == ("psi_re", "psi_im")

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            make_psi_components(name="")

    def test_suffix_collision_rejected(self):
        with pytest.raises(ValueError, match="_re"):
            make_psi_components(name="foo_re")
        with pytest.raises(ValueError, match="_im"):
            make_psi_components(name="foo_im")

    def test_extra_group_member_outside_spec(self):
        with pytest.raises(ValueError, match="not in"):
            make_psi_components(
                name="psi", extra_groups={"bad": ("psi_re", "other")},
            )

    def test_extra_group_name_collides_with_psi(self):
        with pytest.raises(ValueError, match="collides"):
            make_psi_components(
                name="psi", extra_groups={"psi": ("psi_re", "psi_im")},
            )

    def test_returns_componentspec(self):
        spec = make_psi_components()
        assert isinstance(spec, ComponentSpec)
        assert len(spec.names) == 2


class TestAtomicUnitsConstants:
    """Pin the constants to CODATA 2018 values within machine precision."""

    def test_hartree_to_cm_value(self):
        # CODATA 2018: E_h / (h c) = 219474.6313632 cm^-1
        assert math.isclose(HARTREE_TO_CM, 219474.6313632, rel_tol=1e-12)

    def test_hartree_to_ev_value(self):
        # CODATA 2018: E_h in eV
        assert math.isclose(HARTREE_TO_EV, 27.211386245988, rel_tol=1e-12)

    def test_bohr_to_angstrom_value(self):
        # CODATA 2018: a_0 in angstrom
        assert math.isclose(BOHR_TO_ANGSTROM, 0.52917721067, rel_tol=1e-12)

    def test_amu_to_me_value(self):
        # CODATA 2018: 1 amu in electron masses
        assert math.isclose(AMU_TO_ME, 1822.888486209, rel_tol=1e-12)

    def test_constants_are_floats(self):
        for const in (HARTREE_TO_CM, HARTREE_TO_EV, BOHR_TO_ANGSTROM, AMU_TO_ME):
            assert isinstance(const, float)

    @pytest.mark.skipif(
        pytest.importorskip("jax", reason="jax backend not installed") is None,
        reason="needs jax",
    )
    def test_consistent_with_bo_derivatives(self):
        """When jax is installed, our constants must match the source of truth."""
        from omnibias.jax import bo_derivatives as bo

        assert HARTREE_TO_CM == bo.HARTREE_TO_CM
        assert BOHR_TO_ANGSTROM == bo.BOHR_TO_ANGSTROM
        assert AMU_TO_ME == bo.AMU_TO_ME
