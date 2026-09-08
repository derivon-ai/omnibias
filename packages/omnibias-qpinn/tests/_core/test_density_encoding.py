# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Split-real density-matrix ComponentSpec helpers."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from omnibias.pinn._core.components import ComponentSpec
from omnibias.qpinn import (
    LindbladSpec,
    make_rho_components,
    parse_rho_entry_name,
    rho_entry_names,
)


def test_make_rho_components_qubit() -> None:
    spec = make_rho_components()
    assert isinstance(spec, ComponentSpec)
    assert len(spec.names) == 8
    assert spec.group_members("rho") == spec.names
    assert spec.group_members("rho_0_0") == ("rho_re_0_0", "rho_im_0_0")
    assert rho_entry_names("rho", 1, 0) == ("rho_re_1_0", "rho_im_1_0")
    assert parse_rho_entry_name("rho_im_1_0", group="rho") == ("im", 1, 0)


def test_make_rho_components_rejects_bad_dim() -> None:
    with pytest.raises(ValueError, match="positive"):
        make_rho_components(dim=0)
    with pytest.raises(ValueError, match="non-empty"):
        make_rho_components(name="")


def test_lindblad_spec_is_frozen() -> None:
    spec = LindbladSpec(
        hamiltonian=((0.0, 0.0), (0.0, 1.0)),
        jumps=(((0.0, 1.0), (0.0, 0.0)),),
        rates=(0.5,),
        dim=2,
    )
    assert spec.group == "rho"
    with pytest.raises(FrozenInstanceError):
        spec.dim = 3  # type: ignore[misc]
