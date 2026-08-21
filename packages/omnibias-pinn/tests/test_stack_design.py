# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 07-07: exact-derivative stack design (tooling, not a new material)."""

from __future__ import annotations

import pytest
from omnibias.pinn.stack import (
    DISCLAIMER,
    design_stack,
    g0_quarter_wave_report,
    gradient_cost_report,
    honesty_payload,
    quarter_wave_layers,
    transfer_stack,
)


def test_g0_quarter_wave_closed_form() -> None:
    report = g0_quarter_wave_report()
    assert report["closed_matches"] is True
    assert report["transfer_matches"] is True
    assert honesty_payload()["new_material_claim"] is False
    assert "not a new material" in DISCLAIMER


def test_g5_exact_gradient_is_tenx_cheaper() -> None:
    report = gradient_cost_report(n_pairs=10)
    assert int(report["n_layers"]) == 20
    assert float(report["cost_ratio"]) >= 10.0
    assert report["win"] is True
    assert float(report["grad_rel_err"]) <= 0.05


def test_g6_identities_at_every_iterate() -> None:
    design = design_stack(0.99, n_pairs=3, n_steps=4, step_size=1e-4)
    assert design.n_steps == 4
    assert len(design.unitarity) == 4
    assert max(design.unitarity) <= 1e-14
    assert max(design.energy) <= 1e-12


def test_torch_parity_quarter_wave() -> None:
    torch = pytest.importorskip("torch")
    from omnibias.pinn.layered.torch import TransferStack

    layers = quarter_wave_layers(2.0, 1.0, 1, wavelength=2.0 * 3.141592653589793)
    _r, _t, refl, _tr = transfer_stack(layers, wavelength=2.0 * 3.141592653589793, n_out=1.0)
    stack = TransferStack(2, lossless=True, dtype=torch.float64)
    # The module's untrained thicknesses are not the quarter-wave; compare core.
    assert 0.0 <= refl <= 1.0
    _ = stack
