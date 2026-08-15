# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""BiasScan tests (theory 01-02): worked example, interior shift, honesty."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from omnibias.core.scan import BankSpec
from omnibias.torch.scan import (
    BiasScan,
    min_offset_separation,
    scan_response,
    soft_argmax_offset,
    template_from_op,
)


def _ulp_error(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("inf")
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b), 2.2250738585072014e-308)
    return abs(a - b) / (np.finfo(np.float64).eps * scale)


def _scan(
    *,
    bank: BankSpec,
    readout: str = "response",
    gamma: float = 8.0,
    learnable_offsets: bool = False,
) -> BiasScan:
    return BiasScan(
        1,
        bank,
        template="grad",
        base="tanh",
        learnable_offsets=learnable_offsets,
        learnable_scales=False,
        readout=readout,  # type: ignore[arg-type]
        gamma=gamma,
        dtype=torch.float64,
    )


def test_worked_example_matches_spec() -> None:
    """Spec §5: tanh grad bank at z=0.3, gamma=8, tau* ~ -0.283."""
    torch.set_default_dtype(torch.float64)
    bank = BankSpec((-1.0, -0.5, 0.0, 0.5, 1.0))
    z = torch.tensor([[0.3]], dtype=torch.float64)
    resp = _scan(bank=bank, readout="response")(z).squeeze()
    expected_u = z.item() + torch.tensor(bank.offsets, dtype=torch.float64)
    expected_r = 1.0 - torch.tanh(expected_u).pow(2)
    assert torch.equal(resp, expected_r)
    offs = torch.tensor(bank.offsets, dtype=torch.float64)
    tau_star = float(soft_argmax_offset(resp, offs, gamma=8.0))
    weights = torch.softmax(8.0 * resp, dim=0)
    expected_tau = float((weights * offs).sum())
    assert tau_star == expected_tau
    # Spec §5 quoted tau* ~ -0.283 from a rounded R-table; the exact
    # softmax of 1-tanh^2 is within a tenth of a spacing of -z = -0.3.
    assert abs(tau_star + 0.3) <= 0.1 * 0.5


def test_g1_interior_shift_within_4_ulp() -> None:
    """On-lattice shift is an interior slide, not a circular wrap of tanh'."""
    torch.set_default_dtype(torch.float64)
    bank = BankSpec.uniform(-1.0, 1.0, 5)
    spacing = bank.spacing
    assert spacing is not None
    scan = _scan(bank=bank, readout="response")
    z = torch.tensor([[0.15]], dtype=torch.float64)
    r0 = scan(z)
    r_shift = scan(z + spacing)
    left = r_shift[..., :-1]
    right = r0[..., 1:]
    worst = 0.0
    for a, b in zip(left.reshape(-1).tolist(), right.reshape(-1).tolist(), strict=True):
        worst = max(worst, _ulp_error(a, b))
    assert worst <= 4.0, f"interior-shift worst_ulp={worst}"


def test_off_lattice_error_drops_when_spacing_halves() -> None:
    """Off-lattice: interpolating R(z) to a shift δ vs R(z+δ); error drops as h halves."""
    torch.set_default_dtype(torch.float64)
    z0 = 0.3
    delta = 0.1

    def _err(n: int) -> float:
        bank = BankSpec.uniform(-1.0, 1.0, n)
        scan = _scan(bank=bank, readout="response")
        z = torch.tensor([[z0]], dtype=torch.float64)
        r0 = scan(z).detach().numpy().reshape(-1)
        rd = scan(z + delta).detach().numpy().reshape(-1)
        offs = np.asarray(bank.offsets, dtype=np.float64)
        pred = np.interp(offs + delta, offs, r0)
        mask = (offs + delta >= offs[0]) & (offs + delta <= offs[-1])
        return float(np.max(np.abs(pred[mask] - rd[mask])))

    assert _err(9) < _err(5)


def test_soft_argmax_matches_finite_difference_dtau_dz() -> None:
    torch.set_default_dtype(torch.float64)
    bank = BankSpec.uniform(-1.0, 1.0, 5)
    scan = _scan(bank=bank, readout="argmax")
    z = torch.tensor([[0.3]], dtype=torch.float64, requires_grad=True)
    tau = scan(z)
    tau.backward()
    assert z.grad is not None
    analytic = float(z.grad.item())
    eps = 1e-6
    with torch.no_grad():
        tp = float(scan(torch.tensor([[0.3 + eps]], dtype=torch.float64)))
        tm = float(scan(torch.tensor([[0.3 - eps]], dtype=torch.float64)))
    fd = (tp - tm) / (2.0 * eps)
    assert analytic == pytest.approx(fd, rel=1e-4, abs=1e-6)


def test_two_interface_soft_argmax_is_biased_visible() -> None:
    """Two peaks one spacing apart: soft-argmax collapses to 0; do not hide it."""
    torch.set_default_dtype(torch.float64)
    bank = BankSpec((-1.0, -0.5, 0.0, 0.5, 1.0))
    offsets = torch.tensor(bank.offsets, dtype=torch.float64)
    scales = torch.tensor(bank.scales, dtype=torch.float64)
    from omnibias.torch.activations.registry import get_activation

    spec = template_from_op("grad")
    base = get_activation("tanh")
    r1 = scan_response(
        torch.tensor([[0.4]], dtype=torch.float64), offsets, scales, spec, base
    )
    r2 = scan_response(
        torch.tensor([[-0.4]], dtype=torch.float64), offsets, scales, spec, base
    )
    combined = r1 + r2
    tau_star = float(soft_argmax_offset(combined.squeeze(), offsets, gamma=8.0))
    assert abs(tau_star) < 0.15
    assert abs(tau_star - 0.4) > 0.2
    assert abs(tau_star + 0.4) > 0.2


def test_min_offset_separation_on_learnable_offsets() -> None:
    torch.set_default_dtype(torch.float64)
    bank = BankSpec.uniform(-1.0, 1.0, 5)
    scan = _scan(bank=bank, learnable_offsets=True)
    assert float(scan.min_offset_separation().detach()) == pytest.approx(0.5)
    with torch.no_grad():
        scan.offsets[0] = scan.offsets[1]
    assert float(scan.min_offset_separation().detach()) == pytest.approx(0.0)
    collapsed = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    assert float(min_offset_separation(collapsed)) == pytest.approx(0.0)


def test_dtype_none_uses_framework_default() -> None:
    prev = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        scan = BiasScan(1, BankSpec.uniform(-1.0, 1.0, 3), dtype=None)
        assert scan.offsets.dtype == torch.float64
    finally:
        torch.set_default_dtype(prev)


def test_band_and_integral_templates_run() -> None:
    torch.set_default_dtype(torch.float64)
    bank = BankSpec.uniform(-1.0, 1.0, 3)
    z = torch.zeros(2, 1, dtype=torch.float64)
    band = BiasScan(1, bank, template="band", learnable_offsets=False, dtype=torch.float64)
    integ = BiasScan(
        1, bank, template="integral", base="tanh", learnable_offsets=False, dtype=torch.float64
    )
    assert band(z).shape == (2, 1, 3)
    assert integ(z).shape == (2, 1, 3)
    with pytest.raises(ValueError, match="unknown"):
        template_from_op("not-an-op")
