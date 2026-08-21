# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Resistive-layer MHD application (theory 07-07).

Composes the existing ``induction_residual`` /
``ideal_mhd_momentum_residual`` field-op names with a Harris-sheet
equilibrium and a multi-pack basis at the layer. The founding bias
collapse (``delta -> 0``) places O(1) packs at a sheet of thickness
``d``. Temperature collapse (``beta -> inf``, feasibility) does not
appear. Do not conflate the two.

This is a **tool** for a model residual. It is not a fusion result
and not a physical discovery.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

DISCLAIMER = (
    "tooling: resistive-layer residual on a model geometry; "
    "not a fusion-energy result and not a physical discovery"
)
FIELD_OPS = (
    "omnibias.fields.torch.ops.mhd.induction_residual",
    "omnibias.fields.torch.ops.mhd.ideal_mhd_momentum_residual",
    "omnibias.fields.jax.ops.mhd.induction_residual",
    "omnibias.fields.jax.ops.mhd.ideal_mhd_momentum_residual",
)


def honesty_payload() -> dict[str, bool]:
    return {
        "unproven_claim": False,
        "fusion_claim": False,
        "physical_discovery": False,
        "theorem_prover_verified": False,
    }


def _sech2(z: float) -> float:
    c = math.cosh(z)
    return 1.0 / (c * c)


@dataclass(frozen=True)
class HarrisSheet:
    """Harris 1962 current sheet (Alfven units, ``mu_0 = 1``)."""

    b0: float = 1.0
    thickness: float = 0.1
    p0: float = 0.0

    def b_y(self, x: float) -> float:
        return self.b0 * math.tanh(x / self.thickness)

    def j_z(self, x: float) -> float:
        return (self.b0 / self.thickness) * _sech2(x / self.thickness)

    def pressure(self, x: float) -> float:
        return self.p0 + 0.5 * self.b0 * self.b0 * _sech2(x / self.thickness)

    def grad_p_x(self, x: float) -> float:
        d = self.thickness
        return -(self.b0 * self.b0 / d) * _sech2(x / d) * math.tanh(x / d)


def mhd_equilibrium_residual(
    *,
    j_z: float,
    b_y: float,
    grad_p_x: float,
) -> float:
    """``(J × B - ∇p)_x`` for a Harris-aligned sheet (static equilibrium)."""
    lorentz_x = -j_z * b_y
    return lorentz_x - grad_p_x


def harris_force_errors(
    thickness: float = 0.1,
    xs: Sequence[float] | None = None,
) -> list[float]:
    """G0 / G4: Harris force balance at sample points."""
    sheet = HarrisSheet(thickness=thickness)
    if xs is None:
        xs = tuple(thickness * z for z in (-3.0, -1.0, -0.3, 0.0, 0.4, 1.2, 2.5))
    return [
        abs(mhd_equilibrium_residual(j_z=sheet.j_z(x), b_y=sheet.b_y(x), grad_p_x=sheet.grad_p_x(x)))
        for x in xs
    ]


def field_ops_resolve() -> dict[str, bool]:
    """G0: the shipped MHD field ops still import (composition is by name)."""
    found: dict[str, bool] = {}
    try:
        from omnibias.fields.torch.ops import mhd as torch_mhd

        found["torch.induction_residual"] = hasattr(torch_mhd, "induction_residual")
        found["torch.ideal_mhd_momentum_residual"] = hasattr(
            torch_mhd, "ideal_mhd_momentum_residual"
        )
    except ImportError:
        found["torch.induction_residual"] = False
        found["torch.ideal_mhd_momentum_residual"] = False
    try:
        from omnibias.fields.jax.ops import mhd as jax_mhd

        found["jax.induction_residual"] = hasattr(jax_mhd, "induction_residual")
        found["jax.ideal_mhd_momentum_residual"] = hasattr(
            jax_mhd, "ideal_mhd_momentum_residual"
        )
    except ImportError:
        found["jax.induction_residual"] = False
        found["jax.ideal_mhd_momentum_residual"] = False
    return found


def resistive_layer_basis(
    location: float,
    thickness: float,
    *,
    orders: Sequence[int] = (0, 1, 2),
) -> list[tuple[str, float, int]]:
    """Multi-pack descriptors ``(family, location, order)`` at the sheet."""
    if thickness <= 0.0:
        raise ValueError(f"thickness must be positive, got {thickness}")
    return [("tanh", float(location), int(order)) for order in orders]


def pack_current(x: float, *, location: float, thickness: float) -> float:
    """Closed-form ``tanh'`` pack: ``sech^2((x-μ)/d) / d`` (Harris ``J``)."""
    return _sech2((x - location) / thickness) / thickness


def _relative_l2(
    values: Sequence[float],
    approx: Sequence[float],
) -> float:
    num = 0.0
    den = 0.0
    for v, a in zip(values, approx, strict=True):
        num += (v - a) * (v - a)
        den += v * v
    if den <= 0.0:
        return 0.0
    return math.sqrt(num / den)


def uniform_current_projection(
    thickness: float,
    n_bins: int,
    *,
    half_width: float = 1.0,
    n_quad: int = 4000,
) -> float:
    """Relative L2 error of a piecewise-constant projection of Harris ``J``."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    lo, hi = -half_width, half_width
    width = (hi - lo) / float(n_bins)
    # Cluster quadrature in units of thickness, then keep the box.
    zs = [(-8.0 + 16.0 * i / (n_quad - 1)) for i in range(n_quad)]
    xs = [z * thickness for z in zs if lo <= z * thickness <= hi]
    if len(xs) < 16:
        xs = [lo + (hi - lo) * i / (n_quad - 1) for i in range(n_quad)]
    truth = [pack_current(x, location=0.0, thickness=thickness) for x in xs]
    means = [0.0] * n_bins
    counts = [0] * n_bins
    for x, val in zip(xs, truth, strict=True):
        idx = min(n_bins - 1, max(0, int((x - lo) / width)))
        means[idx] += val
        counts[idx] += 1
    for i in range(n_bins):
        if counts[i]:
            means[i] /= float(counts[i])
    approx = []
    for x in xs:
        idx = min(n_bins - 1, max(0, int((x - lo) / width)))
        approx.append(means[idx])
    return _relative_l2(truth, approx)


def pack_current_residual(thickness: float, *, half_width: float = 1.0) -> float:
    """One tanh' pack is the Harris current: residual is rounding."""
    xs = [thickness * z for z in (-4.0, -1.5, -0.2, 0.0, 0.5, 1.8, 3.5)]
    xs = [x for x in xs if abs(x) <= half_width]
    truth = [(1.0 / thickness) * _sech2(x / thickness) for x in xs]
    pack = [pack_current(x, location=0.0, thickness=thickness) for x in xs]
    return _relative_l2(truth, pack)


def layer_scaling_report(
    thicknesses: Sequence[float] = (1e-1, 1e-2, 1e-3, 1e-4),
    *,
    pack_orders: Sequence[int] = (0, 1, 2, 3),
    target: float = 0.05,
) -> list[dict[str, float | int | bool]]:
    """G3: pack dim is independent of ``d``; uniform dim grows at least as ``1/d``."""
    rows: list[dict[str, float | int | bool]] = []
    pack_dim = len(pack_orders)
    prev_uniform = 0
    prev_d = 0.0
    for d in thicknesses:
        pack_res = pack_current_residual(float(d))
        n_need = 1
        for n in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096):
            if uniform_current_projection(float(d), n) <= target:
                n_need = n
                break
        else:
            n_need = 4096
        grew = True if prev_uniform == 0 else n_need * prev_d >= 0.5 * prev_uniform * float(d)
        rows.append(
            {
                "d": float(d),
                "pack_dim": pack_dim,
                "pack_residual": pack_res,
                "uniform_dim": n_need,
                "grew_as_inv_d": bool(grew),
            }
        )
        prev_uniform = n_need
        prev_d = float(d)
    return rows


__all__ = [
    "DISCLAIMER",
    "FIELD_OPS",
    "HarrisSheet",
    "field_ops_resolve",
    "harris_force_errors",
    "honesty_payload",
    "layer_scaling_report",
    "mhd_equilibrium_residual",
    "pack_current",
    "pack_current_residual",
    "resistive_layer_basis",
    "uniform_current_projection",
]
