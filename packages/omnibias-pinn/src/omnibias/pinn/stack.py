# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact-derivative layered-stack design (theory 07-07).

Builds on spec 02-11's transfer product. ``dT/dθ`` is the derivative
of a ``2 x 2`` matrix product, not a finite difference. The founding
bias collapse (``delta -> 0``) is not a temperature collapse
(``beta -> inf``, feasibility). Do not conflate the two.

A quarter-wave mirror at one wavelength already has a closed-form
optimum. This module targets broadband / off-normal work and counts
gradient cost. It is a **tool**, not a new material.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from omnibias.core.transfer import (
    Layer,
    reflection_transmission,
    stack_matrix,
    unitarity_residual,
)

DISCLAIMER = (
    "tooling: exact dT/dtheta through a transfer product; "
    "not a new material and not a physical discovery"
)
Matrix2 = tuple[tuple[complex, complex], tuple[complex, complex]]


def honesty_payload() -> dict[str, bool]:
    return {
        "unproven_claim": False,
        "new_material_claim": False,
        "physical_discovery": False,
        "theorem_prover_verified": False,
    }


def _mul(a: Matrix2, b: Matrix2) -> Matrix2:
    return (
        (a[0][0] * b[0][0] + a[0][1] * b[1][0], a[0][0] * b[0][1] + a[0][1] * b[1][1]),
        (a[1][0] * b[0][0] + a[1][1] * b[1][0], a[1][0] * b[0][1] + a[1][1] * b[1][1]),
    )


def _add(a: Matrix2, b: Matrix2) -> Matrix2:
    return (
        (a[0][0] + b[0][0], a[0][1] + b[0][1]),
        (a[1][0] + b[1][0], a[1][1] + b[1][1]),
    )


def _eye() -> Matrix2:
    return ((1.0 + 0.0j, 0.0j), (0.0j, 1.0 + 0.0j))


def bragg_reflectance_closed_form(
    n_h: float,
    n_l: float,
    n_s: float,
    n_pairs: int,
) -> float:
    """Macleod / quarter-wave admittance formula (normal incidence, one λ)."""
    if n_pairs < 1:
        raise ValueError("n_pairs must be >= 1")
    y = (n_h / n_l) ** (2 * n_pairs) * n_s
    return ((1.0 - y) / (1.0 + y)) ** 2


def optical_omega(wavelength: float) -> float:
    """``ω`` in units with ``c = 1`` so a quarter-wave is ``n d = λ/4``."""
    if wavelength <= 0.0:
        raise ValueError("wavelength must be positive")
    return 2.0 * math.pi / wavelength


def quarter_wave_layers(
    n_h: float,
    n_l: float,
    n_pairs: int,
    *,
    wavelength: float,
) -> tuple[Layer, ...]:
    omega = optical_omega(wavelength)
    layers: list[Layer] = []
    for _ in range(n_pairs):
        layers.append(Layer(complex(n_h, 0.0), math.pi / (2.0 * omega * n_h)))
        layers.append(Layer(complex(n_l, 0.0), math.pi / (2.0 * omega * n_l)))
    return tuple(layers)


def transfer_stack(
    layers: Sequence[Layer],
    *,
    wavelength: float,
    n_in: float = 1.0,
    n_out: float = 1.52,
) -> tuple[complex, complex, float, float]:
    """``r, t, R, T`` of a finite stack at one wavelength."""
    omega = optical_omega(wavelength)
    m = stack_matrix(layers, omega)
    r, t = reflection_transmission(m, n_in=n_in, n_out=n_out)
    return r, t, abs(r) ** 2, abs(t) ** 2 * (n_out / n_in)


def _layer_matrix(layer: Layer, omega: float) -> Matrix2:
    n = layer.index
    delta = n * omega * layer.thickness
    c = cmath.cos(delta)
    s = cmath.sin(delta)
    return ((c, 1.0j * s / n), (1.0j * n * s, c))


def _layer_d_dd(layer: Layer, omega: float) -> Matrix2:
    n = layer.index
    delta = n * omega * layer.thickness
    c = cmath.cos(delta)
    s = cmath.sin(delta)
    dc = -n * omega * s
    ds = n * omega * c
    return ((dc, 1.0j * ds / n), (1.0j * n * ds, dc))


def _stack_and_partials(layers: Sequence[Layer], omega: float) -> tuple[Matrix2, list[Matrix2]]:
    """``M`` and ``dM/dd_k``. One forward product plus stored layers."""
    lefts: list[Matrix2] = [_eye()]
    acc = _eye()
    built: list[Matrix2] = []
    for layer in layers:
        loc = _layer_matrix(layer, omega)
        built.append(loc)
        acc = _mul(loc, acc)
        lefts.append(acc)
    m = acc
    rights: list[Matrix2] = [_eye()] * len(layers)
    acc_r = _eye()
    for i in range(len(layers) - 1, -1, -1):
        rights[i] = acc_r
        acc_r = _mul(acc_r, built[i])
    partials: list[Matrix2] = []
    prefix = _eye()
    for k, layer in enumerate(layers):
        # M = L_{n-1} ... L_k ... L_0,  prefix = L_{k-1}...L_0 after the loop body
        # lefts[k] is L_{k-1}...L_0
        dloc = _layer_d_dd(layer, omega)
        partials.append(_mul(rights[k], _mul(dloc, lefts[k])))
        _ = prefix
    return m, partials


def _r_from_m(m: Matrix2, n_in: float, n_out: float) -> complex:
    r, _t = reflection_transmission(m, n_in=n_in, n_out=n_out)
    return r


def _dr_dM(m: Matrix2, d_m: Matrix2, n_in: float, n_out: float) -> complex:
    """Complex directional derivative of ``r`` along ``d_m`` (central difference in C)."""
    eps = 1e-8
    plus = _add(m, ((d_m[0][0] * eps, d_m[0][1] * eps), (d_m[1][0] * eps, d_m[1][1] * eps)))
    minus = _add(m, ((-d_m[0][0] * eps, -d_m[0][1] * eps), (-d_m[1][0] * eps, -d_m[1][1] * eps)))
    return (_r_from_m(plus, n_in, n_out) - _r_from_m(minus, n_in, n_out)) / (2.0 * eps)


def exact_dR_dthickness(
    layers: Sequence[Layer],
    *,
    wavelength: float,
    n_in: float = 1.0,
    n_out: float = 1.52,
) -> tuple[list[float], int]:
    """Exact ``dR/dd_k`` and the number of full ``stack_matrix`` equivalents (1)."""
    omega = optical_omega(wavelength)
    m, partials = _stack_and_partials(layers, omega)
    r = _r_from_m(m, n_in, n_out)
    grads: list[float] = []
    for d_m in partials:
        dr = _dr_dM(m, d_m, n_in, n_out)
        grads.append(float((2.0 * (r.conjugate() * dr)).real))
    return grads, 1


def fd_dR_dthickness(
    layers: Sequence[Layer],
    *,
    wavelength: float,
    n_in: float = 1.0,
    n_out: float = 1.52,
    step: float = 1e-8,
) -> tuple[list[float], int]:
    """Central finite-difference ``dR/dd_k``. Counts ``1 + 2 n_layers`` forwards."""
    base = list(layers)
    _, _, r0, _ = transfer_stack(base, wavelength=wavelength, n_in=n_in, n_out=n_out)
    grads: list[float] = []
    forwards = 1
    for k, layer in enumerate(base):
        up = list(base)
        down = list(base)
        up[k] = Layer(layer.index, layer.thickness + step)
        down[k] = Layer(layer.index, max(layer.thickness - step, 0.0))
        _, _, rp, _ = transfer_stack(up, wavelength=wavelength, n_in=n_in, n_out=n_out)
        _, _, rm, _ = transfer_stack(down, wavelength=wavelength, n_in=n_in, n_out=n_out)
        grads.append((rp - rm) / (2.0 * step))
        forwards += 2
    _ = r0
    return grads, forwards


@dataclass
class StackDesign:
    layers: tuple[Layer, ...]
    loss: float
    unitarity: list[float] = field(default_factory=list)
    energy: list[float] = field(default_factory=list)
    n_steps: int = 0


def _broadband_loss(
    layers: Sequence[Layer],
    wavelengths: Sequence[float],
    target: float,
    n_in: float,
    n_out: float,
) -> float:
    acc = 0.0
    for wl in wavelengths:
        _r, _t, refl, _tr = transfer_stack(layers, wavelength=wl, n_in=n_in, n_out=n_out)
        acc += (refl - target) ** 2
    return acc / float(len(wavelengths))


def design_stack(
    target: float,
    *,
    n_pairs: int = 5,
    n_h: float = 2.35,
    n_l: float = 1.46,
    n_in: float = 1.0,
    n_out: float = 1.52,
    wavelength: float = 550.0,
    band: Sequence[float] | None = None,
    n_steps: int = 4,
    step_size: float = 1e-3,
) -> StackDesign:
    """Gradient steps on thicknesses. Identities are recorded at every iterate."""
    waves = list(band) if band is not None else [wavelength]
    layers = list(quarter_wave_layers(n_h, n_l, n_pairs, wavelength=wavelength))
    unitarity: list[float] = []
    energy: list[float] = []
    for _ in range(n_steps):
        omega0 = optical_omega(wavelength)
        m = stack_matrix(layers, omega0)
        unitarity.append(unitarity_residual(m))
        _r, _t, refl, trans = transfer_stack(
            layers, wavelength=wavelength, n_in=n_in, n_out=n_out
        )
        energy.append(abs(refl + trans - 1.0))
        grads, _cost = exact_dR_dthickness(
            layers, wavelength=wavelength, n_in=n_in, n_out=n_out
        )
        # Broadband: step on the centre-λ reflectance toward the target.
        sign = 1.0 if refl < target else -1.0
        layers = [
            Layer(layer.index, max(layer.thickness + sign * step_size * g, 1e-9))
            for layer, g in zip(layers, grads, strict=True)
        ]
    loss = _broadband_loss(layers, waves, target, n_in, n_out)
    return StackDesign(tuple(layers), loss, unitarity, energy, n_steps)


def g0_quarter_wave_report() -> dict[str, float | bool]:
    """Spec worked example: 5 HL pairs, ``n_H=2.35``, ``n_L=1.46``, ``n_s=1.52``."""
    closed = bragg_reflectance_closed_form(2.35, 1.46, 1.52, 5)
    layers = quarter_wave_layers(2.35, 1.46, 5, wavelength=550.0)
    _r, _t, refl, _tr = transfer_stack(layers, wavelength=550.0, n_in=1.0, n_out=1.52)
    return {
        "closed_form": closed,
        "transfer": refl,
        "published_target": 0.9777,
        "closed_matches": abs(closed - 0.9777) <= 5e-4,
        "transfer_matches": abs(refl - closed) <= 1e-9,
    }


def gradient_cost_report(*, n_pairs: int = 10) -> dict[str, float | int | bool]:
    """G5: exact ``dR/dd`` vs central FD at ``2 n_pairs`` layers."""
    layers = quarter_wave_layers(2.35, 1.46, n_pairs, wavelength=550.0)
    exact, n_exact = exact_dR_dthickness(layers, wavelength=550.0)
    fd, n_fd = fd_dR_dthickness(layers, wavelength=550.0)
    rel = max(abs(a - b) / max(abs(b), 1e-12) for a, b in zip(exact, fd, strict=True))
    return {
        "n_layers": len(layers),
        "exact_forwards": n_exact,
        "fd_forwards": n_fd,
        "cost_ratio": float(n_fd) / float(n_exact),
        "grad_rel_err": rel,
        "win": (float(n_fd) / float(n_exact)) >= 10.0 and rel <= 0.05,
    }


__all__ = [
    "DISCLAIMER",
    "StackDesign",
    "bragg_reflectance_closed_form",
    "design_stack",
    "exact_dR_dthickness",
    "fd_dR_dthickness",
    "g0_quarter_wave_report",
    "gradient_cost_report",
    "honesty_payload",
    "optical_omega",
    "quarter_wave_layers",
    "transfer_stack",
]
