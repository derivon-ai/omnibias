# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Boundary-integral kernels (theory 02-06).

Linear constant-coefficient homogeneous problems only. The PDE is exact
*off the surface* by construction; the boundary condition is approximated.
No 2-D/3-D FMM. Dense evaluation is small-N unless the 02-07 tree is used.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from omnibias.core.conjugate import (
    HardyAtom,
    HardyDictionary,
    hardy_p_deriv_n,
    hardy_q_deriv_n,
)
from omnibias.core.mollifier import MollifierSpec

EquationName = Literal["laplace", "helmholtz", "stokes"]


@dataclass(frozen=True)
class Surface:
    """A parameterized surface. The unit circle is the validation geometry."""

    kind: Literal["circle"]
    radius: float = 1.0
    n_quad: int = 32
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        if self.kind != "circle":
            raise ValueError("gated BEM ships the circle parameterization only")
        if self.radius <= 0.0:
            raise ValueError("radius must be positive")
        if self.n_quad < 4:
            raise ValueError("n_quad must be >= 4")

    def nodes(self) -> tuple[tuple[float, float], ...]:
        cx, cy = self.center
        r = self.radius
        n = self.n_quad
        return tuple(
            (cx + r * math.cos(2.0 * math.pi * i / n), cy + r * math.sin(2.0 * math.pi * i / n))
            for i in range(n)
        )

    def weights(self) -> tuple[float, ...]:
        # arc-length measure  r dtheta
        ds = 2.0 * math.pi * self.radius / self.n_quad
        return tuple(ds for _ in range(self.n_quad))


@dataclass(frozen=True)
class KernelSpec:
    equation: EquationName
    dimension: int = 2
    wavenumber: complex | None = None
    regularization: MollifierSpec | float | None = None

    def __post_init__(self) -> None:
        if self.equation != "laplace":
            raise ValueError("gated BEM ships Laplace only (linear constant-coeff homogeneous)")
        if self.dimension != 2:
            raise ValueError("gated BEM is 2-D Laplace; 2-D/3-D FMM is out of scope")


def green_laplace_2d(dx: float, dy: float, *, eps: float = 0.0) -> float:
    """``G = -log(sqrt(r^2+eps^2))/(2 pi)``. ``eps = 0`` is the exact kernel."""
    r2 = dx * dx + dy * dy + float(eps) * float(eps)
    if r2 <= 0.0:
        raise ValueError("Green kernel is singular on the surface; evaluate off-surface")
    return -0.5 * math.log(r2) / (2.0 * math.pi)


def single_layer(
    x: tuple[float, float],
    surface: Surface,
    density: Sequence[float],
    kernel: KernelSpec,
) -> float:
    """Single-layer potential. Harmonic off-surface for the exact kernel."""
    nodes = surface.nodes()
    wts = surface.weights()
    if len(density) != len(nodes):
        raise ValueError("density length must match quadrature nodes")
    eps = 0.0
    if isinstance(kernel.regularization, (int, float)):
        eps = float(kernel.regularization)
    acc = 0.0
    for (yx, yy), phi, w in zip(nodes, density, wts, strict=True):
        acc += w * float(phi) * green_laplace_2d(x[0] - yx, x[1] - yy, eps=eps)
    return acc


def pde_residual_off_surface(
    x: tuple[float, float],
    surface: Surface,
    density: Sequence[float],
    kernel: KernelSpec,
    *,
    min_clearance: float = 1e-6,
) -> float:
    """Laplace residual. Exact kernel: identically 0 off-surface (by construction)."""
    nodes = surface.nodes()
    for yx, yy in nodes:
        if math.hypot(x[0] - yx, x[1] - yy) <= min_clearance:
            raise ValueError("query point is on/near the surface; residual is off-surface only")
    if kernel.regularization is None:
        _ = (surface, density)
        return 0.0
    # Mollified kernel: Delta log(r^2+eps^2) = 4 eps^2 / (r^2+eps^2)^2.
    eps = float(kernel.regularization) if isinstance(kernel.regularization, (int, float)) else 0.0
    wts = surface.weights()
    acc = 0.0
    for (yx, yy), phi, w in zip(nodes, density, wts, strict=True):
        dx, dy = x[0] - yx, x[1] - yy
        r2 = dx * dx + dy * dy + eps * eps
        # Delta of -log(r2)/(4 pi)  [since G = -0.5 log(r2) / (2 pi) = -log(r2)/(4 pi)]
        # Delta log(r2+eps^2 wait r2 already includes eps^2) = 4 eps^2 / r2^2
        acc += w * float(phi) * (-1.0 / (4.0 * math.pi)) * (4.0 * eps * eps) / (r2 * r2)
    return acc


def half_plane_dtn(
    dictionary: HardyDictionary,
    coeffs: Sequence[float],
    y: float,
) -> float:
    """Dirichlet-to-Neumann on the half-plane: ``H[f']``, dictionary permutation.

    Closed form, no quadrature. Line Hilbert only.
    """
    if len(coeffs) != len(dictionary.atoms):
        raise ValueError("coeffs length must match the dictionary")
    acc = 0.0
    for atom, c in zip(dictionary.atoms, coeffs, strict=True):
        if not (atom.exponent > 0.0):
            raise ValueError("DtN commutation needs decay (alpha > 0)")
        if atom.parity == "even":
            acc += float(c) * hardy_q_deriv_n(y, atom.scale, atom.exponent, atom.order + 1)
        else:
            acc += float(c) * (-hardy_p_deriv_n(y, atom.scale, atom.exponent, atom.order + 1))
    return acc


def circle_dirichlet_density(
    surface: Surface,
    boundary_values: Sequence[float],
    *,
    mean_tol: float = 1e-12,
) -> tuple[float, ...]:
    """Exterior Dirichlet density on a circle via the single-layer spectrum.

    Mode ``k >= 1`` of the boundary data is scaled by ``2 k``; a nonzero
    mean (net charge / log term) is refused. Not a train.
    """
    if surface.kind != "circle":
        raise ValueError("circle_dirichlet_density is the disc solver")
    if abs(float(surface.radius) - 1.0) > 1e-15:
        raise ValueError("circle_dirichlet_density ships the unit disc (radius=1)")
    n = surface.n_quad
    if len(boundary_values) != n:
        raise ValueError("boundary_values length must match n_quad")
    g = [float(v) for v in boundary_values]
    mean = sum(g) / float(n)
    if abs(mean) > mean_tol:
        raise ValueError(
            "exterior single-layer disc solver refuses a net charge "
            f"(mean Dirichlet {mean}); the log term does not decay"
        )
    # Unnormalized real DFT. Mode k of cos(k θ) has hat = n/2.
    # Density scale for the decaying exterior mode is 2k.
    two_pi_n = 2.0 * math.pi / float(n)
    phi = [0.0] * n
    k_max = (n - 1) // 2
    for k in range(1, k_max + 1):
        ak = 2.0 * sum(g[j] * math.cos(two_pi_n * k * j) for j in range(n)) / float(n)
        bk = 2.0 * sum(g[j] * math.sin(two_pi_n * k * j) for j in range(n)) / float(n)
        scale = 2.0 * float(k)
        for j in range(n):
            ang = two_pi_n * k * j
            phi[j] += scale * (ak * math.cos(ang) + bk * math.sin(ang))
    if n % 2 == 0:
        k = n // 2
        ak = sum(g[j] * math.cos(two_pi_n * k * j) for j in range(n)) / float(n)
        scale = 2.0 * float(k) * ak
        for j in range(n):
            phi[j] += scale * math.cos(two_pi_n * k * j)
    return tuple(phi)


def exterior_disc_field(x: tuple[float, float], *, radius: float = 1.0) -> float:
    """Analytic exterior unit-mode field ``(R/r) cos θ = R x / r^2``."""
    r2 = float(x[0]) * float(x[0]) + float(x[1]) * float(x[1])
    if r2 <= float(radius) * float(radius):
        raise ValueError("exterior_disc_field is defined outside the disc")
    return float(radius) * float(x[0]) / r2


def annulus_rel_l2(
    surface: Surface,
    density: Sequence[float],
    kernel: KernelSpec,
    truth: Callable[[tuple[float, float]], float],
    *,
    r_lo: float = 1.5,
    r_hi: float = 3.0,
    n_radial: int = 8,
    n_theta: int = 32,
) -> tuple[float, float]:
    """Relative ``L2`` and skill vs the zero predictor on an annulus."""
    if r_lo <= surface.radius:
        raise ValueError("annulus must sit outside the surface")
    num = 0.0
    den = 0.0
    for ir in range(n_radial):
        if n_radial == 1:
            r = 0.5 * (r_lo + r_hi)
        else:
            r = r_lo + (r_hi - r_lo) * ir / float(n_radial - 1)
        for it in range(n_theta):
            t = 2.0 * math.pi * it / float(n_theta)
            pt = (r * math.cos(t), r * math.sin(t))
            pred = single_layer(pt, surface, density, kernel)
            tru = float(truth(pt))
            num += (pred - tru) * (pred - tru)
            den += tru * tru
    rel = math.sqrt(num / max(den, 1e-18))
    skill = 1.0 - (num / max(den, 1e-18))
    return rel, skill


def poisson_pair_dictionary(*, scale: float = 1.0) -> tuple[HardyDictionary, tuple[float, float]]:
    """``P_{a,1}`` / ``Q_{a,1}`` pair; coeffs ``(1, 0)`` recover the Poisson kernel."""
    a = float(scale)
    atoms = (
        HardyAtom(scale=a, exponent=1.0, order=0, parity="even"),
        HardyAtom(scale=a, exponent=1.0, order=0, parity="odd"),
    )
    return HardyDictionary(atoms), (1.0, 0.0)


__all__ = [
    "KernelSpec",
    "Surface",
    "annulus_rel_l2",
    "circle_dirichlet_density",
    "exterior_disc_field",
    "green_laplace_2d",
    "half_plane_dtn",
    "pde_residual_off_surface",
    "poisson_pair_dictionary",
    "single_layer",
]
