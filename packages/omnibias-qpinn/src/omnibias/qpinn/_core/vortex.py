# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Vortex diagnostics for 2D rotating-frame condensates.

Backend-agnostic helpers (NumPy-only) for the 2D vortex-lattice demo
and any other rotating-NLS / superfluid-helium / quantum-fluid PINN
that needs:

* a 2D **Thomas-Fermi reference** for the smooth background density,
* **Feynman's vortex-density rule** :math:`n_v = m\Omega/(\pi\hbar)`,
* a **plaquette-winding vortex detector** that counts phase
  singularities in a 2D phase field on a uniform Cartesian grid.

These all operate on plain :mod:`numpy` arrays so they work in both
the torch and JAX experimental pipelines without backend churn. Use
:mod:`omnibias.qpinn.torch.diagnostics.vortex` or
:mod:`omnibias.qpinn.jax.diagnostics.vortex` for backend-tagged
re-exports if you prefer the symmetric import path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def thomas_fermi_density_2d(
    x: np.ndarray, y: np.ndarray, *,
    mu: float, omega_trap: float = 1.0, g: float,
) -> np.ndarray:
    r"""2D Thomas-Fermi density :math:`\rho(x,y) = \max((\mu - V)/g, 0)`.

    Parameters
    ----------
    x, y
        Coordinate arrays (any shape, broadcastable). Atomic units.
    mu
        Chemical potential (Hartree, or whatever consistent energy
        scale was used to compute ``g``).
    omega_trap
        Harmonic-trap frequency. Default 1.0.
    g
        Two-body interaction strength.

    Returns
    -------
    np.ndarray
        Density field, same shape as broadcast ``(x, y)``. Zero outside
        the TF disk.
    """
    V = 0.5 * omega_trap ** 2 * (x ** 2 + y ** 2)
    rho = (mu - V) / g
    return np.where(rho > 0.0, rho, 0.0)


def thomas_fermi_mu_2d(*, g: float, omega_trap: float = 1.0, N_particles: float = 1.0) -> float:
    r"""2D Thomas-Fermi chemical potential :math:`\mu = \omega\sqrt{g N / \pi}`.

    Derivation: integrating the Thomas-Fermi density over the disk and
    requiring :math:`\int |\psi|^2 = N` gives
    :math:`\mu_\text{TF} = \omega_\text{trap}\,\sqrt{g\,N/\pi}`.
    """
    if g <= 0:
        raise ValueError(f"g must be > 0 for the Thomas-Fermi regime, got {g}")
    if omega_trap <= 0:
        raise ValueError(f"omega_trap must be > 0, got {omega_trap}")
    return float(omega_trap * np.sqrt(g * N_particles / np.pi))


def thomas_fermi_radius_2d(*, mu: float, omega_trap: float = 1.0) -> float:
    r"""2D Thomas-Fermi cloud radius :math:`R_\text{TF} = \sqrt{2\mu}/\omega_\text{trap}`."""
    if mu < 0:
        raise ValueError(f"mu must be >= 0, got {mu}")
    if omega_trap <= 0:
        raise ValueError(f"omega_trap must be > 0, got {omega_trap}")
    return float(np.sqrt(2.0 * mu) / omega_trap)


def feynman_vortex_count(
    *, omega_rot: float, R_TF: float, m: float = 1.0, hbar: float = 1.0,
) -> float:
    r"""Feynman vortex-count prediction inside a TF disk.

    .. math::

        N_v \;\approx\; n_v \cdot A_\text{TF}
            \;=\; \frac{m\,\Omega}{\pi\,\hbar}\cdot \pi R_\text{TF}^2
            \;=\; \frac{m\,\Omega\,R_\text{TF}^2}{\hbar}.

    In dimensionless harmonic-oscillator units :math:`m = \hbar = 1`
    this reduces to :math:`N_v = \Omega R_\text{TF}^2`.
    """
    return float(m * omega_rot * R_TF ** 2 / hbar)


@dataclass(frozen=True)
class VortexDetection:
    """Result of :func:`detect_vortices` on a 2D phase field."""

    count: int
    """Total *unsigned* vortex count: ``|charges|.sum()``."""

    charges: np.ndarray
    """Per-plaquette charge array of shape ``(Ny - 1, Nx - 1)`` with
    entries in :math:`\\{-1, 0, +1\\}`."""

    net_winding: int
    """Signed total winding ``charges.sum()`` (vortices minus antivortices)."""


def detect_vortices(
    phase: np.ndarray, *, mask: np.ndarray | None = None,
) -> tuple[int, np.ndarray]:
    r"""Count phase singularities by plaquette winding.

    A vortex is a point around which :math:`\arg\psi` winds by
    :math:`\pm 2\pi`. We scan every 2x2 plaquette and sum the
    branch-resolved phase increments along its boundary; any plaquette
    whose total winding is :math:`\pm 2\pi` contains a vortex of that
    sign.

    Parameters
    ----------
    phase
        Phase field of shape ``(Ny, Nx)``, values in :math:`(-\pi, \pi]`.
    mask
        Optional boolean array of shape ``(Ny, Nx)``. Only plaquettes
        whose four corners all lie inside the mask are counted. Use
        this to restrict the search to the Thomas-Fermi disk.

    Returns
    -------
    (count, charges)
        Total unsigned vortex count and the per-plaquette charge map.
    """
    if phase.ndim != 2:
        raise ValueError(f"phase must be 2D, got shape {phase.shape!r}")

    def _delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        d = a - b
        return d - 2.0 * np.pi * np.round(d / (2.0 * np.pi))

    p = phase
    d1 = _delta(p[:-1, 1:],  p[:-1, :-1])
    d2 = _delta(p[1:,  1:],  p[:-1, 1:])
    d3 = _delta(p[1:,  :-1], p[1:,  1:])
    d4 = _delta(p[:-1, :-1], p[1:,  :-1])
    winding = d1 + d2 + d3 + d4
    charges = np.round(winding / (2.0 * np.pi)).astype(np.int64)
    if mask is not None:
        if mask.shape != phase.shape:
            raise ValueError(
                f"mask shape {mask.shape!r} != phase shape {phase.shape!r}"
            )
        plaq_mask = mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]
        charges = np.where(plaq_mask, charges, 0)
    return int(np.abs(charges).sum()), charges


def detect_vortices_full(
    phase: np.ndarray, *, mask: np.ndarray | None = None,
) -> VortexDetection:
    """Like :func:`detect_vortices` but returns the structured result."""
    count, charges = detect_vortices(phase, mask=mask)
    return VortexDetection(
        count=count, charges=charges, net_winding=int(charges.sum()),
    )


__all__ = [
    "VortexDetection",
    "detect_vortices",
    "detect_vortices_full",
    "feynman_vortex_count",
    "thomas_fermi_density_2d",
    "thomas_fermi_mu_2d",
    "thomas_fermi_radius_2d",
]
