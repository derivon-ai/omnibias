# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared NamedTuple output types for the quantum equation registry (torch)."""

from __future__ import annotations

from typing import NamedTuple

from torch import Tensor


class TISEOutput(NamedTuple):
    """Output of :class:`omnibias.qpinn.torch.equations.TISE`.

    Attributes
    ----------
    residual
        Stacked ``(B, 2)`` residual for the real and imaginary channels
        of ``(H - E) psi``. Column 0 is the real channel, column 1 is
        the imaginary channel.
    energy_estimate
        Scalar variational estimate of :math:`E` from
        ``<psi | H | psi> / <psi | psi>`` on the collocation points.
        ``None`` if no quadrature weights are provided.
    diag
        Plain-Python dict of scalar diagnostics for logging.
    """

    residual: Tensor
    energy_estimate: Tensor | None
    diag: dict[str, float]


class TDSEOutput(NamedTuple):
    """Output of :class:`omnibias.qpinn.torch.equations.TDSE`.

    Attributes
    ----------
    residual
        Stacked ``(B, 2)`` residual. Column 0 is
        ``-hbar * psi_im_t - (H psi)_re``, column 1 is
        ``+hbar * psi_re_t - (H psi)_im``.
    diag
        Diagnostic dict.
    """

    residual: Tensor
    diag: dict[str, float]


class NLSOutput(NamedTuple):
    """Output of :class:`omnibias.qpinn.torch.equations.NLS`
    (Gross-Pitaevskii).

    Attributes
    ----------
    residual
        Stacked ``(B, 2)`` residual including the nonlinear
        ``g |psi|^2 psi`` term in the Hamiltonian.
    diag
        Diagnostic dict (includes ``mean_density`` and
        ``nonlinear_energy``).
    """

    residual: Tensor
    diag: dict[str, float]


class RotatingNLSOutput(NamedTuple):
    """Output of :class:`omnibias.qpinn.torch.equations.RotatingNLS`.

    Stationary 2D Gross-Pitaevskii in the rotating frame::

        R = -hbar^2/(2m) Lap psi + V psi - Omega L_z psi
             + g |psi|^2 psi - mu psi.

    Attributes
    ----------
    residual
        Stacked ``(B, 2)`` residual (real, imaginary).
    density
        ``|psi|^2`` at every collocation point, shape ``(B,)``.
    diag
        Diagnostic dict (``mean_sq_residual``, ``mean_density``,
        ``rotation_energy`` :math:`= \\Omega\\,\\Im(\\psi^* L_z \\psi)`).
    """

    residual: Tensor
    density: Tensor
    diag: dict[str, float]


__all__ = [
    "NLSOutput",
    "RotatingNLSOutput",
    "TDSEOutput",
    "TISEOutput",
]
