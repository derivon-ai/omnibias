# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Shared NamedTuple output types for the equation registry (torch)."""

from __future__ import annotations

from typing import NamedTuple

from torch import Tensor


class NavierStokesOutput(NamedTuple):
    """Output of :class:`omnibias.pinn.torch.equations.NavierStokes`.

    Attributes
    ----------
    residual
        Momentum residual.

        * primitive 3D: shape ``(B, 3)`` with components ``(R_x, R_y, R_z)``.
        * primitive 2D: shape ``(B, 2)`` with components ``(R_x, R_y)``.
        * vorticity-stream 2D: shape ``(B,)`` -- the scalar vorticity-transport
          residual.
    continuity
        Continuity (incompressibility) residual ``nabla . u``. Shape ``(B,)``.
        Identically zero (up to round-off) when ``incompressibility="hard"``
        i.e. the field is wrapped in a divergence-free cage.

        For ``form="vorticity_stream_2d"`` this is identically zero by
        construction.
    diag
        Plain-Python dict of scalar diagnostics: mean residual energy,
        mean divergence, etc. Used for logging.
    """

    residual: Tensor
    continuity: Tensor
    diag: dict[str, float]


class BurgersOutput(NamedTuple):
    """Output of :class:`omnibias.pinn.torch.equations.Burgers`.

    Attributes
    ----------
    residual
        Pointwise residual ``u_t + u . grad u - nu * laplacian u``.
        Shape ``(B,)`` for the 1D scalar form, ``(B, D)`` for vector.
    diag
        Diagnostic dict.
    """

    residual: Tensor
    diag: dict[str, float]


class HeatOutput(NamedTuple):
    """Output of :class:`omnibias.pinn.torch.equations.Heat`.

    Attributes
    ----------
    residual
        ``u_t - alpha * laplacian(u) - source``. Shape ``(B,)``.
    diag
        Diagnostic dict.
    """

    residual: Tensor
    diag: dict[str, float]


class KSOutput(NamedTuple):
    """Output of :class:`omnibias.pinn.torch.equations.KuramotoSivashinsky`.

    Attributes
    ----------
    residual
        Pointwise KS residual ``u_t + c1 * u * u_x + c2 * u_xx + c3 * u_xxxx``.
        Shape ``(B,)``.
    diag
        Diagnostic dict.
    """

    residual: Tensor
    diag: dict[str, float]


class CHOutput(NamedTuple):
    """Output of :class:`omnibias.pinn.torch.equations.CahnHilliard`.

    Attributes
    ----------
    residual
        Pointwise CH residual ``c_t - M * (f''(c) * lap c + f'''(c)
        * |grad c|^2) + M * kappa * biharmonic(c)``. Shape ``(B,)``.
    diag
        Diagnostic dict.
    """

    residual: Tensor
    diag: dict[str, float]


class CCFOutput(NamedTuple):
    """Output of :class:`omnibias.pinn.torch.equations.CordobaCordobaFontelos`.

    Attributes
    ----------
    residual
        Pointwise self-similar CCF residual on the periodic grid. Shape ``(B,)``.
    hilbert
        The (nonlocal) Hilbert transform ``H[theta]`` of the profile, returned
        so callers can inspect / reuse the nonlocal velocity. Shape ``(B,)``.
    diag
        Diagnostic dict (``mean_sq_residual``, ``max_abs_residual``).
    """

    residual: Tensor
    hilbert: Tensor
    diag: dict[str, float]


class CCFCompactifiedOutput(NamedTuple):
    """Output of compactified / line-domain CCF residual.

    Attributes
    ----------
    residual
        Factored residual ``R = E / F``. Shape ``(B,)``.
    equation_residual
        Raw self-similar equation residual ``E``. Shape ``(B,)``.
    hilbert
        Truncated-line Hilbert transform of the profile. Shape ``(B,)``.
    weight
        Factorisation weight ``F(y)``. Shape ``(B,)``.
    q
        Compactified coordinates ``q(y)``. Shape ``(B,)``.
    diag
        Diagnostic dict.
    """

    residual: Tensor
    equation_residual: Tensor
    hilbert: Tensor
    weight: Tensor
    q: Tensor
    diag: dict[str, float]


class FredholmOutput(NamedTuple):
    """Output of :class:`omnibias.pinn.torch.equations.Fredholm`.

    Attributes
    ----------
    residual
        ``u - source - lam * int_Omega K(x,t) u(t) dmu(t)``. Shape ``(B,)``.
    integral
        The nonlocal term ``int_Omega K(x,t) u(t) dmu(t)`` on its own, shape
        ``(B,)``. Returned because it is the expensive part of the evaluation and
        the part worth inspecting: it is the only quadrature-approximated
        quantity in the residual.
    diag
        Diagnostic dict.
    """

    residual: Tensor
    integral: Tensor
    diag: dict[str, float]


class VolterraOutput(NamedTuple):
    """Output of :class:`omnibias.pinn.torch.equations.Volterra`.

    Attributes
    ----------
    residual
        ``u - source - lam * int_a^x K(x,t) u(t) dt``. Shape ``(B,)``.
    integral
        The causal term ``int_a^x K(x,t) u(t) dt`` on its own. Shape ``(B,)``.
    diag
        Diagnostic dict.
    """

    residual: Tensor
    integral: Tensor
    diag: dict[str, float]


class BiharmonicOutput(NamedTuple):
    """Output of :class:`omnibias.pinn.torch.equations.Biharmonic`.

    Attributes
    ----------
    residual
        Pointwise residual ``Delta^2 u - source``. Shape ``(B,)``.
    diag
        Diagnostic dict.
    """

    residual: Tensor
    diag: dict[str, float]


__all__ = [
    "BiharmonicOutput",
    "BurgersOutput",
    "CCFCompactifiedOutput",
    "CCFOutput",
    "CHOutput",
    "FredholmOutput",
    "HeatOutput",
    "KSOutput",
    "NavierStokesOutput",
    "VolterraOutput",
]
