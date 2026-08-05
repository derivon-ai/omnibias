# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Helmholtz equation residual (torch backend).

The (homogeneous) Helmholtz equation in :math:`D` spatial dimensions is

.. math::

    (\nabla^2 + k^2)\,\psi(x) = 0,

with a possibly :math:`x`-dependent wavenumber :math:`k(x)` (the
inhomogeneous case allows a non-constant index of refraction). With
:math:`\psi = \psi_R + i\,\psi_I` the equation splits into two
independent equations on the real and imaginary channels.

A non-zero source :math:`s(x)` gives the inhomogeneous Helmholtz
:math:`(\nabla^2 + k^2) \psi = -s`, used for scattering problems.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from omnibias.pinn._core.state import FieldState
from omnibias.qpinn._core.complex import psi_value
from torch import Tensor


@dataclass
class HelmholtzOutput:
    """Output of :class:`Helmholtz`.

    Attributes
    ----------
    residual
        Stacked ``(B, 2)`` residual ``(R_re, R_im)``.
    diag
        Diagnostic dict.
    """

    residual: Tensor
    diag: dict[str, float]


@dataclass
class Helmholtz:
    r"""Configurable Helmholtz residual.

    Parameters
    ----------
    k
        Wavenumber. Accepts a ``float`` (constant) or a callable
        ``k(state) -> Tensor of shape (B,)`` for a position-dependent
        index of refraction.
    psi
        Wavefunction group name. Default ``"psi"``.
    source
        Optional callable ``s(state) -> Tensor of shape (B, 2)`` added
        to the residual. For inhomogeneous scattering, pass the source
        term :math:`-(\nabla^2 + k^2)\psi_{inc}`.
    """

    k: float | Callable[[FieldState], Tensor] = 1.0
    psi: str = "psi"
    source: Callable[[FieldState], Tensor] | None = None

    def __call__(self, state: FieldState) -> HelmholtzOutput:
        re_name = f"{self.psi}_re"
        im_name = f"{self.psi}_im"
        if not state.components.is_component(re_name):
            raise KeyError(
                f"component {re_name!r} not found; build the field with "
                "omnibias.qpinn.make_psi_components"
            )
        psi_re, psi_im = psi_value(state, self.psi)
        lap_re = state.ops.laplacian(state, re_name)
        lap_im = state.ops.laplacian(state, im_name)
        if callable(self.k):
            k_val = self.k(state)
        else:
            k_val = torch.as_tensor(
                self.k, dtype=psi_re.dtype, device=psi_re.device,
            )
        k_sq = k_val * k_val
        res_re = lap_re + k_sq * psi_re
        res_im = lap_im + k_sq * psi_im
        residual = torch.stack([res_re, res_im], dim=-1)
        if self.source is not None:
            residual = residual + self.source(state)
        return HelmholtzOutput(
            residual=residual,
            diag={"mean_sq_residual": float((residual.detach() ** 2).mean())},
        )


def helmholtz(
    state: FieldState,
    *,
    k: float | Callable[[FieldState], Tensor] = 1.0,
    psi: str = "psi",
    source: Callable[[FieldState], Tensor] | None = None,
) -> HelmholtzOutput:
    """Stateless one-shot wrapper around :class:`Helmholtz`."""
    return Helmholtz(k=k, psi=psi, source=source)(state)


__all__ = ["Helmholtz", "HelmholtzOutput", "helmholtz"]
