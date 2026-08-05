# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Dirac equation residual (torch backend).

The free Dirac equation on Minkowski space is

.. math::

    (i\gamma^\mu \partial_\mu - m)\,\psi = 0,

acting on a 4-spinor :math:`\psi = (\psi^0, \psi^1, \psi^2, \psi^3)`
encoded as 8 real channels via
:func:`omnibias.qpinn.make_spinor_components` with ``n_components=4``.
The choice of gamma representation (``"dirac"`` or ``"weyl"``) is the
caller's. The mass term acts identity-wise on the spinor index, so the
residual at each spinor index :math:`a` and split-real channel is

.. math::

    R^a_R(x) &= -\sum_\mu \big[(\gamma^\mu \partial_\mu \psi)^a\big]_I
                - m\,\psi^a_R \\
    R^a_I(x) &= +\sum_\mu \big[(\gamma^\mu \partial_\mu \psi)^a\big]_R
                - m\,\psi^a_I

(the ``i`` factor swaps re/im and flips sign on the imaginary channel).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from omnibias.pinn._core.state import FieldState
from omnibias.qpinn._core.spinor import gamma_partial_psi, spinor_value
from torch import Tensor


@dataclass
class DiracOutput:
    """Output of :class:`Dirac`.

    Attributes
    ----------
    residual
        Stacked ``(B, 8)`` residual.  Columns are ordered as the spinor
        components ``(R^0_re, R^0_im, R^1_re, R^1_im, R^2_re, R^2_im,
        R^3_re, R^3_im)``.
    diag
        Diagnostic dict.
    """

    residual: Tensor
    diag: dict[str, float]


@dataclass
class Dirac:
    r"""Configurable Dirac residual.

    Parameters
    ----------
    mass
        :math:`m`. Default 1.0.
    representation
        Choice of gamma matrices. Either ``"dirac"`` (standard) or
        ``"weyl"`` (chiral). Default ``"dirac"``.
    spinor
        Spinor group name on the :class:`FieldState`. Default
        ``"spinor"``. Must be a 4-spinor group built by
        :func:`omnibias.qpinn.make_spinor_components` with
        ``n_components=4``.
    source
        Optional callable ``s(state) -> Tensor of shape (B, 8)`` added
        to the residual.
    """

    mass: float = 1.0
    representation: Literal["dirac", "weyl"] = "dirac"
    spinor: str = "spinor"
    source: Callable[[FieldState], Tensor] | None = None

    def __call__(self, state: FieldState) -> DiracOutput:
        gp = gamma_partial_psi(
            state, spinor_group=self.spinor, representation=self.representation,
        )
        cols: list[Tensor] = []
        for a in range(4):
            psi_re_a, psi_im_a = spinor_value(state, self.spinor, a)
            gp_re_a, gp_im_a = gp[a]
            # i * (gp_re + i * gp_im) - m * (psi_re + i * psi_im) = 0
            # Real: -gp_im - m * psi_re
            # Imag: +gp_re - m * psi_im
            r_re = -gp_im_a - self.mass * psi_re_a
            r_im = gp_re_a - self.mass * psi_im_a
            cols.append(r_re)
            cols.append(r_im)
        residual = torch.stack(cols, dim=-1)
        if self.source is not None:
            residual = residual - self.source(state)
        return DiracOutput(
            residual=residual,
            diag={"mean_sq_residual": float((residual.detach() ** 2).mean())},
        )


def dirac(
    state: FieldState,
    *,
    mass: float = 1.0,
    representation: Literal["dirac", "weyl"] = "dirac",
    spinor: str = "spinor",
    source: Callable[[FieldState], Tensor] | None = None,
) -> DiracOutput:
    """Stateless one-shot wrapper around :class:`Dirac`."""
    return Dirac(
        mass=mass, representation=representation, spinor=spinor, source=source,
    )(state)


__all__ = ["Dirac", "DiracOutput", "dirac"]
