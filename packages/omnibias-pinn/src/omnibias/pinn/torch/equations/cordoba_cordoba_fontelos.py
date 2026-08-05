# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Córdoba-Córdoba-Fontelos (CCF) self-similar profile residual (torch twin).

Bit-parity companion of :mod:`omnibias.pinn.jax.equations.cordoba_cordoba_fontelos`,
which carries the full mathematical contract. In brief, this evaluates the
*stationary* self-similar profile residual of the 1D CCF nonlocal-transport
model (blow-up time normalised to ``t = 1``, scalar exponent ``k(lambda) =
lambda``):

* transport:  ``(1+lam) y Theta' - lam Theta + (H Theta) Theta'``
* flux:       ``(1+lam) y Theta' - lam Theta + [Theta'(H Theta) + Theta (H Theta')]``

The residual is **nonlocal**: ``state.coords`` must be an ordered uniform grid
over one period of the single periodic spatial axis (the Hilbert transform is
evaluated spectrally over that grid). Local derivatives come from omnibias's
exact closed-form path; only the Hilbert transform is numerical.
"""

from __future__ import annotations

from dataclasses import dataclass

from omnibias.pinn._core.state import FieldState
from omnibias.pinn.torch.equations._types import CCFOutput
from omnibias.pinn.torch.hilbert import hilbert_transform
from torch import Tensor

_FORMS = ("transport", "flux")


def ccf_residual_samples(
    y: Tensor,
    theta: Tensor,
    theta_y: Tensor,
    lam: Tensor | float,
    *,
    form: str = "transport",
    velocity_sign: float = 1.0,
    hilbert_axis: int = -1,
) -> Tensor:
    """Self-similar CCF residual from sampled profile values (see jax twin)."""
    if form not in _FORMS:
        raise ValueError(f"CCF form must be one of {_FORMS}, got {form!r}")
    h_theta = hilbert_transform(theta, axis=hilbert_axis)
    if form == "transport":
        nonlocal_term = h_theta * theta_y
    else:
        h_theta_y = hilbert_transform(theta_y, axis=hilbert_axis)
        nonlocal_term = theta_y * h_theta + theta * h_theta_y
    linear = (1.0 + lam) * y * theta_y - lam * theta
    return linear + velocity_sign * nonlocal_term


@dataclass
class CordobaCordobaFontelos:
    r"""CCF self-similar profile residual (1D scalar, steady, nonlocal).

    Parameters
    ----------
    lam
        Self-similar scaling parameter :math:`\lambda`.
    component
        Profile component name. Default ``"theta"``.
    form
        ``"transport"`` (default) or ``"flux"``.
    velocity_sign
        Sign :math:`s = \pm 1` on the nonlocal term.
    """

    lam: float = 0.6057
    component: str = "theta"
    form: str = "transport"
    velocity_sign: float = 1.0

    def __call__(self, state: FieldState) -> CCFOutput:
        if state.coordinate_spec.time_axis is not None:
            raise ValueError(
                "CCF self-similar residual is steady; the field must have no "
                f"time axis (got time_axis={state.coordinate_spec.time_axis!r})"
            )
        spatial = state.coordinate_spec.spatial_axes
        if len(spatial) != 1:
            raise ValueError(
                f"CCF residual requires exactly 1 spatial axis, got "
                f"{len(spatial)} ({spatial!r})"
            )
        ax = spatial[0]
        ax_i = state.coordinate_spec.axis_index(ax)
        y = state.coords[:, ax_i]
        theta = state.ops.value(state, self.component)
        theta_y = state.ops.derivative(state, self.component, axis=ax, order=1)
        residual = ccf_residual_samples(
            y, theta, theta_y, self.lam,
            form=self.form, velocity_sign=self.velocity_sign,
        )
        h_theta = hilbert_transform(theta)
        return CCFOutput(
            residual=residual,
            hilbert=h_theta,
            diag={
                "mean_sq_residual": float((residual.detach() ** 2).mean()),
                "max_abs_residual": float(residual.detach().abs().max()),
            },
        )


def cordoba_cordoba_fontelos(
    state: FieldState,
    *,
    lam: float = 0.6057,
    component: str = "theta",
    form: str = "transport",
    velocity_sign: float = 1.0,
) -> CCFOutput:
    """Stateless one-shot wrapper around :class:`CordobaCordobaFontelos`."""
    return CordobaCordobaFontelos(
        lam=lam, component=component, form=form, velocity_sign=velocity_sign,
    )(state)


__all__ = [
    "CordobaCordobaFontelos",
    "ccf_residual_samples",
    "cordoba_cordoba_fontelos",
]
