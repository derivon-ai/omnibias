# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Córdoba-Córdoba-Fontelos (CCF) self-similar profile residual (jax twin).

The CCF model [Cordoba2005]_ is the canonical 1D nonlocal-transport analogue of
the vorticity formulation of the 3D Euler / Navier-Stokes equations. The active
scalar :math:`\theta(x,t)` is transported by a velocity given by its own Hilbert
transform :math:`H\theta`:

.. math::

    \theta_t + (H\theta)\,\theta_x = 0          \quad\text{(transport form)},

with the conservative / flux variant

.. math::

    \theta_t + (\theta\,H\theta)_x = 0          \quad\text{(flux form)}.

The inviscid model develops a finite-time gradient singularity from smooth data
[Cordoba2005]_; this module targets its **self-similar blow-up profile**.

Self-similar reduction (the "contract")
---------------------------------------
Following the ansatz of Wang et al. [Wang2025]_ (arXiv:2509.14185, eq. 2), with
blow-up time normalised to :math:`t = 1`,

.. math::

    \theta(x, t) = (1-t)^{k(\lambda)}\,\Theta(y), \qquad
    y = (1-t)^{-(1+\lambda)}\,x .

Because the Hilbert transform is invariant under positive dilations,
:math:`H_x[\theta](x,t) = (1-t)^{k}\,(H_y\Theta)(y)`. Substituting and requiring
that a single power of :math:`(1-t)` factor out of the equation fixes the scalar
exponent

.. math::

    k(\lambda) = \lambda ,

and yields the **stationary** profile equation :math:`\mathcal{E}(\Theta,\lambda)
= 0` on the real line:

.. math::

    \text{transport:}\quad
      (1+\lambda)\,y\,\Theta'(y) - \lambda\,\Theta(y) + (H\Theta)(y)\,\Theta'(y) = 0,

.. math::

    \text{flux:}\quad
      (1+\lambda)\,y\,\Theta'(y) - \lambda\,\Theta(y)
      + \big[\Theta'(y)\,(H\Theta)(y) + \Theta(y)\,(H\Theta')(y)\big] = 0,

using :math:`(H\Theta)' = H(\Theta')`. (Both reductions are verified by exact
symbolic substitution in the test-suite.) A ``velocity_sign`` :math:`s = \pm 1`
multiplies the nonlocal term so both sign conventions in the literature are
expressible.

Parity / gauge
--------------
The nonlocal term is parity-consistent with the local terms only for an **even**
profile :math:`\Theta` (then :math:`\Theta'` is odd, :math:`H\Theta` is odd, and
every term is even). The blow-up amplitude decays as :math:`(1-t)^{\lambda}` while
the gradient :math:`\theta_x \sim (1-t)^{-1}` diverges. A smooth profile exists
only for isolated *admissible* :math:`\lambda`; published CCF values are the
stable profile and :math:`\lambda_1 \approx 0.6057` (first unstable, Eggers &
Fontelos / Wang et al.) and :math:`\lambda_2 \approx 0.4703` (Wang et al.).

Numerics / honesty
------------------
This residual is **nonlocal**: ``state.coords`` must be an *ordered, uniform
grid over one period* of the single periodic spatial axis, because the Hilbert
transform is evaluated spectrally over that grid (see
:mod:`omnibias.pinn.jax.hilbert`). The local derivatives :math:`\Theta, \Theta'`
come from omnibias's exact closed-form path; only :math:`H` is numerical. The
periodic-grid evaluation is a tractable truncation of the line problem -- it is
honest about being an approximation of the unbounded-domain profile and is **not**
by itself a reproduction of the published line-domain :math:`\lambda` values.

.. [Cordoba2005] A. Córdoba, D. Córdoba, M. A. Fontelos, "Formation of
   singularities for a transport equation with nonlocal velocity", Ann. of
   Math. (2) 162 (2005) 1377-1389.
.. [Wang2025] Y. Wang et al., "Discovery of Unstable Singularities",
   arXiv:2509.14185 (2025).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from omnibias.pinn._core.state import FieldState
from omnibias.pinn.jax.equations._types import CCFOutput
from omnibias.pinn.jax.hilbert import hilbert_transform

_FORMS = ("transport", "flux")


def ccf_residual_samples(
    y: Array,
    theta: Array,
    theta_y: Array,
    lam: Array | float,
    *,
    form: str = "transport",
    velocity_sign: float = 1.0,
    hilbert_axis: int = -1,
) -> Array:
    r"""Self-similar CCF residual from sampled profile values.

    Backend kernel shared by :class:`CordobaCordobaFontelos` and the discovery
    harness. ``y``, ``theta`` and ``theta_y`` are sampled on a uniform periodic
    grid (in grid order). Returns the residual array, same shape as ``theta``.
    """
    if form not in _FORMS:
        raise ValueError(f"CCF form must be one of {_FORMS}, got {form!r}")
    h_theta = hilbert_transform(theta, axis=hilbert_axis)
    if form == "transport":
        nonlocal_term = h_theta * theta_y
    else:  # flux: d_y(Theta * H Theta) = Theta' H Theta + Theta H(Theta')
        h_theta_y = hilbert_transform(theta_y, axis=hilbert_axis)
        nonlocal_term = theta_y * h_theta + theta * h_theta_y
    linear = (1.0 + lam) * y * theta_y - lam * theta
    return jnp.asarray(linear + velocity_sign * nonlocal_term)


@dataclass
class CordobaCordobaFontelos:
    r"""CCF self-similar profile residual (1D scalar, steady, nonlocal).

    Parameters
    ----------
    lam
        Self-similar scaling parameter :math:`\lambda` (admissible values are
        isolated). The scalar exponent is :math:`k(\lambda) = \lambda`.
    component
        Profile component name. Default ``"theta"``.
    form
        ``"transport"`` (default) for :math:`\theta_t + (H\theta)\theta_x = 0`
        or ``"flux"`` for the conservative :math:`\theta_t + (\theta H\theta)_x
        = 0`.
    velocity_sign
        Sign :math:`s = \pm 1` on the nonlocal term (literature convention).
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
                "mean_sq_residual": jnp.mean(residual * residual),
                "max_abs_residual": jnp.max(jnp.abs(residual)),
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
