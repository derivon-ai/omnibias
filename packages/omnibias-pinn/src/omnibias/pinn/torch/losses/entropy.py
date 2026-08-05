# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Entropy-consistent residual loss (torch).

For hyperbolic conservation laws Tadmor's *entropy-stable* schemes
penalise residuals using a strictly convex entropy :math:`\\eta(u)`.
The natural PINN analog weights each squared residual by the local
convexity of :math:`\\eta`:

.. math::

    \\| R \\|_\\eta^2 = \\langle R, \\, \\eta''(u)\\, R \\rangle.

For ``entropy_fn = lambda u: 0.5 * u**2`` this reduces to plain MSE.
For more general entropies (e.g. Boltzmann, Burgers'
:math:`\\eta(u) = u \\log u`, kinetic energy, ...) it gives the
correct entropy-consistent residual norm.

For a *quadratic* entropy this is identical to
:func:`omnibias.pinn.torch.losses.mse_residual_loss`; we keep the
helper public so users can pass arbitrary convex weights.
"""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor


def entropy_consistent_residual(
    residual: Tensor,
    *,
    entropy_weight: Callable[[Tensor], Tensor] | None = None,
    state_for_weight: Tensor | None = None,
) -> Tensor:
    """Entropy-weighted MSE of a residual tensor.

    Parameters
    ----------
    residual
        Tensor of any shape.
    entropy_weight
        Optional callable ``u -> eta''(u)`` returning per-element
        non-negative weights. If ``None``, plain MSE is returned.
    state_for_weight
        State :math:`u` at which to evaluate ``eta''``. Defaults to
        ``residual`` (useful when residual itself is the field, e.g.
        for a transport equation).
    """
    if entropy_weight is None:
        return (residual * residual).mean()
    if state_for_weight is None:
        state_for_weight = residual
    weight = entropy_weight(state_for_weight)
    return (weight * residual * residual).mean()


__all__ = ["entropy_consistent_residual"]
