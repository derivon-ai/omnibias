# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Proximal-class activations: huber, arctan, log1pu2.

These are the *new design-rule* activations introduced by the multi-bias
operator framing. Each one is chosen so that its K=2 bias-collapse limit
(i.e. its first derivative) is a classical proximal / robust-statistics
primitive:

============  ========================================  ===========================
Activation    K=2 collapse output                       Operator role
============  ========================================  ===========================
``huber``     ``clip(z, -tau, tau)``                    LASSO ISTA soft-shrink step
``arctan``    ``1 / (1 + z^2)``                         Cauchy IRLS weight (robust)
``log1pu2``   ``2 z / (1 + z^2)``                       Redescending M-estimator
============  ========================================  ===========================

``arctan`` and ``log1pu2`` are analytic (fast path ``n in {0, 1, 2}``).
``huber`` is piecewise-polynomial: its fast path is closed form to all
orders on the almost-everywhere / regular-part convention (``n = 2`` is the
indicator of the quadratic region, ``n >= 3`` is zero), dropping the singular
part at the ``+/-tau`` kinks.
"""

from __future__ import annotations

import math

from omnibias.torch.activations.registry import ActivationSpec, register_activation

import torch
from torch import Tensor

# --- huber (default tau = 1) ----------------------------------------------


_DEFAULT_HUBER_TAU = 1.0


def _huber_forward(z: Tensor, tau: float = _DEFAULT_HUBER_TAU) -> Tensor:
    abs_z = z.abs()
    quadratic = 0.5 * z * z
    linear = tau * (abs_z - 0.5 * tau)
    return torch.where(abs_z <= tau, quadratic, linear)


def _huber_derivative(z: Tensor, tau: float = _DEFAULT_HUBER_TAU) -> Tensor:
    return torch.clamp(z, min=-tau, max=tau)


def _huber_integral(z: Tensor, tau: float = _DEFAULT_HUBER_TAU) -> Tensor:
    abs_z = z.abs()
    inside = z * z * z / 6.0
    outside_mag = 0.5 * tau * abs_z * abs_z - 0.5 * tau * tau * abs_z + tau**3 / 6.0
    outside = torch.sign(z) * outside_mag
    return torch.where(abs_z <= tau, inside, outside)


def _huber_fastpath(z: Tensor, n: int, tau: float = _DEFAULT_HUBER_TAU) -> Tensor:
    """Closed-form ``huber^(n)`` on the almost-everywhere / regular-part convention.

    The inner quadratic ``0.5 z^2`` (``|z| <= tau``) contributes ``1`` at
    ``n = 2`` and ``0`` beyond; the outer linear arms contribute ``0`` for
    ``n >= 2``. Order 1 is the clip. The singular part at the ``+/-tau`` kinks
    (a step in the second derivative) is dropped.
    """
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _huber_forward(z, tau)
    if n == 1:
        return _huber_derivative(z, tau)
    if n == 2:
        return (z.abs() <= tau).to(z.dtype)
    return torch.zeros_like(z)


def make_huber_spec(
    tau: float = _DEFAULT_HUBER_TAU, *, register: bool = False
) -> ActivationSpec[Tensor]:
    """Build a Huber :class:`ActivationSpec` with custom threshold ``tau``.

    The default-``tau`` spec is registered as ``"huber"`` at module load.
    Pass ``register=True`` to add a non-default spec to the global
    registry (it will be stored under ``f"huber_tau{tau:g}"``).
    """
    if tau <= 0:
        raise ValueError(f"Huber tau must be > 0, got {tau}.")

    def fwd(z: Tensor) -> Tensor:
        return _huber_forward(z, tau)

    def deriv(z: Tensor) -> Tensor:
        return _huber_derivative(z, tau)

    def integ(z: Tensor) -> Tensor:
        return _huber_integral(z, tau)

    def fp(z: Tensor, n: int) -> Tensor:
        return _huber_fastpath(z, n, tau)

    name = "huber" if math.isclose(tau, _DEFAULT_HUBER_TAU) else f"huber_tau{tau:g}"
    spec = ActivationSpec(
        name=name,
        forward=fwd,
        derivative=deriv,
        fastpath=fp,
        integral=integ,
        riccati_polynomial=None,
        noise_model="laplace_smoothed",
        operator_role=(
            "K=2 collapse -> clip(z, -tau, tau); proximal of L1 / ISTA soft-shrink step for LASSO."
        ),
    )
    if register:
        register_activation(spec)
    return spec


HUBER = register_activation(make_huber_spec(_DEFAULT_HUBER_TAU))


# --- arctan ---------------------------------------------------------------


def _arctan_forward(z: Tensor) -> Tensor:
    return torch.atan(z)


def _arctan_derivative(z: Tensor) -> Tensor:
    return 1.0 / (1.0 + z * z)


def _arctan_integral(z: Tensor) -> Tensor:
    return z * torch.atan(z) - 0.5 * torch.log1p(z * z)


def _arctan_fastpath(z: Tensor, n: int) -> Tensor:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _arctan_forward(z)
    if n == 1:
        return _arctan_derivative(z)
    if n == 2:
        denom = (1.0 + z * z) ** 2
        return -2.0 * z / denom
    raise NotImplementedError(f"arctan fast path only implements n in {{0, 1, 2}}, got {n}.")


ARCTAN = register_activation(
    ActivationSpec(
        name="arctan",
        forward=_arctan_forward,
        derivative=_arctan_derivative,
        fastpath=_arctan_fastpath,
        integral=_arctan_integral,
        riccati_polynomial=None,
        noise_model="cauchy",
        operator_role=(
            "K=2 collapse -> 1/(1+z^2); Cauchy IRLS weight, "
            "natural for heavy-tailed regression residuals."
        ),
        aliases=("atan",),
        limit_pos_inf=math.pi / 2.0,
        limit_neg_inf=-math.pi / 2.0,
    )
)


# --- log1pu2 (redescending M-estimator) -----------------------------------


def _log1pu2_forward(z: Tensor) -> Tensor:
    return torch.log1p(z * z)


def _log1pu2_derivative(z: Tensor) -> Tensor:
    return 2.0 * z / (1.0 + z * z)


def _log1pu2_integral(z: Tensor) -> Tensor:
    return z * torch.log1p(z * z) - 2.0 * z + 2.0 * torch.atan(z)


def _log1pu2_fastpath(z: Tensor, n: int) -> Tensor:
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _log1pu2_forward(z)
    if n == 1:
        return _log1pu2_derivative(z)
    if n == 2:
        denom = (1.0 + z * z) ** 2
        return 2.0 * (1.0 - z * z) / denom
    raise NotImplementedError(f"log1pu2 fast path only implements n in {{0, 1, 2}}, got {n}.")


LOG1PU2 = register_activation(
    ActivationSpec(
        name="log1pu2",
        forward=_log1pu2_forward,
        derivative=_log1pu2_derivative,
        fastpath=_log1pu2_fastpath,
        integral=_log1pu2_integral,
        riccati_polynomial=None,
        noise_model="redescending_m",
        operator_role=(
            "K=2 collapse -> 2 z / (1 + z^2); redescending M-estimator "
            "(Black-Anandan family), down-weights large residuals to zero."
        ),
        aliases=("logcosh_dual",),
    )
)


__all__ = ["ARCTAN", "HUBER", "LOG1PU2", "make_huber_spec"]
