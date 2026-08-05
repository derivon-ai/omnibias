# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Proximal-class activations: huber, arctan, log1pu2.

Each one's K=2 bias-collapse limit is a classical proximal /
robust-statistics primitive. Mirrors
:mod:`omnibias.torch.activations.proximal` on ``keras.ops``. ``arctan`` /
``log1pu2`` cover ``n in {0, 1, 2}``; ``huber`` is closed form to all orders on
the almost-everywhere / regular-part convention (``n = 2`` indicator, ``n >= 3``
zero), dropping the singular part at the ``+/-tau`` kinks.
"""

from __future__ import annotations

import math
from typing import Any

from omnibias.keras.activations.registry import ActivationSpec, register_activation

from keras import ops

_DEFAULT_HUBER_TAU = 1.0


# --- huber ----------------------------------------------------------------


def _huber_forward(z: Any, tau: float = _DEFAULT_HUBER_TAU) -> Any:
    abs_z = ops.abs(z)
    quadratic = 0.5 * z * z
    linear = tau * (abs_z - 0.5 * tau)
    return ops.where(abs_z <= tau, quadratic, linear)


def _huber_derivative(z: Any, tau: float = _DEFAULT_HUBER_TAU) -> Any:
    return ops.clip(z, -tau, tau)


def _huber_integral(z: Any, tau: float = _DEFAULT_HUBER_TAU) -> Any:
    abs_z = ops.abs(z)
    inside = z * z * z / 6.0
    outside_mag = 0.5 * tau * abs_z * abs_z - 0.5 * tau * tau * abs_z + tau**3 / 6.0
    outside = ops.sign(z) * outside_mag
    return ops.where(abs_z <= tau, inside, outside)


def _huber_fastpath(z: Any, n: int, tau: float = _DEFAULT_HUBER_TAU) -> Any:
    """Closed-form ``huber^(n)`` on the almost-everywhere / regular-part convention.

    ``n = 2`` is the indicator of the quadratic region, ``n >= 3`` is zero; the
    singular part at the ``+/-tau`` kinks is dropped.
    """
    if n < 0:
        raise ValueError(f"order n must be >= 0, got {n}.")
    if n == 0:
        return _huber_forward(z, tau)
    if n == 1:
        return _huber_derivative(z, tau)
    if n == 2:
        return ops.cast(ops.abs(z) <= tau, dtype=z.dtype)
    return ops.zeros_like(z)


def make_huber_spec(
    tau: float = _DEFAULT_HUBER_TAU, *, register: bool = False
) -> ActivationSpec[Any]:
    """Build a Huber :class:`ActivationSpec` with custom threshold ``tau``."""
    if tau <= 0:
        raise ValueError(f"Huber tau must be > 0, got {tau}.")

    def fwd(z: Any) -> Any:
        return _huber_forward(z, tau)

    def deriv(z: Any) -> Any:
        return _huber_derivative(z, tau)

    def integ(z: Any) -> Any:
        return _huber_integral(z, tau)

    def fp(z: Any, n: int) -> Any:
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


def _arctan_forward(z: Any) -> Any:
    return ops.arctan(z)


def _arctan_derivative(z: Any) -> Any:
    return 1.0 / (1.0 + z * z)


def _arctan_integral(z: Any) -> Any:
    return z * ops.arctan(z) - 0.5 * ops.log1p(z * z)


def _arctan_fastpath(z: Any, n: int) -> Any:
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
    )
)


# --- log1pu2 --------------------------------------------------------------


def _log1pu2_forward(z: Any) -> Any:
    return ops.log1p(z * z)


def _log1pu2_derivative(z: Any) -> Any:
    return 2.0 * z / (1.0 + z * z)


def _log1pu2_integral(z: Any) -> Any:
    return z * ops.log1p(z * z) - 2.0 * z + 2.0 * ops.arctan(z)


def _log1pu2_fastpath(z: Any, n: int) -> Any:
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
