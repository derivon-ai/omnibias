# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Temporal point processes & survival analysis on the measure integral (jax).

Bit-identical twin of :mod:`omnibias.measure.torch.pointprocess`. See that
module for the likelihood derivation. The compensator
:math:`\Lambda = \int_{t_0}^{T} \lambda(t)\,dt` is evaluated either by measure
quadrature (:func:`compensator`, ``numerical``) or, for an OMBU intensity
:math:`\lambda(t) = c\,\sigma(w t + b)`, by the exact antiderivative window
(:func:`closed_form_compensator`, ``closed form``). Everything is differentiable
in the intensity parameters via :func:`jax.grad`.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jax import Array
from omnibias.measure._core.measure import Measure, lebesgue, uniform_mc
from omnibias.measure.jax import ops

#: A jax intensity / hazard ``time (m,) -> rate (m,)`` (non-negative output).
RateFn = Callable[[Array], Array]


def _time_measure(t0: float, t1: float, num: int, rule: str, *, seed: int) -> Measure:
    """Build the 1-D integration measure on ``[t0, t1]`` (numpy, float64)."""
    if not t1 > t0:
        raise ValueError(f"require t0 < t1, got t0={t0}, t1={t1}")
    if rule == "gauss_legendre":
        return lebesgue([(t0, t1)], num)
    if rule == "monte_carlo":
        return uniform_mc([(t0, t1)], num, seed=seed)
    raise ValueError(f"rule must be 'gauss_legendre' or 'monte_carlo', got {rule!r}")


def compensator(
    intensity: RateFn,
    t0: float,
    t1: float,
    *,
    num: int = 256,
    rule: str = "gauss_legendre",
    seed: int = 0,
) -> Array:
    r"""Compensator ``Lambda = int_{t0}^{t1} intensity(t) dt`` via measure quadrature.

    ``intensity`` maps a time array ``(m,)`` to non-negative rates ``(m,)``.
    Differentiable through the intensity parameters; ``numerical`` (a high-order
    Gauss-Legendre rule, exact for polynomial intensities up to degree
    ``2*num - 1``).
    """
    measure = _time_measure(t0, t1, num, rule, seed=seed)

    def f(nodes: Array) -> Array:
        return intensity(nodes[:, 0]).reshape(-1)

    return ops.lebesgue_integral(f, measure)


def closed_form_compensator(
    activation: str,
    w: float | Array,
    b: float | Array,
    t0: float,
    t1: float,
    *,
    scale: float | Array = 1.0,
) -> Array:
    r"""Exact compensator of an OMBU intensity ``scale * activation(w t + b)``.

    Uses the closed-form antiderivative kernel ``S`` (``S' = activation``) from
    the omnibias activation registry:
    ``int_{t0}^{t1} scale * act(w t + b) dt = (scale / w) [S(w t1 + b) - S(w t0 + b)]``.
    **Closed form** (no quadrature) and differentiable in ``scale`` / ``w`` /
    ``b``. Requires the activation to ship an antiderivative (e.g. ``sigmoid``
    -> ``softplus``, ``exp`` -> ``exp``) and ``w != 0``.
    """
    from omnibias.jax.activations import get_activation

    spec = get_activation(activation)
    if spec.integral is None:
        raise ValueError(
            f"activation {activation!r} has no closed-form antiderivative kernel; "
            "use compensator(...) (quadrature) instead"
        )
    s_kernel = spec.integral
    w_a = jnp.asarray(w)
    b_a = jnp.asarray(b)
    scale_a = jnp.asarray(scale)
    if bool(jnp.any(w_a == 0)):
        raise ValueError("closed_form_compensator requires w != 0")
    hi = s_kernel(w_a * t1 + b_a)
    lo = s_kernel(w_a * t0 + b_a)
    out: Array = scale_a / w_a * (hi - lo)
    return out


def poisson_nll(
    intensity: RateFn,
    events: Array,
    t0: float,
    t1: float,
    *,
    num: int = 256,
    rule: str = "gauss_legendre",
    seed: int = 0,
    eps: float = 1e-12,
) -> Array:
    r"""Negative log-likelihood of an inhomogeneous Poisson process.

    ``-[ sum_i log intensity(t_i) - int_{t0}^{t1} intensity ]`` for events
    ``t_i`` in ``[t0, t1]``. Differentiable end-to-end.
    """
    ev = jnp.asarray(events).reshape(-1)
    lam = intensity(ev).reshape(-1)
    comp = compensator(intensity, t0, t1, num=num, rule=rule, seed=seed)
    log_term = jnp.sum(jnp.log(lam + eps))
    out: Array = comp - log_term
    return out


def survival_nll(
    hazard: RateFn,
    durations: Array,
    observed: Array,
    *,
    num: int = 128,
    rule: str = "gauss_legendre",
    seed: int = 0,
    eps: float = 1e-12,
) -> Array:
    r"""Right-censored survival negative log-likelihood.

    For samples ``(d_k, delta_k)`` (``delta_k = 1`` event, ``0`` right-censored)
    with hazard ``h``, returns ``-sum_k [ delta_k log h(d_k) - H(d_k) ]`` with
    cumulative hazard ``H(d_k) = int_0^{d_k} h`` a per-sample rescaled quadrature
    on ``[0, 1]``. Differentiable through the hazard parameters.
    """
    d = jnp.asarray(durations).reshape(-1)
    obs = jnp.asarray(observed).reshape(-1).astype(d.dtype)
    if d.shape != obs.shape:
        raise ValueError(f"durations {tuple(d.shape)} and observed {tuple(obs.shape)} must match")
    unit = (
        lebesgue([(0.0, 1.0)], num)
        if rule == "gauss_legendre"
        else uniform_mc([(0.0, 1.0)], num, seed=seed)
    )
    u = jnp.asarray(unit.nodes[:, 0]).astype(d.dtype)  # (num,)
    wq = jnp.asarray(unit.weights).astype(d.dtype)  # (num,)
    pts = d[:, None] * u[None, :]  # (K, num)
    hz = hazard(pts.reshape(-1)).reshape(pts.shape)  # (K, num)
    cum_hazard = d * jnp.sum(wq[None, :] * hz, axis=1)  # (K,)
    log_h = jnp.log(hazard(d).reshape(-1) + eps)  # (K,)
    out: Array = -jnp.sum(obs * log_h - cum_hazard)
    return out


class TemporalPointProcess:
    r"""Convenience wrapper around an intensity callable (jax, functional).

    Unlike the torch ``nn.Module`` twin this is a plain holder (the intensity's
    parameters live in the closure / a separate pytree, per the jax idiom); it
    simply routes ``.nll`` / ``.compensator`` / ``.log_likelihood`` to the
    functional ops so ``jax.grad`` over the closed-over parameters trains it.
    """

    def __init__(
        self,
        intensity: RateFn,
        *,
        num: int = 256,
        rule: str = "gauss_legendre",
        seed: int = 0,
    ) -> None:
        self.intensity = intensity
        self.num = int(num)
        self.rule = rule
        self.seed = int(seed)

    def compensator(self, t0: float, t1: float) -> Array:
        return compensator(self.intensity, t0, t1, num=self.num, rule=self.rule, seed=self.seed)

    def log_likelihood(self, events: Array, t0: float, t1: float) -> Array:
        return -self.nll(events, t0, t1)

    def nll(self, events: Array, t0: float, t1: float) -> Array:
        return poisson_nll(
            self.intensity, events, t0, t1, num=self.num, rule=self.rule, seed=self.seed
        )


__all__ = [
    "RateFn",
    "TemporalPointProcess",
    "closed_form_compensator",
    "compensator",
    "poisson_nll",
    "survival_nll",
]
