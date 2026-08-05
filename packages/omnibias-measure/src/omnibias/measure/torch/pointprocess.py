# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Temporal point processes & survival analysis on the measure integral (torch).

The likelihood of an inhomogeneous Poisson / temporal point process with
intensity :math:`\lambda(t) \ge 0` and events :math:`t_1 < \dots < t_n` on
:math:`[t_0, T]` is

.. math::

    \log L = \sum_i \log \lambda(t_i) - \underbrace{\int_{t_0}^{T} \lambda(t)\, dt}_{\text{compensator } \Lambda},

and a right-censored survival observation :math:`(d, \delta)` with hazard
:math:`h` has :math:`\log L = \delta \log h(d) - \int_0^{d} h(u)\,du` (the
cumulative hazard is the survival compensator). The hard term is the
**compensator** :math:`\Lambda`, normally Monte-Carlo / quadrature approximated.

omnibias evaluates it two ways:

* :func:`compensator` -- the general route: the measure integral
  :math:`\int \lambda\, d\mu` over a Gauss-Legendre (or Monte-Carlo) measure on
  :math:`[t_0, T]`, differentiable through the intensity network
  (``numerical`` -- a high-order quadrature, exact for polynomial intensities);
* :func:`closed_form_compensator` -- the omnibias-native route for an
  OMBU-style intensity :math:`\lambda(t) = c\,\sigma(w t + b)`: the exact
  antiderivative window :math:`(c/w)\,[S(wT+b) - S(wt_0+b)]` with
  :math:`S' = \sigma`, reusing the registered closed-form antiderivative kernel
  (``closed form``, e.g. ``sigma``'s antiderivative is ``softplus``).

Both are differentiable in the intensity parameters, so a network intensity
trains by maximum likelihood. Bit-identical jax twin in
:mod:`omnibias.measure.jax.pointprocess`.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from omnibias.measure._core.measure import Measure, lebesgue, uniform_mc
from omnibias.measure.torch import ops
from torch import Tensor, nn

#: A torch intensity / hazard ``time (m,) -> rate (m,)`` (non-negative output).
RateFn = Callable[[Tensor], Tensor]


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
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""Compensator ``Lambda = int_{t0}^{t1} intensity(t) dt`` via measure quadrature.

    ``intensity`` maps a time tensor ``(m,)`` to non-negative rates ``(m,)``.
    Differentiable through the intensity parameters; ``numerical`` (a high-order
    Gauss-Legendre rule, exact for polynomial intensities up to degree
    ``2*num - 1``).
    """
    measure = _time_measure(t0, t1, num, rule, seed=seed)

    def f(nodes: Tensor) -> Tensor:
        return intensity(nodes[:, 0]).reshape(-1)

    return ops.lebesgue_integral(f, measure, dtype=dtype, device=device)


def closed_form_compensator(
    activation: str,
    w: float | Tensor,
    b: float | Tensor,
    t0: float,
    t1: float,
    *,
    scale: float | Tensor = 1.0,
) -> Tensor:
    r"""Exact compensator of an OMBU intensity ``scale * activation(w t + b)``.

    Uses the closed-form antiderivative kernel ``S`` (``S' = activation``) from
    the omnibias activation registry:
    ``int_{t0}^{t1} scale * act(w t + b) dt = (scale / w) [S(w t1 + b) - S(w t0 + b)]``.
    This is the fundamental-theorem twin of the derivative tower -- **closed
    form** (no quadrature) -- and differentiable in ``scale`` / ``w`` / ``b``.
    Requires the activation to ship an antiderivative (e.g. ``sigmoid`` ->
    ``softplus``, ``exp`` -> ``exp``) and ``w != 0``.
    """
    from omnibias.torch.activations.registry import get_activation

    spec = get_activation(activation)
    if spec.integral is None:
        raise ValueError(
            f"activation {activation!r} has no closed-form antiderivative kernel; "
            "use compensator(...) (quadrature) instead"
        )
    s_kernel = spec.integral
    w_t = torch.as_tensor(w)
    b_t = torch.as_tensor(b, dtype=w_t.dtype if w_t.is_floating_point() else None)
    scale_t = torch.as_tensor(scale)
    if bool(torch.any(w_t == 0)):
        raise ValueError("closed_form_compensator requires w != 0")
    hi = s_kernel(w_t * t1 + b_t)
    lo = s_kernel(w_t * t0 + b_t)
    out: Tensor = scale_t / w_t * (hi - lo)
    return out


def poisson_nll(
    intensity: RateFn,
    events: Tensor,
    t0: float,
    t1: float,
    *,
    num: int = 256,
    rule: str = "gauss_legendre",
    seed: int = 0,
    eps: float = 1e-12,
) -> Tensor:
    r"""Negative log-likelihood of an inhomogeneous Poisson process.

    ``-[ sum_i log intensity(t_i) - int_{t0}^{t1} intensity ]`` for events
    ``t_i`` in ``[t0, t1]``. Minimising it is maximum-likelihood fitting of the
    intensity network; differentiable end-to-end.
    """
    ev = torch.as_tensor(events).reshape(-1)
    lam = intensity(ev).reshape(-1)
    comp = compensator(intensity, t0, t1, num=num, rule=rule, seed=seed, dtype=lam.dtype)
    log_term = torch.sum(torch.log(lam + eps))
    return comp - log_term


def survival_nll(
    hazard: RateFn,
    durations: Tensor,
    observed: Tensor,
    *,
    num: int = 128,
    rule: str = "gauss_legendre",
    seed: int = 0,
    eps: float = 1e-12,
) -> Tensor:
    r"""Right-censored survival negative log-likelihood.

    For samples ``(d_k, delta_k)`` (``delta_k = 1`` event, ``0`` right-censored)
    with hazard ``h``, returns
    ``-sum_k [ delta_k log h(d_k) - H(d_k) ]`` where the cumulative hazard
    ``H(d_k) = int_0^{d_k} h(u) du`` is a per-sample rescaled quadrature on
    ``[0, 1]`` (so all samples share one reference rule). Differentiable through
    the hazard parameters.
    """
    d = torch.as_tensor(durations).reshape(-1)
    obs = torch.as_tensor(observed, dtype=d.dtype).reshape(-1)
    if d.shape != obs.shape:
        raise ValueError(f"durations {tuple(d.shape)} and observed {tuple(obs.shape)} must match")
    unit = lebesgue([(0.0, 1.0)], num) if rule == "gauss_legendre" else uniform_mc(
        [(0.0, 1.0)], num, seed=seed
    )
    u = torch.as_tensor(unit.nodes[:, 0], dtype=d.dtype, device=d.device)  # (num,)
    wq = torch.as_tensor(unit.weights, dtype=d.dtype, device=d.device)  # (num,)
    pts = d[:, None] * u[None, :]  # (K, num)
    hz = hazard(pts.reshape(-1)).reshape(pts.shape)  # (K, num)
    cum_hazard = d * torch.sum(wq[None, :] * hz, dim=1)  # (K,)
    log_h = torch.log(hazard(d).reshape(-1) + eps)  # (K,)
    return -torch.sum(obs * log_h - cum_hazard)


class TemporalPointProcess(nn.Module):
    r"""Trainable temporal point process wrapping an intensity ``nn.Module``.

    The ``intensity`` submodule maps a time tensor ``(m,)`` to non-negative
    rates ``(m,)`` (end it in ``softplus`` / ``exp`` to guarantee positivity).
    Its parameters are registered, so ``.nll(events, t0, T)`` (the training
    objective) backpropagates into them. ``forward`` is the NLL.
    """

    def __init__(
        self,
        intensity: nn.Module,
        *,
        num: int = 256,
        rule: str = "gauss_legendre",
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.intensity = intensity
        self.num = int(num)
        self.rule = rule
        self.seed = int(seed)

    def compensator(self, t0: float, t1: float) -> Tensor:
        return compensator(self.intensity, t0, t1, num=self.num, rule=self.rule, seed=self.seed)

    def log_likelihood(self, events: Tensor, t0: float, t1: float) -> Tensor:
        return -self.nll(events, t0, t1)

    def nll(self, events: Tensor, t0: float, t1: float) -> Tensor:
        return poisson_nll(
            self.intensity, events, t0, t1, num=self.num, rule=self.rule, seed=self.seed
        )

    def forward(self, events: Tensor, t0: float, t1: float) -> Tensor:
        return self.nll(events, t0, t1)


__all__ = [
    "RateFn",
    "TemporalPointProcess",
    "closed_form_compensator",
    "compensator",
    "poisson_nll",
    "survival_nll",
]
