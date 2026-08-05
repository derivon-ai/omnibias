# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Certified argmax / measure-mode collapse: soft selection + Gibbs moments (torch).

Bit-identical twin of :mod:`omnibias.struct.jax.select` (float64). The Gibbs law
``p_beta(i) ∝ exp(beta s_i)`` over ``N`` logits collapses onto a Dirac at the mode as
``beta -> inf``; :func:`certified_argmax` pairs the relaxed selection with a sound,
closed-form :class:`~omnibias.struct.SelectionCertificate`. The moments reuse the closed-form
log-sum-exp tower (:mod:`omnibias.struct.torch._logsumexp`): the Gibbs mean is ``softmax``, the
covariance is ``hessian(lse_beta)/beta``, and the higher directional cumulants are the exact
log-partition jet ``kappa_m(v_I) = m! * jet_m / beta^m`` obtained with
:func:`omnibias.torch.jet.compose_jet` -- no autodiff, no finite differences.

**Two axes, never conflated.** The ``beta -> inf`` annealing here is the *feasibility /
temperature* sense of collapse (the same axis as ``omnibias-discrete`` / ``omnibias-qubo``) --
it is **not** the founding ``delta -> 0`` bias collapse. The founding derivative tower is only
the exact engine that differentiates ``lse_beta`` (``softplus^(n) = sigma^(n-1)``);
do not conflate the two axes.
"""

from __future__ import annotations

import math

import torch
from omnibias.struct._core.select import SelectionCertificate, certify_argmax
from omnibias.struct.torch._logsumexp import (
    logsumexp_beta,
    logsumexp_beta_hessian,
    softmax_beta,
)
from omnibias.torch.jet import compose_jet
from torch import Tensor


def soft_max_value(logits: Tensor, beta: float = 1.0, *, axis: int = -1) -> Tensor:
    r"""The relaxed max value ``lse_beta(logits)`` (``>= max``, ``-> max`` as ``beta -> inf``)."""
    return logsumexp_beta(logits, beta, axis=axis)


def soft_argmax(logits: Tensor, beta: float = 1.0, *, axis: int = -1) -> Tensor:
    r"""The relaxed one-hot selection ``softmax(beta * logits)`` (``-> e_argmax`` as ``beta -> inf``)."""
    return softmax_beta(logits, beta, axis=axis)


def gibbs_mean(logits: Tensor, beta: float = 1.0, *, axis: int = -1) -> Tensor:
    r"""The Gibbs law ``p_i = softmax(beta * logits)_i`` -- the mean of the one-hot indicator.

    Identical tensor to :func:`soft_argmax`; named for the measure/moment view (it is the
    gradient of :func:`soft_max_value`, i.e. ``E_p[e_i]``).
    """
    return softmax_beta(logits, beta, axis=axis)


def gibbs_covariance(logits: Tensor, beta: float = 1.0, *, axis: int = -1) -> Tensor:
    r"""The Gibbs one-hot covariance ``diag(p) - p p^T`` (``= hessian(lse_beta) / beta``).

    Returns a ``(..., N, N)`` tensor; the exact second moment of the categorical indicator,
    obtained in closed form from :func:`logsumexp_beta_hessian` (never autodiff).
    """
    return logsumexp_beta_hessian(logits, beta, axis=axis) / beta


def gibbs_cumulants_directional(
    logits: Tensor, v: Tensor, beta: float = 1.0, order: int = 1
) -> Tensor:
    r"""Exact cumulants ``kappa_1..kappa_order`` of ``X = v_I`` under the Gibbs law ``I ~ p``.

    ``kappa_1 = E_p[v_I] = <p, v>`` (the directional mean), ``kappa_2 = Var_p(v_I)``
    (``= v^T (diag(p) - p p^T) v``), and higher cumulants follow. They are the derivatives of
    the log-partition ``phi(t) = lse_beta(logits + t v)``: writing
    ``phi(t) = lse_beta(logits) + beta^-1 log sum_i p_i e^{t beta v_i}``, the jet of the inner
    ``log-sum`` is composed from the analytic ``log`` derivative tower via
    :func:`omnibias.torch.jet.compose_jet` (Faà di Bruno), giving
    ``kappa_m = m! * F_jet[m] / beta^m`` exactly -- no autodiff. Reduces over the last axis;
    returns a ``(order, ...)`` tensor (index ``m-1`` holds ``kappa_m``).
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    s = torch.as_tensor(logits)
    vv = torch.as_tensor(v)
    p = softmax_beta(s, beta, axis=-1)
    w = beta * vv
    # Inner jet (coefficient form): S_coeffs[k] = sum_i p_i w_i^k / k!  (S(0) = sum_i p_i = 1).
    s_rows = [p.sum(dim=-1)]
    wk = torch.ones_like(w)
    fact = 1.0
    for k in range(1, order + 1):
        wk = wk * w
        fact *= k
        s_rows.append((p * wk).sum(dim=-1) / fact)
    s_coeffs = torch.stack(s_rows, dim=0)
    s0 = s_coeffs[0]
    # Analytic log derivative tower at S0: f^(k)(S0) = (-1)^(k-1) (k-1)! / S0^k (k >= 1).
    log_rows = [torch.log(s0)]
    for k in range(1, order + 1):
        coef = ((-1.0) ** (k - 1)) * math.factorial(k - 1)
        log_rows.append(coef / (s0**k))
    log_tower = torch.stack(log_rows, dim=0)
    f_jet = compose_jet(s_coeffs, log_tower)
    cumulants = [math.factorial(m) * f_jet[m] / (beta**m) for m in range(1, order + 1)]
    return torch.stack(cumulants, dim=0)


def certified_argmax(
    logits: Tensor, beta: float, *, eps: float | None = None
) -> tuple[Tensor, SelectionCertificate]:
    r"""Relaxed selection ``softmax(beta * logits)`` plus its :class:`SelectionCertificate`.

    ``logits`` must be 1-D (``N`` choices). Returns ``(soft_output, certificate)`` where the
    certificate bounds the ``beta -> inf`` Gibbs-to-Dirac collapse in closed form (value gap
    ``log(N)/beta``, mode-mass concentration, and -- when ``eps`` is given -- ``L^inf``
    argmax-stability ``margin > 2 eps``). The soft output is differentiable; the certificate is
    a frozen numpy object suitable for :func:`~omnibias.struct.seal_selection_certificate`.
    """
    lg = torch.as_tensor(logits)
    if lg.ndim != 1:
        raise ValueError(f"certified_argmax expects 1-D logits (N choices), got shape {tuple(lg.shape)}")
    soft = softmax_beta(lg, beta, axis=-1)
    cert = certify_argmax(lg.detach().cpu().numpy(), beta, eps=eps)
    return soft, cert


__all__ = [
    "certified_argmax",
    "gibbs_covariance",
    "gibbs_cumulants_directional",
    "gibbs_mean",
    "logsumexp_beta",
    "soft_argmax",
    "soft_max_value",
    "softmax_beta",
]
