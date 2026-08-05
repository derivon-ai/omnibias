# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Activation-specific closed-form fractional derivatives (torch).

Some activations have a fractional derivative in *closed form via a special
function*, not merely a truncated jet series. This module registers those forms
in :data:`ACTIVATION_FRACTIONAL` (a fractional-derivative overlay on the
activation dictionary) and dispatches them through
:func:`activation_fractional_derivative`. Every entry is the Riemann-Liouville
(or Caputo) derivative from the terminal ``0`` on ``x > 0``:

* ``exp`` (``e^{lam x}``): ``{}_0 D_x^{alpha} e^{lam x} = x^{-alpha} E_{1, 1-alpha}(lam x)``
  -- exactly the Mittag-Leffler function
  :func:`~omnibias.fractional.torch.ops.special.mittag_leffler`. The Caputo variant
  drops the ``k < ceil(alpha)`` Taylor head.
* ``cosh`` / ``sinh``: linear combinations of the ``exp`` identity at ``lam = +-1``,
  ``{}_0 D^{alpha} cosh = (1/2) x^{-alpha}[E_{1,1-alpha}(x) + E_{1,1-alpha}(-x)]`` (and
  ``-`` for ``sinh``). The Mittag-Leffler arguments stay bounded (``|x|``), so the
  truncated series is machine-accurate.

Everything is differentiable in ``alpha`` (through the Mittag-Leffler ``lgamma``)
and bit-identical to the JAX twin. These are exact identities evaluated by a
differentiable truncated series (the ``terms`` tail); they hold on the stated
domain (``x > 0`` for the ``x^{-alpha}`` branch).

The logistic ``sigmoid`` is intentionally *absent*: its RL derivative from ``0``
expands as ``sum_j (-1)^{j-1} E_{1,1-alpha}(-j x)``, whose large-``j`` arguments blow
up the truncated Mittag-Leffler series, and the incomplete-gamma route is
complex-valued for that sign -- there is no robust real elementary closed form, so
we do not ship a fragile one. Use the grid/spectral operators for the logistic.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from omnibias.fractional.torch.ops.special import _recip_gamma, mittag_leffler
from torch import Tensor


def _as_scalar(value: float | Tensor, ref: Tensor) -> Tensor:
    if isinstance(value, Tensor):
        return value.to(dtype=ref.dtype, device=ref.device)
    return torch.tensor(float(value), dtype=ref.dtype, device=ref.device)


def exp_fractional(
    x: Tensor | float,
    *,
    alpha: float | Tensor,
    lam: float = 1.0,
    kind: str = "riemann_liouville",
    terms: int = 64,
) -> Tensor:
    r"""Closed-form fractional derivative of ``e^{lam x}`` from terminal ``0`` (``x > 0``).

    Riemann-Liouville: ``x^{-alpha} E_{1, 1-alpha}(lam x)``. Caputo: the same minus the
    ``k < ceil(alpha)`` head ``sum_k (lam^k / Gamma(k+1-alpha)) x^{k-alpha}``.
    """
    if kind not in ("riemann_liouville", "caputo"):
        raise ValueError(f"kind must be 'riemann_liouville' or 'caputo', got {kind!r}")
    x_t = torch.as_tensor(x, dtype=torch.get_default_dtype()) if not isinstance(x, Tensor) else x
    a = _as_scalar(alpha, x_t)
    xma = torch.exp(-a * torch.log(x_t))
    rl: Tensor = xma * mittag_leffler(lam * x_t, 1.0, 1.0 - a, terms=terms)
    if kind == "riemann_liouville":
        return rl
    m = int(math.ceil(float(a.detach())))
    head = torch.zeros_like(rl)
    for k in range(m):
        coef = (lam**k) * _recip_gamma(k + 1.0 - a)
        head = head + coef * torch.exp((k - a) * torch.log(x_t))
    return rl - head


def cosh_fractional(
    x: Tensor | float,
    *,
    alpha: float | Tensor,
    kind: str = "riemann_liouville",
    terms: int = 64,
) -> Tensor:
    r"""Closed-form fractional derivative of ``cosh`` from terminal ``0`` (``x > 0``).

    ``(1/2)[exp_fractional(x, lam=1) + exp_fractional(x, lam=-1)]``; the Mittag-Leffler
    arguments ``+-x`` stay bounded so the truncated series is machine-accurate. RL and
    Caputo both reduce to the ordinary derivatives at integer ``alpha`` (``alpha=1``
    gives ``sinh``).
    """
    pos = exp_fractional(x, alpha=alpha, lam=1.0, kind=kind, terms=terms)
    neg = exp_fractional(x, alpha=alpha, lam=-1.0, kind=kind, terms=terms)
    return 0.5 * (pos + neg)


def sinh_fractional(
    x: Tensor | float,
    *,
    alpha: float | Tensor,
    kind: str = "riemann_liouville",
    terms: int = 64,
) -> Tensor:
    r"""Closed-form fractional derivative of ``sinh`` from terminal ``0`` (``x > 0``).

    ``(1/2)[exp_fractional(x, lam=1) - exp_fractional(x, lam=-1)]`` (``alpha=1`` gives
    ``cosh``). See :func:`cosh_fractional`.
    """
    pos = exp_fractional(x, alpha=alpha, lam=1.0, kind=kind, terms=terms)
    neg = exp_fractional(x, alpha=alpha, lam=-1.0, kind=kind, terms=terms)
    return 0.5 * (pos - neg)


ACTIVATION_FRACTIONAL: dict[str, Callable[..., Tensor]] = {
    "exp": exp_fractional,
    "cosh": cosh_fractional,
    "sinh": sinh_fractional,
}


def activation_fractional_derivative(
    name: str,
    x: Tensor | float,
    *,
    alpha: float | Tensor,
    kind: str = "riemann_liouville",
    **kwargs: object,
) -> Tensor:
    r"""Dispatch to the closed-form fractional derivative registered for ``name``.

    ``name`` must be a key of :data:`ACTIVATION_FRACTIONAL` (``"exp"`` / ``"cosh"``
    / ``"sinh"``; the logistic ``sigmoid`` is intentionally absent -- see the module
    docstring). Extra keyword arguments (e.g. ``lam``, ``terms``) are forwarded to
    the identity.
    """
    if name not in ACTIVATION_FRACTIONAL:
        raise KeyError(
            f"no closed-form fractional derivative registered for {name!r}; "
            f"available: {sorted(ACTIVATION_FRACTIONAL)}"
        )
    return ACTIVATION_FRACTIONAL[name](x, alpha=alpha, kind=kind, **kwargs)


__all__ = [
    "ACTIVATION_FRACTIONAL",
    "activation_fractional_derivative",
    "cosh_fractional",
    "exp_fractional",
    "sinh_fractional",
]
