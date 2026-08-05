# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Closed-form fractional derivative of an analytic function (torch).

Unlike the grid / spectral operators in
:mod:`omnibias.fractional.torch.ops.fractional`, this operator is **closed form
on the analytic-function class**: given the Taylor jet ``a_k = f^(k)(a)/k!`` of a
function about a terminal ``a``, the fractional derivative of order ``alpha`` is

.. math::

    {}_a D_x^{\alpha} f(x)
        = \sum_{k} a_k \,\frac{\Gamma(k+1)}{\Gamma(k+1-\alpha)}\,(x-a)^{k-\alpha},

with ``t = x - a >= 0``. It is a single vectorised sum over the jet -- no grid,
no history -- and is differentiable in both the order ``alpha`` (through
``lgamma``) and the coefficients ``a_k`` (so it composes with autograd,
:func:`omnibias.torch.jet.mlp_jet`, and :class:`~omnibias.fractional.torch.order.LearnableOrder`).

Two variants share the gamma ratio:

* **Riemann-Liouville** (``kind="riemann_liouville"``) sums over all ``k``;
  it is singular at ``t = 0`` for non-integer ``alpha``.
* **Caputo** (``kind="caputo"``) drops the terms ``k < ceil(alpha)`` and is
  regular at the terminal (e.g. the Caputo derivative of a constant is ``0``).

``alpha = 0`` recovers the function itself; an integer ``alpha`` recovers the
ordinary derivative for ``t > 0`` (the low-order terms vanish through the Gamma
poles). The intended regime is *non-integer* ``alpha`` -- the alpha-gradient
exactly at an integer order is unstable (``0 * inf`` at a Gamma pole); steer
integer orders to the closed-form derivative tower or the grid ops instead.

The result is an order-``N`` truncation (``N`` = jet order): exact for
polynomials of degree ``<= N`` and otherwise accurate within the Taylor radius.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Sequence

    from omnibias.core.spec import ActivationSpec

    LayerSpec = tuple[Tensor, Tensor | None, ActivationSpec[Tensor] | str | None]


def _gamma_ratio(k: Tensor, alpha: Tensor) -> Tensor:
    r"""Real gamma ratio ``Gamma(k+1) / Gamma(k+1-alpha)``.

    Computed as ``sign * exp(lgamma(k+1) - lgamma(k+1-alpha))`` so it is smooth
    in ``alpha`` and stays real: ``lgamma`` is ``log|Gamma|`` and the explicit
    ``sign`` restores the sign of ``Gamma`` on its negative arguments
    (``Gamma(y) < 0`` for ``y`` in ``(-1, 0), (-3, -2), ...``). At a Gamma pole
    (``k + 1 - alpha`` a non-positive integer) ``lgamma`` is ``+inf`` and the
    ratio collapses to ``0`` -- exactly the vanishing low-order terms of an
    integer-order derivative. This form is shared verbatim with the JAX twin
    because torch has no ``torch.special.gamma``.
    """
    y = (k + 1.0) - alpha
    log_mag = torch.lgamma(k + 1.0) - torch.lgamma(y)
    parity = torch.remainder(torch.ceil(-y), 2.0)
    sign = torch.where(y > 0, torch.ones_like(y), 1.0 - 2.0 * parity)
    return sign * torch.exp(log_mag)


def fractional_derivative(
    jet: Tensor,
    x: Tensor | float,
    *,
    alpha: float | Tensor,
    a: float = 0.0,
    kind: str = "riemann_liouville",
) -> Tensor:
    r"""Closed-form fractional derivative of an analytic function from its jet.

    Parameters
    ----------
    jet
        Taylor coefficients ``a_k = f^(k)(a)/k!`` about the terminal ``a`` along
        the leading axis, shape ``(N+1, *feat)``. Trailing axes are treated as
        independent channels.
    x
        Evaluation point(s); ``t = x - a >= 0`` is required (real branch). The
        output has shape ``(*feat, *x.shape)`` -- every channel evaluated at
        every point.
    alpha
        Fractional order. A tensor (e.g. an ``nn.Parameter`` or
        :class:`~omnibias.fractional.torch.order.LearnableOrder` output) keeps
        the gradient path to the order; a ``float`` is promoted to ``jet``'s
        dtype/device.
    a
        Terminal (expansion point) of the jet; the branch point of the power
        ``t^{k-alpha}``.
    kind
        ``"riemann_liouville"`` (sum over all ``k``) or ``"caputo"`` (drop
        ``k < ceil(alpha)``).
    """
    if kind not in ("riemann_liouville", "caputo"):
        raise ValueError(
            f"kind must be 'riemann_liouville' or 'caputo', got {kind!r}"
        )
    if jet.ndim < 1:
        raise ValueError("jet must have a leading order axis, shape (N+1, ...)")

    n1 = jet.shape[0]
    if isinstance(alpha, Tensor):
        a_t = alpha.to(dtype=jet.dtype, device=jet.device)
    else:
        a_t = torch.tensor(float(alpha), dtype=jet.dtype, device=jet.device)

    k = torch.arange(n1, dtype=jet.dtype, device=jet.device)
    ratio = _gamma_ratio(k, a_t)
    if kind == "caputo":
        ratio = ratio * (k >= torch.ceil(a_t)).to(jet.dtype)

    x_t = torch.as_tensor(x, dtype=jet.dtype, device=jet.device)
    t = x_t - a

    fdims = jet.ndim - 1
    pdims = t.ndim
    shape_k = (n1,) + (1,) * fdims + (1,) * pdims
    ratio_e = ratio.reshape(shape_k)
    exps_e = (k - a_t).reshape(shape_k)
    jet_e = jet.reshape(jet.shape + (1,) * pdims)
    t_e = t.reshape((1,) * (1 + fdims) + t.shape)

    powers = t_e**exps_e
    # Masked / vanishing terms (ratio == 0, e.g. Caputo below ceil(alpha) or a
    # Gamma pole) contribute exactly 0 even at the terminal, where ``t**exps``
    # is ``0**negative = inf`` and ``0 * inf`` would otherwise be ``nan``.
    terms = torch.where(
        ratio_e == 0.0, torch.zeros_like(powers), jet_e * ratio_e * powers
    )
    out: Tensor = terms.sum(dim=0)
    return out


def _patch_weights(term: Tensor, x: Tensor, blend: float) -> Tensor:
    r"""Patch-selection weights ``w`` of shape ``(M, *x)`` that sum to 1 per point.

    ``blend <= 0`` gives the hard indicator (each point routed to the last terminal
    ``<= x``); ``blend > 0`` gives a telescoping sigmoid partition-of-unity across
    the ``M-1`` interior boundaries ``sigmoid((x - term[j]) / blend)`` -- a
    differentiable soft patch selection whose ``blend -> 0`` limit is the hard rule
    (this is what makes the stitched field continuous across patch boundaries).
    """
    m = int(term.shape[0])
    if m == 1:
        return torch.ones((1, *x.shape), dtype=x.dtype, device=x.device)
    if blend > 0.0:
        bnds = term[1:].reshape((m - 1,) + (1,) * x.ndim)
        p = torch.sigmoid((x.unsqueeze(0) - bnds) / blend)  # (M-1, *x)
        w0 = 1.0 - p[:1]
        wlast = p[-1:]
        if m == 2:
            return torch.cat([w0, wlast], dim=0)
        return torch.cat([w0, p[:-1] - p[1:], wlast], dim=0)
    idx = torch.searchsorted(term, x.reshape(-1), right=True).reshape(x.shape) - 1
    idx = idx.clamp(0, m - 1)
    rng = torch.arange(m, device=x.device).reshape((m,) + (1,) * x.ndim)
    return (idx.unsqueeze(0) == rng).to(x.dtype)


def piecewise_fractional_derivative(
    jets: Tensor,
    terminals: Sequence[float] | Tensor,
    x: Tensor | float,
    *,
    alpha: float | Tensor,
    kind: str = "riemann_liouville",
    blend: float = 0.0,
    gap: float = 1e-6,
) -> Tensor:
    r"""Piecewise-analytic (multi-terminal) closed-form fractional derivative.

    Orchestrates the single-terminal :func:`fractional_derivative` over a partition
    of the domain by strictly increasing ``terminals`` ``a_0 < a_1 < ... < a_{M-1}``,
    each carrying its own local Taylor jet ``jets[i]`` (shape ``(M, N+1, *feat)``).
    A point ``x`` is served by the local expansion about the largest terminal
    ``a_i <= x`` (its "matched terminal"), so the evaluation always stays inside one
    patch's Taylor radius -- extending validity to piecewise-analytic ``f`` well
    beyond a single expansion's radius.

    Because the fractional lower terminal is *restarted* at each patch edge, this is
    the closed-form **short-memory** operator (Podlubny's short-memory principle):
    the memory runs from ``a_i``, not from a global origin. With ``blend > 0`` the
    per-patch results are stitched by a smooth sigmoid partition of unity so the
    field is continuous across boundaries; ``blend = 0`` (default) selects hard.

    Fully vectorised over patches and points and differentiable in ``alpha`` (via the
    gamma ratio) and the jets. Each patch is evaluated at ``max(x, a_i + gap)`` so the
    Riemann-Liouville power stays finite at the singular terminal (``gap > 0``); for
    ``x`` inside a patch the result is exact. Requires ``x >= a_0``.
    """
    if jets.ndim < 2:
        raise ValueError("jets must have shape (M, N+1, *feat)")
    term_list = [float(t) for t in terminals]
    m = jets.shape[0]
    if len(term_list) != m:
        raise ValueError(f"expected {m} terminals (one per jet), got {len(term_list)}")
    if any(term_list[i] >= term_list[i + 1] for i in range(m - 1)):
        raise ValueError("terminals must be strictly increasing")

    x_t = torch.as_tensor(x, dtype=jets.dtype, device=jets.device)
    if float(x_t.min()) < term_list[0] - 1e-9:
        raise ValueError(
            f"every x must be >= the first terminal {term_list[0]}, got min {float(x_t.min())}"
        )
    feat = jets.ndim - 2

    parts = [
        fractional_derivative(
            jets[i], torch.clamp(x_t, min=term_list[i] + gap), alpha=alpha, a=term_list[i], kind=kind
        )
        for i in range(m)
    ]
    stack = torch.stack(parts, dim=0)  # (M, *feat, *x)

    term_t = torch.tensor(term_list, dtype=x_t.dtype, device=x_t.device)
    weights = _patch_weights(term_t, x_t, blend)  # (M, *x)
    w_e = weights.reshape((m,) + (1,) * feat + tuple(x_t.shape))
    out: Tensor = (w_e * stack).sum(dim=0)
    return out


def mlp_fractional_derivative(
    x0: Tensor,
    v: Tensor,
    layers: Sequence[LayerSpec],
    t: Tensor | float,
    *,
    alpha: float | Tensor,
    order: int,
    kind: str = "riemann_liouville",
) -> Tensor:
    r"""Closed-form directional fractional derivative of a deep MLP.

    Builds the exact directional Taylor jet of ``f`` along the ray
    ``x(s) = x0 + s v`` with :func:`omnibias.torch.jet.mlp_jet` (terminal at the
    base point ``s = 0``), then applies :func:`fractional_derivative` in the
    scalar path parameter ``s`` evaluated at the offsets ``t >= 0``. Requires the
    ``omnibias-torch`` package (imported lazily).
    """
    from omnibias.torch.jet import mlp_jet

    jet = mlp_jet(x0, v, layers, order)
    return fractional_derivative(jet, t, alpha=alpha, a=0.0, kind=kind)


__all__ = [
    "fractional_derivative",
    "mlp_fractional_derivative",
    "piecewise_fractional_derivative",
]
