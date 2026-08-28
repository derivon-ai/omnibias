# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact multi-layer directional jets via Faà di Bruno (torch).

Bit-identical twin of :mod:`omnibias.jax.jet`; see that module for the full
mathematical exposition. A *jet* stores Taylor coefficients ``a_k = f^(k)(0)/k!``
along the leading axis of a tensor of shape ``(N+1, ...)``. The activation
composition uses the numerically-stable shifted-power identity

.. math::

    \\sigma(u(t)) = \\sum_{k=0}^{N} \\frac{\\sigma^{(k)}(u_0)}{k!}\\,(u(t)-u_0)^k,

with series powers built by truncated convolution and exact ``sigma^(k)`` from
the omnibias activation fast paths.

Truncation-order cost (no ``O(N^2)`` claim for the general kernel) is discussed
in :mod:`omnibias.jax.jet`; the two modules run the same recurrences.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from omnibias.torch.activations.registry import get_activation

import torch
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from omnibias.core.spec import ActivationSpec


def _factorials(np1: int, dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.tensor(
        [float(math.factorial(k)) for k in range(np1)], dtype=dtype, device=device
    )


def _scale_axis0(arr: Tensor, factors: Tensor) -> Tensor:
    """Multiply ``arr`` (leading axis = order) by per-order ``factors``."""
    shape = (factors.shape[0],) + (1,) * (arr.ndim - 1)
    return arr * factors.reshape(shape)


def tower_to_jet(tower: Tensor) -> Tensor:
    """Convert a derivative tower ``(f, f', f'', ...)`` to a Taylor jet.

    ``jet[k] = tower[k] / k!``. Leading axis is the derivative order.
    """
    tower = torch.as_tensor(tower)
    facts = _factorials(tower.shape[0], tower.dtype, tower.device)
    return _scale_axis0(tower, 1.0 / facts)


def jet_to_tower(jet: Tensor) -> Tensor:
    """Convert a Taylor jet to the derivative tower ``(f, f', f'', ...)``.

    ``tower[k] = jet[k] * k!``. Matches the scalar-curve Taylor-coefficient
    convention used by :mod:`jax.experimental.jet` in the validation suite.
    """
    jet = torch.as_tensor(jet)
    facts = _factorials(jet.shape[0], jet.dtype, jet.device)
    return _scale_axis0(jet, facts)


def derivative_jet(jet: Tensor) -> Tensor:
    r"""Taylor jet of ``f'`` from the jet of ``f`` (differentiation; leading axis = order).

    The derivative's Taylor coefficients are ``(k+1) * jet[k+1]``, so the jet
    shortens by one order. Works on scalar ``(N+1,)`` and vector ``(N+1, ...)``
    jets (e.g. an :func:`mlp_jet` output). Exact float inverse of
    :func:`antiderivative_jet`; the differentiation half of the Fundamental
    Theorem of Calculus in the jet register.
    """
    jet = torch.as_tensor(jet)
    np1 = jet.shape[0]
    factors = torch.arange(1, np1, dtype=jet.dtype, device=jet.device)
    return _scale_axis0(jet[1:], factors)


def antiderivative_jet(jet: Tensor, constant: float = 0.0) -> Tensor:
    r"""Taylor jet of ``F(t) = constant + \int_0^t f`` (term-by-term integration).

    ``A_0 = constant``; ``A_m = jet[m-1]/m`` for ``m >= 1``, so the jet lengthens by
    one order. Works on scalar and vector jets; the two-sided partner of
    :func:`jet_to_tower` and the exact float inverse of :func:`derivative_jet`.
    The constant of integration is a free parameter (default 0).
    """
    jet = torch.as_tensor(jet)
    np1 = jet.shape[0]
    divisors = torch.arange(1, np1 + 1, dtype=jet.dtype, device=jet.device)
    scaled = _scale_axis0(jet, 1.0 / divisors)
    const_row = torch.full_like(jet[0], float(constant)).unsqueeze(0)
    return torch.cat([const_row, scaled], dim=0)


def affine_jet(z_jet: Tensor, W: Tensor, b: Tensor | None = None) -> Tensor:
    """Push a jet through an affine map ``u = W z + b``.

    ``z_jet`` has shape ``(N+1, ..., D_in)``; ``W`` has shape ``(D_out, D_in)``;
    ``b`` (optional) has shape ``(D_out,)``. Linear in ``t``, so it acts per
    order with the bias added to the zeroth coefficient only.
    """
    z_jet = torch.as_tensor(z_jet)
    W = torch.as_tensor(W)
    out: Tensor = torch.tensordot(z_jet, W, dims=([-1], [-1]))  # (N+1, ..., D_out)
    if b is not None:
        b = torch.as_tensor(b)
        out0 = (out[0] + b).unsqueeze(0)
        out = torch.cat([out0, out[1:]], dim=0)
    return out


def compose_jet(u_jet: Tensor, sigma_tower: Tensor) -> Tensor:
    """Compose an elementwise activation onto a jet: ``b = sigma(u)``.

    ``u_jet`` is the pre-activation jet of shape ``(N+1, ...)``;
    ``sigma_tower[k] = sigma^(k)(u_jet[0])`` (same trailing shape). Returns the
    jet of ``sigma(u(t))``.

    The shifted power ``v^k``, ``v = u - u_0``, has valuation ``k``, so its
    coefficients below order ``k`` vanish identically and are never formed:
    the recurrence runs over the ``(k, n)`` triangle ``n >= k`` only. That is a
    constant-factor saving (``~N^3/6`` instead of ``~N^3/2`` elementwise
    multiply-adds) returning bit-for-bit the same values as the dense form,
    **not** an asymptotic one -- see :mod:`omnibias.jax.jet` for why no
    ``O(N^2)`` claim is made for an arbitrary tower, and
    :func:`compose_jet_riccati` for the quadratic fastpath that a Riccati
    activation does admit.

    Two documented consequences of no longer forming products that were exactly
    zero: a coefficient that *is* exactly zero may come out with the other sign
    of zero (``-0.0 == 0.0``), and an infinite or NaN tower entry no longer
    contaminates lower orders through ``inf * 0``.
    """
    u_jet = torch.as_tensor(u_jet)
    sigma_tower = torch.as_tensor(sigma_tower)
    np1 = u_jet.shape[0]
    if sigma_tower.shape[0] != np1:
        raise ValueError(
            f"sigma_tower order {sigma_tower.shape[0] - 1} must match jet order "
            f"{np1 - 1}"
        )
    zero = torch.zeros_like(u_jet[0])
    result = [sigma_tower[0] * torch.ones_like(u_jet[0])]
    result += [zero for _ in range(np1 - 1)]
    # p[j] is the order-(k + j) coefficient of v^k; v^1 == u_jet[1:] exactly.
    p = [u_jet[n] for n in range(1, np1)]
    fact = 1.0
    for k in range(1, np1):
        fact *= k
        dk = sigma_tower[k] / fact
        for n in range(k, np1):
            result[n] = result[n] + dk * p[n - k]
        if k + 1 < np1:
            # p <- truncated_conv(p, v), keeping orders k+1 .. N of v^(k+1).
            new_p = []
            for n in range(k + 1, np1):
                acc = zero
                for i in range(k, n):
                    acc = acc + p[i - k] * u_jet[n - i]
                new_p.append(acc)
            p = new_p
    return torch.stack(result, dim=0)


def compose_jet_riccati(
    u_jet: Tensor,
    sigma_u0: Tensor,
    riccati_polynomial: Sequence[float],
) -> Tensor:
    r"""Riccati-class composition ``b = sigma(u)`` in ``O(deg(P) * N^2)``.

    Specialised fastpath for an activation in the **Riccati class**, i.e. one
    whose derivative is a polynomial in the activation itself,
    ``sigma'(z) = P(sigma(z))`` -- the ``riccati_polynomial`` recorded on
    :class:`omnibias.core.spec.ActivationSpec` (``sigmoid``: ``s - s^2``,
    ``tanh``: ``1 - t^2``, ``exp``: ``E``, ``tan``: ``1 + t^2``, ...). The
    chain rule then closes on ``b`` alone,

    .. math::

        b'(t) = P(b(t))\, u'(t),

    so equating Taylor coefficients gives the forward recurrence

    .. math::

        (n+1)\, b_{n+1} = \sum_{i=0}^{n} P(b)_i \, (n-i+1)\, u_{n+1-i},

    where ``P(b)`` needs the truncated powers ``b^m``, ``m <= deg(P)``, each
    advanced one order per step. Every order costs ``O(deg(P) * n)``, hence
    ``O(deg(P) * N^2)`` overall, and the derivative tower is never built: the
    only activation evaluation is ``sigma_u0``.

    Parameters
    ----------
    u_jet
        Pre-activation jet, shape ``(N+1, ...)``.
    sigma_u0
        ``sigma(u_jet[0])``, broadcastable to the trailing shape of ``u_jet``.
    riccati_polynomial
        Ascending coefficients of ``P``, i.e. ``P(s) = sum_m c_m s^m``.

    Returns
    -------
    Tensor
        Jet of ``sigma(u(t))``, shape ``(N+1, ...)``.

    Notes
    -----
    Mathematically identical to :func:`compose_jet` fed the exact tower, but it
    is a *different* sequence of floating-point operations, so it agrees to
    rounding (``~1e-13`` relative in float64) rather than bit-for-bit; against
    its :mod:`omnibias.jax.jet` twin it agrees to a few ULP, the same footing
    :func:`compose_jet` sits on across backends. :func:`compose_jet` remains
    the default everywhere so the pinned cross-backend goldens keep their
    meaning.

    It also reaches *further* than the tower path: ``tan``, ``cot`` and ``coth``
    have order-capped fastpath kernels, so :func:`layer_jet` raises above the
    cap, while this recurrence needs only ``sigma(u_0)`` and ``P``.
    """
    u_jet = torch.as_tensor(u_jet)
    sigma_u0 = torch.as_tensor(sigma_u0)
    poly = tuple(float(c) for c in riccati_polynomial)
    if not poly:
        raise ValueError(
            "riccati_polynomial must hold at least one coefficient (P of sigma)"
        )
    np1 = u_jet.shape[0]
    b = [sigma_u0 * torch.ones_like(u_jet[0])]
    if np1 == 1:
        return torch.stack(b, dim=0)
    deg = len(poly) - 1
    du = derivative_jet(u_jet)  # du[m] = (m+1) * u_jet[m+1]
    zero = torch.zeros_like(b[0])
    # powers[m] holds the coefficients of b^m computed so far; index 1 is b.
    powers: list[list[Tensor]] = [[], b] + [[] for _ in range(max(deg - 1, 0))]
    q: list[Tensor] = []
    for n in range(np1 - 1):
        for m in range(2, deg + 1):
            acc = zero
            for j in range(n + 1):
                acc = acc + powers[m - 1][j] * b[n - j]
            powers[m].append(acc)
        qn = (zero + poly[0]) if n == 0 else zero
        for m in range(1, deg + 1):
            if poly[m] != 0.0:
                qn = qn + poly[m] * powers[m][n]
        q.append(qn)
        acc = zero
        for i in range(n + 1):
            acc = acc + q[i] * du[n - i]
        b.append(acc / float(n + 1))
    return torch.stack(b, dim=0)


def _sigma_tower(spec: ActivationSpec[Tensor], u0: Tensor, order: int) -> Tensor:
    """Stack ``sigma^(k)(u0)`` for ``k = 0..order`` with a clear order-cap error."""
    rows = [spec.forward(u0)]
    fp: Callable[[Tensor, int], Tensor] | None = spec.fastpath
    for k in range(1, order + 1):
        if fp is None:
            raise ValueError(
                f"activation {spec.name!r} has no fastpath kernel; required for "
                "jet composition"
            )
        try:
            rows.append(fp(u0, k))
        except NotImplementedError as exc:
            raise ValueError(
                f"activation {spec.name!r} fastpath does not support order {k} "
                f"required for an order-{order} jet"
            ) from exc
    return torch.stack(rows, dim=0)


def layer_jet(
    z_jet: Tensor,
    W: Tensor,
    b: Tensor | None,
    spec: ActivationSpec[Tensor] | str,
    order: int | None = None,
    *,
    riccati: bool = False,
) -> Tensor:
    """Push a jet through one ``sigma(W z + b)`` layer.

    The activation derivative tower is built from ``spec.fastpath``; passing an
    activation whose fastpath does not reach the jet order raises ``ValueError``.

    ``riccati=True`` opts into :func:`compose_jet_riccati`, which skips the
    tower entirely and costs ``O(N^2)`` instead of ``O(N^3)``; it requires
    ``spec.riccati_polynomial`` and changes only the rounding, not the
    mathematical result. The default stays ``False`` so the pinned goldens keep
    their bit-for-bit meaning.
    """
    z_jet = torch.as_tensor(z_jet)
    jet_order = z_jet.shape[0] - 1
    if order is not None and order != jet_order:
        raise ValueError(
            f"order {order} must equal the jet order {jet_order} carried by z_jet"
        )
    resolved = get_activation(spec)
    u_jet = affine_jet(z_jet, W, b)
    if riccati:
        poly = resolved.riccati_polynomial
        if poly is None:
            raise ValueError(
                f"activation {resolved.name!r} is not in the Riccati class "
                "(no riccati_polynomial); drop riccati=True to use the general "
                "tower kernel"
            )
        return compose_jet_riccati(u_jet, resolved.forward(u_jet[0]), poly)
    sigma_tower = _sigma_tower(resolved, u_jet[0], jet_order)
    return compose_jet(u_jet, sigma_tower)


def _path_jet(x0: Tensor, v: Tensor, order: int) -> Tensor:
    """Input jet for the line ``x(t) = x0 + t v`` truncated at ``order``."""
    x0 = torch.as_tensor(x0)
    v = torch.as_tensor(v)
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    rows = [x0]
    if order >= 1:
        rows.append(v)
    rows.extend(torch.zeros_like(x0) for _ in range(order - 1))
    return torch.stack(rows[: order + 1], dim=0)


def mlp_jet(
    x0: Tensor,
    v: Tensor,
    layers: Sequence[tuple[Tensor, Tensor | None, ActivationSpec[Tensor] | str | None]],
    order: int,
    *,
    riccati: bool = False,
) -> Tensor:
    """Exact directional Taylor jet of a deep MLP along ``x(t) = x0 + t v``.

    ``layers`` is a sequence of ``(W, b, spec)``; ``spec=None`` is a pure affine
    readout. Returns a jet of shape ``(N+1, C)`` whose :func:`jet_to_tower` gives
    the directional derivatives ``d^k/dt^k f(x0 + t v)`` at ``t = 0``.

    ``riccati=True`` forwards to the :func:`compose_jet_riccati` fastpath in
    every activation layer and requires all of them to be Riccati-class.
    """
    jet = _path_jet(x0, v, order)
    for W, b, spec in layers:
        if spec is None:
            jet = affine_jet(jet, W, b)
        else:
            jet = layer_jet(jet, W, b, spec, order, riccati=riccati)
    return jet


def _leading_order_scalar(coeffs: Tensor, atol: float) -> int | None:
    """First Taylor order whose coefficient exceeds ``atol`` in magnitude.

    Returns ``None`` when every coefficient is (numerically) zero. Reads host
    values, so this is a forward-only helper.
    """
    for k in range(coeffs.shape[0]):
        if abs(float(coeffs[k])) > atol:
            return k
    return None


def lhopital_ratio(num_jet: Tensor, den_jet: Tensor, order: int = 1) -> Tensor:
    r"""Differentiable L'Hopital limit of ``f(t)/g(t)`` as ``t -> 0``.

    For a ``0/0`` form whose numerator and denominator both vanish to order
    ``order - 1`` -- so ``f(t) = a_p t^p + O(t^{p+1})`` and
    ``g(t) = b_p t^p + O(t^{p+1})`` with ``p = order`` -- the limit is the ratio
    of the leading Taylor coefficients ``a_p / b_p``, i.e.
    ``num_jet[order] / den_jet[order]`` in the Taylor convention
    ``jet[k] = f^(k)(0)/k!``.

    This is the *differentiable* entry point: a plain elementwise division, so
    the limit backpropagates. Compose it with :func:`mlp_jet` to take the limit
    of a *learned* field along a ray. Use :func:`limit_of_ratio` when the
    vanishing order is not known ahead of time.
    """
    num_jet = torch.as_tensor(num_jet)
    den_jet = torch.as_tensor(den_jet)
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    max_order = min(num_jet.shape[0], den_jet.shape[0]) - 1
    if order > max_order:
        raise ValueError(f"order {order} exceeds available jet order {max_order}")
    return num_jet[order] / den_jet[order]


def limit_of_ratio(num_jet: Tensor, den_jet: Tensor, *, atol: float = 1e-12) -> Tensor:
    r"""Auto-detect the leading order of ``lim_{t->0} f(t)/g(t)`` (forward-only).

    Scans both scalar jets (shape ``(N+1,)``) for their lowest non-vanishing
    Taylor order and returns

    * ``num_jet[q] / den_jet[q]`` when numerator and denominator share leading
      order ``q`` (a genuine ``0/0`` resolved by L'Hopital),
    * ``0`` when the numerator vanishes to strictly higher order,
    * ``+/- inf`` when the denominator vanishes faster (a pole).

    Uses Python control flow on the detected order, so it is a convenience that
    is **not** differentiable at order transitions; the differentiable
    counterpart is :func:`lhopital_ratio`.
    """
    num_jet = torch.as_tensor(num_jet)
    den_jet = torch.as_tensor(den_jet)
    if num_jet.ndim != 1 or den_jet.ndim != 1:
        raise ValueError("limit_of_ratio expects scalar (1-D) jets")
    den_ord = _leading_order_scalar(den_jet, atol)
    if den_ord is None:
        raise ValueError(
            "denominator jet vanishes to all computed orders; the limit is "
            "undefined at this truncation order"
        )
    num_ord = _leading_order_scalar(num_jet, atol)
    dtype = torch.result_type(num_jet, den_jet)
    if num_ord is None or num_ord > den_ord:
        return torch.zeros((), dtype=dtype)
    if num_ord < den_ord:
        sign = float(torch.sign(num_jet[num_ord]) * torch.sign(den_jet[den_ord]))
        return torch.tensor(sign * math.inf, dtype=dtype)
    return (num_jet[den_ord] / den_jet[den_ord]).to(dtype)


def removable_value(jet: Tensor) -> Tensor:
    r"""Limit ``t -> 0`` of a path jet -- its zeroth Taylor coefficient.

    For a jet built along ``x(t) = x0 + t v`` (e.g. via :func:`mlp_jet`) the
    value at the base point is ``jet[0]``; when the jet was assembled to cancel a
    removable singularity this is the finite limit. Differentiable (no host
    synchronisation).
    """
    return torch.as_tensor(jet)[0]


__all__ = [
    "affine_jet",
    "antiderivative_jet",
    "compose_jet",
    "compose_jet_riccati",
    "derivative_jet",
    "jet_to_tower",
    "layer_jet",
    "lhopital_ratio",
    "limit_of_ratio",
    "mlp_jet",
    "removable_value",
    "tower_to_jet",
]
