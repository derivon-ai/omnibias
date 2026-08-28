# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact multi-layer directional jets via Faà di Bruno (jax).

Bit-identical twin of :mod:`omnibias.torch.jet`.

A *jet* is the truncated Taylor expansion of a scalar-parametrised path. We
store the Taylor coefficients ``a_k = f^(k)(0) / k!`` along the leading axis of
an array of shape ``(N+1, ...)`` and broadcast over the trailing axes (hidden
units, output components, batch).

For a deep network evaluated along a line ``x(t) = x0 + t v`` the pre-activation
of every hidden unit is a power series in ``t``. Propagating its jet through the
elementwise activation is exactly Faà di Bruno's formula; because omnibias
supplies the *exact* derivative tower ``sigma^(k)`` (Eulerian / Legendre /
Hermite recurrences in :mod:`omnibias.core.polynomials`), the composition is
exact and needs neither nested autodiff nor finite differences.

The activation composition uses the numerically-stable *shifted-power* identity

.. math::

    \\sigma(u(t)) = \\sum_{k=0}^{N} \\frac{\\sigma^{(k)}(u_0)}{k!}\\,(u(t)-u_0)^k,

truncated at order ``N``; series powers are built by truncated convolution. The
combinatorially-explicit Bell-polynomial form
(:func:`omnibias.core.bell.faa_di_bruno_terms`) is the cross-check used in the
test-suite, not the production path.

Truncation-order cost
---------------------

:func:`compose_jet` takes an **arbitrary** derivative tower, so what it has to
evaluate is the truncated composition of the polynomial
``D(x) = sum_k sigma^(k)(u_0) x^k / k!`` with the valuation-1 series
``v = u - u_0``. The kernel knows nothing about ``D`` beyond its coefficients,
and every output coefficient ``b_n`` really does depend on all ``sigma^(k)(u_0)``
and ``u_j`` with ``k, j <= n``, which already forces ``Omega(N^2)`` work.

* Because ``v^k`` has valuation ``k``, all coefficients of ``v^k`` below order
  ``k`` vanish identically. Restricting the recurrence to the triangle
  ``n >= k`` removes those structurally-zero products: ``~N^3/6`` elementwise
  multiply-adds instead of the dense ``~N^3/2``, returning the same values
  bit-for-bit in eager execution (see :func:`compose_jet`). The exponent is
  unchanged -- this is a constant factor, measured at 3.3x fewer multiplies at
  ``N = 16`` in ``benchmarks/jet_compose_cost.py``.
* Sub-cubic composition is not ruled out in theory: baby-step / giant-step
  (Brent-Kung) reaches ``O(N^{5/2})`` when truncated multiplication is
  schoolbook, and near-linear composition exists in the ring-operation model.
  Both buy the exponent by transforming or reordering the coefficient axis
  (FFT / Karatsuba style multiplication), which is *not* exact in floating
  point -- the Taylor coefficients here span many magnitudes, so a transform
  would trade the kernel's exactness (and the pinned cross-backend goldens) for
  the exponent. At the truncation orders this API is used at (``N`` of order
  ten) ``N^{5/2}`` with its larger constant is not even a win. So **no
  ``O(N^2)`` claim is made for** :func:`compose_jet`; it is cubic in ``N``,
  with a smaller constant than the dense form.
* ``O(N^2)`` *is* available when the outer map is not arbitrary. An activation
  in the Riccati class satisfies ``sigma' = P(sigma)`` for a polynomial ``P``
  (recorded as :attr:`omnibias.core.spec.ActivationSpec.riccati_polynomial`),
  which closes the chain rule on ``b`` itself and gives the ``O(deg(P) * N^2)``
  recurrence in :func:`compose_jet_riccati` -- also skipping tower
  construction. It is opt-in via ``riccati=True`` on :func:`layer_jet` /
  :func:`mlp_jet` because it rounds differently from :func:`compose_jet`.

.. important::

    **Bit-parity requires 64-bit JAX.** The "bit-identical twin" guarantee
    against :mod:`omnibias.torch.jet` holds in double precision. PyTorch defaults
    to ``float64`` for Python floats, whereas JAX silently truncates arrays to
    ``float32`` unless 64-bit mode is enabled. Enable it *before* the first JAX
    import::

        import jax
        jax.config.update("jax_enable_x64", True)

    (equivalently set ``JAX_ENABLE_X64=1`` / ``--jax_enable_x64`` in the env).
    Without it the jets are still internally consistent, but cross-backend
    comparisons will only match to ``float32`` tolerance, not bit-for-bit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from omnibias.jax.activations import JaxActivationSpec, get_activation

import jax.numpy as jnp
from jax import Array

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    LayerSpec = tuple[Array, Array | None, JaxActivationSpec | str | None]


def _factorials(np1: int, dtype: jnp.dtype[Any]) -> Array:
    return jnp.array([float(math.factorial(k)) for k in range(np1)], dtype=dtype)


def _scale_axis0(arr: Array, factors: Array) -> Array:
    """Multiply ``arr`` (leading axis = order) by per-order ``factors``."""
    shape = (factors.shape[0],) + (1,) * (arr.ndim - 1)
    return arr * factors.reshape(shape)


def tower_to_jet(tower: Array) -> Array:
    """Convert a derivative tower ``(f, f', f'', ...)`` to a Taylor jet.

    ``jet[k] = tower[k] / k!``. Leading axis is the derivative order.
    """
    tower = jnp.asarray(tower)
    facts = _factorials(tower.shape[0], tower.dtype)
    return _scale_axis0(tower, 1.0 / facts)


def jet_to_tower(jet: Array) -> Array:
    """Convert a Taylor jet to the derivative tower ``(f, f', f'', ...)``.

    ``tower[k] = jet[k] * k!``. This matches the scalar-curve Taylor-coefficient
    convention used by :mod:`jax.experimental.jet` in the validation suite.
    """
    jet = jnp.asarray(jet)
    facts = _factorials(jet.shape[0], jet.dtype)
    return _scale_axis0(jet, facts)


def derivative_jet(jet: Array) -> Array:
    r"""Taylor jet of ``f'`` from the jet of ``f`` (differentiation; leading axis = order).

    The derivative's Taylor coefficients are ``(k+1) * jet[k+1]``, so the jet
    shortens by one order. Works on scalar ``(N+1,)`` and vector ``(N+1, ...)``
    jets (e.g. an :func:`mlp_jet` output) and is JIT/vmap safe. Exact float
    inverse of :func:`antiderivative_jet`; the differentiation half of the
    Fundamental Theorem of Calculus in the jet register.
    """
    jet = jnp.asarray(jet)
    np1 = jet.shape[0]
    factors = jnp.arange(1, np1, dtype=jet.dtype)
    return _scale_axis0(jet[1:], factors)


def antiderivative_jet(jet: Array, constant: float = 0.0) -> Array:
    r"""Taylor jet of ``F(t) = constant + \int_0^t f`` (term-by-term integration).

    ``A_0 = constant``; ``A_m = jet[m-1]/m`` for ``m >= 1``, so the jet lengthens by
    one order. Works on scalar and vector jets, JIT/vmap safe; the two-sided
    partner of :func:`jet_to_tower` and the exact float inverse of
    :func:`derivative_jet`. The constant of integration is a free parameter
    (default 0).
    """
    jet = jnp.asarray(jet)
    np1 = jet.shape[0]
    divisors = jnp.arange(1, np1 + 1, dtype=jet.dtype)
    scaled = _scale_axis0(jet, 1.0 / divisors)
    const_row = jnp.full_like(jet[0], constant)[None, ...]
    return jnp.concatenate([const_row, scaled], axis=0)


def affine_jet(z_jet: Array, W: Array, b: Array | None = None) -> Array:
    """Push a jet through an affine map ``u = W z + b``.

    ``z_jet`` has shape ``(N+1, ..., D_in)``; ``W`` has shape ``(D_out, D_in)``;
    ``b`` (optional) has shape ``(D_out,)``. The map is linear in ``t`` so it
    acts per order, with the bias added to the zeroth coefficient only.
    """
    z_jet = jnp.asarray(z_jet)
    W = jnp.asarray(W)
    out = jnp.tensordot(z_jet, W, axes=([-1], [-1]))  # (N+1, ..., D_out)
    if b is not None:
        out = out.at[0].add(jnp.asarray(b))
    return out


def compose_jet(u_jet: Array, sigma_tower: Array) -> Array:
    """Compose an elementwise activation onto a jet: ``b = sigma(u)``.

    Parameters
    ----------
    u_jet
        Jet of the pre-activation, shape ``(N+1, ...)`` (Taylor coefficients).
    sigma_tower
        Derivative tower of the activation evaluated at ``u_jet[0]``:
        ``sigma_tower[k] = sigma^(k)(u_jet[0])``, same trailing shape as
        ``u_jet``.

    Returns
    -------
    Array
        Jet of ``sigma(u(t))``, shape ``(N+1, ...)``.

    Notes
    -----
    ``w(t) = u(t) - u_0`` has valuation ``1``, so ``w^k`` cannot contribute
    below order ``k``; the recurrence therefore runs over the triangle
    ``n >= k`` and never forms the structurally-zero coefficients. Values are
    unchanged -- bit-for-bit against the dense form in eager execution, for
    finite inputs -- so the win is a constant factor of about three, not a
    better exponent. Two documented consequences of dropping products that were
    exactly zero: a coefficient that *is* exactly zero may come out with the
    other sign of zero (``-0.0 == 0.0``), and an infinite or NaN tower entry no
    longer contaminates lower orders through ``inf * 0``. See the module
    docstring for the cost discussion and :func:`compose_jet_riccati` for the
    quadratic fastpath.
    """
    u_jet = jnp.asarray(u_jet)
    sigma_tower = jnp.asarray(sigma_tower)
    np1 = u_jet.shape[0]
    if sigma_tower.shape[0] != np1:
        raise ValueError(
            f"sigma_tower order {sigma_tower.shape[0] - 1} must match jet order "
            f"{np1 - 1}"
        )
    zero = jnp.zeros_like(u_jet[0])
    result = [sigma_tower[0] * jnp.ones_like(u_jet[0])]
    result += [zero for _ in range(np1 - 1)]
    # p[j] is the order-(k + j) coefficient of w^k; w^1 == u_jet[1:] exactly.
    p = [u_jet[n] for n in range(1, np1)]
    fact = 1.0
    for k in range(1, np1):
        fact *= k
        dk = sigma_tower[k] / fact
        for n in range(k, np1):
            result[n] = result[n] + dk * p[n - k]
        if k + 1 < np1:
            # p <- truncated_conv(p, w), keeping orders k+1 .. N of w^(k+1).
            new_p = []
            for n in range(k + 1, np1):
                acc = zero
                for i in range(k, n):
                    acc = acc + p[i - k] * u_jet[n - i]
                new_p.append(acc)
            p = new_p
    return jnp.stack(result, axis=0)


def compose_jet_riccati(
    u_jet: Array,
    sigma_u0: Array,
    riccati_polynomial: Sequence[float],
) -> Array:
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
    Array
        Jet of ``sigma(u(t))``, shape ``(N+1, ...)``.

    Notes
    -----
    Mathematically identical to :func:`compose_jet` fed the exact tower, but it
    is a *different* sequence of floating-point operations, so it agrees to
    rounding (``~1e-13`` relative in float64) rather than bit-for-bit; against
    its :mod:`omnibias.torch.jet` twin it agrees to a few ULP, the same footing
    :func:`compose_jet` sits on across backends. JIT/vmap safe.
    :func:`compose_jet` remains the default everywhere so the pinned
    cross-backend goldens keep their meaning.

    It also reaches *further* than the tower path: ``tan``, ``cot`` and ``coth``
    have order-capped fastpath kernels, so :func:`layer_jet` raises above the
    cap, while this recurrence needs only ``sigma(u_0)`` and ``P``.
    """
    u_jet = jnp.asarray(u_jet)
    sigma_u0 = jnp.asarray(sigma_u0)
    poly = tuple(float(c) for c in riccati_polynomial)
    if not poly:
        raise ValueError(
            "riccati_polynomial must hold at least one coefficient (P of sigma)"
        )
    np1 = u_jet.shape[0]
    b = [sigma_u0 * jnp.ones_like(u_jet[0])]
    if np1 == 1:
        return jnp.stack(b, axis=0)
    deg = len(poly) - 1
    du = derivative_jet(u_jet)  # du[m] = (m+1) * u_jet[m+1]
    zero = jnp.zeros_like(b[0])
    # powers[m] holds the coefficients of b^m computed so far; index 1 is b.
    powers: list[list[Array]] = [[], b] + [[] for _ in range(max(deg - 1, 0))]
    q: list[Array] = []
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
    return jnp.stack(b, axis=0)


def _sigma_tower(spec: JaxActivationSpec, u0: Array, order: int) -> Array:
    """Stack ``sigma^(k)(u0)`` for ``k = 0..order`` with a clear order-cap error."""
    rows = [spec.forward(u0)]
    fp: Callable[[Array, int], Array] | None = spec.fastpath
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
    return jnp.stack(rows, axis=0)


def layer_jet(
    z_jet: Array,
    W: Array,
    b: Array | None,
    spec: JaxActivationSpec | str,
    order: int | None = None,
    *,
    riccati: bool = False,
) -> Array:
    """Push a jet through one ``sigma(W z + b)`` layer.

    The activation derivative tower is built from ``spec.fastpath``; passing an
    activation whose fastpath does not reach the jet order raises ``ValueError``.

    ``riccati=True`` opts into :func:`compose_jet_riccati`, which skips the
    tower entirely and costs ``O(N^2)`` instead of ``O(N^3)``; it requires
    ``spec.riccati_polynomial`` and changes only the rounding, not the
    mathematical result. The default stays ``False`` so the pinned goldens keep
    their bit-for-bit meaning.
    """
    z_jet = jnp.asarray(z_jet)
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


def _path_jet(x0: Array, v: Array, order: int) -> Array:
    """Input jet for the line ``x(t) = x0 + t v`` truncated at ``order``."""
    x0 = jnp.asarray(x0)
    v = jnp.asarray(v)
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    rows = [x0]
    if order >= 1:
        rows.append(v)
    rows.extend(jnp.zeros_like(x0) for _ in range(order - 1))
    return jnp.stack(rows[: order + 1], axis=0)


def mlp_jet(
    x0: Array,
    v: Array,
    layers: Sequence[tuple[Array, Array | None, JaxActivationSpec | str | None]],
    order: int,
    *,
    riccati: bool = False,
) -> Array:
    """Exact directional Taylor jet of a deep MLP along ``x(t) = x0 + t v``.

    Parameters
    ----------
    x0, v
        Base point and direction, shape ``(D,)`` each.
    layers
        Sequence of ``(W, b, spec)``. ``spec=None`` is a pure affine readout
        (no activation); otherwise the layer computes ``sigma(W z + b)``.
    order
        Truncation order ``N``; the returned jet has ``N+1`` coefficients.
    riccati
        Opt into the :func:`compose_jet_riccati` fastpath in every activation
        layer; requires all of them to be Riccati-class.

    Returns
    -------
    Array
        Jet of shape ``(N+1, C)`` whose ``jet_to_tower`` gives the directional
        derivatives ``d^k/dt^k f(x0 + t v)`` at ``t = 0``.
    """
    jet = _path_jet(x0, v, order)
    for W, b, spec in layers:
        if spec is None:
            jet = affine_jet(jet, W, b)
        else:
            jet = layer_jet(jet, W, b, spec, order, riccati=riccati)
    return jet


def _leading_order_scalar(coeffs: Array, atol: float) -> int | None:
    """First Taylor order whose coefficient exceeds ``atol`` in magnitude.

    Returns ``None`` when every coefficient is (numerically) zero. Reads host
    values, so this is a forward-only helper (not traceable under ``jit``).
    """
    for k in range(coeffs.shape[0]):
        if abs(float(coeffs[k])) > atol:
            return k
    return None


def lhopital_ratio(num_jet: Array, den_jet: Array, order: int = 1) -> Array:
    r"""Differentiable L'Hopital limit of ``f(t)/g(t)`` as ``t -> 0``.

    For a ``0/0`` form whose numerator and denominator both vanish to order
    ``order - 1`` -- so ``f(t) = a_p t^p + O(t^{p+1})`` and
    ``g(t) = b_p t^p + O(t^{p+1})`` with ``p = order`` -- the limit is the ratio
    of the leading Taylor coefficients ``a_p / b_p``, i.e.
    ``num_jet[order] / den_jet[order]`` in the Taylor convention
    ``jet[k] = f^(k)(0)/k!``.

    This is the *differentiable* entry point: a plain elementwise division, so
    the limit backpropagates and is ``vmap``/``jit`` compatible. Compose it with
    :func:`mlp_jet` to take the limit of a *learned* field along a ray. Use
    :func:`limit_of_ratio` when the vanishing order is not known ahead of time.
    """
    num_jet = jnp.asarray(num_jet)
    den_jet = jnp.asarray(den_jet)
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    max_order = min(num_jet.shape[0], den_jet.shape[0]) - 1
    if order > max_order:
        raise ValueError(f"order {order} exceeds available jet order {max_order}")
    return num_jet[order] / den_jet[order]


def limit_of_ratio(num_jet: Array, den_jet: Array, *, atol: float = 1e-12) -> Array:
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
    num_jet = jnp.asarray(num_jet)
    den_jet = jnp.asarray(den_jet)
    if num_jet.ndim != 1 or den_jet.ndim != 1:
        raise ValueError("limit_of_ratio expects scalar (1-D) jets")
    den_ord = _leading_order_scalar(den_jet, atol)
    if den_ord is None:
        raise ValueError(
            "denominator jet vanishes to all computed orders; the limit is "
            "undefined at this truncation order"
        )
    num_ord = _leading_order_scalar(num_jet, atol)
    dtype = jnp.result_type(num_jet.dtype, den_jet.dtype)
    if num_ord is None or num_ord > den_ord:
        return jnp.asarray(0.0, dtype=dtype)
    if num_ord < den_ord:
        sign = float(jnp.sign(num_jet[num_ord]) * jnp.sign(den_jet[den_ord]))
        return jnp.asarray(sign * jnp.inf, dtype=dtype)
    return (num_jet[den_ord] / den_jet[den_ord]).astype(dtype)


def removable_value(jet: Array) -> Array:
    r"""Limit ``t -> 0`` of a path jet -- its zeroth Taylor coefficient.

    For a jet built along ``x(t) = x0 + t v`` (e.g. via :func:`mlp_jet`) the
    value at the base point is ``jet[0]``; when the jet was assembled to cancel a
    removable singularity this is the finite limit. Differentiable and
    ``jit``-friendly (no host synchronisation).
    """
    return jnp.asarray(jet)[0]


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
