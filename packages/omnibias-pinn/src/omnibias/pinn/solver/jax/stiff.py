# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Stiff integrators as differentiable layers (JAX twin).

Bit-parity twin of :mod:`omnibias.pinn.solver.torch.stiff`. A stiff problem is
one whose fastest mode is far faster than anything you care about, so an
explicit method is stable only at a step size set by physics you are not trying
to resolve. Every scheme here removes that restriction through the same object,
the **phi functions**

.. math::

   \varphi_0(z) = e^z, \qquad
   \varphi_{k}(z) = \sum_{j \ge 0} \frac{z^{j}}{(j+k)!},

evaluated **from the series** (:func:`phi_diagonal`, :func:`phi_matrix`) rather
than from the differenced closed form. ``(e^z - 1)/z`` loses every significant
digit as ``z -> 0``, which is exactly where the well-resolved modes live; that
cancellation is why the reference ETDRK4 evaluates these coefficients by contour
integration. Summing the series is the cancellation-free route to the same
numbers, and it is the same Taylor-coefficient machinery the rest of omnibias
runs on.

* :func:`etdrk4_step` -- Cox-Matthews ETDRK4 for ``u_t = L u + N(u)`` with a
  Fourier symbol ``L``: the linear part is integrated exactly.
* :func:`imex_euler_step`, :func:`imex_cnab2_step` -- cheaper implicit-explicit
  splits, first and second order.
* :func:`rosenbrock_step` -- L-stable ROS2 for a dense ODE system; linearly
  implicit, so two linear solves and no Newton iteration to fail.
* :func:`exponential_rosenbrock_step` -- ``u + h phi_1(hJ) f(u)``, exact for an
  affine right-hand side at any step size.
* :func:`closed_form_jacobian` -- ``J`` for a neural right-hand side from the
  exact multivariate jet, no autodiff and no finite difference.

Honesty
-------
These are **numerical** integrators with genuine local truncation error; only
the linear part of an ETD step and the phi functions themselves are exact.
Certified enclosures of the true flow are a different tier entirely --
:mod:`omnibias.core.verified.lohner` and :mod:`omnibias.dynamics`.

References
----------
Cox & Matthews, JCP 176 (2002). Kassam & Trefethen, SIAM J. Sci. Comput. 26
(2005). Hochbruck & Ostermann, *Exponential integrators*, Acta Numerica 19
(2010). Verwer et al., SIAM J. Sci. Comput. 20 (1999).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jax import Array
from omnibias.jax.jet_mv import jet_gradient, mlp_jet_mv

# gamma = 1 + 1/sqrt(2) is the root of 1 - 2/g + 1/(2 g^2), i.e. the value that
# sends ROS2's stability function to *exactly* zero at infinity (L-stability).
ROS2_GAMMA = 1.0 + 1.0 / math.sqrt(2.0)


def _factorials(n: int) -> list[float]:
    return [float(math.factorial(i)) for i in range(n + 1)]


def _squarings_for(scale: float) -> int:
    """How many doublings bring ``scale`` below ``1/2``."""
    if not math.isfinite(scale):
        raise ValueError("phi functions need a finite operator norm")
    if scale <= 0.5:
        return 0
    return int(math.ceil(math.log2(2.0 * scale)))


def phi_diagonal(
    z: Array, k_max: int, *, order: int = 20, squarings: int | None = None
) -> Array:
    r"""``phi_0..phi_k_max`` evaluated elementwise, shape ``(k_max + 1, *z.shape)``.

    ``z`` is the *scaled* argument ``dt * lambda`` -- a Fourier symbol times the
    step, so entries may be complex and range over many orders of magnitude in a
    single array. Small entries are handled by the series (no cancellation) and
    large ones by scaling-and-squaring, so both ends of a stiff spectrum are
    accurate at once.

    Parameters
    ----------
    squarings:
        Number of doublings. ``None`` picks it from ``max |z|``, which needs
        concrete values; pass an explicit count to stay traceable under
        ``jax.jit``.
    """
    if k_max < 0:
        raise ValueError(f"k_max must be >= 0, got {k_max}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    s = (
        _squarings_for(float(jnp.abs(z).max())) if squarings is None else int(squarings)
    )
    if s < 0:
        raise ValueError(f"squarings must be >= 0, got {s}")
    zs = z / float(2**s)
    fact = _factorials(order + k_max)
    phis = []
    for k in range(k_max + 1):
        acc = jnp.full_like(zs, 1.0 / fact[order + k])
        for j in range(order - 1, -1, -1):
            acc = acc * zs + 1.0 / fact[j + k]
        phis.append(acc)
    for _ in range(s):
        phis = _double_diagonal(phis)
    return jnp.stack(phis, axis=0)


def _double_diagonal(phis: list[Array]) -> list[Array]:
    r"""``phi_k(2z)`` from ``phi_k(z)``: the exponential's doubling identity.

    ``phi_0(2z) = phi_0(z)^2`` and, for ``k >= 1``,
    ``phi_k(2z) = 2^-k [phi_0 phi_k + sum_{j=1..k} phi_j / (k-j)!]``. Every term
    is a sum of like-signed quantities for real ``z``, so undoing the scaling
    costs no accuracy.
    """
    fact = _factorials(len(phis))
    out = [phis[0] * phis[0]]
    for k in range(1, len(phis)):
        acc = phis[0] * phis[k]
        for j in range(1, k + 1):
            acc = acc + phis[j] / fact[k - j]
        out.append(acc / float(2**k))
    return out


def phi_matrix(
    a: Array, k_max: int, *, order: int = 20, squarings: int | None = None
) -> Array:
    r"""``phi_0..phi_k_max`` of a square matrix, shape ``(k_max + 1, n, n)``.

    Same series and the same doubling identity as :func:`phi_diagonal`, with
    matrix products in place of elementwise ones. ``phi_0(A)`` is the matrix
    exponential; it is computed here rather than delegated so the torch and jax
    twins run *the same algorithm* and agree to round-off, which a pair of
    library ``expm`` implementations would not.
    """
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"phi_matrix needs a square matrix, got shape {tuple(a.shape)}")
    if k_max < 0:
        raise ValueError(f"k_max must be >= 0, got {k_max}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    s = (
        _squarings_for(float(jnp.abs(a).sum(axis=-1).max()))
        if squarings is None
        else int(squarings)
    )
    if s < 0:
        raise ValueError(f"squarings must be >= 0, got {s}")
    scaled = a / float(2**s)
    eye = jnp.eye(a.shape[0], dtype=a.dtype)
    fact = _factorials(order + k_max)
    phis = []
    for k in range(k_max + 1):
        acc = eye / fact[order + k]
        for j in range(order - 1, -1, -1):
            acc = acc @ scaled + eye / fact[j + k]
        phis.append(acc)
    for _ in range(s):
        phis = _double_matrix(phis, eye)
    return jnp.stack(phis, axis=0)


def _double_matrix(phis: list[Array], eye: Array) -> list[Array]:
    """Matrix form of the doubling identity used by :func:`phi_matrix`."""
    fact = _factorials(len(phis))
    out = [phis[0] @ phis[0]]
    for k in range(1, len(phis)):
        acc = phis[0] @ phis[k]
        for j in range(1, k + 1):
            acc = acc + phis[j] / fact[k - j]
        out.append(acc / float(2**k))
    return out


# ---------------------------------------------------------------------
# Jacobians
# ---------------------------------------------------------------------


def dense_jacobian(rhs: Callable[[Array], Array], u: Array) -> Array:
    """``(n, n)`` Jacobian ``df/du`` by reverse-mode autodiff.

    The general fallback: correct for any ``rhs``, and labelled autodiff rather
    than closed form because that is what it is. When the right-hand side is an
    omnibias layer stack, :func:`closed_form_jacobian` is exact and cheaper.
    """
    if u.ndim != 1:
        raise ValueError(f"dense_jacobian expects a flat state (n,), got {tuple(u.shape)}")
    out: Array = jax.jacrev(rhs)(u)
    return out


def closed_form_jacobian(
    layers: Sequence[tuple[Array, Array | None, object]], u: Array
) -> Array:
    """``(C, n)`` Jacobian of an omnibias MLP right-hand side, closed form.

    ``layers`` is the ``(W, b, spec)`` stack that
    :func:`~omnibias.jax.jet_mv.mlp_jet_mv` consumes. The order-1 jet holds
    every first partial already, so the Jacobian falls out of one evaluation of
    the exact tower -- no autodiff graph, no finite difference, and the same
    numbers on both backends.
    """
    if u.ndim != 1:
        raise ValueError(
            f"closed_form_jacobian expects a flat state (n,), got {tuple(u.shape)}"
        )
    n = int(u.shape[0])
    jet = mlp_jet_mv(u, layers, 1)  # type: ignore[arg-type]
    grad = jet_gradient(jet, n, 1)  # (n, C): d f_c / d u_i
    return jnp.transpose(grad, (1, 0))


# ---------------------------------------------------------------------
# Dense stiff steps
# ---------------------------------------------------------------------


def rosenbrock_step(
    rhs: Callable[[Array], Array],
    u: Array,
    dt: float,
    *,
    jacobian: Array | Callable[[Array], Array] | None = None,
    gamma: float = ROS2_GAMMA,
) -> Array:
    r"""One L-stable, second-order ROS2 step of ``u' = f(u)``.

    .. math::

       (I - \gamma h J) k_1 &= h f(u) \\
       (I - \gamma h J) k_2 &= h f(u + k_1) - 2 k_1 \\
       u_{n+1} &= u + \tfrac32 k_1 + \tfrac12 k_2

    Linearly implicit: two solves with the *same* matrix, no Newton loop, so
    there is no iteration that can fail to converge -- the step always returns
    an answer, and its cost is known in advance.

    ``J`` need not be exact (this is a W-method: an approximate Jacobian costs
    accuracy in the stiff limit, never correctness of the order). Pass a matrix,
    a callable ``u -> J``, or leave it ``None`` for :func:`dense_jacobian`.

    With ``gamma = 1 + 1/sqrt(2)`` the stability function vanishes at infinity,
    so an infinitely stiff decaying mode is annihilated in one step instead of
    ringing -- the property that separates a usable stiff solver from a merely
    stable one.
    """
    if u.ndim != 1:
        raise ValueError(f"rosenbrock_step expects a flat state (n,), got {tuple(u.shape)}")
    j = _resolve_jacobian(rhs, u, jacobian)
    h = float(dt)
    eye = jnp.eye(u.shape[0], dtype=u.dtype)
    lhs = eye - (float(gamma) * h) * j
    lu = jax.scipy.linalg.lu_factor(lhs)  # one factorisation, two solves
    k1 = jax.scipy.linalg.lu_solve(lu, h * rhs(u))
    b2 = h * rhs(u + k1) - 2.0 * k1
    k2 = jax.scipy.linalg.lu_solve(lu, b2)
    out: Array = u + 1.5 * k1 + 0.5 * k2
    return out


def exponential_rosenbrock_step(
    rhs: Callable[[Array], Array],
    u: Array,
    dt: float,
    *,
    jacobian: Array | Callable[[Array], Array] | None = None,
    order: int = 20,
    squarings: int | None = None,
) -> Array:
    r"""One exponential Rosenbrock-Euler step ``u + h phi_1(h J) f(u)``.

    **Exact** whenever ``f`` is affine (``f(u) = Ju + b``), at any step size:
    the scheme integrates the linearisation in closed form and only the
    remainder is approximated, which is why it beats ROS2 on problems that are
    stiff *and* nearly linear. Second order in general.
    """
    if u.ndim != 1:
        raise ValueError(
            f"exponential_rosenbrock_step expects a flat state (n,), got {tuple(u.shape)}"
        )
    j = _resolve_jacobian(rhs, u, jacobian)
    h = float(dt)
    phis = phi_matrix(h * j, 1, order=order, squarings=squarings)
    return u + h * (phis[1] @ rhs(u))


def _resolve_jacobian(
    rhs: Callable[[Array], Array],
    u: Array,
    jacobian: Array | Callable[[Array], Array] | None,
) -> Array:
    if jacobian is None:
        return dense_jacobian(rhs, u)
    if callable(jacobian):
        return jacobian(u)
    return jacobian


# ---------------------------------------------------------------------
# Diagonal (spectral) stiff steps
# ---------------------------------------------------------------------


def _to_hat(u: Array) -> Array:
    return jnp.fft.fft(u)


def _from_hat(uh: Array) -> Array:
    return jnp.real(jnp.fft.ifft(uh))


def imex_euler_step(
    symbol: Array,
    nonlinear: Callable[[Array], Array],
    u: Array,
    dt: float,
) -> Array:
    r"""First-order IMEX: ``(1 - dt L) u_{n+1} = u_n + dt N(u_n)`` in Fourier space.

    The stiff linear part is implicit (unconditionally stable, and damped at
    every wavenumber), the nonlinear part explicit (one evaluation, no solve).
    The cheapest thing that removes the diffusive step restriction.
    """
    h = float(dt)
    nh = _to_hat(nonlinear(u))
    uh = _to_hat(u)
    lam = symbol.astype(uh.dtype)
    return _from_hat((uh + h * nh) / (1.0 - h * lam))


def imex_cnab2_step(
    symbol: Array,
    nonlinear: Callable[[Array], Array],
    u: Array,
    dt: float,
    *,
    previous_nonlinear: Array | None = None,
) -> tuple[Array, Array]:
    r"""Crank-Nicolson (linear) + Adams-Bashforth 2 (nonlinear); second order.

    Returns ``(u_next, N(u_n))`` -- the second element is what the *next* call
    wants as ``previous_nonlinear``, which is the whole cost of a multistep
    method: it needs a memory. The first step, with no history, falls back to
    explicit Euler on the nonlinear term (the standard bootstrap); that costs
    one step of first-order accuracy and nothing thereafter.
    """
    h = float(dt)
    n_cur = nonlinear(u)
    if previous_nonlinear is None:
        explicit = n_cur
    else:
        explicit = 1.5 * n_cur - 0.5 * previous_nonlinear
    uh = _to_hat(u)
    lam = symbol.astype(uh.dtype)
    num = uh * (1.0 + 0.5 * h * lam) + h * _to_hat(explicit)
    return _from_hat(num / (1.0 - 0.5 * h * lam)), n_cur


def etdrk4_step(
    symbol: Array,
    nonlinear: Callable[[Array], Array],
    u: Array,
    dt: float,
    *,
    order: int = 20,
    squarings: int | None = None,
) -> Array:
    r"""One Cox-Matthews ETDRK4 step of ``u_t = L u + N(u)``.

    The linear part is advanced by ``exp(dt L)`` -- exactly, so its stiffness
    imposes **no** step restriction whatsoever -- and the nonlinear part by a
    fourth-order exponential Runge-Kutta scheme whose weights are

    .. math::

       \alpha = \varphi_1 - 3\varphi_2 + 4\varphi_3, \quad
       \beta = 2\varphi_2 - 4\varphi_3, \quad
       \gamma = -\varphi_2 + 4\varphi_3

    at ``z = dt L``. Written that way the coefficients are evaluated from the
    series and stay accurate for the small-``|z|`` modes where the equivalent
    closed forms cancel catastrophically; that cancellation is the reason the
    canonical implementation reaches for a contour integral instead.
    """
    h = float(dt)
    uh = _to_hat(u)
    lam = symbol.astype(uh.dtype)
    z = h * lam
    p_half = phi_diagonal(0.5 * z, 1, order=order, squarings=squarings)
    p_full = phi_diagonal(z, 3, order=order, squarings=squarings)
    e_half, e_full = p_half[0], p_full[0]
    q = 0.5 * h * p_half[1]
    alpha = h * (p_full[1] - 3.0 * p_full[2] + 4.0 * p_full[3])
    beta = h * (2.0 * p_full[2] - 4.0 * p_full[3])
    gamma = h * (-p_full[2] + 4.0 * p_full[3])

    n_u = _to_hat(nonlinear(u))
    a_hat = e_half * uh + q * n_u
    n_a = _to_hat(nonlinear(_from_hat(a_hat)))
    b_hat = e_half * uh + q * n_a
    n_b = _to_hat(nonlinear(_from_hat(b_hat)))
    c_hat = e_half * a_hat + q * (2.0 * n_b - n_u)
    n_c = _to_hat(nonlinear(_from_hat(c_hat)))
    return _from_hat(e_full * uh + alpha * n_u + beta * (n_a + n_b) + gamma * n_c)


# ---------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------


def stiff_rollout(
    step: Callable[[Array, float], Array],
    u0: Array,
    *,
    dt: float,
    n_steps: int,
) -> Array:
    """Apply ``step`` ``n_steps`` times; return ``(n_steps + 1, *u0.shape)``.

    Nothing is stop-gradiented between steps, so the whole trajectory is one
    differentiable expression: a loss on the final state trains the parameters
    inside the right-hand side through every step of the integration. That is
    what makes these integrators *layers* rather than a post-processing stage.
    """
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0, got {n_steps}")
    u = u0
    out = [u0]
    for _ in range(n_steps):
        u = step(u, float(dt))
        out.append(u)
    return jnp.stack(out, axis=0)


__all__ = [
    "ROS2_GAMMA",
    "closed_form_jacobian",
    "dense_jacobian",
    "etdrk4_step",
    "exponential_rosenbrock_step",
    "imex_cnab2_step",
    "imex_euler_step",
    "phi_diagonal",
    "phi_matrix",
    "rosenbrock_step",
    "stiff_rollout",
]
