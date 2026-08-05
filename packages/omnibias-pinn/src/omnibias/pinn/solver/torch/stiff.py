# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Stiff integrators as differentiable layers (torch).

A stiff problem is one where the fastest mode is far faster than anything you
care about, so an explicit method is stable only at a step size set by physics
you are not trying to resolve. :func:`~omnibias.pinn.solver.torch.integrators.rk4_step`
on the fourth-derivative term of Kuramoto-Sivashinsky needs
``dt ~ (dx)^4``; the solution itself evolves on ``O(1)`` timescales. That is a
three-order-of-magnitude tax for nothing.

The cure is to treat the stiff part *implicitly* or *exactly*, and every scheme
here does it through the same object -- the **phi functions**

.. math::

   \varphi_0(z) = e^z, \qquad
   \varphi_{k}(z) = \sum_{j \ge 0} \frac{z^{j}}{(j+k)!}
   = \frac{\varphi_{k-1}(z) - 1/(k-1)!}{z},

which are precisely the Taylor coefficients of the exponential, regrouped.
:func:`phi_diagonal` and :func:`phi_matrix` evaluate them **from the series**
(Horner, plus scaling-and-squaring), never from the differenced closed form.
That is not a stylistic choice: ``(e^z - 1)/z`` loses every significant digit as
``z -> 0``, which is exactly the regime a well-resolved mode sits in, and it is
why the reference ETDRK4 implementation of Kassam & Trefethen evaluates these
coefficients by contour integration in the complex plane. Summing the series is
the cancellation-free route to the same numbers, and it is the same
Taylor-coefficient machinery the rest of omnibias is built on.

What is here
------------
*Diagonal / spectral* -- for ``u_t = L u + N(u)`` with ``L`` a Fourier symbol:

* :func:`etdrk4_step` -- Cox-Matthews ETDRK4. The linear part is integrated
  **exactly** (no timestep restriction from it at all), the nonlinear part to
  fourth order.
* :func:`imex_euler_step`, :func:`imex_cnab2_step` -- implicit-explicit splits;
  cheaper per step, first / second order.

*Dense* -- for an ODE system ``u' = f(u)`` with a Jacobian:

* :func:`rosenbrock_step` -- the 2-stage, 2nd-order, **L-stable** ROS2. Linearly
  implicit: two linear solves, no Newton iteration, and no convergence test to
  fail.
* :func:`exponential_rosenbrock_step` -- ``u + h phi_1(hJ) f(u)``, exact for a
  linear autonomous problem however large the step.

Every step is a plain differentiable function of its inputs, so a rollout
(:func:`stiff_rollout`) backpropagates to whatever produced the right-hand side
-- which is the point of putting them here rather than in a classical ODE
library. :func:`closed_form_jacobian` supplies ``J`` for a neural RHS from the
exact multivariate jet, so a stiff step on a learned vector field need not
touch autodiff or a finite difference at all.

Honesty
-------
These are **numerical** integrators with genuine local truncation error
(``O(dt^3)`` for ROS2 and the exponential Euler, ``O(dt^5)`` for ETDRK4); only
the *linear* part of an ETD step and the phi functions themselves are exact.
Nothing here is a certified enclosure -- that is
:mod:`omnibias.core.verified.lohner` and :mod:`omnibias.dynamics`, which bound
the true flow rather than approximating it.

References
----------
Cox & Matthews, *Exponential time differencing for stiff systems*, JCP 176
(2002). Kassam & Trefethen, *Fourth-order time-stepping for stiff PDEs*, SIAM J.
Sci. Comput. 26 (2005). Hochbruck & Ostermann, *Exponential integrators*, Acta
Numerica 19 (2010). Verwer et al., *A second-order Rosenbrock method*, SIAM J.
Sci. Comput. 20 (1999).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from omnibias.torch.jet_mv import jet_gradient, mlp_jet_mv
from torch import Tensor

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
    z: Tensor, k_max: int, *, order: int = 20, squarings: int | None = None
) -> Tensor:
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
        ``torch.compile`` / ``jax.jit``.
    """
    if k_max < 0:
        raise ValueError(f"k_max must be >= 0, got {k_max}")
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    s = (
        _squarings_for(float(z.abs().max())) if squarings is None else int(squarings)
    )
    if s < 0:
        raise ValueError(f"squarings must be >= 0, got {s}")
    zs = z / float(2**s)
    fact = _factorials(order + k_max)
    phis = []
    for k in range(k_max + 1):
        acc = torch.full_like(zs, 1.0 / fact[order + k])
        for j in range(order - 1, -1, -1):
            acc = acc * zs + 1.0 / fact[j + k]
        phis.append(acc)
    for _ in range(s):
        phis = _double_diagonal(phis)
    return torch.stack(phis, dim=0)


def _double_diagonal(phis: list[Tensor]) -> list[Tensor]:
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
    a: Tensor, k_max: int, *, order: int = 20, squarings: int | None = None
) -> Tensor:
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
        _squarings_for(float(a.abs().sum(dim=-1).max()))
        if squarings is None
        else int(squarings)
    )
    if s < 0:
        raise ValueError(f"squarings must be >= 0, got {s}")
    scaled = a / float(2**s)
    eye = torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
    fact = _factorials(order + k_max)
    phis = []
    for k in range(k_max + 1):
        acc = eye / fact[order + k]
        for j in range(order - 1, -1, -1):
            acc = acc @ scaled + eye / fact[j + k]
        phis.append(acc)
    for _ in range(s):
        phis = _double_matrix(phis, eye)
    return torch.stack(phis, dim=0)


def _double_matrix(phis: list[Tensor], eye: Tensor) -> list[Tensor]:
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


def dense_jacobian(rhs: Callable[[Tensor], Tensor], u: Tensor) -> Tensor:
    """``(n, n)`` Jacobian ``df/du`` by forward-over-reverse autodiff.

    The general fallback: correct for any ``rhs``, and labelled autodiff rather
    than closed form because that is what it is. When the right-hand side is an
    omnibias layer stack, :func:`closed_form_jacobian` is exact and cheaper.
    """
    if u.ndim != 1:
        raise ValueError(f"dense_jacobian expects a flat state (n,), got {tuple(u.shape)}")
    out: Tensor = torch.func.jacrev(rhs)(u)
    return out


def closed_form_jacobian(
    layers: Sequence[tuple[Tensor, Tensor | None, object]], u: Tensor
) -> Tensor:
    """``(C, n)`` Jacobian of an omnibias MLP right-hand side, closed form.

    ``layers`` is the ``(W, b, spec)`` stack that
    :func:`~omnibias.torch.jet_mv.mlp_jet_mv` consumes. The order-1 jet holds
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
    out: Tensor = torch.transpose(grad, 0, 1)
    return out


# ---------------------------------------------------------------------
# Dense stiff steps
# ---------------------------------------------------------------------


def rosenbrock_step(
    rhs: Callable[[Tensor], Tensor],
    u: Tensor,
    dt: float,
    *,
    jacobian: Tensor | Callable[[Tensor], Tensor] | None = None,
    gamma: float = ROS2_GAMMA,
) -> Tensor:
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
    a callable ``u -> J``, or leave it ``None`` for
    :func:`dense_jacobian`.

    With ``gamma = 1 + 1/sqrt(2)`` the stability function vanishes at infinity,
    so an infinitely stiff decaying mode is annihilated in one step instead of
    ringing -- the property that separates a usable stiff solver from a merely
    stable one.
    """
    if u.ndim != 1:
        raise ValueError(f"rosenbrock_step expects a flat state (n,), got {tuple(u.shape)}")
    j = _resolve_jacobian(rhs, u, jacobian)
    h = float(dt)
    eye = torch.eye(u.shape[0], dtype=u.dtype, device=u.device)
    lhs = eye - (float(gamma) * h) * j
    lu, piv = torch.linalg.lu_factor(lhs)  # one factorisation, two solves
    k1 = torch.linalg.lu_solve(lu, piv, (h * rhs(u)).unsqueeze(-1)).squeeze(-1)
    b2 = h * rhs(u + k1) - 2.0 * k1
    k2 = torch.linalg.lu_solve(lu, piv, b2.unsqueeze(-1)).squeeze(-1)
    out: Tensor = u + 1.5 * k1 + 0.5 * k2
    return out


def exponential_rosenbrock_step(
    rhs: Callable[[Tensor], Tensor],
    u: Tensor,
    dt: float,
    *,
    jacobian: Tensor | Callable[[Tensor], Tensor] | None = None,
    order: int = 20,
    squarings: int | None = None,
) -> Tensor:
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
    rhs: Callable[[Tensor], Tensor],
    u: Tensor,
    jacobian: Tensor | Callable[[Tensor], Tensor] | None,
) -> Tensor:
    if jacobian is None:
        return dense_jacobian(rhs, u)
    if callable(jacobian):
        return jacobian(u)
    return jacobian


# ---------------------------------------------------------------------
# Diagonal (spectral) stiff steps
# ---------------------------------------------------------------------


def _to_hat(u: Tensor) -> Tensor:
    out: Tensor = torch.fft.fft(u)
    return out


def _from_hat(uh: Tensor) -> Tensor:
    out: Tensor = torch.fft.ifft(uh).real
    return out


def imex_euler_step(
    symbol: Tensor,
    nonlinear: Callable[[Tensor], Tensor],
    u: Tensor,
    dt: float,
) -> Tensor:
    r"""First-order IMEX: ``(1 - dt L) u_{n+1} = u_n + dt N(u_n)`` in Fourier space.

    The stiff linear part is implicit (unconditionally stable, and damped at
    every wavenumber), the nonlinear part explicit (one evaluation, no solve).
    The cheapest thing that removes the diffusive step restriction.
    """
    h = float(dt)
    nh = _to_hat(nonlinear(u))
    uh = _to_hat(u)
    lam = symbol.to(uh.dtype)
    return _from_hat((uh + h * nh) / (1.0 - h * lam))


def imex_cnab2_step(
    symbol: Tensor,
    nonlinear: Callable[[Tensor], Tensor],
    u: Tensor,
    dt: float,
    *,
    previous_nonlinear: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
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
    lam = symbol.to(uh.dtype)
    num = uh * (1.0 + 0.5 * h * lam) + h * _to_hat(explicit)
    return _from_hat(num / (1.0 - 0.5 * h * lam)), n_cur


def etdrk4_step(
    symbol: Tensor,
    nonlinear: Callable[[Tensor], Tensor],
    u: Tensor,
    dt: float,
    *,
    order: int = 20,
    squarings: int | None = None,
) -> Tensor:
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
    lam = symbol.to(uh.dtype)
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
    step: Callable[[Tensor, float], Tensor],
    u0: Tensor,
    *,
    dt: float,
    n_steps: int,
) -> Tensor:
    """Apply ``step`` ``n_steps`` times; return ``(n_steps + 1, *u0.shape)``.

    Nothing is detached between steps, so the whole trajectory is one
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
    return torch.stack(out, dim=0)


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
