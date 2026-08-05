# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Fredholm & Volterra integral equations on the measure integral (torch).

An integral equation of the **second kind** puts the unknown both outside and
under the integral,

.. math::

    u(x) = f(x) + \lambda \int_\Omega K(x, t)\, u(t)\, d\mu(t),

the mirror image of a differential equation: a derivative reads the solution
locally, an integral operator reads all of it at once. Nystrom discretisation
replaces the integral with the quadrature the measure already carries, turning
the equation into the dense system :math:`(I - \lambda K W) u = f` in the nodal
values -- and since a :class:`~omnibias.measure._core.measure.Measure` *is* nodes
and weights, that matrix is one outer product away from the measure integral used
throughout this package.

Four solvers with different promises:

* :func:`nystrom_solve` -- the general Fredholm route (``numerical``: the
  quadrature error of the measure's rule, so spectral for Gauss-Legendre on a
  smooth kernel);
* :func:`volterra_solve` -- the causal variant, lower triangular and always
  uniquely solvable (``numerical``, second order, set by the cumulative rule
  rather than by the measure);
* :func:`neumann_series` -- the iterative route, valid only for
  :math:`|\lambda| \rho(KW) < 1`, which **reports divergence** instead of
  returning plausible garbage (``numerical``);
* :func:`degenerate_kernel_solve` -- the finite-rank path, **exact in the
  kernel** (only scalar moments are quadrature), which doubles as the analytic
  oracle for the other three.

Everything here is differentiable in the **kernel**, the **right-hand side** and
:math:`\lambda`, so an integral equation composes with a network on either side:
a learned kernel fitted to data, or a learned source. The weights may also be
passed as tensors, making the quadrature itself learnable.

The cost is honest and worth stating: the Fredholm routes are dense in the nodes,
:math:`O(n^2)` memory and :math:`O(n^3)` per solve, and the backward pass through
:func:`torch.linalg.solve` costs another solve. That is exactly why the
degenerate path exists.

Bit-identical numpy reference in :mod:`omnibias.measure._core.integraleq` and jax
twin in :mod:`omnibias.measure.jax.integraleq`.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from omnibias.measure._core.integraleq import (
    SINGULAR_RCOND,
    NeumannResult,
    solvability_margin,
)
from omnibias.measure._core.measure import Measure
from torch import Tensor

#: A kernel ``K(x, t)`` called with query points ``(n, d)`` and quadrature nodes
#: ``(m, d)``, returning ``(n, m)``. Broadcasts itself, so it stays one readable
#: expression in more than one dimension -- and may carry network parameters.
KernelFn = Callable[[Tensor, Tensor], Tensor]

#: A right-hand side ``f(x)`` mapping points ``(n, d)`` to values ``(n,)``.
SourceFn = Callable[[Tensor], Tensor]


def _nodes_weights(
    measure: Measure | None,
    nodes: Tensor | None,
    weights: Tensor | None,
    *,
    dtype: torch.dtype | None,
    device: torch.device | None,
) -> tuple[Tensor, Tensor]:
    dt = dtype if dtype is not None else torch.get_default_dtype()
    if nodes is None:
        if measure is None:
            raise ValueError("provide either `measure` or explicit `nodes`/`weights`")
        nodes = torch.as_tensor(measure.nodes, dtype=dt, device=device)
    if weights is None:
        if measure is None:
            raise ValueError("provide either `measure` or explicit `weights`")
        w = torch.as_tensor(measure.weights, dtype=dt, device=device)
    else:
        w = weights
    if w.shape[0] != nodes.shape[0]:
        raise ValueError(f"weights length {w.shape[0]} != n_nodes {nodes.shape[0]}")
    return nodes, w


def _operator(kernel: KernelFn, nodes: Tensor, weights: Tensor) -> Tensor:
    """The discretised operator ``K W``."""
    k = kernel(nodes, nodes)
    n = nodes.shape[0]
    if k.shape != (n, n):
        raise ValueError(
            f"kernel returned shape {tuple(k.shape)}, expected {(n, n)} for "
            "(query points, quadrature nodes)"
        )
    return k * weights.unsqueeze(0)


def _check_solvable(a: Tensor, lam: float | Tensor, what: str) -> None:
    """Raise before solving if ``a`` is at (or too near) a Fredholm alternative.

    Reads a scalar off the matrix, which forces a synchronisation, so callers
    inside a compiled region pass ``check_conditioning=False``. Without the check
    the solve returns a large finite vector instead of failing.
    """
    with torch.no_grad():
        rcond = solvability_margin(torch.linalg.svdvals(a).detach().cpu().numpy())
    if not math.isfinite(rcond) or rcond < SINGULAR_RCOND:
        raise ValueError(
            f"the {what} is singular at lam={float(lam):g} (solvability margin "
            f"{rcond:.3g} < {SINGULAR_RCOND:g}): 1/lam is an eigenvalue of "
            "the operator, so the integral equation has no unique solution there "
            "-- a Fredholm alternative. Solving anyway would return a large finite "
            "vector rather than failing"
        )


def _source_values(source: SourceFn | Tensor, nodes: Tensor) -> Tensor:
    values = source(nodes) if callable(source) else source
    if values.shape != (nodes.shape[0],):
        raise ValueError(
            f"the right-hand side has shape {tuple(values.shape)}, expected "
            f"{(nodes.shape[0],)} (one value per quadrature node)"
        )
    return values


def nystrom_solve(
    kernel: KernelFn,
    source: SourceFn | Tensor,
    measure: Measure | None = None,
    *,
    lam: float | Tensor = 1.0,
    nodes: Tensor | None = None,
    weights: Tensor | None = None,
    check_conditioning: bool = True,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""Solve ``u = f + lam int K(x,t) u(t) dmu(t)`` by Nystrom discretisation (torch).

    Returns the nodal values at the measure's nodes, differentiable in the
    kernel's parameters, the source, ``lam`` and ``weights``. The discrete system
    :math:`(I - \lambda K W) u = f` is solved directly, so unlike
    :func:`neumann_series` there is no convergence radius -- only the requirement
    that :math:`1/\lambda` is not an eigenvalue of the discretised operator. That
    case (a Fredholm alternative) raises; see the numpy reference for why the test
    is a condition number rather than exact singularity. It reads a scalar off the
    matrix, so pass ``check_conditioning=False`` inside a compiled region.

    Honesty label: **numerical**, the quadrature error of ``measure``'s rule.
    """
    nd, w = _nodes_weights(measure, nodes, weights, dtype=dtype, device=device)
    kw = _operator(kernel, nd, w)
    f = _source_values(source, nd)
    lam_t = torch.as_tensor(lam, dtype=kw.dtype, device=kw.device)
    a = torch.eye(nd.shape[0], dtype=kw.dtype, device=kw.device) - lam_t * kw
    if check_conditioning:
        _check_solvable(a, lam_t, "Nystrom system")
    out: Tensor = torch.linalg.solve(a, f)
    return out


def volterra_solve(
    kernel: KernelFn,
    source: SourceFn | Tensor,
    measure: Measure | None = None,
    *,
    lam: float | Tensor = 1.0,
    nodes: Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""Solve the causal ``u(x) = f(x) + lam int_a^x K(x,t) u(t) dt`` (torch).

    Causality makes the discretised operator lower triangular, so the system is a
    forward substitution and is *always* uniquely solvable: a Volterra operator of
    the second kind has spectral radius zero, so no Fredholm alternative arises.

    The nodes must be 1-D and **sorted** -- that ordering is what "causal" means.
    The measure's weights are deliberately unused: a global rule's weights are
    coefficients for the whole interval, not the measure of a neighbourhood of
    each node, so they cannot be restricted to the prefix :math:`[a, x_i]`. A
    cumulative trapezoid rule is built from the nodes instead.

    Honesty label: **numerical**, second order in the node spacing -- and that
    order does not improve by supplying a better measure, for the reason above.
    """
    dt = dtype if dtype is not None else torch.get_default_dtype()
    if nodes is None:
        if measure is None:
            raise ValueError("provide either `measure` or explicit `nodes`")
        if measure.dim != 1:
            raise ValueError(
                "volterra_solve needs a 1-D measure (causality is an ordering of "
                f"the line), got dim={measure.dim}"
            )
        nodes = torch.as_tensor(measure.nodes, dtype=dt, device=device)
    if nodes.ndim != 2 or nodes.shape[1] != 1:
        raise ValueError(
            f"volterra_solve needs 1-D nodes of shape (n, 1), got {tuple(nodes.shape)}"
        )
    x = nodes[:, 0]
    if x.shape[0] < 2:
        raise ValueError("volterra_solve needs at least two nodes")
    if bool(torch.any(torch.diff(x) <= 0)):
        raise ValueError(
            "volterra_solve needs strictly increasing nodes; sort the measure "
            "(causality is defined by that order)"
        )
    k = kernel(nodes, nodes)
    n = x.shape[0]
    if k.shape != (n, n):
        raise ValueError(
            f"kernel returned shape {tuple(k.shape)}, expected {(n, n)} for "
            "(query points, quadrature nodes)"
        )
    f = _source_values(source, nodes)
    cum = cumulative_trapezoid_matrix(x)
    lam_t = torch.as_tensor(lam, dtype=k.dtype, device=k.device)
    a = torch.eye(n, dtype=k.dtype, device=k.device) - lam_t * (k * cum)
    out: Tensor = torch.linalg.solve(a, f)
    return out


def cumulative_trapezoid_matrix(x: Tensor) -> Tensor:
    r"""Weights ``C`` with ``(C @ v)_i = int_{x_0}^{x_i} v`` by the trapezoid rule.

    Lower triangular, which is what makes a Volterra system a forward
    substitution. Exact for a piecewise linear integrand, second order otherwise.
    Differentiable in the node positions.
    """
    n = x.shape[0]
    h = torch.diff(x)
    # Panel k contributes h_k/2 to nodes k and k+1, for every row at or past k+1.
    # ``below`` selects the panels wholly left of each row.
    below = torch.tril(
        torch.ones(n, n - 1, dtype=x.dtype, device=x.device), diagonal=-1
    )
    half = 0.5 * h.unsqueeze(0) * below  # (n, n-1): panel contributions per row
    c = torch.zeros(n, n, dtype=x.dtype, device=x.device)
    c[:, :-1] = c[:, :-1] + half
    c[:, 1:] = c[:, 1:] + half
    return c


def neumann_series(
    kernel: KernelFn,
    source: SourceFn | Tensor,
    measure: Measure | None = None,
    *,
    lam: float | Tensor = 1.0,
    max_terms: int = 64,
    tol: float = 1e-12,
    nodes: Tensor | None = None,
    weights: Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> tuple[Tensor, NeumannResult]:
    r"""Sum ``u = sum_k lam^k (K W)^k f``, reporting whether it converged (torch).

    Returns ``(solution, report)``. The tensor is the differentiable object; the
    :class:`~omnibias.measure._core.integraleq.NeumannResult` alongside it is a
    detached, numpy-valued verdict carrying the estimated radius
    :math:`|\lambda| \rho(KW)`, the achieved residual and an earned ``converged``
    flag. The split is deliberate: outside the radius the partial sums diverge but
    every one of them is a perfectly finite tensor, so a bare return value would
    be a plausible wrong answer. Call
    :meth:`~omnibias.measure._core.integraleq.NeumannResult.raise_if_diverged` on
    the report to turn that into an exception.

    Honesty label: **numerical** -- the measure's quadrature error plus the
    truncation at ``max_terms``. Worth using when :math:`\lambda` is small, or
    when only the leading Born-style corrections are wanted; otherwise prefer
    :func:`nystrom_solve`, which has no radius.
    """
    if max_terms < 1:
        raise ValueError(f"max_terms must be >= 1, got {max_terms}")
    nd, w = _nodes_weights(measure, nodes, weights, dtype=dtype, device=device)
    kw = _operator(kernel, nd, w)
    f = _source_values(source, nd)
    lam_t = torch.as_tensor(lam, dtype=kw.dtype, device=kw.device)
    scaled = lam_t * kw

    total = f
    term = f
    n_terms = 1
    for _ in range(max_terms - 1):
        term = scaled @ term
        total = total + term
        n_terms += 1
        if float(term.abs().max()) <= tol * max(1.0, float(total.abs().max())):
            break
    with torch.no_grad():
        radius = float(torch.linalg.eigvals(scaled).abs().max())
        residual = float((total - f - scaled @ total).abs().max())
        scale = max(1.0, float(total.abs().max()))
    report = NeumannResult(
        solution=total.detach().cpu().numpy(),
        converged=bool(radius < 1.0 and residual <= tol * scale * 1e3),
        n_terms=n_terms,
        residual=residual,
        spectral_radius=radius,
    )
    return total, report


def degenerate_kernel_solve(
    factors: list[tuple[SourceFn, SourceFn]],
    source: SourceFn | Tensor,
    measure: Measure | None = None,
    *,
    lam: float | Tensor = 1.0,
    nodes: Tensor | None = None,
    weights: Tensor | None = None,
    check_conditioning: bool = True,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""Exact finite-rank solve for ``K(x,t) = sum_r a_r(x) b_r(t)`` (torch).

    A separable kernel collapses the equation. With
    :math:`c_r = \int b_r u \, d\mu` the solution is
    :math:`u = f + \lambda \sum_r c_r a_r`, and the coefficients satisfy the
    :math:`r \times r` system
    :math:`c_s - \lambda \sum_r c_r \int b_s a_r = \int b_s f`. So a rank-``r``
    kernel costs an ``r x r`` solve rather than ``n x n``.

    Honesty label: **exact in the kernel** -- :math:`K` is never discretised, only
    the scalar moments are quadrature. That is what makes it the analytic oracle
    the other solvers are checked against, and why it is the route to reach for
    when the kernel really is low rank.
    """
    if not factors:
        raise ValueError("degenerate_kernel_solve needs at least one (a, b) factor")
    nd, w = _nodes_weights(measure, nodes, weights, dtype=dtype, device=device)
    f = _source_values(source, nd)
    a_vals = torch.stack([a(nd).reshape(-1) for a, _ in factors], dim=0)
    b_vals = torch.stack([b(nd).reshape(-1) for _, b in factors], dim=0)
    rank = len(factors)
    if a_vals.shape != (rank, nd.shape[0]) or b_vals.shape != a_vals.shape:
        raise ValueError(
            f"each factor must return {(nd.shape[0],)} values per node; got "
            f"a{tuple(a_vals.shape)}, b{tuple(b_vals.shape)}"
        )
    bw = b_vals * w.unsqueeze(0)
    m = bw @ a_vals.T
    rhs = bw @ f
    lam_t = torch.as_tensor(lam, dtype=m.dtype, device=m.device)
    eye = torch.eye(rank, dtype=m.dtype, device=m.device)
    moment_system = eye - lam_t * m
    if check_conditioning:
        _check_solvable(moment_system, lam_t, f"rank-{rank} moment system")
    c = torch.linalg.solve(moment_system, rhs)
    out: Tensor = f + lam_t * (c @ a_vals)
    return out


def fredholm_residual(
    u: Tensor,
    kernel: KernelFn,
    source: SourceFn | Tensor,
    measure: Measure | None = None,
    *,
    lam: float | Tensor = 1.0,
    nodes: Tensor | None = None,
    weights: Tensor | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""Pointwise residual ``u - f - lam int K u dmu`` (torch).

    The quantity a PINN drives to zero, and the way to check any solver above
    without a reference solution. Differentiable in ``u``, so it composes
    directly with a network that predicts the solution.
    """
    nd, w = _nodes_weights(measure, nodes, weights, dtype=dtype, device=device)
    kw = _operator(kernel, nd, w)
    f = _source_values(source, nd)
    if u.shape != (nd.shape[0],):
        raise ValueError(
            f"u has shape {tuple(u.shape)}, expected {(nd.shape[0],)} "
            "(one value per quadrature node)"
        )
    lam_t = torch.as_tensor(lam, dtype=kw.dtype, device=kw.device)
    out: Tensor = u - f - lam_t * (kw @ u)
    return out


__all__ = [
    "KernelFn",
    "SourceFn",
    "cumulative_trapezoid_matrix",
    "degenerate_kernel_solve",
    "fredholm_residual",
    "neumann_series",
    "nystrom_solve",
    "volterra_solve",
]
