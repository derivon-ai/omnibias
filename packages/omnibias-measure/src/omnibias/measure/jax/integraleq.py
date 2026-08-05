# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Fredholm & Volterra integral equations on the measure integral (jax).

Bit-identical twin of :mod:`omnibias.measure.torch.integraleq`, with the numpy
reference in :mod:`omnibias.measure._core.integraleq`. See the torch module for
the full derivation; in brief, an integral equation of the **second kind**

.. math::

    u(x) = f(x) + \lambda \int_\Omega K(x, t)\, u(t)\, d\mu(t)

becomes the dense system :math:`(I - \lambda K W) u = f` under Nystrom
discretisation, where :math:`W` is the diagonal of the measure's quadrature
weights -- so a :class:`~omnibias.measure._core.measure.Measure` is already
everything the discretisation needs.

Four solvers: :func:`nystrom_solve` (general Fredholm, ``numerical``),
:func:`volterra_solve` (causal, lower triangular, ``numerical`` second order),
:func:`neumann_series` (iterative, valid only inside
:math:`|\lambda| \rho(KW) < 1` and honest about it), and
:func:`degenerate_kernel_solve` (finite rank, **exact in the kernel**, the
analytic oracle). All differentiable in the kernel, the source and
:math:`\lambda`.

All are ``jit``-compatible: the only Python-level branching is on shapes, and
:func:`neumann_series` runs a fixed ``max_terms`` under ``jit`` rather than an
early exit. The two direct solvers additionally screen for a Fredholm alternative
(``check_conditioning``, on by default), which needs a concrete matrix -- under a
transform it finds a tracer and steps aside, so a transformed solve at an
eigenvalue returns a large finite vector rather than raising. Solve once eagerly
if that is a live risk.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from omnibias.measure._core.integraleq import (
    SINGULAR_RCOND,
    NeumannResult,
    solvability_margin,
)
from omnibias.measure._core.measure import Measure

#: A kernel ``K(x, t)`` called with query points ``(n, d)`` and quadrature nodes
#: ``(m, d)``, returning ``(n, m)``.
KernelFn = Callable[[Array, Array], Array]

#: A right-hand side ``f(x)`` mapping points ``(n, d)`` to values ``(n,)``.
SourceFn = Callable[[Array], Array]


def _nodes_weights(
    measure: Measure | None, nodes: Array | None, weights: Array | None
) -> tuple[Array, Array]:
    if nodes is None:
        if measure is None:
            raise ValueError("provide either `measure` or explicit `nodes`/`weights`")
        nodes = jnp.asarray(measure.nodes)
    if weights is None:
        if measure is None:
            raise ValueError("provide either `measure` or explicit `weights`")
        w = jnp.asarray(measure.weights)
    else:
        w = weights
    if w.shape[0] != nodes.shape[0]:
        raise ValueError(f"weights length {w.shape[0]} != n_nodes {nodes.shape[0]}")
    return nodes, w


def _operator(kernel: KernelFn, nodes: Array, weights: Array) -> Array:
    """The discretised operator ``K W``."""
    k = kernel(nodes, nodes)
    n = nodes.shape[0]
    if k.shape != (n, n):
        raise ValueError(
            f"kernel returned shape {k.shape}, expected {(n, n)} for "
            "(query points, quadrature nodes)"
        )
    return k * weights[None, :]


def _check_solvable(a: Array, lam: float | Array, what: str) -> None:
    """Raise before solving if ``a`` is at (or too near) a Fredholm alternative.

    The test reads scalars off the matrix, which only exist as numbers when the
    matrix does: under ``jit``, ``grad`` or ``vmap`` there is a tracer and nothing
    to read, so the check is skipped rather than crashing on the concretisation.
    A transformed solve therefore returns a large finite vector at a Fredholm
    alternative instead of raising -- run it once eagerly to find out.
    """
    if isinstance(a, jax.core.Tracer):
        return
    rcond = solvability_margin(np.asarray(jnp.linalg.svd(a, compute_uv=False)))
    if not math.isfinite(rcond) or rcond < SINGULAR_RCOND:
        raise ValueError(
            f"the {what} is singular at lam={float(lam):g} (solvability margin "
            f"{rcond:.3g} < {SINGULAR_RCOND:g}): 1/lam is an eigenvalue of "
            "the operator, so the integral equation has no unique solution there "
            "-- a Fredholm alternative. Solving anyway would return a large finite "
            "vector rather than failing"
        )


def _source_values(source: SourceFn | Array, nodes: Array) -> Array:
    values = source(nodes) if callable(source) else source
    if values.shape != (nodes.shape[0],):
        raise ValueError(
            f"the right-hand side has shape {values.shape}, expected "
            f"{(nodes.shape[0],)} (one value per quadrature node)"
        )
    return values


def nystrom_solve(
    kernel: KernelFn,
    source: SourceFn | Array,
    measure: Measure | None = None,
    *,
    lam: float | Array = 1.0,
    nodes: Array | None = None,
    weights: Array | None = None,
    check_conditioning: bool = True,
) -> Array:
    r"""Solve ``u = f + lam int K(x,t) u(t) dmu(t)`` by Nystrom discretisation (jax).

    Returns nodal values at the measure's nodes, differentiable in the kernel's
    parameters, the source, ``lam`` and ``weights``. Direct solve, so there is no
    convergence radius -- only the requirement that :math:`1/\lambda` is not an
    eigenvalue of the discretised operator. That case (a Fredholm alternative)
    raises; see the numpy reference for why the test is a condition number rather
    than exact singularity. It needs a concrete matrix, so it stands down under a
    ``jit`` / ``grad`` / ``vmap`` trace; ``check_conditioning=False`` skips it
    eagerly too.

    Honesty label: **numerical**, the quadrature error of ``measure``'s rule.
    """
    nd, w = _nodes_weights(measure, nodes, weights)
    kw = _operator(kernel, nd, w)
    f = _source_values(source, nd)
    lam_a = jnp.asarray(lam, dtype=kw.dtype)
    a = jnp.eye(nd.shape[0], dtype=kw.dtype) - lam_a * kw
    if check_conditioning:
        _check_solvable(a, lam_a, "Nystrom system")
    out: Array = jnp.linalg.solve(a, f)
    return out


def volterra_solve(
    kernel: KernelFn,
    source: SourceFn | Array,
    measure: Measure | None = None,
    *,
    lam: float | Array = 1.0,
    nodes: Array | None = None,
) -> Array:
    r"""Solve the causal ``u(x) = f(x) + lam int_a^x K(x,t) u(t) dt`` (jax).

    Causality makes the operator lower triangular, so the system is a forward
    substitution and is always uniquely solvable (a Volterra operator of the
    second kind has spectral radius zero, so no Fredholm alternative arises).

    Nodes must be 1-D and **sorted** -- that ordering is what causality means. The
    measure's weights are deliberately unused: a global rule's weights cannot be
    restricted to the prefix :math:`[a, x_i]`, so a cumulative trapezoid rule is
    built from the nodes instead.

    Honesty label: **numerical**, second order in the node spacing, and that order
    does not improve with a better measure.
    """
    if nodes is None:
        if measure is None:
            raise ValueError("provide either `measure` or explicit `nodes`")
        if measure.dim != 1:
            raise ValueError(
                "volterra_solve needs a 1-D measure (causality is an ordering of "
                f"the line), got dim={measure.dim}"
            )
        nodes = jnp.asarray(measure.nodes)
    if nodes.ndim != 2 or nodes.shape[1] != 1:
        raise ValueError(
            f"volterra_solve needs 1-D nodes of shape (n, 1), got {nodes.shape}"
        )
    x = nodes[:, 0]
    if x.shape[0] < 2:
        raise ValueError("volterra_solve needs at least two nodes")
    k = kernel(nodes, nodes)
    n = x.shape[0]
    if k.shape != (n, n):
        raise ValueError(
            f"kernel returned shape {k.shape}, expected {(n, n)} for "
            "(query points, quadrature nodes)"
        )
    f = _source_values(source, nodes)
    cum = cumulative_trapezoid_matrix(x)
    a = jnp.eye(n, dtype=k.dtype) - jnp.asarray(lam, dtype=k.dtype) * (k * cum)
    out: Array = jnp.linalg.solve(a, f)
    return out


def cumulative_trapezoid_matrix(x: Array) -> Array:
    r"""Weights ``C`` with ``(C @ v)_i = int_{x_0}^{x_i} v`` by the trapezoid rule.

    Lower triangular, which is what makes a Volterra system a forward
    substitution. Exact for a piecewise linear integrand, second order otherwise.
    """
    n = x.shape[0]
    h = jnp.diff(x)
    below = jnp.tril(jnp.ones((n, n - 1), dtype=x.dtype), k=-1)
    half = 0.5 * h[None, :] * below
    c = jnp.zeros((n, n), dtype=x.dtype)
    c = c.at[:, :-1].add(half)
    c = c.at[:, 1:].add(half)
    return c


def neumann_series(
    kernel: KernelFn,
    source: SourceFn | Array,
    measure: Measure | None = None,
    *,
    lam: float | Array = 1.0,
    max_terms: int = 64,
    tol: float = 1e-12,
    nodes: Array | None = None,
    weights: Array | None = None,
) -> tuple[Array, NeumannResult]:
    r"""Sum ``u = sum_k lam^k (K W)^k f``, reporting whether it converged (jax).

    Returns ``(solution, report)``. The array is the differentiable object; the
    :class:`~omnibias.measure._core.integraleq.NeumannResult` is a concrete,
    numpy-valued verdict carrying the estimated radius
    :math:`|\lambda| \rho(KW)`, the achieved residual and an earned ``converged``
    flag -- necessary because outside the radius the partial sums diverge while
    remaining perfectly finite arrays.

    Honesty label: **numerical**. Note the report forces concretisation, so under
    ``jit`` take only the array and check convergence outside the trace.
    """
    if max_terms < 1:
        raise ValueError(f"max_terms must be >= 1, got {max_terms}")
    nd, w = _nodes_weights(measure, nodes, weights)
    kw = _operator(kernel, nd, w)
    f = _source_values(source, nd)
    scaled = jnp.asarray(lam, dtype=kw.dtype) * kw

    total = f
    term = f
    n_terms = 1
    for _ in range(max_terms - 1):
        term = scaled @ term
        total = total + term
        n_terms += 1
        if float(jnp.max(jnp.abs(term))) <= tol * max(
            1.0, float(jnp.max(jnp.abs(total)))
        ):
            break
    radius = float(jnp.max(jnp.abs(jnp.linalg.eigvals(scaled))))
    residual = float(jnp.max(jnp.abs(total - f - scaled @ total)))
    scale = max(1.0, float(jnp.max(jnp.abs(total))))
    report = NeumannResult(
        solution=jnp.asarray(total).__array__(),
        converged=bool(radius < 1.0 and residual <= tol * scale * 1e3),
        n_terms=n_terms,
        residual=residual,
        spectral_radius=radius,
    )
    return total, report


def degenerate_kernel_solve(
    factors: list[tuple[SourceFn, SourceFn]],
    source: SourceFn | Array,
    measure: Measure | None = None,
    *,
    lam: float | Array = 1.0,
    nodes: Array | None = None,
    weights: Array | None = None,
    check_conditioning: bool = True,
) -> Array:
    r"""Exact finite-rank solve for ``K(x,t) = sum_r a_r(x) b_r(t)`` (jax).

    With :math:`c_r = \int b_r u \, d\mu` the solution is
    :math:`u = f + \lambda \sum_r c_r a_r`, and the coefficients satisfy an
    :math:`r \times r` system in the moments -- so a rank-``r`` kernel costs an
    ``r x r`` solve rather than ``n x n``.

    Honesty label: **exact in the kernel** -- :math:`K` is never discretised, only
    the scalar moments are quadrature, which is what makes this the oracle the
    other solvers are checked against.
    """
    if not factors:
        raise ValueError("degenerate_kernel_solve needs at least one (a, b) factor")
    nd, w = _nodes_weights(measure, nodes, weights)
    f = _source_values(source, nd)
    a_vals = jnp.stack([a(nd).reshape(-1) for a, _ in factors], axis=0)
    b_vals = jnp.stack([b(nd).reshape(-1) for _, b in factors], axis=0)
    rank = len(factors)
    if a_vals.shape != (rank, nd.shape[0]) or b_vals.shape != a_vals.shape:
        raise ValueError(
            f"each factor must return {(nd.shape[0],)} values per node; got "
            f"a{a_vals.shape}, b{b_vals.shape}"
        )
    bw = b_vals * w[None, :]
    m = bw @ a_vals.T
    rhs = bw @ f
    lam_a = jnp.asarray(lam, dtype=m.dtype)
    moment_system = jnp.eye(rank, dtype=m.dtype) - lam_a * m
    if check_conditioning:
        _check_solvable(moment_system, lam_a, f"rank-{rank} moment system")
    c = jnp.linalg.solve(moment_system, rhs)
    out: Array = f + lam_a * (c @ a_vals)
    return out


def fredholm_residual(
    u: Array,
    kernel: KernelFn,
    source: SourceFn | Array,
    measure: Measure | None = None,
    *,
    lam: float | Array = 1.0,
    nodes: Array | None = None,
    weights: Array | None = None,
) -> Array:
    r"""Pointwise residual ``u - f - lam int K u dmu`` (jax).

    The quantity a PINN drives to zero, and the way to check any solver above
    without a reference solution.
    """
    nd, w = _nodes_weights(measure, nodes, weights)
    kw = _operator(kernel, nd, w)
    f = _source_values(source, nd)
    if u.shape != (nd.shape[0],):
        raise ValueError(
            f"u has shape {u.shape}, expected {(nd.shape[0],)} "
            "(one value per quadrature node)"
        )
    return u - f - jnp.asarray(lam, dtype=kw.dtype) * (kw @ u)


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
