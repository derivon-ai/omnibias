# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Fredholm & Volterra integral equations on the measure integral (numpy reference).

An integral equation of the **second kind** puts the unknown both outside and
under the integral,

.. math::

    u(x) = f(x) + \lambda \int_\Omega K(x, t)\, u(t)\, d\mu(t),

which is the mirror image of a differential equation: a derivative reads the
solution *locally*, an integral operator reads all of it at once. The classical
route is Nystrom discretisation -- replace the integral by the quadrature the
measure already carries,

.. math::

    u(x_i) = f(x_i) + \lambda \sum_j w_j K(x_i, t_j) u(t_j),

so that the equation becomes the dense linear system :math:`(I - \lambda K W) u = f`
in the nodal values. omnibias gets the quadrature for free: a
:class:`~omnibias.measure._core.measure.Measure` *is* a set of nodes and weights,
so the Nystrom matrix is one outer product away from the measure integral used
everywhere else in this package.

Four solvers, and they differ in what they can honestly promise:

* :func:`nystrom_solve` -- the general Fredholm route. **numerical**: exact up to
  the quadrature error of the measure's rule, so on a smooth kernel a
  Gauss-Legendre measure converges spectrally and a trapezoid rule at
  :math:`O(h^2)`.
* :func:`volterra_solve` -- the causal variant, where :math:`K(x, t) = 0` for
  :math:`t > x` makes the matrix lower triangular and the system solvable by
  forward substitution in :math:`O(n^2)` rather than :math:`O(n^3)`.
  **numerical**, with the extra caveat that the running quadrature is a
  *cumulative* rule (the measure's global weights cannot be restricted to a
  prefix), so it is trapezoid-accurate regardless of the measure supplied.
* :func:`neumann_series` -- the iterative route
  :math:`u = \sum_k \lambda^k K^k f`, which converges only inside the radius
  :math:`|\lambda| \, \rho(KW) < 1`. **numerical**, and it *reports* divergence
  rather than returning a plausible wrong answer: the estimated radius and the
  achieved residual come back in the result, and the converged flag is earned.
* :func:`degenerate_kernel_solve` -- the finite-rank path
  :math:`K(x,t) = \sum_r a_r(x) b_r(t)`, which collapses the problem to an
  :math:`r \times r` system in the moments :math:`\int b_r u`. **exact** in the
  kernel (no discretisation of :math:`K` at all); only the moment integrals are
  quadrature. This is the analytic oracle the other three are checked against.

The honest cost note: every Fredholm route here is dense in the nodes -- forming
:math:`K W` is :math:`O(n^2)` in memory and the solve :math:`O(n^3)` -- which is
why the degenerate path matters whenever the kernel really is low rank.

Bit-identical torch twin in :mod:`omnibias.measure.torch.integraleq` and jax twin
in :mod:`omnibias.measure.jax.integraleq`, both differentiable in the kernel, the
right-hand side and :math:`\lambda`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from omnibias.measure._core.measure import Measure

#: A kernel ``K(x, t)`` called with query points ``(n, d)`` and quadrature nodes
#: ``(m, d)``, returning ``(n, m)``. The kernel broadcasts itself, which keeps it
#: a single readable expression in more than one dimension.
KernelFn = Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]

#: A right-hand side ``f(x)`` mapping points ``(n, d)`` to values ``(n,)``.
SourceFn = Callable[[NDArray[np.float64]], NDArray[np.float64]]


@dataclass(frozen=True)
class NeumannResult:
    """Outcome of a :func:`neumann_series` solve, including whether it worked.

    The point of returning this instead of a bare array is that a Neumann series
    outside its radius does not fail loudly -- it produces a finite array of
    garbage. :attr:`converged` is earned by the measured residual, and
    :attr:`spectral_radius` says how far outside the radius the problem sits.
    """

    #: Nodal values of the (attempted) solution.
    solution: NDArray[np.float64]
    #: Whether the achieved residual met ``tol``.
    converged: bool
    #: Terms actually summed.
    n_terms: int
    #: ``max |u - f - lam K W u|`` at the returned iterate.
    residual: float
    #: Estimated ``|lam| rho(K W)``: the series converges iff this is ``< 1``.
    spectral_radius: float

    def raise_if_diverged(self) -> NDArray[np.float64]:
        """The solution, or a ``ValueError`` naming the radius that was exceeded."""
        if not self.converged:
            raise ValueError(
                f"the Neumann series did not converge: |lam| rho(KW) = "
                f"{self.spectral_radius:.4g} (needs < 1), residual "
                f"{self.residual:.3e} after {self.n_terms} terms. Use "
                "nystrom_solve, which is a direct solve and has no radius"
            )
        return self.solution


def _nystrom_matrix(
    kernel: KernelFn, measure: Measure
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(K W, nodes)``: the discretised operator and the points it acts on."""
    nodes = np.asarray(measure.nodes, dtype=float)
    weights = np.asarray(measure.weights, dtype=float)
    k = np.asarray(kernel(nodes, nodes), dtype=float)
    n = nodes.shape[0]
    if k.shape != (n, n):
        raise ValueError(
            f"kernel returned shape {k.shape}, expected {(n, n)} for "
            "(query points, quadrature nodes)"
        )
    return k * weights[None, :], nodes


#: Floor on ``sigma_min / max(1, sigma_max)`` below which ``I - lam K W`` counts as
#: singular. Roughly ``sqrt(eps)``: past it the solve returns a huge finite vector
#: rather than failing, which is the quiet wrong answer this module exists to
#: avoid. A genuine Fredholm alternative lands here, and so does a problem close
#: enough to one that the answer would be meaningless anyway.
SINGULAR_RCOND = 1e-10


def solvability_margin(singular_values: NDArray[np.floating[Any]]) -> float:
    """How far ``I - lam K W`` is from singular, from its singular values.

    Deliberately *not* the reciprocal condition number, which is scale-invariant
    and so blind to the case that matters: a rank-1 moment system at an eigenvalue
    is the single entry ``1 - lam * m``, and *any* nonzero 1x1 matrix has
    condition number one, however tiny. The identity in ``I - lam K W`` supplies
    an absolute scale, so the denominator is floored at one -- a uniformly tiny
    matrix is then singular in the only sense the equation cares about.

    Returns a number in ``(0, 1]``; the solvers refuse below
    :data:`SINGULAR_RCOND`. Useful directly when sweeping ``lam`` to find where an
    equation stops being solvable, which is where the operator's spectrum is.
    """
    return float(singular_values.min()) / max(1.0, float(singular_values.max()))


def _check_solvable(a: NDArray[np.float64], lam: float, what: str) -> None:
    """Raise before solving if ``a`` is at (or too near) a Fredholm alternative."""
    rcond = solvability_margin(np.linalg.svd(a, compute_uv=False))
    if not np.isfinite(rcond) or rcond < SINGULAR_RCOND:
        raise ValueError(
            f"the {what} is singular at lam={lam:g} (solvability margin "
            f"{rcond:.3g} < {SINGULAR_RCOND:g}): 1/lam is an eigenvalue of the "
            "operator, so the integral equation has no unique solution there -- a "
            "Fredholm alternative. Solving anyway would return a large finite "
            "vector rather than failing"
        )


def _source_values(
    source: SourceFn | NDArray[np.float64], nodes: NDArray[np.float64]
) -> NDArray[np.float64]:
    """``f`` at the nodes, whether supplied as a callable or as nodal values."""
    values: NDArray[np.float64] = (
        np.asarray(source(nodes), dtype=float)
        if callable(source)
        else np.asarray(source, dtype=float)
    )
    if values.shape != (nodes.shape[0],):
        raise ValueError(
            f"the right-hand side has shape {values.shape}, expected "
            f"{(nodes.shape[0],)} (one value per quadrature node)"
        )
    return values


def nystrom_solve(
    kernel: KernelFn,
    source: SourceFn | NDArray[np.float64],
    measure: Measure,
    *,
    lam: float = 1.0,
    check_conditioning: bool = True,
) -> NDArray[np.float64]:
    r"""Solve ``u = f + lam int K(x,t) u(t) dmu(t)`` by Nystrom discretisation.

    Returns the nodal values ``u(x_i)`` at ``measure.nodes``. The discrete system
    is :math:`(I - \lambda K W) u = f`, solved directly, so unlike
    :func:`neumann_series` there is no convergence radius -- only the requirement
    that :math:`1/\lambda` is not an eigenvalue of the discretised operator.

    When it *is* an eigenvalue the equation has no unique solution (a Fredholm
    alternative) and this raises. The check is a condition number rather than a
    test for exact singularity, because in floating point the dangerous case is
    the near-miss: an exactly singular matrix makes the solve fail loudly, while
    one eigenvalue away by a rounding error returns a huge finite vector that
    looks like an answer. Pass ``check_conditioning=False`` to skip it.

    Honesty label: **numerical**. The only error is the quadrature error of
    ``measure``'s rule, so the accuracy is the caller's choice: a Gauss-Legendre
    measure on a smooth kernel converges spectrally.

    Cost is :math:`O(n^2)` memory and :math:`O(n^3)` time in the node count. Use
    :func:`degenerate_kernel_solve` when the kernel is finite rank.
    """
    kw, nodes = _nystrom_matrix(kernel, measure)
    f = _source_values(source, nodes)
    a = np.eye(nodes.shape[0]) - float(lam) * kw
    if check_conditioning:
        _check_solvable(a, float(lam), "Nystrom system")
    solution: NDArray[np.float64] = np.asarray(np.linalg.solve(a, f), dtype=float)
    return solution


def volterra_solve(
    kernel: KernelFn,
    source: SourceFn | NDArray[np.float64],
    measure: Measure,
    *,
    lam: float = 1.0,
) -> NDArray[np.float64]:
    r"""Solve the causal ``u(x) = f(x) + lam int_a^x K(x,t) u(t) dt``.

    Causality makes the discretised operator lower triangular, so the system is
    solved by forward substitution in :math:`O(n^2)`, and -- unlike the Fredholm
    case -- it is *always* uniquely solvable: the diagonal of :math:`I - \lambda K W`
    cannot vanish for small enough spacing, and a Volterra operator of the second
    kind has spectral radius zero, so no Fredholm alternative arises.

    The measure must be 1-D with **sorted** nodes, which is what "causal" refers
    to. Its weights are *not* used: a global rule's weights are coefficients for
    the whole interval rather than the measure of a neighbourhood of each node, so
    they cannot be restricted to the prefix :math:`[a, x_i]`. The running integral
    is a cumulative trapezoid rule built from the nodes instead.

    Honesty label: **numerical**, second order in the node spacing -- and, unlike
    :func:`nystrom_solve`, that order does not improve by handing it a better
    measure, for the reason just given.
    """
    nodes = np.asarray(measure.nodes, dtype=float)
    if measure.dim != 1:
        raise ValueError(
            f"volterra_solve needs a 1-D measure (causality is an ordering of the "
            f"line), got dim={measure.dim}"
        )
    x = nodes[:, 0]
    if x.size < 2:
        raise ValueError("volterra_solve needs at least two nodes")
    if np.any(np.diff(x) <= 0.0):
        raise ValueError(
            "volterra_solve needs strictly increasing nodes; sort the measure "
            "(causality is defined by that order)"
        )
    k = np.asarray(kernel(nodes, nodes), dtype=float)
    n = x.size
    if k.shape != (n, n):
        raise ValueError(
            f"kernel returned shape {k.shape}, expected {(n, n)} for "
            "(query points, quadrature nodes)"
        )
    f = _source_values(source, nodes)
    cum = cumulative_trapezoid_matrix(x)
    a = np.eye(n) - float(lam) * (k * cum)
    # Lower triangular by construction (``cum`` is), so this is forward
    # substitution; np.linalg.solve recognises the structure via LAPACK.
    solution: NDArray[np.float64] = np.asarray(np.linalg.solve(a, f), dtype=float)
    return solution


def cumulative_trapezoid_matrix(x: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""Weights ``C`` with ``(C @ v)_i = int_{x_0}^{x_i} v`` by the trapezoid rule.

    Row ``i`` holds the composite-trapezoid weights for the prefix
    :math:`[x_0, x_i]`, so the whole running integral is one matrix-vector
    product. Lower triangular, which is what makes a Volterra system a forward
    substitution. Exact for a piecewise linear integrand, second order otherwise.
    """
    xs = np.asarray(x, dtype=float).reshape(-1)
    n = xs.size
    c = np.zeros((n, n), dtype=float)
    h = np.diff(xs)
    for i in range(1, n):
        c[i, :i] = c[i - 1, :i]
        c[i, i - 1] += 0.5 * h[i - 1]
        c[i, i] += 0.5 * h[i - 1]
    return c


def neumann_series(
    kernel: KernelFn,
    source: SourceFn | NDArray[np.float64],
    measure: Measure,
    *,
    lam: float = 1.0,
    max_terms: int = 64,
    tol: float = 1e-12,
) -> NeumannResult:
    r"""Sum ``u = sum_k lam^k (K W)^k f``, reporting honestly whether it converged.

    The Neumann (successive-approximation) series is the iterative counterpart of
    :func:`nystrom_solve`, and it is the one route here that can silently fail:
    outside its radius of convergence

    .. math::

        |\lambda| \, \rho(K W) < 1

    the partial sums diverge, but every one of them is a perfectly finite array.
    So this returns a :class:`NeumannResult` rather than a bare array: the
    estimated radius, the achieved residual, and a ``converged`` flag earned by
    that residual against ``tol``. Call
    :meth:`NeumannResult.raise_if_diverged` to turn a failure into an exception.

    Honesty label: **numerical** -- the quadrature error of the measure, plus the
    truncation of the series at ``max_terms``.

    Useful when :math:`\lambda` is small (a few terms beat an :math:`O(n^3)`
    solve) or when only the leading Born-style corrections are wanted. Otherwise
    prefer :func:`nystrom_solve`, which has no radius.
    """
    if max_terms < 1:
        raise ValueError(f"max_terms must be >= 1, got {max_terms}")
    kw, nodes = _nystrom_matrix(kernel, measure)
    f = _source_values(source, nodes)
    scaled = float(lam) * kw
    radius = float(np.max(np.abs(np.linalg.eigvals(scaled))))

    total = f.copy()
    term = f.copy()
    n_terms = 1
    for _ in range(max_terms - 1):
        term = scaled @ term
        total = total + term
        n_terms += 1
        if float(np.max(np.abs(term))) <= tol * max(
            1.0, float(np.max(np.abs(total)))
        ):
            break
    residual = float(np.max(np.abs(total - f - scaled @ total)))
    scale = max(1.0, float(np.max(np.abs(total))))
    return NeumannResult(
        solution=total,
        converged=bool(radius < 1.0 and residual <= tol * scale * 1e3),
        n_terms=n_terms,
        residual=residual,
        spectral_radius=radius,
    )


def degenerate_kernel_solve(
    factors: list[tuple[SourceFn, SourceFn]],
    source: SourceFn | NDArray[np.float64],
    measure: Measure,
    *,
    lam: float = 1.0,
    check_conditioning: bool = True,
) -> NDArray[np.float64]:
    r"""Exact finite-rank solve for ``K(x,t) = sum_r a_r(x) b_r(t)``.

    A degenerate (separable) kernel makes the integral equation collapse. Writing
    :math:`c_r = \int b_r u \, d\mu`, the equation becomes
    :math:`u = f + \lambda \sum_r c_r a_r`, and multiplying by :math:`b_s` and
    integrating gives the :math:`r \times r` system

    .. math::

        c_s - \lambda \sum_r c_r \int b_s a_r \, d\mu = \int b_s f \, d\mu,

    so a rank-``r`` kernel costs an ``r x r`` solve rather than ``n x n`` -- and
    the solution is then evaluable at *any* point, not only at the nodes, because
    :math:`u = f + \lambda \sum_r c_r a_r` is a closed form.

    Honesty label: **exact in the kernel** -- :math:`K` is never discretised, only
    the scalar moments :math:`\int b_s a_r` and :math:`\int b_s f` are quadrature.
    That is what makes this the analytic oracle the other three solvers are
    checked against: for a separable kernel it is right to the accuracy of a
    handful of scalar integrals, where a Nystrom solve carries the quadrature
    error of an ``n x n`` operator.

    ``factors`` is the list of ``(a_r, b_r)`` pairs; both map points ``(n, d)`` to
    values ``(n,)``. Returns the nodal values at ``measure.nodes``. A Fredholm
    alternative of the ``r x r`` moment system raises, as in :func:`nystrom_solve`.
    """
    if not factors:
        raise ValueError("degenerate_kernel_solve needs at least one (a, b) factor")
    nodes = np.asarray(measure.nodes, dtype=float)
    weights = np.asarray(measure.weights, dtype=float)
    f = _source_values(source, nodes)
    rank = len(factors)
    a_vals = np.stack(
        [np.asarray(a(nodes), dtype=float).reshape(-1) for a, _ in factors], axis=0
    )
    b_vals = np.stack(
        [np.asarray(b(nodes), dtype=float).reshape(-1) for _, b in factors], axis=0
    )
    if a_vals.shape != (rank, nodes.shape[0]) or b_vals.shape != a_vals.shape:
        raise ValueError(
            f"each factor must return {(nodes.shape[0],)} values per node; got "
            f"a{a_vals.shape}, b{b_vals.shape}"
        )
    # M[s, r] = int b_s a_r dmu ; rhs[s] = int b_s f dmu
    m = (b_vals * weights[None, :]) @ a_vals.T
    rhs = (b_vals * weights[None, :]) @ f
    moment_system = np.eye(rank) - float(lam) * m
    if check_conditioning:
        _check_solvable(moment_system, float(lam), f"rank-{rank} moment system")
    c = np.linalg.solve(moment_system, rhs)
    solution: NDArray[np.float64] = f + float(lam) * (c @ a_vals)
    return solution


def fredholm_residual(
    u: NDArray[np.float64],
    kernel: KernelFn,
    source: SourceFn | NDArray[np.float64],
    measure: Measure,
    *,
    lam: float = 1.0,
) -> NDArray[np.float64]:
    r"""Pointwise residual ``u - f - lam int K u dmu`` of a candidate solution.

    The quantity a PINN drives to zero, and the way to check any of the solvers
    above without a reference solution. Returns one value per node.
    """
    kw, nodes = _nystrom_matrix(kernel, measure)
    f = _source_values(source, nodes)
    values = np.asarray(u, dtype=float).reshape(-1)
    if values.shape != (nodes.shape[0],):
        raise ValueError(
            f"u has shape {values.shape}, expected {(nodes.shape[0],)} "
            "(one value per quadrature node)"
        )
    residual: NDArray[np.float64] = values - f - float(lam) * (kw @ values)
    return residual


__all__ = [
    "KernelFn",
    "NeumannResult",
    "SINGULAR_RCOND",
    "SourceFn",
    "cumulative_trapezoid_matrix",
    "degenerate_kernel_solve",
    "fredholm_residual",
    "neumann_series",
    "nystrom_solve",
    "solvability_margin",
    "volterra_solve",
]
