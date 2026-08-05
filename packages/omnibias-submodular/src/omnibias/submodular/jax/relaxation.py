# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Differentiable continuous-greedy relaxation (JAX) on the multilinear extension.

Bit-identical twin of :mod:`omnibias.submodular.torch.relaxation` (float64). Continuous
greedy is Frank-Wolfe on the multilinear extension ``F``: from ``p_0 = 0``, at each of
``T`` steps take the (soft) matroid linear-oracle basis and move ``p <- p + (1/T) y``.
The exact gradient is ``dF/dp_i = F(p|p_i=1) - F(p|p_i=0)`` (``F`` is affine in each
coordinate), and the differentiable soft oracle ``y = sigmoid(beta (g - tau))`` is
unrolled for backprop, so a model that predicts the coverage data trains *through* the
optimizer. The returned fractional ``p*`` is rounded by
:func:`omnibias.submodular.pipage_round` and the gap certified by
:func:`omnibias.submodular.certify_submodular_gap`.

Terminology: the multilinear extension relaxes ``{0,1}^n -> [0,1]^n`` and the Frank-Wolfe
LP oracle ``sigmoid(beta (g - tau))``, ``beta -> inf``, hardens onto a ``0/1``
matroid-basis vertex -- the **feasibility** / temperature sense of "collapse" (a soft
indicator becoming a 0/1 step). This is **distinct from the founding bias collapse** (the
multi-bias ``delta -> 0`` limit to the closed-form derivative ``sigma^(K-1)``; see
``docs/theory.md`` and :mod:`omnibias.jax`).
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from jax import Array
from omnibias.submodular._core.multilinear import budget_multilinear_schedule
from omnibias.submodular.functions import BudgetAdditive, Coverage, FacilityLocation
from omnibias.submodular.matroid import Matroid
from omnibias.submodular.problem import ContinuousGreedySchedule, SubmodularProblem

MultilinearFn = Callable[[Array], Array]


def _sigmoid(z: Array) -> Array:
    return 1.0 / (1.0 + jnp.exp(-z))


def coverage_multilinear(p: object, membership: object, weights: object) -> Array:
    r"""The coverage multilinear extension ``F(p) = sum_u w_u (1 - prod_i (1 - C_ui p_i))``.

    Accepts a point ``(n,)`` (returns a scalar) or a batch ``(B, n)`` (returns ``(B,)``).
    Differentiable in ``membership`` / ``weights`` for training *through* the relaxation.
    """
    c = jnp.asarray(membership, dtype=jnp.float64)
    w = jnp.asarray(weights, dtype=jnp.float64)
    pv = jnp.asarray(p, dtype=jnp.float64)
    if pv.ndim == 1:
        uncovered = jnp.prod(1.0 - c * pv[None, :], axis=1)
        return jnp.sum(w * (1.0 - uncovered))
    a = 1.0 - c[None, :, :] * pv[:, None, :]
    uncovered_b = jnp.prod(a, axis=2)
    return jnp.sum(w[None, :] * (1.0 - uncovered_b), axis=1)


def _soft_basis(g: Array, groups: list[Array], caps: list[int], beta: float) -> Array:
    n = int(g.shape[0])
    y = jnp.zeros(n, dtype=jnp.float64)
    for group, cap_raw in zip(groups, caps, strict=True):
        size = int(group.shape[0])
        cap = min(int(cap_raw), size)
        if cap <= 0:
            continue
        if cap >= size:
            y = y.at[group].set(1.0)
            continue
        gg = g[group]
        sorted_desc = jnp.sort(gg)[::-1]
        tau = 0.5 * (sorted_desc[cap - 1] + sorted_desc[cap])
        y = y.at[group].set(_sigmoid(beta * (gg - tau)))
    return y


def continuous_greedy(
    multilinear_fn: MultilinearFn,
    matroid: Matroid,
    schedule: ContinuousGreedySchedule | None = None,
) -> Array:
    r"""Differentiable continuous greedy for a backend ``multilinear_fn`` -> ``p* in [0,1]^n``.

    ``multilinear_fn`` must accept a batch ``(n, n)`` and return ``(n,)`` (used for the
    exact difference gradient). The soft oracle uses ``schedule.beta``; a larger ``beta``
    hardens ``p*`` toward the exact-oracle continuous-greedy point.
    """
    sched = schedule or ContinuousGreedySchedule()
    n = matroid.n
    groups = [jnp.asarray(g) for g in matroid.groups()]
    caps = matroid.caps()
    eye = jnp.eye(n, dtype=jnp.float64)
    p = jnp.zeros(n, dtype=jnp.float64)
    inv = 1.0 / float(sched.steps)
    for _ in range(sched.steps):
        hi = p[None, :] * (1.0 - eye) + eye
        lo = p[None, :] * (1.0 - eye)
        grad = multilinear_fn(hi) - multilinear_fn(lo)
        y = _soft_basis(grad, groups, caps, sched.beta)
        p = p + inv * y
    return p


def coverage_relaxation(
    membership: object,
    weights: object,
    matroid: Matroid,
    schedule: ContinuousGreedySchedule | None = None,
) -> Array:
    r"""Continuous greedy for a coverage instance given (possibly tensor) ``membership`` / ``weights``.

    Pass jax arrays to differentiate the fractional ``p*`` through the coverage data.
    """
    c = jnp.asarray(membership, dtype=jnp.float64)
    w = jnp.asarray(weights, dtype=jnp.float64)

    def fn(p: Array) -> Array:
        return coverage_multilinear(p, c, w)

    return continuous_greedy(fn, matroid, schedule)


def facility_multilinear(p: object, gains: object, weights: object) -> Array:
    r"""The facility-location multilinear extension via the order-statistic expectation.

    For client ``j`` with facility gains sorted descending ``v_1 >= v_2 >= ...`` (a constant
    permutation of the data) and matched probabilities ``q_k = p`` of the sorted facility,
    ``E[max] = sum_k v_k q_k prod_{l < k} (1 - q_l)`` -- the survival product is an exclusive
    cumulative product, so ``F`` is differentiable in ``p`` / ``weights``. Accepts a point
    ``(n,)`` (scalar out) or a batch ``(B, n)`` (``(B,)`` out).
    """
    g = jnp.asarray(gains, dtype=jnp.float64)  # (m, n)
    w = jnp.asarray(weights, dtype=jnp.float64)  # (m,)
    m = int(g.shape[0])
    order = jnp.asarray(np.argsort(-np.asarray(gains, dtype=float), axis=1), dtype=jnp.int64)
    v_sorted = jnp.take_along_axis(g, order, axis=1)  # (m, n) descending gains
    pv = jnp.asarray(p, dtype=jnp.float64)
    if pv.ndim == 1:
        q = pv[order]  # (m, n) probabilities in sorted-gain order
        incl = jnp.cumprod(1.0 - q, axis=1)
        surv = jnp.concatenate([jnp.ones((m, 1), dtype=jnp.float64), incl[:, :-1]], axis=1)
        emax = jnp.sum(v_sorted * q * surv, axis=1)  # (m,)
        return jnp.sum(w * emax)
    q = pv[:, order]  # (B, m, n)
    incl = jnp.cumprod(1.0 - q, axis=2)
    surv = jnp.concatenate([jnp.ones((pv.shape[0], m, 1), dtype=jnp.float64), incl[:, :, :-1]], axis=2)
    emax = jnp.sum(v_sorted[None, :, :] * q * surv, axis=2)  # (B, m)
    return jnp.sum(w[None, :] * emax, axis=1)


def _budget_dp(pv: Array, support: Array, src: Array, capped: Array, zero_index: int) -> Array:
    r"""Differentiable Poisson-binomial convolution driven by a constant schedule.

    Invalid parents (``src < 0``) are zeroed by a multiplicative mask (not a ``where``) so
    the arithmetic is identical to the torch twin.
    """
    length = int(support.shape[0])
    n = int(src.shape[0])
    init = (jnp.arange(length) == zero_index).astype(jnp.float64)
    if pv.ndim == 1:
        prob = init
        for i in range(n):
            si = src[i]
            mask = (si >= 0).astype(jnp.float64)
            gathered = prob[jnp.clip(si, 0, length - 1)] * mask
            prob = prob * (1.0 - pv[i]) + gathered * pv[i]
        return jnp.sum(prob * capped)
    batch = int(pv.shape[0])
    prob = jnp.tile(init, (batch, 1))
    for i in range(n):
        si = src[i]
        idx = jnp.clip(si, 0, length - 1)
        mask = (si >= 0).astype(jnp.float64)
        gathered = prob[:, idx] * mask[None, :]
        pi = pv[:, i][:, None]
        prob = prob * (1.0 - pi) + gathered * pi
    return jnp.sum(prob * capped[None, :], axis=1)


def budget_multilinear(p: object, ground: object, budget: object) -> Array:
    r"""The budget-additive multilinear extension ``F(p) = E[min(sum_i a_i x_i, budget)]``.

    Uses the exact Poisson-binomial convolution over the (constant) subset-sum support from
    :func:`~omnibias.submodular._core.multilinear.budget_multilinear_schedule`, driven
    differentiably in ``p`` (and ``budget`` via the ``min``). Accepts a point ``(n,)`` or a
    batch ``(B, n)``. Constant-support: the support structure is fixed data, not traced.
    """
    support_np, src_np, zero_index = budget_multilinear_schedule(ground)
    support = jnp.asarray(support_np, dtype=jnp.float64)
    src = jnp.asarray(src_np, dtype=jnp.int64)
    capped = jnp.minimum(support, jnp.asarray(budget, dtype=jnp.float64))
    return _budget_dp(jnp.asarray(p, dtype=jnp.float64), support, src, capped, zero_index)


def graphcut_multilinear(p: object, weights: object) -> Array:
    r"""The graph-cut multilinear extension ``F(p) = deg . p - p^T W p`` (differentiable twin).

    ``weights`` is a (possibly tensor) adjacency matrix, symmetrized with its diagonal zeroed
    -- identically in both backends, so the twin stays bit-identical and differentiable in
    ``weights`` / ``p``. Accepts a point ``(n,)`` (scalar out) or a batch ``(B, n)`` (``(B,)``).
    Non-monotone: this is the objective for :func:`~omnibias.submodular.measured_continuous_greedy`,
    not the monotone continuous greedy.
    """
    w = jnp.asarray(weights, dtype=jnp.float64)
    w = 0.5 * (w + w.T)
    w = w - jnp.diag(jnp.diagonal(w))
    deg = jnp.sum(w, axis=1)
    pv = jnp.asarray(p, dtype=jnp.float64)
    if pv.ndim == 1:
        return pv @ deg - pv @ (w @ pv)
    lin = pv @ deg
    quad = jnp.sum((pv @ w) * pv, axis=1)
    return lin - quad


def facility_relaxation(
    gains: object,
    weights: object,
    matroid: Matroid,
    schedule: ContinuousGreedySchedule | None = None,
) -> Array:
    r"""Continuous greedy for a facility-location instance -> fractional ``p* in [0, 1]^n``."""
    g = jnp.asarray(gains, dtype=jnp.float64)
    w = jnp.asarray(weights, dtype=jnp.float64)

    def fn(p: Array) -> Array:
        return facility_multilinear(p, g, w)

    return continuous_greedy(fn, matroid, schedule)


def budget_relaxation(
    ground: object,
    budget: object,
    matroid: Matroid,
    schedule: ContinuousGreedySchedule | None = None,
) -> Array:
    r"""Continuous greedy for a budget-additive instance -> fractional ``p* in [0, 1]^n``."""
    support_np, src_np, zero_index = budget_multilinear_schedule(ground)
    support = jnp.asarray(support_np, dtype=jnp.float64)
    src = jnp.asarray(src_np, dtype=jnp.int64)
    capped = jnp.minimum(support, jnp.asarray(budget, dtype=jnp.float64))

    def fn(p: Array) -> Array:
        return _budget_dp(p, support, src, capped, zero_index)

    return continuous_greedy(fn, matroid, schedule)


def submodular_relaxation(
    problem: SubmodularProblem, schedule: ContinuousGreedySchedule | None = None
) -> Array:
    r"""Differentiable continuous greedy for ``problem`` -> fractional ``p* in [0, 1]^n``.

    Supports the three closed-form multilinear families
    (:class:`~omnibias.submodular.functions.Coverage`,
    :class:`~omnibias.submodular.functions.FacilityLocation`,
    :class:`~omnibias.submodular.functions.BudgetAdditive`); greedy-path functions (no
    closed-form ``F``) run the numpy certified pipeline via :func:`omnibias.submodular.maximize`.
    """
    fn = problem.function
    if isinstance(fn, Coverage):
        return coverage_relaxation(fn.membership, fn.weights, problem.matroid, schedule)
    if isinstance(fn, FacilityLocation):
        return facility_relaxation(fn.gains, fn.weights, problem.matroid, schedule)
    if isinstance(fn, BudgetAdditive):
        return budget_relaxation(fn.ground, fn.budget, problem.matroid, schedule)
    raise NotImplementedError(
        f"the differentiable relaxation twin supports Coverage / FacilityLocation / "
        f"BudgetAdditive, not {type(fn).__name__}; use omnibias.submodular.maximize instead"
    )


__all__ = [
    "budget_multilinear",
    "budget_relaxation",
    "continuous_greedy",
    "coverage_multilinear",
    "coverage_relaxation",
    "facility_multilinear",
    "facility_relaxation",
    "graphcut_multilinear",
    "submodular_relaxation",
]
