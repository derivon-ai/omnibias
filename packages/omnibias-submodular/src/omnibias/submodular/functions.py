# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Monotone submodular functions with exact closed-form multilinear extensions.

The **multilinear extension** of a set function ``f: 2^[n] -> R`` is
``F(p) = E_{x ~ p}[f(x)]`` for ``p in [0, 1]^n`` (each coordinate an independent
Bernoulli), which is exactly the unique multilinear polynomial that interpolates ``f``
on the cube. Continuous greedy maximizes ``F`` over a matroid polytope; the rounding
step lands back on a vertex ``x in {0, 1}^n``. Replacing ``x in {0,1}^n`` by
``p in [0,1]^n`` is a *feasibility / temperature* relaxation hardened back to a vertex,
**not** the founding ``delta -> 0`` bias collapse (see :mod:`omnibias.submodular`).

Several monotone submodular families ship with exact, deterministic (no Monte-Carlo)
multilinear extensions, so the torch / jax relaxation twins stay bit-identical:

* :class:`Coverage` -- weighted (probabilistic) set coverage, the headline max-coverage
  objective, ``F(p) = sum_u w_u (1 - prod_i (1 - C[u,i] p_i))``;
* :class:`FacilityLocation` -- ``f(S) = sum_j w_j max_{i in S} M[j,i]`` with the
  order-statistic expectation for ``F``;
* :class:`BudgetAdditive` -- the concave-of-modular ``f(S) = min(sum_{i in S} a_i, B)``
  with the exact Poisson-binomial expectation for ``F``.

Not every submodular function has a closed-form multilinear extension. :class:`LogDeterminant`
(the determinantal / DPP diversity ``log det(I + K_S)``) is **greedy-path**: its
:meth:`~SubmodularFunction.value` and :meth:`~SubmodularFunction.marginal_gains` are exact
closed form, but ``F(p)`` is a genuine ``2^n`` expectation with no product / order-statistic
/ convolution collapse, so its :meth:`~SubmodularFunction.multilinear` raises
:class:`NotImplementedError` and it carries no differentiable twin -- maximize it with
lazy / stochastic greedy and certify it with the marginal bound.

Not every submodular function is monotone, either. :class:`GraphCut`
(``f(S) = sum_{i in S, j not in S} w_ij``) has the closed-form multilinear extension
``F(p) = deg . p - p^T W p`` (and a differentiable twin), but ``f(empty) = f(V) = 0`` so it
is non-monotone -- maximize it with :func:`~omnibias.submodular.double_greedy` /
:func:`~omnibias.submodular.measured_continuous_greedy`, not the monotone continuous greedy.

Submodularity is closed under a small **algebra** of compositions, so new objectives can be
assembled from the primitives above:

* :class:`Sum` -- ``f(S) = sum_k f_k(S)`` (a nonnegative combination stays monotone
  submodular; by linearity of expectation ``F = sum_k F_k``, so the closed-form extension /
  twin composes whenever every part has one);
* :class:`Scaled` -- ``f(S) = c . g(S)`` for ``c >= 0`` (``F = c . G``);
* :class:`Saturated` -- ``f(S) = min(g(S), cap)`` (a concave, nondecreasing cap preserves
  monotone submodularity, but ``min`` inside the expectation has no closed form, so it is a
  *greedy-path* function like :class:`LogDeterminant`).

Every closed-form-extension function exposes :meth:`~SubmodularFunction.multilinear` (``F``),
the generic :meth:`~SubmodularFunction.multilinear_grad` (exact, since ``F`` is affine in each
coordinate ``dF/dp_i = F(p|p_i=1) - F(p|p_i=0)``), :meth:`~SubmodularFunction.value`
(``f`` on the cube), and :meth:`~SubmodularFunction.to_polynomial` (the multilinear
polynomial via the Moebius / subset transform, for the substrate SOS bound); greedy-path
functions expose everything except the closed-form ``F``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibias.sos import Polynomial

FloatArray = NDArray[np.float64]


def indicator(elements: Iterable[int], n: int) -> FloatArray:
    """The ``0/1`` indicator vector of ``elements`` over the ground set ``[n]``."""
    x = np.zeros(n, dtype=float)
    for i in elements:
        x[int(i)] = 1.0
    return x


def _multilinear_coeffs(values: FloatArray, n: int) -> dict[tuple[int, ...], float]:
    r"""Multilinear (Moebius) coefficients ``c_T`` from ``f`` on all ``2^n`` subsets.

    ``values[mask]`` is ``f`` at the subset whose ``i``-th bit selects element ``i``.
    The unique multilinear interpolation ``f(S) = sum_{T subseteq S} c_T`` inverts to
    ``c_T = sum_{S subseteq T} (-1)^{|T|-|S|} f(S)``, computed in ``O(n 2^n)`` by the
    in-place subset (Moebius) transform.
    """
    h = np.array(values, dtype=float)
    for i in range(n):
        step = 1 << i
        for mask in range(1 << n):
            if mask & step:
                h[mask] -= h[mask ^ step]
    coeffs: dict[tuple[int, ...], float] = {}
    for mask in range(1 << n):
        c = float(h[mask])
        if c != 0.0:
            exp = tuple(1 if (mask >> i) & 1 else 0 for i in range(n))
            coeffs[exp] = c
    return coeffs


class SubmodularFunction(ABC):
    """A submodular set function ``f: 2^[n] -> R`` with a multilinear extension ``F``."""

    @property
    @abstractmethod
    def n(self) -> int:
        """Ground-set size."""

    @property
    def is_monotone(self) -> bool:
        r"""Whether ``f`` is nondecreasing (``f(S) <= f(T)`` for ``S subseteq T``).

        Most shipped families are monotone, so this defaults to ``True`` and only
        :class:`GraphCut` (and any composite built over it) overrides it.

        This is **load-bearing, not documentation**: several upper bounds are only
        valid for monotone ``f``. :func:`~omnibias.submodular.marginal_upper_bound`
        bounds ``f(O)`` by way of ``f(O u S)``, a step that needs monotonicity; on a
        non-monotone ``f`` it can return a value strictly below the true optimum,
        which would make the gap certificate claim a bound that does not hold.
        :func:`~omnibias.submodular.certify_submodular_gap` reads this flag to pick a
        bound whose derivation actually applies.
        """
        return True

    @abstractmethod
    def multilinear(self, p: object) -> float | FloatArray:
        r"""The multilinear extension ``F(p) = E_{x ~ p}[f(x)]`` for ``p in [0, 1]^n``.

        Accepts one point ``(n,)`` (returns a ``float``) or a batch ``(m, n)`` (returns
        an ``(m,)`` array). Agrees with :meth:`value` on the cube ``{0, 1}^n``.
        """

    def value(self, x: object) -> float | FloatArray:
        """``f(x)`` on the cube ``x in {0, 1}^n`` (defaults to ``F`` restricted there)."""
        return self.multilinear(x)

    def multilinear_grad(self, p: object) -> FloatArray:
        r"""The exact gradient ``grad F(p)`` at one point ``p in [0, 1]^n``.

        Because ``F`` is affine in each coordinate, ``dF/dp_i = F(p|p_i=1) - F(p|p_i=0)``
        exactly (no finite-difference error); at a vertex this is the marginal gain.
        """
        pv = np.asarray(p, dtype=float).reshape(-1)
        n = self.n
        rows = np.arange(n)
        hi = np.tile(pv, (n, 1))
        hi[rows, rows] = 1.0
        lo = np.tile(pv, (n, 1))
        lo[rows, rows] = 0.0
        grad: FloatArray = np.asarray(self.multilinear(hi), dtype=float) - np.asarray(
            self.multilinear(lo), dtype=float
        )
        return grad

    def marginal_gains(self, x: object) -> FloatArray:
        r"""The marginal gains ``f(x + e_i) - f(x)`` for every ``i`` (``x`` a ``0/1`` point)."""
        xv = np.asarray(x, dtype=float).reshape(-1)
        n = self.n
        base = float(self.value(xv))
        rows = np.arange(n)
        plus = np.tile(xv, (n, 1))
        plus[rows, rows] = 1.0
        return np.asarray(self.value(plus), dtype=float) - base

    @abstractmethod
    def to_polynomial(self) -> Polynomial:
        """The multilinear polynomial of ``f`` (small ``n``; used by the SOS bound)."""

    def _polynomial_via_moebius(self) -> Polynomial:
        """Shared exact construction of the multilinear polynomial (``O(n 2^n)``)."""
        from omnibias.sos import Polynomial

        n = self.n
        idx = np.arange(1 << n, dtype=np.int64)
        bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
        values = np.asarray(self.value(bits), dtype=float)
        return Polynomial(n, _multilinear_coeffs(values, n))


@dataclass(frozen=True)
class Coverage(SubmodularFunction):
    r"""Weighted (probabilistic) set-coverage ``f(S) = sum_u w_u [u covered by S]``.

    With a membership matrix ``C in [0, 1]^{m x n}`` (``C[u, i]`` the probability set
    ``i`` covers element ``u``; ``0/1`` for ordinary coverage) and element weights
    ``w >= 0``, the coverage of a *soft* selection is
    ``F(p) = sum_u w_u (1 - prod_i (1 - C[u, i] p_i))`` -- element ``u`` is covered with
    probability ``1 - prod_i (1 - C[u, i] p_i)``. Monotone and submodular; the canonical
    ``(1 - 1/e)`` max-coverage objective.
    """

    membership: FloatArray
    weights: FloatArray | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        c = np.asarray(self.membership, dtype=float)
        if c.ndim != 2 or c.shape[1] < 1:
            raise ValueError(f"membership must be (m, n) with n >= 1, got {c.shape}")
        if np.any(c < 0.0) or np.any(c > 1.0):
            raise ValueError("membership entries must lie in [0, 1]")
        m = c.shape[0]
        if self.weights is None:
            w = np.ones(m, dtype=float)
        else:
            w = np.asarray(self.weights, dtype=float).reshape(-1)
            if w.shape[0] != m:
                raise ValueError(f"weights must have length {m}, got {w.shape[0]}")
            if np.any(w < 0.0):
                raise ValueError("weights must be nonnegative (monotone coverage)")
        object.__setattr__(self, "membership", c)
        object.__setattr__(self, "weights", w)

    @property
    def n(self) -> int:
        return int(self.membership.shape[1])

    def multilinear(self, p: object) -> float | FloatArray:
        c = self.membership
        w = np.asarray(self.weights, dtype=float)
        pv = np.asarray(p, dtype=float)
        if pv.ndim == 1:
            uncovered = np.prod(1.0 - c * pv[None, :], axis=1)  # (m,)
            return float(np.sum(w * (1.0 - uncovered)))
        # batch (B, n) -> (B, m, n) -> (B,)
        a = 1.0 - c[None, :, :] * pv[:, None, :]
        uncovered_b = np.prod(a, axis=2)  # (B, m)
        return np.asarray(np.sum(w[None, :] * (1.0 - uncovered_b), axis=1), dtype=float)

    def to_polynomial(self) -> Polynomial:
        return self._polynomial_via_moebius()


@dataclass(frozen=True)
class FacilityLocation(SubmodularFunction):
    r"""Facility location ``f(S) = sum_j w_j max_{i in S} M[j, i]`` (``max`` over empty = 0).

    ``M in R_{>=0}^{m x n}`` is the client-by-facility gain matrix and ``w >= 0`` the
    client weights. The multilinear extension uses the order statistic: for client ``j``
    with facility gains sorted ``v_1 >= v_2 >= ...`` (probabilities ``q_k = p`` of the
    sorted facility), ``E[max] = sum_k v_k q_k prod_{l < k} (1 - q_l)``. Monotone and
    submodular.
    """

    gains: FloatArray
    weights: FloatArray | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        m = np.asarray(self.gains, dtype=float)
        if m.ndim != 2 or m.shape[1] < 1:
            raise ValueError(f"gains must be (m, n) with n >= 1, got {m.shape}")
        if np.any(m < 0.0):
            raise ValueError("gains must be nonnegative (monotone facility location)")
        rows = m.shape[0]
        if self.weights is None:
            w = np.ones(rows, dtype=float)
        else:
            w = np.asarray(self.weights, dtype=float).reshape(-1)
            if w.shape[0] != rows:
                raise ValueError(f"weights must have length {rows}, got {w.shape[0]}")
            if np.any(w < 0.0):
                raise ValueError("weights must be nonnegative")
        object.__setattr__(self, "gains", m)
        object.__setattr__(self, "weights", w)

    @property
    def n(self) -> int:
        return int(self.gains.shape[1])

    def value(self, x: object) -> float | FloatArray:
        gains = self.gains
        w = np.asarray(self.weights, dtype=float)
        xv = np.asarray(x, dtype=float)
        if xv.ndim == 1:
            masked = gains * xv[None, :]  # (m, n)
            return float(np.sum(w * np.max(masked, axis=1)))
        masked_b = gains[None, :, :] * xv[:, None, :]  # (B, m, n)
        return np.asarray(np.sum(w[None, :] * np.max(masked_b, axis=2), axis=1), dtype=float)

    def _multilinear_point(self, pv: FloatArray) -> float:
        gains = self.gains
        w = np.asarray(self.weights, dtype=float)
        total = 0.0
        for j in range(gains.shape[0]):
            order = np.argsort(-gains[j])  # descending gain
            v = gains[j, order]
            q = pv[order]
            surv = 1.0  # prod_{l < k} (1 - q_l)
            exp_max = 0.0
            for vk, qk in zip(v, q, strict=True):
                exp_max += float(vk) * float(qk) * surv
                surv *= 1.0 - float(qk)
            total += float(w[j]) * exp_max
        return total

    def multilinear(self, p: object) -> float | FloatArray:
        pv = np.asarray(p, dtype=float)
        if pv.ndim == 1:
            return self._multilinear_point(pv)
        return np.asarray([self._multilinear_point(row) for row in pv], dtype=float)

    def to_polynomial(self) -> Polynomial:
        return self._polynomial_via_moebius()


@dataclass(frozen=True)
class BudgetAdditive(SubmodularFunction):
    r"""Budget-additive (concave-of-modular) ``f(S) = min(sum_{i in S} a_i, budget)``.

    A concave nondecreasing function of a modular ground term ``a >= 0``, hence monotone
    submodular. The multilinear extension is the exact Poisson-binomial expectation
    ``F(p) = E[min(sum_i a_i x_i, B)]`` with ``x_i ~ Bernoulli(p_i)``, computed by an
    exact convolution DP over the distribution of ``sum_i a_i x_i`` (no Monte-Carlo).
    """

    ground: FloatArray
    budget: float
    name: str | None = None

    def __post_init__(self) -> None:
        a = np.asarray(self.ground, dtype=float).reshape(-1)
        if a.shape[0] < 1:
            raise ValueError("ground must have at least one element")
        if np.any(a < 0.0):
            raise ValueError("ground weights must be nonnegative (monotone)")
        if self.budget < 0.0:
            raise ValueError("budget must be nonnegative")
        object.__setattr__(self, "ground", a)
        object.__setattr__(self, "budget", float(self.budget))

    @property
    def n(self) -> int:
        return int(self.ground.shape[0])

    def value(self, x: object) -> float | FloatArray:
        a = self.ground
        xv = np.asarray(x, dtype=float)
        modular = xv @ a
        return float(np.minimum(modular, self.budget)) if xv.ndim == 1 else np.minimum(
            modular, self.budget
        )

    def _multilinear_point(self, pv: FloatArray) -> float:
        a = self.ground
        # distribution of the modular sum as {value: probability}
        dist: dict[float, float] = {0.0: 1.0}
        for ai, pi in zip(a, pv, strict=True):
            nxt: dict[float, float] = {}
            for s, pr in dist.items():
                if pr == 0.0:
                    continue
                nxt[s] = nxt.get(s, 0.0) + pr * (1.0 - float(pi))
                s2 = s + float(ai)
                nxt[s2] = nxt.get(s2, 0.0) + pr * float(pi)
            dist = nxt
        return float(sum(pr * min(s, self.budget) for s, pr in dist.items()))

    def multilinear(self, p: object) -> float | FloatArray:
        pv = np.asarray(p, dtype=float)
        if pv.ndim == 1:
            return self._multilinear_point(pv)
        return np.asarray([self._multilinear_point(row) for row in pv], dtype=float)

    def to_polynomial(self) -> Polynomial:
        return self._polynomial_via_moebius()


@dataclass(frozen=True)
class LogDeterminant(SubmodularFunction):
    r"""Log-determinant (DPP) diversity ``f(S) = log det(I + K_S)`` -- **greedy-path**.

    ``K`` is a symmetric positive-semidefinite kernel and ``K_S`` its principal submatrix
    on the selected rows / columns; this is the (L-ensemble) determinantal-point-process
    log-likelihood, monotone and submodular, that rewards *diverse* selections. :meth:`value`
    is a single ``slogdet``; :meth:`marginal_gains` is the closed-form Schur complement
    ``f(S + i) - f(S) = log(1 + K_ii - K_{i,S} (I + K_S)^{-1} K_{S,i})`` (``>= 0``, so
    monotone).

    Unlike :class:`Coverage` / :class:`FacilityLocation` / :class:`BudgetAdditive`, the
    log-determinant has **no closed-form multilinear extension** -- ``F(p) = E[log det(I +
    K_X)]`` is a genuine ``2^n`` expectation with no product / order-statistic / convolution
    collapse -- so this is a *greedy-path* function: :meth:`multilinear` raises
    :class:`NotImplementedError` and there is no differentiable twin. Maximize it with
    :func:`~omnibias.submodular.lazy_greedy` / :func:`~omnibias.submodular.stochastic_greedy`
    and certify it with the marginal bound (:func:`~omnibias.submodular.certify_submodular_gap`).
    :meth:`to_polynomial` still builds the exact multilinear *polynomial* from :meth:`value`
    via the Moebius transform (small ``n``) for the substrate SOS bound.
    """

    kernel: FloatArray
    name: str | None = None

    def __post_init__(self) -> None:
        k = np.asarray(self.kernel, dtype=float)
        if k.ndim != 2 or k.shape[0] != k.shape[1] or k.shape[0] < 1:
            raise ValueError(f"kernel must be a square (n, n) matrix with n >= 1, got {k.shape}")
        k = 0.5 * (k + k.T)  # a DPP kernel is symmetric; symmetrize defensively
        min_eig = float(np.linalg.eigvalsh(k).min())
        if min_eig < -1e-8:
            raise ValueError(
                f"kernel must be positive semidefinite (min eigenvalue {min_eig:.3e} < 0)"
            )
        object.__setattr__(self, "kernel", k)

    @property
    def n(self) -> int:
        return int(self.kernel.shape[0])

    def _value_point(self, xv: FloatArray) -> float:
        sel = np.where(xv > 0.5)[0]
        if sel.size == 0:
            return 0.0
        ks = self.kernel[np.ix_(sel, sel)]
        _, logabsdet = np.linalg.slogdet(np.eye(sel.size) + ks)  # I + K_S is PD -> sign +1
        return float(logabsdet)

    def value(self, x: object) -> float | FloatArray:
        xv = np.asarray(x, dtype=float)
        if xv.ndim == 1:
            return self._value_point(xv)
        return np.asarray([self._value_point(row) for row in xv], dtype=float)

    def marginal_gains(self, x: object) -> FloatArray:
        r"""Closed-form Schur-complement marginals ``log(1 + K_ii - K_{i,S} (I+K_S)^{-1} K_{S,i})``."""
        xv = np.asarray(x, dtype=float).reshape(-1)
        k = self.kernel
        gains = np.zeros(self.n, dtype=float)
        free = np.where(xv <= 0.5)[0]
        if free.size == 0:
            return gains
        diag_free = np.diagonal(k)[free]
        sel = np.where(xv > 0.5)[0]
        if sel.size == 0:
            gains[free] = np.log1p(diag_free)  # empty S: schur = 1 + K_ii
            return gains
        m_s = np.eye(sel.size) + k[np.ix_(sel, sel)]
        b = k[np.ix_(sel, free)]  # (|S|, |free|): K_{S,i} columns
        z = np.linalg.solve(m_s, b)  # (I + K_S)^{-1} K_{S,i}
        quad = np.sum(b * z, axis=0)  # K_{i,S} (I + K_S)^{-1} K_{S,i}
        schur = 1.0 + diag_free - quad  # >= 1 in exact arithmetic (monotone)
        gains[free] = np.log(np.maximum(schur, 1e-12))
        return gains

    def multilinear(self, p: object) -> float | FloatArray:
        raise NotImplementedError(
            "LogDeterminant has no closed-form multilinear extension (F(p) is a 2^n "
            "expectation); it is a greedy-path function -- maximize with lazy_greedy / "
            "stochastic_greedy and certify with certify_submodular_gap, not continuous_greedy "
            "or the differentiable twins."
        )

    def to_polynomial(self) -> Polynomial:
        return self._polynomial_via_moebius()


@dataclass(frozen=True)
class GraphCut(SubmodularFunction):
    r"""Undirected graph cut ``f(S) = sum_{i in S, j not in S} w_ij`` -- submodular, **non-monotone**.

    ``weights`` is a symmetric nonnegative ``(n, n)`` adjacency matrix (zero diagonal); the
    cut value sums every edge crossing the boundary of ``S``. It is submodular but *not*
    monotone (``f(empty) = f(V) = 0``), so monotone greedy / continuous greedy carry no
    guarantee -- maximize it with :func:`~omnibias.submodular.double_greedy` (unconstrained,
    ``1/2``) or :func:`~omnibias.submodular.measured_continuous_greedy` (matroid, ``1/e``).

    Because ``E[x_i + x_j - 2 x_i x_j] = p_i + p_j - 2 p_i p_j`` factorizes over independent
    Bernoullis, the multilinear extension is closed form,
    ``F(p) = deg . p - p^T W p`` with ``deg_i = sum_j w_ij``, and the exact gradient is
    ``dF/dp = deg - 2 W p``. So GraphCut also carries a bit-identical differentiable twin
    (:func:`omnibias.submodular.torch.graphcut_multilinear` /
    :func:`omnibias.submodular.jax.graphcut_multilinear`).
    """

    weights: FloatArray
    name: str | None = None

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=float)
        if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] < 1:
            raise ValueError(f"weights must be a square (n, n) matrix with n >= 1, got {w.shape}")
        w = 0.5 * (w + w.T)  # an undirected graph has symmetric weights
        np.fill_diagonal(w, 0.0)  # self-loops never cross a cut
        if np.any(w < 0.0):
            raise ValueError("weights must be nonnegative")
        object.__setattr__(self, "weights", w)

    @property
    def n(self) -> int:
        return int(self.weights.shape[0])

    @property
    def is_monotone(self) -> bool:
        """``f(empty) = f(V) = 0`` while interior sets cut edges, so ``f`` decreases."""
        return False

    def multilinear(self, p: object) -> float | FloatArray:
        w = self.weights
        deg = w.sum(axis=1)
        pv = np.asarray(p, dtype=float)
        if pv.ndim == 1:
            return float(pv @ deg - pv @ (w @ pv))
        lin = pv @ deg
        quad = np.sum((pv @ w) * pv, axis=1)
        return np.asarray(lin - quad, dtype=float)

    def multilinear_grad(self, p: object) -> FloatArray:
        r"""The exact closed-form gradient ``dF/dp = deg - 2 W p``."""
        w = self.weights
        pv = np.asarray(p, dtype=float).reshape(-1)
        return np.asarray(w.sum(axis=1) - 2.0 * (w @ pv), dtype=float)

    def to_polynomial(self) -> Polynomial:
        return self._polynomial_via_moebius()


@dataclass(frozen=True)
class Sum(SubmodularFunction):
    r"""The sum ``f(S) = sum_k f_k(S)`` of submodular parts -- submodular-closed.

    A sum of monotone submodular functions is monotone submodular, and by linearity of
    expectation the multilinear extension composes, ``F(p) = sum_k F_k(p)`` -- so
    :meth:`multilinear` (and hence the differentiable twin) is closed form **iff every part
    is**. If any part is greedy-path (e.g. :class:`LogDeterminant`), :meth:`multilinear`
    propagates its :class:`NotImplementedError`, while :meth:`value` still composes on the cube.
    """

    parts: tuple[SubmodularFunction, ...]
    name: str | None = None

    def __post_init__(self) -> None:
        parts = tuple(self.parts)
        if not parts:
            raise ValueError("Sum needs at least one part")
        n0 = parts[0].n
        for part in parts:
            if part.n != n0:
                raise ValueError(f"all parts must share n; got {part.n} vs {n0}")
        object.__setattr__(self, "parts", parts)

    @property
    def n(self) -> int:
        return int(self.parts[0].n)

    @property
    def is_monotone(self) -> bool:
        """A sum is monotone exactly when every part is."""
        return all(part.is_monotone for part in self.parts)

    def value(self, x: object) -> float | FloatArray:
        acc = [np.asarray(part.value(x), dtype=float) for part in self.parts]
        total = np.sum(acc, axis=0)
        return float(total) if total.ndim == 0 else np.asarray(total, dtype=float)

    def multilinear(self, p: object) -> float | FloatArray:
        acc = [np.asarray(part.multilinear(p), dtype=float) for part in self.parts]
        total = np.sum(acc, axis=0)
        return float(total) if total.ndim == 0 else np.asarray(total, dtype=float)

    def to_polynomial(self) -> Polynomial:
        return self._polynomial_via_moebius()


@dataclass(frozen=True)
class Scaled(SubmodularFunction):
    r"""A nonnegative scaling ``f(S) = c . g(S)`` -- submodular-closed for ``c >= 0``.

    Scaling by ``c >= 0`` preserves monotonicity and submodularity and commutes with the
    expectation, so ``F(p) = c . G(p)`` (closed form iff the base is). A negative scale would
    flip diminishing returns into increasing returns, so it is rejected.
    """

    base: SubmodularFunction
    scale: float
    name: str | None = None

    def __post_init__(self) -> None:
        if self.scale < 0.0:
            raise ValueError("scale must be nonnegative to preserve monotone submodularity")

    @property
    def n(self) -> int:
        return int(self.base.n)

    @property
    def is_monotone(self) -> bool:
        """``c >= 0`` preserves the ordering, so monotonicity passes straight through."""
        return bool(self.base.is_monotone)

    def value(self, x: object) -> float | FloatArray:
        v = np.asarray(self.base.value(x), dtype=float) * self.scale
        return float(v) if v.ndim == 0 else np.asarray(v, dtype=float)

    def multilinear(self, p: object) -> float | FloatArray:
        v = np.asarray(self.base.multilinear(p), dtype=float) * self.scale
        return float(v) if v.ndim == 0 else np.asarray(v, dtype=float)

    def to_polynomial(self) -> Polynomial:
        return self._polynomial_via_moebius()


@dataclass(frozen=True)
class Saturated(SubmodularFunction):
    r"""A saturating cap ``f(S) = min(g(S), cap)`` -- submodular-closed, **greedy-path**.

    Composing a monotone submodular ``g`` with the concave, nondecreasing scalar ``min(., cap)``
    keeps ``f`` monotone submodular (a concave nondecreasing function of a monotone submodular
    function is submodular). But ``min`` does not pass through the expectation, so
    ``F(p) = E[min(g(X), cap)]`` has no product / order-statistic collapse: like
    :class:`LogDeterminant` this is a *greedy-path* function whose :meth:`multilinear` raises
    :class:`NotImplementedError`. :meth:`value` and the marginal gains are exact, so maximize
    it with lazy / stochastic greedy and certify it with the marginal bound.
    """

    base: SubmodularFunction
    cap: float
    name: str | None = None

    @property
    def n(self) -> int:
        return int(self.base.n)

    @property
    def is_monotone(self) -> bool:
        """``min(., cap)`` is nondecreasing, so it preserves the base's monotonicity."""
        return bool(self.base.is_monotone)

    def value(self, x: object) -> float | FloatArray:
        v = np.asarray(self.base.value(x), dtype=float)
        capped = np.minimum(v, self.cap)
        return float(capped) if capped.ndim == 0 else np.asarray(capped, dtype=float)

    def multilinear(self, p: object) -> float | FloatArray:
        raise NotImplementedError(
            "Saturated is greedy-path: E[min(g, cap)] has no closed-form multilinear "
            "extension; maximize with lazy_greedy / stochastic_greedy and certify with "
            "certify_submodular_gap, not continuous_greedy or the differentiable twins."
        )

    def to_polynomial(self) -> Polynomial:
        return self._polynomial_via_moebius()


def is_monotone_submodular(
    function: SubmodularFunction, *, samples: int = 64, seed: int = 0, tol: float = 1e-9
) -> tuple[bool, bool]:
    r"""Empirically check monotonicity and submodularity on random subset pairs.

    Returns ``(monotone, submodular)``. Monotone: ``f(S) <= f(T)`` for ``S subseteq T``.
    Submodular: diminishing returns ``f(S + i) - f(S) >= f(T + i) - f(T)`` for
    ``S subseteq T`` and ``i notin T``. A guard for tests, not a proof.
    """
    n = function.n
    rng = np.random.default_rng(seed)
    monotone = True
    submodular = True
    for _ in range(samples):
        mask_s = rng.integers(0, 2, size=n).astype(bool)
        extra = rng.integers(0, 2, size=n).astype(bool)
        mask_t = mask_s | extra  # S subseteq T
        s = mask_s.astype(float)
        t = mask_t.astype(float)
        fs = float(function.value(s))
        ft = float(function.value(t))
        if ft < fs - tol:
            monotone = False
        free = np.where(~mask_t)[0]
        if free.size:
            i = int(rng.choice(free))
            s_i = s.copy()
            s_i[i] = 1.0
            t_i = t.copy()
            t_i[i] = 1.0
            gain_s = float(function.value(s_i)) - fs
            gain_t = float(function.value(t_i)) - ft
            if gain_t > gain_s + tol:
                submodular = False
    return monotone, submodular


__all__ = [
    "BudgetAdditive",
    "Coverage",
    "FacilityLocation",
    "GraphCut",
    "LogDeterminant",
    "Saturated",
    "Scaled",
    "SubmodularFunction",
    "Sum",
    "indicator",
    "is_monotone_submodular",
]
