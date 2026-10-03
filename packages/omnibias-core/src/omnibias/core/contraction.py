# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Combinatorics for the deep-network Laplacian / poly-Laplacian fast lane.

:mod:`omnibias.core.multi_index` materialises *every* mixed partial up to a
total order -- the right tool for "give me the whole jet", but the wrong one
for an operator that only ever *contracts* the jet down to one number (the
Laplacian, or its ``k``-th iterate). This module supplies the
backend-agnostic bookkeeping the ``deep_field_laplacian`` /
``deep_field_polylaplacian`` kernels in :mod:`omnibias.jax.laplacian` and
:mod:`omnibias.torch.laplacian` need to do that contraction directly, so a
deep network's Laplacian at any input dimension ``D`` never has to pay the
``comb(D + 2k, D)`` multi-index ceiling.

Three routes are exposed, mirroring the tiers documented on the field
operators:

* **Tier A (forward, k = 1 only)** -- the classic forward-Laplacian
  recursion carries a Jacobian and one scalar per hidden unit, at
  ``O(B * H * D)``, with no combinatorial term at all. It has no ceiling and
  applies at every ``D``, so :func:`select_mode` always returns
  ``"forward"`` for ``k = 1``.
* **Tier B (support-grouped local jets, k >= 2, exact while it fits a budget)**
  -- the multinomial expansion ``Delta^k = (sum_i d_i^2)^k = k! * sum_{|a|=k}
  D^(2a) f / a!`` is evaluated exactly by grouping the terms ``a`` by their
  *support* (the axes with ``a_i > 0``, at most ``k`` of them) and reading
  each group off one small local multivariate jet restricted to that
  support. :func:`polylaplacian_support_terms` returns that grouping;
  :func:`support_jet_count` is the number of local-jet evaluations it costs.
* **Tier C (estimator, k >= 2, unbounded D)** -- the spherical identity in
  :func:`polylaplacian_normalizer` turns ``Delta^k f`` into a normalised
  expectation of a *directional* ``2k``-th derivative over the unit sphere,
  estimated by sampling directions; :mod:`omnibias.core.verified.sampled`
  supplies the concentration bound around the sample mean.

Every function here is pure Python (no numpy / torch / jax), so both backend
kernels share the exact same combinatorics and stay bit-identical by
construction, the same discipline as :mod:`omnibias.core.multi_index`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from math import comb, factorial
from typing import TYPE_CHECKING, Literal

from omnibias.core.multi_index import multi_index_factorial, multi_indices

if TYPE_CHECKING:
    from omnibias.core.verified.sampled import ConcentrationReport

#: Number of local-jet evaluations above which :func:`select_mode` prefers
#: the unbiased estimator (Tier C) over the exact support-grouped route (Tier B).
#: This is deliberately smaller than :data:`omnibias.core.multi_index.MAX_MULTI_INDICES`:
#: support cost is dominated by the *number of support sets*, not by the size
#: of any one jet, so a much smaller ceiling already protects against runaway
#: work.
DEFAULT_SUPPORT_BUDGET: int = 20_000

Mode = Literal["forward", "support", "estimator"]

_CACHE_SIZE: int = 256


@dataclass(frozen=True)
class PolylaplacianReport:
    """Metadata returned by ``deep_field_polylaplacian_with_report``.

    Attributes
    ----------
    mode : Mode
        Resolved tier after ``select_mode`` (or an explicit ``mode`` override).
    dim : int
        Ambient input dimension ``D``.
    k : int
        Poly-Laplacian order.
    budget : int
        Support-jet budget used for the Tier B/C decision.
    n_directions : int | None
        Sample count when ``mode == "estimator"``, else ``None``.
    concentration : ConcentrationReport | None
        Hoeffding enclosure around the Tier C sample mean, when applicable.
    """

    mode: Mode
    dim: int
    k: int
    budget: int
    n_directions: int | None = None
    concentration: ConcentrationReport | None = None


def polylaplacian_normalizer(dim: int, k: int) -> float:
    r"""Constant relating ``Delta^k f`` to a directional-derivative expectation.

    .. math::

        \Delta^k f(x) = \frac{2^k\,k!\,\prod_{j=0}^{k-1}(D + 2j)}{(2k)!}\;
            \mathbb{E}_{v \sim S^{D-1}}\!\left[\partial_v^{2k} f(x)\right]

    Derivation sketch: for ``g ~ N(0, I_D)``, ``v = g / \|g\|`` is uniform on
    the sphere and independent of ``\|g\|``; matching the even Gaussian
    moments ``E[g^alpha]`` for ``|alpha| = 2k`` against the multinomial
    expansion ``Delta^k f = k! sum_{|a|=k} D^(2a) f / a!`` gives the constant
    above (``E[\|g\|^{2k}] = prod_{j=0}^{k-1}(D + 2j)`` is the standard
    chi-squared moment). Checked directly against
    :func:`polylaplacian_multinomial_terms` at ``k = 1, 2`` in the test suite
    (``k = 1`` collapses to the familiar ``Delta f = D * E_v[partial_v^2 f]``).

    Parameters
    ----------
    dim : int
        Ambient input dimension ``D >= 1``.
    k : int
        Poly-Laplacian order ``>= 1``.
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    rising = 1
    for j in range(k):
        rising *= dim + 2 * j
    return float(2**k * factorial(k) * rising) / factorial(2 * k)


def polylaplacian_multinomial_terms(
    n_spatial: int, k: int
) -> tuple[tuple[tuple[int, ...], int], ...]:
    r"""Multinomial expansion of ``Delta^k = (sum_i d_i^2)^k``.

    Returns ``(beta, k! / beta!)`` pairs over ``n_spatial`` axes with
    ``|beta| = k``, so that ``Delta^k f = sum_beta (k! / beta!) D^(2 beta) f``.
    Shared by both ``JetMLPVectorField`` twins (promoted out of the
    previously duplicated private ``_polylaplacian_terms`` in
    ``omnibias.pinn.{jax,torch}.fields.jet_mlp``) and by
    :func:`polylaplacian_support_terms` below, which further groups these
    terms by support so a deep field never needs the full order-``2k`` jet.
    """
    if n_spatial < 1:
        raise ValueError(f"n_spatial must be >= 1, got {n_spatial}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    k_fact = factorial(k)
    return tuple(
        (beta, k_fact // multi_index_factorial(beta))
        for beta in multi_indices(n_spatial, k)
        if sum(beta) == k
    )


def _positive_compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Yield every length-``parts`` tuple of positive ints summing to ``total``.

    Standard stars-and-bars enumeration, smallest-first-part order. Every
    entry is ``>= 1`` by construction, so the tuples are exactly the
    multi-indices ``a`` restricted to a support of size ``parts`` with no
    zero entries (the zero entries live outside the support and are dropped).
    """
    if parts < 1:
        raise ValueError(f"parts must be >= 1, got {parts}")
    if total < parts:
        return
    if parts == 1:
        yield (total,)
        return
    for first in range(1, total - parts + 2):
        for rest in _positive_compositions(total - first, parts - 1):
            yield (first, *rest)


def support_jet_count(dim: int, k: int) -> int:
    """Number of local-jet evaluations :func:`polylaplacian_support_terms` costs.

    Equal to ``sum_{s=1}^{min(k, dim)} comb(dim, s)`` -- one local
    multivariate jet per *support set* of size ``s`` (each costing
    ``comb(s + 2k, s)``, bounded purely by ``k`` since ``s <= k``). This is
    the quantity :func:`select_mode` checks against a budget: it grows
    polynomially in ``dim`` for fixed ``k`` (``O(dim^k)``), unlike the
    ``comb(dim + 2k, dim)`` the full multivariate jet would need.
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    s_max = min(k, dim)
    return sum(comb(dim, s) for s in range(1, s_max + 1))


@lru_cache(maxsize=_CACHE_SIZE)
def polylaplacian_support_terms(
    dim: int, k: int
) -> tuple[tuple[tuple[int, ...], tuple[tuple[tuple[int, ...], int], ...]], ...]:
    r"""Group the ``|a| = k`` multinomial terms of ``Delta^k`` by support.

    ``Delta^k f = sum_{|a|=k} (k! / a!) D^(2a) f``, and every surviving ``a``
    has at most ``k`` nonzero entries (since each is ``>= 1`` and they sum to
    ``k``). Grouping by the *support* ``S = {i : a_i > 0}`` (an increasing
    tuple of axis indices, ``|S| <= min(k, dim)``) means every ``a`` sharing a
    support can be read off **one** local multivariate jet of dimension
    ``|S|`` and order ``2k`` -- restricting the network's first affine layer
    to the columns in ``S`` (see ``deep_field_polylaplacian`` in
    :mod:`omnibias.jax.laplacian` / :mod:`omnibias.torch.laplacian``) -- rather
    than one jet per ``a`` or, worse, the full order-``2k`` jet over all
    ``dim`` axes.

    Returns
    -------
    tuple of ``(support, terms)``
        ``support`` is an increasing tuple of axis indices. ``terms`` is a
        tuple of ``(a_local, coeff)`` pairs where ``a_local`` is the
        multi-index restricted to ``support`` (same length and order as
        ``support``, every entry ``>= 1``) and ``coeff = k! / a_local!`` is
        its multinomial weight. Cached on ``(dim, k)``: the grouping is pure
        combinatorics, shared by every point in a batch.
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    from itertools import combinations

    k_fact = factorial(k)
    s_max = min(k, dim)
    groups: list[tuple[tuple[int, ...], tuple[tuple[tuple[int, ...], int], ...]]] = []
    for s in range(1, s_max + 1):
        for support in combinations(range(dim), s):
            terms = tuple(
                (a_local, k_fact // multi_index_factorial(a_local))
                for a_local in _positive_compositions(k, s)
            )
            groups.append((support, terms))
    return tuple(groups)


def select_mode(
    dim: int, k: int, *, budget: int | None = None
) -> Mode:
    """Choose the cheapest ceiling-free route for ``Delta^k`` at this ``(dim, k)``.

    ``k = 1`` always returns ``"forward"`` (Tier A has no ceiling at any
    dimension). For ``k >= 2``, ``"support"`` (Tier B, exact) is returned
    while :func:`support_jet_count` stays within ``budget``
    (:data:`DEFAULT_SUPPORT_BUDGET` when ``budget`` is ``None``); beyond that
    it returns ``"estimator"`` (Tier C, unbiased in expectation, unbounded
    ``dim``).
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k == 1:
        return "forward"
    resolved_budget = DEFAULT_SUPPORT_BUDGET if budget is None else budget
    if resolved_budget < 1:
        raise ValueError(f"budget must be >= 1, got {resolved_budget}")
    if support_jet_count(dim, k) <= resolved_budget:
        return "support"
    return "estimator"


__all__ = [
    "DEFAULT_SUPPORT_BUDGET",
    "Mode",
    "PolylaplacianReport",
    "polylaplacian_multinomial_terms",
    "polylaplacian_normalizer",
    "polylaplacian_support_terms",
    "select_mode",
    "support_jet_count",
]
