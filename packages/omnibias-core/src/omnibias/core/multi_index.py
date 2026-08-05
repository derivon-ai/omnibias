# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Multi-index combinatorics for multivariate Faà di Bruno jets (pure Python).

This module supplies the backend-agnostic bookkeeping for the *multivariate*
multi-layer jet primitive, the dim-``D`` generalisation of the directional
(1-D path) jets in :mod:`omnibias.jax.jet` / :mod:`omnibias.torch.jet`.

A multivariate jet truncated at total order ``N`` over ``D`` variables
represents a function ``g`` around a base point ``x0`` by its Taylor
coefficients

.. math::

    g(x_0 + \delta) = \sum_{|\alpha| \le N} c_\alpha\, \delta^\alpha,
    \qquad c_\alpha = \frac{D^\alpha g(x_0)}{\alpha!},

where :math:`\alpha \in \mathbb{N}^D` is a *multi-index*,
:math:`|\alpha| = \sum_i \alpha_i`, :math:`\alpha! = \prod_i \alpha_i!` and
:math:`\delta^\alpha = \prod_i \delta_i^{\alpha_i}`.

The coefficients :math:`c_\alpha` are stored densely along the leading axis of a
backend array, in the canonical order returned by :func:`multi_indices`. The
only combinatorics the backend kernels need are:

* :func:`multi_indices` -- the canonical, deterministic ordering of all
  multi-indices with :math:`|\alpha| \le N` (so torch / jax agree row-for-row);
* :func:`multiply_table` -- the truncated-Cauchy-product table that drives
  multivariate polynomial multiplication ``(a*b)_\gamma =
  \sum_{\alpha+\beta=\gamma} a_\alpha b_\beta``, the multivariate replacement
  for the 1-D convolution in the directional kernel;
* :func:`multi_index_factorial` -- :math:`\alpha!`, converting between Taylor
  coefficients and raw partial derivatives.

Everything is pure integer combinatorics (no numpy / torch / jax) and cached on
``(dim, order)`` so the two backends are bit-identical by construction.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product
from math import comb, factorial

MultiIndex = tuple[int, ...]

#: Ceiling on how many multi-indices a single request may materialise. The count
#: is ``comb(dim + order, dim)``, which grows fast in *both* arguments, so one
#: joint bound on the result size is the honest guard -- capping ``dim`` and
#: ``order`` separately would still admit a combinatorial explosion. Checked
#: before anything is allocated, since the count itself is cheap.
MAX_MULTI_INDICES: int = 200_000

#: Bound on each memo. These tables are keyed on caller-supplied ``(dim, order)``,
#: so an unbounded memo would pin every shape ever requested for the process's
#: lifetime.
_CACHE_SIZE: int = 128


def _check_shape(dim: int, order: int) -> None:
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    count = comb(dim + order, dim)
    if count > MAX_MULTI_INDICES:
        raise ValueError(
            f"dim={dim}, order={order} needs {count} multi-indices, above "
            f"MAX_MULTI_INDICES ({MAX_MULTI_INDICES}); lower the total order or the "
            "dimension"
        )


def _generate(dim: int, order: int) -> list[MultiIndex]:
    """All multi-indices in ``N^dim`` with total degree ``<= order``.

    Built up one coordinate at a time rather than by recursion on ``dim``, so a
    wide, low-order request (thousands of variables at order 1) costs memory
    proportional to its output instead of blowing the interpreter stack.
    """
    partial: list[MultiIndex] = [()]
    for _ in range(dim):
        grown: list[MultiIndex] = []
        for head in partial:
            remaining = order - sum(head)
            for value in range(remaining + 1):
                grown.append((*head, value))
        partial = grown
    return partial


@lru_cache(maxsize=_CACHE_SIZE)
def _multi_indices(dim: int, order: int) -> tuple[MultiIndex, ...]:
    _check_shape(dim, order)
    indices = _generate(dim, order)
    indices.sort(key=lambda a: (sum(a), a))
    return tuple(indices)


@lru_cache(maxsize=_CACHE_SIZE)
def _index_position(dim: int, order: int) -> dict[MultiIndex, int]:
    return {a: i for i, a in enumerate(_multi_indices(dim, order))}


@lru_cache(maxsize=_CACHE_SIZE)
def _multiply_table(dim: int, order: int) -> tuple[tuple[int, int, int], ...]:
    indices = _multi_indices(dim, order)
    pos = _index_position(dim, order)
    table: list[tuple[int, int, int]] = []
    for g_idx, gamma in enumerate(indices):
        for alpha in product(*(range(gi + 1) for gi in gamma)):
            beta = tuple(gamma[i] - alpha[i] for i in range(dim))
            table.append((g_idx, pos[alpha], pos[beta]))
    return tuple(table)


def num_multi_indices(dim: int, order: int) -> int:
    """Number of multi-indices with ``|alpha| <= order`` in ``dim`` variables.

    Equals :math:`\\binom{D + N}{D}`, the dimension of the truncated Taylor
    coefficient space (and the length of :func:`multi_indices`).
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    return comb(dim + order, dim)


def multi_indices(dim: int, order: int) -> list[MultiIndex]:
    """Canonical ordering of all multi-indices with ``|alpha| <= order``.

    Indices are sorted by total degree ``|alpha|`` and then lexicographically,
    so element ``0`` is always the zero multi-index ``(0, ..., 0)`` and the next
    ``dim`` entries are the unit vectors ``e_0, ..., e_{dim-1}`` (the gradient
    block). The ordering is deterministic, so dense jets share row indexing
    across backends.
    """
    return list(_multi_indices(dim, order))


def index_position(dim: int, order: int) -> dict[MultiIndex, int]:
    """Map each multi-index to its row in :func:`multi_indices`."""
    return dict(_index_position(dim, order))


def multiply_table(dim: int, order: int) -> list[tuple[int, int, int]]:
    """Truncated Cauchy-product table for multivariate polynomial multiply.

    Returns a list of ``(gamma, alpha, beta)`` *row-index* triples such that
    ``indices[alpha] + indices[beta] == indices[gamma]`` componentwise (with
    ``indices = multi_indices(dim, order)``). The product of two dense jets
    ``a``, ``b`` is then ``c[gamma] = sum a[alpha] * b[beta]`` over all triples
    sharing ``gamma``; pairs whose sum exceeds ``order`` are absent (truncated).
    """
    return list(_multiply_table(dim, order))


def multi_index_factorial(alpha: MultiIndex) -> int:
    """Multi-index factorial :math:`\\alpha! = \\prod_i \\alpha_i!`."""
    result = 1
    for a in alpha:
        if a < 0:
            raise ValueError(f"multi-index entries must be >= 0, got {alpha}")
        result *= factorial(a)
    return result


__all__ = [
    "MultiIndex",
    "index_position",
    "multi_index_factorial",
    "multi_indices",
    "multiply_table",
    "num_multi_indices",
]
