# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Continuous greedy: the exact (numpy) Frank-Wolfe on the multilinear extension.

Continuous greedy maximizes the multilinear extension ``F`` over a matroid polytope by a
``T``-step Frank-Wolfe flow: from ``p_0 = 0``, at each step take the linear-oracle basis
``y = argmax_{y in P} <grad F(p), y>`` and move ``p <- p + (1/T) y``. For monotone
submodular ``f`` this yields a fractional ``p*`` in the matroid polytope with
``F(p*) >= (1 - 1/e) OPT`` (up to the ``O(1/T)`` discretization loss), which pipage /
swap rounding then turns into an integral independent set without losing value.

This is the exact hard-oracle path that carries the ``(1 - 1/e)`` guarantee; the
bit-identical differentiable torch / jax twins live in
:mod:`omnibias.submodular.torch` / :mod:`omnibias.submodular.jax`.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from omnibias.submodular.functions import SubmodularFunction
from omnibias.submodular.matroid import Matroid

FloatArray = NDArray[np.float64]


def continuous_greedy(
    function: SubmodularFunction, matroid: Matroid, *, steps: int = 25
) -> tuple[FloatArray, list[FloatArray]]:
    r"""Run ``steps``-step continuous greedy; return ``(p_star, bases)``.

    ``p_star in [0, 1]^n`` is the fractional point in the matroid polytope; ``bases`` is
    the sequence of ``0/1`` basis indicators chosen at each step (their mean is
    ``p_star``), consumed by swap rounding.
    """
    if steps < 1:
        raise ValueError("steps must be >= 1")
    n = function.n
    p = np.zeros(n, dtype=float)
    bases: list[FloatArray] = []
    inv = 1.0 / float(steps)
    for _ in range(steps):
        grad = function.multilinear_grad(p)
        # Move along a *full* matroid basis (textbook continuous greedy): for monotone f
        # the gradient is nonnegative so this is still a linear-oracle maximizer, and the
        # equal per-group counts let swap rounding merge the bases without bias.
        y = matroid.fill_basis(grad)
        bases.append(y)
        p = p + inv * y
    return p, bases


__all__ = ["continuous_greedy"]
