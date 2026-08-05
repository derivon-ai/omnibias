# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Shared pytest config: float64 JAX + a self-contained toy ``DiscreteProblem``.

The toy quadratic exercises the generic substrate through the ``DiscreteProblem`` seam
*without* importing ``omnibias.qubo`` (which depends on this package). ``make_toy`` builds
either the closed-form ``flip_deltas`` fast-path variant or an energy-only variant that
forces the generic batched-energy fallback -- letting the tests prove the two agree.
"""

from __future__ import annotations

import numpy as np
import pytest

try:  # float64 parity for the torch <-> jax relaxation twins
    import jax

    jax.config.update("jax_enable_x64", True)
except ImportError:  # pragma: no cover - jax optional
    pass


class _ToyQUBO:
    """A tiny quadratic pseudo-Boolean problem with the closed-form flip fast path."""

    def __init__(self, Q: object, c: object | None = None, const: float = 0.0) -> None:
        q = np.asarray(Q, dtype=float)
        q = 0.5 * (q + q.T)
        self.Q = q
        self.c = np.zeros(q.shape[0]) if c is None else np.asarray(c, dtype=float)
        self.const = float(const)

    @property
    def n(self) -> int:
        return int(self.Q.shape[0])

    def energy(self, x: object) -> float | np.ndarray:
        xv = np.asarray(x, dtype=float)
        out = np.sum((xv @ self.Q) * xv, axis=-1) + xv @ self.c + self.const
        return float(out) if xv.ndim == 1 else out

    def flip_deltas(self, x: object) -> np.ndarray:
        diag = np.diag(self.Q).copy()
        xv = np.asarray(x, dtype=float)
        grad = diag + 2.0 * (self.Q @ xv) - 2.0 * diag * xv + self.c
        return (1.0 - 2.0 * xv) * grad

    def to_polynomial(self) -> object:
        from omnibias.sos import Polynomial

        n = self.n
        coeffs: dict[tuple[int, ...], float] = {}

        def add(exp: tuple[int, ...], value: float) -> None:
            if value != 0.0:
                coeffs[exp] = coeffs.get(exp, 0.0) + value

        add((0,) * n, self.const)
        for i in range(n):
            add(tuple(1 if k == i else 0 for k in range(n)), float(self.c[i]))
            for j in range(n):
                qij = float(self.Q[i, j])
                if qij:
                    exp = [0] * n
                    exp[i] += 1
                    exp[j] += 1
                    add(tuple(exp), qij)
        return Polynomial(n, coeffs)


class _ToyEnergyOnly:
    """Wraps a toy but exposes only ``n`` / ``energy`` / ``to_polynomial`` (no fast path)."""

    def __init__(self, base: _ToyQUBO) -> None:
        self._base = base

    @property
    def n(self) -> int:
        return self._base.n

    def energy(self, x: object) -> float | np.ndarray:
        return self._base.energy(x)

    def to_polynomial(self) -> object:
        return self._base.to_polynomial()


@pytest.fixture
def make_toy():  # type: ignore[no-untyped-def]
    def _make(Q: object, c: object | None = None, const: float = 0.0, *, fast: bool = True):  # type: ignore[no-untyped-def]
        base = _ToyQUBO(Q, c, const)
        return base if fast else _ToyEnergyOnly(base)

    return _make
