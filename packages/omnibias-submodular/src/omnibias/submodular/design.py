# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Eligible D-optimal design adapter for the existing greedy/certificate stack.

Each candidate supplies a factor J_i, so its information is J_i.T J_i.
Fixed independent blocks and an SPD prior make normalized logdet monotone
submodular. Nuisance-profiled and robust objectives are deliberately not
accepted by this adapter. Numerical values retain floating-point error.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from omnibias.submodular.functions import SubmodularFunction
from omnibias.submodular.matroid import UniformMatroid
from omnibias.submodular.problem import SubmodularProblem

if TYPE_CHECKING:
    from omnibias.sos import Polynomial

Array = NDArray[np.float64]


class InformationLogDet(SubmodularFunction):
    """f(S)=logdet(P+sum_i J_i.T J_i)-logdet(P), one block per item."""

    def __init__(self, factors: Sequence[Array], prior: Array) -> None:
        p = np.array(prior, dtype=float, copy=True)
        if (
            p.ndim != 2
            or p.shape[0] != p.shape[1]
            or p.shape[0] == 0
            or not np.all(np.isfinite(p))
            or not np.allclose(p, p.T, atol=1e-12, rtol=0)
        ):
            raise ValueError("finite symmetric positive definite prior required")
        try:
            np.linalg.cholesky(p)
        except np.linalg.LinAlgError as exc:
            raise ValueError("positive definite prior required") from exc
        mats = []
        for factor in factors:
            j = np.asarray(factor, dtype=float)
            if j.ndim != 2 or j.shape[1] != p.shape[0] or not np.all(np.isfinite(j)):
                raise ValueError("each factor must be finite (observations,parameters)")
            mats.append(j.T @ j)
        if not mats:
            raise ValueError("nonempty candidate factors required")
        self._prior = p
        self._blocks = np.stack(mats)
        self._base = float(np.linalg.slogdet(p)[1])

    @property
    def n(self) -> int:
        return int(len(self._blocks))

    def value(self, x: object) -> float | Array:
        v = np.asarray(x, dtype=float)
        if v.ndim not in (1, 2) or v.shape[-1] != self.n or not np.all((v == 0) | (v == 1)):
            raise ValueError("binary selection(s) matching candidate count required")
        mat = self._prior + np.einsum("...c,cij->...ij", v, self._blocks)
        result = np.linalg.slogdet(mat)[1] - self._base
        return float(result) if v.ndim == 1 else np.asarray(result, dtype=float)

    def multilinear(self, p: object) -> float | Array:
        raise NotImplementedError(
            "logdet of weighted information is not the multilinear extension; use greedy"
        )

    def to_polynomial(self) -> Polynomial:
        if self.n > 16:
            raise ValueError("exact subset interpolation limited to 16 candidates")
        return self._polynomial_via_moebius()


def information_design_problem(
    factors: Sequence[Array], prior: Array, *, budget: int
) -> SubmodularProblem:
    function = InformationLogDet(factors, prior)
    if budget < 1 or budget > function.n:
        raise ValueError("budget must be in 1..candidate count")
    return SubmodularProblem(
        function, UniformMatroid(function.n, budget), name="fixed additive D-optimal design"
    )


__all__ = ["InformationLogDet", "information_design_problem"]
