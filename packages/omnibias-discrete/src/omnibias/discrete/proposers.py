# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Optional discovery proposers that wrap ``anneal_descent``.

Not registered on :func:`omnibias.core.proof.discovery.get_proposer`. Pass an
instance to :func:`~omnibias.core.proof.discovery.run_discovery`. Missing torch
or jax makes :meth:`AnnealDescentProposer.propose` yield nothing so the caller
stays ``BLOCKED``.
"""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from typing import Any

from omnibias.core.proof.discovery import Candidate, FiniteFamily, ScoreGuidedWalk
from omnibias.discrete._core.schedule import AnnealSchedule


def _bit_tuple(family: FiniteFamily, candidate: Candidate) -> tuple[int, ...] | None:
    nbits = getattr(family, "bit_length", None)
    if isinstance(candidate, int) and isinstance(nbits, int) and nbits > 0:
        return tuple((int(candidate) >> i) & 1 for i in range(nbits))
    if isinstance(candidate, tuple) and candidate and all(item in (0, 1) for item in candidate):
        return tuple(int(item) for item in candidate)
    return None


def _from_bits(family: FiniteFamily, bits: tuple[int, ...], origin: Candidate) -> Candidate:
    nbits = getattr(family, "bit_length", None)
    if isinstance(origin, int) and isinstance(nbits, int):
        value = 0
        for i, bit in enumerate(bits):
            if bit:
                value |= 1 << i
        return value
    return bits


class AnnealDescentProposer:
    """Relax ``-score`` with ``anneal_descent``, then snap to a 0/1 vertex."""

    name = "anneal_descent"

    def __init__(self, *, backend: str = "torch", seed: int = 0) -> None:
        self.backend = backend
        self.seed = seed

    def propose(self, family: FiniteFamily, budget: int) -> Iterator[Candidate]:
        if budget <= 0:
            return
        origin = family.origin()
        bits = _bit_tuple(family, origin)
        if bits is None:
            yield from ScoreGuidedWalk().propose(family, budget)
            return
        snapped = self._anneal_bits(family, bits)
        if snapped is None:
            return
        seen: set[Candidate] = set()
        start = _from_bits(family, snapped, origin)
        for candidate in (origin, start, *family.neighbors(start)):
            if candidate in seen:
                continue
            seen.add(candidate)
            yield candidate
            if len(seen) >= budget:
                return
        leftover = budget - len(seen)
        if leftover > 0:
            for candidate in ScoreGuidedWalk().propose(family, leftover + len(seen)):
                if candidate in seen:
                    continue
                seen.add(candidate)
                yield candidate
                if len(seen) >= budget:
                    return

    def _anneal_bits(self, family: FiniteFamily, bits: tuple[int, ...]) -> tuple[int, ...] | None:
        n = len(bits)
        schedule = AnnealSchedule(beta0=0.5, beta_growth=2.0, stages=3, steps=4)

        def energy_of(values: tuple[int, ...]) -> float:
            origin = family.origin()
            candidate = _from_bits(family, values, origin)
            value = family.score(candidate)
            score = value if isinstance(value, Fraction) else Fraction(value)
            return -float(score)

        if self.backend == "jax":
            return self._anneal_jax(family, n, schedule, energy_of)
        return self._anneal_torch(family, n, schedule, energy_of)

    def _anneal_torch(
        self,
        family: FiniteFamily,
        n: int,
        schedule: AnnealSchedule,
        energy_of: Any,
    ) -> tuple[int, ...] | None:
        try:
            import torch
            from omnibias.discrete.torch import anneal_descent
        except ImportError:
            return None

        def grad_x_fn(x: Any) -> Any:
            grads = []
            snapped = tuple(1 if float(v) >= 0.5 else 0 for v in x.detach().tolist())
            base = energy_of(snapped)
            for i in range(n):
                flipped = list(snapped)
                flipped[i] = 1 - flipped[i]
                delta = energy_of(tuple(flipped)) - base
                grads.append(delta)
            return torch.tensor(grads, dtype=x.dtype)

        vertex = anneal_descent(grad_x_fn, scale=1.0, n=n, schedule=schedule)
        return tuple(1 if float(v) >= 0.5 else 0 for v in vertex.detach().tolist())

    def _anneal_jax(
        self,
        family: FiniteFamily,
        n: int,
        schedule: AnnealSchedule,
        energy_of: Any,
    ) -> tuple[int, ...] | None:
        try:
            import jax.numpy as jnp
            from omnibias.discrete.jax import anneal_descent
        except ImportError:
            return None

        def grad_x_fn(x: Any) -> Any:
            snapped = tuple(1 if float(v) >= 0.5 else 0 for v in list(x))
            base = energy_of(snapped)
            grads = []
            for i in range(n):
                flipped = list(snapped)
                flipped[i] = 1 - flipped[i]
                grads.append(energy_of(tuple(flipped)) - base)
            return jnp.asarray(grads, dtype=x.dtype)

        vertex = anneal_descent(grad_x_fn, scale=1.0, n=n, schedule=schedule)
        return tuple(1 if float(v) >= 0.5 else 0 for v in list(vertex))


__all__ = [
    "AnnealDescentProposer",
]
