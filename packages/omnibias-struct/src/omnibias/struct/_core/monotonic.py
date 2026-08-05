# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Monotonic Alignment Search (MAS) lattice: hard MAS + brute-force oracle (pure numpy).

MAS (Glow-TTS) finds the highest-scoring **monotonic, surjective** alignment between ``L``
tokens and ``T >= L`` frames: every frame ``j`` is assigned to exactly one token ``i``, and
the assignment is non-decreasing in ``j``. With a score matrix ``S`` ``(L, T)`` the DP is
``Q[i, j] = S[i, j] + max(Q[i-1, j-1], Q[i, j-1])`` from ``(0, 0)`` to ``(L-1, T-1)`` -- two
moves (advance a token, or stay). The differentiable soft-MAS replaces ``max`` with
``lse_beta`` and anneals back with a certified ``log(N)/beta`` gap, where
``N = C(T-1, L-1)`` is the number of valid alignments. :func:`brute_force_mas` enumerates
them all (the oracle).
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _validate(score: FloatArray) -> tuple[int, int]:
    s = np.asarray(score, dtype=float)
    if s.ndim != 2:
        raise ValueError(f"score must be (L, T), got shape {s.shape}")
    n_tokens, n_frames = s.shape
    if n_frames < n_tokens:
        raise ValueError(f"MAS needs T >= L (frames >= tokens), got L={n_tokens}, T={n_frames}")
    return n_tokens, n_frames


def hard_mas(score: FloatArray) -> float:
    r"""Best monotonic-surjective alignment score of an ``(L, T)`` score matrix."""
    s = np.asarray(score, dtype=float)
    n_tokens, n_frames = _validate(s)
    q = np.full((n_tokens, n_frames), -np.inf)
    q[0, 0] = s[0, 0]
    for j in range(1, n_frames):
        for i in range(min(j, n_tokens - 1) + 1):
            stay = q[i, j - 1]
            advance = q[i - 1, j - 1] if i > 0 else -np.inf
            q[i, j] = s[i, j] + max(stay, advance)
    return float(q[n_tokens - 1, n_frames - 1])


def enumerate_alignments(n_tokens: int, n_frames: int) -> Iterator[tuple[int, ...]]:
    r"""Yield every monotonic-surjective token-per-frame assignment (length ``T``)."""
    if n_frames < n_tokens:
        return

    def dfs(j: int, i: int, acc: tuple[int, ...]) -> Iterator[tuple[int, ...]]:
        if j == n_frames:
            if i == n_tokens - 1:
                yield acc
            return
        # stay on token i
        yield from dfs(j + 1, i, (*acc, i))
        # advance to token i+1 (if any tokens remain, keeping surjectivity feasible)
        if i + 1 < n_tokens:
            yield from dfs(j + 1, i + 1, (*acc, i + 1))

    yield from dfs(1, 0, (0,))


def brute_force_mas(score: FloatArray) -> float:
    r"""Best alignment score by enumerating every valid alignment (oracle)."""
    s = np.asarray(score, dtype=float)
    n_tokens, n_frames = _validate(s)
    return max(
        float(sum(s[i, j] for j, i in enumerate(a)))
        for a in enumerate_alignments(n_tokens, n_frames)
    )


def brute_force_soft_mas(score: FloatArray, beta: float) -> float:
    r"""Exact soft-MAS ``beta^-1 log sum_alignments exp(beta score)`` (global softmax)."""
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    s = np.asarray(score, dtype=float)
    n_tokens, n_frames = _validate(s)
    scores = np.array(
        [float(sum(s[i, j] for j, i in enumerate(a))) for a in enumerate_alignments(n_tokens, n_frames)]
    )
    mx = float(np.max(scores))
    return mx + math.log(float(np.sum(np.exp(beta * (scores - mx))))) / beta


def count_alignments(n_tokens: int, n_frames: int) -> int:
    r"""Number of monotonic-surjective alignments ``C(T-1, L-1)``."""
    if n_frames < n_tokens:
        return 0
    return int(math.comb(n_frames - 1, n_tokens - 1))


__all__ = [
    "brute_force_mas",
    "brute_force_soft_mas",
    "count_alignments",
    "enumerate_alignments",
    "hard_mas",
]
