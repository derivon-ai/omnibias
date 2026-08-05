# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Encode integer factoring ``N = p * q`` as a soft-gate Boolean system.

The unknowns are the ``a_bits`` of ``p`` and the ``b_bits`` of ``q`` (LSB-first),
``n = a_bits + b_bits`` Boolean variables in total. The single constraint is the
predicate ``int(p) * int(q) == N``; its multilinear extension is the soft residual
the annealed solver minimizes. The constraint truth table has ``2**n`` rows, so
even *building* the system is exponential in the bit-length -- the first honest
signal that the relaxation does not change the problem's size.
"""

from __future__ import annotations

from collections.abc import Sequence

from omnibias.boolean.torch.ops.solver import BooleanSystem


def factor_system(n_value: int, a_bits: int, b_bits: int) -> BooleanSystem:
    """Boolean system whose solutions ``(p_bits, q_bits)`` satisfy ``p * q == N``."""
    if a_bits < 1 or b_bits < 1:
        raise ValueError("a_bits and b_bits must be >= 1")

    def pred(*bits: int) -> bool:
        p = sum(bits[i] << i for i in range(a_bits))
        q = sum(bits[a_bits + j] << j for j in range(b_bits))
        return p * q == n_value

    return BooleanSystem.from_predicates([pred], a_bits + b_bits)


def bits_to_factors(
    bits: Sequence[int], a_bits: int, b_bits: int
) -> tuple[int, int]:
    """Decode a bit assignment back into the integer pair ``(p, q)``."""
    p = sum(int(bits[i]) << i for i in range(a_bits))
    q = sum(int(bits[a_bits + j]) << j for j in range(b_bits))
    return p, q


def semiprimes_for_width(width: int) -> list[int]:
    """Semiprimes ``N = p * q`` with both factors representable in ``width`` bits.

    Restricted to ``p, q in [2, 2**width - 1]`` and ``N > 2**width - 1`` so the only
    in-range factorizations are the non-trivial ones (no ``1 * N`` shortcut).
    """
    hi = (1 << width) - 1
    out = []
    for p in range(2, hi + 1):
        for q in range(p, hi + 1):
            n = p * q
            if n > hi:
                out.append(n)
    return sorted(set(out))


__all__ = ["bits_to_factors", "factor_system", "semiprimes_for_width"]
