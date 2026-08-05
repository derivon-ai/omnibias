# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Truth-table representation of Boolean functions (pure Python).

A Boolean function of ``n`` variables is stored as a tuple of ``2**n`` entries in
``{0, 1}``. The integer index ``i`` encodes an assignment **LSB-first**: bit ``j``
of ``i`` is the value of variable ``x_j`` (so ``x_0`` is the least-significant
bit). This convention is shared by every module in :mod:`omnibias.boolean._core`,
so the ANF (:mod:`~omnibias.boolean._core.anf`), Walsh
(:mod:`~omnibias.boolean._core.walsh`) and multilinear
(:mod:`~omnibias.boolean._core.multilinear`) transforms index their subsets the
same way.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

TruthTable = tuple[int, ...]


def is_power_of_two(value: int) -> bool:
    """Return ``True`` iff ``value`` is a positive power of two."""
    return value > 0 and (value & (value - 1)) == 0


def num_vars(table: TruthTable) -> int:
    """Number of variables ``n`` such that ``len(table) == 2**n``."""
    size = len(table)
    if not is_power_of_two(size):
        raise ValueError(f"truth-table length must be a power of two, got {size}")
    return size.bit_length() - 1


def check_truth_table(table: TruthTable) -> None:
    """Validate that ``table`` has power-of-two length and ``{0, 1}`` entries."""
    num_vars(table)
    for v in table:
        if v not in (0, 1):
            raise ValueError(f"truth-table entries must be 0 or 1, got {v}")


def assignment(index: int, n: int) -> tuple[int, ...]:
    """Decode row ``index`` into the bit tuple ``(x_0, ..., x_{n-1})`` (LSB-first)."""
    if not 0 <= index < (1 << n):
        raise ValueError(f"index {index} out of range for n={n}")
    return tuple((index >> j) & 1 for j in range(n))


def index_of(bits: tuple[int, ...]) -> int:
    """Encode a bit tuple ``(x_0, ..., x_{n-1})`` (LSB-first) into its row index."""
    idx = 0
    for j, b in enumerate(bits):
        if b not in (0, 1):
            raise ValueError(f"bits must be 0 or 1, got {b}")
        idx |= (b & 1) << j
    return idx


def all_assignments(n: int) -> Iterator[tuple[int, ...]]:
    """Iterate over every assignment ``(x_0, ..., x_{n-1})`` in row order."""
    for i in range(1 << n):
        yield assignment(i, n)


def truth_table_from_callable(func: Callable[..., int], n: int) -> TruthTable:
    """Tabulate ``func(x_0, ..., x_{n-1})`` over all ``2**n`` assignments.

    ``func`` receives the bits as positional arguments and returns ``0``/``1``
    (any truthy value is coerced to ``1``).
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    rows = []
    for bits in all_assignments(n):
        rows.append(1 if func(*bits) else 0)
    return tuple(rows)


def truth_table_to_callable(table: TruthTable) -> Callable[..., int]:
    """Return ``func(x_0, ..., x_{n-1}) -> {0, 1}`` reading from ``table``."""
    n = num_vars(table)

    def func(*bits: int) -> int:
        if len(bits) != n:
            raise ValueError(f"expected {n} bits, got {len(bits)}")
        return table[index_of(bits)]

    return func


def bit_to_spin(b: int) -> int:
    """Map a bit ``{0, 1}`` to a spin ``{+1, -1}`` via ``s = 1 - 2b``."""
    if b not in (0, 1):
        raise ValueError(f"bit must be 0 or 1, got {b}")
    return 1 - 2 * b


def spin_to_bit(s: int) -> int:
    """Map a spin ``{+1, -1}`` back to a bit ``{0, 1}`` via ``b = (1 - s) / 2``."""
    if s not in (1, -1):
        raise ValueError(f"spin must be +1 or -1, got {s}")
    return (1 - s) // 2


def pm1_values(table: TruthTable) -> tuple[int, ...]:
    """Map a ``{0, 1}`` truth table to its ``{+1, -1}`` output encoding."""
    return tuple(bit_to_spin(v) for v in table)


__all__ = [
    "TruthTable",
    "all_assignments",
    "assignment",
    "bit_to_spin",
    "check_truth_table",
    "index_of",
    "is_power_of_two",
    "num_vars",
    "pm1_values",
    "spin_to_bit",
    "truth_table_from_callable",
    "truth_table_to_callable",
]
