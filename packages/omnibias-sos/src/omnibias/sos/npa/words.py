# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Noncommutative word algebra: the alphabet an NPA moment hierarchy is built on.

A finite set of ``n`` **Hermitian** operator symbols ``X_0, ..., X_{n-1}`` generates
a free unital ``*``-algebra of noncommuting "words" (ordered products). A word is
represented as a plain exponent-free tuple of generator indices, e.g. ``(0, 1, 0)``
for ``X_0 X_1 X_0``; the empty tuple ``()`` is the identity.

**Hermitian conjugation.** Every generator is self-adjoint, so a product's adjoint
reverses the order: ``(g_1 g_2 ... g_k)^* = g_k ... g_2 g_1``, i.e.
``word^* = reverse(word)`` -- exactly the rule the work item specifies for
self-adjoint generators.

**Relations (both optional, both empty by default -- the "simplest useful case" of
free noncommuting Hermitian generators with no relations at all).**

* ``involutions`` -- a subset of generator indices declared self-inverse,
  ``X_i^2 = 1``. Two adjacent occurrences of such a generator cancel.
* ``anticommuting`` -- a set of unordered generator-index pairs declared to
  anticommute, ``X_i X_j = -X_j X_i`` (all other cross-generator pairs are left
  genuinely **free** -- never reordered, since there is no relation to justify it).
  Combined with ``involutions`` on both members this instantiates the real
  (non-complex) restriction of the qubit Pauli algebra used by
  :mod:`omnibias.sos.npa`'s worked ground-state example (``X_0 X_1 = -X_1 X_0``,
  ``X_0^2 = X_1^2 = 1``) -- a *strict subset* of the full Pauli algebra: there is
  no third, complex generator and no claim of the full anticommutation structure
  of a multi-qubit tensor product. If a future lattice-Hamiltonian / Pauli-string
  package lands with its own concrete operator representation, that representation
  is a natural *instantiation* to check this free algebra's relaxation against
  (a further, larger concrete Hilbert space), not a replacement for it.

A word canonicalizes to a ``SignedWord = (sign, word)`` by repeatedly applying, at
adjacent positions, whichever of these two local rewrite rules applies (a
confluent rewriting system for the small, explicitly declared relation sets this
module is scoped to):

1. two adjacent equal involutive generators cancel (``sign`` unchanged);
2. two adjacent *different* generators ``a > b`` whose pair is declared
   (anti)commuting are swapped into ascending order (``sign`` flips iff the pair
   anticommutes).

Any pair with **no** declared relation is left exactly as encountered -- this is
what keeps the default (no relations) case genuinely free.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import product as _iter_product

Word = tuple[int, ...]
"""An ordered product of generator indices; ``()`` is the identity word."""

SignedWord = tuple[int, Word]
"""``(sign, word)`` with ``sign in {+1, -1}`` and ``word`` already reduced."""


@dataclass(frozen=True)
class NCGenerators:
    r"""A finite set of noncommuting Hermitian operator symbols plus relations.

    Parameters
    ----------
    n:
        Number of generators (``X_0 .. X_{n-1}``), each Hermitian by construction.
    involutions:
        Subset of generator indices with ``X_i^2 = 1``.
    anticommuting:
        Unordered generator-index pairs (as 2-element ``frozenset``s) with
        ``X_i X_j = -X_j X_i``. Pairs not listed here (and not equal) are genuinely
        **free** -- never reordered.
    """

    n: int
    involutions: frozenset[int] = frozenset()
    anticommuting: frozenset[frozenset[int]] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        bad = {i for i in self.involutions if not (0 <= i < self.n)}
        if bad:
            raise ValueError(f"involutions index out of range 0..{self.n - 1}: {sorted(bad)}")
        for pair in self.anticommuting:
            if len(pair) != 2 or any(not (0 <= i < self.n) for i in pair):
                raise ValueError(
                    f"anticommuting pair must be 2 distinct indices in 0..{self.n - 1}: {pair!r}"
                )

    @property
    def identity(self) -> Word:
        """The empty word ``()``, i.e. the algebra's unit."""
        return ()

    def validate_word(self, word: Sequence[int]) -> None:
        """Raise :class:`ValueError` if any generator index in ``word`` is out of range."""
        bad = {g for g in word if not (0 <= g < self.n)}
        if bad:
            raise ValueError(f"generator index out of range 0..{self.n - 1}: {sorted(bad)}")

    def reduce(self, word: Sequence[int]) -> SignedWord:
        r"""Canonicalize ``word`` to a fixed point of the declared rewrite rules.

        Repeatedly cancels adjacent equal involutive generators and swaps adjacent
        *declared* (anti)commuting pairs into ascending index order (flipping the
        running sign on each anticommuting swap) until no rule applies. Pairs with
        no declared relation are left in their original relative order.
        """
        self.validate_word(word)
        sign = 1
        buf = list(word)
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(buf) - 1:
                a, b = buf[i], buf[i + 1]
                if a == b:
                    if a in self.involutions:
                        del buf[i : i + 2]
                        changed = True
                        continue
                    i += 1
                    continue
                if a > b and frozenset((a, b)) in self.anticommuting:
                    buf[i], buf[i + 1] = b, a
                    sign = -sign
                    changed = True
                    i += 1
                    continue
                i += 1
        return sign, tuple(buf)

    def adjoint(self, word: Sequence[int]) -> SignedWord:
        r"""The Hermitian conjugate ``word^*``, canonicalized: ``reduce(reverse(word))``.

        Every generator is self-adjoint, so as an operator identity
        ``(g_1 ... g_k)^* = g_k^* ... g_1^* = g_k ... g_1`` -- literal reversal of
        the (possibly unreduced) input, then canonicalized by :meth:`reduce`, which
        already computes the correct sign/word for *any* raw sequence.
        """
        return self.reduce(tuple(reversed(word)))

    def multiply(self, a: Sequence[int], b: Sequence[int]) -> SignedWord:
        """Reduced product ``a . b`` (concatenation then canonicalization)."""
        return self.reduce(tuple(a) + tuple(b))

    def multiply_signed(self, a: SignedWord, b: SignedWord) -> SignedWord:
        """Reduced product of two already-signed words."""
        sign_a, word_a = a
        sign_b, word_b = b
        sign_c, word_c = self.reduce(word_a + word_b)
        return sign_a * sign_b * sign_c, word_c

    def words_up_to_length(self, max_length: int) -> tuple[Word, ...]:
        r"""Every distinct **reduced** word of length ``0 .. max_length``.

        This is the generating set :math:`\mathcal S_d` of an NPA hierarchy at
        level ``d = max_length``: raw words are enumerated up to the requested
        length and each is canonicalized, so redundant (relation-reducible) raw
        words collapse onto the shorter word they reduce to rather than appearing
        as spurious duplicate basis entries. Returned in graded (length, then
        lexicographic) order; the identity word is always first.
        """
        if max_length < 0:
            raise ValueError(f"max_length must be >= 0, got {max_length}")
        seen: set[Word] = set()
        for length in range(max_length + 1):
            for raw in _iter_product(range(self.n), repeat=length):
                _sign, word = self.reduce(raw)
                seen.add(word)
        return tuple(sorted(seen, key=lambda w: (len(w), w)))


__all__ = ["NCGenerators", "SignedWord", "Word"]
