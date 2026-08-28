# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Rationalize-and-certify: seal a *discovered* law as a checkable ``Q`` obligation.

This module is **not** a new discovery search, a new SINDy variant, or a new
way to find laws from scratch. It is a *verify-and-seal* step that sits
downstream of one: given floating-point-fitted coefficients that an existing
discovery pipeline (or a Lie-symmetry search) already proposed, it

1. snaps each float to the nearest :class:`~fractions.Fraction` with a
   caller-controlled denominator bound (:func:`omnibias.core.proof.lift.as_fraction`
   -- the same primitive :mod:`omnibias.symbolic.symmetry` already uses to
   snap a determining matrix), then
2. re-evaluates the candidate law's *governing residual* at the rationalized
   coefficients using **exact** :class:`~fractions.Fraction` arithmetic, and
3. seals a ``rational_identity`` :mod:`omnibias.core.proof.certificate` (the
   same payload shape :mod:`omnibias.holonomic` already uses for a Lean-
   checkable finite obligation) only when every check's residual is
   *identically* zero over ``Q`` -- never a float tolerance.

This mirrors :func:`omnibias.holonomic.rank_syzygy.certify_holonomic_syzygy`'s
"integerize then rank collapse" pattern one level up the stack: that function
clears denominators of an already-exact matrix and hands it to
:func:`~omnibias.core.collapse.rank.rank_collapse`; this module additionally
does the *snapping* step (float -> nearby rational) that a genuine numerical
discovery pipeline needs first, then performs its own exact check rather than
delegating to rank collapse (the residual here is an arbitrary caller-supplied
finite sum, not a matrix nullspace).

**Scope.** A refusal (``certified=False``) means only that *this* rationalization
at *this* ``max_denominator`` does not close the residual; it is not a proof
that no rational law exists. A pass means the rationalized coefficients satisfy
the declared residual exactly on the declared finite checks -- not that the
float SVD / STLSQ / SINDy proposer that produced the original floats was
itself sound, and not that the law holds beyond the checks actually run.
:data:`omnibias.core.proof.certificate.THEOREM_PROVER_VERIFIED_KEY` is never
supplied here; that flag is earned only by a genuine Lean kernel pass elsewhere
(:mod:`omnibias.core.proof.lean_check`), and :func:`omnibias.core.proof.certificate.make_certificate`
refuses it outright if a caller ever tried.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Any

from omnibias.core.proof.certificate import Cert, make_certificate
from omnibias.core.proof.lift import as_fraction
from omnibias.symbolic.symmetry import Generator, LinearPoly, Sample, SymmetryBasis

#: An exact rational or integer scalar -- never a float.
Number = int | Fraction
#: One term ``coefficient * value`` of a residual sum; both entries exact.
ResidualPair = tuple[Number, Number]
#: One finite exact check: the claim is ``sum(c * v for c, v in check) == 0``.
ResidualCheck = Sequence[ResidualPair]
#: Given the rationalized coefficient vector, return one or more exact checks.
ResidualFn = Callable[[Sequence[Fraction]], Sequence[ResidualCheck]]


def honesty_payload() -> dict[str, bool]:
    """The honesty flags stamped onto every certificate this module seals."""
    return {
        "unproven_claim": False,
        "exact_rational_check": True,
        "float_tolerance_used": False,
        "rationalization_is_a_snap_not_a_derivation": True,
    }


@dataclass(frozen=True)
class DiscoveryCertificate:
    """Verdict of rationalizing, then exactly checking, one discovered coefficient vector.

    ``certified`` is ``True`` only when *every* check :attr:`residuals` reports
    is exactly ``Fraction(0)``; :attr:`certificates` then holds one sealed
    ``rational_identity`` certificate per check (see
    :func:`rationalize_and_certify_discovery`). On refusal, :attr:`certificates`
    is empty and :attr:`residuals` still records every check's actual (generally
    nonzero) exact residual, so the caller can see precisely where the
    rationalization failed to close -- never a forged certificate.
    """

    certified: bool
    rationalized_coefficients: tuple[Fraction, ...]
    max_denominator: int
    residuals: tuple[Fraction, ...]
    certificates: tuple[Cert, ...]
    detail: str

    @property
    def n_checks(self) -> int:
        """Number of exact checks ``residual_fn`` returned."""
        return len(self.residuals)


def _exact_pair_value(value: object, check_index: int) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, int | Fraction):
        raise TypeError(
            "rationalize_and_certify_discovery requires residual_fn to return "
            f"exact Fraction/int (coefficient, value) pairs (check {check_index}); "
            "a float residual is not a certificate"
        )
    return Fraction(value)


def _rational_identity_payload(products: Sequence[Fraction]) -> dict[str, Any]:
    """``{"type": "rational_identity", "lhs_terms": [[int, 1], ...], "rhs": 0}``.

    Cross-multiplies every (already exactly zero-summing) product onto a
    shared integer denominator, mirroring
    ``omnibias.holonomic._core.certify._zero_identity``'s pattern -- the Lean
    kernel obligation this payload feeds
    (:func:`omnibias.core.proof.lean_check.generate_obligation`) checks the
    *un-collapsed* per-term arithmetic sums to zero, not a Python-precomputed
    tautology.
    """
    denom = 1
    for product in products:
        denom = denom * product.denominator // gcd(denom, product.denominator)
    lhs_terms = [[int(product * denom), 1] for product in products]
    return {"type": "rational_identity", "lhs_terms": lhs_terms, "rhs": 0}


def rationalize_and_certify_discovery(
    discovered_coeffs: Sequence[float | Number],
    residual_fn: ResidualFn,
    *,
    max_denominator: int = 1_000_000,
    claim: str,
    meta: Mapping[str, Any] | None = None,
) -> DiscoveryCertificate:
    r"""Snap ``discovered_coeffs`` to ``Q`` and exactly check ``residual_fn`` against them.

    Parameters
    ----------
    discovered_coeffs:
        The floating-point-fitted coefficients a discovery pipeline proposed
        (e.g. a :class:`~omnibias.symbolic.discovery.SparseEquation`'s
        ``coefficients`` / ``intercept``, or a Lie-symmetry generator's linear
        combination weights). Each entry is snapped with
        :func:`omnibias.core.proof.lift.as_fraction` at ``max_denominator``;
        an entry that is already an ``int`` or :class:`~fractions.Fraction`
        passes through exactly, unsnapped.
    residual_fn:
        Given the rationalized coefficients, returns one or more
        :data:`ResidualCheck` -- each a finite list of exact ``(coefficient,
        value)`` pairs whose sum must be *identically* zero over ``Q`` for the
        candidate law to hold at that check. ``residual_fn`` must itself use
        only exact :class:`~fractions.Fraction` / ``int`` arithmetic; any
        pair entry that is a ``float`` (or ``bool``) raises :class:`TypeError`
        -- a float residual is not a certificate, by construction, not by
        convention. See :func:`lie_symmetry_residual_fn` for a worked builder.
    max_denominator:
        Upper bound on the denominator :func:`~omnibias.core.proof.lift.as_fraction`
        may introduce while snapping a float coefficient. Must be positive.
    claim:
        Human-readable label for the sealed certificate(s) (one ``[check i/N]``
        suffix is appended per check).
    meta:
        Extra JSON-serialisable provenance merged into every certificate's
        ``meta`` (e.g. ``{"pde": "transport", "generator": "x d/dx + t d/dt"}``).

    Returns
    -------
    DiscoveryCertificate
        ``certified=True`` with one sealed certificate per check when every
        check's residual is exactly zero; otherwise ``certified=False`` with
        no certificates and the actual (generally nonzero) residuals recorded
        in :attr:`DiscoveryCertificate.residuals` for an honest report.

    Raises
    ------
    ValueError
        ``max_denominator < 1``, ``residual_fn`` returns no checks, or a check
        carries no terms.
    TypeError
        A pair from ``residual_fn`` is not an exact ``int``/``Fraction``.
    """
    if max_denominator < 1:
        raise ValueError("max_denominator must be positive")
    rationalized = tuple(
        as_fraction(coeff, denom_bound=max_denominator) for coeff in discovered_coeffs
    )
    checks = [list(check) for check in residual_fn(rationalized)]
    if not checks:
        raise ValueError("residual_fn must return at least one exact check")

    residuals: list[Fraction] = []
    all_products: list[list[Fraction]] = []
    for index, pairs in enumerate(checks):
        if not pairs:
            raise ValueError(f"check {index} carries no (coefficient, value) terms")
        products = [
            _exact_pair_value(coeff, index) * _exact_pair_value(value, index)
            for coeff, value in pairs
        ]
        all_products.append(products)
        residuals.append(sum(products, Fraction(0)))

    first_nonzero = next((i for i, r in enumerate(residuals) if r != 0), None)
    if first_nonzero is not None:
        detail = (
            f"rationalization refused: check {first_nonzero} of {len(checks)} has exact "
            f"residual {residuals[first_nonzero]} (!= 0) over Q at "
            f"max_denominator={max_denominator}; the rationalized coefficients "
            "do not close this residual"
        )
        return DiscoveryCertificate(
            certified=False,
            rationalized_coefficients=rationalized,
            max_denominator=max_denominator,
            residuals=tuple(residuals),
            certificates=(),
            detail=detail,
        )

    meta_base: dict[str, Any] = dict(meta) if meta is not None else {}
    certificates: list[Cert] = []
    for index, products in enumerate(all_products):
        cert_meta = dict(meta_base)
        cert_meta["check_index"] = index
        cert_meta["n_checks"] = len(checks)
        certificates.append(
            make_certificate(
                claim=f"{claim} [check {index + 1}/{len(checks)}]",
                payload=_rational_identity_payload(products),
                honesty=honesty_payload(),
                meta=cert_meta,
            )
        )
    detail = (
        f"certified: {len(checks)} exact check(s) over Q identically zero at "
        f"max_denominator={max_denominator}"
    )
    return DiscoveryCertificate(
        certified=True,
        rationalized_coefficients=rationalized,
        max_denominator=max_denominator,
        residuals=tuple(residuals),
        certificates=tuple(certificates),
        detail=detail,
    )


def _exact_linear_poly(poly: LinearPoly) -> LinearPoly:
    return LinearPoly(Fraction(poly.c0), Fraction(poly.cx), Fraction(poly.ct), Fraction(poly.cu))


def _exact_generator(gen: Generator) -> Generator:
    return Generator(
        _exact_linear_poly(gen.xi_x),
        _exact_linear_poly(gen.xi_t),
        _exact_linear_poly(gen.eta),
        gen.label,
    )


def _exact_sample(sample: Sample) -> Sample:
    return Sample(*(Fraction(value) for value in dataclasses.astuple(sample)))


def lie_symmetry_residual_fn(
    pr: Callable[[Generator, Sample], float],
    restrict: Callable[[Sample], Sample],
    basis: SymmetryBasis,
    samples: Sequence[Sample],
) -> ResidualFn:
    r"""Build a :data:`ResidualFn` for a Lie-symmetry determining equation.

    A determining equation is linear in the generator once the prolongation is
    fixed (see :mod:`omnibias.symbolic.symmetry`), so the residual of a proposed
    linear combination ``sum_i coeff_i * basis.generators[i]`` at one sample
    equals ``sum_i coeff_i * pr(basis.generators[i], sample)`` exactly. The
    returned callable evaluates every ``basis.generators[i]`` and every
    ``sample`` with :class:`~fractions.Fraction` arithmetic (never floats,
    since dyadic-rational sample coordinates such as
    :func:`omnibias.symbolic.symmetry.designed_samples` convert to
    :class:`~fractions.Fraction` losslessly) and hands
    :func:`rationalize_and_certify_discovery` one check per sample, with one
    ``(coefficient, exact pr value)`` pair per basis generator -- so the sealed
    certificate records genuine per-term arithmetic, not a pre-collapsed zero.

    Only a ``pr`` built purely from ``+``, ``-``, ``*`` on exact inputs stays
    exact under this substitution. ``pr_transport`` qualifies. Several shipped
    residuals do **not**: ``pr_heat`` / ``pr_wave`` / ``pr_laplace`` /
    ``pr_fisher`` multiply by the Python literal ``2.0`` inside their shared
    ``_dxx_q`` / ``_dtt_q`` helper, which silently downgrades a
    :class:`~fractions.Fraction` operand to ``float``; ``pr_burgers`` /
    ``pr_kdv`` / ``pr_kdv_linear`` / ``pr_boussinesq`` route through a
    complex-step ``eta_xxx`` that is not exact at any denominator. This
    function does not special-case those: it raises :class:`TypeError` the
    moment ``pr`` returns a non-exact value at any sample, so an incompatible
    ``pr`` is refused loudly rather than silently certifying a float.

    Raises
    ------
    ValueError
        ``basis`` has no generators, or ``samples`` is empty.
    """
    exact_generators = tuple(_exact_generator(g) for g in basis.generators)
    exact_samples = tuple(restrict(_exact_sample(s)) for s in samples)
    n_generators = len(exact_generators)
    if n_generators == 0:
        raise ValueError("basis must declare at least one generator")
    if not exact_samples:
        raise ValueError("samples must be non-empty")

    def _residual(coeffs: Sequence[Fraction]) -> list[ResidualCheck]:
        if len(coeffs) != n_generators:
            raise ValueError(
                f"expected {n_generators} coefficients (one per basis generator), "
                f"got {len(coeffs)}"
            )
        checks: list[ResidualCheck] = []
        for sample in exact_samples:
            pairs: list[ResidualPair] = []
            for coeff, generator in zip(coeffs, exact_generators, strict=True):
                value = pr(generator, sample)
                if isinstance(value, bool) or not isinstance(value, int | Fraction):
                    raise TypeError(
                        "the determining-equation residual produced a non-exact "
                        f"value ({type(value).__name__}); pr must use only +, -, "
                        "* on Fraction inputs to stay exact here -- see "
                        "lie_symmetry_residual_fn's docstring for which shipped "
                        "pr_* callables qualify"
                    )
                pairs.append((coeff, value))
            checks.append(pairs)
        return checks

    return _residual


__all__ = [
    "DiscoveryCertificate",
    "ResidualCheck",
    "ResidualFn",
    "ResidualPair",
    "honesty_payload",
    "lie_symmetry_residual_fn",
    "rationalize_and_certify_discovery",
]
