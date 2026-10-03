# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bounded tests for the finite Log-Noetherian certificate substrate."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import replace
from fractions import Fraction

import pytest
from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.core.proof.lean_check import generate_obligation
from omnibias.core.realization.polynomial import (
    AlgebraBudgetExceeded,
    SparsePolynomial,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.log_noetherian import (
    DEFAULT_LN_BUDGET,
    ChainFunctionRef,
    CoordinateRef,
    LNCell,
    LNChain,
    LNFiber,
    LNResourceBudget,
    NamedPolynomial,
    ln_format,
    log_chart_derivative_bound,
    monomial_eigenvalue_bound,
    replay_ln_chain,
    seal_ln_certificate,
    seal_ln_chain_closure_obligation,
    seal_ln_format_bound_obligation,
    verify_ln_certificate,
    verify_ln_chain,
)

mpmath = pytest.importorskip("mpmath")


def _constant(nvars: int, value: int | Fraction) -> SparsePolynomial:
    return SparsePolynomial.constant(nvars, value)


def _valid_chain(*, budget: LNResourceBudget = DEFAULT_LN_BUDGET) -> LNChain:
    cell = LNCell(
        (
            LNFiber("Disc", (Fraction(2),)),
            LNFiber("Annulus", (Fraction(1, 2), Fraction(2))),
        )
    )
    x = SparsePolynomial.variable(2, 0)
    y = SparsePolynomial.variable(2, 1)
    zero_base = _constant(2, 0)
    two_base = _constant(2, 2)
    zero_chain = _constant(2, 0)
    two_chain = _constant(2, 2)
    chain_y = SparsePolynomial.variable(2, 1)
    return LNChain(
        cell=cell,
        functions=(NamedPolynomial("x", x), NamedPolynomial("y", y)),
        closure_matrix=((two_chain, zero_chain), (zero_chain, chain_y)),
        claimed_derivatives=((two_base, zero_base), (zero_base, y)),
        budget=budget,
    )


def _perturbed_chain() -> LNChain:
    chain = _valid_chain()
    twice_y = 2 * SparsePolynomial.variable(2, 1)
    return LNChain(
        cell=chain.cell,
        functions=chain.functions,
        closure_matrix=(
            chain.closure_matrix[0],
            (chain.closure_matrix[1][0], twice_y),
        ),
        claimed_derivatives=chain.claimed_derivatives,
        budget=chain.budget,
    )


def test_exact_closure_accepts_and_perturbation_rejects() -> None:
    chain = _valid_chain()
    report = replay_ln_chain(chain)
    assert report.valid
    assert report.failures == ()
    assert verify_ln_chain(chain)

    perturbed = _perturbed_chain()
    bad = replay_ln_chain(perturbed)
    assert not bad.valid
    assert not verify_ln_chain(perturbed)
    assert len(bad.failures) == 1
    assert bad.failures[0].function == "y"
    assert bad.failures[0].axis == 1
    assert bad.failures[0].stage == "closure_composition"


def test_claimed_derivative_is_independently_replayed() -> None:
    chain = _valid_chain()
    wrong = LNChain(
        cell=chain.cell,
        functions=chain.functions,
        closure_matrix=chain.closure_matrix,
        claimed_derivatives=(
            ((_constant(2, 3)), chain.claimed_derivatives[0][1]),
            chain.claimed_derivatives[1],
        ),
        budget=chain.budget,
    )
    report = replay_ln_chain(wrong)
    assert not report.valid
    assert {failure.stage for failure in report.failures} == {
        "claimed_derivative",
        "closure_composition",
    }


def test_named_disc_radius_participates_in_exact_derivative() -> None:
    radius = _constant(1, 2)
    x = SparsePolynomial.variable(1, 0)
    zero_base = _constant(1, 0)
    two_base = _constant(1, 2)
    zero_chain = _constant(2, 0)
    radius_chain = SparsePolynomial.variable(2, 0)
    chain = LNChain(
        cell=LNCell((LNFiber("Disc", (ChainFunctionRef("radius"),)),)),
        functions=(
            NamedPolynomial("radius", radius),
            NamedPolynomial("x", x),
        ),
        closure_matrix=((zero_chain,), (radius_chain,)),
        claimed_derivatives=((zero_base,), (two_base,)),
    )
    assert verify_ln_chain(chain)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LNFiber("Line", ()),  # type: ignore[arg-type]
        lambda: LNFiber("Point", (Fraction(1),)),
        lambda: LNFiber("Disc"),
        lambda: LNFiber("PuncturedDisc", (Fraction(0),)),
        lambda: LNFiber("Annulus", (Fraction(2), Fraction(1))),
        lambda: LNFiber(
            "Annulus",
            (
                ChainFunctionRef("r", Fraction(2)),
                ChainFunctionRef("r", Fraction(1)),
            ),
        ),
        lambda: LNFiber("Disc", (0.5,)),  # type: ignore[arg-type]
    ],
)
def test_fiber_refusals(factory: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()  # type: ignore[operator]


def test_cell_extension_derivations_and_real_marker() -> None:
    cell = LNCell(
        (
            LNFiber("Point"),
            LNFiber("Disc", (Fraction(2),)),
            LNFiber("PuncturedDisc", (Fraction(3),)),
            LNFiber("Annulus", (Fraction(1, 2), Fraction(2))),
        ),
        real_part=True,
    )
    extended = cell.delta_extension(Fraction(1, 2))
    assert extended.real_part
    assert extended.fibers[0].radii == ()
    assert extended.fibers[1].radii == (Fraction(4),)
    assert extended.fibers[2].radii == (Fraction(6),)
    assert extended.fibers[3].radii == (Fraction(1, 4), Fraction(4))

    disc_derivation = cell.standard_derivation(1)
    assert disc_derivation.operator == "radius_d_z"
    assert disc_derivation.multiplier == Fraction(2)
    assert disc_derivation.real_part
    log_derivation = cell.standard_derivation(2)
    assert log_derivation.operator == "z_d_z"
    assert log_derivation.multiplier == CoordinateRef(2)


@pytest.mark.parametrize("delta", [Fraction(0), Fraction(1), Fraction(-1)])
def test_delta_extension_refuses_invalid_delta(delta: Fraction) -> None:
    with pytest.raises(ValueError, match="0 < delta < 1"):
        LNCell((LNFiber("Disc", (Fraction(1),)),)).delta_extension(delta)


def test_delta_extension_refuses_uncertified_symbolic_annulus_margin() -> None:
    cell = LNCell(
        (
            LNFiber(
                "Annulus",
                (ChainFunctionRef("inner"), ChainFunctionRef("outer")),
            ),
        )
    )
    with pytest.raises(ValueError, match="positive annulus margin"):
        cell.delta_extension(Fraction(1, 2))


def test_cell_refusals() -> None:
    with pytest.raises(TypeError, match="real_part"):
        LNCell((), real_part=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="LNFiber"):
        LNCell(("Disc",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="out of range"):
        LNCell((LNFiber("Point"),)).standard_derivation(1)
    with pytest.raises(TypeError, match="exact"):
        LNCell((LNFiber("Disc", (Fraction(1),)),)).delta_extension(0.5)  # type: ignore[arg-type]


def test_ln_format_l2_values_are_split_exactly() -> None:
    sup_bound = Interval.from_rational(Fraction(3))
    report = ln_format(
        _valid_chain(),
        cell_format=Fraction(5),
        sup_bound=sup_bound,
    )
    # 5 cell + 2 dimensions + 2 functions
    # + (deg 0 + norm 2) + (deg 1 + norm 1).
    assert report.cell_format == Fraction(5)
    assert report.combinatorial_part == Fraction(13)
    assert report.sup_contribution == sup_bound
    assert report.total.contains(16.0)

    with pytest.raises(ValueError, match="closure replay failed"):
        ln_format(
            _perturbed_chain(),
            cell_format=Fraction(5),
            sup_bound=sup_bound,
        )
    with pytest.raises(ValueError, match="cell_format"):
        ln_format(_valid_chain(), cell_format=0, sup_bound=sup_bound)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ln_format(
            _valid_chain(),
            cell_format=1,
            sup_bound=Interval(-1.0, 1.0),
        )


def test_l1_bounds_laurent_monomial_on_annulus_dense_and_random() -> None:
    # f(z)=z^-2 is holomorphic on A(1/4, 8), the delta-extension of A(1,2)
    # at delta=1/4. Its extension supremum is 16. In w=log(z),
    # d_w^k f=(-2)^k f.
    order = 2
    cauchy = log_chart_derivative_bound(
        Interval.from_rational(Fraction(16)),
        Fraction(1, 4),
        order,
    )
    eigen = monomial_eigenvalue_bound(
        Fraction(-2),
        order,
        Interval.from_rational(Fraction(1)),
    )
    assert monomial_eigenvalue_bound(Fraction(-2), order) == Fraction(4)
    assert eigen.contains(4.0)
    assert cauchy.lo >= 0.0
    assert math.isfinite(cauchy.hi)

    deterministic = [
        (radius, angle)
        for radius in (1.0, 1.5, 2.0)
        for angle in tuple(-math.pi + 2 * math.pi * index / 24 for index in range(25))
    ]
    rng = random.Random(1729)
    random_points = [
        (rng.uniform(1.0, 2.0), rng.uniform(-math.pi, math.pi))
        for _ in range(48)
    ]
    with mpmath.workdps(60):
        for radius, angle in deterministic + random_points:
            z = mpmath.mpc(
                radius * math.cos(angle),
                radius * math.sin(angle),
            )
            w = mpmath.log(z)
            reference = abs(
                mpmath.diff(lambda value: mpmath.exp(-2 * value), w, order)
            )
            assert float(reference) <= eigen.hi
            assert float(reference) <= cauchy.hi


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"delta": Fraction(0), "order": 1}, ValueError),
        ({"delta": Fraction(1), "order": 1}, ValueError),
        ({"delta": Fraction(1, 2), "order": -1}, ValueError),
        (
            {
                "delta": Fraction(1, 2),
                "order": 1,
                "margin_ratio": Fraction(0),
            },
            ValueError,
        ),
        (
            {
                "delta": Fraction(1, 2),
                "order": 1,
                "margin_ratio": Fraction(2),
            },
            ValueError,
        ),
    ],
)
def test_log_chart_bound_refusals(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        log_chart_derivative_bound(Interval.point(1.0), **kwargs)  # type: ignore[arg-type]


def test_unbounded_and_budgeted_bound_refusals() -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        log_chart_derivative_bound(Interval(0.0, math.inf), Fraction(1, 2), 1)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        monomial_eigenvalue_bound(2, 1, Interval(-1.0, 1.0))
    tiny = replace(LNResourceBudget(), max_derivative_order=1)
    with pytest.raises(AlgebraBudgetExceeded, match="order budget"):
        log_chart_derivative_bound(
            Interval.point(1.0),
            Fraction(1, 2),
            2,
            budget=tiny,
        )
    with pytest.raises(AlgebraBudgetExceeded, match="order budget"):
        monomial_eigenvalue_bound(2, 2, budget=tiny)
    with pytest.raises(TypeError, match="alpha"):
        monomial_eigenvalue_bound(0.5, 1)  # type: ignore[call-overload]
    with pytest.raises(ValueError, match="nonnegative integer"):
        monomial_eigenvalue_bound(2, -1)


def test_chain_shape_name_reference_and_resource_refusals() -> None:
    chain = _valid_chain()
    with pytest.raises(ValueError, match="unique"):
        LNChain(
            chain.cell,
            (
                NamedPolynomial("same", chain.functions[0].polynomial),
                NamedPolynomial("same", chain.functions[1].polynomial),
            ),
            chain.closure_matrix,
            chain.claimed_derivatives,
        )
    with pytest.raises(ValueError, match="shape"):
        LNChain(
            chain.cell,
            chain.functions,
            (chain.closure_matrix[0],),
            chain.claimed_derivatives,
        )
    with pytest.raises(ValueError, match="wrong variable count"):
        LNChain(
            chain.cell,
            (NamedPolynomial("bad", SparsePolynomial.variable(1, 0)),),
            ((_constant(1, 0), _constant(1, 0)),),
            ((_constant(2, 0), _constant(2, 0)),),
        )
    with pytest.raises(ValueError, match="unknown chain functions"):
        LNChain(
            LNCell((LNFiber("Disc", (ChainFunctionRef("missing"),)),)),
            (NamedPolynomial("x", SparsePolynomial.variable(1, 0)),),
            ((_constant(1, 1),),),
            ((_constant(1, 1),),),
        )
    with pytest.raises(AlgebraBudgetExceeded, match="dimension budget"):
        _valid_chain(budget=replace(LNResourceBudget(), max_dimension=1))
    with pytest.raises(ValueError, match="positive integers"):
        LNResourceBudget(max_terms=0)


def test_composition_resource_refusal() -> None:
    cell = LNCell((LNFiber("PuncturedDisc", (Fraction(2),)),))
    x = SparsePolynomial.variable(1, 0)
    square_in_chain = SparsePolynomial(1, {(2,): 1})
    chain = LNChain(
        cell,
        (NamedPolynomial("x", x),),
        ((square_in_chain,),),
        ((x,),),
        budget=replace(LNResourceBudget(), max_composition_products=1),
    )
    with pytest.raises(AlgebraBudgetExceeded, match="composition product"):
        replay_ln_chain(chain)


def test_certificate_seal_tamper_semantic_replay_and_refusals() -> None:
    chain = _valid_chain()
    cert = seal_ln_certificate(
        chain,
        cell_format=Fraction(5),
        sup_bound=Interval.from_rational(Fraction(3)),
    )
    assert verify_certificate_digest(cert)
    assert verify_ln_certificate(cert)
    assert cert["meta"]["transcend_backend"] == "not_used"
    assert cert["honesty"]["dulac_map_log_noetherian_claim"] is False

    tampered = copy.deepcopy(cert)
    tampered["payload"]["format"]["combinatorial_part"] = [12, 1]
    assert not verify_certificate_digest(tampered)
    assert not verify_ln_certificate(tampered)

    semantic_payload = copy.deepcopy(cert["payload"])
    semantic_payload["chain"]["closure_matrix"][1][1]["terms"][0][1] = [2, 1]
    semantically_bad = make_certificate(
        claim=cert["claim"],
        payload=semantic_payload,
        honesty=cert["honesty"],
        meta=cert["meta"],
    )
    assert verify_certificate_digest(semantically_bad)
    assert not verify_ln_certificate(semantically_bad)

    boolean_payload = copy.deepcopy(cert["payload"])
    boolean_payload["closure_valid"] = True
    boolean_claim = make_certificate(
        claim=cert["claim"],
        payload=boolean_payload,
        honesty=cert["honesty"],
        meta=cert["meta"],
    )
    assert not verify_ln_certificate(boolean_claim)

    with pytest.raises(ValueError, match="closure replay failed"):
        seal_ln_certificate(
            _perturbed_chain(),
            cell_format=5,
            sup_bound=Interval.point(3.0),
        )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        seal_ln_certificate(
            chain,
            cell_format=5,
            sup_bound=Interval(0.0, math.inf),
        )


def test_certificate_refuses_wrong_provenance_and_malformed_payload() -> None:
    cert = seal_ln_certificate(
        _valid_chain(),
        cell_format=5,
        sup_bound=Interval.point(3.0),
    )
    wrong_meta = make_certificate(
        claim=cert["claim"],
        payload=cert["payload"],
        honesty=cert["honesty"],
        meta={"transcend_backend": "mpmath"},
    )
    assert verify_certificate_digest(wrong_meta)
    assert not verify_ln_certificate(wrong_meta)

    wrong_claim = make_certificate(
        claim="physical Dulac map is Log-Noetherian",
        payload=cert["payload"],
        honesty=cert["honesty"],
        meta=cert["meta"],
    )
    assert verify_certificate_digest(wrong_claim)
    assert not verify_ln_certificate(wrong_claim)

    malformed_payload = copy.deepcopy(cert["payload"])
    malformed_payload["chain"]["functions"][0]["polynomial"]["terms"][0][0] = [-1, 0]
    malformed = make_certificate(
        claim=cert["claim"],
        payload=malformed_payload,
        honesty=cert["honesty"],
        meta=cert["meta"],
    )
    assert not verify_ln_certificate(malformed)


def test_minimal_kernel_obligation_payloads_are_rederived() -> None:
    chain = _valid_chain()
    closure = seal_ln_chain_closure_obligation(chain)
    assert verify_certificate_digest(closure)
    closure_source = generate_obligation(closure)
    assert closure_source is not None
    assert "allRatEq" in closure_source

    format_report = ln_format(
        chain,
        cell_format=5,
        sup_bound=Interval.point(3.0),
    )
    format_cert = seal_ln_format_bound_obligation(
        format_report,
        strict_upper=14,
    )
    assert verify_certificate_digest(format_cert)
    format_source = generate_obligation(format_cert)
    assert format_source is not None
    assert "allRatLt" in format_source
    with pytest.raises(ValueError, match="must exceed"):
        seal_ln_format_bound_obligation(format_report, strict_upper=13)
