# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact identities, ambiguity, and refusals of finite-library PDE discovery."""

from dataclasses import FrozenInstanceError
from fractions import Fraction

import pytest
from omnibias.core.operator_spec import OperatorSpec
from omnibias.core.profile_discovery import discover_profile_relations
from omnibias.core.profile_system import ProfilePack, ProfileSystem
from omnibias.core.ridge_system import RidgeBudget


def heat_pack(frequency: int) -> ProfilePack:
    return ProfilePack((0, frequency), (-frequency**2, 0), (1, 0, 1), ((1,), (0,)))


def heat_candidates():
    return tuple((0, OperatorSpec.mixed(alpha)) for alpha in ((0, 0), (1, 0), (0, 2)))


def test_distinct_heat_frequencies_remove_single_mode_ambiguity():
    candidates = heat_candidates()
    single = discover_profile_relations((heat_pack(1),), candidates)
    ensemble = discover_profile_relations((heat_pack(1), heat_pack(2)), candidates)
    assert single.rank == 1
    assert len(single.relations) == 2
    assert ensemble.rank == 2
    assert ensemble.candidate_count == 3
    assert ensemble.constraint_count == 4
    assert ensemble.relations == ((Fraction(0), Fraction(-1), Fraction(1)),)
    # Independently: I, Dt and Dxx act on exp(-k²t) sin(kx) by 1,-k²,-k².
    relation, = ensemble.relations
    for frequency in (3, 5, 11):
        assert sum(c * value for c, value in zip(relation, (1, -frequency**2, -frequency**2), strict=True)) == 0
    # Check the same inferred row against an unseen structured family.
    inferred = OperatorSpec.polynomial({(1, 0): relation[1], (0, 2): relation[2]})
    held_out = heat_pack(3)
    system = ProfileSystem.from_operators(
        ((inferred,),), held_out.direction, carrier=held_out.carrier,
        annihilator=held_out.annihilator,
    )
    assert system.residual_coefficients(held_out) == ((Fraction(0),), (Fraction(0),))
    with pytest.raises(FrozenInstanceError):
        ensemble.rank = 0


def test_coupled_component_relation_and_independent_profile_choices():
    identity = OperatorSpec.mixed((0, 0))
    dx = OperatorSpec.mixed((0, 1))
    packs = tuple(
        ProfilePack((0, k), (-k*k, 0), (1, 0, 1), ((1, 0), (0, k))) for k in (1, 2)
    )
    result = discover_profile_relations(packs, ((0, dx), (1, identity), (0, identity)))
    assert result.rank == 2
    assert result.relations == ((Fraction(-1), Fraction(1), Fraction(0)),)
    # Opposite coefficients can cancel for a tied profile, but independently
    # selectable profiles do not make the identity operator a PDE relation.
    positive = heat_pack(1)
    negative = ProfilePack(positive.direction, positive.carrier, positive.annihilator, ((-1,), (0,)))
    independent = discover_profile_relations((positive, negative), ((0, identity),))
    assert independent.rank == 1
    assert independent.relations == ()


def test_full_rank_library_has_no_relation():
    pack = heat_pack(1)
    result = discover_profile_relations(
        (pack,), ((0, OperatorSpec.mixed((0, 0))), (0, OperatorSpec.mixed((0, 1)))),
    )
    assert result.rank == result.candidate_count == 2
    assert result.relations == ()


def test_repeated_roots_and_different_annihilators():
    polynomial = ProfilePack((1,), (0,), (0, 0, 1), ((1,), (0,)))
    oscillatory = ProfilePack((1,), (0,), (1, 0, 1), ((1,), (0,)))
    candidates = tuple((0, OperatorSpec.mixed((order,))) for order in (0, 1, 2))
    result = discover_profile_relations((polynomial,), candidates)
    assert result.relations == ((Fraction(0), Fraction(0), Fraction(1)),)
    combined = discover_profile_relations((polynomial, oscillatory), candidates)
    assert combined.constraint_count == 4
    assert combined.rank == 3
    assert combined.relations == ()


def test_zero_pack_and_duplicate_candidates_retain_full_nullspace():
    zero = ProfilePack((1,), (0,), (1, 0, 1), ((0,), (0,)))
    identity = OperatorSpec.mixed((0,))
    result = discover_profile_relations((zero,), ((0, identity), (0, identity)))
    assert result.rank == 0
    assert result.relations == ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    nonzero = ProfilePack((1,), (0,), (1, 0, 1), ((1,), (0,)))
    duplicate = discover_profile_relations((nonzero,), ((0, identity), (0, identity)))
    assert duplicate.relations == ((Fraction(-1), Fraction(1)),)


@pytest.mark.parametrize("candidate", [(), (0,), (0, None), (False, OperatorSpec.mixed((0, 0))),
                                      (-1, OperatorSpec.mixed((0, 0))), (1, OperatorSpec.mixed((0, 0))),
                                      (0.0, OperatorSpec.mixed((0, 0)))])
def test_malformed_candidates_are_refused(candidate):
    with pytest.raises((TypeError, ValueError)):
        discover_profile_relations((heat_pack(1),), (candidate,))


def test_metadata_dimension_and_directional_refusals():
    with pytest.raises(ValueError, match="nonempty"):
        discover_profile_relations((), heat_candidates())
    with pytest.raises(ValueError, match="nonempty"):
        discover_profile_relations((heat_pack(1),), ())
    with pytest.raises(TypeError, match="ProfilePack"):
        discover_profile_relations((object(),), heat_candidates())
    with pytest.raises(TypeError, match="RidgeBudget"):
        discover_profile_relations((heat_pack(1),), heat_candidates(), budget=object())
    other_dimension = ProfilePack((1,), (0,), (1, 0, 1), ((1,), (0,)))
    with pytest.raises(ValueError, match="share"):
        discover_profile_relations((heat_pack(1), other_dimension), heat_candidates())
    other_outputs = ProfilePack((0, 1), (0, 0), (1, 0, 1), ((1, 0), (0, 0)))
    with pytest.raises(ValueError, match="share"):
        discover_profile_relations((heat_pack(1), other_outputs), heat_candidates())
    with pytest.raises(ValueError, match="dimension"):
        discover_profile_relations((heat_pack(1),), ((0, OperatorSpec.mixed((0,))),))
    with pytest.raises(ValueError, match="directional"):
        discover_profile_relations((heat_pack(1),), ((0, OperatorSpec.directional(1)),))


def test_aggregate_storage_work_and_exact_bit_budgets():
    pack = heat_pack(1)
    candidates = heat_candidates()
    with pytest.raises(ValueError, match="matrix entries"):
        discover_profile_relations((pack, pack), candidates, budget=RidgeBudget(max_matrix_entries=11))
    work_budget = RidgeBudget(max_elimination_work=300)
    assert discover_profile_relations((pack,), candidates, budget=work_budget).rank == 1
    # Each pack alone fits; the total construction/solve must still be bounded.
    with pytest.raises(ValueError, match="arithmetic work"):
        discover_profile_relations((pack, pack), candidates, budget=work_budget)
    huge = ProfilePack(pack.direction, pack.carrier, pack.annihilator, ((257,), (0,)))
    with pytest.raises(ValueError, match="bit budget"):
        discover_profile_relations((huge,), candidates, budget=RidgeBudget(max_coefficient_bits=8))
    with pytest.raises(ValueError, match="quotient order"):
        discover_profile_relations((pack,), candidates, budget=RidgeBudget(max_degree=0))
    with pytest.raises(ValueError, match="operator order"):
        discover_profile_relations((pack,), ((0, OperatorSpec.mixed((3, 0))),),
                                   budget=RidgeBudget(max_order=2))
