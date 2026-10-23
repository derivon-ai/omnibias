# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact sheet sources, jumps and coupled weak PDE identities."""

from fractions import Fraction as F

import pytest
from omnibias.core.operator_spec import OperatorSpec
from omnibias.core.ridge_distribution import (
    RidgeDistribution,
    collapse_dirac_cluster,
    gaussian_pairing_orders,
)
from omnibias.core.ridge_system import RidgeBudget, RidgeSystem


def test_laplacian_relu_retains_sheet_source_and_normal_scaling() -> None:
    field = RidgeDistribution((3, 4), ((1, (1,)),), bias=F(1, 3))
    laplace = RidgeSystem.from_operators(((OperatorSpec.laplacian(),),), (3, 4))
    source = field.apply(laplace)
    assert source.terms == ((-1, (F(25),)),)
    assert source.bias == F(1, 3)
    assert source.regular_terms == ()
    assert source.singular_terms == source.terms
    assert source.apply(laplace).terms == ((-3, (F(625),)),)


def test_jump_derivatives_and_mixed_spatial_operator() -> None:
    field = RidgeDistribution((2, -3), ((0, (2,)), (1, (5,)), (3, (7,))))
    mixed = RidgeSystem.from_operators(((OperatorSpec.mixed((1, 1)),),), field.direction)
    result = field.apply(mixed)
    assert result.terms == ((-2, (F(-12),)), (-1, (F(-30),)), (1, (F(-42),)))


@pytest.mark.parametrize("profile_index", [-3, -1, 0, 1, 3])
def test_coupled_wave_is_zero_including_interface_distributions(profile_index: int) -> None:
    # (u,v,r)=(phi,2 phi',phi') at z=2t+x. The check must retain
    # the singular derivatives; pointwise a.e. differentiation is inadequate.
    field = RidgeDistribution((2, 1), ((profile_index, (1, 0, 0)),
                                       (profile_index - 1, (0, 2, 1))), outputs=3)
    identity = OperatorSpec.mixed((0, 0))
    wave = RidgeSystem.from_operators((
        (OperatorSpec.polynomial({(1, 0): -1}), identity, None),
        (OperatorSpec.polynomial({(0, 1): -1}), None, identity),
        (None, OperatorSpec.mixed((1, 0)), OperatorSpec.polynomial({(0, 1): -4}))), (2, 1))
    residual = field.apply(wave)
    assert residual.outputs == 3
    assert residual.terms == ()


def test_exact_duplicate_combination_and_zero_field_dimension() -> None:
    field = RidgeDistribution((1,), ((2, (F(1, 3),)), (2, (F(2, 3),)),
                                    (-1, (2,)), (-1, (-2,))))
    assert field.terms == ((2, (F(1),)),)
    assert gaussian_pairing_orders(field) == (2, -1)
    assert gaussian_pairing_orders(RidgeDistribution((1,), (), outputs=3)) == (-1, -1)


def test_compiled_kernel_pack_accepts_a_singular_profile() -> None:
    # u_x + v = 0, Q=(1,-2s), w=2. With phi=delta the field has
    # both a delta component and a delta-prime component.
    system = RidgeSystem.from_operators(((OperatorSpec.mixed((1,)),
                                          OperatorSpec.mixed((0,))),), (2,))
    pack, = system.kernel_basis(max_degree=1)
    field = RidgeDistribution.from_syzygy(pack, -1, bias=F(1, 7))
    assert field.bias == F(1, 7) and field.singular_terms == field.terms
    assert field.apply(system).terms == ()
    with pytest.raises(TypeError):
        RidgeDistribution.from_syzygy(None)


@pytest.mark.parametrize("kwargs, error", [
    ({"direction": (0,)}, ValueError),
    ({"direction": (1.0,)}, TypeError),
    ({"bias": 0.1}, TypeError),
    ({"terms": ((True, (1,)),)}, ValueError),
    ({"terms": ((1.0, (1,)),)}, ValueError),
    ({"terms": ((1, (1.0,)),)}, TypeError),
    ({"terms": ((1, (1, 2)),)}, ValueError),
    ({"outputs": 0}, ValueError),
    ({"terms": ((33, (1,)),)}, ValueError),
    ({"terms": ((-130, (1,)),)}, ValueError),
    ({"budget": None}, TypeError),
])
def test_metadata_refuses_ambiguous_or_unbounded_inputs(kwargs, error) -> None:
    arguments = {"direction": (1,), "terms": ((1, (1,)),)}
    arguments.update(kwargs)
    with pytest.raises(error):
        RidgeDistribution(**arguments)


def test_operator_mismatch_and_result_budget_refusals() -> None:
    field = RidgeDistribution((1,), ((-1, (1,)),), budget=RidgeBudget(max_order=0))
    derivative = RidgeSystem.from_operators(((OperatorSpec.mixed((1,)),),), (1,))
    with pytest.raises(ValueError, match="order budget"):
        field.apply(derivative)
    with pytest.raises(ValueError, match="direction"):
        field.apply(RidgeSystem.from_operators(((OperatorSpec.mixed((1,)),),), (2,)))
    with pytest.raises(TypeError):
        field.apply(None)
    with pytest.raises(ValueError, match="bit budget"):
        RidgeDistribution((1,), ((1, (2**100,)),), budget=RidgeBudget(max_coefficient_bits=8))


def test_cluster_moments_and_uniform_bound_are_exact_rationals() -> None:
    collapse = collapse_dirac_cluster((1,), (-F(1, 100), F(1, 100)), (1, 1),
                                      max_order=3, min_probe_std=F(1, 5))
    assert collapse.field.terms == ((-3, (F(1, 10_000),)), (-1, (F(2),)))
    # Fourth Gaussian derivative <=3/(2 sigma^5); two remainder terms h^4/24.
    assert collapse.weak_error_bound == F(1, 256_000)
    twice = collapse_dirac_cluster((1,), (-F(1, 50), F(1, 50)), (1, 1),
                                   max_order=3, min_probe_std=F(1, 5))
    assert twice.weak_error_bound == 16*collapse.weak_error_bound


def test_zero_spread_and_signed_strength_collapse() -> None:
    zero = collapse_dirac_cluster((2,), (0, 0), (2, -2), min_probe_std=1)
    assert zero.field.terms == () and zero.weak_error_bound == 0
    derivative = collapse_dirac_cluster((1,), (-F(1, 10), F(1, 10)), (-5, 5),
                                        max_order=1, min_probe_std=1)
    assert derivative.field.terms == ((-2, (F(1),)),)


@pytest.mark.parametrize("kwargs", [
    {"min_probe_std": 0}, {"min_probe_std": -1}, {"min_probe_std": 0.1},
    {"offsets": ()}, {"strengths": ()}, {"max_order": -1}, {"max_order": 129},
    {"offsets": (0.1,)}, {"strengths": (1.0,)},
])
def test_cluster_refusals(kwargs) -> None:
    arguments = {"direction": (1,), "offsets": (0,), "strengths": (1,), "min_probe_std": 1}
    arguments.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        collapse_dirac_cluster(**arguments)
