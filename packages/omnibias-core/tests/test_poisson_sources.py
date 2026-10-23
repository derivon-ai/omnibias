# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent Green-kernel, knot-extrema and exhaustive partition checks."""

from dataclasses import replace
from fractions import Fraction as F
from itertools import product
from random import Random

import pytest
from omnibias.core.operator_spec import OperatorSpec
from omnibias.core.poisson_sources import (
    PositivePoissonSources,
    SourceBudget,
    compress_poisson_sources,
)
from omnibias.core.ridge_distribution import RidgeDistribution, collapse_dirac_cluster
from omnibias.core.ridge_system import RidgeSystem


def _green(x, s, left=F(0), right=F(1)):
    # Product form, independently of the production prefix expression.
    return (min(x, s)-left)*(right-max(x, s))/(right-left)


def _direct(sources, x):
    return sum((a*_green(x, s, sources.left, sources.right)
                for s, a in zip(sources.points, sources.masses, strict=True)), F(0))


def _block_error(points, masses):
    mass = sum(masses, F(0))
    center = sum((a*s for s, a in zip(points, masses, strict=True)), F(0))/mass
    # The difference is piecewise affine; its extrema are among these knots.
    return max(abs(mass*_green(x, center)-sum(
        (a*_green(x, s) for s, a in zip(points, masses, strict=True)), F(0)))
               for x in (*points, center, F(0), F(1)))


def test_symmetric_pair_has_sharp_solution_error_not_gaussian_weak_error() -> None:
    h = F(1, 100)
    sources = PositivePoissonSources((F(1, 2)-h, F(1, 2)+h), (1, 1))
    report = compress_poisson_sources(sources, h)
    assert report.verify()
    assert report.compressed.points == (F(1, 2),)
    assert report.compressed.masses == (F(2),)
    assert report.max_error == h
    assert report.compressed.value(F(1, 2))-sources.value(F(1, 2)) == h
    for x in (F(0), F(1, 4), F(3, 4), F(1)):
        assert report.compressed.value(x) == sources.value(x)
    assert len(compress_poisson_sources(sources, h/2).clusters) == 2


def test_four_moment_weak_source_collapse_can_have_a_singular_poisson_inverse() -> None:
    h = F(1, 100)
    source = collapse_dirac_cluster((1,), (-h, h), (1, 1), min_probe_std=F(1, 5),
                                    max_order=3, bias=-F(1, 2)).field
    # Up to an affine boundary correction, the inverse is -2 ReLU(z)-h² delta(z).
    # Retaining the singular term is essential: weak source accuracy alone
    # gives no maximum-norm error for a finite-valued solution.
    inverse = RidgeDistribution((1,), ((1, (-2,)), (-1, (-h*h,))), bias=-F(1, 2))
    negative_laplacian = RidgeSystem.from_operators(((OperatorSpec.polynomial({(2,): -1}),),), (1,))
    assert inverse.apply(negative_laplacian).terms == source.terms
    assert inverse.singular_terms == ((-1, (-h*h,)),)


def test_unsorted_duplicates_nonunit_domain_and_boundary_values() -> None:
    sources = PositivePoissonSources((1, -1, 1), (2, 3, 4), left=-2, right=3)
    assert sources.points == (F(-1), F(1)) and sources.masses == (F(3), F(6))
    for x in (F(-2), F(-3, 2), F(0), F(7, 3), F(3)):
        assert sources.value(x) == _direct(sources, x)
    assert sources.value(-2) == sources.value(3) == 0


def test_slope_jumps_match_the_full_distributional_pde() -> None:
    sources = PositivePoissonSources((F(1, 4), F(3, 5)), (F(2), F(5, 2)))
    h = F(1, 1000)
    for s, a in zip(sources.points, sources.masses, strict=True):
        left_slope = (sources.value(s)-sources.value(s-h))/h
        right_slope = (sources.value(s+h)-sources.value(s))/h
        assert right_slope-left_slope == -a  # -u'' = a delta_s.


def test_global_sharp_error_and_locality_on_random_rational_sources() -> None:
    rng = Random(19)
    for _ in range(30):
        points = tuple(F(s, 101) for s in sorted(rng.sample(range(1, 101), 14)))
        masses = tuple(F(rng.randint(1, 8), 20) for _ in points)
        sources = PositivePoissonSources(points, masses)
        report = compress_poisson_sources(sources, F(1, 40))
        assert report.verify()
        knots = sorted(set((F(0), F(1), *sources.points, *report.compressed.points)))
        differences = [report.compressed.value(x)-_direct(sources, x) for x in knots]
        assert min(differences) >= 0
        assert max(differences) == report.max_error <= report.tolerance
        assert sum(report.compressed.masses) == sum(masses)
        assert sum(a*s for a, s in zip(report.compressed.masses, report.compressed.points, strict=True)) == sum(
            a*s for a, s in zip(masses, points, strict=True))


def test_greedy_cluster_count_matches_exhaustive_partitions() -> None:
    rng = Random(41)
    for _ in range(25):
        n = rng.randint(2, 7)
        points = tuple(F(s, 41) for s in sorted(rng.sample(range(1, 41), n)))
        masses = tuple(F(rng.randint(1, 5), 10) for _ in points)
        tolerance = F(rng.randint(1, 8), 100)
        best = n
        for cuts in product((False, True), repeat=n-1):
            endpoints = (0, *(i+1 for i, cut in enumerate(cuts) if cut), n)
            errors = [_block_error(points[a:b], masses[a:b])
                      for a, b in zip(endpoints, endpoints[1:], strict=False)]
            if max(errors) <= tolerance:
                best = min(best, len(endpoints)-1)
        report = compress_poisson_sources(PositivePoissonSources(points, masses), tolerance)
        assert report.verify()
        assert len(report.clusters) == best


def test_domain_scaling_and_mass_scaling_preserve_partition() -> None:
    points, masses = (F(1, 5), F(1, 4), F(3, 5), F(4, 5)), (1, 2, 1, 3)
    original = compress_poisson_sources(PositivePoissonSources(points, masses), F(1, 5))
    scaled = compress_poisson_sources(PositivePoissonSources(tuple(3*x-2 for x in points),
                                                            tuple(5*a for a in masses), -2, 1), F(3))
    assert [(c.start, c.stop) for c in scaled.clusters] == [(c.start, c.stop) for c in original.clusters]
    assert scaled.max_error == 15*original.max_error


def test_empty_zero_tolerance_and_tamper_refusal() -> None:
    empty = compress_poisson_sources(PositivePoissonSources((), ()), 0)
    assert empty.verify() and empty.max_error == 0 and empty.compressed.value(F(1, 2)) == 0
    sources = PositivePoissonSources((F(1, 3), F(2, 3)), (1, 1))
    unchanged = compress_poisson_sources(sources, 0)
    assert unchanged.verify() and len(unchanged.clusters) == 2
    merged = compress_poisson_sources(sources, F(1))
    assert merged.verify()
    assert not replace(merged, max_error=F(0)).verify()
    assert not replace(merged, tolerance=F(0)).verify()
    assert not replace(merged, clusters=(replace(merged.clusters[0], center=F(1, 4)),)).verify()
    assert not replace(merged, clusters=(replace(merged.clusters[0], stop="2"),)).verify()
    assert not replace(merged, clusters=(replace(merged.clusters[0], center=0.5),)).verify()
    assert not replace(unchanged, tolerance=F(1)).verify()  # Not a maximal-prefix construction.


@pytest.mark.parametrize("kwargs", [
    {"points": (0,)}, {"points": (1,)}, {"points": (0.5,)}, {"masses": (1.0,)},
    {"masses": (-1,)}, {"masses": (0,)}, {"masses": ()}, {"left": 1},
    {"budget": SourceBudget(max_coefficient_bits=2), "masses": (F(1, 100),)},
])
def test_source_refusals(kwargs) -> None:
    arguments = {"points": (F(1, 2),), "masses": (1,)}
    arguments.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        PositivePoissonSources(**arguments)


def test_compression_budget_and_query_refusals() -> None:
    sources = PositivePoissonSources((F(1, 3), F(2, 3)), (1, 1), budget=SourceBudget(max_work=100))
    with pytest.raises(ValueError, match="work"):
        compress_poisson_sources(sources, 1)
    with pytest.raises(ValueError):
        compress_poisson_sources(sources, -1)
    with pytest.raises(TypeError):
        compress_poisson_sources(sources, 0.1)
    with pytest.raises(ValueError):
        sources.value(2)
    with pytest.raises(TypeError):
        sources.value(0.5)
