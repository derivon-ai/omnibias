# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact solution-error accounting for positive 1-D Poisson source merging.

For -u''=sum a_i delta(x-s_i), u(left)=u(right)=0, the Green kernel is
G(x,s)=(min(x,s)-left)-(x-left)*(s-left)/(right-left). Replacing a positive
cluster by its total mass at its barycenter c preserves the field outside
the cluster hull. The replacement minus original is nonnegative, increases
up to c and decreases after c. Its EXACT sup norm is

    sum_(s_i<c) a_i*(c-s_i) = sum_i a_i*abs(s_i-c)/2.

Contiguous clusters have disjoint error supports, so their global sup error
is the maximum cluster error. This supplies a finite rational replay check,
not a certificate of floating-point evaluation or a theorem-prover build.

Cluster error is monotone under extending either end. Greedily taking the
longest feasible prefix minimizes the number of CONTIGUOUS, MASS/MEAN-
PRESERVING positive clusters at the requested tolerance. It does not solve
unrestricted optimal spline approximation. Green functions, convex splines
and particle/moment merging are classical; no novelty claim is made here.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from fractions import Fraction

from omnibias.core.ridge_system import _exact, _integer

Rational = int | Fraction


@dataclass(frozen=True)
class SourceBudget:
    """Finite source count, rational bit growth and host-operation budgets."""

    max_sources: int = 100_000
    max_coefficient_bits: int = 8192
    max_work: int = 16_000_000

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _integer(getattr(self, name), name, 1))

    def coefficient(self, value: Fraction) -> Fraction:
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > self.max_coefficient_bits:
            raise ValueError("source coefficient exceeds the exact bit budget")
        return value

    def work(self, count: int) -> None:
        if count > self.max_work:
            raise ValueError("source work exceeds the resource budget")


@dataclass(frozen=True)
class SourceCluster:
    """A half-open index block of the normalized source array and its error."""

    start: int
    stop: int
    center: Fraction
    mass: Fraction
    max_error: Fraction


@dataclass(frozen=True)
class PositivePoissonSources:
    """Positive rational point sources strictly inside a rational interval.

    Input order is arbitrary; duplicate positions are combined. Zero and
    negative masses are refused. The empty measure is supported. Prefix
    sums permit exact O(log N) value and block-error queries after sorting.
    Arithmetic counts are not bit-complexity or wall-clock guarantees.
    """

    points: tuple[Fraction, ...]
    masses: tuple[Fraction, ...]
    left: Fraction = Fraction(0)
    right: Fraction = Fraction(1)
    budget: SourceBudget = SourceBudget()
    _mass_prefix: tuple[Fraction, ...] = field(init=False, repr=False)
    _moment_prefix: tuple[Fraction, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.budget, SourceBudget):
            raise TypeError("budget must be a SourceBudget")
        if len(self.points) != len(self.masses):
            raise ValueError("points and masses must have matching lengths")
        if len(self.points) > self.budget.max_sources:
            raise ValueError("source count exceeds the resource budget")
        self.budget.work(len(self.points)*(len(self.points).bit_length()+16))
        check = self.budget.coefficient
        left, right = check(_exact(self.left)), check(_exact(self.right))
        if left >= right:
            raise ValueError("require left < right")
        rows: dict[Fraction, Fraction] = {}
        for point, mass in zip(self.points, self.masses, strict=True):
            point, mass = check(_exact(point)), check(_exact(mass))
            if not left < point < right or mass <= 0:
                raise ValueError("sources must be interior and masses strictly positive")
            rows[point] = check(rows.get(point, Fraction(0))+mass)
        points = tuple(sorted(rows))
        masses = tuple(rows[point] for point in points)
        pm, ps = [Fraction(0)], [Fraction(0)]
        for point, mass in zip(points, masses, strict=True):
            pm.append(check(pm[-1]+mass))
            ps.append(check(ps[-1]+check(point*mass)))
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "masses", masses)
        object.__setattr__(self, "_mass_prefix", tuple(pm))
        object.__setattr__(self, "_moment_prefix", tuple(ps))

    def value(self, x: Rational) -> Fraction:
        """Exact continuous Green solution at a rational x in [left,right]."""
        check = self.budget.coefficient
        x = check(_exact(x))
        if not self.left <= x <= self.right:
            raise ValueError("query must lie in the source interval")
        split = bisect_right(self.points, x)
        mass, moment = self._mass_prefix, self._moment_prefix
        shifted_left = check(moment[split]-check(self.left*mass[split]))
        shifted_total = check(moment[-1]-check(self.left*mass[-1]))
        tail_mass = check(mass[-1]-mass[split])
        value = check(shifted_left+check((x-self.left)*tail_mass))
        return check(value-check((x-self.left)*shifted_total/(self.right-self.left)))

    def cluster(self, start: int, stop: int) -> SourceCluster:
        """Exact sharp block error, with a binary search for its barycenter."""
        start, stop = _integer(start, "start"), _integer(stop, "stop")
        if not 0 <= start < stop <= len(self.points):
            raise ValueError("require a nonempty source block")
        check = self.budget.coefficient
        mass = check(self._mass_prefix[stop]-self._mass_prefix[start])
        center = check((self._moment_prefix[stop]-self._moment_prefix[start])/mass)
        split = bisect_left(self.points, center, start, stop)
        return self._cluster_at_split(start, stop, center, mass, split)

    def _cluster_at_split(
        self, start: int, stop: int, center: Fraction, mass: Fraction, split: int,
    ) -> SourceCluster:
        check = self.budget.coefficient
        left_mass = check(self._mass_prefix[split]-self._mass_prefix[start])
        left_moment = check(self._moment_prefix[split]-self._moment_prefix[start])
        error = check(check(center*left_mass)-left_moment)
        return SourceCluster(start, stop, center, mass, error)


@dataclass(frozen=True)
class PoissonCompression:
    """Replayable source merge and its sharp real-arithmetic solution error.

    Direct construction alone proves nothing; call ``verify()``. The finite
    checker replays partition coverage, masses, centers, errors, tolerance and
    maximal-prefix choices. Its interpretation uses the analytic Green identity
    in this module. It is not a Lean build or an enclosure of tensor rounding.
    """

    original: PositivePoissonSources
    compressed: PositivePoissonSources
    clusters: tuple[SourceCluster, ...]
    tolerance: Fraction
    max_error: Fraction

    def verify(self) -> bool:
        """Check the exact merge, sharp error and stated greedy optimality scope."""
        if not isinstance(self.original, PositivePoissonSources) or not isinstance(
            self.compressed, PositivePoissonSources
        ):
            return False
        if not isinstance(self.tolerance, Fraction) or not isinstance(self.max_error, Fraction):
            return False
        if self.tolerance < 0 or (self.original.left, self.original.right) != (
            self.compressed.left, self.compressed.right
        ):
            return False
        if len(self.clusters) != len(self.compressed.points):
            return False
        self.original.budget.work(len(self.clusters)*(len(self.original.points).bit_length()+32))
        cursor = 0
        errors = []
        for index, cluster in enumerate(self.clusters):
            if not isinstance(cluster, SourceCluster):
                return False
            if type(cluster.start) is not int or type(cluster.stop) is not int:
                return False
            if not all(isinstance(v, Fraction) for v in (cluster.center, cluster.mass, cluster.max_error)):
                return False
            if cluster.start != cursor or not cursor < cluster.stop <= len(self.original.points):
                return False
            try:
                expected = self.original.cluster(cursor, cluster.stop)
            except (TypeError, ValueError):
                return False
            if cluster != expected or cluster.max_error > self.tolerance:
                return False
            if self.compressed.points[index] != cluster.center or self.compressed.masses[index] != cluster.mass:
                return False
            if cluster.stop < len(self.original.points) and self.original.cluster(
                cursor, cluster.stop+1
            ).max_error <= self.tolerance:
                return False
            cursor = cluster.stop
            errors.append(cluster.max_error)
        return cursor == len(self.original.points) and self.max_error == max(errors, default=Fraction(0))


def compress_poisson_sources(
    sources: PositivePoissonSources, tolerance: Rational,
) -> PoissonCompression:
    """Greedy minimum-count contiguous barycenter merges at sharp sup error.

    After the source object is normalized, construction uses a monotone split
    pointer and O(N) rational operations, before independent replay. Positivity
    is essential for monotonicity, locality and maximum rather than sum of
    block errors. This does not assert optimality among arbitrary spline spaces.
    """
    if not isinstance(sources, PositivePoissonSources):
        raise TypeError("sources must be PositivePoissonSources")
    check = sources.budget.coefficient
    tolerance = check(_exact(tolerance))
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    sources.budget.work(64*len(sources.points))
    clusters: list[SourceCluster] = []
    start, split = 0, 0
    accepted: SourceCluster | None = None
    for stop in range(1, len(sources.points)+1):
        mass = check(sources._mass_prefix[stop]-sources._mass_prefix[start])
        center = check((sources._moment_prefix[stop]-sources._moment_prefix[start])/mass)
        while split < stop and sources.points[split] < center:
            split += 1
        candidate = sources._cluster_at_split(start, stop, center, mass, split)
        if candidate.max_error > tolerance:
            assert accepted is not None  # A singleton has exactly zero error.
            clusters.append(accepted)
            start, split = stop-1, stop-1
            accepted = SourceCluster(start, stop, sources.points[start], sources.masses[start], Fraction(0))
        else:
            accepted = candidate
    if accepted is not None:
        clusters.append(accepted)
    compressed = PositivePoissonSources(tuple(c.center for c in clusters), tuple(c.mass for c in clusters),
                                        sources.left, sources.right, sources.budget)
    return PoissonCompression(sources, compressed, tuple(clusters), tolerance,
                              max((c.max_error for c in clusters), default=Fraction(0)))


__all__ = ["PoissonCompression", "PositivePoissonSources", "SourceBudget", "SourceCluster",
           "compress_poisson_sources"]
