# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Differentiable proposal and exact acceptance for Viro patchworks.

The discrete and convex packages are used only as optional proposers.  A hit is
accepted by :func:`patchwork_curve` plus an exact rational regular-height
certificate.  Direct algebraic realization is a separate gate: an explicit
``F_t`` must replay through the complex-smoothness and whole-boundary verifier.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from omnibias.core.proof.catalog import CatalogEntry, register_catalog
from omnibias.core.proof.discovery import (
    ExactCheck,
    FiniteFamily,
    OneHotAnneal,
    Statement,
    run_discovery,
)
from omnibias.core.realization.polynomial import SparsePolynomial
from omnibias.geometry.algebraic import (
    BarrierAnnulus,
    HomogeneousPlaneCurve,
    ProjectiveCurveCertificate,
    certify_curve,
    find_smoothness_witness,
    replay_curve_certificate,
)
from omnibias.geometry.part_a_target import (
    sibling_octic_region_tree,
    wide_deep_octic_region_tree,
)
from omnibias.geometry.patchwork import (
    RootedTree,
    SignDistribution,
    Triangulation,
    flip_neighbors,
    patchwork_curve,
    staircase_triangulation,
    target_octic_region_tree,
)
from omnibias.geometry.patchwork_height_lp import (
    RegularHeightCertificate,
    certify_regular_heights,
    clear_denominators,
    propose_regular_heights,
    verify_regular_height_certificate,
)
from omnibias.geometry.patchwork_symmetry import canonical_sign_orbit, sign_orbit_representatives

PATCHWORK_OCTIC_KIND = "hilbert16_patchwork_octic"


@dataclass(frozen=True)
class PatchworkCandidate:
    """A triangulation-bank index and one sign bit per Newton point."""

    triangulation_index: int
    bits: tuple[int, ...]


def statement_for_patchwork_target(target: RootedTree) -> Statement:
    """Name the rooted tree this family actually checks.

    The two open degree-eight schemes stay distinct. Any other tree is
    described as the family's declared target, without borrowing either
    octic scheme name.
    """
    if target == sibling_octic_region_tree():
        scheme = "14 + 1<2 + 1<4>>"
    elif target == wide_deep_octic_region_tree():
        scheme = "4 + 1<2 + 1<14>>"
    else:
        scheme = "the declared rooted region tree"
    return Statement(
        name="regular_patchwork_target_tree",
        obligation=f"a regular unimodular patchwork has rooted region tree {scheme}",
        parent="Hilbert XVI Part A selected octic scheme",
        parent_status="open",
    )


def _tree_edges(tree: RootedTree) -> int:
    return len(tree) + sum(_tree_edges(child) for child in tree)


def _tree_profile(tree: RootedTree, depth: int = 0) -> tuple[tuple[int, int], ...]:
    return ((depth, len(tree)),) + tuple(
        item
        for child in tree
        for item in _tree_profile(child, depth + 1)
    )


def _tree_score(actual: RootedTree, target: RootedTree) -> int:
    actual_profile = sorted(_tree_profile(actual))
    target_profile = sorted(_tree_profile(target))
    width = max(len(actual_profile), len(target_profile))
    padded_actual = actual_profile + [(99, 99)] * (width - len(actual_profile))
    padded_target = target_profile + [(99, 99)] * (width - len(target_profile))
    penalty = 100 * abs(_tree_edges(actual) - _tree_edges(target))
    penalty += sum(
        abs(a_depth - t_depth) + abs(a_degree - t_degree)
        for (a_depth, a_degree), (t_depth, t_degree) in zip(
            padded_actual,
            padded_target,
            strict=True,
        )
    )
    return -penalty


def triangulation_bank(
    degree: int,
    *,
    flip_depth: int = 1,
    limit: int = 64,
) -> tuple[Triangulation, ...]:
    """Build a bounded, exact, duplicate-free edge-flip bank."""
    if type(flip_depth) is not int or flip_depth < 0:
        raise ValueError("flip_depth must be a nonnegative integer")
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be positive")
    origin = staircase_triangulation(degree)
    queue = deque([(origin, 0)])
    output: list[Triangulation] = []
    seen: set[tuple[tuple[tuple[int, int], ...], ...]] = set()
    while queue and len(output) < limit:
        current, depth = queue.popleft()
        key = tuple(tuple(sorted(triangle)) for triangle in current.triangles)
        if key in seen:
            continue
        seen.add(key)
        output.append(current)
        if depth < flip_depth:
            queue.extend((neighbor, depth + 1) for neighbor in flip_neighbors(current))
    return tuple(output)


def annealed_sign_seed(triangulation: Triangulation) -> tuple[int, ...]:
    """Use ``anneal_descent`` on the nonmonochromatic-triangle surrogate."""
    import torch
    from omnibias.discrete import AnnealSchedule
    from omnibias.discrete.torch import anneal_descent

    points = triangulation.vertices
    point_index = {point: i for i, point in enumerate(points)}
    triangles = tuple(
        tuple(point_index[point] for point in triangle)
        for triangle in triangulation.triangles
    )

    def gradient(x):
        result = torch.zeros_like(x)
        for i, j, k in triangles:
            result[i] += x[j] * x[k] - (1 - x[j]) * (1 - x[k])
            result[j] += x[i] * x[k] - (1 - x[i]) * (1 - x[k])
            result[k] += x[i] * x[j] - (1 - x[i]) * (1 - x[j])
        return result

    incidence = max(
        sum(index in triangle for triangle in triangles)
        for index in range(len(points))
    )
    relaxed = anneal_descent(
        gradient,
        max(1, 4 * incidence),
        len(points),
        schedule=AnnealSchedule.fast(),
    )
    return tuple(int(value >= 0.5) for value in relaxed.detach().cpu().tolist())


def csp_sign_seed(
    triangulation: Triangulation,
    *,
    seed: int = 0,
) -> tuple[int, ...]:
    """Use soft arc consistency and two-temperature CSP descent as proposer."""
    from omnibias.discrete import AnnealSchedule
    from omnibias.discrete.csp import CSP, Relation, Variable, csp_solve

    points = triangulation.vertices
    point_index = {point: i for i, point in enumerate(points)}
    variables = tuple(Variable(f"s{i}", (0, 1)) for i in range(len(points)))
    allowed = frozenset(
        bits
        for bits in (
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            (0, 1, 1),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (1, 1, 1),
        )
        if bits not in ((0, 0, 0), (1, 1, 1))
    )
    relations = tuple(
        Relation(
            tuple(point_index[point] for point in triangle),
            allowed,
        )
        for triangle in triangulation.triangles
    )
    csp = CSP(variables, relations, name="patchwork_nonmonochromatic_triangles")
    result = csp_solve(
        csp,
        simplex_schedule=AnnealSchedule.fast(),
        clause_schedule=AnnealSchedule.fast(),
        steps=12,
        restarts=2,
        seed=seed,
    )
    return result.assignment


@dataclass(frozen=True)
class PatchworkSeedReport:
    """The two differentiable proposals and the exact-topology winner."""

    annealed_bits: tuple[int, ...]
    csp_bits: tuple[int, ...]
    selected_bits: tuple[int, ...]
    annealed_score: int
    csp_score: int


def optimized_sign_seed(
    triangulation: Triangulation,
    target: RootedTree,
    *,
    seed: int = 0,
) -> PatchworkSeedReport:
    """Run both differentiable proposers and select by exact tree score."""
    annealed = annealed_sign_seed(triangulation)
    csp = csp_sign_seed(triangulation, seed=seed)

    def score(bits: tuple[int, ...]) -> int:
        result = patchwork_curve(
            triangulation,
            SignDistribution(triangulation.degree, bits),
        )
        return _tree_score(result.rooted_tree, target)

    annealed_score, csp_score = score(annealed), score(csp)
    selected = annealed if annealed_score >= csp_score else csp
    return PatchworkSeedReport(
        annealed,
        csp,
        selected,
        annealed_score,
        csp_score,
    )


def _regularity_for_candidate(
    triangulation: Triangulation,
) -> RegularHeightCertificate:
    explicit = {
        point: point[0] ** 2 + point[1] ** 2 + point[0] * point[1]
        for point in triangulation.vertices
    }
    try:
        return certify_regular_heights(triangulation, explicit)
    except ValueError:
        return propose_regular_heights(triangulation)


@dataclass
class PatchworkSearchFamily(FiniteFamily):
    """Bounded sign/triangulation search with an exact target-tree gate."""

    triangulations: tuple[Triangulation, ...]
    seed_bits: tuple[int, ...]
    target: RootedTree
    name: str = "hilbert16_octic_patchwork"
    complete: bool = False
    statement: Statement = field(
        default_factory=lambda: Statement(
            name="regular_patchwork_target_tree",
            obligation=(
                "a regular unimodular patchwork has the family's declared "
                "rooted region tree"
            ),
            parent="Hilbert XVI Part A selected octic scheme",
            parent_status="open",
        )
    )

    def __post_init__(self) -> None:
        if self.complete:
            raise ValueError("PatchworkSearchFamily.complete stays false")
        if not self.triangulations:
            raise ValueError("at least one triangulation is required")
        degree = self.triangulations[0].degree
        if any(item.degree != degree for item in self.triangulations):
            raise ValueError("all triangulations must have the same degree")
        SignDistribution(degree, self.seed_bits)
        self.statement = statement_for_patchwork_target(self.target)

    def origin(self) -> PatchworkCandidate:
        return PatchworkCandidate(0, self.seed_bits)

    def neighbors(self, candidate: Any) -> Sequence[PatchworkCandidate]:
        if not isinstance(candidate, PatchworkCandidate):
            return ()
        bit_neighbors = tuple(
            PatchworkCandidate(
                candidate.triangulation_index,
                candidate.bits[:i] + (1 - bit,) + candidate.bits[i + 1:],
            )
            for i, bit in enumerate(candidate.bits)
        )
        triangulation_neighbors = tuple(
            PatchworkCandidate(index, candidate.bits)
            for index in range(len(self.triangulations))
            if index != candidate.triangulation_index
        )
        return (*bit_neighbors, *triangulation_neighbors)

    def score(self, candidate: Any) -> int:
        if not isinstance(candidate, PatchworkCandidate):
            return -10**9
        try:
            triangulation = self.triangulations[candidate.triangulation_index]
            result = patchwork_curve(
                triangulation,
                SignDistribution(triangulation.degree, candidate.bits),
            )
        except (ArithmeticError, IndexError, TypeError, ValueError):
            return -10**9
        return _tree_score(result.rooted_tree, self.target)

    def check(self, candidate: Any) -> ExactCheck | None:
        if not isinstance(candidate, PatchworkCandidate):
            return None
        try:
            triangulation = self.triangulations[candidate.triangulation_index]
            signs = SignDistribution(triangulation.degree, candidate.bits)
            result = patchwork_curve(triangulation, signs)
        except (ArithmeticError, IndexError, TypeError, ValueError):
            return None
        if result.rooted_tree != self.target:
            return ExactCheck(
                False,
                {
                    "component_count": result.component_count,
                    "honesty": {
                        "target_region_tree_exact": False,
                        "search_family_complete": False,
                        "full_hilbert16_solved": False,
                    },
                },
            )
        regularity = _regularity_for_candidate(triangulation)
        return ExactCheck(
            True,
            {
                "triangulation_index": candidate.triangulation_index,
                "triangles": [
                    [list(point) for point in triangle]
                    for triangle in triangulation.triangles
                ],
                "bits": list(candidate.bits),
                "component_count": result.component_count,
                "regular_height_digest": regularity.feasibility.seal["digest"],
                "honesty": {
                    "target_region_tree_exact": True,
                    "regular_height_exact_q": True,
                    "algebraic_polynomial_directly_certified": False,
                    "search_family_complete": False,
                    "full_hilbert16_solved": False,
                },
            },
        )

    def cardinality(self) -> int | None:
        return len(self.triangulations) * (1 << len(self.seed_bits))


def symmetry_reduced_sign_candidates(
    triangulation: Triangulation,
    bits: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """Enumerate sign-bit candidates modulo the `(±x, ±y)` symmetry group."""
    orbit_bits = {canonical_sign_orbit(SignDistribution(triangulation.degree, candidate)) for candidate in (
        bits,
        tuple(reversed(bits)),
    )}
    representatives = sign_orbit_representatives(
        tuple(SignDistribution(triangulation.degree, candidate) for candidate in orbit_bits)
    )
    return tuple(sign.bits for sign in representatives)


def run_patchwork_discovery(
    *,
    degree: int = 8,
    budget: int = 64,
    flip_depth: int = 1,
    triangulation_limit: int = 64,
    seed: int = 0,
    target: RootedTree | None = None,
    triangulations: tuple[Triangulation, ...] | None = None,
) -> dict[str, Any]:
    """Run the two differentiable seeds, then the exact discovery walk.

    ``triangulations`` reuses a bank already built for this degree. The
    exact gate is unchanged: ``patchwork_curve`` plus rational regular heights.
    """
    if triangulations is None:
        bank = triangulation_bank(
            degree,
            flip_depth=flip_depth,
            limit=triangulation_limit,
        )
    else:
        if not triangulations:
            raise ValueError("triangulations must be non-empty")
        if any(item.degree != degree for item in triangulations):
            raise ValueError("triangulations must match degree")
        bank = tuple(triangulations)
    if target is None:
        target = target_octic_region_tree() if degree == 8 else tuple(() for _ in range(4))
    seed_report = optimized_sign_seed(bank[0], target, seed=seed)
    reduced_bits = symmetry_reduced_sign_candidates(bank[0], seed_report.selected_bits)
    family = PatchworkSearchFamily(bank, reduced_bits[0], target)
    result = run_discovery(
        family.statement,
        family,
        OneHotAnneal(seed=seed),
        budget=budget,
    )
    return {
        "seed": seed_report,
        "symmetry_reduced_seed_count": len(reduced_bits),
        "triangulation_count": len(bank),
        "result": result,
    }


def patchwork_polynomial(
    triangulation: Triangulation,
    signs: SignDistribution,
    regularity: RegularHeightCertificate,
    *,
    t: int | Fraction,
) -> HomogeneousPlaneCurve:
    """Form the explicit homogeneous rational ``F_t`` from a regular patchwork."""
    if isinstance(t, bool) or not isinstance(t, int | Fraction):
        raise TypeError("t must be an exact int or Fraction, not a float")
    parameter = Fraction(t)
    if not 0 < parameter < 1:
        raise ValueError("t must be an exact rational strictly between zero and one")
    if signs.degree != triangulation.degree:
        raise ValueError("signs and triangulation must have the same degree")
    if (
        regularity.triangulation != triangulation
        or not verify_regular_height_certificate(regularity)
    ):
        raise ValueError("regular-height certificate does not replay for this triangulation")
    integer_heights = clear_denominators(regularity.heights)
    shift = min(integer_heights)
    exponents = tuple(value - shift for value in integer_heights)
    terms = {
        (point[0], point[1], triangulation.degree - point[0] - point[1]): (
            (-1 if signs.bit(point) else 1) * parameter**height
        )
        for point, height in zip(triangulation.vertices, exponents, strict=True)
    }
    return HomogeneousPlaneCurve(SparsePolynomial(3, terms))


def _parents_tree(parents: tuple[int, ...]) -> RootedTree:
    children: list[list[int]] = [[] for _ in range(len(parents) + 1)]
    for index, parent in enumerate(parents):
        children[0 if parent < 0 else parent + 1].append(index + 1)

    def signature(node: int) -> RootedTree:
        return tuple(sorted((signature(child) for child in children[node]), key=repr))

    return signature(0)


@dataclass(frozen=True)
class PatchworkRealizationCertificate:
    """A direct coefficient-level acceptance of an exact patchwork candidate."""

    triangulation: Triangulation
    signs: SignDistribution
    regularity: RegularHeightCertificate
    t: Fraction
    max_multiplier_degree: int
    curve: HomogeneousPlaneCurve
    curve_certificate: ProjectiveCurveCertificate
    target: RootedTree


def certify_patchwork_realization(
    triangulation: Triangulation,
    signs: SignDistribution,
    regularity: RegularHeightCertificate,
    annuli: Sequence[BarrierAnnulus],
    *,
    t: int | Fraction,
    target: RootedTree,
    max_multiplier_degree: int = 12,
    fixed_axis: int = 2,
    subdivision_depth: int = 8,
) -> PatchworkRealizationCertificate:
    """Require both the exact PL tree and direct coefficient-level realization."""
    result = patchwork_curve(triangulation, signs)
    if result.rooted_tree != target:
        raise ValueError("patchwork rooted tree does not match the declared target")
    curve = patchwork_polynomial(triangulation, signs, regularity, t=t)
    witness = find_smoothness_witness(
        curve,
        max_multiplier_degree=max_multiplier_degree,
    )
    if witness is None:
        raise ValueError("explicit F_t lacks a bounded complex-smoothness witness")
    certificate = certify_curve(
        curve,
        witness,
        annuli,
        fixed_axis=fixed_axis,
        subdivision_depth=subdivision_depth,
    )
    if not certificate.complete_real_scheme or _parents_tree(certificate.barrier_parents) != target:
        raise ValueError("direct polynomial certificate does not realize the target tree")
    return PatchworkRealizationCertificate(
        triangulation,
        signs,
        regularity,
        Fraction(t),
        max_multiplier_degree,
        curve,
        certificate,
        target,
    )


def verify_patchwork_realization(
    certificate: PatchworkRealizationCertificate,
) -> bool:
    """Replay exact regularity, PL topology, coefficients, and curve certificate."""
    if (
        not verify_regular_height_certificate(certificate.regularity)
        or not replay_curve_certificate(certificate.curve, certificate.curve_certificate)
    ):
        return False
    try:
        expected = certify_patchwork_realization(
            certificate.triangulation,
            certificate.signs,
            certificate.regularity,
            certificate.curve_certificate.annuli,
            t=certificate.t,
            target=certificate.target,
            max_multiplier_degree=certificate.max_multiplier_degree,
            fixed_axis=certificate.curve_certificate.fixed_axis,
            subdivision_depth=certificate.curve_certificate.subdivision_depth,
        )
    except (ArithmeticError, TypeError, ValueError):
        return False
    return expected == certificate


def _catalog_factory(**kwargs: Any) -> dict[str, Any]:
    return run_patchwork_discovery(**kwargs)


register_catalog(
    CatalogEntry(
        kind=PATCHWORK_OCTIC_KIND,
        obligation="regular unimodular patchwork with the declared 22-oval region tree",
        parent="Hilbert XVI Part A selected octic scheme",
        parent_status="open",
        package="omnibias.geometry.patchwork_search",
        mode="exact_search",
        complete=False,
        existential=True,
    ),
    _catalog_factory,
)


__all__ = [
    "PATCHWORK_OCTIC_KIND",
    "PatchworkCandidate",
    "PatchworkRealizationCertificate",
    "PatchworkSearchFamily",
    "PatchworkSeedReport",
    "annealed_sign_seed",
    "certify_patchwork_realization",
    "csp_sign_seed",
    "optimized_sign_seed",
    "patchwork_polynomial",
    "run_patchwork_discovery",
    "statement_for_patchwork_target",
    "symmetry_reduced_sign_candidates",
    "triangulation_bank",
    "verify_patchwork_realization",
]
