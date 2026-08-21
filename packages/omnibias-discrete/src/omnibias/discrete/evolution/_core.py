# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
r"""Soft-population evolution (theory 03-01).

Four narrow contributions on existing omnibias pieces, not a new EA family:

* softmax selection ``w = softmax(-beta E)`` with a closed-form
  ``log(P)/beta`` gap (reuses ``logsumexp_gap_bound``);
* geometry-aware mutation of pack genes (mean additive, spread
  log-normal, order discrete);
* exact-curvature memetic polish versus gradient-descent polish, with
  every fitness call counted;
* a certify-loop stop on ``DiscreteProblem`` (``certify_gap`` /
  negative-coeff floor). Never an exactness or P = NP claim.

Selection ``beta -> inf`` is **temperature collapse** (feasibility:
soft weights harden to an argmin). It is **not** the founding bias collapse
(``delta -> 0`` coalescence of ``K`` biases into ``sigma^(K-1)``),
which is the geometry the pack genes mutate. Do not conflate the two.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray
from omnibias.discrete._core.bound import negative_coeff_lower_bound
from omnibias.discrete._core.decode import brute_force_min
from omnibias.discrete._core.problem import DiscreteProblem
from omnibias.discrete._core.schedule import AnnealSchedule
from omnibias.discrete.certify import certify_gap

FloatArray = NDArray[np.float64]
BoundKind = Literal["negative_coeff", "certify_gap", "brute_force"]
PolishKind = Literal["newton", "gd"]

_GAP_ATOL = 1e-9


def _as_f64(x: object) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


def _rng(seed: int | None) -> Generator:
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class PopulationConfig:
    """Soft-population size, selection temperature, and elitism.

    ``selection_beta`` may be a scalar or an :class:`AnnealSchedule`
    (``beta -> inf`` temperature collapse of selection pressure).
    """

    size: int
    selection_beta: float | AnnealSchedule
    elitism: int = 1
    mutation_sigma: float = 0.15
    bit_flip_p: float = 0.25

    def __post_init__(self) -> None:
        if int(self.size) < 2:
            raise ValueError(f"size must be >= 2, got {self.size}")
        if int(self.elitism) < 0 or int(self.elitism) >= int(self.size):
            raise ValueError("elitism must satisfy 0 <= elitism < size")
        if self.mutation_sigma <= 0.0:
            raise ValueError("mutation_sigma must be > 0")
        if not 0.0 < float(self.bit_flip_p) <= 1.0:
            raise ValueError("bit_flip_p must be in (0, 1]")
        if isinstance(self.selection_beta, AnnealSchedule):
            return
        if float(self.selection_beta) < 0.0:
            raise ValueError("selection_beta must be >= 0")
        object.__setattr__(self, "size", int(self.size))
        object.__setattr__(self, "elitism", int(self.elitism))


@dataclass(frozen=True)
class GeometryMutation:
    """Pack-gene mutation. Spread is log-normal; order is discrete."""

    mean_sigma: float = 0.05
    spread_log_sigma: float = 0.3
    order_step_prob: float = 0.2
    weight_sigma: float = 0.1
    normal_rotation: float = 0.0
    order_min: int = 0
    order_max: int = 6


@dataclass(frozen=True)
class PackGenes:
    """One pack: sample location, collapse tightness, order, amplitude."""

    mu: float
    delta: float
    order: int
    weight: float

    def __post_init__(self) -> None:
        if float(self.delta) <= 0.0:
            raise ValueError("delta must be > 0 (pack spread)")
        if int(self.order) < 0:
            raise ValueError("order must be >= 0")
        object.__setattr__(self, "mu", float(self.mu))
        object.__setattr__(self, "delta", float(self.delta))
        object.__setattr__(self, "order", int(self.order))
        object.__setattr__(self, "weight", float(self.weight))

    def as_vector(self) -> FloatArray:
        return np.array([self.mu, self.delta, float(self.order), self.weight], dtype=np.float64)

    @classmethod
    def from_vector(cls, vec: object) -> PackGenes:
        v = _as_f64(vec).reshape(-1)
        if v.size != 4:
            raise ValueError("PackGenes vector must have length 4")
        return cls(float(v[0]), max(float(v[1]), 1e-12), int(round(float(v[2]))), float(v[3]))


@dataclass(frozen=True)
class GenerationRecord:
    """One generation: energies, soft mean, measured gap, certified bound."""

    generation: int
    beta: float
    energies: tuple[float, ...]
    e_soft: float
    e_best: float
    measured_gap: float
    bound: float
    n_eval: int


@dataclass
class EvolveResult:
    """Best member after a budgeted run. ``n_eval`` counts every fitness call."""

    best: FloatArray
    best_value: float
    n_eval: int
    records: list[GenerationRecord] = field(default_factory=list)
    gap_violations: int = 0


@dataclass(frozen=True)
class CertifiedEvolveResult:
    """Instance-wise certified gap. Not a complexity or P vs NP claim."""

    best_assignment: tuple[int, ...]
    best_value: float
    lower_bound: float
    gap: float
    gap_closed: bool
    generations_used: int
    n_eval: int
    bound_method: str
    records: tuple[GenerationRecord, ...]


def honesty_payload() -> dict[str, bool]:
    return {
        "unproven_claim": False,
        "p_equals_np_claim": False,
        "selection_is_temperature_collapse": True,
        "founding_bias_collapse_in_selection": False,
        "cma_es_beaten": False,
    }


def selection_gap_bound(*, population: int, beta: float) -> float:
    """``log(P) / beta``, reusing ``omnibias.struct.logsumexp_gap_bound``."""
    from omnibias.struct import logsumexp_gap_bound

    return float(logsumexp_gap_bound(int(population), float(beta)))


def soft_weights(energies: object, *, beta: float) -> FloatArray:
    """``w_i = softmax(-beta E_i)``. ``beta == 0`` is uniform."""
    e = _as_f64(energies).reshape(-1)
    if e.size < 1:
        raise ValueError("energies must be non-empty")
    b = float(beta)
    if b < 0.0:
        raise ValueError(f"beta must be >= 0, got {b}")
    if b == 0.0:
        return np.full(e.shape, 1.0 / float(e.size), dtype=np.float64)
    z = -b * e
    z = z - float(np.max(z))
    w = np.exp(z)
    return (w / float(np.sum(w))).astype(np.float64, copy=False)


def selection_stats(energies: object, *, beta: float) -> tuple[FloatArray, float, float, float]:
    """Weights, soft mean, best energy, measured soft-versus-best gap."""
    e = _as_f64(energies).reshape(-1)
    w = soft_weights(e, beta=beta)
    e_soft = float(np.dot(w, e))
    e_best = float(np.min(e))
    return w, e_soft, e_best, e_soft - e_best


def _beta_at(config: PopulationConfig, generation: int) -> float:
    raw = config.selection_beta
    if isinstance(raw, AnnealSchedule):
        betas = raw.betas()
        return float(betas[min(int(generation), len(betas) - 1)])
    return float(raw)


def _record(
    generation: int,
    energies: FloatArray,
    *,
    beta: float,
    n_eval: int,
) -> GenerationRecord:
    _w, e_soft, e_best, measured = selection_stats(energies, beta=beta)
    bound = selection_gap_bound(population=int(energies.size), beta=beta) if beta > 0.0 else float("inf")
    return GenerationRecord(
        generation=int(generation),
        beta=float(beta),
        energies=tuple(float(v) for v in energies.tolist()),
        e_soft=e_soft,
        e_best=e_best,
        measured_gap=measured,
        bound=bound,
        n_eval=int(n_eval),
    )


def mutate_geometry(pack: PackGenes, mut: GeometryMutation, rng: Generator) -> PackGenes:
    mu = pack.mu + float(rng.normal(0.0, mut.mean_sigma))
    delta = pack.delta * float(np.exp(rng.normal(0.0, mut.spread_log_sigma)))
    delta = max(delta, 1e-12)
    order = pack.order
    if float(rng.random()) < mut.order_step_prob:
        step = int(rng.choice(np.array([-1, 1], dtype=np.int64)))
        order = int(np.clip(order + step, mut.order_min, mut.order_max))
    weight = pack.weight + float(rng.normal(0.0, mut.weight_sigma))
    return PackGenes(mu, delta, order, weight)


def mutate_isotropic(pack: PackGenes, sigma: float, rng: Generator, *, mut: GeometryMutation) -> PackGenes:
    """Single-scale Gaussian on ``(mu, delta, order, weight)``. Destroys log-delta."""
    v = pack.as_vector() + rng.normal(0.0, float(sigma), size=4)
    v[1] = max(float(v[1]), 1e-12)
    v[2] = float(np.clip(round(float(v[2])), mut.order_min, mut.order_max))
    return PackGenes.from_vector(v)


def crossover_packs(a: Sequence[PackGenes], b: Sequence[PackGenes], rng: Generator) -> list[PackGenes]:
    """Exchange whole packs (representation-respecting crossover)."""
    if len(a) != len(b):
        raise ValueError("crossover parents must have the same number of packs")
    if not a:
        return []
    out: list[PackGenes] = []
    for left, right in zip(a, b, strict=True):
        out.append(left if float(rng.random()) < 0.5 else right)
    return out


def pack_fitness(pack: PackGenes, *, target: PackGenes) -> float:
    """Named pack-structure mismatch (log-spread + scaled mean / order / weight)."""
    return float(
        ((pack.mu - target.mu) / 0.2) ** 2
        + (np.log(pack.delta / target.delta)) ** 2
        + 4.0 * (pack.order - target.order) ** 2
        + (pack.weight - target.weight) ** 2
    )


def _select_indices(weights: FloatArray, n: int, rng: Generator) -> NDArray[np.intp]:
    return rng.choice(weights.size, size=n, replace=True, p=weights)


def soft_population_evolve(
    fitness: Callable[[FloatArray], float],
    init: object,
    config: PopulationConfig,
    *,
    generations: int,
    seed: int | None = None,
    sigma_decay: float = 0.0,
) -> EvolveResult:
    """Continuous EA. Mutation is isotropic Gaussian on the vector genome.

    Seed policy: ``numpy.random.Generator(seed)``. The same seed yields the
    same parent draws and mutations on every backend; torch / jax only
    specialise :func:`soft_weights`.
    """
    pop = _as_f64(init).copy()
    if pop.ndim != 2 or pop.shape[0] != config.size:
        raise ValueError(f"init must have shape ({config.size}, d), got {pop.shape}")
    if int(generations) < 1:
        raise ValueError("generations must be >= 1")
    rng = _rng(seed)
    n_eval = 0
    records: list[GenerationRecord] = []
    violations = 0
    values = np.empty(config.size, dtype=np.float64)
    for i in range(config.size):
        values[i] = float(fitness(pop[i]))
        n_eval += 1
    best_idx = int(np.argmin(values))
    best = pop[best_idx].copy()
    best_value = float(values[best_idx])
    for g in range(int(generations)):
        beta = _beta_at(config, g)
        rec = _record(g, values, beta=beta, n_eval=n_eval)
        if rec.measured_gap > rec.bound + 1e-12:
            violations += 1
        records.append(rec)
        w = soft_weights(values, beta=beta)
        order = np.argsort(values)
        elite_n = config.elitism
        new_pop = pop.copy()
        if elite_n:
            new_pop[:elite_n] = pop[order[:elite_n]]
        n_fill = config.size - elite_n
        if n_fill:
            parents = _select_indices(w, n_fill, rng)
            sigma = config.mutation_sigma / (1.0 + float(sigma_decay) * float(g))
            noise = rng.normal(0.0, sigma, size=(n_fill, pop.shape[1]))
            new_pop[elite_n:] = pop[parents] + noise
        pop = new_pop
        for i in range(elite_n, config.size):
            values[i] = float(fitness(pop[i]))
            n_eval += 1
        if elite_n:
            values[:elite_n] = values[order[:elite_n]]
        idx = int(np.argmin(values))
        if float(values[idx]) < best_value:
            best_value = float(values[idx])
            best = pop[idx].copy()
    return EvolveResult(best=best, best_value=best_value, n_eval=n_eval, records=records, gap_violations=violations)


def geometry_evolve(
    target: PackGenes,
    init: Sequence[PackGenes],
    config: PopulationConfig,
    *,
    generations: int,
    seed: int | None = None,
    mutation: GeometryMutation | None = None,
    isotropic: bool = False,
    isotropic_sigma: float | None = None,
) -> EvolveResult:
    """Pack-structure search. ``isotropic=True`` is the G3 control."""
    mut = mutation or GeometryMutation()
    packs = list(init)
    if len(packs) != config.size:
        raise ValueError("init population size must match config.size")
    rng = _rng(seed)
    n_eval = 0
    values = np.array([pack_fitness(p, target=target) for p in packs], dtype=np.float64)
    n_eval += config.size
    records: list[GenerationRecord] = []
    violations = 0
    best = packs[int(np.argmin(values))]
    best_value = float(np.min(values))
    iso_sigma = float(isotropic_sigma if isotropic_sigma is not None else config.mutation_sigma)
    for g in range(int(generations)):
        beta = _beta_at(config, g)
        rec = _record(g, values, beta=beta, n_eval=n_eval)
        if rec.measured_gap > rec.bound + 1e-12:
            violations += 1
        records.append(rec)
        w = soft_weights(values, beta=beta)
        order = np.argsort(values)
        elite_n = config.elitism
        new_packs = list(packs)
        if elite_n:
            new_packs[:elite_n] = [packs[int(i)] for i in order[:elite_n]]
        n_fill = config.size - elite_n
        if n_fill:
            parents = _select_indices(w, n_fill, rng)
            for k, pidx in enumerate(parents):
                parent = packs[int(pidx)]
                if isotropic:
                    new_packs[elite_n + k] = mutate_isotropic(parent, iso_sigma, rng, mut=mut)
                else:
                    new_packs[elite_n + k] = mutate_geometry(parent, mut, rng)
        packs = new_packs
        for i in range(elite_n, config.size):
            values[i] = pack_fitness(packs[i], target=target)
            n_eval += 1
        if elite_n:
            values[:elite_n] = values[order[:elite_n]]
        idx = int(np.argmin(values))
        if float(values[idx]) < best_value:
            best_value = float(values[idx])
            best = packs[idx]
    return EvolveResult(
        best=best.as_vector(),
        best_value=best_value,
        n_eval=n_eval,
        records=records,
        gap_violations=violations,
    )


def newton_polish(x: FloatArray, grad: FloatArray, hess: FloatArray) -> FloatArray:
    """Exact Newton step ``x - H^{-1} g``. One linear solve, not a fitness call."""
    return np.asarray(x - np.linalg.solve(hess, grad), dtype=np.float64)


def gd_polish(x: FloatArray, grad: FloatArray, *, lr: float) -> FloatArray:
    return np.asarray(x - float(lr) * grad, dtype=np.float64)


def memetic_evolve(
    fitness: Callable[[FloatArray], float],
    grad_hess: Callable[[FloatArray], tuple[FloatArray, FloatArray]],
    init: object,
    config: PopulationConfig,
    *,
    generations: int,
    polish: PolishKind,
    polish_steps: int = 3,
    gd_lr: float = 0.1,
    seed: int | None = None,
    sigma_decay: float = 0.0,
) -> EvolveResult:
    """EA + local polish. Every ``fitness`` call is counted, including polish."""
    if int(polish_steps) < 0:
        raise ValueError("polish_steps must be >= 0")

    def _polished(x: FloatArray) -> FloatArray:
        y = x.copy()
        for _ in range(int(polish_steps)):
            g, h = grad_hess(y)
            if polish == "newton":
                y = newton_polish(y, g, h)
            elif polish == "gd":
                y = gd_polish(y, g, lr=gd_lr)
            else:
                raise ValueError(f"unknown polish {polish!r}")
        return y

    pop = _as_f64(init).copy()
    if pop.ndim != 2 or pop.shape[0] != config.size:
        raise ValueError(f"init must have shape ({config.size}, d), got {pop.shape}")
    rng = _rng(seed)
    n_eval = 0
    for i in range(config.size):
        pop[i] = _polished(pop[i])
    values = np.empty(config.size, dtype=np.float64)
    for i in range(config.size):
        values[i] = float(fitness(pop[i]))
        n_eval += 1
    records: list[GenerationRecord] = []
    violations = 0
    best = pop[int(np.argmin(values))].copy()
    best_value = float(np.min(values))
    for g in range(int(generations)):
        beta = _beta_at(config, g)
        rec = _record(g, values, beta=beta, n_eval=n_eval)
        if rec.measured_gap > rec.bound + 1e-12:
            violations += 1
        records.append(rec)
        w = soft_weights(values, beta=beta)
        order = np.argsort(values)
        elite_n = config.elitism
        new_pop = pop.copy()
        if elite_n:
            new_pop[:elite_n] = pop[order[:elite_n]]
        n_fill = config.size - elite_n
        if n_fill:
            parents = _select_indices(w, n_fill, rng)
            sigma = config.mutation_sigma / (1.0 + float(sigma_decay) * float(g))
            new_pop[elite_n:] = pop[parents] + rng.normal(0.0, sigma, size=(n_fill, pop.shape[1]))
            for i in range(elite_n, config.size):
                new_pop[i] = _polished(new_pop[i])
        pop = new_pop
        for i in range(elite_n, config.size):
            values[i] = float(fitness(pop[i]))
            n_eval += 1
        if elite_n:
            values[:elite_n] = values[order[:elite_n]]
        idx = int(np.argmin(values))
        if float(values[idx]) < best_value:
            best_value = float(values[idx])
            best = pop[idx].copy()
    return EvolveResult(best=best, best_value=best_value, n_eval=n_eval, records=records, gap_violations=violations)


def _lower_bound(problem: DiscreteProblem, assignment: FloatArray, kind: BoundKind) -> tuple[float, str]:
    if kind == "brute_force":
        _x, emin = brute_force_min(problem)
        return float(emin), "brute_force"
    if kind == "certify_gap":
        cert = certify_gap(problem, assignment)
        return float(cert.lower_bound), str(cert.method)
    poly = problem.to_polynomial()
    return float(negative_coeff_lower_bound(poly)), "negative_coeff"


def _mutate_bits(x: FloatArray, rng: Generator, p: float) -> FloatArray:
    y = x.copy()
    flips = rng.random(x.size) < float(p)
    if not bool(np.any(flips)):
        flips[int(rng.integers(0, x.size))] = True
    y[flips] = 1.0 - y[flips]
    return y


def certified_discrete_evolve(
    problem: DiscreteProblem,
    config: PopulationConfig,
    *,
    generations: int,
    seed: int | None = None,
    bound: BoundKind = "negative_coeff",
    init: object | None = None,
) -> CertifiedEvolveResult:
    """EA on the ``DiscreteProblem`` seam; stop when the certified gap closes.

    ``gap_closed`` requires a sound ``lower_bound <= best_value`` sandwich
    that has collapsed to numerical zero. A loose bound never forges a close.
    The brute-force bound is exponential in ``n`` and is labelled as such.
    """
    n = int(problem.n)
    rng = _rng(seed)
    if init is None:
        pop = rng.integers(0, 2, size=(config.size, n)).astype(np.float64)
    else:
        pop = _as_f64(init).copy()
        if pop.shape != (config.size, n):
            raise ValueError(f"init must have shape ({config.size}, {n})")
    n_eval = 0
    values = np.empty(config.size, dtype=np.float64)
    for i in range(config.size):
        values[i] = float(problem.energy(pop[i]))
        n_eval += 1
    records: list[GenerationRecord] = []
    best_idx = int(np.argmin(values))
    best = pop[best_idx].copy()
    best_value = float(values[best_idx])
    used = 0
    method = str(bound)
    lower = float("-inf")
    closed = False
    gap = float("inf")
    for g in range(int(generations)):
        used = g + 1
        beta = _beta_at(config, g)
        rec = _record(g, values, beta=beta, n_eval=n_eval)
        records.append(rec)
        lower, method = _lower_bound(problem, best, bound)
        gap = float(best_value - lower)
        closed = bool(np.isfinite(lower) and lower <= best_value + _GAP_ATOL and gap <= _GAP_ATOL)
        if closed:
            break
        w = soft_weights(values, beta=beta)
        order = np.argsort(values)
        elite_n = config.elitism
        new_pop = pop.copy()
        if elite_n:
            new_pop[:elite_n] = pop[order[:elite_n]]
        n_fill = config.size - elite_n
        if n_fill:
            parents = _select_indices(w, n_fill, rng)
            for k, pidx in enumerate(parents):
                new_pop[elite_n + k] = _mutate_bits(pop[int(pidx)], rng, config.bit_flip_p)
        pop = new_pop
        for i in range(elite_n, config.size):
            values[i] = float(problem.energy(pop[i]))
            n_eval += 1
        if elite_n:
            values[:elite_n] = values[order[:elite_n]]
        idx = int(np.argmin(values))
        if float(values[idx]) < best_value:
            best_value = float(values[idx])
            best = pop[idx].copy()
    if not closed:
        lower, method = _lower_bound(problem, best, bound)
        gap = float(best_value - lower)
        closed = bool(np.isfinite(lower) and lower <= best_value + _GAP_ATOL and gap <= _GAP_ATOL)
    assignment = tuple(int(v) for v in np.rint(best).tolist())
    return CertifiedEvolveResult(
        best_assignment=assignment,
        best_value=best_value,
        lower_bound=lower,
        gap=gap,
        gap_closed=closed,
        generations_used=used,
        n_eval=n_eval,
        bound_method=method,
        records=tuple(records),
    )


def named_quadratic() -> tuple[FloatArray, FloatArray, Callable[[FloatArray], float], Callable[[FloatArray], tuple[FloatArray, FloatArray]]]:
    """G4 plant: ``f(x) = 1/2 x^T H x`` with ``H = diag(4, 1)``. Min is 0 at 0."""
    hess = np.diag(np.array([4.0, 1.0], dtype=np.float64))

    def fitness(x: FloatArray) -> float:
        v = _as_f64(x).reshape(-1)
        return float(0.5 * v @ hess @ v)

    def grad_hess(x: FloatArray) -> tuple[FloatArray, FloatArray]:
        v = _as_f64(x).reshape(-1)
        return hess @ v, hess

    x0 = np.array([1.0, 1.0], dtype=np.float64)
    return x0, hess, fitness, grad_hess


def named_g2_objective(x: object) -> float:
    """Nonnegative 2-D bowl plus a mild ripple (G2 continuous suite)."""
    v = _as_f64(x).reshape(-1)
    target = np.array([0.2, -0.4], dtype=np.float64)
    return float(np.sum((v - target) ** 2) + 0.15 * np.sum(np.sin(3.0 * v) ** 2))


def named_pack_target() -> PackGenes:
    """Spec §5 pack: ``(mu, delta, n, c) = (0.5, 0.01, 2, 1.3)``."""
    return PackGenes(0.5, 0.01, 2, 1.3)


__all__ = [
    "BoundKind",
    "CertifiedEvolveResult",
    "EvolveResult",
    "GenerationRecord",
    "GeometryMutation",
    "PackGenes",
    "PolishKind",
    "PopulationConfig",
    "certified_discrete_evolve",
    "crossover_packs",
    "gd_polish",
    "geometry_evolve",
    "honesty_payload",
    "memetic_evolve",
    "mutate_geometry",
    "mutate_isotropic",
    "named_g2_objective",
    "named_pack_target",
    "named_quadratic",
    "newton_polish",
    "pack_fitness",
    "selection_gap_bound",
    "selection_stats",
    "soft_population_evolve",
    "soft_weights",
]
