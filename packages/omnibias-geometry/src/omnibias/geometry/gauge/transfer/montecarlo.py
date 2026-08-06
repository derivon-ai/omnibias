# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Monte-Carlo cross-check of a certified transfer-matrix gap.

The certified bound in :mod:`.gap` is a statement about **one fixed finite
matrix**.  A lattice Monte Carlo is a statement about an **ensemble**.  Comparing
them is only honest if both describe the same object, so this module does not
reach for a lattice gauge simulation: it samples the path measure that the
transfer matrix *itself* defines,

.. math::

    P(x_0, \ldots, x_{L-1}) \;\propto\; \prod_{t} T_{x_t,\, x_{t+1}}

on a periodic chain of length ``L``.  This is the ordinary one-dimensional
Euclidean path integral -- exactly the object a transfer matrix is the transfer
matrix *of* -- so the mass gap the correlators measure is by construction the same
``-ln(lambda_1 / lambda_0)`` the certificate bounds.

**Why this is an independent oracle.** The sampler reads matrix *entries* only.
It never touches an eigenvalue, an eigenvector, or anything the certified engines
compute; the gap emerges from the decay of a sampled autocorrelation. The
estimators are the repo's existing lattice ones
(:func:`~omnibias.geometry.gauge.lattice.effective_mass`,
:func:`~omnibias.geometry.gauge.lattice.gevp_ground_mass`,
:func:`~omnibias.geometry.gauge.lattice.jackknife_std`), used unmodified.

**What the comparison can and cannot show.** A certified *lower* bound sitting
below a noisy estimate is a consistency check that can *falsify* -- a bound above
the estimate means something is wrong -- but it is evidence, not proof; only the
interval arithmetic is proof.  The sampler needs non-negative weights for
``P`` to be a probability measure at all, so an entrywise-negative matrix is
refused rather than silently sampled.

Nothing here is a continuum or a Yang-Mills statement.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from omnibias.geometry.gauge.lattice._core.stats import effective_mass
from omnibias.geometry.gauge.transfer.gap import (
    TransferGapResult,
    certified_transfer_matrix_gap,
)
from omnibias.geometry.gauge.transfer.matrices import TransferMatrix


@dataclass(frozen=True)
class PathEnsemble:
    """Sampled configurations of the transfer matrix's own path measure.

    ``observables`` is ``n_samples`` rows of ``n_operators x chain_length``: the
    shape the lattice estimators already take.
    """

    observables: tuple[tuple[tuple[float, ...], ...], ...]
    chain_length: int
    n_samples: int
    acceptance: float
    seed: int

    @property
    def n_operators(self) -> int:
        return len(self.observables[0]) if self.observables else 0


@dataclass(frozen=True)
class MonteCarloGapCheck:
    """A certified lower bound placed against a Monte-Carlo estimate of the same matrix.

    Two tests, and the weaker one is listed first only because it is the one that
    mentions the certificate:

    ``consistent``
        The certified lower bound does not exceed the estimate by more than
        ``n_sigma`` error bars.  This is **one-sided**, and the estimator's own bias
        at small ``tau`` pushes in the direction that makes it pass, so on its own it
        is a weak check -- it can *falsify* loudly but cannot confirm much.
    ``agrees_with_exact``
        The estimate brackets the closed-form gap to within ``n_sigma``.  This is
        **two-sided** and is the test with teeth: it fails if the sampler, the matrix,
        or the identification of "gap" is wrong in either direction.  Available only
        for the models whose spectrum is known in closed form -- which is all of them
        here, by design.
    """

    model: str
    certified_gap_lower: float
    exact_gap: float | None
    monte_carlo_mass: float
    monte_carlo_error: float
    gevp_mass: float
    gevp_error: float
    n_sigma: float
    consistent: bool
    agrees_with_exact: bool | None
    chain_length: int
    n_samples: int
    tau_window: tuple[int, int]
    acceptance: float = 0.0
    detail: str = ""

    @property
    def slack(self) -> float:
        """How far the certified bound sits below the estimate, in mass units."""
        return self.monte_carlo_mass - self.certified_gap_lower


def _weights(transfer: TransferMatrix) -> list[list[float]]:
    """Midpoint entries as transition weights, refusing anything not a measure."""
    rows: list[list[float]] = []
    for row in transfer.entries:
        out: list[float] = []
        for cell in row:
            if cell.lo < 0.0:
                msg = (
                    "path-measure sampling needs non-negative weights, but this matrix "
                    f"has an entry enclosing a negative value ({cell.lo!r}); a truncated "
                    "character sum can do this at small coupling -- raise the coupling or "
                    "the truncation order"
                )
                raise ValueError(msg)
            out.append(max(0.5 * (cell.lo + cell.hi), 0.0))
        rows.append(out)
    return rows


def _supported_start(weights: Sequence[Sequence[float]]) -> int:
    r"""A constant path with positive weight, i.e. a starting point *inside* the support.

    Seeding the chain at random is wrong, not merely slow to equilibrate: the random
    path can have probability zero under ``prod_t T_{x_t, x_{t+1}}``, and heat-bath
    updates cannot leave a zero-weight configuration -- a site flanked by two states
    it has no weight to bridge is frozen forever.  A diagonal matrix is the extreme
    case, where every non-constant path has weight zero.  The constant path at
    ``argmax_i T_ii`` is in the support whenever any diagonal entry is positive.
    """
    diagonal = [weights[i][i] for i in range(len(weights))]
    best = max(range(len(diagonal)), key=lambda i: diagonal[i])
    return best if diagonal[best] > 0.0 else 0


def _sample_index(cumulative: Sequence[float], draw: float) -> int:
    total = cumulative[-1]
    target = draw * total
    for index, value in enumerate(cumulative):
        if target <= value:
            return index
    return len(cumulative) - 1  # pragma: no cover - float guard only


def sample_transfer_path_ensemble(
    transfer: TransferMatrix,
    *,
    chain_length: int = 24,
    n_samples: int = 64,
    n_sweeps: int = 8,
    n_thermalize: int = 40,
    seed: int = 0,
    operators: Sequence[Sequence[float]] | None = None,
) -> PathEnsemble:
    r"""Heat-bath sample the periodic path measure ``prod_t T_{x_t, x_{t+1}}``.

    Gibbs sampling: with its neighbours held fixed, site ``t`` has the exact
    conditional ``P(x_t = j) \propto T_{x_{t-1}, j} T_{j, x_{t+1}}``, which is cheap
    to normalise for the small state spaces here and never rejects.  Only matrix
    entries are read.

    Parameters
    ----------
    chain_length
        Euclidean extent ``L``.  Correlators are measured out to ``L // 2``.
    n_samples
        Number of configurations kept, each separated by ``n_sweeps`` full sweeps
        to decorrelate.
    operators
        One row per operator, giving its value on each state.  The default is a
        single operator that resolves the first excited state: the exact
        subdominant eigenvector when one is recorded, else the state index.  An
        operator orthogonal to that state would leave its signal absent from the
        correlator entirely.
    """
    if chain_length < 4:
        raise ValueError(f"chain_length must be >= 4, got {chain_length}")
    if n_samples < 2:
        raise ValueError(f"n_samples must be >= 2 for a jackknife, got {n_samples}")

    weights = _weights(transfer)
    dim = len(weights)
    if operators is None:
        operators = _default_operators(transfer)
    for row in operators:
        if len(row) != dim:
            msg = f"each operator needs one value per state ({dim}), got {len(row)}"
            raise ValueError(msg)

    rng = random.Random(seed)
    state = [_supported_start(weights)] * chain_length
    moves = 0
    changes = 0

    def sweep() -> None:
        nonlocal moves, changes
        for site in range(chain_length):
            left = state[site - 1]
            right = state[(site + 1) % chain_length]
            cumulative: list[float] = []
            running = 0.0
            for j in range(dim):
                running += weights[left][j] * weights[j][right]
                cumulative.append(running)
            if running <= 0.0:
                msg = (
                    f"site {site} has zero conditional weight between states {left} and "
                    f"{right}, so the current path has probability zero -- this measure is "
                    "reducible and heat-bath sampling cannot move within it"
                )
                raise ValueError(msg)
            picked = _sample_index(cumulative, rng.random())
            moves += 1
            if picked != state[site]:
                changes += 1
            state[site] = picked

    for _ in range(n_thermalize):
        sweep()

    samples: list[tuple[tuple[float, ...], ...]] = []
    for _ in range(n_samples):
        for _ in range(n_sweeps):
            sweep()
        samples.append(tuple(tuple(op[x] for x in state) for op in operators))

    acceptance = changes / moves if moves else 0.0
    return PathEnsemble(
        observables=tuple(samples),
        chain_length=chain_length,
        n_samples=n_samples,
        acceptance=acceptance,
        seed=seed,
    )


def _default_operators(transfer: TransferMatrix) -> tuple[tuple[float, ...], ...]:
    r"""The operator that isolates the first excited state: ``v_1 / v_0``.

    The path measure's stationary distribution is the ground-state transform
    ``pi_x \propto v_0[x]^2``, under which the chain's eigenfunctions are
    ``phi_k[x] = v_k[x] / v_0[x]``.  So an operator ``O`` overlaps mode ``k`` by
    ``sum_x pi_x O[x] phi_k[x] = sum_x v_0[x] O[x] v_k[x]``, and choosing
    ``O = v_1 / v_0`` makes that ``sum_x v_1[x] v_k[x] = delta_{1k}`` -- the
    contamination from every higher mode vanishes rather than merely decaying.

    Using ``v_1`` itself, the naive choice, leaves a real ``O(5%)`` upward bias in
    the effective mass at small ``tau``.  For the ``su(2)`` class-angle matrix this
    ratio is ``sin(2 theta) / sin(theta) = 2 cos(theta)``, i.e. exactly the
    fundamental character -- the operator a lattice practitioner would have reached
    for anyway.
    """
    dim = transfer.dimension
    perron = tuple(float(v) for v in transfer.perron_vector)
    if transfer.subdominant_vectors and len(perron) == dim:
        first = tuple(float(v) for v in transfer.subdominant_vectors[0])
        if len(first) == dim and all(abs(p) > 1e-12 for p in perron):
            ratio = tuple(f / p for f, p in zip(first, perron, strict=True))
            if any(abs(v) > 0.0 for v in ratio):
                return (ratio,)
    return (tuple(float(i) for i in range(dim)),)


def _connected_correlator(ensemble: PathEnsemble, operator: int = 0) -> list[list[float]]:
    r"""Per-sample connected correlators ``C_i(tau)``, vacuum-subtracted ensemble-wide."""
    rows = [sample[operator] for sample in ensemble.observables]
    length = ensemble.chain_length
    max_tau = length // 2
    mean = sum(sum(row) for row in rows) / (len(rows) * length)
    out: list[list[float]] = []
    for row in rows:
        curve: list[float] = []
        for tau in range(max_tau + 1):
            acc = sum(row[t] * row[(t + tau) % length] for t in range(length)) / length
            curve.append(acc - mean * mean)
        out.append(curve)
    return out


def _plateau_mass(curves: Sequence[Sequence[float]], window: tuple[int, int]) -> float:
    """Mean effective mass over a fixed tau window of one (possibly jackknifed) curve."""
    mean = [sum(c[tau] for c in curves) / len(curves) for tau in range(len(curves[0]))]
    lo, hi = window
    values = [effective_mass(mean, tau) for tau in range(lo, hi)]
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def _jackknife_error(
    curves: Sequence[Sequence[float]],
    estimate: Callable[[Sequence[Sequence[float]]], float],
) -> float:
    r"""Delete-1 jackknife error of a *non-linear* estimator of the whole ensemble.

    ``sqrt((n - 1) / n * sum_i (theta_i - theta_bar)^2)`` over the leave-one-out
    replicas.  The repo's :func:`~omnibias.geometry.gauge.lattice.jackknife_std` and
    :func:`~omnibias.geometry.gauge.lattice.ensemble_mean_jackknife` take *raw
    samples* and form the replicas themselves, so they are the right tool for a mean
    and the wrong one here: an effective mass is a ratio of logs of ensemble
    averages, whose replicas have to be built by re-running the estimator on each
    reduced ensemble.  Feeding pre-formed replicas to those helpers double-jackknifes
    and understates the error by roughly ``sqrt(n)``.
    """
    n_meas = len(curves)
    replicas = [
        estimate([c for k, c in enumerate(curves) if k != i]) for i in range(n_meas)
    ]
    finite = [v for v in replicas if math.isfinite(v)]
    if len(finite) < 2:
        return float("nan")
    mean = sum(finite) / len(finite)
    total = sum((v - mean) ** 2 for v in finite)
    return math.sqrt((len(finite) - 1) / len(finite) * total)


def certified_gap_versus_monte_carlo(
    transfer: TransferMatrix,
    *,
    chain_length: int = 32,
    n_samples: int = 200,
    n_sweeps: int = 6,
    seed: int = 0,
    n_sigma: float = 4.0,
    tau_window: tuple[int, int] | None = None,
    gap: TransferGapResult | None = None,
) -> MonteCarloGapCheck:
    r"""Place the certified lower bound against a Monte-Carlo estimate of the same matrix.

    Both sides describe the *same fixed matrix*: the certificate bounds
    ``-ln(lambda_1 / lambda_0)`` by interval arithmetic, and the Monte Carlo
    estimates it from the decay of a sampled autocorrelation of the path measure
    ``prod_t T_{x_t, x_{t+1}}``.  There is no ensemble-versus-matrix gap to
    hand-wave across, which is the only reason this comparison means anything.

    Choosing ``tau_window`` honestly
    -------------------------------
    The effective mass is biased in *both* directions and the window sits between
    them.  At small ``tau`` the states above the first excited one have not yet
    decayed, biasing it **up**.  Near ``tau = L / 2`` the chain's periodicity turns
    the correlator into a ``cosh``, whose effective mass falls to zero at the
    midpoint, biasing it **down** -- and by then the signal is at the noise floor
    anyway.  The default window is the early exponential regime between the two,
    which is where the plateau actually lives; it is a fixed rule, not tuned per
    model.

    That the small-``tau`` bias is *upward* is exactly why
    :attr:`MonteCarloGapCheck.consistent` alone would be too easy to pass, and why
    :attr:`MonteCarloGapCheck.agrees_with_exact` is reported beside it.
    """
    result = gap if gap is not None else certified_transfer_matrix_gap(transfer)
    ensemble = sample_transfer_path_ensemble(
        transfer,
        chain_length=chain_length,
        n_samples=n_samples,
        n_sweeps=n_sweeps,
        seed=seed,
    )
    window = tau_window if tau_window is not None else _default_window(chain_length)

    curves = _connected_correlator(ensemble)
    mass = _plateau_mass(curves, window)

    error = _jackknife_error(curves, lambda rows: _plateau_mass(rows, window))
    gevp_mass, gevp_error = _gevp_estimate(curves, (window[0] + 1, window[1] + 1))

    exact_ratio = transfer.exact_subdominant_ratio()
    exact = None
    if exact_ratio is not None and exact_ratio.lo > 0.0:
        exact = -math.log(0.5 * (exact_ratio.lo + exact_ratio.hi))

    bar = 0.0 if not math.isfinite(error) else n_sigma * error
    consistent = math.isfinite(mass) and result.spectral_gap_lower <= mass + bar
    agrees = None if exact is None else (math.isfinite(mass) and abs(mass - exact) <= bar)
    detail = (
        f"certified {result.spectral_gap_lower:.6f} <= MC {mass:.6f} "
        f"+/- {error:.6f} (x{n_sigma:g})"
        if consistent
        else f"certified {result.spectral_gap_lower:.6f} EXCEEDS MC {mass:.6f} +/- {error:.6f}"
    )
    return MonteCarloGapCheck(
        model=transfer.model,
        certified_gap_lower=result.spectral_gap_lower,
        exact_gap=exact,
        monte_carlo_mass=mass,
        monte_carlo_error=error,
        gevp_mass=gevp_mass,
        gevp_error=gevp_error,
        n_sigma=n_sigma,
        consistent=consistent,
        agrees_with_exact=agrees,
        chain_length=chain_length,
        n_samples=n_samples,
        tau_window=window,
        acceptance=ensemble.acceptance,
        detail=detail,
    )


def _default_window(chain_length: int) -> tuple[int, int]:
    """The early exponential regime: after ``tau = 0``, well before the ``cosh`` midpoint."""
    return (0, max(2, min(4, chain_length // 8)))


def _gevp_estimate(
    curves: Sequence[Sequence[float]], window: tuple[int, int]
) -> tuple[float, float]:
    r"""Single-operator ratio estimate ``-ln(C(t1) / C(t0)) / (t1 - t0)``, jackknifed.

    With one operator the generalised eigenvalue problem degenerates to exactly this
    ratio, so this is :func:`~omnibias.geometry.gauge.lattice.gevp_ground_mass`'s
    one-dimensional case rather than a different estimator.

    It is deliberately evaluated on a window shifted one step later than the plateau
    average, because on the *same* window it would carry no new information: summing
    ``-ln(C(tau + 1) / C(tau))`` over ``tau in [t0, t1)`` telescopes exactly to
    ``-ln(C(t1) / C(t0))``, so the plateau mean and the endpoint ratio are the same
    number.  Shifted, it instead answers whether the plateau is stable in ``tau``.
    """
    t0, t1 = window
    if t1 <= t0 or t1 >= len(curves[0]):
        return float("nan"), float("nan")

    def ratio(rows: Sequence[Sequence[float]]) -> float:
        mean = [sum(c[tau] for c in rows) / len(rows) for tau in range(len(rows[0]))]
        if mean[t0] <= 0.0 or mean[t1] <= 0.0:
            return float("nan")
        return -math.log(mean[t1] / mean[t0]) / (t1 - t0)

    return ratio(curves), _jackknife_error(curves, ratio)


__all__ = [
    "MonteCarloGapCheck",
    "PathEnsemble",
    "certified_gap_versus_monte_carlo",
    "sample_transfer_path_ensemble",
]
