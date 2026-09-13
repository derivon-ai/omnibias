# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""General finite-observation information geometry; legacy location API unchanged.

The default uses a fixed-covariance mean Jacobian. Parameter-dependent noise
requires explicit likelihood scores. Information rank and finite-jet visibility
are local numerical diagnostics, not global identifiability theorems.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
ObservationFn = Callable[[Array, Array], Array]
ScoreProvenance = Literal["exact_finite", "quadrature", "monte_carlo"]
DirectionalJetProvider = Callable[[Array, Array, Array, int], Array]


@dataclass(frozen=True)
class LikelihoodScores:
    """Supplied likelihood-score rows with normalized expectation weights.

    Each row is ``grad_theta log p(y | theta, design)`` for one outcome of
    the whole declared experiment. Weights are finite, nonnegative and
    normalized by their positive sum. Provenance describes the expectation
    rule; it does not prove that these rows are scores of the intended law.
    Even ``exact_finite`` uses numerical arithmetic, not a certificate.
    """

    scores: Array
    weights: Array
    provenance: ScoreProvenance
    description: str = ""

    def __post_init__(self) -> None:
        scores = np.asarray(self.scores, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if scores.ndim != 2 or 0 in scores.shape or not np.all(np.isfinite(scores)):
            raise ValueError("finite nonempty score matrix required")
        if (
            weights.shape != (scores.shape[0],)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
            or not np.any(weights > 0)
        ):
            raise ValueError(
                "one finite nonnegative weight per score row, with positive mass, required"
            )
        if self.provenance not in ("exact_finite", "quadrature", "monte_carlo"):
            raise ValueError("score provenance must be exact_finite, quadrature or monte_carlo")
        # Scaling before summation avoids overflow for valid large weights.
        normalized = weights / np.max(weights)
        normalized /= np.sum(normalized)
        object.__setattr__(self, "scores", scores.copy())
        object.__setattr__(self, "weights", normalized)


LikelihoodScoreProvider = Callable[[Array, Array], LikelihoodScores]


@dataclass(frozen=True)
class ObservationModel:
    """Mean observations plus optional likelihood scores and directional jets.

    ``directional_jet(theta, design, direction, max_order)`` returns ordinary
    directional derivatives, including order zero, shaped
    ``(max_order + 1, *value.shape)``. These are not Taylor coefficients.
    """

    value: ObservationFn
    jacobian: ObservationFn
    parameter_names: tuple[str, ...]
    derivative_kind: str = "supplied analytic"
    score_provider: LikelihoodScoreProvider | None = None
    directional_jet: DirectionalJetProvider | None = None

    def __post_init__(self) -> None:
        if not self.parameter_names or len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("nonempty unique parameter names required")
        if self.score_provider is not None and not callable(self.score_provider):
            raise ValueError("score_provider must be callable")
        if self.directional_jet is not None and not callable(self.directional_jet):
            raise ValueError("directional_jet must be callable")


@dataclass(frozen=True)
class ObservationInformation:
    jacobian: Array
    whitened_jacobian: Array
    fisher: Array
    parameter_names: tuple[str, ...]
    derivative_kind: str
    score_factor: Array | None = None
    score_provenance: ScoreProvenance | None = None
    score_description: str = ""

    @property
    def information_factor(self) -> Array:
        """Rows whose Gram matrix is Fisher; use these for nuisance profiling."""
        return self.whitened_jacobian if self.score_factor is None else self.score_factor

    @property
    def information_kind(self) -> str:
        return (
            "fixed_covariance_mean_jacobian" if self.score_factor is None else "likelihood_scores"
        )


@dataclass(frozen=True)
class IdentifiabilityReport:
    rank: int
    singular_values: Array
    weak_directions: Array
    threshold: float
    parameter_names: tuple[str, ...]
    scope: str = "local first-order finite observations"
    certified: bool = False


@dataclass(frozen=True)
class ProfileInformation:
    fisher: Array
    projected_jacobian: Array
    nuisance_rank: int
    target: tuple[int, ...]
    nuisance: tuple[int, ...]


@dataclass(frozen=True)
class DirectionalVisibilityReport:
    direction: Array
    derivatives: Array
    first_visible_order: int | None
    leading_vector: Array | None
    status: Literal["visible", "inconclusive_at_supplied_order"]
    derivative_kind: str
    scope: str = "finite-order local observation visibility along the supplied direction"
    certified: bool = False
    threshold: float = 1e-12

    @property
    def max_order(self) -> int:
        return int(self.derivatives.shape[0])


def _evaluate_model(
    model: ObservationModel, theta: Array, design: Array
) -> tuple[Array, Array, Array, Array]:
    t = np.asarray(theta, dtype=float)
    d = np.asarray(design, dtype=float)
    if t.shape != (len(model.parameter_names),) or not np.all(np.isfinite(t)):
        raise ValueError("theta must match parameter names and be finite")
    values = np.asarray(model.value(t, d), dtype=float)
    raw = np.asarray(model.jacobian(t, d), dtype=float)
    m, p = values.size, t.size
    if (
        m == 0
        or raw.shape not in ((m, p), (*values.shape, p))
        or not np.all(np.isfinite(raw))
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("finite nonempty observations and matching Jacobian required")
    return t, d, values, raw.reshape(m, p)


def observation_information(
    model: ObservationModel,
    theta: Array,
    design: Array,
    *,
    covariance: object = 1.0,
    score_provider: LikelihoodScoreProvider | None = None,
) -> ObservationInformation:
    """Form fixed-covariance mean information or an explicit score expectation.

    A scalar covariance denotes variance (not standard deviation); a vector
    denotes diagonal variances. A matrix may model correlated observations.
    The Jacobian has value.shape + (n_parameters,), or its flattened matrix.
    A callable covariance is evaluated at ``(theta, design)`` and requires a
    score provider. With scores, Fisher is their weighted outer-product sum;
    the whitened mean Jacobian is retained separately for compatibility. No
    covariance derivatives are silently omitted or inferred from a callable.
    """
    provider = score_provider if score_provider is not None else model.score_provider
    if provider is not None and not callable(provider):
        raise ValueError("score_provider must be callable")
    if callable(covariance) and provider is None:
        raise ValueError("parameter-dependent covariance requires an explicit score_provider")
    t, d, values, jac = _evaluate_model(model, theta, design)
    m = values.size
    cov = np.asarray(covariance(t, d) if callable(covariance) else covariance, dtype=float)
    if cov.ndim == 0:
        if not np.isfinite(cov) or cov <= 0:
            raise ValueError("covariance must be positive")
        white = jac / np.sqrt(cov)
    elif cov.ndim == 1:
        if cov.shape != (m,) or not np.all(np.isfinite(cov)) or np.any(cov <= 0):
            raise ValueError("positive variance per observation required")
        white = jac / np.sqrt(cov[:, None])
    else:
        if (
            cov.shape != (m, m)
            or not np.all(np.isfinite(cov))
            or not np.allclose(cov, cov.T, rtol=0, atol=1e-12)
        ):
            raise ValueError("covariance must be finite symmetric (m,m)")
        try:
            white = np.linalg.solve(np.linalg.cholesky(cov), jac)
        except np.linalg.LinAlgError as exc:
            raise ValueError("covariance must be positive definite") from exc
    factor = None
    provenance = None
    description = ""
    if provider is not None:
        supplied = provider(t, d)
        if not isinstance(supplied, LikelihoodScores):
            raise ValueError("score_provider must return LikelihoodScores")
        # Revalidate arrays because frozen dataclasses can still contain arrays
        # whose contents were mutated after construction.
        supplied = LikelihoodScores(
            supplied.scores, supplied.weights, supplied.provenance, supplied.description
        )
        if supplied.scores.shape[1] != t.size:
            raise ValueError("one score column per named parameter required")
        factor = np.sqrt(supplied.weights[:, None]) * supplied.scores
        provenance = supplied.provenance
        description = supplied.description
    information_factor = white if factor is None else factor
    fisher = information_factor.T @ information_factor
    if not np.all(np.isfinite(fisher)):
        raise ValueError("information matrix must be finite")
    return ObservationInformation(
        jac.copy(),
        white,
        fisher,
        model.parameter_names,
        model.derivative_kind,
        factor,
        provenance,
        description,
    )


def observation_visibility(
    model: ObservationModel,
    theta: Array,
    design: Array,
    direction: Array,
    *,
    max_order: int = 4,
    atol: float = 1e-12,
) -> DirectionalVisibilityReport:
    """First visible supplied directional derivative, not global identification.

    A vanishing finite tower is inconclusive. This concerns the observation
    map, not a complete likelihood law or the intrinsic image's smoothness.
    The supplied order-zero and first derivative must match value and Jv.
    """
    if model.directional_jet is None:
        raise ValueError("observation_visibility requires an explicit directional_jet provider")
    if not isinstance(max_order, int) or isinstance(max_order, bool) or max_order < 1:
        raise ValueError("positive integer max_order required")
    if atol < 0 or not np.isfinite(atol):
        raise ValueError("finite nonnegative visibility tolerance required")
    t, d, values, jac = _evaluate_model(model, theta, design)
    v = np.asarray(direction, dtype=float)
    if v.shape != t.shape or not np.all(np.isfinite(v)) or not np.any(v != 0):
        raise ValueError("finite nonzero direction matching parameters required")
    tower = np.asarray(model.directional_jet(t, d, v, max_order), dtype=float)
    if tower.shape != (max_order + 1, *values.shape) or not np.all(np.isfinite(tower)):
        raise ValueError("finite directional derivative tower with matching value shape required")
    if not np.allclose(tower[0], values, rtol=1e-10, atol=1e-12) or not np.allclose(
        tower[1].reshape(-1), jac @ v, rtol=1e-10, atol=1e-12
    ):
        raise ValueError("directional jet must match observation value and Jacobian action")
    derivatives = tower[1:].reshape(max_order, values.size).copy()
    for order, derivative in enumerate(derivatives, 1):
        if np.max(np.abs(derivative)) > atol:
            return DirectionalVisibilityReport(
                v.copy(),
                derivatives,
                order,
                derivative.copy(),
                "visible",
                model.derivative_kind,
                threshold=atol,
            )
    return DirectionalVisibilityReport(
        v.copy(),
        derivatives,
        None,
        None,
        "inconclusive_at_supplied_order",
        model.derivative_kind,
        threshold=atol,
    )


def identifiability_report(
    info: ObservationInformation, *, rtol: float = 1e-10, atol: float = 0.0
) -> IdentifiabilityReport:
    if rtol < 0 or atol < 0 or not np.isfinite(rtol + atol):
        raise ValueError("finite nonnegative rank tolerances required")
    matrix = info.information_factor
    _, s, vh = np.linalg.svd(matrix, full_matrices=matrix.shape[0] < matrix.shape[1])
    threshold = max(atol, rtol * float(s[0]))
    rank = int(np.count_nonzero(s > threshold))
    scope = (
        "local first-order finite observations"
        if info.score_factor is None
        else f"local likelihood-score information; {info.score_provenance} expectation"
    )
    return IdentifiabilityReport(rank, s, vh[rank:].copy(), threshold, info.parameter_names, scope)


def profile_information(
    whitened_jacobian: Array,
    target: Sequence[int],
    nuisance: Sequence[int],
    *,
    rtol: float = 1e-10,
) -> ProfileInformation:
    """Eliminate nuisance tangents, including singular nuisances.

    Pass ``info.information_factor`` to include likelihood-score information.
    The historical argument name also accepts a weighted score factor.
    """
    jac = np.asarray(whitened_jacobian, dtype=float)
    ti, ni = tuple(target), tuple(nuisance)
    if (
        jac.ndim != 2
        or not np.all(np.isfinite(jac))
        or jac.shape[0] == 0
        or not ti
        or rtol < 0
        or not np.isfinite(rtol)
    ):
        raise ValueError("finite observation Jacobian, targets and tolerance required")
    if len(set(ti + ni)) != len(ti + ni) or any(i < 0 or i >= jac.shape[1] for i in ti + ni):
        raise ValueError("target and nuisance indices must be distinct and in range")
    projected = jac[:, ti].copy()
    rank = 0
    if ni:
        u, s, _ = np.linalg.svd(jac[:, ni], full_matrices=False)
        rank = int(np.count_nonzero(s > rtol * float(s[0])))
        basis = u[:, :rank]
        projected -= basis @ (basis.T @ projected)
    return ProfileInformation(projected.T @ projected, projected, rank, ti, ni)


def candidate_information(
    model: ObservationModel,
    theta: Array,
    candidates: Array,
    *,
    covariance: object = 1.0,
    score_provider: LikelihoodScoreProvider | None = None,
) -> Array:
    """One independent observation block per candidate; no correlated-block claim."""
    ds = np.asarray(candidates, dtype=float)
    if ds.ndim < 1 or len(ds) == 0:
        raise ValueError("nonempty candidates required")
    return np.stack(
        [
            observation_information(
                model, theta, d[None, ...], covariance=covariance, score_provider=score_provider
            ).fisher
            for d in ds
        ]
    )


__all__ = [
    "DirectionalJetProvider",
    "DirectionalVisibilityReport",
    "IdentifiabilityReport",
    "LikelihoodScoreProvider",
    "LikelihoodScores",
    "ObservationInformation",
    "ObservationModel",
    "ProfileInformation",
    "ScoreProvenance",
    "candidate_information",
    "identifiability_report",
    "observation_information",
    "observation_visibility",
    "profile_information",
]
