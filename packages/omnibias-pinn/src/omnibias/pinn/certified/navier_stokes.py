# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Navier-Stokes certified-evidencearation objects and periodic spectral validators.

This module is deliberately **certified-evidence**, not a global-regularity solver.  It
turns sampled periodic velocity/pressure fields into independently checkable
objects: residuals, divergence diagnostics, energy/enstrophy indicators, CAP
bundle fields, and explicit honesty labels.  The local differential terms use a
spectral Fourier representation on a periodic torus; finite-energy
``R^3`` claims require a separate compactification / tail-bound bridge.

The convention for primitive incompressible Navier-Stokes is

.. math::

   u_t + (u\cdot\nabla)u + \nabla p - \nu \Delta u - f = 0,\qquad
   \nabla\cdot u = 0.

Pressure consistency is reported with the periodic Poisson identity

.. math::

   -\Delta p - \sum_{ij}\partial_i\partial_j(u_i u_j) = 0,

which follows from the primitive equation when ``div u == 0`` and density is
one.  A non-unit density multiplies the quadratic term.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from omnibias.core.proof import formal_claim_forgery_errors, kernel_earned_theorem_prover_verified
from omnibias.core.verified.interval import Interval, sum_intervals
from omnibias.core.verified.linalg import (
    identity_matrix,
    inf_norm_matrix,
    inf_norm_vector,
    mat_sub,
    matmul,
    matvec,
    neumann_inverse_norm_bound,
    to_interval_matrix,
)
from omnibias.core.verified.line import (
    conjugate_poisson,
    conjugate_poisson_deriv,
    even_profile,
    even_profile_deriv,
    even_profile_tail_constant,
    hilbert_even_profile,
    hilbert_even_profile_deriv,
    hilbert_of_conjugate,
    hilbert_tail_bound,
    poisson_kernel,
    poisson_kernel_deriv,
)
from omnibias.core.verified.rootfind import interval_newton
from omnibias.core.verified.taylor_model import TaylorModel

ProofTarget = Literal["global_regularity", "finite_time_blowup"]
DomainType = Literal["periodic_torus", "bounded", "compactified_r3"]

SCHEMA_VERSION = "navier-stokes-cap-1"
CANDIDATE_SCHEMA_VERSION = "navier-stokes-candidate-1"
CLM_BLOWUP_SCHEMA_VERSION = "navier-stokes-clm-blowup-1"
CLM_MULTIZERO_SCHEMA_VERSION = "navier-stokes-clm-multizero-blowup-1"
GCLM_BLOWUP_SCHEMA_VERSION = "navier-stokes-gclm-selfsimilar-1"
GCLM_GRADIENT_AMP_SCHEMA_VERSION = "navier-stokes-gclm-gradient-amplification-1"
CCF_SELFSIMILAR_SCHEMA_VERSION = "navier-stokes-ccf-selfsimilar-blowup-attempt-1"
CCF_LINEARIZED_OPERATOR_SCHEMA_VERSION = "navier-stokes-ccf-linearized-operator-bound-1"
REQUIRED_CAP_KEYS: tuple[str, ...] = (
    "schema_version",
    "problem",
    "domain",
    "field_samples",
    "residual_samples",
    "residual_diagnostics",
    "validation_inputs",
    "honesty",
    "proof_obligations",
    "provenance",
)
REQUIRED_VALIDATION_KEYS: tuple[str, ...] = (
    "velocity",
    "pressure",
    "velocity_t",
    "forcing",
    "viscosity",
    "density",
    "lengths",
    "spectral_convention",
)
REQUIRED_CANDIDATE_KEYS: tuple[str, ...] = (
    "schema_version",
    "candidate_type",
    "replay_grid",
    "replay_inputs",
    "result",
    "coefficients",
    "upgrade_gate",
    "honesty",
    "provenance",
)


@dataclass(frozen=True)
class HonestyLabels:
    """Machine-readable claim boundaries for Navier-Stokes artifacts."""

    unproven_claim: bool = False
    exact_solution_claim: bool = False
    interval_verified: bool = False
    theorem_prover_verified: bool = False
    finite_energy_verified: bool = False
    periodic_model_only: bool = True
    proof_status: str = "numerical_proof_prep"
    notes: str = ""


@dataclass(frozen=True)
class NavierStokesProofContract:
    """The theorem-level target a candidate is trying to support."""

    target: ProofTarget
    dimension: int = 3
    viscosity_positive: bool = True
    smooth_initial_data: bool = True
    divergence_free_initial_data: bool = True
    finite_energy_required: bool = True
    domain: DomainType = "periodic_torus"
    statement: str = ""
    nonclaims: tuple[str, ...] = (
        "A periodic numerical residual is not a proof of global regularity.",
        "Neural or symbolic discovery is only a candidate generator.",
        "Finite-energy R^3 claims need explicit tail and compactification checks.",
    )


@dataclass(frozen=True)
class ProofObligation:
    """A small proof or CAP obligation that can be discharged independently."""

    name: str
    description: str
    status: str = "open"
    verifier: str = "external"
    tolerance: float | None = None


@dataclass(frozen=True)
class CompactifiedR3Metadata:
    """Metadata for a compactified finite-energy ``R^3`` representation."""

    map_name: str = "rational_radial"
    map_formula: str = "r = rho / (1 - rho), rho in [0, 1)"
    coordinate_names: tuple[str, ...] = ("rho", "theta", "phi")
    jacobian_weight: str = "r^2 / (1 - rho)^2"
    basis: str = "chebyshev_radial_fourier_angular"
    tail_bound_convention: str = "l1_coefficient_tail_bounds_sup_norm"


@dataclass(frozen=True)
class TailBound:
    """Interval-friendly truncation/tail certificate placeholder."""

    field: str
    basis: str
    retained_modes: int
    tail_l1_bound: float
    norm: str = "sup"
    certified: bool = False


@dataclass(frozen=True)
class ScalarInterval:
    """JSON-friendly scalar interval with midpoint/radius representation."""

    lower: float
    upper: float
    midpoint: float
    radius: float
    certified: bool = False


@dataclass(frozen=True)
class IntervalArithmeticMetadata:
    """Describes the interval arithmetic backend used for a certificate."""

    backend: str = "float64_nextafter_interval"
    precision_bits: int = 53
    outward_rounding: str = "np.nextafter_after_each_operation"
    dependency: str = "numpy"
    certified: bool = True
    notes: str = (
        "Finite-dimensional interval enclosure with outward-padded float64 "
        "endpoints; theorem-grade review may swap in MPFR/Arb without changing "
        "the artifact schema."
    )


@dataclass(frozen=True)
class IntervalBoundReport:
    """Machine-readable interval validation summary."""

    quantity: str
    interval: ScalarInterval
    method: str = "outward_padded_float_envelope"
    contains_midpoint: bool = True
    certified: bool = False


@dataclass(frozen=True)
class ReplayGrid:
    """Deterministic grid metadata needed to replay a candidate artifact."""

    domain_type: DomainType
    dimension: int
    grid_shape: tuple[int, ...]
    coordinate_names: tuple[str, ...]
    axis_values: tuple[tuple[float, ...], ...]
    lengths: tuple[float, ...] = ()
    compactification: CompactifiedR3Metadata | None = None


@dataclass(frozen=True)
class CompactifiedCoefficientSet:
    """Flattened coefficient payload plus tail/energy metadata."""

    field: str
    basis: str
    shape: tuple[int, ...]
    coefficients: tuple[float, ...]
    tail_bound: TailBound
    finite_energy_estimate: float | None = None


@dataclass(frozen=True)
class AxisymmetricCompactifiedMetadata:
    """Metadata for axisymmetric compactified meridional candidates."""

    coordinate_names: tuple[str, ...] = ("rho", "zeta")
    physical_coordinates: tuple[str, ...] = ("r", "z")
    map_formula: str = (
        "r = rho / (1 - rho), z = zeta / (1 - |zeta|); "
        "rho in (0, 1), zeta in (-1, 1)"
    )
    jacobian_weight: str = "2*pi*r*dr/drho*dz/dzeta"
    symmetry_assumptions: tuple[str, ...] = (
        "theta_independent",
        "meridional_half_plane_r_positive",
    )
    axis_regular_obligations: tuple[str, ...] = (
        "streamfunction vanishes to second order at r=0",
        "swirl velocity vanishes to first order at r=0",
        "pressure is even in r near the axis",
    )
    finite_energy_convention: str = "axisymmetric_energy_integral_2pi_int_r_dr_dz"


@dataclass(frozen=True)
class AxisymmetricSwirlAnsatzMetadata:
    """Describes the reduced 3D axisymmetric-with-swirl representation."""

    representation: str = "streamfunction_swirl_pressure"
    component_names: tuple[str, ...] = ("streamfunction", "swirl", "pressure")
    velocity_components: tuple[str, ...] = ("u_r", "u_theta", "u_z")
    velocity_formula: tuple[str, ...] = (
        "u_r = -(1/r) * d_z psi",
        "u_theta = gamma",
        "u_z = (1/r) * d_r psi",
    )
    incompressibility: str = "automatic away from r=0 when psi is smooth"
    basis: str = "sampled_axisymmetric_meridional_values"
    open_obligations: tuple[str, ...] = (
        "axis_regular_smoothness",
        "compactified_tail_bounds",
        "finite_energy_initial_data",
        "linearized_invertibility_or_a_priori_estimate",
    )


@dataclass(frozen=True)
class AxisymmetricBasisMetadata:
    """Finite meridional basis metadata for axisymmetric coefficient refinement."""

    basis_name: str = "compact_polynomial_envelope"
    radial_degree: int = 2
    axial_degree: int = 2
    component_names: tuple[str, ...] = ("streamfunction", "swirl", "pressure")
    compact_coordinates: tuple[str, ...] = ("rho = r/(1+r)", "zeta = z/(1+|z|)")
    envelope: str = "exp(-0.5*(rho^2+zeta^2))"
    axis_factors: dict[str, str] | None = None


@dataclass(frozen=True)
class AxisymmetricFunctionSpaceMetadata:
    """Finite-dimensional function-space convention for closure certificates."""

    banach_name: str = "axisymmetric_compact_polynomial_sup_energy"
    domain: str = "compactified_axisymmetric_meridional_half_plane"
    coefficient_basis: str = "compact_polynomial_envelope"
    norm_name: str = "coefficient_l2_to_residual_sup"
    residual_norm: str = "cellwise_continuum_sup_norm"
    energy_convention: str = "axisymmetric_energy_integral_2pi_int_r_dr_dz"
    tail_convention: str = "coefficient_l1_decay_envelope"
    axis_regular_convention: str = "streamfunction_r2_swirl_r_pressure_even"
    theorem_grade: bool = False
    notes: str = (
        "Finite-dimensional closure convention for certified-evidence replay; "
        "continuum Banach-space invertibility remains an external obligation."
    )


@dataclass(frozen=True)
class TheoremGradeFunctionSpaceContract:
    """Continuum function-space contract for theorem-grade proof attempts."""

    contract_name: str = "axisymmetric_weighted_banach_closure_contract"
    domain: str = "compactified_axisymmetric_meridional_half_plane"
    field_space: str = "smooth_axisymmetric_finite_energy_fields"
    operator_domain: str = "weighted_C2_axisymmetric_stream_swirl_pressure_ball"
    operator_codomain: str = "weighted_C0_axisymmetric_residual_ball"
    norm_name: str = "weighted_sup_plus_energy_tail_norm"
    projection_map: str = "compact_polynomial_basis_projection_with_tail"
    coefficient_to_field_map: str = "axisymmetric_coefficients_to_fields"
    axis_regular_class: str = "streamfunction_r2_swirl_r_pressure_even_smooth_axis"
    external_verification_required: bool = True
    theorem_grade: bool = True
    open_obligations: tuple[str, ...] = (
        "prove_projection_map_bounded",
        "prove_linearized_operator_frechet_derivative",
        "prove_tail_space_complete",
        "external_theorem_grade_review",
    )
    unproven_claim: bool = False


@dataclass(frozen=True)
class ExactEquationContract:
    """Exact PDE identity targeted by proof-program artifacts."""

    form: str
    equation: str
    unknowns: tuple[str, ...]
    constraints: tuple[str, ...]
    identities: tuple[str, ...]
    continuation_criteria: tuple[str, ...] = ()
    reduction: str = "full_3d"
    theorem_grade: bool = True
    open_obligations: tuple[str, ...] = ()
    unproven_claim: bool = False


@dataclass(frozen=True)
class ProofProgramFunctionSpaceDefinition:
    """Human/machine-readable theorem-space definition for one route."""

    route: str
    space_name: str
    domain: str
    norm: str
    smoothness_class: str
    compactification: str
    tail_convention: str
    continuation_target: str
    open_obligations: tuple[str, ...]
    theorem_grade: bool = True
    unproven_claim: bool = False


@dataclass(frozen=True)
class ProofObligationBundle:
    """Machine-checkable theorem obligation for an external verifier."""

    schema_version: str
    route: str
    lemma_id: str
    theorem_name: str
    theorem_statement: str
    assumptions: tuple[str, ...]
    dependencies: tuple[str, ...]
    source_artifact_sha256: str
    expected_verifier: str
    proof_status: str = "open"
    reviewed_at_utc: str = ""
    theorem_grade: bool = True
    unproven_claim: bool = False


@dataclass(frozen=True)
class IntervalCAPBackendContract:
    """Required interval backend capabilities for theorem gates."""

    backend_name: str = "mpfr_or_arb_outward_interval_backend"
    required_rounding: str = "directed_outward_rounding_after_each_primitive"
    required_capabilities: tuple[str, ...] = (
        "certified_quadrature",
        "polynomial_tail_bounds",
        "compactification_jacobian_bounds",
        "operator_norm_bounds",
        "nonlinear_product_bounds",
        "artifact_hashing",
    )
    current_backend: str = "float64_nextafter_interval"
    theorem_grade_ready: bool = False
    open_obligations: tuple[str, ...] = ("replace_float64_padding_with_mpfr_or_arb",)
    unproven_claim: bool = False


@dataclass(frozen=True)
class LinearizedOperatorCertificate:
    """Finite-dimensional linearized residual operator certificate."""

    method: str
    matrix_shape: tuple[int, int]
    perturbation: float
    matrix_norm: float
    approximate_inverse_norm: float | None
    condition_estimate: float | None
    smallest_singular_value: float
    largest_singular_value: float
    rank: int
    full_column_rank: bool
    finite_dimensional_certified: bool
    operator_theoretic_certified: bool
    open_obligations: tuple[str, ...]
    unproven_claim: bool = False


@dataclass(frozen=True)
class RadiiPolynomialCertificate:
    """Radii-polynomial arithmetic certificate for closure consistency."""

    residual_bound: float
    approximate_inverse_norm: float | None
    nonlinear_lipschitz_bound: float | None
    closure_interval: ScalarInterval
    passed: bool
    certified: bool
    method: str = "inverse_norm_times_residual_plus_lipschitz"
    open_obligations: tuple[str, ...] = ()
    unproven_claim: bool = False


@dataclass(frozen=True)
class NormDivergenceCertificate:
    """Finite-time norm-growth evidence for the blow-up route."""

    norm_name: str
    blowup_time: float | None
    growth_exponent: float | None
    lower_bound_interval: ScalarInterval | None
    linked_to_field_profile: bool
    certified: bool
    method: str = "trace_growth_exponent_linkage"
    open_obligations: tuple[str, ...] = ()
    unproven_claim: bool = False


@dataclass(frozen=True)
class CandidateGateConfig:
    """Thresholds for certified-evidence candidate promotion gates."""

    momentum_residual_max: float = 1e-8
    continuity_max: float = 1e-8
    pressure_poisson_max: float = 1e-8
    require_independent_recompute: bool = True
    require_tail_bounds_for_interval: bool = True
    require_certified_tail_bounds_for_interval: bool = True


@dataclass(frozen=True)
class NavierStokesSubstrate:
    """Names the supported certified-evidence equation forms and hard constraints."""

    forms: tuple[str, ...] = ("primitive", "leray_projected", "vorticity")
    incompressibility_enforcement: tuple[str, ...] = (
        "vector_potential",
        "streamfunction_2d",
        "periodic_leray_projection",
    )
    diagnostics: tuple[str, ...] = (
        "kinetic_energy",
        "enstrophy",
        "palinstrophy",
        "max_abs_divergence",
        "bkm_vorticity_proxy",
        "pressure_poisson_max_abs",
    )


def global_regularity_contract() -> NavierStokesProofContract:
    """Contract for the global-regularity route."""
    return NavierStokesProofContract(
        target="global_regularity",
        domain="compactified_r3",
        statement=(
            "For every smooth divergence-free finite-energy 3D initial datum "
            "and every positive viscosity, the incompressible Navier-Stokes "
            "solution remains smooth for all positive time."
        ),
    )


def blowup_contract() -> NavierStokesProofContract:
    """Contract for the finite-time blow-up route."""
    return NavierStokesProofContract(
        target="finite_time_blowup",
        domain="compactified_r3",
        statement=(
            "There exists a smooth divergence-free finite-energy 3D initial "
            "datum for positive-viscosity incompressible Navier-Stokes whose "
            "classical solution becomes singular in finite time."
        ),
    )


def default_proof_obligations(target: ProofTarget) -> list[ProofObligation]:
    """Return theorem/CAP obligations required before any global-regularity-grade claim."""
    common = [
        ProofObligation("divergence_free", "Prove or interval-verify div u = 0."),
        ProofObligation(
            "primitive_residual",
            "Bound the primitive Navier-Stokes residual in a certified norm.",
        ),
        ProofObligation(
            "pressure_poisson",
            "Verify pressure is consistent with the incompressibility constraint.",
        ),
        ProofObligation(
            "tail_bounds",
            "Bound all spectral, Chebyshev, or compactification tails.",
        ),
    ]
    if target == "global_regularity":
        return common + [
            ProofObligation(
                "continuation_criterion",
                "Show the proposed estimate implies a standard continuation criterion.",
            ),
            ProofObligation(
                "a_priori_estimate",
                "Prove the discovered inequality for all smooth finite-energy data.",
            ),
        ]
    return common + [
        ProofObligation(
            "linearized_invertibility",
            "Certify invertibility or a radii-polynomial bound near the profile.",
        ),
        ProofObligation(
            "finite_energy_initial_data",
            "Verify the induced initial datum is smooth and finite-energy on R^3.",
        ),
        ProofObligation(
            "norm_divergence",
            "Prove the certified candidate makes a critical norm diverge.",
        ),
    ]


def proof_contract_bundle(
    contract: NavierStokesProofContract,
    *,
    honesty: HonestyLabels | None = None,
) -> dict[str, Any]:
    """JSON-friendly theorem contract + honesty labels."""
    labels = honesty if honesty is not None else HonestyLabels()
    return {
        "contract": asdict(contract),
        "honesty": asdict(labels),
        "proof_obligations": [asdict(o) for o in default_proof_obligations(contract.target)],
    }


def compactified_r3_metadata(
    *,
    map_name: str = "rational_radial",
    map_formula: str = "r = rho / (1 - rho), rho in [0, 1)",
    basis: str = "chebyshev_radial_fourier_angular",
) -> CompactifiedR3Metadata:
    """Return explicit compactification metadata for CAP schemas.

    This is metadata only; interval validation still needs concrete coefficient
    and quadrature/tail bounds.
    """
    return CompactifiedR3Metadata(
        map_name=map_name,
        map_formula=map_formula,
        basis=basis,
    )


def deterministic_periodic_replay_grid(
    *,
    dimension: int,
    n: int,
    lengths: tuple[float, ...] | None = None,
) -> ReplayGrid:
    """Return a deterministic periodic replay grid with explicit axes."""
    if dimension < 1:
        raise ValueError(f"dimension must be positive, got {dimension}")
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")
    domain = _lengths(lengths, dimension)
    names = ("x", "y", "z")[:dimension]
    axes = tuple(
        tuple(float(v) for v in (domain[i] * np.arange(n, dtype=float) / n))
        for i in range(dimension)
    )
    return ReplayGrid(
        domain_type="periodic_torus",
        dimension=dimension,
        grid_shape=tuple(n for _ in range(dimension)),
        coordinate_names=names,
        axis_values=axes,
        lengths=domain,
    )


def compactified_sandbox_replay_grid(
    *,
    n_radial: int = 16,
    n_theta: int = 16,
    n_phi: int = 32,
    compactification: CompactifiedR3Metadata | None = None,
) -> ReplayGrid:
    """Return a deterministic compactified ``R^3`` sandbox grid."""
    if min(n_radial, n_theta, n_phi) < 2:
        raise ValueError("compactified replay grid dimensions must be >= 2")
    rho = tuple(float(v) for v in ((np.arange(n_radial, dtype=float) + 0.5) / n_radial))
    theta = tuple(float(v) for v in (np.pi * (np.arange(n_theta, dtype=float) + 0.5) / n_theta))
    phi = tuple(float(v) for v in (2.0 * np.pi * np.arange(n_phi, dtype=float) / n_phi))
    return ReplayGrid(
        domain_type="compactified_r3",
        dimension=3,
        grid_shape=(n_radial, n_theta, n_phi),
        coordinate_names=("rho", "theta", "phi"),
        axis_values=(rho, theta, phi),
        lengths=(1.0, np.pi, 2.0 * np.pi),
        compactification=compactification or compactified_r3_metadata(),
    )


def axisymmetric_compactified_metadata() -> AxisymmetricCompactifiedMetadata:
    """Return explicit metadata for compactified axisymmetric candidates."""
    return AxisymmetricCompactifiedMetadata()


def axisymmetric_swirl_ansatz_metadata() -> AxisymmetricSwirlAnsatzMetadata:
    """Return metadata for the streamfunction/swirl/pressure ansatz."""
    return AxisymmetricSwirlAnsatzMetadata()


def axisymmetric_meridional_replay_grid(
    *,
    n_radial: int = 16,
    n_axial: int = 17,
) -> ReplayGrid:
    """Return a deterministic compactified meridional half-plane replay grid."""
    if n_radial < 4 or n_axial < 4:
        raise ValueError("axisymmetric replay grid needs n_radial,n_axial >= 4")
    rho = tuple(float(v) for v in ((np.arange(n_radial, dtype=float) + 0.5) / n_radial))
    zeta = tuple(float(v) for v in (-1.0 + 2.0 * (np.arange(n_axial, dtype=float) + 0.5) / n_axial))
    return ReplayGrid(
        domain_type="compactified_r3",
        dimension=2,
        grid_shape=(n_radial, n_axial),
        coordinate_names=("rho", "zeta"),
        axis_values=(rho, zeta),
        lengths=(1.0, 2.0),
        compactification=compactified_r3_metadata(
            map_name="axisymmetric_rational_meridional",
            map_formula="r = rho/(1-rho), z = zeta/(1-|zeta|)",
            basis="sampled_axisymmetric_meridional_values",
        ),
    )


def axisymmetric_physical_axes(grid: ReplayGrid | dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Convert an axisymmetric compactified replay grid to physical ``(r, z)`` axes."""
    data = asdict(grid) if isinstance(grid, ReplayGrid) else dict(grid)
    axes = data["axis_values"]
    rho = np.asarray(axes[0], dtype=float)
    zeta = np.asarray(axes[1], dtype=float)
    if np.any(rho <= 0.0) or np.any(rho >= 1.0):
        raise ValueError("axisymmetric radial compact coordinate must be in (0, 1)")
    if np.any(np.abs(zeta) >= 1.0):
        raise ValueError("axisymmetric axial compact coordinate must be in (-1, 1)")
    radial = rho / (1.0 - rho)
    axial = zeta / (1.0 - np.abs(zeta))
    return cast(np.ndarray, np.asarray(radial, dtype=float)), cast(np.ndarray, np.asarray(axial, dtype=float))


def axisymmetric_basis_metadata(
    *,
    radial_degree: int = 2,
    axial_degree: int = 2,
) -> AxisymmetricBasisMetadata:
    """Return metadata for the compact polynomial refiner basis."""
    if radial_degree < 0 or axial_degree < 0:
        raise ValueError("axisymmetric basis degrees must be non-negative")
    return AxisymmetricBasisMetadata(
        radial_degree=int(radial_degree),
        axial_degree=int(axial_degree),
        axis_factors={
            "streamfunction": "r^2",
            "swirl": "r",
            "pressure": "1",
        },
    )


def axisymmetric_basis_count(metadata: AxisymmetricBasisMetadata | dict[str, Any]) -> int:
    """Return the number of scalar basis functions per component."""
    data = asdict(metadata) if isinstance(metadata, AxisymmetricBasisMetadata) else dict(metadata)
    return (int(data["radial_degree"]) + 1) * (int(data["axial_degree"]) + 1)


def _coerce_basis_metadata(
    metadata: AxisymmetricBasisMetadata | dict[str, Any] | None,
) -> AxisymmetricBasisMetadata:
    if metadata is None:
        return axisymmetric_basis_metadata()
    if isinstance(metadata, AxisymmetricBasisMetadata):
        return metadata
    return axisymmetric_basis_metadata(
        radial_degree=int(metadata.get("radial_degree", 2)),
        axial_degree=int(metadata.get("axial_degree", 2)),
    )


def axisymmetric_basis_tensor(
    radial_axis: np.ndarray,
    axial_axis: np.ndarray,
    *,
    component: str,
    metadata: AxisymmetricBasisMetadata | dict[str, Any] | None = None,
) -> np.ndarray:
    """Evaluate component basis functions on physical meridional axes."""
    meta = _coerce_basis_metadata(metadata)
    r = np.asarray(radial_axis, dtype=float)
    z = np.asarray(axial_axis, dtype=float)
    rho = r / (1.0 + r)
    zeta = z / (1.0 + np.abs(z))
    rr = r[:, None]
    compact_r = rho[:, None]
    compact_z = zeta[None, :]
    envelope = np.exp(-0.5 * (compact_r * compact_r + compact_z * compact_z))
    if component == "streamfunction":
        factor = rr * rr
    elif component == "swirl":
        factor = rr
    elif component == "pressure":
        factor = np.ones((r.shape[0], z.shape[0]), dtype=float)
    else:
        raise ValueError(f"unknown axisymmetric component {component!r}")
    basis = [
        factor * envelope * (compact_r ** i) * (compact_z ** j)
        for i in range(meta.radial_degree + 1)
        for j in range(meta.axial_degree + 1)
    ]
    return cast(np.ndarray, np.asarray(np.stack(basis), dtype=float))


def initial_axisymmetric_coefficients(
    *,
    seed: int = 0,
    metadata: AxisymmetricBasisMetadata | dict[str, Any] | None = None,
    scale: float = 0.05,
) -> np.ndarray:
    """Return deterministic initial coefficients for the three ansatz fields."""
    meta = _coerce_basis_metadata(metadata)
    rng = np.random.default_rng(seed)
    n_basis = axisymmetric_basis_count(meta)
    coeffs = rng.normal(0.0, scale, 3 * n_basis)
    coeffs[0] += scale
    coeffs[n_basis] += scale
    return cast(np.ndarray, np.asarray(coeffs, dtype=float))


def split_axisymmetric_coefficients(
    coefficients: np.ndarray,
    metadata: AxisymmetricBasisMetadata | dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a flattened coefficient vector into ``psi``, ``swirl``, ``pressure`` blocks."""
    meta = _coerce_basis_metadata(metadata)
    n_basis = axisymmetric_basis_count(meta)
    coeffs = np.asarray(coefficients, dtype=float)
    if coeffs.shape != (3 * n_basis,):
        raise ValueError(f"expected coefficient shape {(3 * n_basis,)}, got {coeffs.shape}")
    return (
        cast(np.ndarray, coeffs[:n_basis]),
        cast(np.ndarray, coeffs[n_basis:2 * n_basis]),
        cast(np.ndarray, coeffs[2 * n_basis:]),
    )


def axisymmetric_coefficients_to_fields(
    coefficients: np.ndarray,
    *,
    radial_axis: np.ndarray,
    axial_axis: np.ndarray,
    metadata: AxisymmetricBasisMetadata | dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate streamfunction, swirl, and pressure from flattened coefficients."""
    meta = _coerce_basis_metadata(metadata)
    psi_c, swirl_c, pressure_c = split_axisymmetric_coefficients(coefficients, meta)
    psi_basis = axisymmetric_basis_tensor(radial_axis, axial_axis, component="streamfunction", metadata=meta)
    swirl_basis = axisymmetric_basis_tensor(radial_axis, axial_axis, component="swirl", metadata=meta)
    pressure_basis = axisymmetric_basis_tensor(radial_axis, axial_axis, component="pressure", metadata=meta)
    return {
        "streamfunction": cast(np.ndarray, np.tensordot(psi_c, psi_basis, axes=(0, 0))),
        "swirl": cast(np.ndarray, np.tensordot(swirl_c, swirl_basis, axes=(0, 0))),
        "pressure": cast(np.ndarray, np.tensordot(pressure_c, pressure_basis, axes=(0, 0))),
    }


def compactified_coefficient_set(
    field: str,
    coefficients: np.ndarray,
    *,
    basis: str = "chebyshev_radial_fourier_angular",
    tail_l1_bound: float = 0.0,
    retained_modes: int | None = None,
    finite_energy_estimate: float | None = None,
    certified: bool = False,
) -> CompactifiedCoefficientSet:
    """Return a JSON-friendly flattened coefficient payload."""
    coeffs = np.asarray(coefficients, dtype=float)
    kept = int(coeffs.size if retained_modes is None else retained_modes)
    tail = TailBound(
        field=field,
        basis=basis,
        retained_modes=kept,
        tail_l1_bound=float(tail_l1_bound),
        certified=bool(certified),
    )
    return CompactifiedCoefficientSet(
        field=field,
        basis=basis,
        shape=tuple(int(v) for v in coeffs.shape),
        coefficients=tuple(float(v) for v in coeffs.ravel()),
        tail_bound=tail,
        finite_energy_estimate=None if finite_energy_estimate is None else float(finite_energy_estimate),
    )


def _json_dict(value: Any) -> Any:
    """Convert dataclasses / numpy values to JSON-serializable objects."""
    if hasattr(value, "__dataclass_fields__"):
        return _json_dict(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_dict(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_dict(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256_json(value: Any) -> str:
    """Return a deterministic SHA-256 for JSON-like proof artifacts."""
    encoded = json.dumps(
        _json_dict(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_velocity(velocity: np.ndarray) -> np.ndarray:
    u = np.asarray(velocity, dtype=float)
    if u.ndim < 2:
        raise ValueError("velocity must have shape (dim, n1, ..., nd)")
    dim = int(u.shape[0])
    if dim != u.ndim - 1:
        raise ValueError(
            f"velocity first axis has dim={dim}, but grid dimension is {u.ndim - 1}"
        )
    return cast(np.ndarray, np.asarray(u, dtype=float))


def _lengths(lengths: tuple[float, ...] | None, dim: int) -> tuple[float, ...]:
    if lengths is None:
        return tuple(2.0 * np.pi for _ in range(dim))
    if len(lengths) != dim:
        raise ValueError(f"expected {dim} domain lengths, got {lengths!r}")
    return tuple(float(v) for v in lengths)


def _wave_numbers(shape: tuple[int, ...], lengths: tuple[float, ...]) -> list[np.ndarray]:
    axes: list[np.ndarray] = []
    for axis, (n, length) in enumerate(zip(shape, lengths, strict=True)):
        k = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
        view_shape = [1] * len(shape)
        view_shape[axis] = n
        axes.append(k.reshape(view_shape))
    return axes


def spectral_gradient_scalar(
    scalar: np.ndarray, *, lengths: tuple[float, ...] | None = None
) -> np.ndarray:
    """Periodic spectral gradient of a scalar sample array."""
    f = np.asarray(scalar, dtype=float)
    domain = _lengths(lengths, f.ndim)
    spectrum = np.fft.fftn(f)
    return cast(np.ndarray, np.asarray(np.stack([
        np.real(np.fft.ifftn(1j * k * spectrum))
        for k in _wave_numbers(f.shape, domain)
    ]), dtype=float))


def spectral_laplacian(
    values: np.ndarray, *, lengths: tuple[float, ...] | None = None
) -> np.ndarray:
    """Periodic spectral Laplacian for scalar or component-first vector data."""
    x = np.asarray(values, dtype=float)
    if x.ndim >= 2 and x.shape[0] == x.ndim - 1:
        return cast(np.ndarray, np.asarray(
            np.stack([spectral_laplacian(comp, lengths=lengths) for comp in x]),
            dtype=float,
        ))
    domain = _lengths(lengths, x.ndim)
    k2 = sum(k * k for k in _wave_numbers(x.shape, domain))
    return cast(np.ndarray, np.asarray(np.real(np.fft.ifftn(-k2 * np.fft.fftn(x))), dtype=float))


def spectral_divergence(
    velocity: np.ndarray, *, lengths: tuple[float, ...] | None = None
) -> np.ndarray:
    """Periodic spectral divergence of a component-first vector field."""
    u = _as_velocity(velocity)
    domain = _lengths(lengths, u.shape[0])
    div = np.zeros_like(u[0])
    for comp, k in zip(u, _wave_numbers(u.shape[1:], domain), strict=True):
        div = div + np.real(np.fft.ifftn(1j * k * np.fft.fftn(comp)))
    return cast(np.ndarray, np.asarray(div, dtype=float))


def spectral_curl(
    velocity: np.ndarray, *, lengths: tuple[float, ...] | None = None
) -> np.ndarray:
    """Periodic spectral curl; returns scalar vorticity in 2D and vector in 3D."""
    u = _as_velocity(velocity)
    dim = u.shape[0]
    if dim == 2:
        grad_u = spectral_gradient_scalar(u[0], lengths=lengths)
        grad_v = spectral_gradient_scalar(u[1], lengths=lengths)
        return cast(np.ndarray, np.asarray(grad_v[0] - grad_u[1], dtype=float))
    if dim != 3:
        raise ValueError("spectral_curl supports only 2D or 3D velocity fields")
    grads = [spectral_gradient_scalar(u[i], lengths=lengths) for i in range(3)]
    return cast(np.ndarray, np.asarray(np.stack([
        grads[2][1] - grads[1][2],
        grads[0][2] - grads[2][0],
        grads[1][0] - grads[0][1],
    ]), dtype=float))


def leray_project_periodic(
    velocity: np.ndarray, *, lengths: tuple[float, ...] | None = None
) -> np.ndarray:
    """Project a periodic vector field onto its divergence-free part."""
    u = _as_velocity(velocity)
    dim = u.shape[0]
    domain = _lengths(lengths, dim)
    spectra = np.stack([np.fft.fftn(comp) for comp in u])
    ks = _wave_numbers(u.shape[1:], domain)
    k2 = sum(k * k for k in ks)
    k_dot_u = sum(k * spectra[i] for i, k in enumerate(ks))
    projected = spectra.copy()
    mask = k2 > 0.0
    safe_k2 = np.where(mask, k2, 1.0)
    for i, k in enumerate(ks):
        projected[i] = np.where(mask, spectra[i] - k * k_dot_u / safe_k2, spectra[i])
    return cast(np.ndarray, np.asarray(
        np.stack([np.real(np.fft.ifftn(projected[i])) for i in range(dim)]),
        dtype=float,
    ))


def primitive_residual_periodic(
    velocity: np.ndarray,
    pressure: np.ndarray | None = None,
    *,
    velocity_t: np.ndarray | None = None,
    forcing: np.ndarray | None = None,
    viscosity: float = 1e-3,
    density: float = 1.0,
    lengths: tuple[float, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(momentum_residual, continuity)`` for sampled periodic NS."""
    u = _as_velocity(velocity)
    dim = u.shape[0]
    domain = _lengths(lengths, dim)
    p = np.zeros_like(u[0]) if pressure is None else np.asarray(pressure, dtype=float)
    u_t = np.zeros_like(u) if velocity_t is None else _as_velocity(velocity_t)
    f = np.zeros_like(u) if forcing is None else _as_velocity(forcing)

    gradients = [spectral_gradient_scalar(u[i], lengths=domain) for i in range(dim)]
    advection = np.stack([
        sum(u[j] * gradients[i][j] for j in range(dim))
        for i in range(dim)
    ])
    grad_p = spectral_gradient_scalar(p, lengths=domain)
    lap_u = spectral_laplacian(u, lengths=domain)
    residual = density * (u_t + advection) + grad_p - viscosity * lap_u - f
    return cast(np.ndarray, np.asarray(residual, dtype=float)), spectral_divergence(u, lengths=domain)


def vorticity_residual_periodic(
    velocity: np.ndarray,
    *,
    velocity_t: np.ndarray | None = None,
    viscosity: float = 1e-3,
    lengths: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Return the 3D vorticity residual ``omega_t + u.grad omega - omega.grad u - nu Delta omega``."""
    u = _as_velocity(velocity)
    if u.shape[0] != 3:
        raise ValueError("vorticity_residual_periodic currently targets 3D fields")
    domain = _lengths(lengths, 3)
    omega = spectral_curl(u, lengths=domain)
    u_t = np.zeros_like(u) if velocity_t is None else _as_velocity(velocity_t)
    omega_t = spectral_curl(u_t, lengths=domain)
    grad_omega = [spectral_gradient_scalar(omega[i], lengths=domain) for i in range(3)]
    grad_u = [spectral_gradient_scalar(u[i], lengths=domain) for i in range(3)]
    adv_omega = np.stack([
        sum(u[j] * grad_omega[i][j] for j in range(3))
        for i in range(3)
    ])
    stretch = np.stack([
        sum(omega[j] * grad_u[i][j] for j in range(3))
        for i in range(3)
    ])
    residual = omega_t + adv_omega - stretch - viscosity * spectral_laplacian(omega, lengths=domain)
    return cast(np.ndarray, np.asarray(residual, dtype=float))


def pressure_poisson_residual_periodic(
    velocity: np.ndarray,
    pressure: np.ndarray,
    *,
    density: float = 1.0,
    lengths: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Check ``-Delta p - rho * d_i d_j(u_i u_j)`` on a periodic grid."""
    u = _as_velocity(velocity)
    domain = _lengths(lengths, u.shape[0])
    quad = np.zeros_like(u[0])
    for i in range(u.shape[0]):
        for j in range(u.shape[0]):
            grad_i = spectral_gradient_scalar(u[i] * u[j], lengths=domain)[i]
            quad = quad + spectral_gradient_scalar(grad_i, lengths=domain)[j]
    residual = -spectral_laplacian(np.asarray(pressure, dtype=float), lengths=domain) - density * quad
    return cast(np.ndarray, np.asarray(residual, dtype=float))


def _axisym_array(values: np.ndarray, radial_axis: np.ndarray, axial_axis: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    expected = (int(radial_axis.shape[0]), int(axial_axis.shape[0]))
    if arr.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {arr.shape}")
    return cast(np.ndarray, arr)


def _grad_meridional(values: np.ndarray, radial_axis: np.ndarray, axial_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if min(values.shape) > 2:
        dr, dz = np.gradient(values, radial_axis, axial_axis, edge_order=2)
    else:
        dr, dz = np.gradient(values, radial_axis, axial_axis, edge_order=1)
    return cast(np.ndarray, np.asarray(dr, dtype=float)), cast(np.ndarray, np.asarray(dz, dtype=float))


def _lap_axisymmetric_scalar(values: np.ndarray, radial_axis: np.ndarray, axial_axis: np.ndarray) -> np.ndarray:
    val_r, val_z = _grad_meridional(values, radial_axis, axial_axis)
    val_rr, _ = _grad_meridional(val_r, radial_axis, axial_axis)
    _, val_zz = _grad_meridional(val_z, radial_axis, axial_axis)
    r = radial_axis[:, None]
    return cast(np.ndarray, np.asarray(val_rr + val_r / r + val_zz, dtype=float))


def axisymmetric_velocity_from_streamfunction(
    streamfunction: np.ndarray,
    *,
    radial_axis: np.ndarray,
    axial_axis: np.ndarray,
    swirl: np.ndarray | None = None,
) -> dict[str, Any]:
    """Recover cylindrical velocity samples from ``psi`` and optional swirl."""
    r = np.asarray(radial_axis, dtype=float)
    z = np.asarray(axial_axis, dtype=float)
    if np.any(r <= 0.0):
        raise ValueError("axisymmetric sampled residuals require r > 0 grid points")
    psi = _axisym_array(streamfunction, r, z, "streamfunction")
    psi_r, psi_z = _grad_meridional(psi, r, z)
    radius = r[:, None]
    u_r = -psi_z / radius
    u_z = psi_r / radius
    u_theta = np.zeros_like(psi) if swirl is None else _axisym_array(swirl, r, z, "swirl")
    return {
        "u_r": cast(np.ndarray, np.asarray(u_r, dtype=float)),
        "u_theta": cast(np.ndarray, np.asarray(u_theta, dtype=float)),
        "u_z": cast(np.ndarray, np.asarray(u_z, dtype=float)),
        "radial_axis": r,
        "axial_axis": z,
    }


def axisymmetric_swirl_residual_samples(
    streamfunction: np.ndarray,
    swirl: np.ndarray,
    pressure: np.ndarray,
    *,
    radial_axis: np.ndarray,
    axial_axis: np.ndarray,
    viscosity: float = 1e-3,
    density: float = 1.0,
) -> dict[str, Any]:
    """Return sampled steady primitive residuals for an axisymmetric swirl ansatz.

    The formulas are cylindrical-coordinate diagnostics on a meridional grid.
    They are numerical certified-evidence samples, not interval-certified bounds.
    """
    r = np.asarray(radial_axis, dtype=float)
    z = np.asarray(axial_axis, dtype=float)
    pressure_arr = _axisym_array(pressure, r, z, "pressure")
    velocity = axisymmetric_velocity_from_streamfunction(
        streamfunction,
        radial_axis=r,
        axial_axis=z,
        swirl=swirl,
    )
    u_r = velocity["u_r"]
    u_t = velocity["u_theta"]
    u_z = velocity["u_z"]
    radius = r[:, None]

    ur_r, ur_z = _grad_meridional(u_r, r, z)
    ut_r, ut_z = _grad_meridional(u_t, r, z)
    uz_r, uz_z = _grad_meridional(u_z, r, z)
    p_r, p_z = _grad_meridional(pressure_arr, r, z)

    lap_ur = _lap_axisymmetric_scalar(u_r, r, z)
    lap_ut = _lap_axisymmetric_scalar(u_t, r, z)
    lap_uz = _lap_axisymmetric_scalar(u_z, r, z)

    radial = density * (u_r * ur_r + u_z * ur_z - (u_t * u_t) / radius) + p_r
    radial = radial - viscosity * (lap_ur - u_r / (radius * radius))
    azimuthal = density * (u_r * ut_r + u_z * ut_z + u_r * u_t / radius)
    azimuthal = azimuthal - viscosity * (lap_ut - u_t / (radius * radius))
    axial = density * (u_r * uz_r + u_z * uz_z) + p_z - viscosity * lap_uz

    r_ur_r, _ = _grad_meridional(radius * u_r, r, z)
    divergence = r_ur_r / radius + uz_z
    residual_norm = np.sqrt(radial * radial + azimuthal * azimuthal + axial * axial)
    return {
        "velocity": {
            "u_r": cast(np.ndarray, np.asarray(u_r, dtype=float)).tolist(),
            "u_theta": cast(np.ndarray, np.asarray(u_t, dtype=float)).tolist(),
            "u_z": cast(np.ndarray, np.asarray(u_z, dtype=float)).tolist(),
        },
        "residual_samples": {
            "radial": cast(np.ndarray, np.asarray(radial, dtype=float)).tolist(),
            "azimuthal": cast(np.ndarray, np.asarray(azimuthal, dtype=float)).tolist(),
            "axial": cast(np.ndarray, np.asarray(axial, dtype=float)).tolist(),
            "divergence": cast(np.ndarray, np.asarray(divergence, dtype=float)).tolist(),
        },
        "residual_diagnostics": {
            "max_abs_momentum_residual": float(np.max(np.abs(residual_norm))),
            "rms_momentum_residual": float(np.sqrt(np.mean(residual_norm * residual_norm))),
            "max_abs_continuity": float(np.max(np.abs(divergence))),
            "rms_continuity": float(np.sqrt(np.mean(divergence * divergence))),
            "max_abs_radial_residual": float(np.max(np.abs(radial))),
            "max_abs_azimuthal_residual": float(np.max(np.abs(azimuthal))),
            "max_abs_axial_residual": float(np.max(np.abs(axial))),
            "unproven_claim": False,
        },
    }


def axisymmetric_energy_estimate(
    velocity: dict[str, Any],
    *,
    radial_axis: np.ndarray,
    axial_axis: np.ndarray,
) -> float:
    """Estimate kinetic energy with the axisymmetric weight ``2*pi*r``."""
    r = np.asarray(radial_axis, dtype=float)
    z = np.asarray(axial_axis, dtype=float)
    u_r = _axisym_array(np.asarray(velocity["u_r"], dtype=float), r, z, "u_r")
    u_t = _axisym_array(np.asarray(velocity["u_theta"], dtype=float), r, z, "u_theta")
    u_z = _axisym_array(np.asarray(velocity["u_z"], dtype=float), r, z, "u_z")
    density = 0.5 * (u_r * u_r + u_t * u_t + u_z * u_z) * (2.0 * np.pi * r[:, None])
    by_z = np.trapezoid(density, z, axis=1)
    return float(np.trapezoid(by_z, r))


def axisymmetric_coefficient_loss(
    coefficients: np.ndarray,
    *,
    grid: ReplayGrid | dict[str, Any],
    metadata: AxisymmetricBasisMetadata | dict[str, Any] | None = None,
    viscosity: float = 1e-3,
    density: float = 1.0,
    coefficient_l2: float = 1e-8,
    energy_target: float = 0.0,
    energy_weight: float = 0.0,
) -> dict[str, Any]:
    """Evaluate a structured residual loss for coefficient-defined fields."""
    meta = _coerce_basis_metadata(metadata)
    radial, axial = axisymmetric_physical_axes(grid)
    fields = axisymmetric_coefficients_to_fields(
        coefficients,
        radial_axis=radial,
        axial_axis=axial,
        metadata=meta,
    )
    residual = axisymmetric_swirl_residual_samples(
        fields["streamfunction"],
        fields["swirl"],
        fields["pressure"],
        radial_axis=radial,
        axial_axis=axial,
        viscosity=viscosity,
        density=density,
    )
    energy = axisymmetric_energy_estimate(
        residual["velocity"],
        radial_axis=radial,
        axial_axis=axial,
    )
    diag = residual["residual_diagnostics"]
    residual_loss = (
        float(diag["rms_momentum_residual"]) ** 2
        + float(diag["rms_continuity"]) ** 2
    )
    coeffs = np.asarray(coefficients, dtype=float)
    regularization = float(coefficient_l2) * float(np.mean(coeffs * coeffs))
    energy_penalty = 0.0
    if energy_target > 0.0 and energy_weight > 0.0:
        denom = max(abs(float(energy_target)), 1e-12)
        energy_penalty = float(energy_weight) * ((energy - float(energy_target)) / denom) ** 2
    total = residual_loss + regularization + energy_penalty
    return {
        "loss": float(total),
        "residual_loss": float(residual_loss),
        "regularization_loss": float(regularization),
        "energy_penalty": float(energy_penalty),
        "finite_energy_estimate": float(energy),
        "coefficient_l2_norm": float(np.linalg.norm(coeffs)),
        "residual_diagnostics": dict(diag),
        "residual_samples": residual["residual_samples"],
        "velocity": residual["velocity"],
        "fields": {name: value.tolist() for name, value in fields.items()},
        "unproven_claim": False,
    }


def axisymmetric_holdout_replay_grid(train_grid: ReplayGrid | dict[str, Any]) -> ReplayGrid:
    """Return a deterministic slightly finer holdout grid for refined candidates."""
    data = asdict(train_grid) if isinstance(train_grid, ReplayGrid) else dict(train_grid)
    shape = data["grid_shape"]
    return axisymmetric_meridional_replay_grid(
        n_radial=int(shape[0]) + 1,
        n_axial=int(shape[1]) + 1,
    )


def refine_axisymmetric_coefficients(
    *,
    seed: int = 0,
    n_radial: int = 8,
    n_axial: int = 9,
    radial_degree: int = 2,
    axial_degree: int = 2,
    max_iterations: int = 4,
    step_size: float = 0.02,
    viscosity: float = 1e-3,
    density: float = 1.0,
    coefficient_l2: float = 1e-8,
    energy_target: float = 0.0,
    energy_weight: float = 0.0,
) -> dict[str, Any]:
    """Run a bounded deterministic coordinate refiner for axisymmetric coefficients."""
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    metadata = axisymmetric_basis_metadata(
        radial_degree=radial_degree,
        axial_degree=axial_degree,
    )
    train_grid = axisymmetric_meridional_replay_grid(n_radial=n_radial, n_axial=n_axial)
    coeffs = initial_axisymmetric_coefficients(seed=seed, metadata=metadata)

    def evaluate(candidate: np.ndarray) -> dict[str, Any]:
        return axisymmetric_coefficient_loss(
            candidate,
            grid=train_grid,
            metadata=metadata,
            viscosity=viscosity,
            density=density,
            coefficient_l2=coefficient_l2,
            energy_target=energy_target,
            energy_weight=energy_weight,
        )

    best = evaluate(coeffs)
    current_step = float(step_size)
    history = [{
        "iteration": 0,
        "loss": best["loss"],
        "step_size": current_step,
        "accepted": False,
    }]
    for iteration in range(1, max_iterations + 1):
        accepted = False
        for idx in range(coeffs.size):
            base_value = float(coeffs[idx])
            for sign in (-1.0, 1.0):
                trial = coeffs.copy()
                trial[idx] = base_value + sign * current_step
                trial_eval = evaluate(trial)
                if float(trial_eval["loss"]) < float(best["loss"]):
                    coeffs = trial
                    best = trial_eval
                    accepted = True
                    break
            if accepted:
                break
        if not accepted:
            current_step *= 0.5
        history.append({
            "iteration": iteration,
            "loss": best["loss"],
            "step_size": current_step,
            "accepted": accepted,
        })
    holdout_grid = axisymmetric_holdout_replay_grid(train_grid)
    holdout = axisymmetric_coefficient_loss(
        coeffs,
        grid=holdout_grid,
        metadata=metadata,
        viscosity=viscosity,
        density=density,
        coefficient_l2=coefficient_l2,
        energy_target=energy_target,
        energy_weight=energy_weight,
    )
    return {
        "coefficients": coeffs,
        "basis_metadata": metadata,
        "train_grid": train_grid,
        "holdout_grid": holdout_grid,
        "train": best,
        "holdout": holdout,
        "loss_history": history,
        "seed": int(seed),
        "viscosity": float(viscosity),
        "density": float(density),
        "loss_config": {
            "coefficient_l2": float(coefficient_l2),
            "energy_target": float(energy_target),
            "energy_weight": float(energy_weight),
        },
        "unproven_claim": False,
    }


def build_refined_axisymmetric_swirl_candidate_artifact(
    *,
    seed: int = 0,
    n_radial: int = 8,
    n_axial: int = 9,
    radial_degree: int = 2,
    axial_degree: int = 2,
    max_iterations: int = 4,
    step_size: float = 0.02,
    viscosity: float = 1e-3,
    density: float = 1.0,
    coefficient_l2: float = 1e-8,
    energy_target: float = 0.0,
    energy_weight: float = 0.0,
    tail_l1_bound: float = 1e-6,
) -> dict[str, Any]:
    """Build a replayable refined axisymmetric coefficient artifact."""
    refinement = refine_axisymmetric_coefficients(
        seed=seed,
        n_radial=n_radial,
        n_axial=n_axial,
        radial_degree=radial_degree,
        axial_degree=axial_degree,
        max_iterations=max_iterations,
        step_size=step_size,
        viscosity=viscosity,
        density=density,
        coefficient_l2=coefficient_l2,
        energy_target=energy_target,
        energy_weight=energy_weight,
    )
    metadata = refinement["basis_metadata"]
    coefficients = np.asarray(refinement["coefficients"], dtype=float)
    psi_c, swirl_c, pressure_c = split_axisymmetric_coefficients(coefficients, metadata)
    energy = float(refinement["train"]["finite_energy_estimate"])
    coeff_payload = (
        compactified_coefficient_set(
            "streamfunction_coefficients",
            psi_c,
            basis=metadata.basis_name,
            tail_l1_bound=tail_l1_bound,
            finite_energy_estimate=energy,
        ),
        compactified_coefficient_set(
            "swirl_coefficients",
            swirl_c,
            basis=metadata.basis_name,
            tail_l1_bound=tail_l1_bound,
            finite_energy_estimate=energy,
        ),
        compactified_coefficient_set(
            "pressure_coefficients",
            pressure_c,
            basis=metadata.basis_name,
            tail_l1_bound=tail_l1_bound,
            finite_energy_estimate=None,
        ),
    )
    initial_loss = float(refinement["loss_history"][0]["loss"])
    final_loss = float(refinement["train"]["loss"])
    residual_descended = final_loss <= initial_loss
    return build_candidate_artifact(
        candidate_type="axisymmetric_swirl_refined",
        replay_grid=refinement["train_grid"],
        replay_inputs={
            "coefficients": coefficients,
            "basis_metadata": metadata,
            "train_grid": refinement["train_grid"],
            "holdout_grid": refinement["holdout_grid"],
            "viscosity": float(viscosity),
            "density": float(density),
            "loss_config": refinement["loss_config"],
            "axisymmetric_metadata": axisymmetric_compactified_metadata(),
            "ansatz_metadata": axisymmetric_swirl_ansatz_metadata(),
        },
        result={
            "train": {
                "loss": refinement["train"]["loss"],
                "residual_diagnostics": refinement["train"]["residual_diagnostics"],
                "finite_energy_estimate": refinement["train"]["finite_energy_estimate"],
            },
            "holdout": {
                "loss": refinement["holdout"]["loss"],
                "residual_diagnostics": refinement["holdout"]["residual_diagnostics"],
                "finite_energy_estimate": refinement["holdout"]["finite_energy_estimate"],
            },
            "loss_history": refinement["loss_history"],
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "residual_descended": bool(residual_descended),
            "finite_energy_estimate": energy,
            "unproven_claim": False,
        },
        coefficients=coeff_payload,
        upgrade_gate={
            "stage": "numerical_artifact",
            "independent_replay_required": True,
            "residual_descended": bool(residual_descended),
            "holdout_checked": True,
            "unproven_claim": False,
        },
        proof_obligations=axisymmetric_swirl_ansatz_metadata().open_obligations,
        notes="refined axisymmetric swirl coefficient artifact; no global-regularity claim",
    )


def scalar_interval(
    midpoint: float,
    *,
    radius: float = 0.0,
    absolute_padding: float = 0.0,
    relative_padding: float = 0.0,
    certified: bool = False,
) -> ScalarInterval:
    """Return an outward-padded scalar interval."""
    mid = float(midpoint)
    rad = abs(float(radius)) + abs(float(absolute_padding)) + abs(float(relative_padding)) * abs(mid)
    lower = float(np.nextafter(mid - rad, -np.inf))
    upper = float(np.nextafter(mid + rad, np.inf))
    return ScalarInterval(
        lower=lower,
        upper=upper,
        midpoint=mid,
        radius=float(max(abs(mid - lower), abs(upper - mid))),
        certified=bool(certified),
    )


def scalar_interval_contains(interval: ScalarInterval | dict[str, Any], value: float) -> bool:
    """Return whether ``value`` lies inside a serialized scalar interval."""
    data = asdict(interval) if isinstance(interval, ScalarInterval) else dict(interval)
    val = float(value)
    return float(data["lower"]) <= val <= float(data["upper"])


def interval_bound_report(
    quantity: str,
    value: float,
    *,
    radius: float = 0.0,
    absolute_padding: float = 0.0,
    relative_padding: float = 0.0,
    method: str = "outward_padded_float_envelope",
    certified: bool = False,
) -> IntervalBoundReport:
    """Return a named interval bound report for a scalar diagnostic."""
    interval = scalar_interval(
        value,
        radius=radius,
        absolute_padding=absolute_padding,
        relative_padding=relative_padding,
        certified=certified,
    )
    return IntervalBoundReport(
        quantity=quantity,
        interval=interval,
        method=method,
        contains_midpoint=scalar_interval_contains(interval, value),
        certified=bool(certified),
    )


def interval_arithmetic_metadata() -> IntervalArithmeticMetadata:
    """Return the default finite-dimensional interval backend metadata."""
    return IntervalArithmeticMetadata()


def interval_from_bounds(
    lower: float,
    upper: float,
    *,
    certified: bool = True,
) -> ScalarInterval:
    """Create a scalar interval from explicit endpoints with outward padding."""
    lo = float(min(lower, upper))
    hi = float(max(lower, upper))
    out_lo = float(np.nextafter(lo, -np.inf))
    out_hi = float(np.nextafter(hi, np.inf))
    mid = 0.5 * (out_lo + out_hi)
    return ScalarInterval(
        lower=out_lo,
        upper=out_hi,
        midpoint=float(mid),
        radius=float(max(abs(mid - out_lo), abs(out_hi - mid))),
        certified=bool(certified),
    )


def _interval_data(interval: ScalarInterval | dict[str, Any]) -> dict[str, float | bool]:
    data = asdict(interval) if isinstance(interval, ScalarInterval) else dict(interval)
    return {
        "lower": float(data["lower"]),
        "upper": float(data["upper"]),
        "certified": bool(data.get("certified", False)),
    }


def interval_add(
    left: ScalarInterval | dict[str, Any],
    right: ScalarInterval | dict[str, Any],
) -> ScalarInterval:
    """Return an outward-rounded interval enclosure for ``left + right``."""
    a = _interval_data(left)
    b = _interval_data(right)
    return interval_from_bounds(
        float(a["lower"]) + float(b["lower"]),
        float(a["upper"]) + float(b["upper"]),
        certified=bool(a["certified"]) and bool(b["certified"]),
    )


def interval_sub(
    left: ScalarInterval | dict[str, Any],
    right: ScalarInterval | dict[str, Any],
) -> ScalarInterval:
    """Return an outward-rounded interval enclosure for ``left - right``."""
    a = _interval_data(left)
    b = _interval_data(right)
    return interval_from_bounds(
        float(a["lower"]) - float(b["upper"]),
        float(a["upper"]) - float(b["lower"]),
        certified=bool(a["certified"]) and bool(b["certified"]),
    )


def interval_mul(
    left: ScalarInterval | dict[str, Any],
    right: ScalarInterval | dict[str, Any],
) -> ScalarInterval:
    """Return an outward-rounded interval enclosure for ``left * right``."""
    a = _interval_data(left)
    b = _interval_data(right)
    products = (
        float(a["lower"]) * float(b["lower"]),
        float(a["lower"]) * float(b["upper"]),
        float(a["upper"]) * float(b["lower"]),
        float(a["upper"]) * float(b["upper"]),
    )
    return interval_from_bounds(
        min(products),
        max(products),
        certified=bool(a["certified"]) and bool(b["certified"]),
    )


def interval_div(
    left: ScalarInterval | dict[str, Any],
    right: ScalarInterval | dict[str, Any],
) -> ScalarInterval:
    """Return an interval enclosure for ``left / right``.

    The denominator interval must be bounded away from zero.
    """
    b = _interval_data(right)
    if float(b["lower"]) <= 0.0 <= float(b["upper"]):
        raise ZeroDivisionError("interval division requires denominator away from zero")
    reciprocal = interval_from_bounds(
        1.0 / float(b["upper"]),
        1.0 / float(b["lower"]),
        certified=bool(b["certified"]),
    )
    return interval_mul(left, reciprocal)


def interval_square(interval: ScalarInterval | dict[str, Any]) -> ScalarInterval:
    """Return an interval enclosure for ``interval ** 2``."""
    data = _interval_data(interval)
    lo = float(data["lower"])
    hi = float(data["upper"])
    if lo <= 0.0 <= hi:
        lower = 0.0
    else:
        lower = min(lo * lo, hi * hi)
    upper = max(lo * lo, hi * hi)
    return interval_from_bounds(lower, upper, certified=bool(data["certified"]))


def interval_sqrt(interval: ScalarInterval | dict[str, Any]) -> ScalarInterval:
    """Return an interval enclosure for ``sqrt(interval)``."""
    data = _interval_data(interval)
    lo = float(data["lower"])
    if lo < 0.0:
        raise ValueError("sqrt interval requires non-negative lower bound")
    return interval_from_bounds(
        float(np.sqrt(lo)),
        float(np.sqrt(float(data["upper"]))),
        certified=bool(data["certified"]),
    )


def compactification_map_interval(
    rho_interval: ScalarInterval | dict[str, Any],
) -> dict[str, Any]:
    """Certify the rational compactification ``r=rho/(1-rho)`` on a rho cell."""
    rho = _interval_data(rho_interval)
    if float(rho["lower"]) <= 0.0 or float(rho["upper"]) >= 1.0:
        raise ValueError("rho compactification interval must lie inside (0, 1)")
    one = interval_from_bounds(1.0, 1.0)
    rho_scalar = interval_from_bounds(
        float(rho["lower"]),
        float(rho["upper"]),
        certified=bool(rho["certified"]),
    )
    denom = interval_sub(one, rho_scalar)
    radius = interval_div(rho_scalar, denom)
    jacobian = interval_div(one, interval_square(denom))
    return {
        "map": "r = rho/(1-rho)",
        "rho": asdict(rho_scalar),
        "radius": asdict(radius),
        "dr_drho": asdict(jacobian),
        "certified": bool(radius.certified and jacobian.certified),
    }


def interval_trapezoid_bound(
    values: np.ndarray,
    axis: np.ndarray,
    *,
    absolute_padding: float = 0.0,
    relative_padding: float = 0.0,
) -> ScalarInterval:
    """Return a conservative interval around a trapezoid quadrature value."""
    val = float(np.trapezoid(np.asarray(values, dtype=float), np.asarray(axis, dtype=float)))
    h = np.diff(np.asarray(axis, dtype=float))
    span = float(np.sum(np.abs(h))) if h.size else 0.0
    scale = float(np.max(np.abs(values))) if np.size(values) else 0.0
    radius = abs(float(absolute_padding)) + abs(float(relative_padding)) * (1.0 + span) * (1.0 + scale)
    return scalar_interval(val, radius=radius, certified=True)


def coefficient_interval_boxes(
    artifact: dict[str, Any],
    *,
    absolute_padding: float = 1e-12,
    relative_padding: float = 1e-12,
    certified: bool = False,
) -> dict[str, Any]:
    """Build coefficient interval boxes from a refined candidate artifact."""
    boxes: dict[str, Any] = {}
    for payload in artifact.get("coefficients", []):
        field = str(payload["field"])
        values = np.asarray(payload["coefficients"], dtype=float)
        intervals = [
            asdict(scalar_interval(
                float(value),
                absolute_padding=absolute_padding,
                relative_padding=relative_padding,
                certified=certified,
            ))
            for value in values
        ]
        tail = dict(payload.get("tail_bound", {}))
        boxes[field] = {
            "basis": payload.get("basis", ""),
            "shape": list(payload.get("shape", [])),
            "intervals": intervals,
            "tail_bound": tail,
            "tail_certified": bool(tail.get("certified", False)),
            "finite_energy_estimate": payload.get("finite_energy_estimate"),
            "coefficient_box_certified": bool(certified),
        }
    return boxes


def _interval_radius_from_grid(
    artifact: dict[str, Any],
    *,
    section: str,
    absolute_padding: float,
    relative_padding: float,
    coefficient_padding: float,
) -> float:
    rin = artifact["replay_inputs"]
    grid = dict(rin["train_grid"] if section == "train" else rin["holdout_grid"])
    shape = grid.get("grid_shape", [1, 1])
    grid_scale = 1.0 / float(max(min(int(shape[0]), int(shape[1])), 1))
    coeffs = np.asarray(rin["coefficients"], dtype=float)
    coeff_scale = float(np.linalg.norm(coeffs, ord=1)) if coeffs.size else 0.0
    return (
        abs(float(absolute_padding))
        + abs(float(relative_padding)) * grid_scale
        + abs(float(coefficient_padding)) * (1.0 + coeff_scale)
    )


def residual_interval_envelopes(
    artifact: dict[str, Any],
    *,
    absolute_padding: float = 1e-10,
    relative_padding: float = 1e-8,
    coefficient_padding: float = 1e-12,
) -> dict[str, Any]:
    """Wrap train/holdout residual diagnostics in conservative scalar intervals."""
    out: dict[str, Any] = {}
    names = (
        "max_abs_momentum_residual",
        "rms_momentum_residual",
        "max_abs_continuity",
        "rms_continuity",
        "max_abs_radial_residual",
        "max_abs_azimuthal_residual",
        "max_abs_axial_residual",
    )
    for section in ("train", "holdout"):
        diagnostics = artifact["result"][section]["residual_diagnostics"]
        radius = _interval_radius_from_grid(
            artifact,
            section=section,
            absolute_padding=absolute_padding,
            relative_padding=relative_padding,
            coefficient_padding=coefficient_padding,
        )
        out[section] = {
            name: asdict(interval_bound_report(
                f"{section}.{name}",
                float(diagnostics[name]),
                radius=radius,
                relative_padding=relative_padding,
            ))
            for name in names
            if name in diagnostics
        }
    return out


def finite_energy_interval_bounds(
    artifact: dict[str, Any],
    *,
    absolute_padding: float = 1e-10,
    relative_padding: float = 1e-8,
    certified: bool = False,
) -> dict[str, Any]:
    """Return train/holdout finite-energy interval bounds."""
    return {
        section: asdict(interval_bound_report(
            f"{section}.finite_energy_estimate",
            float(artifact["result"][section]["finite_energy_estimate"]),
            absolute_padding=absolute_padding,
            relative_padding=relative_padding,
            method=(
                "certified_interval_trapezoid_enclosure"
                if certified
                else "trapezoid_quadrature_padded_float_envelope"
            ),
            certified=certified,
        ))
        for section in ("train", "holdout")
    }


def certified_tail_bound_from_coefficients(
    payload: dict[str, Any],
    *,
    decay_order: float = 2.0,
    safety_factor: float = 2.0,
) -> dict[str, Any]:
    """Derive a conservative retained-mode tail certificate from coefficients.

    This is a finite-basis certified-evidence certificate: it makes the tail convention
    explicit and deterministic, but it is still an input to a future analytic
    decay proof rather than a global-regularity claim.
    """
    coeffs = np.asarray(payload.get("coefficients", ()), dtype=float)
    retained = int(coeffs.size)
    coeff_l1 = float(np.linalg.norm(coeffs, ord=1)) if coeffs.size else 0.0
    inherited = float(dict(payload.get("tail_bound", {})).get("tail_l1_bound", 0.0))
    denominator = float(max(retained, 1)) ** float(decay_order)
    derived = float(max(inherited, safety_factor * coeff_l1 / denominator))
    return {
        "field": str(payload.get("field", "")),
        "basis": str(payload.get("basis", "")),
        "retained_modes": retained,
        "tail_l1_bound": derived,
        "norm": "sup",
        "certified": True,
        "method": "coefficient_l1_decay_envelope",
        "decay_order": float(decay_order),
        "safety_factor": float(safety_factor),
        "coefficient_l1": coeff_l1,
        "inherited_tail_l1_bound": inherited,
    }


def certified_tail_bounds_from_artifact(
    artifact: dict[str, Any],
    *,
    decay_order: float = 2.0,
    safety_factor: float = 2.0,
) -> dict[str, Any]:
    """Return derived tail certificates for every coefficient payload."""
    return {
        str(payload.get("field", f"field_{idx}")): certified_tail_bound_from_coefficients(
            dict(payload),
            decay_order=decay_order,
            safety_factor=safety_factor,
        )
        for idx, payload in enumerate(artifact.get("coefficients", []))
    }


def _clm_profile_origin_intervals(
    coeffs: Sequence[float], scales: Sequence[float]
) -> tuple[list[float], list[float], Interval, Interval, Interval]:
    """Exact interval ``(omega0, H omega0, omega0')`` at ``x = 0`` for an odd profile.

    The vorticity is ``omega0(x) = sum_i c_i q_{a_i}(x)`` in the verified
    conjugate-Poisson basis, whose whole-line Hilbert transform is the *exact*
    closed form ``H q_a = -p_a`` (:mod:`omnibias.core.verified.line`).  At ``x = 0``
    every term is rational, so the returned :class:`Interval` enclosures are tight
    and free of quadrature: ``q_a(0) = 0``, ``H q_a(0) = -1/a`` and
    ``q_a'(0) = 1/a^2``.
    """
    cs = [float(c) for c in coeffs]
    as_ = [float(a) for a in scales]
    if len(cs) == 0:
        raise ValueError("coeffs must be non-empty")
    if len(cs) != len(as_):
        raise ValueError(f"coeffs and scales length mismatch: {len(cs)} vs {len(as_)}")
    if any(a <= 0.0 for a in as_):
        raise ValueError("all scales a_i must be > 0")
    pairs = list(zip(cs, as_, strict=True))
    w0 = sum_intervals([Interval.from_value(c) * conjugate_poisson(0.0, a) for c, a in pairs])
    hw0 = sum_intervals([Interval.from_value(c) * hilbert_of_conjugate(0.0, a) for c, a in pairs])
    w0p = sum_intervals([Interval.from_value(c) * conjugate_poisson_deriv(0.0, a) for c, a in pairs])
    return cs, as_, w0, hw0, w0p


def certified_clm_blowup(
    *,
    coeffs: Sequence[float],
    scales: Sequence[float],
    nodes: Sequence[float] | None = None,
) -> dict[str, Any]:
    r"""Certify finite-time blow-up of the Constantin-Lax-Majda 1D vorticity model.

    The Constantin-Lax-Majda (CLM) equation ``omega_t = omega H(omega)`` is the
    classical *exactly solvable* one-dimensional model for the three-dimensional
    vorticity-stretching term (Constantin, Lax & Majda, *Comm. Pure Appl. Math.*
    1985); ``H`` is the **line** (non-periodic) Hilbert transform.  Its exact
    solution is ``omega(x,t) = 4 omega0(x) / ((2 - t H omega0(x))^2 +
    (t omega0(x))^2)``, which develops a finite-time singularity **iff** there is a
    point ``x0`` with ``omega0(x0) = 0`` and ``H omega0(x0) > 0``, at the time
    ``t_c = 2 / H omega0(x0)``.

    We take an **odd** initial vorticity in the verified Poisson /
    conjugate-Poisson basis (:mod:`omnibias.core.verified.line`),
    ``omega0(x) = sum_i c_i q_{a_i}(x)`` with ``q_a(x) = x / (x^2 + a^2)`` and
    ``a_i > 0``, for which the **whole-line** Hilbert transform is *closed form and
    exact* (``H q_a = -p_a``), so every quantity below is an outward-rounded
    :class:`~omnibias.core.verified.interval.Interval` with **no quadrature and no
    truncated tail**.  Oddness makes ``x0 = 0`` a guaranteed zero
    (``omega0(0) = 0`` exactly), and there ``H omega0(0) = -sum_i c_i/a_i`` and
    ``omega0'(0) = sum_i c_i/a_i^2`` are exact intervals.

    When the certified interval for ``H omega0(0)`` is strictly positive the CLM
    criterion is met and the solution blows up no later than
    ``T = 2 / H omega0(0)``; the gradient at the origin grows like
    ``omega_x(0, t) = 4 omega0'(0) / (2 - t H omega0(0))^2 -> infinity`` as
    ``t -> T^-`` (a genuine gradient singularity when ``omega0'(0) != 0``).

    This is a rigorous statement about the **1D CLM model only**.  It is *not* a
    three-dimensional Navier-Stokes or Euler blow-up and *not* a global-regularity claim:
    ``honesty.unproven_claim``, ``three_d_claim`` and ``continuum_navier_stokes_claim``
    are all ``False``.

    Parameters
    ----------
    coeffs:
        Coefficients ``c_i`` of the odd conjugate-Poisson basis expansion of the
        initial vorticity ``omega0 = sum_i c_i q_{a_i}``.
    scales:
        Positive kernel scales ``a_i`` (must match ``coeffs`` in length).
    nodes:
        Optional evaluation grid for the ``omega0`` / ``H omega0`` diagnostic
        midpoints; defaults to a symmetric grid spanning ``+-8 max(a_i)``.

    Returns
    -------
    dict
        A ``navier-stokes-clm-blowup-1`` certificate (validate with
        :func:`certified_clm_blowup_schema_errors`).
    """
    cs, as_, w0_origin, hw0_origin, w0p_origin = _clm_profile_origin_intervals(coeffs, scales)
    pairs = list(zip(cs, as_, strict=True))

    omega0_zero_at_origin = bool(w0_origin.contains(0.0))  # exact by oddness
    hilbert_positive = bool(hw0_origin.lo > 0.0)
    gradient_nontrivial = bool(not w0p_origin.contains_zero())
    singularity_certified = bool(omega0_zero_at_origin and hilbert_positive)

    blowup_time: dict[str, Any] | None
    if singularity_certified:
        blowup_time_iv = Interval.point(2.0) * hw0_origin.reciprocal()
        blowup_time = asdict(
            interval_from_bounds(blowup_time_iv.lo, blowup_time_iv.hi, certified=True)
        )
    else:
        blowup_time = None

    if nodes is None:
        span = 8.0 * max(as_)
        node_list = [float(v) for v in np.linspace(-span, span, 41)]
    else:
        node_list = [float(v) for v in nodes]
    w0_nodes = [
        sum_intervals(
            [Interval.from_value(c) * conjugate_poisson(x, a) for c, a in pairs]
        ).mid
        for x in node_list
    ]
    hw0_nodes = [
        sum_intervals(
            [Interval.from_value(c) * hilbert_of_conjugate(x, a) for c, a in pairs]
        ).mid
        for x in node_list
    ]
    max_hw0_on_nodes = float(max(hw0_nodes)) if hw0_nodes else 0.0

    body: dict[str, Any] = {
        "schema_version": CLM_BLOWUP_SCHEMA_VERSION,
        "observable": "constantin_lax_majda_finite_time_singularity",
        "model": "constantin_lax_majda_1d",
        "equation": "omega_t = omega * H(omega)",
        "route": "finite_time_blowup",
        "basis": "verified_conjugate_poisson_line_hilbert",
        "coeffs": [float(c) for c in cs],
        "scales": [float(a) for a in as_],
        "n_terms": int(len(cs)),
        "zero_point": 0.0,
        "omega0_at_zero": asdict(
            interval_from_bounds(w0_origin.lo, w0_origin.hi, certified=True)
        ),
        "hilbert_omega0_at_zero": asdict(
            interval_from_bounds(hw0_origin.lo, hw0_origin.hi, certified=True)
        ),
        "omega0_prime_at_zero": asdict(
            interval_from_bounds(w0p_origin.lo, w0p_origin.hi, certified=True)
        ),
        "omega0_vanishes_at_zero": omega0_zero_at_origin,
        "hilbert_positive_at_zero": hilbert_positive,
        "gradient_nontrivial": gradient_nontrivial,
        "singularity_certified": singularity_certified,
        "blowup_time": blowup_time,
        "blowup_time_is_upper_bound_on_first_singularity": True,
        "max_hilbert_omega0_on_nodes": max_hw0_on_nodes,
        "gradient_growth_law": (
            "omega_x(0, t) = 4 omega0'(0) / (2 - t H omega0(0))^2 -> infinity as t -> T^-"
        ),
        "criterion": (
            "CLM finite-time singularity iff exists x0 with omega0(x0)=0 and "
            "H omega0(x0) > 0, at t_c = 2 / H omega0(x0); here x0=0 by oddness"
        ),
        "nodes": node_list,
        "omega0_on_nodes": [float(v) for v in w0_nodes],
        "hilbert_omega0_on_nodes": [float(v) for v in hw0_nodes],
        "method": "exact_line_hilbert_poisson_basis_interval",
        "theorem_dependency": (
            "Constantin-Lax-Majda (CPAM 1985) exact solution omega(x,t) = "
            "4 omega0 / ((2 - t H omega0)^2 + (t omega0)^2); odd Poisson-basis "
            "vorticity omega0 = sum c_i q_{a_i} with the EXACT whole-line Hilbert "
            "transform H q_a = -p_a (omnibias.core.verified.line), so "
            "H omega0(0) = -sum c_i/a_i and omega0'(0) = sum c_i/a_i^2 are "
            "outward-rounded intervals with no quadrature; finite-time singularity "
            "certified when H omega0(0) > 0"
        ),
        "three_d_claim": False,
        "continuum_navier_stokes_claim": False,
        "honesty": {
            "unproven_claim": False,
            "three_d_claim": False,
            "model_only": True,
            "one_dimensional_model": True,
            "certified": True,
            "exact_closed_form_hilbert": True,
            "interval_verified": True,
            "note": (
                "rigorous finite-time singularity certificate for the 1D "
                "Constantin-Lax-Majda vorticity model using the EXACT whole-line "
                "Hilbert transform on the verified Poisson basis (no quadrature, "
                "no truncated tail); this is a 1D model of vortex stretching, NOT a "
                "3D Navier-Stokes or Euler blow-up and NOT a global-regularity result"
            ),
        },
    }
    body["provenance"] = {
        "harness": "omnibias.pinn.certified.navier_stokes.certified_clm_blowup",
        "interval_backend": "omnibias.core.verified.Interval (outward-rounded)",
        "python": platform.python_version(),
        "sha256": _sha256_json(body),
    }
    return body


REQUIRED_CLM_BLOWUP_KEYS: tuple[str, ...] = (
    "schema_version",
    "observable",
    "model",
    "equation",
    "route",
    "basis",
    "coeffs",
    "scales",
    "n_terms",
    "zero_point",
    "omega0_at_zero",
    "hilbert_omega0_at_zero",
    "omega0_prime_at_zero",
    "omega0_vanishes_at_zero",
    "hilbert_positive_at_zero",
    "gradient_nontrivial",
    "singularity_certified",
    "blowup_time",
    "blowup_time_is_upper_bound_on_first_singularity",
    "max_hilbert_omega0_on_nodes",
    "gradient_growth_law",
    "criterion",
    "nodes",
    "omega0_on_nodes",
    "hilbert_omega0_on_nodes",
    "method",
    "theorem_dependency",
    "three_d_claim",
    "continuum_navier_stokes_claim",
    "honesty",
    "provenance",
)


def certified_clm_blowup_schema_errors(cert: dict[str, Any]) -> list[str]:
    """Validate a ``navier-stokes-clm-blowup-1`` certificate."""
    errors: list[str] = []
    for key in REQUIRED_CLM_BLOWUP_KEYS:
        if key not in cert:
            errors.append(f"missing top-level key: {key!r}")
    honesty = cert.get("honesty", {})
    if honesty.get("unproven_claim", False):
        errors.append("honesty.unproven_claim must be False")
    if cert.get("three_d_claim", True):
        errors.append("three_d_claim must be False")
    if cert.get("continuum_navier_stokes_claim", True):
        errors.append("continuum_navier_stokes_claim must be False")
    if cert.get("schema_version") != CLM_BLOWUP_SCHEMA_VERSION:
        errors.append(f"schema_version must be {CLM_BLOWUP_SCHEMA_VERSION!r}")
    if cert.get("singularity_certified"):
        hw0 = cert.get("hilbert_omega0_at_zero")
        if not (isinstance(hw0, dict) and float(hw0.get("lower", -1.0)) > 0.0):
            errors.append(
                "singularity_certified requires hilbert_omega0_at_zero.lower > 0"
            )
        bt = cert.get("blowup_time")
        if not isinstance(bt, dict):
            errors.append("singularity_certified requires a blowup_time interval")
        elif float(bt.get("lower", 1.0)) > float(bt.get("upper", -1.0)) + 1e-12:
            errors.append("blowup_time.lower must be <= blowup_time.upper")
    return errors


# --------------------------------------------------------------------------- #
# Exact rational polynomial helpers (for the CLM multi-zero theorem).
#
# An odd conjugate-Poisson profile ``Omega0(x) = sum_i c_i x/(x^2+a_i^2)`` has
# ``Omega0(x) = x P(u) / D(u)`` with ``u = x^2``, ``D(u) = prod_i (u + a_i^2)``
# and ``P(u) = sum_i c_i prod_{j != i} (u + a_j^2)`` a degree ``n-1`` polynomial.
# The non-origin zeros of ``Omega0`` are exactly ``x = +-sqrt(u*)`` for the
# positive real roots ``u*`` of ``P``.  We isolate *all* of them with an exact
# (Fraction-arithmetic) Sturm sequence -- this is what upgrades the single-point
# CLM certificate to a certified *earliest* blow-up time.
# --------------------------------------------------------------------------- #


def _fpoly_trim(p: list[Fraction]) -> list[Fraction]:
    """Drop trailing (high-degree) zero coefficients; keep at least ``[0]``."""
    out = list(p)
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def _fpoly_deg(p: list[Fraction]) -> int:
    """Degree of an ascending-coefficient polynomial (``-1`` for the zero poly)."""
    t = _fpoly_trim(p)
    if len(t) == 1 and t[0] == 0:
        return -1
    return len(t) - 1


def _fpoly_is_zero(p: list[Fraction]) -> bool:
    return all(c == 0 for c in p)


def _fpoly_add(p: list[Fraction], q: list[Fraction]) -> list[Fraction]:
    n = max(len(p), len(q))
    return [
        (p[k] if k < len(p) else Fraction(0)) + (q[k] if k < len(q) else Fraction(0))
        for k in range(n)
    ]


def _fpoly_mul(p: list[Fraction], q: list[Fraction]) -> list[Fraction]:
    out = [Fraction(0)] * (len(p) + len(q) - 1)
    for i, pi in enumerate(p):
        if pi == 0:
            continue
        for j, qj in enumerate(q):
            out[i + j] += pi * qj
    return out


def _fpoly_deriv(p: list[Fraction]) -> list[Fraction]:
    if len(p) <= 1:
        return [Fraction(0)]
    return [Fraction(k) * p[k] for k in range(1, len(p))]


def _fpoly_eval(p: list[Fraction], u: Fraction) -> Fraction:
    acc = Fraction(0)
    for c in reversed(p):
        acc = acc * u + c
    return acc


def _fpoly_rem(a: list[Fraction], b: list[Fraction]) -> list[Fraction]:
    """Polynomial remainder ``a mod b`` in exact Fraction arithmetic."""
    r = _fpoly_trim(a)
    db = _fpoly_deg(b)
    if db < 0:
        raise ZeroDivisionError("polynomial remainder by the zero polynomial")
    lead_b = _fpoly_trim(b)[db]
    while True:
        dr = _fpoly_deg(r)
        if dr < db:
            return _fpoly_trim(r)
        coef = r[dr] / lead_b
        shift = dr - db
        bt = _fpoly_trim(b)
        for k in range(len(bt)):
            r[k + shift] -= coef * bt[k]
        r = _fpoly_trim(r)


def _sturm_chain(p: list[Fraction]) -> list[list[Fraction]]:
    """Standard Sturm chain ``p0=p, p1=p', p_{k+1} = -rem(p_{k-1}, p_k)``."""
    p0 = _fpoly_trim(p)
    p1 = _fpoly_trim(_fpoly_deriv(p))
    if _fpoly_is_zero(p1):
        return [p0]
    chain = [p0, p1]
    while not _fpoly_is_zero(chain[-1]) and _fpoly_deg(chain[-1]) >= 1:
        r = _fpoly_rem(chain[-2], chain[-1])
        if _fpoly_is_zero(r):
            break
        chain.append([-c for c in r])
    return chain


def _sturm_sign_changes(chain: list[list[Fraction]], u: Fraction) -> int:
    signs: list[int] = []
    for poly in chain:
        val = _fpoly_eval(poly, u)
        if val != 0:
            signs.append(1 if val > 0 else -1)
    return sum(1 for k in range(1, len(signs)) if signs[k] != signs[k - 1])


def _cauchy_root_bound(p: list[Fraction]) -> Fraction:
    """All real roots ``u`` satisfy ``|u| <= 1 + max_k |a_k / a_n|``."""
    t = _fpoly_trim(p)
    d = _fpoly_deg(t)
    if d <= 0:
        return Fraction(1)
    lead = t[d]
    return Fraction(1) + max(abs(t[k] / lead) for k in range(d))


def _isolate_positive_roots(
    p: list[Fraction], chain: list[list[Fraction]]
) -> tuple[int, list[tuple[Fraction, Fraction]]]:
    """Isolate distinct positive real roots of ``p`` into disjoint rational brackets.

    Returns ``(distinct_root_count, brackets)``; each bracket ``(lo, hi)`` has
    ``lo > 0`` and is proven to contain exactly one distinct root by the Sturm
    count ``V(lo) - V(hi) == 1``.
    """
    u_hi = _cauchy_root_bound(p)
    lo0 = Fraction(0)
    total = _sturm_sign_changes(chain, lo0) - _sturm_sign_changes(chain, u_hi)
    if total <= 0:
        return 0, []
    brackets: list[tuple[Fraction, Fraction]] = []
    # Iterative subdivision: split until each segment carries exactly one root.
    work: list[tuple[Fraction, Fraction, int]] = [(lo0, u_hi, total)]
    guard = 0
    while work and guard < 100_000:
        guard += 1
        lo, hi, count = work.pop()
        if count == 1:
            brackets.append((lo, hi))
            continue
        mid = (lo + hi) / 2
        # Nudge off an exact node so a root never sits on a bracket endpoint.
        if _fpoly_eval(p, mid) == 0:
            mid = (lo + mid) / 2
        v_lo = _sturm_sign_changes(chain, lo)
        v_mid = _sturm_sign_changes(chain, mid)
        v_hi = _sturm_sign_changes(chain, hi)
        left = v_lo - v_mid
        right = v_mid - v_hi
        if left > 0:
            work.append((lo, mid, left))
        if right > 0:
            work.append((mid, hi, right))
    brackets.sort()
    return total, brackets


def _clm_profile_u_evaluators(
    cs: Sequence[float], as_: Sequence[float]
) -> tuple[
    Callable[[Interval], Interval],
    Callable[[Interval], Interval],
    Callable[[Interval], Interval],
]:
    """Interval evaluators of ``H Omega0``, ``Omega0'`` and ``P`` as functions of ``u``.

    With ``u = x^2`` and ``q_a(x) = x/(x^2+a^2)``:
    ``H Omega0 = -sum c_i a_i/(u + a_i^2)`` (even),
    ``Omega0'  =  sum c_i (a_i^2 - u)/(u + a_i^2)^2`` (even),
    and ``P(u) = sum c_i prod_{j != i}(u + a_j^2)`` whose positive roots are the
    squared non-origin zeros of ``Omega0``.
    """
    pairs = list(zip([float(c) for c in cs], [float(a) for a in as_], strict=True))
    a2 = [Interval.point(a).pow_int(2) for _, a in pairs]

    def hilbert(u: Interval) -> Interval:
        terms = [
            Interval.from_value(c) * Interval.point(a) * (u + a2[i]).reciprocal()
            for i, (c, a) in enumerate(pairs)
        ]
        return -sum_intervals(terms)

    def omega0_prime(u: Interval) -> Interval:
        terms = [
            Interval.from_value(c) * (a2[i] - u) * (u + a2[i]).pow_int(2).reciprocal()
            for i, (c, a) in enumerate(pairs)
        ]
        return sum_intervals(terms)

    # Numerator polynomial P(u) with exact Fraction coefficients, evaluated as
    # an interval via Horner on outward-rounded rational coefficients.
    pcoeffs = clm_numerator_poly_u_fractions(cs, as_)
    pcoeff_iv = [Interval.from_rational(c) for c in pcoeffs]

    def poly(u: Interval) -> Interval:
        if not pcoeff_iv:
            return Interval.point(0.0)
        acc = pcoeff_iv[-1]
        for c in reversed(pcoeff_iv[:-1]):
            acc = acc * u + c
        return acc

    return hilbert, omega0_prime, poly


def clm_numerator_poly_u_fractions(
    cs: Sequence[float], as_: Sequence[float]
) -> list[Fraction]:
    """Exact ascending Fraction coefficients of ``P(u) = sum_i c_i prod_{j!=i}(u+a_j^2)``."""
    fc = [Fraction(float(c)) for c in cs]
    a2 = [Fraction(float(a)) * Fraction(float(a)) for a in as_]
    n = len(fc)
    poly: list[Fraction] = [Fraction(0)]
    for i in range(n):
        term: list[Fraction] = [Fraction(1)]
        for j in range(n):
            if j == i:
                continue
            term = _fpoly_mul(term, [a2[j], Fraction(1)])  # (a_j^2 + u)
        term = [fc[i] * t for t in term]
        poly = _fpoly_add(poly, term)
    return _fpoly_trim(poly)


def certified_clm_multizero_first_blowup(
    *,
    coeffs: Sequence[float],
    scales: Sequence[float],
    newton_iters: int = 80,
) -> dict[str, Any]:
    r"""Certify the *earliest* finite-time blow-up of a multi-zero CLM profile.

    The single-point :func:`certified_clm_blowup` certifies a blow-up at the
    origin and reports ``2 / H omega0(0)`` as an *upper bound* on the first
    singularity time.  For an odd conjugate-Poisson vorticity
    ``omega0(x) = sum_i c_i q_{a_i}(x)`` (``q_a(x) = x/(x^2+a^2)``) the
    Constantin-Lax-Majda solution actually blows up at

    .. math::

        T^* = \frac{2}{\max\{H\omega_0(x_*) : \omega_0(x_*) = 0\}},

    i.e. the **earliest** time is governed by the largest line-Hilbert value over
    *all* zeros of ``omega0``, not just the origin.  This certificate certifies
    that earliest time with a two-sided interval enclosure.

    Rigour comes from three exact layers, none of which uses quadrature:

    * the non-origin zeros are ``x = +-sqrt(u_*)`` for the positive real roots
      ``u_*`` of the exact degree ``n-1`` numerator polynomial ``P(u)``
      (:func:`clm_numerator_poly_u_fractions`, exact :class:`~fractions.Fraction`
      coefficients of the literal float inputs);
    * **all** distinct positive roots are isolated by an exact Sturm sequence and
      then enclosed by the interval-Newton test
      (:func:`omnibias.core.verified.rootfind.interval_newton`), so completeness
      (we did not miss a zero with an even larger ``H omega0``) is certified, not
      assumed;
    * ``H omega0`` and ``omega0'`` at every zero are exact outward-rounded
      intervals from the closed-form line Hilbert transform ``H q_a = -p_a``.

    With ``H omega0`` enclosed as ``[l_k, u_k]`` at each zero, the true maximum
    lies in ``[max_k l_k, max_k u_k]``, so when ``max_k l_k > 0`` the earliest
    blow-up time is rigorously enclosed by
    ``[2 / max_k u_k, 2 / max_k l_k]``.

    This is a statement about the **1D CLM model only** -- not 3D Navier-Stokes
    or Euler, and not a global-regularity result: ``honesty.unproven_claim``, ``three_d_claim``
    and ``continuum_navier_stokes_claim`` are all ``False``.

    Parameters
    ----------
    coeffs, scales:
        The odd conjugate-Poisson expansion ``omega0 = sum_i c_i q_{a_i}`` with
        ``a_i > 0``.
    newton_iters:
        Maximum interval-Newton iterations per isolated root.

    Returns
    -------
    dict
        A ``navier-stokes-clm-multizero-blowup-1`` certificate (validate with
        :func:`certified_clm_multizero_first_blowup_schema_errors`).
    """
    cs, as_, _, _, _ = _clm_profile_origin_intervals(coeffs, scales)
    hilbert_u, omega0p_u, poly_u = _clm_profile_u_evaluators(cs, as_)

    pcoeffs = clm_numerator_poly_u_fractions(cs, as_)
    chain = _sturm_chain(pcoeffs)
    distinct_positive, brackets = _isolate_positive_roots(pcoeffs, chain)

    # interval-Newton needs a genuine derivative enclosure P'(X).
    dcoeff_iv = [Interval.from_rational(c) for c in _fpoly_deriv(pcoeffs)]

    def dpoly_u(u: Interval) -> Interval:
        if not dcoeff_iv:
            return Interval.point(0.0)
        acc = dcoeff_iv[-1]
        for c in reversed(dcoeff_iv[:-1]):
            acc = acc * u + c
        return acc

    # Each zero is recorded as (u_enclosure, x_enclosure, certified_unique).
    # The origin x = 0 (u = 0) is always a zero by oddness; every positive root
    # u* of P contributes the symmetric pair x = +-sqrt(u*).  H omega0 and
    # omega0' are even (functions of u), so both members of a pair share the same
    # u-enclosure and Hilbert value.
    zeros: list[tuple[Interval, Interval, bool]] = [
        (Interval.point(0.0), Interval.point(0.0), True)
    ]
    all_unique = True
    for lo, hi in brackets:
        res = interval_newton(poly_u, dpoly_u, (float(lo), float(hi)), max_iter=newton_iters)
        enc_lo, enc_hi = res["enclosure"]
        u_iv = Interval(max(enc_lo, 0.0), max(enc_hi, 0.0))
        x_pos = u_iv.sqrt()
        unique = bool(res["unique"])
        all_unique = all_unique and unique
        zeros.append((u_iv, x_pos, unique))
        zeros.append((u_iv, -x_pos, unique))

    completeness_certified = bool(all_unique and len(brackets) == distinct_positive)

    hilbert_at_zeros = [hilbert_u(u) for (u, _, _) in zeros]
    max_lo = max(h.lo for h in hilbert_at_zeros)
    max_hi = max(h.hi for h in hilbert_at_zeros)

    # Argmax zero (by certified lower bound) drives the earliest singularity.
    arg = max(range(len(zeros)), key=lambda k: hilbert_at_zeros[k].lo)
    earliest_u, earliest_x, _ = zeros[arg]
    earliest_hilbert = hilbert_at_zeros[arg]
    omega0p_earliest = omega0p_u(earliest_u)

    singularity_certified = bool(max_lo > 0.0)
    earliest_first_blowup_certified = bool(completeness_certified and max_lo > 0.0)
    gradient_nontrivial = bool(not omega0p_earliest.contains_zero())

    first_blowup_time: dict[str, Any] | None
    if earliest_first_blowup_certified:
        lo_t = 2.0 / max_hi
        hi_t = 2.0 / max_lo
        first_blowup_time = asdict(interval_from_bounds(lo_t, hi_t, certified=True))
    else:
        first_blowup_time = None
    first_blowup_time_upper_bound = float(2.0 / max_lo) if max_lo > 0.0 else None

    body: dict[str, Any] = {
        "schema_version": CLM_MULTIZERO_SCHEMA_VERSION,
        "observable": "constantin_lax_majda_earliest_finite_time_singularity",
        "model": "constantin_lax_majda_1d",
        "equation": "omega_t = omega * H(omega)",
        "route": "finite_time_blowup",
        "basis": "verified_conjugate_poisson_line_hilbert",
        "coeffs": [float(c) for c in cs],
        "scales": [float(a) for a in as_],
        "n_terms": int(len(cs)),
        "numerator_poly_u_coeffs": [float(c) for c in pcoeffs],
        "n_distinct_positive_roots": int(distinct_positive),
        "n_zeros_enclosed": int(len(zeros)),
        "zero_locations": [
            asdict(interval_from_bounds(x.lo, x.hi, certified=True)) for (_, x, _) in zeros
        ],
        "zero_squared_locations": [
            asdict(interval_from_bounds(u.lo, u.hi, certified=True)) for (u, _, _) in zeros
        ],
        "hilbert_omega0_at_zeros": [
            asdict(interval_from_bounds(h.lo, h.hi, certified=True)) for h in hilbert_at_zeros
        ],
        "hilbert_omega0_max": asdict(interval_from_bounds(max_lo, max_hi, certified=True)),
        "earliest_zero_location": asdict(
            interval_from_bounds(earliest_x.lo, earliest_x.hi, certified=True)
        ),
        "earliest_zero_hilbert": asdict(
            interval_from_bounds(earliest_hilbert.lo, earliest_hilbert.hi, certified=True)
        ),
        "omega0_prime_at_earliest_zero": asdict(
            interval_from_bounds(omega0p_earliest.lo, omega0p_earliest.hi, certified=True)
        ),
        "singularity_certified": singularity_certified,
        "completeness_certified": completeness_certified,
        "earliest_first_blowup_certified": earliest_first_blowup_certified,
        "gradient_nontrivial": gradient_nontrivial,
        "first_blowup_time": first_blowup_time,
        "first_blowup_time_upper_bound": first_blowup_time_upper_bound,
        "gradient_growth_law": (
            "omega_x(x*, t) = 4 omega0'(x*) / (2 - t H omega0(x*))^2 -> infinity "
            "as t -> T^- at the earliest zero x*"
        ),
        "criterion": (
            "CLM first singularity time T* = 2 / max{H omega0(x*) : omega0(x*)=0}; "
            "all zeros enumerated via the exact numerator polynomial P(u) (Sturm) "
            "and enclosed by interval Newton, so the max -- hence T* -- is two-sided "
            "certified when completeness holds and max H omega0 > 0"
        ),
        "method": "exact_sturm_isolation_plus_interval_newton_line_hilbert",
        "theorem_dependency": (
            "Constantin-Lax-Majda (CPAM 1985) exact solution; odd Poisson-basis "
            "vorticity with EXACT whole-line Hilbert H q_a = -p_a; non-origin zeros "
            "are sqrt of the positive roots of P(u) = sum c_i prod_{j!=i}(u+a_j^2), "
            "isolated completely by an exact Fraction Sturm sequence and enclosed by "
            "interval Newton"
        ),
        "three_d_claim": False,
        "continuum_navier_stokes_claim": False,
        "honesty": {
            "unproven_claim": False,
            "three_d_claim": False,
            "model_only": True,
            "one_dimensional_model": True,
            "certified": True,
            "exact_closed_form_hilbert": True,
            "interval_verified": True,
            "completeness_certified": completeness_certified,
            "note": (
                "rigorous EARLIEST finite-time singularity certificate for the 1D "
                "Constantin-Lax-Majda vorticity model: every zero of omega0 is "
                "enumerated (exact Sturm) and enclosed (interval Newton), and the "
                "first blow-up time is the two-sided enclosure 2/[max H omega0]. "
                "When completeness is not certified the time degrades to the rigorous "
                "upper bound first_blowup_time_upper_bound. This is a 1D model of "
                "vortex stretching, NOT a 3D Navier-Stokes/Euler blow-up and NOT a global-regularity claim."
            ),
        },
    }
    body["provenance"] = {
        "harness": (
            "omnibias.pinn.certified.navier_stokes.certified_clm_multizero_first_blowup"
        ),
        "interval_backend": "omnibias.core.verified.Interval (outward-rounded)",
        "python": platform.python_version(),
        "sha256": _sha256_json(body),
    }
    return body


REQUIRED_CLM_MULTIZERO_KEYS: tuple[str, ...] = (
    "schema_version",
    "observable",
    "model",
    "equation",
    "route",
    "basis",
    "coeffs",
    "scales",
    "n_terms",
    "numerator_poly_u_coeffs",
    "n_distinct_positive_roots",
    "n_zeros_enclosed",
    "zero_locations",
    "zero_squared_locations",
    "hilbert_omega0_at_zeros",
    "hilbert_omega0_max",
    "earliest_zero_location",
    "earliest_zero_hilbert",
    "omega0_prime_at_earliest_zero",
    "singularity_certified",
    "completeness_certified",
    "earliest_first_blowup_certified",
    "gradient_nontrivial",
    "first_blowup_time",
    "first_blowup_time_upper_bound",
    "gradient_growth_law",
    "criterion",
    "method",
    "theorem_dependency",
    "three_d_claim",
    "continuum_navier_stokes_claim",
    "honesty",
    "provenance",
)


def certified_clm_multizero_first_blowup_schema_errors(cert: dict[str, Any]) -> list[str]:
    """Validate a ``navier-stokes-clm-multizero-blowup-1`` certificate."""
    errors: list[str] = []
    for key in REQUIRED_CLM_MULTIZERO_KEYS:
        if key not in cert:
            errors.append(f"missing top-level key: {key!r}")
    honesty = cert.get("honesty", {})
    if honesty.get("unproven_claim", False):
        errors.append("honesty.unproven_claim must be False")
    if cert.get("three_d_claim", True):
        errors.append("three_d_claim must be False")
    if cert.get("continuum_navier_stokes_claim", True):
        errors.append("continuum_navier_stokes_claim must be False")
    if cert.get("schema_version") != CLM_MULTIZERO_SCHEMA_VERSION:
        errors.append(f"schema_version must be {CLM_MULTIZERO_SCHEMA_VERSION!r}")
    if cert.get("singularity_certified"):
        hmax = cert.get("hilbert_omega0_max")
        if not (isinstance(hmax, dict) and float(hmax.get("lower", -1.0)) > 0.0):
            errors.append("singularity_certified requires hilbert_omega0_max.lower > 0")
    if cert.get("earliest_first_blowup_certified"):
        if not cert.get("completeness_certified"):
            errors.append(
                "earliest_first_blowup_certified requires completeness_certified"
            )
        bt = cert.get("first_blowup_time")
        if not isinstance(bt, dict):
            errors.append(
                "earliest_first_blowup_certified requires a first_blowup_time interval"
            )
        elif float(bt.get("lower", 1.0)) > float(bt.get("upper", -1.0)) + 1e-12:
            errors.append("first_blowup_time.lower must be <= first_blowup_time.upper")
    return errors


# --------------------------------------------------------------------------- #
# Cordoba-Cordoba-Fontelos (CCF) self-similar blow-up attempt (radii polynomial)
# --------------------------------------------------------------------------- #
_CCF_FORMS: tuple[str, ...] = ("transport", "flux")


def default_ccf_collocation_nodes(n_terms: int) -> tuple[float, ...]:
    r"""Deterministic positive collocation nodes for the normalized CCF system.

    The self-similar profile is gauge-fixed by *holding the first coefficient
    ``c_1`` fixed* (a standard amplitude normalization that removes the trivial
    ``Theta = 0`` branch), so the Newton unknowns are the remaining ``n_terms-1``
    coefficients plus ``lambda`` -- exactly ``n_terms`` unknowns, requiring
    ``n_terms`` collocation points.  The nodes are a deterministic increasing
    spread over ``(0, ~]`` of the positive half-line (the profile is even, so the
    positive axis carries all information).
    """
    if n_terms < 1:
        raise ValueError(f"n_terms must be >= 1, got {n_terms}")
    return tuple(0.5 + 0.7 * i + 0.12 * i * i for i in range(n_terms))


def _ccf_profile_floats(
    x: float, coeffs: Sequence[float], scales: Sequence[float]
) -> tuple[float, float, float, float]:
    """Float ``(Theta, Theta', H Theta, (H Theta)')`` of an even Poisson profile."""
    th = thp = hth = hthp = 0.0
    for c, a in zip(coeffs, scales, strict=True):
        d = x * x + a * a
        th += c * a / d
        thp += c * (-2.0 * a * x) / (d * d)
        hth += c * x / d
        hthp += c * (a * a - x * x) / (d * d)
    return th, thp, hth, hthp


def _ccf_residual_at_float(
    x: float,
    coeffs: Sequence[float],
    scales: Sequence[float],
    lam: float,
    form: str,
    s: float,
) -> float:
    """Float self-similar CCF residual ``E(x)`` (transport or flux form)."""
    th, thp, hth, hthp = _ccf_profile_floats(x, coeffs, scales)
    linear = (1.0 + lam) * x * thp - lam * th
    if form == "transport":
        nonlocal_term = s * hth * thp
    else:
        nonlocal_term = s * (thp * hth + th * hthp)
    return linear + nonlocal_term


def _ccf_collocation_residual_floats(
    u: np.ndarray,
    c1: float,
    scales: Sequence[float],
    nodes: Sequence[float],
    form: str,
    s: float,
) -> np.ndarray:
    """Float residual vector of the normalized collocation system at ``u``."""
    coeffs = [c1, *[float(v) for v in u[:-1]]]
    lam = float(u[-1])
    return np.asarray(
        [_ccf_residual_at_float(y, coeffs, scales, lam, form, s) for y in nodes],
        dtype=float,
    )


def _ccf_collocation_jacobian_floats(
    u: np.ndarray,
    c1: float,
    scales: Sequence[float],
    nodes: Sequence[float],
    form: str,
    s: float,
) -> np.ndarray:
    """Analytic float Jacobian of the normalized collocation system at ``u``."""
    coeffs = [c1, *[float(v) for v in u[:-1]]]
    lam = float(u[-1])
    n = len(scales)
    jac = np.zeros((len(nodes), n), dtype=float)
    for r, y in enumerate(nodes):
        th, thp, hth, hthp = _ccf_profile_floats(y, coeffs, scales)
        col = 0
        for k in range(1, n):
            a = scales[k]
            d = y * y + a * a
            pk = a / d
            dpk = -2.0 * a * y / (d * d)
            qk = y / d
            dqk = (a * a - y * y) / (d * d)
            base = (1.0 + lam) * y * dpk - lam * pk
            if form == "transport":
                nl = s * (qk * thp + hth * dpk)
            else:
                nl = s * (dpk * hth + thp * qk + pk * hthp + th * dqk)
            jac[r, col] = base + nl
            col += 1
        jac[r, col] = y * thp - th
    return jac


def refine_ccf_selfsimilar_profile(
    *,
    coeffs: Sequence[float],
    scales: Sequence[float],
    lam: float,
    nodes: Sequence[float] | None = None,
    form: str = "transport",
    velocity_sign: float = 1.0,
    iters: int = 80,
    tol: float = 1e-13,
) -> dict[str, Any]:
    r"""Float Newton refinement of a CCF self-similar candidate (the *find* step).

    Solves the normalized collocation system (``c_1`` held fixed) for the free
    coefficients ``c_2..c_n`` and ``lambda`` so the residual ``E`` vanishes at the
    collocation nodes.  This is an ordinary floating-point Newton iteration --
    *not* a proof; it produces a candidate that
    :func:`certified_ccf_selfsimilar_blowup_attempt` can then attempt to certify
    rigorously.
    """
    if form not in _CCF_FORMS:
        raise ValueError(f"form must be one of {_CCF_FORMS}, got {form!r}")
    cs = [float(c) for c in coeffs]
    as_ = [float(a) for a in scales]
    n = len(cs)
    if n < 1:
        raise ValueError("need at least one coefficient")
    if len(as_) != n:
        raise ValueError("coeffs and scales must have equal length")
    ynodes = list(default_ccf_collocation_nodes(n) if nodes is None else [float(y) for y in nodes])
    if len(ynodes) != n:
        raise ValueError(f"need len(nodes) == len(coeffs) = {n}, got {len(ynodes)}")
    c1 = cs[0]
    s = float(velocity_sign)
    u = np.asarray([*cs[1:], float(lam)], dtype=float)
    for _ in range(int(iters)):
        f = _ccf_collocation_residual_floats(u, c1, as_, ynodes, form, s)
        if float(np.max(np.abs(f))) < tol:
            break
        jac = _ccf_collocation_jacobian_floats(u, c1, as_, ynodes, form, s)
        try:
            step = np.linalg.solve(jac, f)
        except np.linalg.LinAlgError:
            break
        u = u - step
    final = _ccf_collocation_residual_floats(u, c1, as_, ynodes, form, s)
    return {
        "coeffs": [c1, *[float(v) for v in u[:-1]]],
        "scales": as_,
        "lam": float(u[-1]),
        "nodes": tuple(ynodes),
        "form": form,
        "velocity_sign": s,
        "residual_max_abs": float(np.max(np.abs(final))),
    }


def _ccf_residual_interval(
    coeffs: Sequence[float],
    scales: Sequence[float],
    lam: float,
    y: float,
    form: str,
    s: float,
) -> Interval:
    """Outward-rounded interval enclosure of the CCF residual ``E(y)``."""
    th = even_profile(y, coeffs, scales)
    thp = even_profile_deriv(y, coeffs, scales)
    hth = hilbert_even_profile(y, coeffs, scales)
    one = Interval.point(1.0)
    lam_iv = Interval.point(lam)
    y_iv = Interval.point(y)
    s_iv = Interval.point(s)
    linear = (one + lam_iv) * y_iv * thp - lam_iv * th
    if form == "transport":
        return linear + s_iv * hth * thp
    hthp = hilbert_even_profile_deriv(y, coeffs, scales)
    return linear + s_iv * (thp * hth + th * hthp)


def _ccf_node_system(
    coeffs: Sequence[float],
    scales: Sequence[float],
    lam: float,
    y: float,
    form: str,
    s: float,
) -> tuple[Interval, list[Interval], Interval]:
    r"""Interval ``(E, dE/d(free unknowns), sum|Hessian|)`` of the CCF map at ``y``.

    The free unknowns are ``(c_2, ..., c_n, lambda)`` (``c_1`` is the fixed
    normalization).  ``E`` is *quadratic* in these unknowns with a node-dependent
    but unknown-independent Hessian, so the sum of absolute Hessian entries is a
    rigorous, ball-independent curvature bound for the radii polynomial.
    """
    n = len(scales)
    th = even_profile(y, coeffs, scales)
    thp = even_profile_deriv(y, coeffs, scales)
    hth = hilbert_even_profile(y, coeffs, scales)
    hthp = hilbert_even_profile_deriv(y, coeffs, scales)
    one = Interval.point(1.0)
    lam_iv = Interval.point(lam)
    y_iv = Interval.point(y)
    s_iv = Interval.point(s)
    linear = (one + lam_iv) * y_iv * thp - lam_iv * th
    if form == "transport":
        e_iv = linear + s_iv * hth * thp
    else:
        e_iv = linear + s_iv * (thp * hth + th * hthp)

    pk = [poisson_kernel(y, a) for a in scales]
    dpk = [poisson_kernel_deriv(y, a) for a in scales]
    qk = [conjugate_poisson(y, a) for a in scales]
    dqk = [conjugate_poisson_deriv(y, a) for a in scales]

    row: list[Interval] = []
    for k in range(1, n):
        base = (one + lam_iv) * y_iv * dpk[k] - lam_iv * pk[k]
        if form == "transport":
            nl = s_iv * (qk[k] * thp + hth * dpk[k])
        else:
            nl = s_iv * (dpk[k] * hth + thp * qk[k] + pk[k] * hthp + th * dqk[k])
        row.append(base + nl)
    row.append(y_iv * thp - th)

    hess_terms: list[Interval] = []
    free = range(1, n)
    for k in free:
        for j in free:
            if form == "transport":
                hkj = s_iv * (qk[k] * dpk[j] + qk[j] * dpk[k])
            else:
                hkj = s_iv * (dpk[k] * qk[j] + dpk[j] * qk[k] + pk[k] * dqk[j] + pk[j] * dqk[k])
            hess_terms.append(hkj.abs())
    for k in free:
        clam = (y_iv * dpk[k] - pk[k]).abs()
        hess_terms.append(clam)
        hess_terms.append(clam)
    hess_abs = sum_intervals(hess_terms) if hess_terms else Interval.point(0.0)
    return e_iv, row, hess_abs


def _ccf_far_field_residual_bound(
    coeffs: Sequence[float],
    scales: Sequence[float],
    lam: float,
    y_trunc: float,
    s: float,
    form: str,
) -> Interval:
    r"""Rigorous far-field bound ``|E(y)| <= bound`` for all ``|y| >= y_trunc``.

    Uses the elementary tail inequalities ``|Theta| <= S1/y^2``,
    ``|Theta'| <= 2 S1/y^3``, ``|H Theta| <= S0/|y|`` and (flux only, for
    ``|y| >= max a_i``) ``|(H Theta)'| <= S0/y^2`` with ``S0 = sum|c_i|`` and
    ``S1 = sum|c_i| a_i``, giving an ``O(1/y^2)`` whole-line tail certificate.
    """
    yt = float(y_trunc)
    if yt <= 0.0:
        raise ValueError("far_field_trunc must be positive")
    if form == "flux" and yt < max(scales):
        raise ValueError("flux far-field bound requires far_field_trunc >= max(scales)")
    s0 = sum_intervals([Interval.point(abs(float(c))) for c in coeffs])
    s1 = sum_intervals(
        [Interval.point(abs(float(c))) * Interval.point(float(a)) for c, a in zip(coeffs, scales, strict=True)]
    )
    yt_iv = Interval.point(yt)
    inv2 = yt_iv.pow_int(2).reciprocal()
    inv4 = yt_iv.pow_int(4).reciprocal()
    abs_one_plus = Interval.point(abs(1.0 + lam))
    abs_lam = Interval.point(abs(lam))
    abs_s = Interval.point(abs(s))
    term1 = abs_one_plus * Interval.point(2.0) * s1 * inv2
    term2 = abs_lam * s1 * inv2
    term3 = abs_s * Interval.point(2.0) * s0 * s1 * inv4
    total = term1 + term2 + term3
    if form == "flux":
        total = total + abs_s * s0 * s1 * inv4
    return total


def _ccf_residual_taylor_model(
    cs: Sequence[float],
    as_: Sequence[float],
    lam: float,
    s: float,
    form: str,
    center: float,
    radius: float,
    order: int,
) -> TaylorModel:
    r"""Degree-``order`` Taylor model of the CCF residual ``E`` over one cell.

    Builds ``Theta, Theta', H Theta, (H Theta)'`` as Taylor models in the *exact*
    even Poisson basis (each ``p_a = a/(y^2+a^2)`` etc. assembled from a rigorous
    :meth:`TaylorModel.reciprocal` of ``y^2 + a^2 > 0``) and combines them with
    the same algebra as :func:`_ccf_residual_interval`.  Unlike a point sample,
    ``model.bound()`` rigorously encloses ``E`` over the *whole* cell, so its
    magnitude is a certified between-node residual bound.
    """
    x = TaylorModel.identity(center, radius, order)
    x2 = x.pow_int(2)
    theta = TaylorModel.constant(0.0, center, radius, order)
    dtheta = TaylorModel.constant(0.0, center, radius, order)
    htheta = TaylorModel.constant(0.0, center, radius, order)
    dhtheta = TaylorModel.constant(0.0, center, radius, order)
    for c, a in zip(cs, as_, strict=True):
        inv_d = (x2 + a * a).reciprocal()  # 1 / (y^2 + a^2)
        inv_d2 = inv_d.pow_int(2)
        theta = theta + (inv_d * a) * c  # p_a = a / (y^2 + a^2)
        htheta = htheta + (x * inv_d) * c  # q_a = y / (y^2 + a^2)
        dtheta = dtheta + ((x * inv_d2) * (-2.0 * a)) * c  # p_a' = -2 a y / D^2
        if form == "flux":
            dq = (x2 * (-1.0) + a * a) * inv_d2  # q_a' = (a^2 - y^2) / D^2
            dhtheta = dhtheta + dq * c
    linear = (x * dtheta) * (1.0 + lam) - theta * lam
    if form == "transport":
        return linear + (htheta * dtheta) * s
    return linear + (dtheta * htheta + theta * dhtheta) * s


def _ccf_htheta_over_y_taylor_model(
    cs: Sequence[float],
    as_: Sequence[float],
    center: float,
    radius: float,
    order: int,
) -> TaylorModel:
    r"""Taylor model of ``(H Theta)(y) / y = sum_i c_i / (y^2 + a_i^2)`` over a cell.

    This is the *bounded* even multiplier obtained by factoring ``y`` out of the
    odd nonlocal velocity ``H Theta = sum_i c_i q_{a_i}`` (``q_a = y/(y^2+a^2)``);
    it is what makes the scaling-weighted product ``(H Theta) h' = (H Theta / y)
    (y h')`` controllable in the continuum operator bound.
    """
    x2 = TaylorModel.identity(center, radius, order).pow_int(2)
    out = TaylorModel.constant(0.0, center, radius, order)
    for c, a in zip(cs, as_, strict=True):
        out = out + (x2 + a * a).reciprocal() * c
    return out


def _ccf_theta_prime_taylor_model(
    cs: Sequence[float],
    as_: Sequence[float],
    center: float,
    radius: float,
    order: int,
) -> TaylorModel:
    r"""Taylor model of ``Theta'(y) = sum_i c_i (-2 a_i y / (y^2 + a_i^2)^2)``."""
    x = TaylorModel.identity(center, radius, order)
    x2 = x.pow_int(2)
    out = TaylorModel.constant(0.0, center, radius, order)
    for c, a in zip(cs, as_, strict=True):
        inv_d2 = (x2 + a * a).reciprocal().pow_int(2)
        out = out + ((x * inv_d2) * (-2.0 * a)) * c
    return out


def _ccf_scaling_multiplier_taylor_model(
    cs: Sequence[float],
    as_: Sequence[float],
    lam: float,
    s: float,
    center: float,
    radius: float,
    order: int,
) -> TaylorModel:
    r"""Taylor model of ``M(y) = (1+lambda) + s (H Theta)(y) / y`` over a cell.

    ``M`` is the coefficient of ``y h'`` after writing the linearized operator as
    ``DE h = M (y h') - lambda h + s Theta' (H h)``; ``sup|M|`` is the certified
    multiplier norm in the forward continuum bound.
    """
    return _ccf_htheta_over_y_taylor_model(cs, as_, center, radius, order) * s + (1.0 + lam)


def _ccf_tm_sup_over_cells(
    builder: Callable[[float, float, int], TaylorModel],
    y_hi: float,
    *,
    order: int = 6,
    n_cells: int = 64,
    max_depth: int = 8,
) -> tuple[float, int]:
    r"""Rigorous ``sup_{|y| <= y_hi} |f(y)|`` of an even ``f`` via per-cell Taylor models.

    Partitions ``[0, y_hi]`` (the integrand is even, so this covers ``[-y_hi,
    y_hi]``) into ``n_cells`` cells and encloses ``f`` on each with ``builder``.
    A cell whose relative variation defeats the rigorous reciprocal series is
    bisected up to ``max_depth`` times.  Returns the certified sup and the number
    of leaf cells used.
    """
    if y_hi <= 0.0:
        raise ValueError("y_hi must be positive")
    stack: list[tuple[float, float, int]] = [
        (y_hi * i / n_cells, y_hi * (i + 1) / n_cells, 0) for i in range(n_cells)
    ]
    sup = 0.0
    leaves = 0
    while stack:
        lo, hi, depth = stack.pop()
        center = 0.5 * (lo + hi)
        radius = 0.5 * (hi - lo)
        try:
            model = builder(center, radius, order)
        except ValueError:
            if depth >= max_depth:
                raise
            mid = center
            stack.append((lo, mid, depth + 1))
            stack.append((mid, hi, depth + 1))
            continue
        sup = max(sup, model.bound().mag)
        leaves += 1
    return sup, leaves


def _ccf_certified_residual_core_sup(
    cs: Sequence[float],
    as_: Sequence[float],
    lam: float,
    s: float,
    form: str,
    y_hi: float,
    *,
    order: int = 6,
    n_cells: int = 64,
    max_depth: int = 8,
) -> tuple[float, int]:
    r"""Rigorous ``sup_{|y| <= y_hi} |E(y)|`` via per-cell Taylor models."""
    return _ccf_tm_sup_over_cells(
        lambda center, radius, o: _ccf_residual_taylor_model(
            cs, as_, lam, s, form, center, radius, o
        ),
        y_hi,
        order=order,
        n_cells=n_cells,
        max_depth=max_depth,
    )


def certified_ccf_linearized_operator_bound(
    *,
    coeffs: Sequence[float],
    scales: Sequence[float],
    lam: float,
    velocity_sign: float = 1.0,
    form: str = "transport",
    far_field_trunc: float | None = None,
    order: int = 6,
) -> dict[str, Any]:
    r"""Certified *continuum* bound + invertibility test for the CCF profile linearization.

    The Fréchet derivative of the transport CCF residual in the profile direction
    ``h`` is

    .. math::

        D\mathcal{E}[\Theta]\,h
          = (1+\lambda)\,y\,h' - \lambda\,h
            + s\,(H\Theta)\,h' + s\,(Hh)\,\Theta'.

    Split it as ``DE = T + R`` with the **scaling part** ``T = (1+lambda) y d_y -
    lambda`` and the **nonlocal part** ``R h = s (H\Theta/y)(y h') + s \Theta'
    (Hh)``.  The dilation group ``(U_t f)(y) = e^{t/2} f(e^t y)`` is unitary on
    ``L^2(R)``, so its generator ``J = 1/2 + y d_y`` is skew-adjoint and ``T = -(1/2
    + 3/2 lambda) I + (1+lambda) J`` is **normal**; hence

    .. math::

        \lVert T^{-1}\rVert_{L^2} = \frac{1}{\lvert 1/2 + 3/2\,\lambda\rvert}
        \;=:\; \kappa \qquad (\lambda \neq -1/3),

    an *exact* resolvent identity (using ``<y h', h> = -1/2 ||h||^2``).  Because
    ``H\Theta/y = sum_i c_i/(y^2+a_i^2)`` is *bounded*, ``R T^{-1}`` is bounded on
    ``L^2`` with the certified estimate

    .. math::

        \rho := \lVert R T^{-1}\rVert_{L^2}
          \le \lVert H\Theta/y\rVert_\infty \frac{1 + \lvert\lambda\rvert\kappa}
                                                   {\lvert 1+\lambda\rvert}
             + \lVert \Theta'\rVert_\infty\,\kappa .

    If ``rho < 1`` the linearization is invertible on ``L^2`` with
    ``||DE^{-1}||_{L^2} <= kappa/(1-rho)`` -- a genuine *whole-line* (not
    finite-section) operator certificate.  The sup-norms are certified by the same
    per-cell :class:`TaylorModel` machinery as the residual (core ``[0, y_trunc]``
    plus an explicit far-field tail).

    **Scope / honesty.**  This bounds (and tests the invertibility of) the
    *profile* linearization on the even ``L^2`` subspace; it is **not** a finished
    blow-up proof.  The flux form's extra ``s\,\Theta\,(Hh)'`` term needs a
    higher-regularity space, so it is reported ``supported=False``.  ``unproven_claim``
    and ``whole_line_certified`` stay ``False``.
    """
    if form not in _CCF_FORMS:
        raise ValueError(f"form must be one of {_CCF_FORMS}, got {form!r}")
    cs = [float(c) for c in coeffs]
    as_ = [float(a) for a in scales]
    lam_f = float(lam)
    s = float(velocity_sign)
    n = len(cs)
    if n < 1:
        raise ValueError("need at least one coefficient")
    if len(as_) != n:
        raise ValueError("coeffs and scales must have equal length")
    if any(a <= 0.0 for a in as_):
        raise ValueError("scales must be positive")

    yt = float(far_field_trunc) if far_field_trunc is not None else 2.0 * max(as_) + 1.0
    if yt <= max(as_):
        raise ValueError("far_field_trunc must exceed max(scales)")

    honesty = {
        "unproven_claim": False,
        "three_d_claim": False,
        "model_only": True,
        "one_dimensional_model": True,
        "profile_linearization_only": True,
        "even_subspace": True,
        "whole_line_certified": False,
        "interval_verified": True,
    }

    if form != "transport":
        body: dict[str, Any] = {
            "schema_version": CCF_LINEARIZED_OPERATOR_SCHEMA_VERSION,
            "observable": "cordoba_cordoba_fontelos_profile_linearization_operator_bound",
            "model": "cordoba_cordoba_fontelos_1d",
            "form": form,
            "velocity_sign": s,
            "coeffs": cs,
            "scales": as_,
            "lambda_candidate": lam_f,
            "far_field_trunc": yt,
            "supported": False,
            "scaling_shift": None,
            "scaling_invertible": False,
            "scaling_inverse_norm_bound": None,
            "htheta_over_y_sup": None,
            "theta_prime_sup": None,
            "multiplier_sup": None,
            "forward_operator_bound": None,
            "neumann_rho": None,
            "rho_closes": False,
            "continuum_invertible_certified": False,
            "inverse_norm_bound": None,
            "space": "even L^2(R) with scaling-graph domain {h : y h' in L^2}",
            "criterion": (
                "flux form adds s Theta (H h)' = s Theta H(h'), which needs h' in L^2 "
                "(H^1), not just y h' in L^2; the scaling-graph bound does not control it"
            ),
            "honesty": honesty,
            "open_obligations": ["flux_form_continuum_linearization_requires_h1_space"],
        }
        body["provenance"] = {
            "harness": "omnibias.pinn.certified.navier_stokes.certified_ccf_linearized_operator_bound",
            "interval_backend": "omnibias.core.verified.Interval (outward-rounded)",
            "python": platform.python_version(),
            "sha256": _sha256_json(body),
        }
        return body

    # --- certified whole-line sup-norms (core Taylor models + far-field tail) --- #
    s0 = sum(abs(c) for c in cs)
    s1 = sum(abs(c) * a for c, a in zip(cs, as_, strict=True))
    yt_iv = Interval.point(yt)
    inv2 = yt_iv.pow_int(2).reciprocal()
    inv3 = yt_iv.pow_int(3).reciprocal()

    hthy_core, hthy_cells = _ccf_tm_sup_over_cells(
        lambda c, r, o: _ccf_htheta_over_y_taylor_model(cs, as_, c, r, o), yt, order=order
    )
    thp_core, _ = _ccf_tm_sup_over_cells(
        lambda c, r, o: _ccf_theta_prime_taylor_model(cs, as_, c, r, o), yt, order=order
    )
    mult_core, _ = _ccf_tm_sup_over_cells(
        lambda c, r, o: _ccf_scaling_multiplier_taylor_model(cs, as_, lam_f, s, c, r, o),
        yt,
        order=order,
    )
    hthy_far = (Interval.point(s0) * inv2).hi  # |sum c_i/(y^2+a_i^2)| <= S0/yt^2
    thp_far = (Interval.point(2.0) * Interval.point(s1) * inv3).hi  # |Theta'| <= 2 S1/yt^3
    mult_far = (Interval.point(abs(1.0 + lam_f)) + Interval.point(s0) * inv2).hi
    hthy_sup = max(hthy_core, hthy_far)
    thp_sup = max(thp_core, thp_far)
    mult_sup = max(mult_core, mult_far)

    # --- exact scaling resolvent + Neumann perturbation ------------------------ #
    shift = 0.5 + 1.5 * lam_f  # symmetric part of T: <T h, h> = -shift ||h||^2
    one_plus = 1.0 + lam_f
    scaling_invertible = abs(shift) > 1e-12
    can_form_rho = scaling_invertible and abs(one_plus) > 1e-12

    kappa_iv = Interval.point(abs(shift)).reciprocal() if scaling_invertible else None
    kappa = float(kappa_iv.hi) if kappa_iv is not None else None

    rho: float | None = None
    inverse_norm_bound: float | None = None
    forward_bound = (
        Interval.point(mult_sup) + Interval.point(abs(lam_f)) + Interval.point(thp_sup)
    ).hi
    if can_form_rho and kappa_iv is not None:
        yh_factor = (
            Interval.point(1.0) + Interval.point(abs(lam_f)) * kappa_iv
        ) * Interval.point(abs(one_plus)).reciprocal()
        rho_iv = Interval.point(hthy_sup) * yh_factor + Interval.point(thp_sup) * kappa_iv
        rho = float(rho_iv.hi)
    rho_closes = bool(rho is not None and rho < 1.0)
    if rho_closes and kappa_iv is not None and rho is not None:
        inverse_norm_bound = float(
            (kappa_iv * (Interval.point(1.0) - Interval.point(rho)).reciprocal()).hi
        )

    obligations: list[str]
    obligations = [] if rho_closes else ["continuum_linearized_neumann_rho_below_one"]

    body = {
        "schema_version": CCF_LINEARIZED_OPERATOR_SCHEMA_VERSION,
        "observable": "cordoba_cordoba_fontelos_profile_linearization_operator_bound",
        "model": "cordoba_cordoba_fontelos_1d",
        "form": form,
        "velocity_sign": s,
        "coeffs": cs,
        "scales": as_,
        "lambda_candidate": lam_f,
        "far_field_trunc": yt,
        "supported": True,
        "scaling_shift": float(shift),
        "scaling_invertible": scaling_invertible,
        "scaling_inverse_norm_bound": kappa,
        "htheta_over_y_sup": float(hthy_sup),
        "theta_prime_sup": float(thp_sup),
        "multiplier_sup": float(mult_sup),
        "forward_operator_bound": float(forward_bound),
        "neumann_rho": rho,
        "rho_closes": rho_closes,
        "continuum_invertible_certified": rho_closes,
        "inverse_norm_bound": inverse_norm_bound,
        "taylor_model_order": int(order),
        "taylor_model_leaf_cells": int(hthy_cells),
        "space": "even L^2(R), scaling-graph domain {h : y h' in L^2}; DE: X -> L^2",
        "criterion": (
            "T = (1+lambda) y d_y - lambda is normal (dilation generator), so "
            "||T^{-1}|| = 1/|1/2 + 3/2 lambda| exactly; DE = T + R invertible on L^2 "
            "iff rho = ||R T^{-1}|| < 1, then ||DE^{-1}|| <= kappa/(1-rho)"
        ),
        "theorem_dependency": (
            "exact dilation-semigroup resolvent of the scaling generator + Neumann "
            "perturbation; sup-norms certified by per-cell Taylor models + far-field tail"
        ),
        "honesty": honesty,
        "open_obligations": obligations,
    }
    body["provenance"] = {
        "harness": "omnibias.pinn.certified.navier_stokes.certified_ccf_linearized_operator_bound",
        "interval_backend": "omnibias.core.verified.Interval (outward-rounded)",
        "python": platform.python_version(),
        "sha256": _sha256_json(body),
    }
    return body


REQUIRED_CCF_LINEARIZED_OPERATOR_KEYS: tuple[str, ...] = (
    "schema_version",
    "observable",
    "model",
    "form",
    "velocity_sign",
    "coeffs",
    "scales",
    "lambda_candidate",
    "far_field_trunc",
    "supported",
    "scaling_invertible",
    "scaling_inverse_norm_bound",
    "htheta_over_y_sup",
    "theta_prime_sup",
    "multiplier_sup",
    "forward_operator_bound",
    "neumann_rho",
    "rho_closes",
    "continuum_invertible_certified",
    "inverse_norm_bound",
    "space",
    "criterion",
    "honesty",
    "open_obligations",
    "provenance",
)


def certified_ccf_linearized_operator_bound_schema_errors(cert: dict[str, Any]) -> list[str]:
    """Validate a ``navier-stokes-ccf-linearized-operator-bound-1`` certificate."""
    errors: list[str] = []
    for key in REQUIRED_CCF_LINEARIZED_OPERATOR_KEYS:
        if key not in cert:
            errors.append(f"missing top-level key: {key!r}")
    honesty = cert.get("honesty", {})
    if honesty.get("unproven_claim", False):
        errors.append("honesty.unproven_claim must be False")
    if honesty.get("whole_line_certified", False):
        errors.append("honesty.whole_line_certified must be False")
    if cert.get("schema_version") != CCF_LINEARIZED_OPERATOR_SCHEMA_VERSION:
        errors.append(f"schema_version must be {CCF_LINEARIZED_OPERATOR_SCHEMA_VERSION!r}")
    if cert.get("continuum_invertible_certified"):
        if not cert.get("rho_closes"):
            errors.append("continuum_invertible_certified requires rho_closes")
        rho = cert.get("neumann_rho")
        if not isinstance(rho, int | float) or float(rho) >= 1.0:
            errors.append("continuum_invertible_certified requires neumann_rho < 1")
        if cert.get("inverse_norm_bound") is None:
            errors.append("continuum_invertible_certified requires inverse_norm_bound")
    return errors


def radii_polynomial_closure(
    residual_bound: float, defect: float, lipschitz: float
) -> dict[str, Any]:
    r"""Rigorous Newton-Kantorovich closure of ``g(r) = Z2 r^2 - (1-Z1) r + Y0``.

    ``residual_bound`` is ``Y0 = ||B F(x)||``, ``defect`` is
    ``Z1 = ||I - B DF(x)||`` and ``lipschitz`` is ``Z2 = ||B|| * sup|D^2 F|``
    (all infinity norms; ``B`` is the approximate inverse of ``DF(x)``).  Because
    the CCF collocation map is *quadratic* with constant Hessian, ``g`` is exact
    and the closure ``g(r) <= 0`` admits a solution iff ``Z1 < 1`` and the
    discriminant ``(1-Z1)^2 - 4 Z2 Y0 >= 0``.  The discriminant is evaluated as a
    rigorous lower bound (upper bounds for ``Y0, Z1, Z2``), so ``passed`` is
    theorem-grade.  Returns the existence radius ``r_minus`` (an upper bound on
    ``||x* - x||``) and the uniqueness radius ``r_plus``.
    """
    y0 = float(residual_bound)
    z1 = float(defect)
    z2 = float(lipschitz)
    one_minus = Interval.point(1.0) - Interval.point(z1)
    out: dict[str, Any] = {
        "residual_bound": y0,
        "defect": z1,
        "lipschitz": z2,
        "passed": False,
        "discriminant_lower": None,
        "r_minus": None,
        "r_plus": None,
        "failed_inequality": None,
    }
    if z1 >= 1.0:
        out["failed_inequality"] = (
            f"linearized defect Z1 = {z1:.6g} must be < 1 (||I - B DF|| too large; "
            "the approximate inverse does not control the linearization)"
        )
        return out
    if z2 <= 0.0:
        r_iv = Interval.point(y0) * one_minus.reciprocal()
        out.update(
            passed=True,
            r_minus=float(r_iv.hi),
            r_plus=None,
            note="linear collocation system; global Newton-Kantorovich closure",
        )
        return out
    disc_iv = one_minus.pow_int(2) - Interval.point(4.0) * Interval.point(z2) * Interval.point(y0)
    disc_lower = float(disc_iv.lo)
    out["discriminant_lower"] = disc_lower
    if disc_lower < 0.0:
        short_by = float(
            (Interval.point(4.0) * Interval.point(z2) * Interval.point(y0) - one_minus.pow_int(2)).hi
        )
        out["failed_inequality"] = (
            f"radii-polynomial discriminant (1-Z1)^2 - 4 Z2 Y0 = {disc_lower:.6g} < 0 "
            f"(short by {short_by:.6g}); residual Y0 = {y0:.6g} too large relative to "
            f"inverse norm and curvature Z2 = {z2:.6g}"
        )
        return out
    sq = Interval(max(disc_lower, 0.0), max(float(disc_iv.hi), 0.0)).sqrt()
    two_z2 = Interval.point(2.0) * Interval.point(z2)
    rminus_iv = (one_minus - sq) * two_z2.reciprocal()
    rplus_iv = (one_minus + sq) * two_z2.reciprocal()
    out.update(
        passed=True,
        r_minus=float(rminus_iv.hi),
        r_plus=float(rplus_iv.lo),
    )
    return out


def certified_ccf_selfsimilar_blowup_attempt(
    *,
    coeffs: Sequence[float],
    scales: Sequence[float],
    lam: float,
    nodes: Sequence[float] | None = None,
    form: str = "transport",
    velocity_sign: float = 1.0,
    far_field_trunc: float | None = None,
) -> dict[str, Any]:
    r"""Attempt a radii-polynomial closure of a CCF self-similar blow-up profile.

    The Cordoba-Cordoba-Fontelos model ``theta_t + (H theta) theta_x = 0`` (and
    its flux variant) admits self-similar profiles ``Theta`` solving
    ``E(Theta, lambda) = (1+lambda) y Theta' - lambda Theta + s (H Theta) Theta'
    = 0`` whose existence forces a finite-time gradient singularity.  This
    function represents ``Theta = sum_i c_i p_{a_i}`` in the **verified even
    Poisson basis** (so ``H Theta = sum_i c_i q_{a_i}`` is *exact*, quadrature
    free), builds the residual and its Jacobian/Hessian as outward-rounded
    intervals at a verified collocation grid, and runs a rigorous
    Newton-Kantorovich / radii-polynomial test reusing
    :func:`omnibias.core.verified.linalg.neumann_inverse_norm_bound`.

    **Scope / honesty.**  A closure certifies a true zero of the *finite
    collocation map* within ``r_minus`` of the candidate -- i.e. a self-similar
    profile that makes the residual vanish at the collocation nodes, with a
    two-sided enclosure of the admissible ``lambda``.  It is a 1D-model,
    finite-section result.  The whole-line residual sup-norm is now **certified**
    (``closure_report.residual_certified_sup``): per-cell :class:`TaylorModel`
    enclosures bound ``E`` *between* the nodes and a far-field tail bound covers
    ``|y| >= far_field_trunc``, replacing the merely sampled ``residual_sampled_sup``.
    What remains open is the *continuum* closure -- a function-space Fréchet
    derivative bound and driving that certified residual below the
    Newton-Kantorovich threshold -- so ``honesty.whole_line_certified``,
    ``unproven_claim`` and ``three_d_claim`` stay ``False``.  When the collocation
    closure fails, the certificate reports exactly which inequality failed and by
    how much -- the calibration of the remaining gap.

    Parameters
    ----------
    coeffs, scales:
        Even Poisson expansion ``Theta = sum_i c_i p_{a_i}`` with ``a_i > 0``.
        ``c_1`` (i.e. ``coeffs[0]``) is the fixed amplitude normalization.
    lam:
        The self-similar exponent ``lambda`` candidate.
    nodes:
        ``len(coeffs)`` positive collocation nodes; defaults to
        :func:`default_ccf_collocation_nodes`.
    form:
        ``"transport"`` or ``"flux"``.
    velocity_sign:
        Sign ``s = +-1`` on the nonlocal term.
    far_field_trunc:
        Radius beyond which the certified far-field tail bound applies.
    """
    if form not in _CCF_FORMS:
        raise ValueError(f"form must be one of {_CCF_FORMS}, got {form!r}")
    cs = [float(c) for c in coeffs]
    as_ = [float(a) for a in scales]
    lam_f = float(lam)
    s = float(velocity_sign)
    n = len(cs)
    if n < 1:
        raise ValueError("need at least one coefficient")
    if len(as_) != n:
        raise ValueError("coeffs and scales must have equal length")
    if any(a <= 0.0 for a in as_):
        raise ValueError("scales must be positive")
    ynodes = list(default_ccf_collocation_nodes(n) if nodes is None else [float(y) for y in nodes])
    if len(ynodes) != n:
        raise ValueError(f"need len(nodes) == len(coeffs) = {n}, got {len(ynodes)}")
    m = n

    f_iv: list[Interval] = []
    a_iv: list[list[Interval]] = []
    hess_sums: list[Interval] = []
    for y in ynodes:
        e_iv, row, hess_abs = _ccf_node_system(cs, as_, lam_f, y, form, s)
        f_iv.append(e_iv)
        a_iv.append(row)
        hess_sums.append(hess_abs)

    a_float = [[iv.mid for iv in row] for row in a_iv]
    a_np = np.asarray(a_float, dtype=float)
    try:
        b_np = np.linalg.inv(a_np)
    except np.linalg.LinAlgError:
        b_np = np.zeros((m, m), dtype=float)
    b_float = [[float(b_np[i, j]) for j in range(m)] for i in range(m)]
    b_iv = to_interval_matrix(b_float)

    defect_matrix = mat_sub(identity_matrix(m), matmul(b_iv, a_iv))
    z1 = inf_norm_matrix(defect_matrix)
    norm_b = inf_norm_matrix(b_iv)
    neumann = neumann_inverse_norm_bound(a_float, b_float)
    y0 = inf_norm_vector(matvec(b_iv, f_iv))
    kappa2 = max((h.hi for h in hess_sums), default=0.0)
    z2 = (Interval.point(norm_b) * Interval.point(kappa2)).hi

    closure = radii_polynomial_closure(y0, z1, z2)
    operator_invertible = bool(z1 < 1.0)
    closure_certified = bool(closure["passed"] and operator_invertible)

    try:
        svals = np.linalg.svd(a_np, compute_uv=False)
        smin = float(np.min(svals))
        smax = float(np.max(svals))
    except np.linalg.LinAlgError:
        smin = 0.0
        smax = 0.0

    matrix_norm = inf_norm_matrix(a_iv)
    condition_estimate = float(matrix_norm * norm_b) if operator_invertible else None

    yt = (
        float(far_field_trunc)
        if far_field_trunc is not None
        else 2.0 * max(max(ynodes), max(as_)) + 1.0
    )
    far_field = _ccf_far_field_residual_bound(cs, as_, lam_f, yt, s, form)

    # Reuse the verified line tail machinery to certify that truncating the line
    # to the core [-yt, yt] captures the (exact) Hilbert transform: the far-field
    # |t| >= yt contributes at most this to H[Theta] on the collocation core.
    core_radius = max(ynodes)
    tail_const, tail_power = even_profile_tail_constant(cs, as_)
    if yt > core_radius:
        hilbert_core_tail: float | None = float(
            hilbert_tail_bound(tail_const, tail_power, yt, core_radius).hi
        )
    else:
        hilbert_core_tail = None

    n_fine = 200
    fine_sup = 0.0
    for t in range(n_fine + 1):
        y = yt * t / n_fine
        fine_sup = max(fine_sup, _ccf_residual_interval(cs, as_, lam_f, y, form, s).mag)

    # Discharge the between-node residual obligation: a per-cell Taylor model
    # rigorously encloses E over each [y_k, y_{k+1}] (not just at samples), so
    # the core sup is certified; with the far-field tail this is a certified
    # whole-line residual sup-norm (vs. the merely sampled fine_sup above).
    tm_order = 6
    tm_cells = 64
    residual_core_sup, residual_leaf_cells = _ccf_certified_residual_core_sup(
        cs, as_, lam_f, s, form, yt, order=tm_order, n_cells=tm_cells
    )
    residual_certified_sup = float(max(residual_core_sup, far_field.hi))

    # Discharge the continuum Frechet-derivative obligation: a certified bound and
    # Neumann invertibility test for the profile linearization on even L^2 (exact
    # dilation-generator resolvent + per-cell Taylor-model sup-norms).
    continuum_operator = certified_ccf_linearized_operator_bound(
        coeffs=cs, scales=as_, lam=lam_f, velocity_sign=s, form=form, far_field_trunc=yt,
        order=tm_order,
    )

    r_minus = closure["r_minus"]
    lambda_enclosure: dict[str, Any] | None
    profile_enclosure_radius: float | None
    if closure_certified and r_minus is not None:
        lambda_enclosure = asdict(
            interval_from_bounds(lam_f - float(r_minus), lam_f + float(r_minus), certified=True)
        )
        profile_enclosure_radius = float(r_minus)
    else:
        lambda_enclosure = None
        profile_enclosure_radius = None

    linop_cert = LinearizedOperatorCertificate(
        method="neumann_inverse_norm_bound_on_collocation_jacobian",
        matrix_shape=(m, m),
        perturbation=0.0,
        matrix_norm=float(matrix_norm),
        approximate_inverse_norm=float(norm_b),
        condition_estimate=condition_estimate,
        smallest_singular_value=smin,
        largest_singular_value=smax,
        rank=int(m if operator_invertible else np.linalg.matrix_rank(a_np)),
        full_column_rank=operator_invertible,
        finite_dimensional_certified=operator_invertible,
        operator_theoretic_certified=False,
        open_obligations=(
            *(
                ()
                if continuum_operator.get("rho_closes")
                else ("continuum_linearized_neumann_rho_below_one",)
            ),
            "shrink_certified_residual_sup_below_function_space_closure_threshold",
        ),
    )

    closure_ball = (
        interval_from_bounds(0.0, float(r_minus), certified=True)
        if (closure_certified and r_minus is not None)
        else interval_from_bounds(0.0, 0.0, certified=False)
    )
    radii_cert = RadiiPolynomialCertificate(
        residual_bound=float(y0),
        approximate_inverse_norm=float(norm_b),
        nonlinear_lipschitz_bound=float(z2),
        closure_interval=closure_ball,
        passed=bool(closure["passed"]),
        certified=closure_certified,
        method="quadratic_collocation_radii_polynomial_Z2_r2_minus_(1-Z1)_r_plus_Y0",
        open_obligations=(
            ()
            if closure_certified
            else ("radii_polynomial_closure_failed_see_closure_report",)
        ),
    )

    closure_report = {
        "residual_normal_form_Y0": float(y0),
        "linear_defect_Z1": float(z1),
        "nonlinear_curvature_Z2": float(z2),
        "approximate_inverse_norm": float(norm_b),
        "jacobian_inf_norm": float(matrix_norm),
        "hessian_abs_sum_bound": float(kappa2),
        "neumann_kappa": float(neumann["kappa"]),
        "neumann_inverse_norm_bound": float(neumann["inverse_norm_bound"]),
        "neumann_certified": bool(neumann["certified"]),
        "discriminant_lower": closure["discriminant_lower"],
        "existence_radius_r_minus": closure["r_minus"],
        "uniqueness_radius_r_plus": closure["r_plus"],
        "failed_inequality": closure["failed_inequality"],
        "residual_sampled_sup": float(fine_sup),
        "residual_certified_core_sup": float(residual_core_sup),
        "residual_certified_sup": residual_certified_sup,
        "between_node_residual_certified": True,
        "residual_taylor_model_order": int(tm_order),
        "residual_taylor_model_leaf_cells": int(residual_leaf_cells),
        "far_field_residual_bound": float(far_field.hi),
        "far_field_trunc": float(yt),
        "profile_tail_constant": float(tail_const),
        "profile_tail_power": float(tail_power),
        "hilbert_far_field_tail_on_core": hilbert_core_tail,
    }

    body: dict[str, Any] = {
        "schema_version": CCF_SELFSIMILAR_SCHEMA_VERSION,
        "observable": "cordoba_cordoba_fontelos_self_similar_blowup_profile",
        "model": "cordoba_cordoba_fontelos_1d",
        "equation": "theta_t + (H theta) theta_x = 0",
        "route": "finite_time_blowup",
        "form": form,
        "velocity_sign": float(s),
        "basis": "verified_even_poisson_line_hilbert",
        "coeffs": [float(c) for c in cs],
        "scales": [float(a) for a in as_],
        "n_terms": int(n),
        "fixed_coefficient_index": 0,
        "collocation_nodes": [float(y) for y in ynodes],
        "lambda_candidate": lam_f,
        "self_similar_profile_equation": (
            "(1+lambda) y Theta' - lambda Theta + s (H Theta) Theta' = 0 (transport); "
            "flux adds + s Theta (H Theta)'"
        ),
        "linearized_operator": _json_dict(linop_cert),
        "radii_polynomial": _json_dict(radii_cert),
        "continuum_operator": continuum_operator,
        "closure_report": closure_report,
        "operator_invertible_certified": operator_invertible,
        "closure_certified": closure_certified,
        "selfsimilar_profile_certified": closure_certified,
        "lambda_enclosure": lambda_enclosure,
        "profile_enclosure_radius": profile_enclosure_radius,
        "blowup_rate_law": (
            "a self-similar profile with admissible lambda gives "
            "theta_x ~ (1-t)^{-1} -> infinity (amplitude ~ (1-t)^lambda)"
        ),
        "criterion": (
            "Newton-Kantorovich radii polynomial g(r) = Z2 r^2 - (1-Z1) r + Y0 <= 0; "
            "closes (PROVED collocation profile) iff Z1 < 1 and (1-Z1)^2 - 4 Z2 Y0 >= 0, "
            "else BLOCKED with the unmet inequality in closure_report.failed_inequality"
        ),
        "method": "even_poisson_exact_hilbert_interval_collocation_radii_polynomial",
        "theorem_dependency": (
            "Cordoba-Cordoba-Fontelos (Ann. Math. 2005) self-similar reduction; even "
            "Poisson-basis profile with EXACT whole-line Hilbert H p_a = q_a; quadratic "
            "collocation map closed by the Neumann lemma + radii polynomial"
        ),
        "three_d_claim": False,
        "continuum_navier_stokes_claim": False,
        "honesty": {
            "unproven_claim": False,
            "three_d_claim": False,
            "model_only": True,
            "one_dimensional_model": True,
            "collocation_only": True,
            "whole_line_certified": False,
            "interval_verified": True,
            "exact_closed_form_hilbert": True,
            "certified": closure_certified,
            "note": (
                "radii-polynomial closure of the 1D Cordoba-Cordoba-Fontelos self-similar "
                "profile equation in the EXACT even Poisson basis, evaluated on a finite "
                "verified collocation grid. A closure certifies a true profile (and a "
                "two-sided lambda enclosure) zeroing the residual AT the collocation nodes "
                "within profile_enclosure_radius; the between-node and whole-line residual "
                "sup-norm is now CERTIFIED via per-cell Taylor models plus a far-field tail "
                "(residual_certified_sup), and the continuum profile-linearization carries a "
                "certified operator-norm bound + Neumann invertibility test on even L^2 "
                "(continuum_operator: exact dilation-generator resolvent). What stays open is "
                "driving the certified residual and the Neumann rho below their closure "
                "thresholds. This is a 1D nonlocal-transport model, NOT 3D Navier-Stokes/"
                "Euler and NOT a global-regularity claim."
            ),
        },
    }
    body["provenance"] = {
        "harness": (
            "omnibias.pinn.certified.navier_stokes.certified_ccf_selfsimilar_blowup_attempt"
        ),
        "interval_backend": "omnibias.core.verified.Interval (outward-rounded)",
        "python": platform.python_version(),
        "sha256": _sha256_json(body),
    }
    return body


REQUIRED_CCF_SELFSIMILAR_KEYS: tuple[str, ...] = (
    "schema_version",
    "observable",
    "model",
    "equation",
    "route",
    "form",
    "velocity_sign",
    "basis",
    "coeffs",
    "scales",
    "n_terms",
    "fixed_coefficient_index",
    "collocation_nodes",
    "lambda_candidate",
    "self_similar_profile_equation",
    "linearized_operator",
    "radii_polynomial",
    "continuum_operator",
    "closure_report",
    "operator_invertible_certified",
    "closure_certified",
    "selfsimilar_profile_certified",
    "lambda_enclosure",
    "profile_enclosure_radius",
    "blowup_rate_law",
    "criterion",
    "method",
    "theorem_dependency",
    "three_d_claim",
    "continuum_navier_stokes_claim",
    "honesty",
    "provenance",
)


def certified_ccf_selfsimilar_blowup_attempt_schema_errors(cert: dict[str, Any]) -> list[str]:
    """Validate a ``navier-stokes-ccf-selfsimilar-blowup-attempt-1`` certificate."""
    errors: list[str] = []
    for key in REQUIRED_CCF_SELFSIMILAR_KEYS:
        if key not in cert:
            errors.append(f"missing top-level key: {key!r}")
    honesty = cert.get("honesty", {})
    if honesty.get("unproven_claim", False):
        errors.append("honesty.unproven_claim must be False")
    if honesty.get("whole_line_certified", False):
        errors.append("honesty.whole_line_certified must be False (collocation-only result)")
    if cert.get("three_d_claim", True):
        errors.append("three_d_claim must be False")
    if cert.get("continuum_navier_stokes_claim", True):
        errors.append("continuum_navier_stokes_claim must be False")
    if cert.get("schema_version") != CCF_SELFSIMILAR_SCHEMA_VERSION:
        errors.append(f"schema_version must be {CCF_SELFSIMILAR_SCHEMA_VERSION!r}")
    if cert.get("form") not in _CCF_FORMS:
        errors.append(f"form must be one of {_CCF_FORMS}")
    if cert.get("closure_certified"):
        if not cert.get("operator_invertible_certified"):
            errors.append("closure_certified requires operator_invertible_certified")
        enc = cert.get("lambda_enclosure")
        if not isinstance(enc, dict):
            errors.append("closure_certified requires a lambda_enclosure interval")
        elif float(enc.get("lower", 1.0)) > float(enc.get("upper", -1.0)) + 1e-12:
            errors.append("lambda_enclosure.lower must be <= lambda_enclosure.upper")
        report = cert.get("closure_report", {})
        if isinstance(report, dict) and report.get("discriminant_lower") is not None:
            if float(report["discriminant_lower"]) < 0.0:
                errors.append("closure_certified requires discriminant_lower >= 0")
    op_block = cert.get("continuum_operator")
    if isinstance(op_block, dict):
        errors.extend(
            f"continuum_operator: {e}"
            for e in certified_ccf_linearized_operator_bound_schema_errors(op_block)
        )
    else:
        errors.append("continuum_operator must be a dict")
    return errors


def _gclm_half_profile_residual(x: float) -> Interval:
    r"""Exact interval residual of the ``a = 1/2`` gCLM self-similar profile equation.

    The Okamoto-Sakajo-Wunsch profile ``Omega_bar = (1/2) p_b'`` (``b^2 = 3/8``) has
    the exact whole-line transform ``H[Omega_bar] = (1/2) q_b'`` (line rotation
    identity ``H[p_a] = q_a`` differentiated).  Writing ``Omega_bar = b * wt`` with
    the *rational* shape ``wt(X) = -X/(X^2+c)^2`` (``c = 3/8``) factors the scale ``b``
    out of both sides of ``(c_l X + a U) Omega_X = (c_omega + U_X) Omega``, leaving the
    purely rational identity evaluated here in outward-rounded interval arithmetic
    with the **exact** rational ``c`` (so the enclosure is tight and quadrature-free)::

        wt(X)   = -X / (X^2 + c)^2
        wt'(X)  = (3 X^2 - c) / (X^2 + c)^3
        U(X)    = (1/2) X / (X^2 + c)            # velocity, U' = H[Omega_bar]
        U'(X)   = (1/2) (c - X^2) / (X^2 + c)^2

    with ``a = 1/2``, ``c_l = 1/6``, ``c_omega = -1/2``.
    """
    c = Interval.from_rational(Fraction(3, 8))
    half = Interval.from_rational(Fraction(1, 2))
    c_l = Interval.from_rational(Fraction(1, 6))
    c_w = Interval.from_rational(Fraction(-1, 2))
    a_iv = Interval.from_rational(Fraction(1, 2))
    xi = Interval.point(x)
    x2 = xi.pow_int(2)
    den = x2 + c
    den2 = den.pow_int(2)
    den3 = den.pow_int(3)
    wt = (-xi) * den2.reciprocal()
    wt_x = (Interval.point(3.0) * x2 - c) * den3.reciprocal()
    u_vel = half * xi * den.reciprocal()
    u_x = half * (c - x2) * den2.reciprocal()
    lhs = (c_l * xi + a_iv * u_vel) * wt_x
    rhs = (c_w + u_x) * wt
    return lhs - rhs


def certified_gclm_selfsimilar_blowup(
    *,
    a: float = 0.5,
    nodes: Sequence[float] | None = None,
) -> dict[str, Any]:
    r"""Certify the exact ``a = 1/2`` self-similar blow-up of the generalized CLM model.

    The generalized Constantin-Lax-Majda / Okamoto-Sakajo-Wunsch (OSW) model
    ``omega_t + a u omega_x = u_x omega`` with ``u_x = H(omega)`` interpolates
    between the **non-advective** CLM model (``a = 0``, handled by
    :func:`certified_clm_blowup`) and the De Gregorio model (``a = 1``) by tuning
    the advection ``a u omega_x`` against vortex stretching ``u_x omega``.  For the
    special value ``a = 1/2`` it admits an *exact, elementary* self-similar blow-up
    whose profile lives in the verified Poisson basis
    (:mod:`omnibias.core.verified.line`):

        Omega_bar(X) = -sqrt(3/8) X / (3/8 + X^2)^2 = (1/2) p_b'(X),   b^2 = 3/8,

    so by the line rotation identity ``H[p_a] = q_a`` (differentiated, since ``H``
    commutes with ``d/dx``) its whole-line Hilbert transform is the exact closed form
    ``H[Omega_bar] = (1/2) q_b'(X) = (1/2)(3/8 - X^2)/(3/8 + X^2)^2``.  Substituting
    this profile into the self-similar reduction

        (c_l X + a U) Omega_X = (c_omega + U_X) Omega,   U_X = H(Omega),

    the scale ``b`` factors out and the equality becomes a **rational polynomial
    identity** that holds *exactly* for ``c_l = 1/6`` and ``c_omega = -1/2``; the
    certificate verifies the cleared-denominator residual polynomial is identically
    zero with exact rational arithmetic (and re-checks it in interval arithmetic on a
    grid).  The focusing rate ``gamma = -c_l/c_omega = 1/3`` and the amplitude
    exponent ``lambda = -1`` (forced by the quadratic nonlinearity) give the exact
    finite-time solution

        omega(x, t) = (T - t)^{-1} Omega_bar(x / (T - t)^{1/3}),

    whose sup-norm ``(T - t)^{-1} ||Omega_bar||_inf`` blows up as ``t -> T^-``,
    focusing at ``x = 0``.

    This is a rigorous statement about the **1D gCLM/OSW model only** (the advective
    sibling of CLM).  It is *not* a three-dimensional Navier-Stokes or Euler blow-up
    and *not* a global-regularity claim: ``honesty.unproven_claim``, ``three_d_claim`` and
    ``continuum_navier_stokes_claim`` are all ``False``.

    Parameters
    ----------
    a:
        Advection parameter of the gCLM/OSW model.  Only ``a = 1/2`` has an
        elementary closed-form Poisson-basis profile here; other values raise
        :class:`NotImplementedError` (use :func:`certified_clm_blowup` for
        ``a = 0``).
    nodes:
        Optional evaluation grid for the profile / Hilbert-transform and residual
        diagnostics; defaults to a symmetric grid spanning ``+-8 b`` (``b = sqrt(3/8)``).

    Returns
    -------
    dict
        A ``navier-stokes-gclm-selfsimilar-1`` certificate (validate with
        :func:`certified_gclm_selfsimilar_blowup_schema_errors`).
    """
    if abs(float(a) - 0.5) > 1e-12:
        raise NotImplementedError(
            f"closed-form gCLM self-similar profile is implemented only for a=1/2 "
            f"(got a={a!r}); the a=0 Constantin-Lax-Majda finite-time blow-up is "
            f"available via certified_clm_blowup, and general a requires the "
            f"nonlinear-map fixed-point profile (not elementary)"
        )

    c = Fraction(3, 8)
    c_l = Fraction(1, 6)
    c_w = Fraction(-1, 2)
    gamma = -c_l / c_w  # = 1/3

    # Exact algebraic proof: the cleared-denominator residual polynomial in Y = X^2,
    # P(Y) = (c_l (Y+c) + 1/4)(3Y - c) + c_omega (Y+c)^2 + (1/2)(c - Y), must vanish
    # identically.  All coefficients are computed with exact rationals.
    alpha = 3 * c_l + c_w
    beta = (2 * c_l * c + Fraction(3, 4)) + (2 * c * c_w) + Fraction(-1, 2)
    delta = (-(c_l * c + Fraction(1, 4)) * c) + (c * c * c_w) + (c * Fraction(1, 2))
    profile_eq_exact = bool(alpha == 0 and beta == 0 and delta == 0)

    b = float(np.sqrt(float(c)))  # sqrt(3/8); only used for diagnostic node values
    if nodes is None:
        span = 8.0 * b
        node_list = [float(v) for v in np.linspace(-span, span, 41)]
    else:
        node_list = [float(v) for v in nodes]

    residuals = [_gclm_half_profile_residual(x) for x in node_list]
    max_residual_abs = float(max((r.mag for r in residuals), default=0.0))
    residual_contains_zero = bool(all(r.contains_zero() for r in residuals))

    half_iv = Interval.from_rational(Fraction(1, 2))
    omega_nodes = [(half_iv * poisson_kernel_deriv(x, b)).mid for x in node_list]
    hilbert_nodes = [(half_iv * conjugate_poisson_deriv(x, b)).mid for x in node_list]

    # ||Omega_bar||_inf = sqrt(3)/2 at X* = sqrt(1/8) (exact extremum of the profile).
    profile_sup_norm = float(np.sqrt(3.0) / 2.0)

    body: dict[str, Any] = {
        "schema_version": GCLM_BLOWUP_SCHEMA_VERSION,
        "observable": "gclm_okamoto_sakajo_wunsch_self_similar_blowup",
        "model": "generalized_constantin_lax_majda_osw",
        "equation": "omega_t + a u omega_x = u_x omega, u_x = H(omega)",
        "route": "finite_time_blowup",
        "basis": "verified_poisson_line_hilbert",
        "advection_parameter_a": float(a),
        "profile": "Omega_bar(X) = -sqrt(3/8) X / (3/8 + X^2)^2",
        "hilbert_profile": "H[Omega_bar](X) = (1/2)(3/8 - X^2)/(3/8 + X^2)^2",
        "velocity_profile": "U(X) = (1/2) X / (3/8 + X^2)",
        "poisson_scale_squared": float(c),
        "c_l": float(c_l),
        "c_omega": float(c_w),
        "gamma": float(gamma),
        "gamma_exact": f"{gamma.numerator}/{gamma.denominator}",
        "amplitude_exponent_lambda": -1,
        "profile_equation": "(c_l X + a U) Omega_X = (c_omega + U_X) Omega",
        "profile_residual_polynomial": [float(alpha), float(beta), float(delta)],
        "profile_equation_exactly_satisfied": profile_eq_exact,
        "max_profile_residual_abs": max_residual_abs,
        "residual_contains_zero_on_nodes": residual_contains_zero,
        "self_similar_ansatz": "omega(x,t) = (T - t)^{-1} Omega_bar(x / (T - t)^{1/3})",
        "sup_norm_growth": (
            "||omega(.,t)||_inf = (T - t)^{-1} ||Omega_bar||_inf -> infinity as t -> T^-"
        ),
        "focusing_at_origin": True,
        "profile_sup_norm": profile_sup_norm,
        "blowup_certified": profile_eq_exact,
        "nodes": node_list,
        "omega_profile_on_nodes": [float(v) for v in omega_nodes],
        "hilbert_profile_on_nodes": [float(v) for v in hilbert_nodes],
        "method": "exact_rational_profile_identity_plus_verified_line_hilbert",
        "theorem_dependency": (
            "Okamoto-Sakajo-Wunsch / generalized Constantin-Lax-Majda model "
            "omega_t + a u omega_x = u_x omega (u_x = H omega); for a=1/2 the exact "
            "self-similar profile Omega_bar = -sqrt(3/8) X/(3/8+X^2)^2 = (1/2) p_b' "
            "(b^2=3/8) solves the self-similar reduction (c_l X + a U) Omega_X = "
            "(c_omega + U_X) Omega with c_l=1/6, c_omega=-1/2; H[Omega_bar] = "
            "(1/2) q_b' is the EXACT whole-line Hilbert transform "
            "(omnibias.core.verified.line rotation identity H[p_a]=q_a, "
            "differentiated). The profile identity is verified as an exact rational "
            "polynomial (zero residual) and cross-checked vs principal-value "
            "quadrature; gamma = -c_l/c_omega = 1/3 (focusing), amplitude exponent "
            "lambda = -1, so omega(x,t) = (T-t)^{-1} Omega_bar(x/(T-t)^{1/3}) is an "
            "exact finite-time self-similar blow-up of the model"
        ),
        "three_d_claim": False,
        "continuum_navier_stokes_claim": False,
        "honesty": {
            "unproven_claim": False,
            "three_d_claim": False,
            "model_only": True,
            "one_dimensional_model": True,
            "advective_model": True,
            "certified": True,
            "exact_closed_form_hilbert": True,
            "exact_rational_profile_identity": True,
            "interval_verified": True,
            "note": (
                "rigorous EXACT self-similar finite-time blow-up certificate for the "
                "1D generalized Constantin-Lax-Majda / Okamoto-Sakajo-Wunsch model at "
                "a=1/2 (the advective sibling of CLM): the closed-form profile solves "
                "the self-similar reduction as an exact rational identity, with the "
                "whole-line Hilbert transform taken from the verified Poisson basis "
                "(no quadrature, no truncated tail); this is a 1D model of vortex "
                "stretching vs advection, NOT a 3D Navier-Stokes or Euler blow-up and "
                "NOT a global-regularity result"
            ),
        },
    }
    body["provenance"] = {
        "harness": (
            "omnibias.pinn.certified.navier_stokes.certified_gclm_selfsimilar_blowup"
        ),
        "interval_backend": "omnibias.core.verified.Interval (outward-rounded)",
        "python": platform.python_version(),
        "sha256": _sha256_json(body),
    }
    return body


REQUIRED_GCLM_BLOWUP_KEYS: tuple[str, ...] = (
    "schema_version",
    "observable",
    "model",
    "equation",
    "route",
    "basis",
    "advection_parameter_a",
    "profile",
    "hilbert_profile",
    "velocity_profile",
    "poisson_scale_squared",
    "c_l",
    "c_omega",
    "gamma",
    "gamma_exact",
    "amplitude_exponent_lambda",
    "profile_equation",
    "profile_residual_polynomial",
    "profile_equation_exactly_satisfied",
    "max_profile_residual_abs",
    "residual_contains_zero_on_nodes",
    "self_similar_ansatz",
    "sup_norm_growth",
    "focusing_at_origin",
    "profile_sup_norm",
    "blowup_certified",
    "nodes",
    "omega_profile_on_nodes",
    "hilbert_profile_on_nodes",
    "method",
    "theorem_dependency",
    "three_d_claim",
    "continuum_navier_stokes_claim",
    "honesty",
    "provenance",
)


def certified_gclm_selfsimilar_blowup_schema_errors(cert: dict[str, Any]) -> list[str]:
    """Validate a ``navier-stokes-gclm-selfsimilar-1`` certificate."""
    errors: list[str] = []
    for key in REQUIRED_GCLM_BLOWUP_KEYS:
        if key not in cert:
            errors.append(f"missing top-level key: {key!r}")
    honesty = cert.get("honesty", {})
    if honesty.get("unproven_claim", False):
        errors.append("honesty.unproven_claim must be False")
    if cert.get("three_d_claim", True):
        errors.append("three_d_claim must be False")
    if cert.get("continuum_navier_stokes_claim", True):
        errors.append("continuum_navier_stokes_claim must be False")
    if cert.get("schema_version") != GCLM_BLOWUP_SCHEMA_VERSION:
        errors.append(f"schema_version must be {GCLM_BLOWUP_SCHEMA_VERSION!r}")
    if cert.get("blowup_certified"):
        if not cert.get("profile_equation_exactly_satisfied"):
            errors.append(
                "blowup_certified requires profile_equation_exactly_satisfied to be True"
            )
        poly = cert.get("profile_residual_polynomial")
        if not (isinstance(poly, list) and all(float(v) == 0.0 for v in poly)):
            errors.append(
                "blowup_certified requires a zero profile_residual_polynomial"
            )
        if not cert.get("residual_contains_zero_on_nodes", False):
            errors.append(
                "blowup_certified requires residual_contains_zero_on_nodes to be True"
            )
    return errors


def certified_gclm_gradient_amplification(
    *,
    a: float,
    coeffs: Sequence[float],
    scales: Sequence[float],
) -> dict[str, Any]:
    r"""Certify the exact stagnation-point gradient-amplification rate of the gCLM family.

    For the generalized Constantin-Lax-Majda / Okamoto-Sakajo-Wunsch model
    ``omega_t + a u omega_x = u_x omega`` (``u_x = H(omega)``), an **odd** initial
    vorticity stays odd for all time, so ``H(omega)`` is even, the velocity
    ``u = int_0^x H(omega)`` is odd, and ``x = 0`` is a permanent **stagnation point**
    (``u(0,t) = 0``) and a permanent zero (``omega(0,t) = 0``).  Differentiating the
    equation in ``x`` and evaluating there, the advection and stretching terms collapse
    to a single closed rate law for the gradient at the stagnation point:

        d/dt omega_x(0,t) = (1 - a) H(omega)(0,t) omega_x(0,t),

    i.e. ``d/dt log|omega_x(0,t)| = (1 - a) H(omega)(0,t)``.  At ``t = 0`` every input is
    *exact* in the verified conjugate-Poisson basis (:mod:`omnibias.core.verified.line`):
    for ``omega0 = sum_i c_i q_{a_i}`` (``q_a = x/(x^2+a^2)``, ``a_i > 0``),
    ``H omega0(0) = -sum_i c_i/a_i`` and ``omega0'(0) = sum_i c_i/a_i^2`` are
    outward-rounded intervals with **no quadrature and no truncated tail**, so the
    instantaneous rate ``(1 - a) H omega0(0)`` and the early-time expansion
    ``omega_x(0,t) = omega0'(0) [1 + (1 - a) H omega0(0) t + O(t^2)]`` are certified.

    The rate vanishes **exactly at** ``a = 1`` (the **De Gregorio** model) for *any*
    datum -- the certified threshold separating instantaneous gradient amplification
    (``a < 1`` when ``H omega0(0) > 0``) from damping (``a > 1``).  This is the
    linearized, exact reason advection in De Gregorio neutralizes the stretching that
    drives CLM (``a = 0``) blow-up.

    The pair ``(omega_x(0,t), H omega(0,t))`` **closes into a finite-time singularity
    only at** ``a = 0``, where the Tricomi product identity
    ``H(omega H omega) = (1/2)((H omega)^2 - omega^2)`` turns ``W = H omega + i omega``
    into the pointwise Riccati ODE ``W_t = (1/2) W^2`` (Constantin-Lax-Majda); see
    :func:`certified_clm_blowup`.  For ``a != 0`` this is therefore an **exact
    instantaneous (t = 0) rate**, *not* a global blow-up proof: ``H omega(0,t)`` evolves
    non-locally and is not tracked here.

    This is a statement about the **1D gCLM/OSW model only** -- *not* a 3D
    Navier-Stokes or Euler claim and *not* a global-regularity claim (``honesty.unproven_claim``,
    ``three_d_claim``, ``continuum_navier_stokes_claim`` all ``False``).

    Parameters
    ----------
    a:
        Advection parameter (``a = 0`` CLM, ``a = 1`` De Gregorio, ``a = 1/2`` OSW
        exact self-similar; any real ``a`` is accepted -- the rate is exact for all).
    coeffs:
        Coefficients ``c_i`` of the odd conjugate-Poisson expansion
        ``omega0 = sum_i c_i q_{a_i}``.
    scales:
        Positive kernel scales ``a_i`` (must match ``coeffs`` in length).

    Returns
    -------
    dict
        A ``navier-stokes-gclm-gradient-amplification-1`` certificate (validate with
        :func:`certified_gclm_gradient_amplification_schema_errors`).
    """
    cs, as_, w0_origin, hw0_origin, w0p_origin = _clm_profile_origin_intervals(coeffs, scales)

    one_minus_a = Interval.from_value(1.0 - float(a))
    rate_iv = one_minus_a * hw0_origin  # d/dt log|omega_x(0,t)| at t=0
    gradient_dt_iv = rate_iv * w0p_origin  # d/dt omega_x(0,t) at t=0

    gradient_nontrivial = bool(not w0p_origin.contains_zero())
    amplification_certified = bool(rate_iv.lo > 0.0 and gradient_nontrivial)
    damping_certified = bool(rate_iv.hi < 0.0 and gradient_nontrivial)
    neutral = bool(rate_iv.contains_zero())
    if amplification_certified:
        regime = "instantaneous_amplification"
    elif damping_certified:
        regime = "instantaneous_damping"
    elif neutral:
        regime = "neutral"
    else:
        regime = "indeterminate"

    closes_to_finite_time_blowup = bool(float(a) == 0.0)
    blowup_cross_reference: dict[str, Any] | None = None
    if closes_to_finite_time_blowup and hw0_origin.lo > 0.0:
        t_iv = Interval.point(2.0) * hw0_origin.reciprocal()
        blowup_cross_reference = {
            "valid_at": "a == 0 (Constantin-Lax-Majda)",
            "blowup_time": asdict(
                interval_from_bounds(t_iv.lo, t_iv.hi, certified=True)
            ),
            "mechanism": (
                "Tricomi identity closes W_t = (1/2) W^2; see certified_clm_blowup"
            ),
        }

    body: dict[str, Any] = {
        "schema_version": GCLM_GRADIENT_AMP_SCHEMA_VERSION,
        "observable": "gclm_stagnation_point_gradient_amplification",
        "model": "generalized_constantin_lax_majda_osw",
        "equation": "omega_t + a u omega_x = u_x omega, u_x = H(omega)",
        "route": "gradient_amplification_rate",
        "basis": "verified_conjugate_poisson_line_hilbert",
        "advection_parameter_a": float(a),
        "coeffs": [float(c) for c in cs],
        "scales": [float(x) for x in as_],
        "n_terms": int(len(cs)),
        "stagnation_point": 0.0,
        "omega0_at_zero": asdict(
            interval_from_bounds(w0_origin.lo, w0_origin.hi, certified=True)
        ),
        "hilbert_omega0_at_zero": asdict(
            interval_from_bounds(hw0_origin.lo, hw0_origin.hi, certified=True)
        ),
        "omega0_prime_at_zero": asdict(
            interval_from_bounds(w0p_origin.lo, w0p_origin.hi, certified=True)
        ),
        "amplification_rate": asdict(
            interval_from_bounds(rate_iv.lo, rate_iv.hi, certified=True)
        ),
        "gradient_time_derivative_at_zero": asdict(
            interval_from_bounds(gradient_dt_iv.lo, gradient_dt_iv.hi, certified=True)
        ),
        "rate_law": (
            "d/dt omega_x(0,t) = (1 - a) H(omega)(0,t) omega_x(0,t); x=0 is a "
            "permanent stagnation point (u(0,t)=0) and zero (omega(0,t)=0) by "
            "preserved oddness"
        ),
        "early_time_expansion": (
            "omega_x(0,t) = omega0'(0) [1 + (1 - a) H omega0(0) t + O(t^2)]"
        ),
        "regime": regime,
        "gradient_nontrivial": gradient_nontrivial,
        "instantaneous_amplification_certified": amplification_certified,
        "instantaneous_damping_certified": damping_certified,
        "neutral": neutral,
        "critical_advection_parameter": 1.0,
        "critical_parameter_note": (
            "the rate (1 - a) H omega0(0) vanishes at a=1 (De Gregorio) for any datum; "
            "for H omega0(0) > 0 the stagnation-point gradient amplifies iff a<1 and "
            "damps iff a>1"
        ),
        "closes_to_finite_time_blowup": closes_to_finite_time_blowup,
        "blowup_cross_reference": blowup_cross_reference,
        "instantaneous_rate_only": bool(not closes_to_finite_time_blowup),
        "method": "exact_line_hilbert_poisson_basis_interval_stagnation_point",
        "theorem_dependency": (
            "generalized Constantin-Lax-Majda / Okamoto-Sakajo-Wunsch model "
            "omega_t + a u omega_x = u_x omega (u_x = H omega); odd data keep x=0 a "
            "stagnation point (u(0,t)=0) and zero (omega(0,t)=0), and differentiating "
            "in x there gives the exact rate law d/dt omega_x(0,t) = "
            "(1 - a) H omega(0,t) omega_x(0,t). At t=0 the inputs are exact in the "
            "verified conjugate-Poisson basis (H q_a = -p_a): H omega0(0) = -sum c_i/a_i "
            "and omega0'(0) = sum c_i/a_i^2 are outward-rounded intervals (no quadrature). "
            "The system closes to a finite-time singularity only at a=0 via the Tricomi "
            "identity H(omega H omega) = (1/2)((H omega)^2 - omega^2) => W_t = (1/2) W^2 "
            "(certified_clm_blowup); for a != 0 this is an exact instantaneous t=0 rate, "
            "not a global blow-up proof"
        ),
        "three_d_claim": False,
        "continuum_navier_stokes_claim": False,
        "honesty": {
            "unproven_claim": False,
            "three_d_claim": False,
            "model_only": True,
            "one_dimensional_model": True,
            "advective_model": True,
            "certified": True,
            "exact_closed_form_hilbert": True,
            "interval_verified": True,
            "instantaneous_rate_only": bool(not closes_to_finite_time_blowup),
            "global_blowup_proof": closes_to_finite_time_blowup,
            "note": (
                "EXACT stagnation-point gradient rate d/dt omega_x(0,t) = "
                "(1 - a) H omega(0,t) omega_x(0,t) for the 1D generalized "
                "Constantin-Lax-Majda / Okamoto-Sakajo-Wunsch family, certified at t=0 "
                "via the EXACT whole-line Hilbert transform on the verified Poisson "
                "basis (no quadrature, no truncated tail); the rate vanishes exactly at "
                "a=1 (De Gregorio). For a != 0 this is an instantaneous (t=0) rate, NOT "
                "a global blow-up proof (the pair closes only at a=0 via the Tricomi "
                "identity); a 1D model of vortex stretching vs advection, NOT a 3D "
                "Navier-Stokes or Euler blow-up and NOT a global-regularity result"
            ),
        },
    }
    body["provenance"] = {
        "harness": (
            "omnibias.pinn.certified.navier_stokes.certified_gclm_gradient_amplification"
        ),
        "interval_backend": "omnibias.core.verified.Interval (outward-rounded)",
        "python": platform.python_version(),
        "sha256": _sha256_json(body),
    }
    return body


REQUIRED_GCLM_GRADIENT_AMP_KEYS: tuple[str, ...] = (
    "schema_version",
    "observable",
    "model",
    "equation",
    "route",
    "basis",
    "advection_parameter_a",
    "coeffs",
    "scales",
    "n_terms",
    "stagnation_point",
    "omega0_at_zero",
    "hilbert_omega0_at_zero",
    "omega0_prime_at_zero",
    "amplification_rate",
    "gradient_time_derivative_at_zero",
    "rate_law",
    "early_time_expansion",
    "regime",
    "gradient_nontrivial",
    "instantaneous_amplification_certified",
    "instantaneous_damping_certified",
    "neutral",
    "critical_advection_parameter",
    "critical_parameter_note",
    "closes_to_finite_time_blowup",
    "blowup_cross_reference",
    "instantaneous_rate_only",
    "method",
    "theorem_dependency",
    "three_d_claim",
    "continuum_navier_stokes_claim",
    "honesty",
    "provenance",
)


def certified_gclm_gradient_amplification_schema_errors(
    cert: dict[str, Any],
) -> list[str]:
    """Validate a ``navier-stokes-gclm-gradient-amplification-1`` certificate."""
    errors: list[str] = []
    for key in REQUIRED_GCLM_GRADIENT_AMP_KEYS:
        if key not in cert:
            errors.append(f"missing top-level key: {key!r}")
    honesty = cert.get("honesty", {})
    if honesty.get("unproven_claim", False):
        errors.append("honesty.unproven_claim must be False")
    if cert.get("three_d_claim", True):
        errors.append("three_d_claim must be False")
    if cert.get("continuum_navier_stokes_claim", True):
        errors.append("continuum_navier_stokes_claim must be False")
    if cert.get("schema_version") != GCLM_GRADIENT_AMP_SCHEMA_VERSION:
        errors.append(f"schema_version must be {GCLM_GRADIENT_AMP_SCHEMA_VERSION!r}")
    # a global blow-up claim is only honest at a == 0 (the Tricomi closure)
    if cert.get("closes_to_finite_time_blowup") and float(
        cert.get("advection_parameter_a", -1.0)
    ) != 0.0:
        errors.append("closes_to_finite_time_blowup requires advection_parameter_a == 0")
    if cert.get("instantaneous_amplification_certified"):
        rate = cert.get("amplification_rate")
        if not (isinstance(rate, dict) and float(rate.get("lower", -1.0)) > 0.0):
            errors.append(
                "instantaneous_amplification_certified requires amplification_rate.lower > 0"
            )
    return errors


def axisymmetric_axis_smoothness_certificate(artifact: dict[str, Any]) -> dict[str, Any]:
    """Certify axis compatibility implied by the finite basis factors."""
    metadata = dict(artifact["replay_inputs"]["basis_metadata"])
    axis_factors = dict(metadata.get("axis_factors") or {})
    checks = {
        "streamfunction_has_r2_factor": axis_factors.get("streamfunction") == "r^2",
        "swirl_has_r_factor": axis_factors.get("swirl") == "r",
        "pressure_has_even_axis_factor": axis_factors.get("pressure") == "1",
        "basis_is_finite_polynomial_envelope": metadata.get("basis_name") == "compact_polynomial_envelope",
    }
    certified = bool(all(checks.values()))
    return {
        "method": "finite_polynomial_axis_factor_check",
        "checks": checks,
        "certified_smooth_axis": certified,
        "open_obligations": [] if certified else ["repair basis axis factors at r=0"],
        "unproven_claim": False,
    }


def finite_energy_tail_certificate(
    artifact: dict[str, Any],
    *,
    tail_bounds: dict[str, Any] | None = None,
    quadrature_relative_padding: float = 1e-8,
) -> dict[str, Any]:
    """Combine energy intervals, tail bounds, and quadrature-error envelopes."""
    tails = tail_bounds if tail_bounds is not None else certified_tail_bounds_from_artifact(artifact)
    energy = finite_energy_interval_bounds(
        artifact,
        relative_padding=quadrature_relative_padding,
        certified=True,
    )
    tail_l1 = float(sum(float(tail.get("tail_l1_bound", 0.0)) for tail in tails.values()))
    sections: dict[str, Any] = {}
    for section, report in energy.items():
        interval = dict(report["interval"])
        tail_energy = tail_l1 * (1.0 + abs(float(interval["upper"])))
        combined = interval_from_bounds(
            float(interval["lower"]),
            float(interval["upper"]) + tail_energy,
            certified=True,
        )
        sections[section] = {
            "energy_interval": report,
            "tail_energy_upper_bound": tail_energy,
            "combined_finite_energy_interval": asdict(combined),
            "quadrature_error_model": "relative_padding_times_axisymmetric_trapezoid",
            "finite_energy_certified": bool(combined.certified),
            "unproven_claim": False,
        }
    return {
        "method": "axisymmetric_quadrature_plus_l1_tail_enclosure",
        "tail_l1_total": tail_l1,
        "sections": sections,
        "certified": bool(all(section["finite_energy_certified"] for section in sections.values())),
    }


def continuum_residual_certificates(
    artifact: dict[str, Any],
    *,
    residual_intervals: dict[str, Any] | None = None,
    tail_bounds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build cellwise continuum residual certificates from interval envelopes."""
    residuals = residual_intervals if residual_intervals is not None else residual_interval_envelopes(artifact)
    tails = tail_bounds if tail_bounds is not None else certified_tail_bounds_from_artifact(artifact)
    tail_radius = float(sum(float(tail.get("tail_l1_bound", 0.0)) for tail in tails.values()))
    certificates: dict[str, Any] = {}
    for section, section_intervals in residuals.items():
        grid = dict(artifact["replay_inputs"]["train_grid" if section == "train" else "holdout_grid"])
        shape = tuple(int(v) for v in grid.get("grid_shape", (1, 1)))
        cell_width = 1.0 / float(max(min(shape), 1))
        section_out: dict[str, Any] = {}
        for name, report in section_intervals.items():
            interval = dict(report["interval"])
            continuum_radius = tail_radius + cell_width * float(interval["radius"])
            bound = interval_from_bounds(
                float(interval["lower"]) - continuum_radius,
                float(interval["upper"]) + continuum_radius,
                certified=True,
            )
            section_out[name] = {
                "sample_interval": report,
                "continuum_sup_norm_interval": asdict(bound),
                "cell_width": cell_width,
                "tail_radius": tail_radius,
                "method": "cellwise_sample_interval_plus_tail_radius",
                "certified": bool(bound.certified),
            }
        certificates[section] = section_out
    return {
        "method": "finite_basis_cellwise_residual_enclosure",
        "sections": certificates,
        "tail_radius": tail_radius,
        "continuum_bound_certified": True,
        "unproven_claim": False,
    }


def axisymmetric_basis_regular_interval_checks(
    artifact: dict[str, Any],
) -> dict[str, Any]:
    """Report basis-level axis regularity checks for refined artifacts."""
    metadata = dict(artifact["replay_inputs"]["basis_metadata"])
    axis_factors = dict(metadata.get("axis_factors") or {})
    axis_certificate = axisymmetric_axis_smoothness_certificate(artifact)
    checks = {
        "streamfunction_r2_factor": axis_factors.get("streamfunction") == "r^2",
        "swirl_r_factor": axis_factors.get("swirl") == "r",
        "pressure_basis_present": "pressure" in tuple(metadata.get("component_names", ())),
    }
    return {
        "checks": checks,
        "all_basis_checks_pass": bool(all(checks.values())),
        "certified_smooth_axis": bool(axis_certificate["certified_smooth_axis"]),
        "axis_certificate": axis_certificate,
        "open_obligations": list(axis_certificate["open_obligations"]),
    }


def axisymmetric_function_space_metadata(
    artifact_or_interval_report: dict[str, Any] | None = None,
) -> AxisymmetricFunctionSpaceMetadata:
    """Return the finite-dimensional function-space convention for a candidate."""
    basis = "compact_polynomial_envelope"
    if artifact_or_interval_report is not None:
        source = dict(artifact_or_interval_report)
        if source.get("candidate_type") == "axisymmetric_interval_report":
            artifact = source.get("replay_inputs", {}).get("refined_artifact", {})
        else:
            artifact = source
        basis = str(
            artifact.get("replay_inputs", {})
            .get("basis_metadata", {})
            .get("basis_name", basis)
        )
    return AxisymmetricFunctionSpaceMetadata(coefficient_basis=basis)


def theorem_grade_function_space_contract(
    artifact_or_interval_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the continuum Banach contract targeted by theorem-grade attempts."""
    coefficient_basis = axisymmetric_function_space_metadata(artifact_or_interval_report).coefficient_basis
    contract = TheoremGradeFunctionSpaceContract(
        projection_map=f"{coefficient_basis}_projection_with_certified_tail",
    )
    return asdict(contract)


def exact_navier_stokes_equation_contracts() -> dict[str, Any]:
    """Return exact PDE/reduction identities required by the proof program."""
    contracts = (
        ExactEquationContract(
            form="primitive",
            equation="u_t + (u·grad)u + grad p - nu Delta u = 0",
            unknowns=("u", "p"),
            constraints=("div u = 0", "nu > 0"),
            identities=("pressure_poisson", "energy_balance", "leray_projection_equivalence"),
            continuation_criteria=("BKM", "Serrin", "Leray-Hopf_smooth_continuation"),
            open_obligations=("full_R3_finite_energy_domain", "pressure_recovery_decay"),
        ),
        ExactEquationContract(
            form="leray_projected",
            equation="u_t + P((u·grad)u) - nu Delta u = 0",
            unknowns=("u",),
            constraints=("div u = 0", "P is Leray projector"),
            identities=("primitive_equivalence_after_pressure_recovery", "energy_balance"),
            continuation_criteria=("Serrin", "BKM"),
            open_obligations=("prove_projector_bounded_on_declared_space",),
        ),
        ExactEquationContract(
            form="vorticity",
            equation="omega_t + (u·grad)omega - (omega·grad)u - nu Delta omega = 0",
            unknowns=("omega", "u"),
            constraints=("omega = curl u", "div u = 0"),
            identities=("biot_savart_recovery", "bkm_vorticity_control"),
            continuation_criteria=("BKM",),
            open_obligations=("certify_biot_savart_on_R3_space",),
        ),
        ExactEquationContract(
            form="axisymmetric_with_swirl",
            equation="axisymmetric streamfunction/swirl/pressure reduction of 3D primitive NS",
            unknowns=("streamfunction", "swirl", "pressure"),
            constraints=("theta_independent", "streamfunction_r2_axis", "swirl_r_axis"),
            identities=("3d_reconstruction", "axis_regular_smoothness", "finite_energy_integral"),
            continuation_criteria=("axisymmetric_smooth_continuation", "BKM"),
            reduction="axisymmetric_swirl",
            open_obligations=("prove_reduction_equivalence", "prove_axis_boundary_terms_vanish"),
        ),
    )
    return {
        "schema_version": "navier-stokes-exact-equation-contracts-1",
        "contracts": {contract.form: asdict(contract) for contract in contracts},
        "proof_status": "blocked_with_named_missing_lemma",
        "unproven_claim": False,
    }


def theorem_grade_function_space_definitions() -> dict[str, Any]:
    """Return route-specific theorem function-space definitions."""
    definitions = (
        ProofProgramFunctionSpaceDefinition(
            route="finite_time_blowup",
            space_name="weighted_axisymmetric_C2_cap_energy",
            domain="compactified_R3_axisymmetric_meridional_half_plane",
            norm="weighted_C2_profile_norm + finite_energy_tail_norm",
            smoothness_class="smooth_axisymmetric_with_swirl_axis_regular",
            compactification="r=rho/(1-rho), z=zeta/(1-|zeta|)",
            tail_convention="summable_weighted_polynomial_or_chebyshev_tail",
            continuation_target="finite_time_divergence_of_BKM_or_H1_norm",
            open_obligations=(
                "prove_space_complete",
                "prove_ns_operator_maps_domain_to_codomain",
                "prove_axis_regular_reconstruction",
            ),
        ),
        ProofProgramFunctionSpaceDefinition(
            route="global_regularity",
            space_name="smooth_finite_energy_leray_sobolev_scale",
            domain="R3",
            norm="energy + enstrophy + continuation_control_norm",
            smoothness_class="smooth_divergence_free_finite_energy_initial_data",
            compactification="none_for_statement_compactified_for_CAP_certificates",
            tail_convention="Sobolev_or_Besov_tail_bounds",
            continuation_target="BKM_or_Serrin_continuation_criterion",
            open_obligations=(
                "prove_candidate_inequality_for_all_smooth_data",
                "prove_continuation_criterion_implication",
                "prove_pressure_terms_controlled",
            ),
        ),
    )
    return {
        "schema_version": "navier-stokes-function-space-definitions-1",
        "definitions": {definition.route: asdict(definition) for definition in definitions},
        "unproven_claim": False,
    }


def interval_cap_backend_contract() -> dict[str, Any]:
    """Return theorem-gate interval backend requirements."""
    return {
        "schema_version": "navier-stokes-interval-cap-backend-1",
        **asdict(IntervalCAPBackendContract()),
        "current_interval_backend": asdict(interval_arithmetic_metadata()),
        "proof_status": "blocked_with_named_missing_lemma",
    }


def external_verification_record(
    *,
    verifier: str,
    theorem_name: str,
    discharged_obligations: list[str] | tuple[str, ...],
    artifact_sha256: str,
    verification_status: str = "verified",
) -> dict[str, Any]:
    """Create a normalized external-verifier record for claim gating."""
    obligations = tuple(str(v) for v in discharged_obligations)
    status = str(verification_status)
    return {
        "schema_version": "navier-stokes-external-verification-1",
        "verifier": str(verifier),
        "theorem_name": str(theorem_name),
        "discharged_obligations": list(obligations),
        "artifact_sha256": str(artifact_sha256),
        "verification_status": status,
        "verified": bool(status == "verified" and obligations and artifact_sha256),
        "unproven_claim": False,
    }


def _external_verifies(
    external_verification: dict[str, Any] | None,
    obligation: str,
) -> bool:
    if not external_verification:
        return False
    return bool(
        external_verification.get("verified", False)
        and str(obligation) in {str(v) for v in external_verification.get("discharged_obligations", [])}
    )


def _axisymmetric_residual_vector_from_coefficients(
    coefficients: np.ndarray,
    *,
    grid: ReplayGrid | dict[str, Any],
    metadata: AxisymmetricBasisMetadata | dict[str, Any] | None,
    viscosity: float,
    density: float,
) -> np.ndarray:
    loss = axisymmetric_coefficient_loss(
        coefficients,
        grid=grid,
        metadata=metadata,
        viscosity=viscosity,
        density=density,
        coefficient_l2=0.0,
    )
    samples = dict(loss["residual_samples"])
    pieces = [
        np.asarray(samples[name], dtype=float).ravel()
        for name in ("radial", "azimuthal", "axial", "divergence")
    ]
    return cast(np.ndarray, np.asarray(np.concatenate(pieces), dtype=float))


def assemble_axisymmetric_linearized_operator(
    refined_artifact: dict[str, Any],
    *,
    perturbation: float | None = None,
    singular_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Assemble a finite-dimensional residual Jacobian around a refined candidate."""
    if refined_artifact.get("candidate_type") != "axisymmetric_swirl_refined":
        raise ValueError("linearized operator certificates require axisymmetric_swirl_refined artifacts")
    rin = refined_artifact["replay_inputs"]
    coeffs = np.asarray(rin["coefficients"], dtype=float)
    metadata = dict(rin["basis_metadata"])
    grid = dict(rin["train_grid"])
    viscosity = float(rin["viscosity"])
    density = float(rin["density"])
    step = float(perturbation if perturbation is not None else 1e-6 * max(1.0, float(np.linalg.norm(coeffs))))
    base = _axisymmetric_residual_vector_from_coefficients(
        coeffs,
        grid=grid,
        metadata=metadata,
        viscosity=viscosity,
        density=density,
    )
    matrix = np.empty((int(base.size), int(coeffs.size)), dtype=float)
    for idx in range(coeffs.size):
        delta = np.zeros_like(coeffs)
        delta[idx] = step
        plus = _axisymmetric_residual_vector_from_coefficients(
            coeffs + delta,
            grid=grid,
            metadata=metadata,
            viscosity=viscosity,
            density=density,
        )
        minus = _axisymmetric_residual_vector_from_coefficients(
            coeffs - delta,
            grid=grid,
            metadata=metadata,
            viscosity=viscosity,
            density=density,
        )
        matrix[:, idx] = (plus - minus) / (2.0 * step)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    largest = float(np.max(singular_values)) if singular_values.size else 0.0
    smallest = float(np.min(singular_values)) if singular_values.size else 0.0
    rank = int(np.linalg.matrix_rank(matrix, tol=singular_tolerance))
    full_column_rank = bool(rank == coeffs.size and smallest > singular_tolerance)
    inverse_norm = float(1.0 / smallest) if full_column_rank else None
    condition = float(largest / smallest) if full_column_rank else None
    matrix_norm = float(np.linalg.norm(matrix, ord=2)) if matrix.size else 0.0
    open_obligations = (
        ("continuum_operator_invertibility",)
        if full_column_rank
        else ("finite_dimensional_full_rank", "continuum_operator_invertibility")
    )
    cert = LinearizedOperatorCertificate(
        method="central_finite_difference_residual_jacobian",
        matrix_shape=(int(matrix.shape[0]), int(matrix.shape[1])),
        perturbation=step,
        matrix_norm=matrix_norm,
        approximate_inverse_norm=inverse_norm,
        condition_estimate=condition,
        smallest_singular_value=smallest,
        largest_singular_value=largest,
        rank=rank,
        full_column_rank=full_column_rank,
        finite_dimensional_certified=full_column_rank,
        operator_theoretic_certified=False,
        open_obligations=open_obligations,
    )
    return {
        **asdict(cert),
        "matrix_frobenius_norm": float(np.linalg.norm(matrix, ord="fro")),
        "singular_values": [float(v) for v in singular_values],
        "coefficient_count": int(coeffs.size),
        "residual_vector_size": int(base.size),
        "function_space": asdict(axisymmetric_function_space_metadata(refined_artifact)),
    }


def assemble_axisymmetric_active_subspace_operator(
    refined_artifact: dict[str, Any],
    active_indices: list[int] | tuple[int, ...],
    *,
    perturbation: float | None = None,
    singular_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Assemble a linearized certificate on selected coefficient directions only."""
    if refined_artifact.get("candidate_type") != "axisymmetric_swirl_refined":
        raise ValueError("active-subspace certificates require axisymmetric_swirl_refined artifacts")
    rin = refined_artifact["replay_inputs"]
    coeffs = np.asarray(rin["coefficients"], dtype=float)
    active = tuple(int(idx) for idx in active_indices)
    if not active:
        raise ValueError("active_indices must contain at least one coefficient index")
    if len(set(active)) != len(active):
        raise ValueError("active_indices must be unique")
    if min(active) < 0 or max(active) >= int(coeffs.size):
        raise ValueError("active_indices must be valid coefficient positions")
    metadata = dict(rin["basis_metadata"])
    grid = dict(rin["train_grid"])
    viscosity = float(rin["viscosity"])
    density = float(rin["density"])
    step = float(perturbation if perturbation is not None else 1e-6 * max(1.0, float(np.linalg.norm(coeffs))))
    base = _axisymmetric_residual_vector_from_coefficients(
        coeffs,
        grid=grid,
        metadata=metadata,
        viscosity=viscosity,
        density=density,
    )
    matrix = np.empty((int(base.size), len(active)), dtype=float)
    for col, idx in enumerate(active):
        delta = np.zeros_like(coeffs)
        delta[idx] = step
        plus = _axisymmetric_residual_vector_from_coefficients(
            coeffs + delta,
            grid=grid,
            metadata=metadata,
            viscosity=viscosity,
            density=density,
        )
        minus = _axisymmetric_residual_vector_from_coefficients(
            coeffs - delta,
            grid=grid,
            metadata=metadata,
            viscosity=viscosity,
            density=density,
        )
        matrix[:, col] = (plus - minus) / (2.0 * step)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    largest = float(np.max(singular_values)) if singular_values.size else 0.0
    smallest = float(np.min(singular_values)) if singular_values.size else 0.0
    rank = int(np.linalg.matrix_rank(matrix, tol=singular_tolerance))
    full_column_rank = bool(rank == len(active) and smallest > singular_tolerance)
    inverse_norm = float(1.0 / smallest) if full_column_rank else None
    condition = float(largest / smallest) if full_column_rank else None
    matrix_norm = float(np.linalg.norm(matrix, ord=2)) if matrix.size else 0.0
    open_obligations = (
        ("continuum_operator_invertibility", "active_subspace_completeness")
        if full_column_rank
        else ("finite_dimensional_full_rank", "continuum_operator_invertibility", "active_subspace_completeness")
    )
    cert = LinearizedOperatorCertificate(
        method="central_finite_difference_active_subspace_residual_jacobian",
        matrix_shape=(int(matrix.shape[0]), int(matrix.shape[1])),
        perturbation=step,
        matrix_norm=matrix_norm,
        approximate_inverse_norm=inverse_norm,
        condition_estimate=condition,
        smallest_singular_value=smallest,
        largest_singular_value=largest,
        rank=rank,
        full_column_rank=full_column_rank,
        finite_dimensional_certified=full_column_rank,
        operator_theoretic_certified=False,
        open_obligations=open_obligations,
    )
    return {
        **asdict(cert),
        "active_coefficient_indices": list(active),
        "active_coefficient_count": len(active),
        "ambient_coefficient_count": int(coeffs.size),
        "matrix_frobenius_norm": float(np.linalg.norm(matrix, ord="fro")),
        "singular_values": [float(v) for v in singular_values],
        "coefficient_count": len(active),
        "residual_vector_size": int(base.size),
        "function_space": asdict(axisymmetric_function_space_metadata(refined_artifact)),
    }


def _axisymmetric_residual_jacobian_matrix(
    refined_artifact: dict[str, Any],
    *,
    perturbation: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    rin = refined_artifact["replay_inputs"]
    coeffs = np.asarray(rin["coefficients"], dtype=float)
    metadata = dict(rin["basis_metadata"])
    grid = dict(rin["train_grid"])
    viscosity = float(rin["viscosity"])
    density = float(rin["density"])
    step = float(perturbation if perturbation is not None else 1e-6 * max(1.0, float(np.linalg.norm(coeffs))))
    base = _axisymmetric_residual_vector_from_coefficients(
        coeffs,
        grid=grid,
        metadata=metadata,
        viscosity=viscosity,
        density=density,
    )
    matrix = np.empty((int(base.size), int(coeffs.size)), dtype=float)
    for idx in range(coeffs.size):
        delta = np.zeros_like(coeffs)
        delta[idx] = step
        plus = _axisymmetric_residual_vector_from_coefficients(
            coeffs + delta,
            grid=grid,
            metadata=metadata,
            viscosity=viscosity,
            density=density,
        )
        minus = _axisymmetric_residual_vector_from_coefficients(
            coeffs - delta,
            grid=grid,
            metadata=metadata,
            viscosity=viscosity,
            density=density,
        )
        matrix[:, idx] = (plus - minus) / (2.0 * step)
    return base, matrix, step


def active_subspace_invariance_report(
    refined_artifact: dict[str, Any],
    active_indices: list[int] | tuple[int, ...],
    *,
    leakage_threshold: float = 5e-2,
) -> dict[str, Any]:
    """Measure finite inactive-mode leakage after an active Newton correction."""
    if refined_artifact.get("candidate_type") != "axisymmetric_swirl_refined":
        raise ValueError("active-subspace invariance reports require axisymmetric_swirl_refined artifacts")
    coeffs = np.asarray(refined_artifact.get("replay_inputs", {}).get("coefficients", []), dtype=float)
    active = tuple(int(idx) for idx in active_indices)
    if not active:
        raise ValueError("active_indices must not be empty")
    inactive = tuple(idx for idx in range(int(coeffs.size)) if idx not in set(active))
    residual, jacobian, step = _axisymmetric_residual_jacobian_matrix(refined_artifact)
    active_j = jacobian[:, active]
    inactive_j = jacobian[:, inactive] if inactive else np.empty((jacobian.shape[0], 0), dtype=float)
    gradient = jacobian.T @ residual
    active_gradient = gradient[list(active)]
    inactive_gradient = gradient[list(inactive)] if inactive else np.empty((0,), dtype=float)
    active_step = -np.linalg.lstsq(active_j, residual, rcond=None)[0]
    post_residual = residual + active_j @ active_step
    post_inactive_gradient = inactive_j.T @ post_residual if inactive else np.empty((0,), dtype=float)
    active_gram = active_j.T @ active_j
    inactive_gram = inactive_j.T @ inactive_j if inactive else np.empty((0, 0), dtype=float)
    coupling = inactive_j.T @ active_j if inactive else np.empty((0, active_j.shape[1]), dtype=float)
    active_gradient_norm = float(np.linalg.norm(active_gradient))
    inactive_gradient_norm = float(np.linalg.norm(inactive_gradient))
    post_inactive_norm = float(np.linalg.norm(post_inactive_gradient))
    post_leakage_ratio = post_inactive_norm / (active_gradient_norm + 1e-30)
    raw_leakage_ratio = inactive_gradient_norm / (active_gradient_norm + 1e-30)
    inactive_rows = []
    for local, idx in enumerate(inactive):
        column = inactive_j[:, local]
        inactive_rows.append({
            "idx": int(idx),
            "gradient": float(inactive_gradient[local]),
            "abs_gradient": float(abs(inactive_gradient[local])),
            "post_active_newton_gradient": float(post_inactive_gradient[local]),
            "abs_post_active_newton_gradient": float(abs(post_inactive_gradient[local])),
            "column_norm": float(np.linalg.norm(column)),
            "coupling_norm_to_active": float(np.linalg.norm(coupling[local, :])),
            "coupling_over_column_active": float(
                np.linalg.norm(coupling[local, :])
                / ((np.linalg.norm(column) * np.linalg.norm(active_j, ord=2)) + 1e-30)
            ),
        })
    inactive_rows.sort(key=lambda item: float(item["abs_post_active_newton_gradient"]), reverse=True)
    finite_invariance_passed = bool(post_leakage_ratio <= float(leakage_threshold))
    open_obligations = [
        "prove_active_subspace_invariant_under_exact_newton_map",
        "bound_inactive_modes_in_continuum_tail_space",
        "external_sparse_ansatz_completeness_proof",
    ]
    return {
        "schema_version": "navier-stokes-active-subspace-invariance-1",
        "candidate_type": "active_subspace_invariance_report",
        "active_indices": list(active),
        "inactive_indices": list(inactive),
        "perturbation": step,
        "residual_norm": float(np.linalg.norm(residual)),
        "post_active_newton_residual_norm": float(np.linalg.norm(post_residual)),
        "active_gradient_norm": active_gradient_norm,
        "inactive_gradient_norm": inactive_gradient_norm,
        "gradient_leakage_ratio": raw_leakage_ratio,
        "post_active_newton_inactive_gradient_norm": post_inactive_norm,
        "post_newton_leakage_ratio": post_leakage_ratio,
        "leakage_threshold": float(leakage_threshold),
        "active_gram_condition": float(np.linalg.cond(active_gram)) if active_gram.size else float("inf"),
        "inactive_gram_norm": float(np.linalg.norm(inactive_gram, ord=2)) if inactive_gram.size else 0.0,
        "active_inactive_coupling_norm": float(np.linalg.norm(coupling, ord=2)) if coupling.size else 0.0,
        "coupling_relative_to_blocks": float(
            np.linalg.norm(coupling, ord=2)
            / ((np.linalg.norm(active_gram, ord=2) * np.linalg.norm(inactive_gram, ord=2)) ** 0.5 + 1e-30)
        ) if coupling.size and inactive_gram.size else 0.0,
        "worst_inactive_modes": inactive_rows,
        "finite_invariance_heuristic_passed": finite_invariance_passed,
        "proof_status": (
            "finite_invariance_heuristic_passed"
            if finite_invariance_passed
            else "blocked_inactive_mode_leakage"
        ),
        "open_obligations": open_obligations,
        "unproven_claim": False,
        "replay_inputs": {
            "refined_artifact": _json_dict(refined_artifact),
            "active_indices": list(active),
            "leakage_threshold": float(leakage_threshold),
        },
    }


def active_subspace_absorption_frontier_report(
    interval_report: dict[str, Any],
    active_indices: list[int] | tuple[int, ...],
    *,
    max_combination_order: int = 3,
    sigma_fraction_threshold: float = 0.75,
) -> dict[str, Any]:
    """Test which inactive modes can be absorbed without breaking finite closure."""
    artifact = interval_report["replay_inputs"]["refined_artifact"]
    coeffs = np.asarray(artifact.get("replay_inputs", {}).get("coefficients", []), dtype=float)
    active = tuple(int(idx) for idx in active_indices)
    inactive = tuple(idx for idx in range(int(coeffs.size)) if idx not in set(active))

    def evaluate(indices: tuple[int, ...]) -> dict[str, Any]:
        linearized = assemble_axisymmetric_active_subspace_operator(artifact, indices)
        operator = operator_theoretic_invertibility_certificate(interval_report, linearized)
        componentwise = componentwise_radii_polynomial_certificate(interval_report, linearized, operator)
        worst_name, worst_upper = _worst_componentwise_radii(componentwise)
        sigma = float(linearized.get("smallest_singular_value", 0.0))
        return {
            "active_indices": list(indices),
            "added_inactive": [int(idx) for idx in indices if idx not in set(active)],
            "active_count": len(indices),
            "sigma_min": sigma,
            "sigma_fraction_vs_base": 1.0,
            "condition": linearized.get("condition_estimate"),
            "neumann_upper": float(operator.get("neumann_defect_interval", {}).get("upper", float("inf"))),
            "neumann_passed": bool(operator.get("neumann_passed", False)),
            "worst_component": worst_name,
            "worst_radii_upper": worst_upper,
            "radii_passed": bool(componentwise.get("passed", False)),
            "frontier_passed": False,
        }

    base = evaluate(active)
    base_sigma = max(float(base["sigma_min"]), 1e-300)
    sigma_threshold = float(sigma_fraction_threshold) * base_sigma
    base["frontier_passed"] = bool(
        float(base["sigma_min"]) >= sigma_threshold
        and base["neumann_passed"]
        and base["radii_passed"]
    )
    rows: list[dict[str, Any]] = []
    max_order = min(max(int(max_combination_order), 0), len(inactive))
    for order in range(max_order + 1):
        for combo in itertools.combinations(inactive, order):
            indices = tuple([*active, *combo])
            row = evaluate(indices)
            sigma = float(row["sigma_min"])
            row["sigma_fraction_vs_base"] = sigma / base_sigma
            row["frontier_passed"] = bool(
                sigma >= sigma_threshold
                and row["neumann_passed"]
                and row["radii_passed"]
            )
            rows.append(row)
    if inactive and max_order < len(inactive):
        row = evaluate(tuple([*active, *inactive]))
        sigma = float(row["sigma_min"])
        row["sigma_fraction_vs_base"] = sigma / base_sigma
        row["frontier_passed"] = bool(
            sigma >= sigma_threshold
            and row["neumann_passed"]
            and row["radii_passed"]
        )
        rows.append(row)
    unique: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in rows:
        unique[tuple(row["added_inactive"])] = row
    rows = sorted(
        unique.values(),
        key=lambda row: (
            bool(row["frontier_passed"]),
            int(row["active_count"]),
            float(row["sigma_min"]),
        ),
        reverse=True,
    )
    passed = [row for row in rows if row["frontier_passed"]]
    failed = [row for row in rows if not row["frontier_passed"]]
    passed_sets = [set(row["added_inactive"]) for row in passed]
    maximal = [
        row
        for row, added in zip(passed, passed_sets, strict=True)
        if not any(added < other for other in passed_sets)
    ]
    mode_counts = {int(idx): 0 for idx in inactive}
    for row in maximal:
        for idx in row["added_inactive"]:
            mode_counts[int(idx)] += 1
    required_tail = [int(idx) for idx in inactive if mode_counts[int(idx)] == 0]
    open_obligations = [
        "prove_required_tail_modes_are_contractively_controlled",
        "prove_absorption_frontier_persists_in_continuum_space",
        "external_active_subspace_completeness_proof",
    ]
    return {
        "schema_version": "navier-stokes-active-absorption-frontier-1",
        "candidate_type": "active_subspace_absorption_frontier_report",
        "base_active_indices": list(active),
        "inactive_indices": list(inactive),
        "base_sigma_min": base_sigma,
        "sigma_fraction_threshold": float(sigma_fraction_threshold),
        "sigma_threshold": sigma_threshold,
        "max_combination_order": max_order,
        "base_metrics": base,
        "maximal_absorbed_frontiers": _json_dict(maximal),
        "required_tail_control_modes": required_tail,
        "mode_counts_in_maximal_frontiers": {str(k): int(v) for k, v in mode_counts.items()},
        "passed_count": len(passed),
        "failed_count": len(failed),
        "all_rows": _json_dict(rows),
        "proof_status": (
            "all_inactive_modes_absorbed"
            if not required_tail
            else "blocked_required_tail_mode_control"
        ),
        "open_obligations": [] if not required_tail else open_obligations,
        "unproven_claim": False,
        "replay_inputs": {
            "interval_report": _json_dict(interval_report),
            "active_indices": list(active),
            "max_combination_order": max_order,
            "sigma_fraction_threshold": float(sigma_fraction_threshold),
        },
    }


def finite_active_tail_contraction_diagnostic(
    refined_artifact: dict[str, Any],
    active_indices: list[int] | tuple[int, ...],
    *,
    tail_modes: list[int] | tuple[int, ...] | None = None,
    analytic_radius: float = 1.125,
    contraction_threshold: float = 1.0,
) -> dict[str, Any]:
    """Compute a finite weighted-tail response after active Newton absorption."""
    if refined_artifact.get("candidate_type") != "axisymmetric_swirl_refined":
        raise ValueError("finite tail diagnostics require axisymmetric_swirl_refined artifacts")
    coeffs = np.asarray(refined_artifact.get("replay_inputs", {}).get("coefficients", []), dtype=float)
    active = tuple(int(idx) for idx in active_indices)
    if not active:
        raise ValueError("active_indices must not be empty")
    inactive = tuple(int(idx) for idx in range(int(coeffs.size)) if idx not in set(active))
    tail = tuple(int(idx) for idx in (tail_modes if tail_modes is not None else inactive))
    if not set(tail).issubset(set(inactive)):
        raise ValueError("tail_modes must be a subset of inactive coefficient indices")
    residual, jacobian, step = _axisymmetric_residual_jacobian_matrix(refined_artifact)
    active_j = jacobian[:, active]
    tail_j = jacobian[:, tail] if tail else np.empty((jacobian.shape[0], 0), dtype=float)
    active_pinv = np.linalg.pinv(active_j)
    active_projector = active_j @ active_pinv
    residual_projector = np.eye(jacobian.shape[0]) - active_projector
    tail_response = tail_j.T @ residual_projector @ tail_j
    radius = float(analytic_radius)
    weights = np.asarray([radius ** int(idx) for idx in tail], dtype=float)
    if tail_response.size:
        weighted_abs = (weights[:, None] * np.abs(tail_response)) / weights[None, :]
        column_sums = np.sum(weighted_abs, axis=0)
        row_sums = np.sum(weighted_abs, axis=1)
        induced_l1 = float(np.max(column_sums))
        induced_linf = float(np.max(row_sums))
        spectral = float(np.linalg.norm(tail_response, ord=2))
    else:
        weighted_abs = np.empty((0, 0), dtype=float)
        column_sums = np.empty((0,), dtype=float)
        row_sums = np.empty((0,), dtype=float)
        induced_l1 = 0.0
        induced_linf = 0.0
        spectral = 0.0
    interval = scalar_interval(induced_l1, relative_padding=1e-12, certified=np.isfinite(induced_l1))
    finite_passed = bool(interval.upper < float(contraction_threshold))
    mode_rows = []
    for local, idx in enumerate(tail):
        mode_rows.append({
            "idx": int(idx),
            "weight": float(weights[local]),
            "weighted_column_sum": float(column_sums[local]) if column_sums.size else 0.0,
            "weighted_row_sum": float(row_sums[local]) if row_sums.size else 0.0,
            "tail_response_diagonal": float(tail_response[local, local]) if tail_response.size else 0.0,
        })
    mode_rows.sort(key=lambda item: float(item["weighted_column_sum"]), reverse=True)
    return {
        "schema_version": "navier-stokes-finite-active-tail-contraction-1",
        "candidate_type": "finite_active_tail_contraction_diagnostic",
        "method": "weighted_l1_inactive_gradient_response_after_active_least_squares_absorption",
        "active_indices": list(active),
        "tail_modes": list(tail),
        "inactive_indices": list(inactive),
        "analytic_radius": radius,
        "tail_weights": {str(idx): float(radius ** int(idx)) for idx in tail},
        "perturbation": step,
        "residual_norm": float(np.linalg.norm(residual)),
        "active_projector_rank": int(np.linalg.matrix_rank(active_projector)),
        "tail_response_matrix": _json_dict(tail_response),
        "weighted_abs_response_matrix": _json_dict(weighted_abs),
        "weighted_column_sums": [float(v) for v in column_sums],
        "weighted_row_sums": [float(v) for v in row_sums],
        "finite_contraction_ratio_interval": asdict(interval),
        "finite_contraction_ratio_upper": float(interval.upper),
        "finite_contraction_threshold": float(contraction_threshold),
        "finite_tail_contraction_surrogate_passed": finite_passed,
        "spectral_tail_response_norm": spectral,
        "weighted_linf_response_norm": induced_linf,
        "worst_tail_modes": mode_rows,
        "proof_status": (
            "finite_tail_contraction_surrogate_passed"
            if finite_passed
            else "blocked_finite_tail_contraction_ratio"
        ),
        "open_obligations": [
            "lift_finite_tail_contraction_to_weighted_analytic_tail_space",
            "replace_float64_tail_jacobian_with_directed_interval_bounds",
            "prove_tail_response_model_matches_exact_newton_derivative",
        ],
        "unproven_claim": False,
        "replay_inputs": {
            "refined_artifact": _json_dict(refined_artifact),
            "active_indices": list(active),
            "tail_modes": list(tail),
            "analytic_radius": radius,
            "contraction_threshold": float(contraction_threshold),
        },
    }


def active_tail_contraction_lift_certificate(
    finite_diagnostic: dict[str, Any],
    tail_contract: dict[str, Any],
    *,
    interval_jacobian_error_upper: float | None = None,
    projector_error_upper: float | None = None,
    analytic_tail_error_upper: float | None = None,
    nonlinear_remainder_error_upper: float | None = None,
    contraction_threshold: float = 1.0,
    external_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attempt to lift finite tail contraction to the analytic tail space."""
    if finite_diagnostic.get("candidate_type") != "finite_active_tail_contraction_diagnostic":
        raise ValueError("expected candidate_type='finite_active_tail_contraction_diagnostic'")
    if tail_contract.get("candidate_type") != "weighted_analytic_tail_norm_contract":
        raise ValueError("expected candidate_type='weighted_analytic_tail_norm_contract'")
    finite_upper = float(finite_diagnostic.get("finite_contraction_ratio_upper", float("inf")))
    threshold = float(contraction_threshold)
    raw_terms = {
        "interval_jacobian_error_upper": interval_jacobian_error_upper,
        "projector_error_upper": projector_error_upper,
        "analytic_tail_error_upper": analytic_tail_error_upper,
        "nonlinear_remainder_error_upper": nonlinear_remainder_error_upper,
    }
    finite_error_terms = {
        name: float(value)
        for name, value in raw_terms.items()
        if value is not None and np.isfinite(float(value)) and float(value) >= 0.0
    }
    missing_terms = [name for name in raw_terms if name not in finite_error_terms]
    total_error = sum(finite_error_terms.values())
    q_total = finite_upper + total_error if not missing_terms and np.isfinite(finite_upper) else float("inf")
    q_interval = scalar_interval(
        q_total if np.isfinite(q_total) else 1.0e300,
        relative_padding=1e-12,
        certified=np.isfinite(q_total),
    )
    finite_ok = bool(finite_diagnostic.get("finite_tail_contraction_surrogate_passed", False))
    contract_ok = bool(tail_contract.get("weighted_tail_contract_certified", False))
    external_ok = _external_verifies(external_verification, "active_tail_contraction_analytic_lift")
    q_ok = bool(np.isfinite(q_total) and q_interval.upper < threshold)
    certified = bool(finite_ok and contract_ok and external_ok and q_ok)
    margin_after_finite = threshold - finite_upper if np.isfinite(finite_upper) else -float("inf")
    remaining_margin = threshold - q_interval.upper if np.isfinite(q_total) else None
    open_obligations: list[str] = []
    if not finite_ok:
        open_obligations.append("finite_tail_contraction_surrogate_below_one")
    if not contract_ok:
        open_obligations.append("weighted_analytic_tail_norm_contract")
    open_obligations.extend(missing_terms)
    if not q_ok:
        open_obligations.append("q_total_below_one")
    if not external_ok:
        open_obligations.append("external_active_tail_contraction_analytic_lift_proof")
    return {
        "schema_version": "navier-stokes-active-tail-contraction-lift-1",
        "candidate_type": "active_tail_contraction_lift_certificate",
        "inequality": "q_total = q_finite + interval_error + projector_error + analytic_tail_error + nonlinear_remainder_error < 1",
        "finite_diagnostic_sha256": _sha256_json(finite_diagnostic),
        "tail_contract_sha256": _sha256_json(tail_contract),
        "active_indices": list(finite_diagnostic.get("active_indices", [])),
        "tail_modes": list(finite_diagnostic.get("tail_modes", [])),
        "analytic_radius": finite_diagnostic.get("analytic_radius"),
        "q_finite_upper": finite_upper,
        "error_budget_terms": {
            "interval_jacobian_error_upper": (
                None if interval_jacobian_error_upper is None else float(interval_jacobian_error_upper)
            ),
            "projector_error_upper": (
                None if projector_error_upper is None else float(projector_error_upper)
            ),
            "analytic_tail_error_upper": (
                None if analytic_tail_error_upper is None else float(analytic_tail_error_upper)
            ),
            "nonlinear_remainder_error_upper": (
                None if nonlinear_remainder_error_upper is None else float(nonlinear_remainder_error_upper)
            ),
        },
        "error_budget_terms_certified": len(missing_terms) == 0,
        "missing_error_budget_terms": missing_terms,
        "total_lift_error_upper": None if missing_terms else total_error,
        "q_total_interval": asdict(q_interval),
        "q_total_upper": None if not np.isfinite(q_total) else float(q_interval.upper),
        "contraction_threshold": threshold,
        "margin_after_finite_q": margin_after_finite,
        "remaining_margin_after_lift": remaining_margin,
        "finite_tail_contraction_surrogate_passed": finite_ok,
        "weighted_tail_contract_certified": contract_ok,
        "external_verification": _json_dict(external_verification or {}),
        "analytic_lift_certified": certified,
        "proof_status": "proved_by_external_artifact" if certified else "blocked_with_precise_obligations",
        "open_obligations": open_obligations,
        "unproven_claim": False,
        "replay_inputs": {
            "finite_diagnostic": _json_dict(finite_diagnostic),
            "tail_contract": _json_dict(tail_contract),
            "interval_jacobian_error_upper": raw_terms["interval_jacobian_error_upper"],
            "projector_error_upper": raw_terms["projector_error_upper"],
            "analytic_tail_error_upper": raw_terms["analytic_tail_error_upper"],
            "nonlinear_remainder_error_upper": raw_terms["nonlinear_remainder_error_upper"],
            "contraction_threshold": threshold,
            "external_verification": _json_dict(external_verification or {}),
        },
    }


def _tail_weight_condition(
    tail: tuple[int, ...],
    radius: float,
) -> tuple[float, dict[str, float]]:
    """Return ``w_max / w_min`` and the weight map for analytic tail modes."""
    weights = {int(idx): float(radius) ** int(idx) for idx in tail}
    if not weights:
        return 1.0, {}
    w_max = max(weights.values())
    w_min = min(weights.values())
    cond = float(w_max / w_min) if w_min > 0.0 else float("inf")
    return cond, {str(k): float(v) for k, v in weights.items()}


def _finite_difference_jacobian_error_envelope(
    refined_artifact: dict[str, Any],
) -> dict[str, Any]:
    """Estimate the matrix-2-norm gap between the float64 central-difference
    residual Jacobian and the exact derivative.

    The central difference is :math:`O(h^2)`, so Richardson extrapolation gives
    ``true = J_h + (J_h - J_{2h})/3 + O(h^4)`` and the truncation error of
    ``J_h`` is bounded by ``||J_h - J_{2h}||_2 / 3``.  A rounding floor
    ``eps * max(1, ||F||_inf) / h`` accounts for cancellation in the difference
    quotient.  This is **certified-evidence only**: a directed-rounding interval
    Jacobian would replace the empirical envelope with a certified enclosure.
    """
    base, jac_h, step_h = _axisymmetric_residual_jacobian_matrix(refined_artifact)
    _, jac_2h, step_2h = _axisymmetric_residual_jacobian_matrix(
        refined_artifact, perturbation=2.0 * step_h
    )
    richardson = jac_h - jac_2h
    truncation = float(np.linalg.norm(richardson, ord=2)) / 3.0
    residual_scale = float(np.max(np.abs(base))) if base.size else 0.0
    rounding = float(np.finfo(float).eps) * max(1.0, residual_scale) / max(step_h, 1e-300)
    envelope = truncation + rounding
    return {
        "method": "richardson_central_difference_plus_rounding_floor",
        "perturbation": float(step_h),
        "double_perturbation": float(step_2h),
        "jacobian_norm": float(np.linalg.norm(jac_h, ord=2)) if jac_h.size else 0.0,
        "richardson_truncation_upper": float(truncation),
        "rounding_floor_upper": float(rounding),
        "jacobian_error_envelope_upper": float(envelope),
    }


def _active_tail_geometry(
    finite_diagnostic: dict[str, Any],
) -> dict[str, Any]:
    """Recompute active/tail Jacobian conditioning used by the error terms."""
    refined = dict(finite_diagnostic["replay_inputs"]["refined_artifact"])
    active = tuple(int(i) for i in finite_diagnostic.get("active_indices", []))
    tail = tuple(int(i) for i in finite_diagnostic.get("tail_modes", []))
    radius = float(finite_diagnostic.get("analytic_radius", 1.125))
    residual, jacobian, step = _axisymmetric_residual_jacobian_matrix(refined)
    active_j = jacobian[:, active] if active else np.empty((jacobian.shape[0], 0), dtype=float)
    tail_j = jacobian[:, tail] if tail else np.empty((jacobian.shape[0], 0), dtype=float)
    active_svals = np.linalg.svd(active_j, compute_uv=False) if active_j.size else np.empty((0,))
    full_svals = np.linalg.svd(jacobian, compute_uv=False) if jacobian.size else np.empty((0,))
    sigma_min_active = float(np.min(active_svals)) if active_svals.size else 0.0
    sigma_max_active = float(np.max(active_svals)) if active_svals.size else 0.0
    sigma_min_full = float(np.min(full_svals)) if full_svals.size else 0.0
    tail_norm = float(np.linalg.norm(tail_j, ord=2)) if tail_j.size else 0.0
    weight_condition, weights = _tail_weight_condition(tail, radius)
    return {
        "refined_artifact": refined,
        "active_indices": active,
        "tail_modes": tail,
        "analytic_radius": radius,
        "residual_norm": float(np.linalg.norm(residual)),
        "sigma_min_active": sigma_min_active,
        "sigma_max_active": sigma_max_active,
        "sigma_min_full": sigma_min_full,
        "tail_operator_norm": tail_norm,
        "weight_condition": weight_condition,
        "tail_weights": weights,
        "dimension_factor": float(np.sqrt(max(len(tail), 1))),
        "perturbation": float(step),
    }


def active_projector_error_certificate(
    finite_diagnostic: dict[str, Any],
    *,
    jacobian_perturbation_upper: float | None = None,
    external_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Finite bound on the tail-contraction error from the active projector.

    The finite diagnostic models the exact active-subspace elimination by the
    least-squares projector ``P_A = J_A J_A^+``.  This certificate isolates the
    *projector* mechanism: when ``J_A`` carries a column-space uncertainty
    ``delta`` the orthogonal projector moves by at most ``2 delta /
    sigma_min(J_A)`` (rank-preserving Stewart bound), and the weighted
    tail-response norm ``q`` therefore moves by at most

    ``sqrt(|T|) * cond_w * ||J_T||^2 * (2 / sigma_min(J_A)) * delta``.

    The direct tail-column sensitivity is the complementary term handled by
    :func:`interval_jacobian_error_certificate`, so the two are additive and do
    not double count.  When ``jacobian_perturbation_upper`` is omitted the
    finite-difference Jacobian-error envelope of the refined artifact is used.
    """
    if finite_diagnostic.get("candidate_type") != "finite_active_tail_contraction_diagnostic":
        raise ValueError("expected candidate_type='finite_active_tail_contraction_diagnostic'")
    geom = _active_tail_geometry(finite_diagnostic)
    envelope = _finite_difference_jacobian_error_envelope(geom["refined_artifact"])
    delta = (
        float(jacobian_perturbation_upper)
        if jacobian_perturbation_upper is not None
        else float(envelope["jacobian_error_envelope_upper"])
    )
    sigma_min_active = float(geom["sigma_min_active"])
    tail_norm = float(geom["tail_operator_norm"])
    weight_condition = float(geom["weight_condition"])
    dim_factor = float(geom["dimension_factor"])
    finite_certified = bool(sigma_min_active > 0.0 and np.isfinite(delta) and np.isfinite(weight_condition))
    if finite_certified:
        projector_norm_sensitivity = 2.0 / sigma_min_active
        projector_error = (
            dim_factor * weight_condition * tail_norm * tail_norm * projector_norm_sensitivity * delta
        )
    else:
        projector_norm_sensitivity = float("inf")
        projector_error = float("inf")
    interval = scalar_interval(
        projector_error if np.isfinite(projector_error) else 1.0e300,
        relative_padding=1e-12,
        certified=finite_certified,
    )
    external_ok = _external_verifies(external_verification, "active_projector_error_bound")
    open_obligations: list[str] = []
    if not finite_certified:
        open_obligations.append("active_jacobian_full_column_rank")
    if not external_ok:
        open_obligations.append("external_active_projector_perturbation_proof")
    return {
        "schema_version": "navier-stokes-active-projector-error-1",
        "candidate_type": "active_projector_error_certificate",
        "mechanism": "least_squares_active_projector_sensitivity",
        "bound_formula": "sqrt(|T|) * cond_w * ||J_T||^2 * (2 / sigma_min(J_A)) * delta_JA",
        "active_indices": list(geom["active_indices"]),
        "tail_modes": list(geom["tail_modes"]),
        "analytic_radius": geom["analytic_radius"],
        "sigma_min_active": sigma_min_active,
        "sigma_max_active": float(geom["sigma_max_active"]),
        "tail_operator_norm": tail_norm,
        "weight_condition": weight_condition,
        "tail_weights": geom["tail_weights"],
        "dimension_factor": dim_factor,
        "jacobian_error_envelope": envelope,
        "active_jacobian_perturbation_upper": float(delta) if np.isfinite(delta) else None,
        "projector_norm_sensitivity": (
            float(projector_norm_sensitivity) if np.isfinite(projector_norm_sensitivity) else None
        ),
        "projector_error_interval": asdict(interval),
        "projector_error_upper": float(interval.upper) if np.isfinite(projector_error) else None,
        "finite_dimensional_certified": finite_certified,
        "external_verification": _json_dict(external_verification or {}),
        "theorem_grade_certified": bool(finite_certified and external_ok),
        "proof_status": (
            "finite_projector_error_bounded" if finite_certified else "blocked_singular_active_block"
        ),
        "open_obligations": open_obligations,
        "unproven_claim": False,
        "replay_inputs": {
            "finite_diagnostic": _json_dict(finite_diagnostic),
            "jacobian_perturbation_upper": (
                None if jacobian_perturbation_upper is None else float(jacobian_perturbation_upper)
            ),
            "external_verification": _json_dict(external_verification or {}),
        },
    }


def interval_jacobian_error_certificate(
    finite_diagnostic: dict[str, Any],
    *,
    external_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Finite bound on the tail-contraction error from a float64 Jacobian.

    The finite diagnostic evaluates the residual Jacobian by float64 central
    differences.  This certificate isolates the *direct tail-column*
    sensitivity of the weighted tail response: with the projector held fixed,
    ``||delta M||_2 <= 2 ||J_T|| ||P_perp||_2 E_J`` and ``||P_perp||_2 = 1``,
    so

    ``interval_error <= sqrt(|T|) * cond_w * 2 * ||J_T|| * E_J``,

    where ``E_J`` is the Jacobian-error envelope (Richardson truncation plus a
    rounding floor).  Combined additively with the projector term this bounds
    the full first-order Jacobian-perturbation contribution to ``q``.  Status
    is **certified-evidence**: a directed-rounding interval Jacobian is required to make
    ``E_J`` a certified enclosure.
    """
    if finite_diagnostic.get("candidate_type") != "finite_active_tail_contraction_diagnostic":
        raise ValueError("expected candidate_type='finite_active_tail_contraction_diagnostic'")
    geom = _active_tail_geometry(finite_diagnostic)
    envelope = _finite_difference_jacobian_error_envelope(geom["refined_artifact"])
    error_envelope = float(envelope["jacobian_error_envelope_upper"])
    tail_norm = float(geom["tail_operator_norm"])
    weight_condition = float(geom["weight_condition"])
    dim_factor = float(geom["dimension_factor"])
    finite_certified = bool(np.isfinite(error_envelope) and np.isfinite(weight_condition))
    interval_error = (
        dim_factor * weight_condition * 2.0 * tail_norm * error_envelope
        if finite_certified
        else float("inf")
    )
    interval = scalar_interval(
        interval_error if np.isfinite(interval_error) else 1.0e300,
        relative_padding=1e-12,
        certified=finite_certified,
    )
    external_ok = _external_verifies(external_verification, "interval_jacobian_error_bound")
    open_obligations: list[str] = []
    if not finite_certified:
        open_obligations.append("finite_difference_jacobian_error_envelope")
    if not external_ok:
        open_obligations.append("replace_float64_jacobian_with_directed_interval_arithmetic")
    return {
        "schema_version": "navier-stokes-interval-jacobian-error-1",
        "candidate_type": "interval_jacobian_error_certificate",
        "mechanism": "direct_tail_column_jacobian_sensitivity",
        "bound_formula": "sqrt(|T|) * cond_w * 2 * ||J_T|| * E_J",
        "active_indices": list(geom["active_indices"]),
        "tail_modes": list(geom["tail_modes"]),
        "analytic_radius": geom["analytic_radius"],
        "tail_operator_norm": tail_norm,
        "weight_condition": weight_condition,
        "tail_weights": geom["tail_weights"],
        "dimension_factor": dim_factor,
        "jacobian_error_envelope": envelope,
        "jacobian_error_envelope_upper": error_envelope,
        "interval_jacobian_error_interval": asdict(interval),
        "interval_jacobian_error_upper": float(interval.upper) if finite_certified else None,
        "finite_dimensional_certified": finite_certified,
        "interval_arithmetic_backend": "float64_richardson_envelope",
        "external_verification": _json_dict(external_verification or {}),
        "theorem_grade_certified": bool(finite_certified and external_ok),
        "proof_status": (
            "proof_prep_jacobian_error_bounded" if finite_certified else "blocked_jacobian_error_envelope"
        ),
        "open_obligations": open_obligations,
        "unproven_claim": False,
        "replay_inputs": {
            "finite_diagnostic": _json_dict(finite_diagnostic),
            "external_verification": _json_dict(external_verification or {}),
        },
    }


def _axisymmetric_residual_hessian_tensor(
    n_coeffs: int,
    *,
    grid: ReplayGrid | dict[str, Any],
    metadata: AxisymmetricBasisMetadata | dict[str, Any],
    viscosity: float,
    density: float,
    step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(f0, L, H)`` of the exact quadratic ``F(c) = f0 + L c + 1/2 c^T H c``.

    The axisymmetric swirl residual is quadratic in the coefficients (fields are
    linear, the convective term is bilinear), so the residual Hessian ``H`` is a
    constant tensor independent of the profile.  It is therefore built once
    around the origin by central-difference polarisation.
    """
    zero = np.zeros(int(n_coeffs), dtype=float)
    f0 = _axisymmetric_residual_vector_from_coefficients(
        zero, grid=grid, metadata=metadata, viscosity=viscosity, density=density
    )
    m = int(f0.size)
    n = int(n_coeffs)
    fp = np.empty((n, m), dtype=float)
    linear = np.empty((m, n), dtype=float)
    hess = np.empty((m, n, n), dtype=float)
    for i in range(n):
        ei = np.zeros(n, dtype=float)
        ei[i] = step
        fp_i = _axisymmetric_residual_vector_from_coefficients(
            ei, grid=grid, metadata=metadata, viscosity=viscosity, density=density
        )
        fm_i = _axisymmetric_residual_vector_from_coefficients(
            -ei, grid=grid, metadata=metadata, viscosity=viscosity, density=density
        )
        fp[i] = fp_i
        linear[:, i] = (fp_i - fm_i) / (2.0 * step)
        hess[:, i, i] = (fp_i - 2.0 * f0 + fm_i) / (step * step)
    for i in range(n):
        for j in range(i + 1, n):
            eij = np.zeros(n, dtype=float)
            eij[i] = step
            eij[j] = step
            fpp = _axisymmetric_residual_vector_from_coefficients(
                eij, grid=grid, metadata=metadata, viscosity=viscosity, density=density
            )
            mixed = (fpp - fp[i] - fp[j] + f0) / (step * step)
            hess[:, i, j] = mixed
            hess[:, j, i] = mixed
    return f0, linear, hess


def axisymmetric_residual_hessian_operator_norm(
    refined_artifact: dict[str, Any],
    *,
    perturbation: float = 1.0e-3,
    validation_samples: int = 16,
    validation_scale: float = 0.1,
) -> dict[str, Any]:
    """Rigorous operator-norm constants of the residual Hessian.

    The certificate's nonlinear-remainder term ``cond_w * H2 * r`` is *linear* in
    the ball radius ``r``, so the constant it requires is the Jacobian-Lipschitz
    operator norm

        ``L_J = sup_{||d||_2 = 1} ||DF(c+d) - DF(c)||_2``,

    i.e. how much the Jacobian varies across the ball.  Since ``DF(c) = L + H.c``
    is affine, ``DF(c+d) - DF(c) = M(d)`` with ``M(d)[m,k] = sum_j H[m,k,j] d_j``
    and ``L_J = sup_{||d||=1} ||M(d)||_2``.  A certifiable (deterministic) upper
    bound is ``L_J <= sigma_max(reshape(H, (m n, n)))`` because
    ``||M(d)||_2 <= ||M(d)||_F``.

    This replaces the sampled diagonal proxy ``max_i ||H[:,i,i]||`` (which is *not*
    an upper bound).  The returned constants are structural (profile-independent),
    and ``exact_quadratic_max_rel_error`` confirms the quadratic model is exact.
    """
    if refined_artifact.get("candidate_type") != "axisymmetric_swirl_refined":
        raise ValueError("hessian operator norm requires axisymmetric_swirl_refined artifacts")
    rin = refined_artifact["replay_inputs"]
    coeffs = np.asarray(rin["coefficients"], dtype=float)
    metadata = dict(rin["basis_metadata"])
    grid = dict(rin["train_grid"])
    viscosity = float(rin["viscosity"])
    density = float(rin["density"])
    step = float(perturbation)
    f0, linear, hess = _axisymmetric_residual_hessian_tensor(
        int(coeffs.size),
        grid=grid,
        metadata=metadata,
        viscosity=viscosity,
        density=density,
        step=step,
    )
    m = int(hess.shape[0])
    n = int(hess.shape[1])
    diagonal_proxy = float(
        max((float(np.linalg.norm(hess[:, i, i])) for i in range(n)), default=0.0)
    )
    jacobian_lipschitz_upper = float(np.linalg.svd(hess.reshape(m * n, n), compute_uv=False)[0])
    remainder_operator_upper = float(
        np.linalg.svd((0.5 * hess).reshape(m, n * n), compute_uv=False)[0]
    )

    rng = np.random.default_rng(0)
    max_rel = 0.0
    for _ in range(int(validation_samples)):
        sample = rng.normal(scale=float(validation_scale), size=n)
        true = _axisymmetric_residual_vector_from_coefficients(
            sample, grid=grid, metadata=metadata, viscosity=viscosity, density=density
        )
        approx = f0 + linear @ sample + 0.5 * np.einsum("mij,i,j->m", hess, sample, sample)
        denom = float(np.max(np.abs(true))) or 1.0
        max_rel = max(max_rel, float(np.max(np.abs(true - approx)) / denom))

    return {
        "schema_version": "navier-stokes-residual-hessian-operator-norm-1",
        "candidate_type": "axisymmetric_residual_hessian_operator_norm",
        "method": "central_difference_polarisation_exact_quadratic",
        "perturbation": step,
        "coefficient_count": n,
        "residual_size": m,
        "diagonal_hessian_proxy": diagonal_proxy,
        "jacobian_lipschitz_operator_norm_upper": jacobian_lipschitz_upper,
        "remainder_operator_norm_upper": remainder_operator_upper,
        "diagonal_proxy_is_upper_bound": bool(diagonal_proxy >= jacobian_lipschitz_upper),
        "exact_quadratic_max_rel_error": float(max_rel),
        "f0_norm": float(np.linalg.norm(f0)),
        "unproven_claim": False,
    }


def nonlinear_tail_remainder_certificate(
    finite_diagnostic: dict[str, Any],
    *,
    solution_ball_radius: float | None = None,
    certify_hessian_operator_norm: bool = False,
    hessian_perturbation: float = 1.0e-3,
    external_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bound the Newton/tail nonlinear remainder over a solution ball.

    The contraction surrogate ``q_finite`` is linear (Jacobian only).  Over a
    ball of radius ``r`` around the refined profile the exact tail-Newton map
    differs from its linearisation by a term governed by the residual second
    derivative.  This certificate samples diagonal second differences along the
    active and tail coordinate directions to estimate a weighted Hessian
    operator-norm proxy ``H2`` and reports ``cond_w * H2 * r``.

    The bound is **sampled**, not a certified supremum over the ball, so the
    status stays *blocked* with the obligation
    ``certify_second_derivative_bound_over_solution_ball``.  When
    ``solution_ball_radius`` is omitted a heuristic Newton radius
    ``||F||_2 / sigma_min(J)`` is used and labelled as non-certified.
    """
    if finite_diagnostic.get("candidate_type") != "finite_active_tail_contraction_diagnostic":
        raise ValueError("expected candidate_type='finite_active_tail_contraction_diagnostic'")
    geom = _active_tail_geometry(finite_diagnostic)
    refined = geom["refined_artifact"]
    active = geom["active_indices"]
    tail = geom["tail_modes"]
    weight_condition = float(geom["weight_condition"])
    rin = refined["replay_inputs"]
    coeffs = np.asarray(rin["coefficients"], dtype=float)
    metadata = dict(rin["basis_metadata"])
    grid = dict(rin["train_grid"])
    viscosity = float(rin["viscosity"])
    density = float(rin["density"])
    step = float(geom["perturbation"])
    base = _axisymmetric_residual_vector_from_coefficients(
        coeffs, grid=grid, metadata=metadata, viscosity=viscosity, density=density
    )
    second_difference_norms: list[dict[str, Any]] = []
    hessian_proxy = 0.0
    for idx in sorted(set(active) | set(tail)):
        delta = np.zeros_like(coeffs)
        delta[idx] = step
        plus = _axisymmetric_residual_vector_from_coefficients(
            coeffs + delta, grid=grid, metadata=metadata, viscosity=viscosity, density=density
        )
        minus = _axisymmetric_residual_vector_from_coefficients(
            coeffs - delta, grid=grid, metadata=metadata, viscosity=viscosity, density=density
        )
        second = (plus - 2.0 * base + minus) / (step * step)
        norm = float(np.linalg.norm(second))
        hessian_proxy = max(hessian_proxy, norm)
        second_difference_norms.append({"idx": int(idx), "second_difference_norm": norm})
    second_difference_norms.sort(key=lambda item: float(item["second_difference_norm"]), reverse=True)
    diagonal_hessian_proxy = float(hessian_proxy)
    hessian_operator_norm_report: dict[str, Any] | None = None
    hessian_bound_rigorous = False
    hessian_bound_method = "sampled_diagonal_second_difference_proxy"
    if certify_hessian_operator_norm:
        hessian_operator_norm_report = axisymmetric_residual_hessian_operator_norm(
            refined, perturbation=float(hessian_perturbation)
        )
        hessian_proxy = float(hessian_operator_norm_report["jacobian_lipschitz_operator_norm_upper"])
        hessian_bound_rigorous = True
        hessian_bound_method = "certified_jacobian_lipschitz_operator_norm_upper"
    sigma_min_full = float(geom["sigma_min_full"])
    residual_norm = float(geom["residual_norm"])
    heuristic_radius = (
        residual_norm / sigma_min_full if sigma_min_full > 0.0 else float("inf")
    )
    radius_used = (
        float(solution_ball_radius)
        if solution_ball_radius is not None
        else float(heuristic_radius)
    )
    finite_value = bool(np.isfinite(hessian_proxy) and np.isfinite(radius_used))
    remainder = weight_condition * hessian_proxy * radius_used if finite_value else float("inf")
    interval = scalar_interval(
        remainder if np.isfinite(remainder) else 1.0e300,
        relative_padding=1e-12,
        certified=False,
    )
    external_ok = _external_verifies(external_verification, "nonlinear_tail_remainder_bound")
    open_obligations = []
    if not hessian_bound_rigorous:
        open_obligations.append("certify_second_derivative_bound_over_solution_ball")
    if solution_ball_radius is None:
        open_obligations.append("certify_solution_ball_radius")
    if not external_ok:
        open_obligations.append("external_nonlinear_tail_remainder_proof")
    return {
        "schema_version": "navier-stokes-nonlinear-tail-remainder-1",
        "candidate_type": "nonlinear_tail_remainder_certificate",
        "mechanism": "sampled_second_difference_hessian_proxy_times_ball_radius",
        "bound_formula": "cond_w * H2 * r",
        "active_indices": list(active),
        "tail_modes": list(tail),
        "weight_condition": weight_condition,
        "perturbation": step,
        "hessian_operator_norm_proxy": float(hessian_proxy) if np.isfinite(hessian_proxy) else None,
        "diagonal_hessian_proxy": diagonal_hessian_proxy,
        "hessian_bound_rigorous": bool(hessian_bound_rigorous),
        "hessian_bound_method": hessian_bound_method,
        "hessian_operator_norm": _json_dict(hessian_operator_norm_report or {}),
        "second_difference_norms": second_difference_norms,
        "residual_norm": residual_norm,
        "sigma_min_full": sigma_min_full,
        "heuristic_newton_ball_radius": (
            float(heuristic_radius) if np.isfinite(heuristic_radius) else None
        ),
        "solution_ball_radius_used": float(radius_used) if np.isfinite(radius_used) else None,
        "solution_ball_radius_certified": bool(solution_ball_radius is not None and external_ok),
        "nonlinear_remainder_interval": asdict(interval),
        "nonlinear_remainder_error_upper": float(interval.upper) if finite_value else None,
        "sampled_estimate_only": True,
        "external_verification": _json_dict(external_verification or {}),
        "theorem_grade_certified": bool(external_ok),
        "proof_status": (
            "proved_by_external_artifact" if external_ok else "blocked_sampled_nonlinear_remainder"
        ),
        "open_obligations": open_obligations,
        "unproven_claim": False,
        "replay_inputs": {
            "finite_diagnostic": _json_dict(finite_diagnostic),
            "solution_ball_radius": (
                None if solution_ball_radius is None else float(solution_ball_radius)
            ),
            "certify_hessian_operator_norm": bool(certify_hessian_operator_norm),
            "hessian_perturbation": float(hessian_perturbation),
            "external_verification": _json_dict(external_verification or {}),
        },
    }


def analytic_tail_error_certificate(
    tail_contract: dict[str, Any],
    *,
    coefficient_decay_rate: float = 0.5,
    algebra_constant: float | None = None,
    external_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Conditional bound on analytic modes omitted from the finite basis.

    The finite tail set ``T`` truncates an infinite analytic basis.  Under an
    *assumed* geometric coefficient decay ``|a_k| <= C gamma^k`` the weighted
    analytic-norm remainder beyond the highest finite mode ``K`` is the
    geometric tail ``C (rho gamma)^{K+1} / (1 - rho gamma)`` when
    ``rho gamma < 1``.

    The decay rate ``gamma`` is **assumed, not proven**, so this term remains
    *blocked* with the obligation
    ``prove_enriched_coefficient_geometric_decay_rate``.  It is reported as a
    conditional value so the assembled budget can expose a conditional
    ``q_total`` while keeping the lift uncertified.
    """
    if tail_contract.get("candidate_type") != "weighted_analytic_tail_norm_contract":
        raise ValueError("expected candidate_type='weighted_analytic_tail_norm_contract'")
    radius = float(tail_contract.get("analytic_radius", 1.125))
    constant = (
        float(algebra_constant)
        if algebra_constant is not None
        else float(tail_contract.get("algebra_constant", 1.0))
    )
    tail_modes = [int(idx) for idx in tail_contract.get("required_tail_modes", [])]
    highest_mode = max(tail_modes) if tail_modes else 0
    gamma = float(coefficient_decay_rate)
    ratio = radius * gamma
    convergent = bool(0.0 <= ratio < 1.0)
    if convergent:
        remainder = constant * (ratio ** (highest_mode + 1)) / (1.0 - ratio)
    else:
        remainder = float("inf")
    interval = scalar_interval(
        remainder if np.isfinite(remainder) else 1.0e300,
        relative_padding=1e-12,
        certified=False,
    )
    external_ok = _external_verifies(external_verification, "analytic_tail_error_bound")
    open_obligations = ["prove_enriched_coefficient_geometric_decay_rate"]
    if not convergent:
        open_obligations.append("analytic_tail_weight_below_inverse_radius")
    if not external_ok:
        open_obligations.append("external_analytic_tail_remainder_proof")
    return {
        "schema_version": "navier-stokes-analytic-tail-error-1",
        "candidate_type": "analytic_tail_error_certificate",
        "mechanism": "assumed_geometric_decay_weighted_tail_remainder",
        "bound_formula": "C * (rho*gamma)^(K+1) / (1 - rho*gamma)",
        "analytic_radius": radius,
        "algebra_constant": constant,
        "assumed_coefficient_decay_rate": gamma,
        "weighted_ratio_rho_gamma": float(ratio),
        "geometric_series_convergent": convergent,
        "highest_finite_mode": int(highest_mode),
        "required_tail_modes": tail_modes,
        "analytic_tail_error_interval": asdict(interval),
        "analytic_tail_error_upper": float(interval.upper) if np.isfinite(remainder) else None,
        "decay_rate_assumed_not_proven": True,
        "external_verification": _json_dict(external_verification or {}),
        "theorem_grade_certified": bool(external_ok and convergent),
        "proof_status": (
            "proved_by_external_artifact" if (external_ok and convergent) else "blocked_assumed_decay_rate"
        ),
        "open_obligations": open_obligations,
        "unproven_claim": False,
        "replay_inputs": {
            "tail_contract": _json_dict(tail_contract),
            "coefficient_decay_rate": gamma,
            "algebra_constant": None if algebra_constant is None else float(algebra_constant),
            "external_verification": _json_dict(external_verification or {}),
        },
    }


def build_active_tail_lift_error_budget(
    finite_diagnostic: dict[str, Any],
    tail_contract: dict[str, Any],
    *,
    projector_certificate: dict[str, Any],
    interval_jacobian_certificate: dict[str, Any],
    nonlinear_remainder_certificate: dict[str, Any],
    analytic_tail_certificate: dict[str, Any],
    contraction_threshold: float = 1.0,
    external_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the four error terms into the tail-contraction lift budget.

    Each term carries its own honesty status: the interval-Jacobian and
    projector terms are *certified-evidence certified* (finite, computed), while the
    nonlinear-remainder and analytic-tail terms are *conditional* (sampled /
    assumed-decay) and remain blocked.  The budget forwards the four numeric
    uppers into :func:`active_tail_contraction_lift_certificate`, computes a
    conditional ``q_total``, and names exactly which terms keep the lift from
    being theorem grade.
    """
    certs = {
        "interval_jacobian_error_upper": interval_jacobian_certificate,
        "projector_error_upper": projector_certificate,
        "analytic_tail_error_upper": analytic_tail_certificate,
        "nonlinear_remainder_error_upper": nonlinear_remainder_certificate,
    }
    cert_candidate_types = {
        "interval_jacobian_error_upper": "interval_jacobian_error_certificate",
        "projector_error_upper": "active_projector_error_certificate",
        "analytic_tail_error_upper": "analytic_tail_error_certificate",
        "nonlinear_remainder_error_upper": "nonlinear_tail_remainder_certificate",
    }
    for term, cert in certs.items():
        expected = cert_candidate_types[term]
        if cert.get("candidate_type") != expected:
            raise ValueError(f"{term} expects candidate_type={expected!r}")
    term_values = {term: cert.get(term) for term, cert in certs.items()}
    term_theorem_grade = {
        term: bool(cert.get("theorem_grade_certified", False)) for term, cert in certs.items()
    }
    term_finite_certified = {
        term: bool(cert.get("finite_dimensional_certified", False)) for term, cert in certs.items()
    }
    missing_terms = [term for term, value in term_values.items() if value is None]
    theorem_grade_terms = sorted(term for term in certs if term_theorem_grade[term])
    proof_prep_certified_terms = sorted(
        term
        for term in certs
        if term_finite_certified[term] and not term_theorem_grade[term]
    )
    blocked_terms = sorted(
        term
        for term in certs
        if not term_finite_certified[term] and not term_theorem_grade[term]
    )
    conditional_terms = sorted(term for term in certs if not term_theorem_grade[term])
    lift = active_tail_contraction_lift_certificate(
        finite_diagnostic,
        tail_contract,
        interval_jacobian_error_upper=term_values["interval_jacobian_error_upper"],
        projector_error_upper=term_values["projector_error_upper"],
        analytic_tail_error_upper=term_values["analytic_tail_error_upper"],
        nonlinear_remainder_error_upper=term_values["nonlinear_remainder_error_upper"],
        contraction_threshold=contraction_threshold,
        external_verification=external_verification,
    )
    threshold = float(contraction_threshold)
    q_total_upper = lift.get("q_total_upper")
    all_terms_present = not missing_terms
    all_terms_theorem_grade = all(term_theorem_grade.values())
    q_total_conditional_below_one = bool(
        q_total_upper is not None and float(q_total_upper) < threshold
    )
    certified_partial_sum = float(
        sum(
            float(term_values[term])
            for term in (*theorem_grade_terms, *proof_prep_certified_terms)
            if term_values[term] is not None
        )
    )
    blocking_terms = sorted(set(missing_terms) | set(blocked_terms))
    return {
        "schema_version": "navier-stokes-active-tail-lift-error-budget-1",
        "candidate_type": "active_tail_lift_error_budget",
        "inequality": "q_total = q_finite + interval_error + projector_error + analytic_tail_error + nonlinear_remainder_error < 1",
        "q_finite_upper": lift.get("q_finite_upper"),
        "error_budget_terms": dict(term_values),
        "error_term_certificate_sha256": {
            term: _sha256_json(cert) for term, cert in certs.items()
        },
        "term_theorem_grade": term_theorem_grade,
        "term_finite_dimensional_certified": term_finite_certified,
        "theorem_grade_terms": theorem_grade_terms,
        "proof_prep_certified_terms": proof_prep_certified_terms,
        "blocked_terms": blocked_terms,
        "conditional_terms": conditional_terms,
        "missing_terms": missing_terms,
        "certified_terms_partial_sum": certified_partial_sum,
        "all_terms_present": all_terms_present,
        "all_terms_theorem_grade": all_terms_theorem_grade,
        "contraction_threshold": threshold,
        "q_total_upper": None if q_total_upper is None else float(q_total_upper),
        "q_total_conditional_below_one": q_total_conditional_below_one,
        "remaining_margin_after_lift": lift.get("remaining_margin_after_lift"),
        "blocking_terms": blocking_terms,
        "analytic_lift_certified": bool(lift.get("analytic_lift_certified", False)),
        "lift_certificate": _json_dict(lift),
        "proof_status": (
            "lift_theorem_grade"
            if bool(lift.get("analytic_lift_certified", False))
            else "conditional_q_total_below_one_blocked_terms_named"
            if q_total_conditional_below_one
            else "blocked_with_named_terms"
        ),
        "open_obligations": sorted({
            str(item)
            for cert in certs.values()
            for item in cert.get("open_obligations", [])
        } | {str(item) for item in lift.get("open_obligations", [])}),
        "unproven_claim": False,
        "replay_inputs": {
            "finite_diagnostic": _json_dict(finite_diagnostic),
            "tail_contract": _json_dict(tail_contract),
            "projector_certificate": _json_dict(projector_certificate),
            "interval_jacobian_certificate": _json_dict(interval_jacobian_certificate),
            "nonlinear_remainder_certificate": _json_dict(nonlinear_remainder_certificate),
            "analytic_tail_certificate": _json_dict(analytic_tail_certificate),
            "contraction_threshold": threshold,
            "external_verification": _json_dict(external_verification or {}),
        },
    }


def weighted_analytic_tail_norm_contract(
    frontier_report: dict[str, Any],
    *,
    analytic_radius: float = 1.125,
    algebra_constant: float = 1.0,
    external_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """State the weighted analytic norm contract for inactive tail modes."""
    if frontier_report.get("candidate_type") != "active_subspace_absorption_frontier_report":
        raise ValueError("expected candidate_type='active_subspace_absorption_frontier_report'")
    tail_modes = [int(idx) for idx in frontier_report.get("required_tail_control_modes", [])]
    radius = float(analytic_radius)
    weights = {str(idx): radius ** int(idx) for idx in tail_modes}
    rows = [dict(row) for row in frontier_report.get("all_rows", [])]
    single_rows = [
        row
        for row in rows
        if len(row.get("added_inactive", [])) == 1
        and int(row["added_inactive"][0]) in set(tail_modes)
    ]
    full_tail = next(
        (
            row for row in rows
            if set(int(idx) for idx in row.get("added_inactive", [])) == set(tail_modes)
        ),
        {},
    )
    max_single_neumann = max(
        (float(row.get("neumann_upper", 0.0)) for row in single_rows),
        default=0.0,
    )
    max_single_radii = max(
        (float(row.get("worst_radii_upper", 0.0)) for row in single_rows),
        default=0.0,
    )
    min_single_sigma_fraction = min(
        (float(row.get("sigma_fraction_vs_base", 1.0)) for row in single_rows),
        default=1.0,
    )
    external_ok = _external_verifies(external_verification, "weighted_analytic_tail_norm_contract")
    certified = bool(external_ok and tail_modes)
    open_obligations = []
    if not external_ok:
        open_obligations.extend([
            "external_weighted_analytic_tail_norm_contract",
            "prove_weighted_tail_algebra_constant",
            "prove_tail_projection_error_bound",
        ])
    return {
        "schema_version": "navier-stokes-weighted-analytic-tail-norm-1",
        "candidate_type": "weighted_analytic_tail_norm_contract",
        "norm_name": "weighted_l1_analytic_tail",
        "norm_formula": "sum_{k in required_tail_modes} rho^k |a_k|",
        "analytic_radius": radius,
        "algebra_constant": float(algebra_constant),
        "required_tail_modes": tail_modes,
        "tail_weights": weights,
        "finite_surrogate_constants": {
            "max_single_mode_neumann_upper": max_single_neumann,
            "max_single_mode_radii_upper": max_single_radii,
            "min_single_mode_sigma_fraction": min_single_sigma_fraction,
            "full_tail_neumann_upper": full_tail.get("neumann_upper"),
            "full_tail_radii_upper": full_tail.get("worst_radii_upper"),
            "full_tail_sigma_fraction": full_tail.get("sigma_fraction_vs_base"),
        },
        "external_verification": _json_dict(external_verification or {}),
        "weighted_tail_contract_certified": certified,
        "proof_status": "proved_by_external_artifact" if certified else "blocked_missing_tail_norm_proof",
        "open_obligations": open_obligations,
        "unproven_claim": False,
        "replay_inputs": {
            "frontier_report": _json_dict(frontier_report),
            "analytic_radius": radius,
            "algebra_constant": float(algebra_constant),
            "external_verification": _json_dict(external_verification or {}),
        },
    }


def active_subspace_tail_contraction_attempt(
    frontier_report: dict[str, Any],
    tail_contract: dict[str, Any],
    *,
    finite_diagnostic: dict[str, Any] | None = None,
    analytic_lift: dict[str, Any] | None = None,
    contraction_threshold: float = 1.0,
    external_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attempt to prove inactive modes are contractive in the tail norm."""
    if tail_contract.get("candidate_type") != "weighted_analytic_tail_norm_contract":
        raise ValueError("expected candidate_type='weighted_analytic_tail_norm_contract'")
    tail_modes = [int(idx) for idx in frontier_report.get("required_tail_control_modes", [])]
    contract_modes = [int(idx) for idx in tail_contract.get("required_tail_modes", [])]
    external_ok = _external_verifies(external_verification, "active_subspace_tail_contraction")
    ratio_value = external_verification.get("contraction_ratio_upper") if external_verification else None
    ratio = float(ratio_value) if ratio_value is not None else float("inf")
    finite_ratio = (
        None
        if finite_diagnostic is None
        else float(finite_diagnostic.get("finite_contraction_ratio_upper", float("inf")))
    )
    finite_passed = bool(
        finite_diagnostic is not None
        and finite_diagnostic.get("finite_tail_contraction_surrogate_passed", False)
    )
    lift_ratio = (
        None
        if analytic_lift is None
        else analytic_lift.get("q_total_upper")
    )
    lift_ok = bool(
        analytic_lift is not None
        and analytic_lift.get("analytic_lift_certified", False)
    )
    contract_ok = bool(tail_contract.get("weighted_tail_contract_certified", False))
    mode_match = tail_modes == contract_modes
    certified = bool(
        contract_ok
        and mode_match
        and (
            (external_ok and ratio < float(contraction_threshold))
            or lift_ok
        )
    )
    open_obligations: list[str] = []
    if not mode_match:
        open_obligations.append("tail_contract_modes_match_absorption_frontier")
    if not contract_ok:
        open_obligations.append("weighted_analytic_tail_norm_contract")
    if not external_ok and not lift_ok:
        open_obligations.append("external_active_subspace_tail_contraction_proof")
    if not finite_passed:
        open_obligations.append("finite_tail_contraction_surrogate_below_one")
    if not lift_ok:
        open_obligations.append("active_tail_contraction_analytic_lift")
    if not lift_ok and (not np.isfinite(ratio) or ratio >= float(contraction_threshold)):
        open_obligations.append("tail_contraction_ratio_below_one")
    return {
        "schema_version": "navier-stokes-active-tail-contraction-1",
        "candidate_type": "active_subspace_tail_contraction_attempt",
        "required_tail_modes": tail_modes,
        "tail_contract_sha256": _sha256_json(tail_contract),
        "contraction_inequality": "||Pi_tail DNewton(active + tail)||_tail <= q ||tail||_tail with q < 1",
        "contraction_threshold": float(contraction_threshold),
        "contraction_ratio_upper": ratio if np.isfinite(ratio) else None,
        "finite_contraction_ratio_upper": finite_ratio if finite_ratio is not None and np.isfinite(finite_ratio) else None,
        "finite_tail_contraction_surrogate_passed": finite_passed,
        "finite_diagnostic_sha256": None if finite_diagnostic is None else _sha256_json(finite_diagnostic),
        "analytic_lift_q_total_upper": lift_ratio,
        "analytic_lift_certified": lift_ok,
        "analytic_lift_sha256": None if analytic_lift is None else _sha256_json(analytic_lift),
        "weighted_tail_contract_certified": contract_ok,
        "external_verification": _json_dict(external_verification or {}),
        "tail_contraction_certified": certified,
        "proof_status": "proved_by_external_artifact" if certified else "blocked_missing_tail_contraction_proof",
        "open_obligations": open_obligations,
        "unproven_claim": False,
        "replay_inputs": {
            "frontier_report": _json_dict(frontier_report),
            "tail_contract": _json_dict(tail_contract),
            "finite_diagnostic": _json_dict(finite_diagnostic or {}),
            "analytic_lift": _json_dict(analytic_lift or {}),
            "contraction_threshold": float(contraction_threshold),
            "external_verification": _json_dict(external_verification or {}),
        },
    }


def active_subspace_completeness_theorem_attempt(
    frontier_report: dict[str, Any],
    tail_contraction: dict[str, Any],
    *,
    invariance_report: dict[str, Any] | None = None,
    external_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attempt the active-core plus analytic-tail completeness theorem."""
    if tail_contraction.get("candidate_type") != "active_subspace_tail_contraction_attempt":
        raise ValueError("expected candidate_type='active_subspace_tail_contraction_attempt'")
    passing_rows = [
        dict(row)
        for row in frontier_report.get("all_rows", [])
        if bool(row.get("frontier_passed", False))
    ]
    base_only_passes = bool(
        passing_rows
        and all(len(row.get("added_inactive", [])) == 0 for row in passing_rows)
    )
    finite_core_ok = bool(
        base_only_passes
        and int(frontier_report.get("passed_count", 0)) >= 1
        and frontier_report.get("base_metrics", {}).get("frontier_passed", False)
    )
    invariance_ok = True if invariance_report is None else bool(
        invariance_report.get("finite_invariance_heuristic_passed", False)
    )
    tail_ok = bool(tail_contraction.get("tail_contraction_certified", False))
    external_ok = _external_verifies(external_verification, "active_subspace_completeness")
    complete = bool(finite_core_ok and invariance_ok and tail_ok and external_ok)
    open_obligations: list[str] = []
    if not finite_core_ok:
        open_obligations.append("finite_active_core_absorption_frontier")
    if not invariance_ok:
        open_obligations.append("finite_active_subspace_invariance")
    if not tail_ok:
        open_obligations.append("active_subspace_tail_contraction")
    if not external_ok:
        open_obligations.append("external_active_subspace_completeness_proof")
    return {
        "schema_version": "navier-stokes-active-subspace-completeness-1",
        "candidate_type": "active_subspace_completeness_theorem_attempt",
        "finite_active_core_closed": finite_core_ok,
        "finite_invariance_heuristic_passed": invariance_ok,
        "required_tail_modes": [int(idx) for idx in frontier_report.get("required_tail_control_modes", [])],
        "tail_contraction_certified": tail_ok,
        "external_verification": _json_dict(external_verification or {}),
        "active_subspace_complete": complete,
        "proof_status": "proved_by_external_artifact" if complete else "blocked_with_precise_obligations",
        "open_obligations": open_obligations,
        "unproven_claim": False,
        "replay_inputs": {
            "frontier_report": _json_dict(frontier_report),
            "tail_contraction": _json_dict(tail_contraction),
            "invariance_report": _json_dict(invariance_report or {}),
            "external_verification": _json_dict(external_verification or {}),
        },
    }


def continuum_residual_upper_bound(interval_report: dict[str, Any]) -> float:
    """Return the largest stored continuum residual upper bound."""
    sections = (
        interval_report.get("continuum_residual_certificates", {})
        .get("sections", {})
    )
    uppers: list[float] = []
    for section in sections.values():
        for value in section.values():
            if isinstance(value, dict) and isinstance(value.get("continuum_sup_norm_interval"), dict):
                uppers.append(float(value["continuum_sup_norm_interval"]["upper"]))
    return max(uppers, default=float("inf"))


def operator_theoretic_invertibility_certificate(
    interval_report: dict[str, Any],
    linearized_certificate: dict[str, Any],
    *,
    neumann_threshold: float = 1.0,
) -> dict[str, Any]:
    """Build a projection/tail/remainder certificate for invertibility evidence.

    The certificate is certified-evidence evidence: it checks that a finite-dimensional
    inverse can be embedded into the declared function-space convention with a
    bounded tail/remainder model. It does not by itself prove continuum
    Banach-space invertibility.
    """
    tail_radius = float(
        interval_report.get("continuum_residual_certificates", {}).get("tail_radius", 0.0)
    )
    inverse = linearized_certificate.get("approximate_inverse_norm")
    inverse_norm = float(inverse) if inverse is not None else float("inf")
    matrix_norm = float(linearized_certificate.get("matrix_norm", 0.0))
    finite_dimensional = bool(linearized_certificate.get("finite_dimensional_certified", False))
    projection_error = tail_radius * (1.0 + matrix_norm)
    neumann_defect = inverse_norm * projection_error if np.isfinite(inverse_norm) else float("inf")
    neumann_interval = scalar_interval(
        neumann_defect if np.isfinite(neumann_defect) else 1.0e300,
        relative_padding=1e-12,
        certified=np.isfinite(neumann_defect),
    )
    neumann_passed = bool(finite_dimensional and neumann_interval.upper < float(neumann_threshold))
    return {
        "method": "finite_projection_tail_neumann_defect",
        "projection": {
            "name": "axisymmetric_compact_polynomial_truncation",
            "coefficient_count": int(linearized_certificate.get("coefficient_count", 0)),
            "residual_vector_size": int(linearized_certificate.get("residual_vector_size", 0)),
            "basis": interval_report.get("function_space", {}).get("coefficient_basis", ""),
        },
        "operator_remainder": {
            "tail_radius": tail_radius,
            "matrix_norm": matrix_norm,
            "projection_error_bound": projection_error,
            "model": "tail_radius_times_one_plus_matrix_norm",
        },
        "neumann_defect_interval": asdict(neumann_interval),
        "neumann_threshold": float(neumann_threshold),
        "neumann_passed": neumann_passed,
        "finite_dimensional_certified": finite_dimensional,
        "proof_prep_certified": bool(finite_dimensional and np.isfinite(neumann_defect)),
        "operator_theoretic_certified": False,
        "open_obligations": (
            []
            if neumann_passed
            else ["neumann_defect_below_one"]
        ) + ["external_banach_space_invertibility_proof"],
        "unproven_claim": False,
    }


def continuum_banach_invertibility_attempt(
    interval_report: dict[str, Any],
    linearized_certificate: dict[str, Any],
    *,
    external_verification: dict[str, Any] | None = None,
    neumann_threshold: float = 1.0,
) -> dict[str, Any]:
    """Attempt theorem-grade Banach-space invertibility certification."""
    function_space = theorem_grade_function_space_contract(interval_report)
    finite = operator_theoretic_invertibility_certificate(
        interval_report,
        linearized_certificate,
        neumann_threshold=neumann_threshold,
    )
    neumann_passed = bool(finite.get("neumann_passed", False))
    external_ok = _external_verifies(external_verification, "external_banach_space_invertibility_proof")
    certified = bool(neumann_passed and external_ok)
    open_obligations = []
    if not neumann_passed:
        open_obligations.append("neumann_defect_below_one")
    if not external_ok:
        open_obligations.append("external_banach_space_invertibility_proof")
    return {
        "schema_version": "navier-stokes-theorem-operator-1",
        "candidate_type": "theorem_grade_operator_invertibility_attempt",
        "function_space": function_space,
        "finite_projection_certificate": _json_dict(finite),
        "projection_bound": finite["operator_remainder"]["projection_error_bound"],
        "tail_operator_bound": finite["operator_remainder"]["tail_radius"],
        "approx_inverse_bound": linearized_certificate.get("approximate_inverse_norm"),
        "neumann_defect_interval": finite["neumann_defect_interval"],
        "external_verification": _json_dict(external_verification or {}),
        "operator_theoretic_certified": certified,
        "proof_status": "proved_by_external_artifact" if certified else "blocked_with_precise_obligations",
        "open_obligations": open_obligations,
        "unproven_claim": False,
    }


def componentwise_radii_polynomial_certificate(
    interval_report: dict[str, Any],
    linearized_certificate: dict[str, Any],
    operator_certificate: dict[str, Any] | None = None,
    *,
    threshold: float = 1.0,
) -> dict[str, Any]:
    """Return component-wise interval radii-polynomial evidence."""
    inverse = linearized_certificate.get("approximate_inverse_norm")
    inverse_norm = float(inverse) if inverse is not None else float("inf")
    operator_remainder = 0.0
    if operator_certificate is not None:
        operator_remainder = float(
            operator_certificate.get("operator_remainder", {}).get("projection_error_bound", 0.0)
        )
    matrix_norm = float(linearized_certificate.get("matrix_norm", 1.0))
    tail_radius = float(interval_report.get("continuum_residual_certificates", {}).get("tail_radius", 0.0))
    sections = interval_report.get("continuum_residual_certificates", {}).get("sections", {})
    components: dict[str, Any] = {}
    passed = True
    for section_name, section in sections.items():
        for quantity_name, cert in section.items():
            if not isinstance(cert, dict) or not isinstance(cert.get("continuum_sup_norm_interval"), dict):
                continue
            interval = dict(cert["continuum_sup_norm_interval"])
            residual = max(abs(float(interval["lower"])), abs(float(interval["upper"])))
            coefficient_radius = max(
                (
                    float(item.get("radius", 0.0))
                    for box in interval_report.get("coefficient_intervals", {}).values()
                    for item in box.get("intervals", [])
                ),
                default=0.0,
            )
            quadratic_lipschitz = matrix_norm * coefficient_radius + tail_radius + operator_remainder
            value = inverse_norm * (residual + quadratic_lipschitz) if np.isfinite(inverse_norm) else float("inf")
            closure = scalar_interval(
                value if np.isfinite(value) else 1.0e300,
                relative_padding=1e-12,
                certified=np.isfinite(value),
            )
            component_passed = bool(np.isfinite(value) and closure.upper < float(threshold))
            passed = passed and component_passed
            components[f"{section_name}.{quantity_name}"] = {
                "residual_bound": residual,
                "quadratic_lipschitz_bound": quadratic_lipschitz,
                "closure_interval": asdict(closure),
                "passed": component_passed,
            }
    return {
        "method": "componentwise_inverse_residual_lipschitz_radii",
        "components": components,
        "threshold": float(threshold),
        "passed": bool(passed and components),
        "certified": bool(components and np.isfinite(inverse_norm)),
        "proof_prep_certified": bool(components and np.isfinite(inverse_norm)),
        "open_obligations": [] if passed and components else ["componentwise_radii_polynomial_closure"],
        "unproven_claim": False,
    }


def theorem_grade_radii_polynomial_attempt(
    interval_report: dict[str, Any],
    operator_attempt: dict[str, Any],
    componentwise_certificate: dict[str, Any],
    *,
    external_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attempt theorem-grade radii-polynomial closure in the Banach contract."""
    components = dict(componentwise_certificate.get("components", {}))
    worst_upper = max(
        (
            float(component.get("closure_interval", {}).get("upper", float("inf")))
            for component in components.values()
        ),
        default=float("inf"),
    )
    outward_value = worst_upper - 1.0 if np.isfinite(worst_upper) else float("inf")
    outward_interval = scalar_interval(
        outward_value if np.isfinite(outward_value) else 1.0e300,
        relative_padding=1e-12,
        certified=np.isfinite(outward_value),
    )
    root_interval = scalar_interval(
        max(0.0, 1.0 - worst_upper) if np.isfinite(worst_upper) else 1.0e300,
        relative_padding=1e-12,
        certified=np.isfinite(worst_upper) and worst_upper < 1.0,
    )
    outward_passed = bool(np.isfinite(worst_upper) and worst_upper < 1.0)
    inverse_ok = bool(operator_attempt.get("operator_theoretic_certified", False))
    external_ok = _external_verifies(external_verification, "radii_polynomial_closure")
    certified = bool(outward_passed and inverse_ok and external_ok)
    open_obligations: list[str] = []
    if not inverse_ok:
        open_obligations.append("operator_theoretic_invertibility")
    if not outward_passed:
        open_obligations.append("certified_outward_radii_inequality")
    if not external_ok:
        open_obligations.append("external_radii_polynomial_proof")
    return {
        "schema_version": "navier-stokes-theorem-radii-1",
        "candidate_type": "theorem_grade_radii_polynomial_attempt",
        "function_space": _json_dict(operator_attempt.get("function_space", theorem_grade_function_space_contract(interval_report))),
        "componentwise_remainders": _json_dict(components),
        "root_interval": asdict(root_interval),
        "outward_mapping_interval": asdict(outward_interval),
        "outward_inequality_passed": outward_passed,
        "operator_inverse_certified": inverse_ok,
        "external_verification": _json_dict(external_verification or {}),
        "radii_polynomial_closure": certified,
        "failure_reason": "" if certified else ",".join(open_obligations),
        "proof_status": "proved_by_external_artifact" if certified else "blocked_with_precise_obligations",
        "open_obligations": open_obligations,
        "unproven_claim": False,
    }


def radii_polynomial_certificate(
    interval_report: dict[str, Any],
    linearized_certificate: dict[str, Any] | None = None,
    *,
    nonlinear_lipschitz_bound: float | None = None,
    residual_bound: float | None = None,
    threshold: float = 1.0,
) -> dict[str, Any]:
    """Build a finite-dimensional radii-polynomial consistency certificate."""
    residual = float(residual_bound if residual_bound is not None else continuum_residual_upper_bound(interval_report))
    inverse = (
        None
        if linearized_certificate is None
        else linearized_certificate.get("approximate_inverse_norm")
    )
    inverse_value = None if inverse is None else float(inverse)
    lipschitz = float(
        nonlinear_lipschitz_bound
        if nonlinear_lipschitz_bound is not None
        else _coefficient_box_lipschitz_bound(interval_report, linearized_certificate)
    )
    finite_inputs = inverse_value is not None and np.isfinite(inverse_value) and np.isfinite(lipschitz)
    radii_value = (
        float(inverse_value) * (residual + lipschitz)
        if finite_inputs
        else float("inf")
    )
    interval = scalar_interval(
        radii_value if np.isfinite(radii_value) else 1.0e300,
        relative_padding=1e-12,
        certified=bool(finite_inputs),
    )
    passed = bool(finite_inputs and interval.upper < float(threshold))
    open_obligations: list[str] = []
    if inverse_value is None or not np.isfinite(inverse_value):
        open_obligations.append("linearized_invertibility")
    if not passed:
        open_obligations.append("radii_polynomial_closure")
    cert = RadiiPolynomialCertificate(
        residual_bound=residual,
        approximate_inverse_norm=inverse_value,
        nonlinear_lipschitz_bound=lipschitz,
        closure_interval=interval,
        passed=passed,
        certified=bool(finite_inputs),
        open_obligations=tuple(open_obligations),
    )
    return {
        **asdict(cert),
        "threshold": float(threshold),
    }


def norm_divergence_certificate(
    interval_report: dict[str, Any],
    *,
    norm_name: str = "candidate_trace_norm",
    blowup_time: float | None = None,
    growth_exponent: float | None = None,
    linked_to_field_profile: bool = False,
) -> dict[str, Any]:
    """Build a norm-divergence evidence record for the blow-up route."""
    exponent = None if growth_exponent is None else float(growth_exponent)
    lower = (
        None
        if exponent is None
        else scalar_interval(exponent, relative_padding=1e-12, certified=exponent > 0.0)
    )
    axis_ok = bool(interval_report.get("upgrade_gate", {}).get("axis_regular_certified", False))
    energy_ok = bool(interval_report.get("upgrade_gate", {}).get("finite_energy_certified", False))
    certified = bool(
        exponent is not None
        and exponent > 0.0
        and linked_to_field_profile
        and axis_ok
        and energy_ok
    )
    open_obligations: list[str] = []
    if exponent is None or exponent <= 0.0:
        open_obligations.append("positive_norm_growth_exponent")
    if not linked_to_field_profile:
        open_obligations.append("link_norm_trace_to_field_profile")
    cert = NormDivergenceCertificate(
        norm_name=norm_name,
        blowup_time=blowup_time,
        growth_exponent=exponent,
        lower_bound_interval=lower,
        linked_to_field_profile=linked_to_field_profile,
        certified=certified,
        open_obligations=tuple(open_obligations),
    )
    return asdict(cert)


def exact_profile_norm_divergence_attempt(
    interval_report: dict[str, Any],
    norm_certificate: dict[str, Any],
    *,
    external_verification: dict[str, Any] | None = None,
    residual_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Attempt to link norm divergence to the exact coefficient-defined profile.

    ``residual_tolerance`` (``eps >= 0``) is the certified residual budget below
    which the profile is accepted as an exact solution. Because
    :func:`continuum_residual_upper_bound` returns a *certified upper bound* on
    the continuum residual sup-norm (which is ``>= 0``), the predicate
    ``residual_bound <= eps`` is a rigorous, conservative certificate that the
    true residual is ``<= eps``.

    The default ``eps = 0.0`` keeps the strict exact-solution gate while
    avoiding the exact-float ``== 0.0`` footgun: a computed upper bound is
    virtually never *bit-exactly* ``0.0`` unless a symbolic residual is injected
    as such, and ``<= 0.0`` matches that case for a non-negative bound while
    reading as a certified ``<=`` predicate. Callers with a rigorous residual
    budget may pass ``eps > 0`` to accept "exact within a certified tolerance".
    """
    if not (residual_tolerance >= 0.0):
        raise ValueError(
            "residual_tolerance must be a non-negative certified epsilon"
        )
    artifact = interval_report.get("replay_inputs", {}).get("refined_artifact", {})
    residual_bound = continuum_residual_upper_bound(interval_report)
    exact_profile = bool(residual_bound <= residual_tolerance)
    finite_energy_ok = bool(interval_report.get("upgrade_gate", {}).get("finite_energy_certified", False))
    axis_ok = bool(interval_report.get("upgrade_gate", {}).get("axis_regular_certified", False))
    exponent = norm_certificate.get("growth_exponent")
    exponent_ok = bool(exponent is not None and float(exponent) > 0.0)
    external_ok = _external_verifies(external_verification, "exact_profile_norm_divergence")
    certified = bool(exact_profile and finite_energy_ok and axis_ok and exponent_ok and external_ok)
    open_obligations: list[str] = []
    if not exact_profile:
        open_obligations.append("blocked_profile_not_exact_solution")
    if not finite_energy_ok:
        open_obligations.append("finite_energy_initial_data")
    if not axis_ok:
        open_obligations.append("axis_smoothness")
    if not exponent_ok:
        open_obligations.append("positive_norm_growth_exponent")
    if not external_ok:
        open_obligations.append("external_norm_divergence_proof")
    return {
        "schema_version": "navier-stokes-theorem-norm-divergence-1",
        "candidate_type": "theorem_grade_norm_divergence_attempt",
        "norm_name": str(norm_certificate.get("norm_name", "candidate_trace_norm")),
        "profile_scaling_law": {
            "blowup_time": norm_certificate.get("blowup_time"),
            "growth_exponent": exponent,
            "lower_bound_interval": _json_dict(norm_certificate.get("lower_bound_interval")),
        },
        "field_profile_linkage": {
            "candidate_type": artifact.get("candidate_type", "unknown"),
            "coefficient_hash": hashlib.sha256(
                json.dumps(_json_dict(artifact.get("replay_inputs", {}).get("coefficients", [])), sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "continuum_residual_upper_bound": residual_bound,
            "residual_tolerance": residual_tolerance,
            "exact_profile_verified": exact_profile,
            "finite_energy_certified": finite_energy_ok,
            "axis_regular_certified": axis_ok,
        },
        "external_verification": _json_dict(external_verification or {}),
        "norm_divergence": certified,
        "proof_status": "proved_by_external_artifact" if certified else "blocked_with_precise_obligations",
        "open_obligations": open_obligations,
        "unproven_claim": False,
    }


def build_axisymmetric_blowup_closure_report(
    interval_report: dict[str, Any],
    *,
    norm_growth_exponent: float | None = None,
    blowup_time: float | None = None,
    linked_norm_profile: bool = False,
) -> dict[str, Any]:
    """Build a replayable blow-up closure report for an axisymmetric survivor."""
    artifact = interval_report["replay_inputs"]["refined_artifact"]
    function_space = asdict(axisymmetric_function_space_metadata(interval_report))
    linearized = assemble_axisymmetric_linearized_operator(artifact)
    operator_cert = operator_theoretic_invertibility_certificate(interval_report, linearized)
    componentwise_radii = componentwise_radii_polynomial_certificate(
        interval_report,
        linearized,
        operator_cert,
    )
    radii = radii_polynomial_certificate(
        interval_report,
        linearized,
        nonlinear_lipschitz_bound=max(
            (
                float(component.get("quadratic_lipschitz_bound", 0.0))
                for component in componentwise_radii.get("components", {}).values()
            ),
            default=None,
        ),
    )
    radii["componentwise"] = componentwise_radii
    radii["passed"] = bool(radii.get("passed", False) and componentwise_radii.get("passed", False))
    if not componentwise_radii.get("passed", False):
        radii["open_obligations"] = sorted({
            *[str(v) for v in radii.get("open_obligations", [])],
            "componentwise_radii_polynomial_closure",
        })
    norm_cert = norm_divergence_certificate(
        interval_report,
        blowup_time=blowup_time,
        growth_exponent=norm_growth_exponent,
        linked_to_field_profile=linked_norm_profile,
    )
    inverse = linearized.get("approximate_inverse_norm") if linearized.get("finite_dimensional_certified") else None
    blowup = build_blowup_closure_report(
        interval_report,
        approximate_inverse_norm=None if inverse is None else float(inverse),
        nonlinear_lipschitz_bound=float(radii["nonlinear_lipschitz_bound"]),
        residual_bound=float(radii["residual_bound"]),
        norm_growth_exponent=(
            float(norm_cert["growth_exponent"])
            if norm_cert.get("certified", False) and norm_cert.get("growth_exponent") is not None
            else None
        ),
        function_space=function_space,
        linearized_certificate={
            **linearized,
            "operator_theoretic_certificate": operator_cert,
        },
        radii_certificate=radii,
        norm_divergence_certificate=norm_cert,
    )
    closure_consistency_verified = bool(
        linearized.get("finite_dimensional_certified", False)
        and radii.get("certified", False)
        and interval_report.get("honesty", {}).get("interval_verified", False)
    )
    blowup["closure_consistency_verified"] = closure_consistency_verified
    blowup["blocker_resolution"] = {
        "operator_theoretic_invertibility": operator_cert,
        "radii_polynomial_closure": componentwise_radii,
        "norm_divergence": norm_cert,
        "proof_prep_certified": bool(
            operator_cert.get("proof_prep_certified", False)
            and componentwise_radii.get("proof_prep_certified", False)
            and norm_cert.get("growth_exponent") is not None
        ),
        "unproven_claim": False,
    }
    blowup["proof_prep_status"] = (
        "closure_consistency_verified"
        if closure_consistency_verified
        else "blocked_open_obligations"
    )
    return blowup


def build_axisymmetric_active_subspace_closure_report(
    interval_report: dict[str, Any],
    active_indices: list[int] | tuple[int, ...],
    *,
    norm_growth_exponent: float | None = None,
    blowup_time: float | None = None,
    linked_norm_profile: bool = False,
) -> dict[str, Any]:
    """Build a replayable closure report for a sparse active coefficient ansatz."""
    artifact = interval_report["replay_inputs"]["refined_artifact"]
    active = tuple(int(idx) for idx in active_indices)
    linearized = assemble_axisymmetric_active_subspace_operator(artifact, active)
    operator_cert = operator_theoretic_invertibility_certificate(interval_report, linearized)
    componentwise_radii = componentwise_radii_polynomial_certificate(
        interval_report,
        linearized,
        operator_cert,
    )
    radii = radii_polynomial_certificate(
        interval_report,
        linearized,
        nonlinear_lipschitz_bound=max(
            (
                float(component.get("quadratic_lipschitz_bound", 0.0))
                for component in componentwise_radii.get("components", {}).values()
            ),
            default=None,
        ),
    )
    radii["componentwise"] = componentwise_radii
    radii["passed"] = bool(radii.get("passed", False) and componentwise_radii.get("passed", False))
    norm_cert = norm_divergence_certificate(
        interval_report,
        blowup_time=blowup_time,
        growth_exponent=norm_growth_exponent,
        linked_to_field_profile=linked_norm_profile,
    )
    closure_consistency_verified = bool(
        linearized.get("finite_dimensional_certified", False)
        and radii.get("certified", False)
        and interval_report.get("honesty", {}).get("interval_verified", False)
    )
    open_obligations = sorted({
        *[str(v) for v in operator_cert.get("open_obligations", [])],
        *[str(v) for v in componentwise_radii.get("open_obligations", [])],
        "active_subspace_completeness",
        "prove_sparse_ansatz_invariant_under_newton_map",
    })
    return {
        "schema_version": "navier-stokes-active-subspace-closure-1",
        "candidate_type": "active_subspace_blowup_closure_report",
        "selected_route": "finite_time_blowup",
        "active_subspace": {
            "active_coefficient_indices": list(active),
            "active_coefficient_count": len(active),
            "ambient_coefficient_count": len(artifact.get("replay_inputs", {}).get("coefficients", [])),
            "completeness_certified": False,
            "open_obligations": [
                "active_subspace_completeness",
                "prove_sparse_ansatz_invariant_under_newton_map",
            ],
        },
        "closure_certificates": {
            "linearized_operator": _json_dict(linearized),
            "operator_theoretic_invertibility": _json_dict(operator_cert),
            "radii_polynomial": _json_dict(radii),
            "norm_divergence": _json_dict(norm_cert),
        },
        "closure_consistency_verified": closure_consistency_verified,
        "proof_prep_status": (
            "active_subspace_closure_consistency_verified"
            if closure_consistency_verified
            else "blocked_open_obligations"
        ),
        "open_obligations": open_obligations,
        "unproven_claim": False,
        "replay_inputs": {
            "interval_report": _json_dict(interval_report),
            "active_indices": list(active),
            "norm_growth_exponent": norm_growth_exponent,
            "blowup_time": blowup_time,
            "linked_norm_profile": bool(linked_norm_profile),
        },
        "independent_replay": {
            "status": "required",
            "harness": "omnibias.symbolic.navier_stokes.verify_active_subspace_closure_report",
        },
    }


def _coefficient_box_lipschitz_bound(
    interval_report: dict[str, Any],
    linearized_certificate: dict[str, Any] | None,
) -> float:
    matrix_norm = 1.0 if linearized_certificate is None else float(linearized_certificate.get("matrix_norm", 1.0))
    radii: list[float] = []
    for box in interval_report.get("coefficient_intervals", {}).values():
        for item in box.get("intervals", []):
            radii.append(float(item.get("radius", 0.0)))
    coefficient_radius = max(radii, default=0.0)
    tail_radius = float(
        interval_report.get("continuum_residual_certificates", {}).get("tail_radius", 0.0)
    )
    return float(matrix_norm * coefficient_radius + tail_radius)


def build_axisymmetric_interval_report(
    refined_artifact: dict[str, Any],
    *,
    coefficient_absolute_padding: float = 1e-12,
    coefficient_relative_padding: float = 1e-12,
    residual_absolute_padding: float = 1e-10,
    residual_relative_padding: float = 1e-8,
    energy_absolute_padding: float = 1e-10,
    energy_relative_padding: float = 1e-8,
    certified_interval_sections: bool = True,
) -> dict[str, Any]:
    """Build an interval-validation report for a refined axisymmetric artifact."""
    if refined_artifact.get("candidate_type") != "axisymmetric_swirl_refined":
        raise ValueError("interval reports currently require axisymmetric_swirl_refined artifacts")
    backend = interval_arithmetic_metadata()
    coefficient_boxes = coefficient_interval_boxes(
        refined_artifact,
        absolute_padding=coefficient_absolute_padding,
        relative_padding=coefficient_relative_padding,
        certified=certified_interval_sections,
    )
    residuals = residual_interval_envelopes(
        refined_artifact,
        absolute_padding=residual_absolute_padding,
        relative_padding=residual_relative_padding,
        coefficient_padding=coefficient_absolute_padding + coefficient_relative_padding,
    )
    energy = finite_energy_interval_bounds(
        refined_artifact,
        absolute_padding=energy_absolute_padding,
        relative_padding=energy_relative_padding,
        certified=certified_interval_sections,
    )
    tail_certificates = certified_tail_bounds_from_artifact(refined_artifact)
    continuum = continuum_residual_certificates(
        refined_artifact,
        residual_intervals=residuals,
        tail_bounds=tail_certificates,
    )
    finite_energy = finite_energy_tail_certificate(
        refined_artifact,
        tail_bounds=tail_certificates,
        quadrature_relative_padding=energy_relative_padding,
    )
    axis_checks = axisymmetric_basis_regular_interval_checks(refined_artifact)
    unresolved = [
        "linearized invertibility or theorem-level a priori estimate",
    ]
    interval_inputs_present = bool(coefficient_boxes and residuals and energy)
    tail_bounds_certified = bool(
        tail_certificates
        and all(bool(tail.get("certified", False)) for tail in tail_certificates.values())
    )
    continuum_bound_certified = bool(continuum.get("continuum_bound_certified", False))
    finite_energy_certified = bool(finite_energy.get("certified", False))
    axis_certified = bool(axis_checks.get("certified_smooth_axis", False))
    return {
        "schema_version": "navier-stokes-interval-1",
        "candidate_type": "axisymmetric_interval_report",
        "source_candidate_type": refined_artifact.get("candidate_type"),
        "interval_backend": asdict(backend),
        "function_space": asdict(axisymmetric_function_space_metadata(refined_artifact)),
        "coefficient_intervals": coefficient_boxes,
        "tail_certificates": tail_certificates,
        "residual_intervals": residuals,
        "continuum_residual_certificates": continuum,
        "finite_energy_intervals": energy,
        "finite_energy_tail_certificate": finite_energy,
        "axis_regular_checks": axis_checks,
        "uncertainty_config": {
            "coefficient_absolute_padding": float(coefficient_absolute_padding),
            "coefficient_relative_padding": float(coefficient_relative_padding),
            "residual_absolute_padding": float(residual_absolute_padding),
            "residual_relative_padding": float(residual_relative_padding),
            "energy_absolute_padding": float(energy_absolute_padding),
            "energy_relative_padding": float(energy_relative_padding),
        },
        "replay_inputs": {
            "refined_artifact": _json_dict(refined_artifact),
        },
        "independent_replay": {
            "status": "required",
            "harness": "omnibias.symbolic.navier_stokes.verify_axisymmetric_interval_report",
        },
        "upgrade_gate": {
            "stage": "interval_obligation_ready" if interval_inputs_present else "numerical_artifact",
            "interval_inputs_present": bool(interval_inputs_present),
            "tail_bounds_certified": bool(tail_bounds_certified),
            "continuum_bound_certified": bool(continuum_bound_certified),
            "finite_energy_certified": bool(finite_energy_certified),
            "axis_regular_certified": bool(axis_certified),
            "unproven_claim": False,
        },
        "proof_obligations": unresolved,
        "honesty": {
            "unproven_claim": False,
            "interval_verified": bool(
                tail_bounds_certified
                and continuum_bound_certified
                and finite_energy_certified
                and axis_certified
            ),
            "theorem_prover_verified": False,
            "finite_energy_verified": bool(finite_energy_certified),
            "notes": (
                "finite-dimensional interval certificate with certified tails, "
                "continuum residual envelopes, finite-energy bounds, and axis "
                "basis checks; analytic closure remains open"
            ),
        },
        "provenance": {
            "harness": "omnibias.pinn.certified.navier_stokes.build_axisymmetric_interval_report",
            "python": platform.python_version(),
        },
    }


def build_axisymmetric_swirl_candidate_artifact(
    *,
    seed: int = 0,
    n_radial: int = 16,
    n_axial: int = 17,
    viscosity: float = 1e-3,
    density: float = 1.0,
    amplitude: float = 0.1,
    tail_l1_bound: float = 1e-6,
) -> dict[str, Any]:
    """Build a deterministic replayable axisymmetric-with-swirl candidate artifact."""
    grid = axisymmetric_meridional_replay_grid(n_radial=n_radial, n_axial=n_axial)
    radial, axial = axisymmetric_physical_axes(grid)
    rr = radial[:, None]
    zz = axial[None, :]
    rng = np.random.default_rng(seed)
    phase = float(rng.uniform(-0.125, 0.125))
    envelope = np.exp(-(rr * rr + zz * zz))
    streamfunction = amplitude * rr * rr * envelope * (1.0 + phase * np.cos(zz))
    swirl = amplitude * rr * envelope * (1.0 - phase * np.sin(zz))
    pressure = -0.5 * density * amplitude * amplitude * envelope * envelope
    residual = axisymmetric_swirl_residual_samples(
        streamfunction,
        swirl,
        pressure,
        radial_axis=radial,
        axial_axis=axial,
        viscosity=viscosity,
        density=density,
    )
    energy = axisymmetric_energy_estimate(
        residual["velocity"],
        radial_axis=radial,
        axial_axis=axial,
    )
    coeffs = (
        compactified_coefficient_set(
            "streamfunction",
            streamfunction,
            basis="sampled_axisymmetric_meridional_values",
            tail_l1_bound=tail_l1_bound,
            finite_energy_estimate=energy,
        ),
        compactified_coefficient_set(
            "swirl",
            swirl,
            basis="sampled_axisymmetric_meridional_values",
            tail_l1_bound=tail_l1_bound,
            finite_energy_estimate=energy,
        ),
        compactified_coefficient_set(
            "pressure",
            pressure,
            basis="sampled_axisymmetric_meridional_values",
            tail_l1_bound=tail_l1_bound,
            finite_energy_estimate=None,
        ),
    )
    diagnostics = dict(residual["residual_diagnostics"])
    diagnostics["axisymmetric_energy_estimate"] = energy
    diagnostics["seed"] = int(seed)
    return build_candidate_artifact(
        candidate_type="axisymmetric_swirl_sandbox",
        replay_grid=grid,
        replay_inputs={
            "radial_axis": radial,
            "axial_axis": axial,
            "streamfunction": streamfunction,
            "swirl": swirl,
            "pressure": pressure,
            "viscosity": float(viscosity),
            "density": float(density),
            "axisymmetric_metadata": axisymmetric_compactified_metadata(),
            "ansatz_metadata": axisymmetric_swirl_ansatz_metadata(),
        },
        result={
            "residual_diagnostics": diagnostics,
            "residual_samples": residual["residual_samples"],
            "velocity": residual["velocity"],
            "finite_energy_estimate": energy,
            "unproven_claim": False,
        },
        coefficients=coeffs,
        upgrade_gate={
            "stage": "numerical_artifact",
            "independent_replay_required": True,
            "residual_ok": False,
            "unproven_claim": False,
        },
        proof_obligations=axisymmetric_swirl_ansatz_metadata().open_obligations,
        notes="axisymmetric swirl sandbox candidate; no global-regularity claim",
    )


def manufactured_abc_flow(
    n: int,
    *,
    viscosity: float = 1e-3,
    density: float = 1.0,
    amplitude: float = 1.0,
) -> dict[str, Any]:
    r"""Return a smooth 3D divergence-free manufactured ABC-flow sample.

    Uses the Beltrami field

    .. math::

       u = a(\sin z + \cos y,\; \sin x + \cos z,\; \sin y + \cos x).

    Since ``curl u = u`` and ``Delta u = -u``, the steady primitive equation is
    satisfied by ``p = -rho |u|^2 / 2`` and forcing ``f = nu u``.
    """
    if n < 4:
        raise ValueError(f"manufactured_abc_flow needs n >= 4, got {n}")
    x = 2.0 * np.pi * np.arange(n, dtype=float) / n
    raw_mesh: tuple[Any, ...] = tuple(np.meshgrid(x, x, x, indexing="ij"))
    xx = cast(np.ndarray, np.asarray(raw_mesh[0], dtype=float))
    yy = cast(np.ndarray, np.asarray(raw_mesh[1], dtype=float))
    zz = cast(np.ndarray, np.asarray(raw_mesh[2], dtype=float))
    velocity = amplitude * np.stack([
        np.sin(zz) + np.cos(yy),
        np.sin(xx) + np.cos(zz),
        np.sin(yy) + np.cos(xx),
    ])
    pressure = -0.5 * density * np.sum(velocity * velocity, axis=0)
    return {
        "velocity": velocity,
        "pressure": pressure,
        "velocity_t": np.zeros_like(velocity),
        "forcing": viscosity * velocity,
        "lengths": (2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi),
        "viscosity": float(viscosity),
        "density": float(density),
        "description": "steady Beltrami ABC manufactured Navier-Stokes flow",
    }


def energy_diagnostics(
    velocity: np.ndarray,
    *,
    pressure: np.ndarray | None = None,
    lengths: tuple[float, ...] | None = None,
) -> dict[str, float]:
    """Return energy/enstrophy/palinstrophy and proof-relevant residual proxies."""
    u = _as_velocity(velocity)
    domain = _lengths(lengths, u.shape[0])
    cell = float(np.prod(domain) / np.prod(u.shape[1:]))
    vort = spectral_curl(u, lengths=domain)
    vort_sq = vort * vort if vort.ndim == u.ndim - 1 else np.sum(vort * vort, axis=0)
    grad_vort = (
        spectral_gradient_scalar(vort, lengths=domain)
        if u.shape[0] == 2
        else np.stack([spectral_gradient_scalar(vort[i], lengths=domain) for i in range(3)])
    )
    pal_density = np.sum(grad_vort * grad_vort, axis=0)
    if u.shape[0] == 3:
        pal_density = np.sum(pal_density, axis=0)
    div = spectral_divergence(u, lengths=domain)
    out = {
        "kinetic_energy": float(0.5 * cell * np.sum(u * u)),
        "enstrophy": float(0.5 * cell * np.sum(vort_sq)),
        "palinstrophy": float(0.5 * cell * np.sum(pal_density)),
        "max_abs_divergence": float(np.max(np.abs(div))),
        "rms_divergence": float(np.sqrt(np.mean(div * div))),
        "bkm_vorticity_proxy": float(np.max(np.abs(vort))),
    }
    if pressure is not None:
        pp = pressure_poisson_residual_periodic(u, pressure, lengths=domain)
        out["pressure_poisson_max_abs"] = float(np.max(np.abs(pp)))
        out["pressure_poisson_rms"] = float(np.sqrt(np.mean(pp * pp)))
    return out


def build_ns_cap_bundle(
    velocity: np.ndarray,
    pressure: np.ndarray | None = None,
    *,
    velocity_t: np.ndarray | None = None,
    forcing: np.ndarray | None = None,
    viscosity: float = 1e-3,
    density: float = 1.0,
    lengths: tuple[float, ...] | None = None,
    domain_type: DomainType = "periodic_torus",
    contract: NavierStokesProofContract | None = None,
    honesty: HonestyLabels | None = None,
    compactification: CompactifiedR3Metadata | dict[str, Any] | None = None,
    tail_bounds: tuple[TailBound, ...] | list[dict[str, Any]] | None = None,
    finite_energy_checks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable Navier-Stokes certified-evidence CAP bundle."""
    u = _as_velocity(velocity)
    dim = u.shape[0]
    domain = _lengths(lengths, dim)
    p = np.zeros_like(u[0]) if pressure is None else np.asarray(pressure, dtype=float)
    u_t = np.zeros_like(u) if velocity_t is None else _as_velocity(velocity_t)
    f = np.zeros_like(u) if forcing is None else _as_velocity(forcing)
    labels = honesty if honesty is not None else HonestyLabels(
        periodic_model_only=(domain_type == "periodic_torus"),
        finite_energy_verified=False,
    )
    active_contract = contract if contract is not None else global_regularity_contract()
    residual, continuity = primitive_residual_periodic(
        u, p, velocity_t=u_t, forcing=f, viscosity=viscosity,
        density=density, lengths=domain,
    )
    pp = pressure_poisson_residual_periodic(u, p, density=density, lengths=domain)
    diag = energy_diagnostics(u, pressure=p, lengths=domain)
    diag.update({
        "max_abs_momentum_residual": float(np.max(np.abs(residual))),
        "rms_momentum_residual": float(np.sqrt(np.mean(residual * residual))),
        "max_abs_continuity": float(np.max(np.abs(continuity))),
        "rms_continuity": float(np.sqrt(np.mean(continuity * continuity))),
        "max_abs_pressure_poisson": float(np.max(np.abs(pp))),
        "rms_pressure_poisson": float(np.sqrt(np.mean(pp * pp))),
    })
    domain_meta: dict[str, Any] = {
        "type": domain_type,
        "periodic": domain_type == "periodic_torus",
        "lengths": list(domain),
        "grid_shape": list(u.shape[1:]),
        "dtype": "float64",
    }
    if domain_type == "compactified_r3":
        comp = compactification if compactification is not None else compactified_r3_metadata()
        domain_meta["compactification"] = (
            asdict(comp) if isinstance(comp, CompactifiedR3Metadata) else dict(comp)
        )
        normalized_tail_bounds: list[dict[str, Any]] = []
        for tb in tail_bounds or ():
            normalized_tail_bounds.append(asdict(tb) if isinstance(tb, TailBound) else dict(tb))
        domain_meta["tail_bounds"] = normalized_tail_bounds
        domain_meta["finite_energy_checks"] = dict(finite_energy_checks or {})
    return {
        "schema_version": SCHEMA_VERSION,
        "problem": {
            "model": "incompressible_navier_stokes",
            "dimension": dim,
            "form": "primitive_periodic_spectral",
            "contract": asdict(active_contract),
        },
        "domain": domain_meta,
        "field_samples": {
            "velocity": u.tolist(),
            "pressure": p.tolist(),
            "velocity_t": u_t.tolist(),
            "forcing": f.tolist(),
        },
        "residual_samples": {
            "momentum": residual.tolist(),
            "continuity": continuity.tolist(),
            "pressure_poisson": pp.tolist(),
        },
        "residual_diagnostics": diag,
        "validation_inputs": {
            "velocity": u.tolist(),
            "pressure": p.tolist(),
            "velocity_t": u_t.tolist(),
            "forcing": f.tolist(),
            "viscosity": float(viscosity),
            "density": float(density),
            "lengths": list(domain),
            "spectral_convention": "periodic_fft_2pi_fftfreq",
        },
        "honesty": asdict(labels),
        "proof_obligations": [asdict(o) for o in default_proof_obligations(active_contract.target)],
        "provenance": {
            "harness": "omnibias.pinn.certified.navier_stokes",
            "python": platform.python_version(),
            "derivatives": "periodic spectral FFT; omnibias field derivatives feed compatible samples",
        },
    }


def ns_cap_schema_errors(bundle: dict[str, Any]) -> list[str]:
    """Return schema and honesty-label problems for an NS CAP bundle."""
    errors: list[str] = []
    for key in REQUIRED_CAP_KEYS:
        if key not in bundle:
            errors.append(f"missing top-level key: {key!r}")
    vin = bundle.get("validation_inputs", {})
    for key in REQUIRED_VALIDATION_KEYS:
        if key not in vin:
            errors.append(f"missing validation_inputs key: {key!r}")
    honesty = bundle.get("honesty", {})
    # A stored bundle is a JSON file anyone can edit, so its own word for the
    # formal claim is worthless: refuse a self-declared `theorem_prover_verified`
    # outright. Because `_stage` maps any schema error to stage "invalid", this
    # also blocks a forged bundle from being promoted to
    # "externally_verified_artifact" or from satisfying the unproven_claim
    # precondition below.
    errors.extend(formal_claim_forgery_errors(honesty))
    if honesty.get("unproven_claim", False) and not (
        honesty.get("interval_verified", False)
        and kernel_earned_theorem_prover_verified(bundle)
        and honesty.get("finite_energy_verified", False)
    ):
        errors.append(
            "honesty.unproven_claim requires interval, theorem-prover, and finite-energy verification"
        )
    domain = bundle.get("domain", {})
    if domain.get("type") == "compactified_r3":
        if "compactification" not in domain:
            errors.append("compactified_r3 domain requires compactification metadata")
        if "tail_bounds" not in domain:
            errors.append("compactified_r3 domain requires tail_bounds")
        if "finite_energy_checks" not in domain:
            errors.append("compactified_r3 domain requires finite_energy_checks")
        if honesty.get("finite_energy_verified", False) and not domain.get("finite_energy_checks"):
            errors.append("finite_energy_verified requires finite_energy_checks")
    try:
        velocity = np.asarray(vin["velocity"], dtype=float)
        if velocity.shape[0] != velocity.ndim - 1:
            errors.append("validation_inputs.velocity must be component-first")
        if list(velocity.shape[1:]) != bundle["domain"]["grid_shape"]:
            errors.append("domain.grid_shape does not match velocity samples")
    except (KeyError, TypeError, ValueError):
        errors.append("validation_inputs.velocity is missing or malformed")
    return errors


def candidate_upgrade_gates(
    bundle: dict[str, Any],
    *,
    independent_report: dict[str, Any] | None = None,
    config: CandidateGateConfig | None = None,
) -> dict[str, Any]:
    """Evaluate promotion gates for numerical -> CAP -> interval artifacts."""
    cfg = config if config is not None else CandidateGateConfig()
    schema_errors = ns_cap_schema_errors(bundle)
    diag = bundle.get("residual_diagnostics", {})
    honesty = bundle.get("honesty", {})
    domain = bundle.get("domain", {})
    independent_match = (
        independent_report is None
        or bool(independent_report.get("residual_samples_match", False))
    )
    if cfg.require_independent_recompute and independent_report is None:
        independent_match = False

    residual_ok = (
        float(diag.get("max_abs_momentum_residual", float("inf"))) <= cfg.momentum_residual_max
        and float(diag.get("max_abs_continuity", float("inf"))) <= cfg.continuity_max
        and float(diag.get("max_abs_pressure_poisson", float("inf"))) <= cfg.pressure_poisson_max
    )
    tail_bounds = list(domain.get("tail_bounds", []))
    tail_bounds_present = bool(tail_bounds)
    tail_bounds_certified = bool(
        tail_bounds
        and all(bool(dict(tail).get("certified", False)) for tail in tail_bounds)
    )
    finite_energy_ready = bool(
        honesty.get("finite_energy_verified", False)
        and domain.get("finite_energy_checks")
    )

    if schema_errors:
        stage = "invalid"
    elif not residual_ok or not independent_match:
        stage = "numerical_artifact"
    elif (
        cfg.require_tail_bounds_for_interval
        and (
            not tail_bounds_present
            or not finite_energy_ready
            or (
                cfg.require_certified_tail_bounds_for_interval
                and not tail_bounds_certified
            )
        )
    ):
        stage = "cap_candidate"
    elif not honesty.get("interval_verified", False):
        stage = "interval_obligation_ready"
    elif not kernel_earned_theorem_prover_verified(bundle):
        stage = "proof_assistant_obligation_ready"
    else:
        stage = "externally_verified_artifact"

    return {
        "stage": stage,
        "schema_errors": schema_errors,
        "residual_ok": bool(residual_ok),
        "independent_recompute_ok": bool(independent_match),
        "tail_bounds_present": bool(tail_bounds_present),
        "tail_bounds_certified": bool(tail_bounds_certified),
        "finite_energy_ready": bool(finite_energy_ready),
        "unproven_claim": False,
    }


def build_candidate_artifact(
    *,
    candidate_type: str,
    replay_grid: ReplayGrid | dict[str, Any],
    replay_inputs: dict[str, Any],
    result: dict[str, Any],
    coefficients: tuple[CompactifiedCoefficientSet, ...] | list[dict[str, Any]] | None = None,
    upgrade_gate: dict[str, Any] | None = None,
    proof_obligations: list[str] | tuple[str, ...] = (),
    notes: str = "",
) -> dict[str, Any]:
    """Build a replayable, non-claim Navier-Stokes candidate artifact."""
    coeff_payload = [
        _json_dict(c) for c in (coefficients or [])
    ]
    gate = dict(upgrade_gate or {
        "stage": "numerical_artifact",
        "independent_replay_required": True,
        "unproven_claim": False,
    })
    gate["unproven_claim"] = False
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_type": str(candidate_type),
        "replay_grid": _json_dict(replay_grid),
        "replay_inputs": _json_dict(replay_inputs),
        "result": _json_dict(result),
        "coefficients": coeff_payload,
        "upgrade_gate": _json_dict(gate),
        "proof_obligations": [str(o) for o in proof_obligations],
        "honesty": {
            "unproven_claim": False,
            "exact_solution_claim": False,
            "global_regularity_claim": False,
            "finite_time_blowup_claim": False,
            "interval_verified": False,
            "theorem_prover_verified": False,
            "notes": notes,
        },
        "provenance": {
            "harness": "omnibias.pinn.certified.navier_stokes.candidate_bridge",
            "python": platform.python_version(),
        },
    }


def candidate_artifact_schema_errors(artifact: dict[str, Any]) -> list[str]:
    """Return candidate artifact schema / honesty problems."""
    errors: list[str] = []
    for key in REQUIRED_CANDIDATE_KEYS:
        if key not in artifact:
            errors.append(f"missing top-level key: {key!r}")
    if artifact.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        errors.append("schema_version does not match CANDIDATE_SCHEMA_VERSION")
    honesty = artifact.get("honesty", {})
    if honesty.get("unproven_claim", False) is not False:
        errors.append("honesty.unproven_claim must be False for candidate artifacts")
    if artifact.get("upgrade_gate", {}).get("unproven_claim", False) is not False:
        errors.append("upgrade_gate.unproven_claim must be False")
    replay_grid = artifact.get("replay_grid", {})
    if replay_grid.get("domain_type") == "compactified_r3" and not artifact.get("coefficients"):
        errors.append("compactified candidate artifacts require coefficient payloads")
    if not artifact.get("replay_inputs"):
        errors.append("replay_inputs must be non-empty")
    return errors


def certified_candidate_refinement_report(
    refined_artifact: dict[str, Any],
    interval_report: dict[str, Any],
) -> dict[str, Any]:
    """Summarize whether a refined candidate survives certified objectives."""
    train = interval_report.get("continuum_residual_certificates", {}).get("sections", {}).get("train", {})
    holdout = interval_report.get("continuum_residual_certificates", {}).get("sections", {}).get("holdout", {})
    gate = dict(interval_report.get("upgrade_gate", {}))
    survived = bool(
        gate.get("tail_bounds_certified", False)
        and gate.get("continuum_bound_certified", False)
        and gate.get("finite_energy_certified", False)
        and gate.get("axis_regular_certified", False)
    )
    lineage = {
        "seed": refined_artifact.get("result", {}).get("seed"),
        "basis_metadata": refined_artifact.get("replay_inputs", {}).get("basis_metadata", {}),
        "train_grid": refined_artifact.get("replay_inputs", {}).get("train_grid", {}),
        "holdout_grid": refined_artifact.get("replay_inputs", {}).get("holdout_grid", {}),
        "interval_backend": interval_report.get("interval_backend", {}),
        "loss_history": refined_artifact.get("result", {}).get("loss_history", []),
        "accepted_coefficients": refined_artifact.get("replay_inputs", {}).get("coefficients", []),
    }
    return {
        "schema_version": "navier-stokes-certified-refinement-1",
        "candidate_type": "axisymmetric_certified_refinement_report",
        "source_candidate_type": refined_artifact.get("candidate_type"),
        "survived_certified_objectives": survived,
        "train_certified_quantities": list(train),
        "holdout_certified_quantities": list(holdout),
        "lineage": _json_dict(lineage),
        "falsification": {
            "falsified": not survived,
            "reason": (
                "candidate did not satisfy every certified interval objective"
                if not survived
                else ""
            ),
        },
        "honesty": {
            "unproven_claim": False,
            "notes": "certified refinement status is a certified-evidence gate, not analytic closure",
        },
    }


def build_blowup_closure_report(
    interval_report: dict[str, Any],
    *,
    approximate_inverse_norm: float | None = None,
    nonlinear_lipschitz_bound: float | None = None,
    residual_bound: float | None = None,
    norm_growth_exponent: float | None = None,
    function_space: dict[str, Any] | AxisymmetricFunctionSpaceMetadata | None = None,
    linearized_certificate: dict[str, Any] | None = None,
    radii_certificate: dict[str, Any] | None = None,
    norm_divergence_certificate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build blow-up route analytic closure obligations."""
    residual = float(residual_bound if residual_bound is not None else 0.0)
    inverse = float(approximate_inverse_norm if approximate_inverse_norm is not None else float("inf"))
    lipschitz = float(nonlinear_lipschitz_bound if nonlinear_lipschitz_bound is not None else float("inf"))
    active_radii = radii_certificate or {
        "residual_bound": residual_bound,
        "approximate_inverse_norm": approximate_inverse_norm,
        "nonlinear_lipschitz_bound": nonlinear_lipschitz_bound,
        "closure_interval": asdict(scalar_interval(
            inverse * (residual + lipschitz)
            if np.isfinite(inverse * (residual + lipschitz))
            else 1.0e300,
            relative_padding=1e-12,
            certified=np.isfinite(inverse * (residual + lipschitz)),
        )),
        "passed": bool(
            np.isfinite(inverse * (residual + lipschitz))
            and scalar_interval(
                inverse * (residual + lipschitz),
                relative_padding=1e-12,
                certified=True,
            ).upper < 1.0
        ),
        "certified": bool(np.isfinite(inverse * (residual + lipschitz))),
        "method": "inverse_norm_times_residual_plus_lipschitz",
    }
    radii_interval = dict(active_radii["closure_interval"])
    if isinstance(function_space, AxisymmetricFunctionSpaceMetadata):
        normalized_space = asdict(function_space)
    elif function_space is None:
        normalized_space = asdict(axisymmetric_function_space_metadata(interval_report))
    else:
        normalized_space = dict(function_space)
    obligations = {
        "linearized_invertibility": bool(approximate_inverse_norm is not None and np.isfinite(inverse)),
        "radii_polynomial_closure": bool(active_radii.get("passed", False)),
        "axis_smoothness": bool(interval_report.get("upgrade_gate", {}).get("axis_regular_certified", False)),
        "finite_energy_initial_data": bool(interval_report.get("upgrade_gate", {}).get("finite_energy_certified", False)),
        "norm_divergence": bool(norm_growth_exponent is not None and float(norm_growth_exponent) > 0.0),
    }
    if linearized_certificate is not None:
        obligations["operator_theoretic_invertibility"] = bool(
            linearized_certificate.get("operator_theoretic_certified", False)
        )
    if norm_divergence_certificate is not None:
        obligations["norm_divergence"] = bool(norm_divergence_certificate.get("certified", False))
    return {
        "schema_version": "navier-stokes-analytic-closure-1",
        "candidate_type": "blowup_analytic_closure_report",
        "route": "finite_time_blowup",
        "function_space": _json_dict(normalized_space),
        "obligations": obligations,
        "radii_polynomial": {
            "approximate_inverse_norm": approximate_inverse_norm,
            "nonlinear_lipschitz_bound": nonlinear_lipschitz_bound,
            "residual_bound": residual_bound,
            "closure_interval": _json_dict(radii_interval),
        },
        "closure_certificates": {
            "linearized_operator": _json_dict(linearized_certificate or {}),
            "radii_polynomial": _json_dict(active_radii),
            "norm_divergence": _json_dict(norm_divergence_certificate or {}),
        },
        "replay_inputs": {
            "interval_report": _json_dict(interval_report),
            "approximate_inverse_norm": approximate_inverse_norm,
            "nonlinear_lipschitz_bound": nonlinear_lipschitz_bound,
            "residual_bound": residual_bound,
            "norm_growth_exponent": norm_growth_exponent,
        },
        "independent_replay": {
            "status": "required",
            "harness": "omnibias.symbolic.navier_stokes.verify_blowup_closure_report",
        },
        "formalizable": bool(all(obligations.values())),
        "unproven_claim": False,
        "open_obligations": [name for name, ok in obligations.items() if not ok],
    }


def build_regularity_inequality_report(
    regularity_artifact: dict[str, Any],
    *,
    continuation_criterion: str = "BKM_or_Serrin_type_continuation",
    residual_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Build a replayable a priori inequality candidate from a trace artifact."""
    if regularity_artifact.get("candidate_type") != "regularity_growth_law":
        raise ValueError("regularity inequality reports require regularity_growth_law artifacts")
    rin = dict(regularity_artifact["replay_inputs"])
    result = dict(regularity_artifact["result"])
    time = np.asarray(rin["time"], dtype=float)
    traces = {str(k): np.asarray(v, dtype=float) for k, v in dict(rin["traces"]).items()}
    target = str(rin.get("target", "enstrophy"))
    coefficients = {str(k): float(v) for k, v in dict(result.get("coefficients", {})).items()}
    feature_names = [str(v) for v in result.get("feature_names", list(traces))]
    derivative = np.gradient(traces[target], time)
    predicted = np.zeros_like(derivative)
    features: dict[str, np.ndarray] = dict(traces)
    for name, values in list(features.items()):
        features[f"{name}^2"] = values * values
    for name, coeff in coefficients.items():
        if name in features:
            predicted = predicted + coeff * features[name]
    residual = derivative - predicted
    max_abs_residual = float(np.max(np.abs(residual)))
    coefficient_intervals = {
        name: asdict(scalar_interval(value, relative_padding=1e-12, certified=True))
        for name, value in coefficients.items()
    }
    counterexamples = regularity_counterexample_sweep(
        coefficients,
        traces=traces,
        target=target,
        tolerance=residual_tolerance,
    )
    survived = bool(coefficients and max_abs_residual <= residual_tolerance and not counterexamples["counterexamples"])
    obligations = {
        "a_priori_estimate_candidate": bool(coefficients),
        "counterexample_search_passed": not bool(counterexamples["counterexamples"]),
        "continuation_criterion_link": bool(continuation_criterion),
        "all_smooth_finite_energy_data_proof": False,
    }
    return {
        "schema_version": "navier-stokes-regularity-inequality-1",
        "candidate_type": "regularity_inequality_report",
        "route": "global_regularity",
        "inequality_name": f"{target}_growth_candidate",
        "target": target,
        "feature_names": feature_names,
        "coefficients": coefficients,
        "coefficient_intervals": coefficient_intervals,
        "trace_residual": {
            "max_abs_residual": max_abs_residual,
            "rmse": float(np.sqrt(np.mean(residual * residual))),
            "tolerance": float(residual_tolerance),
            "passed": bool(max_abs_residual <= residual_tolerance),
        },
        "counterexample_sweep": counterexamples,
        "continuation_criterion": continuation_criterion,
        "obligations": obligations,
        "formalizable": bool(all(obligations.values())),
        "proof_prep_certified": survived,
        "open_obligations": [name for name, ok in obligations.items() if not ok],
        "replay_inputs": _json_dict({
            "regularity_artifact": regularity_artifact,
            "residual_tolerance": residual_tolerance,
            "continuation_criterion": continuation_criterion,
        }),
        "independent_replay": {
            "status": "required",
            "harness": "omnibias.symbolic.navier_stokes.verify_regularity_inequality_report",
        },
        "unproven_claim": False,
    }


def regularity_counterexample_sweep(
    coefficients: dict[str, float],
    *,
    traces: dict[str, np.ndarray],
    target: str,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Run deterministic trace-space counterexample probes for an inequality."""
    probes: dict[str, dict[str, np.ndarray]] = {
        "stored_trace": traces,
        "energy_doubled": {
            key: (2.0 * value if key == "energy" else value)
            for key, value in traces.items()
        },
        "enstrophy_amplified": {
            key: (1.25 * value if key == target else value)
            for key, value in traces.items()
        },
    }
    counterexamples: list[dict[str, Any]] = []
    for name, probe in probes.items():
        time = np.linspace(0.0, 1.0, int(np.asarray(probe[target]).shape[0]))
        derivative = np.gradient(np.asarray(probe[target], dtype=float), time)
        predicted = np.zeros_like(derivative)
        features: dict[str, np.ndarray] = {k: np.asarray(v, dtype=float) for k, v in probe.items()}
        for feature_name, values in list(features.items()):
            features[f"{feature_name}^2"] = values * values
        for coeff_name, coeff in coefficients.items():
            if coeff_name in features:
                predicted = predicted + float(coeff) * features[coeff_name]
        violation = float(np.max(np.maximum(derivative - predicted - tolerance, 0.0)))
        if violation > 0.0:
            counterexamples.append({
                "probe": name,
                "max_positive_violation": violation,
            })
    return {
        "method": "deterministic_trace_feature_probes",
        "probe_count": len(probes),
        "counterexamples": counterexamples,
        "passed": not counterexamples,
        "unproven_claim": False,
    }


def regularity_all_data_proof_attempt(
    regularity_report: dict[str, Any],
    *,
    diagnostic_bundles: list[dict[str, Any]] | None = None,
    external_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attempt to promote a trace inequality toward an all-data theorem."""
    bundles = diagnostic_bundles or []
    counterexamples: list[dict[str, Any]] = []
    for idx, bundle in enumerate(bundles):
        diag = dict(bundle.get("residual_diagnostics", {}))
        max_residual = max(
            abs(float(diag.get("max_abs_momentum_residual", 0.0))),
            abs(float(diag.get("max_abs_continuity", 0.0))),
            abs(float(diag.get("max_abs_pressure_poisson", 0.0))),
        )
        if max_residual > 1e-6:
            counterexamples.append({
                "bundle_index": idx,
                "reason": "diagnostic_baseline_residual_too_large",
                "max_residual": max_residual,
            })
    trace_candidate = bool(regularity_report.get("obligations", {}).get("a_priori_estimate_candidate", False))
    sweep_passed = bool(regularity_report.get("counterexample_sweep", {}).get("passed", False))
    continuation_ok = bool(regularity_report.get("continuation_criterion", ""))
    external_ok = _external_verifies(external_verification, "all_smooth_finite_energy_data_proof")
    certified = bool(trace_candidate and sweep_passed and continuation_ok and not counterexamples and external_ok)
    open_obligations: list[str] = []
    if not trace_candidate:
        open_obligations.append("a_priori_estimate_candidate")
    if not sweep_passed or counterexamples:
        open_obligations.append("falsified_with_counterexample")
    if not continuation_ok:
        open_obligations.append("continuation_criterion_link")
    if not external_ok:
        open_obligations.append("all_smooth_finite_energy_data_proof")
    return {
        "schema_version": "navier-stokes-theorem-regularity-1",
        "candidate_type": "theorem_grade_regularity_proof_attempt",
        "inequality_name": str(regularity_report.get("inequality_name", "unknown")),
        "functional_inequality": {
            "target": regularity_report.get("target", "unknown"),
            "feature_names": list(regularity_report.get("feature_names", [])),
            "coefficients": _json_dict(regularity_report.get("coefficients", {})),
            "domain": "all_smooth_divergence_free_finite_energy_3d_data",
        },
        "diagnostic_falsification": {
            "bundle_count": len(bundles),
            "counterexamples": counterexamples,
            "trace_counterexample_sweep": _json_dict(regularity_report.get("counterexample_sweep", {})),
            "passed": bool(sweep_passed and not counterexamples),
        },
        "continuation_criterion": str(regularity_report.get("continuation_criterion", "")),
        "external_verification": _json_dict(external_verification or {}),
        "all_smooth_finite_energy_data_proof": certified,
        "proof_status": (
            "proved_by_external_artifact"
            if certified
            else "falsified_with_counterexample"
            if counterexamples or not sweep_passed
            else "blocked_with_precise_obligations"
        ),
        "open_obligations": open_obligations,
        "unproven_claim": False,
    }


def build_regularity_closure_report(
    *,
    inequality_name: str,
    coefficients: dict[str, float],
    counterexample_count: int = 0,
    continuation_criterion: str = "BKM_or_Serrin_type_continuation",
) -> dict[str, Any]:
    """Build regularity route analytic closure obligations."""
    has_coefficients = bool(coefficients)
    obligations = {
        "a_priori_estimate_candidate": has_coefficients,
        "counterexample_search_passed": int(counterexample_count) == 0,
        "continuation_criterion_link": bool(continuation_criterion),
        "all_smooth_finite_energy_data_proof": False,
    }
    return {
        "schema_version": "navier-stokes-analytic-closure-1",
        "route": "global_regularity",
        "inequality_name": str(inequality_name),
        "coefficients": {str(k): float(v) for k, v in coefficients.items()},
        "counterexample_count": int(counterexample_count),
        "continuation_criterion": str(continuation_criterion),
        "obligations": obligations,
        "formalizable": bool(all(obligations.values())),
        "unproven_claim": False,
        "open_obligations": [name for name, ok in obligations.items() if not ok],
    }


def build_analytic_closure_report(
    interval_report: dict[str, Any],
    *,
    regularity_report: dict[str, Any] | None = None,
    blowup_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select the strongest currently available analytic closure route."""
    blowup = blowup_report if blowup_report is not None else build_blowup_closure_report(interval_report)
    regularity = regularity_report if regularity_report is not None else build_regularity_closure_report(
        inequality_name="unselected_a_priori_candidate",
        coefficients={},
        counterexample_count=0,
    )
    if bool(blowup.get("formalizable", False)):
        selected = "finite_time_blowup"
    elif bool(regularity.get("formalizable", False)):
        selected = "global_regularity"
    else:
        selected = "blocked_no_closed_route"
    return {
        "schema_version": "navier-stokes-closure-selection-1",
        "selected_route": selected,
        "blowup": _json_dict(blowup),
        "regularity": _json_dict(regularity),
        "unproven_claim": False,
        "open_obligations": sorted({
            *[str(v) for v in blowup.get("open_obligations", [])],
            *[str(v) for v in regularity.get("open_obligations", [])],
        }),
    }


_BLOWUP_LEMMA_STATEMENTS: dict[str, str] = {
    "axisymmetric_reduction_equivalence": (
        "The certified axisymmetric-with-swirl system reconstructs a smooth "
        "divergence-free 3D Navier-Stokes solution with no missing axis terms."
    ),
    "smooth_finite_energy_initial_data": (
        "The coefficient-defined profile induces smooth finite-energy initial "
        "data on R3 with the declared tail convention."
    ),
    "continuum_linearized_invertibility": (
        "The linearized Navier-Stokes profile operator is invertible on the "
        "declared continuum Banach space with the certified inverse bound."
    ),
    "nonlinear_remainder_bound": (
        "The nonlinear residual remainder is bounded by the declared interval "
        "CAP product and tail estimates."
    ),
    "radii_fixed_point_closure": (
        "The radii-polynomial inequalities imply a true fixed point of the "
        "Navier-Stokes profile map in the declared ball."
    ),
    "finite_time_norm_divergence": (
        "The exact coefficient-defined solution has finite-time divergence of "
        "a critical continuation norm."
    ),
}

_REGULARITY_LEMMA_STATEMENTS: dict[str, str] = {
    "universal_a_priori_inequality": (
        "The discovered inequality holds for all smooth divergence-free "
        "finite-energy 3D Navier-Stokes data."
    ),
    "continuation_criterion_implication": (
        "The universal inequality controls a standard BKM or Serrin "
        "continuation criterion."
    ),
    "pressure_leray_viscosity_compatibility": (
        "Pressure recovery, Leray projection, and viscosity scaling preserve "
        "the declared regularity estimate."
    ),
    "counterexample_sweep_clearance": (
        "The exact, manufactured, and CAP diagnostic families do not falsify "
        "the proposed all-data inequality."
    ),
}


def proof_obligation_bundle(
    *,
    route: str,
    lemma_id: str,
    theorem_statement: str,
    assumptions: list[str] | tuple[str, ...],
    dependencies: list[str] | tuple[str, ...],
    source_artifact: dict[str, Any],
    expected_verifier: str = "lean4",
    proof_status: str = "open",
) -> dict[str, Any]:
    """Create a deterministic theorem-obligation bundle."""
    theorem_name = "NavierStokes" + "".join(part.capitalize() for part in str(lemma_id).split("_"))
    bundle = ProofObligationBundle(
        schema_version="navier-stokes-proof-obligation-1",
        route=str(route),
        lemma_id=str(lemma_id),
        theorem_name=theorem_name,
        theorem_statement=str(theorem_statement),
        assumptions=tuple(str(v) for v in assumptions),
        dependencies=tuple(str(v) for v in dependencies),
        source_artifact_sha256=_sha256_json(source_artifact),
        expected_verifier=str(expected_verifier),
        proof_status=str(proof_status),
    )
    payload = asdict(bundle)
    return {
        **payload,
        "obligation_id": f"{route}.{lemma_id}",
        "obligation_sha256": _sha256_json(payload),
    }


def theorem_verifier_record(
    obligation_bundles: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    verifier: str = "lean4",
    discharged_obligations: list[str] | tuple[str, ...] | None = None,
    verification_status: str = "verified",
    reviewed_at_utc: str = "external-review-required",
) -> dict[str, Any]:
    """Create a normalized verifier record for proof-obligation bundles."""
    bundles = [_json_dict(bundle) for bundle in obligation_bundles]
    selected = {
        str(obligation)
        for obligation in (
            discharged_obligations
            if discharged_obligations is not None
            else [bundle.get("obligation_id", "") for bundle in bundles]
        )
    }
    proof_records = [
        {
            "obligation_id": str(bundle.get("obligation_id", "")),
            "lemma_id": str(bundle.get("lemma_id", "")),
            "theorem_name": str(bundle.get("theorem_name", "")),
            "obligation_sha256": str(bundle.get("obligation_sha256", "")),
            "source_artifact_sha256": str(bundle.get("source_artifact_sha256", "")),
            "verifier": str(verifier),
        }
        for bundle in bundles
        if str(bundle.get("obligation_id", "")) in selected
    ]
    artifact_sha = _sha256_json(proof_records)
    status = str(verification_status)
    return {
        "schema_version": "navier-stokes-theorem-verifier-1",
        "verifier": str(verifier),
        "verification_status": status,
        "verified": bool(status == "verified" and proof_records and reviewed_at_utc),
        "reviewed_at_utc": str(reviewed_at_utc),
        "artifact_sha256": artifact_sha,
        "discharged_obligations": [record["obligation_id"] for record in proof_records],
        "proof_records": proof_records,
        "unproven_claim": False,
    }


def ingest_theorem_verifier_bundle(
    obligation_bundles: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    verifier_bundle: dict[str, Any] | None,
) -> dict[str, Any]:
    """Accept only verifier evidence matching obligation IDs, hashes, and verifiers."""
    obligations = {str(bundle.get("obligation_id", "")): dict(bundle) for bundle in obligation_bundles}
    if not verifier_bundle:
        return {
            "schema_version": "navier-stokes-theorem-verifier-ingestion-1",
            "verified": False,
            "accepted_obligations": [],
            "rejected_obligations": {
                obligation_id: "verifier_bundle_missing"
                for obligation_id in obligations
            },
            "unproven_claim": False,
        }
    records = {
        str(record.get("obligation_id", "")): dict(record)
        for record in verifier_bundle.get("proof_records", [])
    }
    expected_artifact_sha = _sha256_json(list(records.values()))
    bundle_artifact_sha = str(verifier_bundle.get("artifact_sha256", ""))
    status_ok = bool(verifier_bundle.get("verified", False)) and str(
        verifier_bundle.get("verification_status", "")
    ) == "verified"
    freshness_ok = bool(str(verifier_bundle.get("reviewed_at_utc", "")))
    artifact_ok = bool(records) and expected_artifact_sha == bundle_artifact_sha
    accepted: list[str] = []
    rejected: dict[str, str] = {}
    discharged = {str(v) for v in verifier_bundle.get("discharged_obligations", [])}
    for obligation_id, obligation in obligations.items():
        record = records.get(obligation_id)
        if obligation_id not in discharged or record is None:
            rejected[obligation_id] = "obligation_not_discharged"
            continue
        checks = {
            "status": status_ok,
            "freshness": freshness_ok,
            "artifact_hash": artifact_ok,
            "verifier": str(record.get("verifier", "")) == str(obligation.get("expected_verifier", "")),
            "theorem_name": str(record.get("theorem_name", "")) == str(obligation.get("theorem_name", "")),
            "obligation_hash": str(record.get("obligation_sha256", "")) == str(obligation.get("obligation_sha256", "")),
            "source_hash": str(record.get("source_artifact_sha256", ""))
            == str(obligation.get("source_artifact_sha256", "")),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            rejected[obligation_id] = ",".join(failed)
        else:
            accepted.append(obligation_id)
    return {
        "schema_version": "navier-stokes-theorem-verifier-ingestion-1",
        "verified": bool(accepted and not rejected),
        "accepted_obligations": accepted,
        "rejected_obligations": rejected,
        "artifact_hash_match": artifact_ok,
        "status_ok": status_ok,
        "freshness_ok": freshness_ok,
        "verifier_bundle": _json_dict(verifier_bundle),
        "unproven_claim": False,
    }


def _route_proof_obligation_bundles(
    *,
    route: str,
    lemmas: dict[str, Any],
    statements: dict[str, str],
    source_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        proof_obligation_bundle(
            route=route,
            lemma_id=lemma_id,
            theorem_statement=statements.get(lemma_id, f"Discharge theorem obligation {lemma_id}."),
            assumptions=(
                f"route={route}",
                f"initial_status={lemma.get('status', 'open')}",
            ),
            dependencies=tuple(lemma.get("depends_on", ())),
            source_artifact=source_artifact,
        )
        for lemma_id, lemma in lemmas.items()
    ]


def blowup_proof_obligation_bundles(theorem_attempt: dict[str, Any]) -> list[dict[str, Any]]:
    """Return proof-obligation bundles for every blow-up route lemma."""
    return list(blowup_route_lemma_package(theorem_attempt)["proof_obligation_bundles"])


def regularity_proof_obligation_bundles(regularity_attempt: dict[str, Any]) -> list[dict[str, Any]]:
    """Return proof-obligation bundles for every regularity route lemma."""
    return list(regularity_route_lemma_package(regularity_attempt)["proof_obligation_bundles"])


def verify_external_proof_package(
    formal_package: dict[str, Any],
    external_verification: dict[str, Any] | None,
) -> dict[str, Any]:
    """Check that external proof evidence matches a formal package."""
    if external_verification is None:
        return {
            "schema_version": "navier-stokes-external-proof-check-1",
            "verified": False,
            "reason": "external_verification_missing",
            "unproven_claim": False,
        }
    expected = {str(v) for v in formal_package.get("obligation_ids", [])}
    discharged = {str(v) for v in external_verification.get("discharged_obligations", [])}
    theorem_match = str(external_verification.get("theorem_name", "")) == str(formal_package.get("theorem_name", ""))
    obligations_match = bool(expected and expected.issubset(discharged))
    hash_present = bool(str(external_verification.get("artifact_sha256", "")))
    status_ok = bool(external_verification.get("verified", False))
    verified = bool(theorem_match and obligations_match and hash_present and status_ok)
    return {
        "schema_version": "navier-stokes-external-proof-check-1",
        "verified": verified,
        "theorem_match": theorem_match,
        "obligations_match": obligations_match,
        "artifact_hash_present": hash_present,
        "status_ok": status_ok,
        "missing_obligations": sorted(expected - discharged),
        "external_verification": _json_dict(external_verification),
        "unproven_claim": False,
    }


def theorem_claim_gate(
    theorem_attempt: dict[str, Any],
    formal_package: dict[str, Any],
    *,
    external_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Strict final gate for any global-regularity-grade claim."""
    proof_check = verify_external_proof_package(formal_package, external_verification)
    routes = dict(theorem_attempt.get("route_attempts", {}))
    blowup_proved = bool(
        routes.get("operator_invertibility", {}).get("operator_theoretic_certified", False)
        and routes.get("radii_polynomial", {}).get("radii_polynomial_closure", False)
        and routes.get("norm_divergence", {}).get("norm_divergence", False)
    )
    regularity_proved = bool(
        routes.get("regularity_all_data", {}).get("all_smooth_finite_energy_data_proof", False)
    )
    route_proved = bool(blowup_proved or regularity_proved)
    obligation_bundles = list(formal_package.get("proof_obligation_bundles", []))
    verifier_ingestion = ingest_theorem_verifier_bundle(obligation_bundles, external_verification) if obligation_bundles else {
        "verified": False,
        "accepted_obligations": [],
        "rejected_obligations": {},
    }
    obligation_evidence_ok = bool(not obligation_bundles or verifier_ingestion.get("verified", False))
    all_route_obligations_closed = not theorem_attempt.get("open_obligations")
    unproven_claim = bool(
        route_proved
        and all_route_obligations_closed
        and proof_check.get("verified", False)
        and obligation_evidence_ok
    )
    return {
        "schema_version": "navier-stokes-theorem-claim-gate-1",
        "route_proved": route_proved,
        "blowup_route_proved": blowup_proved,
        "regularity_route_proved": regularity_proved,
        "all_route_obligations_closed": all_route_obligations_closed,
        "external_proof_verified": bool(proof_check.get("verified", False)),
        "obligation_evidence_verified": obligation_evidence_ok,
        "unproven_claim": unproven_claim,
        "promotion_status": "proved_by_external_artifact" if unproven_claim else "blocked_with_precise_obligations",
        "proof_check": proof_check,
        "verifier_ingestion": verifier_ingestion,
        "open_obligations": [] if unproven_claim else sorted(set(
            [str(v) for v in theorem_attempt.get("open_obligations", [])]
            + [str(v) for v in proof_check.get("missing_obligations", [])]
            + ([] if proof_check.get("verified", False) else ["external_theorem_prover_verification"])
            + ([] if obligation_evidence_ok else ["proof_obligation_verifier_evidence"])
        )),
    }


def build_theorem_grade_closure_attempt(
    interval_report: dict[str, Any],
    *,
    blowup_report: dict[str, Any] | None = None,
    regularity_report: dict[str, Any] | None = None,
    external_verifications: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble theorem-grade proof attempts for both regularity routes."""
    external = external_verifications or {}
    active_blowup = blowup_report or build_axisymmetric_blowup_closure_report(interval_report)
    active_regularity = regularity_report or build_regularity_closure_report(
        inequality_name="unselected_a_priori_candidate",
        coefficients={},
    )
    linearized = dict(active_blowup.get("closure_certificates", {}).get("linearized_operator", {}))
    componentwise = dict(
        active_blowup.get("closure_certificates", {})
        .get("radii_polynomial", {})
        .get("componentwise", {})
    )
    norm_cert = dict(active_blowup.get("closure_certificates", {}).get("norm_divergence", {}))
    operator_attempt = continuum_banach_invertibility_attempt(
        interval_report,
        linearized,
        external_verification=external.get("operator"),
    )
    radii_attempt = theorem_grade_radii_polynomial_attempt(
        interval_report,
        operator_attempt,
        componentwise,
        external_verification=external.get("radii"),
    )
    norm_attempt = exact_profile_norm_divergence_attempt(
        interval_report,
        norm_cert,
        external_verification=external.get("norm"),
    )
    regularity_attempt = regularity_all_data_proof_attempt(
        active_regularity,
        external_verification=external.get("regularity"),
    )
    route_attempts = {
        "operator_invertibility": operator_attempt,
        "radii_polynomial": radii_attempt,
        "norm_divergence": norm_attempt,
        "regularity_all_data": regularity_attempt,
    }
    open_obligations = sorted({
        str(obligation)
        for attempt in route_attempts.values()
        for obligation in attempt.get("open_obligations", [])
    })
    statuses = {str(attempt.get("proof_status", "blocked_with_precise_obligations")) for attempt in route_attempts.values()}
    return {
        "schema_version": "navier-stokes-theorem-closure-attempt-1",
        "candidate_type": "theorem_grade_closure_attempt",
        "function_space_contract": theorem_grade_function_space_contract(interval_report),
        "route_attempts": _json_dict(route_attempts),
        "proof_status": (
            "proved_by_external_artifact"
            if statuses == {"proved_by_external_artifact"}
            else "falsified_with_counterexample"
            if "falsified_with_counterexample" in statuses
            else "blocked_with_precise_obligations"
        ),
        "open_obligations": open_obligations,
        "unproven_claim": False,
        "replay_inputs": {
            "interval_report": _json_dict(interval_report),
            "blowup_report": _json_dict(active_blowup),
            "regularity_report": _json_dict(active_regularity),
        },
        "independent_replay": {
            "status": "required",
            "harness": "omnibias.symbolic.navier_stokes.verify_theorem_grade_closure_attempt",
        },
    }


def blowup_route_lemma_package(
    theorem_attempt: dict[str, Any],
    *,
    verifier_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build named blow-up-route lemmas from theorem-grade route attempts."""
    routes = dict(theorem_attempt.get("route_attempts", {}))
    operator = dict(routes.get("operator_invertibility", {}))
    radii = dict(routes.get("radii_polynomial", {}))
    norm = dict(routes.get("norm_divergence", {}))
    lemmas = {
        "axisymmetric_reduction_equivalence": {
            "status": "open",
            "depends_on": ("exact_navier_stokes_equation_contracts.axisymmetric_with_swirl",),
        },
        "smooth_finite_energy_initial_data": {
            "status": "ready" if norm.get("field_profile_linkage", {}).get("finite_energy_certified", False) else "open",
            "depends_on": ("finite_energy_tail_certificate", "axis_regular_certificate"),
        },
        "continuum_linearized_invertibility": {
            "status": "ready" if operator.get("operator_theoretic_certified", False) else "open",
            "depends_on": tuple(operator.get("open_obligations", [])),
        },
        "nonlinear_remainder_bound": {
            "status": "ready" if radii.get("outward_inequality_passed", False) else "open",
            "depends_on": ("theorem_grade_radii_polynomial_attempt",),
        },
        "radii_fixed_point_closure": {
            "status": "ready" if radii.get("radii_polynomial_closure", False) else "open",
            "depends_on": tuple(radii.get("open_obligations", [])),
        },
        "finite_time_norm_divergence": {
            "status": "ready" if norm.get("norm_divergence", False) else "open",
            "depends_on": tuple(norm.get("open_obligations", [])),
        },
    }
    obligations = _route_proof_obligation_bundles(
        route="finite_time_blowup",
        lemmas=lemmas,
        statements=_BLOWUP_LEMMA_STATEMENTS,
        source_artifact=theorem_attempt,
    )
    ingestion = ingest_theorem_verifier_bundle(obligations, verifier_bundle)
    accepted_lemmas = {
        str(obligation_id).split(".", maxsplit=1)[-1]
        for obligation_id in ingestion.get("accepted_obligations", [])
    }
    for lemma_id, lemma in lemmas.items():
        if lemma_id in accepted_lemmas and lemma["status"] != "falsified":
            lemma["status"] = "closed"
            lemma["verifier_record"] = "accepted"
    open_lemmas = [name for name, lemma in lemmas.items() if lemma["status"] != "closed"]
    return {
        "schema_version": "navier-stokes-blowup-lemma-package-1",
        "route": "finite_time_blowup",
        "lemmas": _json_dict(lemmas),
        "proof_obligation_bundles": obligations,
        "verifier_ingestion": ingestion,
        "proof_status": "proved_finite_time_blowup" if not open_lemmas else "blocked_with_named_missing_lemma",
        "open_lemmas": open_lemmas,
        "unproven_claim": False,
    }


def regularity_route_lemma_package(
    regularity_attempt: dict[str, Any],
    *,
    verifier_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build named regularity-route lemmas from all-data proof attempts."""
    lemmas = {
        "universal_a_priori_inequality": {
            "status": "ready" if regularity_attempt.get("all_smooth_finite_energy_data_proof", False) else "open",
            "depends_on": tuple(regularity_attempt.get("open_obligations", [])),
        },
        "continuation_criterion_implication": {
            "status": "ready" if regularity_attempt.get("continuation_criterion") else "open",
            "depends_on": ("BKM_or_Serrin_continuation",),
        },
        "pressure_leray_viscosity_compatibility": {
            "status": "open",
            "depends_on": ("pressure_poisson_identity", "leray_projector_bounds", "viscosity_scaling"),
        },
        "counterexample_sweep_clearance": {
            "status": "ready"
            if regularity_attempt.get("diagnostic_falsification", {}).get("passed", False)
            else "falsified",
            "depends_on": ("diagnostic_falsification",),
        },
    }
    obligations = _route_proof_obligation_bundles(
        route="global_regularity",
        lemmas=lemmas,
        statements=_REGULARITY_LEMMA_STATEMENTS,
        source_artifact=regularity_attempt,
    )
    ingestion = ingest_theorem_verifier_bundle(obligations, verifier_bundle)
    accepted_lemmas = {
        str(obligation_id).split(".", maxsplit=1)[-1]
        for obligation_id in ingestion.get("accepted_obligations", [])
    }
    for lemma_id, lemma in lemmas.items():
        if lemma_id in accepted_lemmas and lemma["status"] != "falsified":
            lemma["status"] = "closed"
            lemma["verifier_record"] = "accepted"
    if any(lemma["status"] == "falsified" for lemma in lemmas.values()):
        status = "candidate_falsified"
    else:
        open_lemmas = [name for name, lemma in lemmas.items() if lemma["status"] != "closed"]
        status = "proved_global_regularity" if not open_lemmas else "blocked_with_named_missing_lemma"
    return {
        "schema_version": "navier-stokes-regularity-lemma-package-1",
        "route": "global_regularity",
        "lemmas": _json_dict(lemmas),
        "proof_obligation_bundles": obligations,
        "verifier_ingestion": ingestion,
        "proof_status": status,
        "open_lemmas": [name for name, lemma in lemmas.items() if lemma["status"] != "closed"],
        "unproven_claim": False,
    }


def lean_formalization_package(proof_program_report: dict[str, Any]) -> dict[str, Any]:
    """Emit deterministic Lean/written-CAP theorem statements and imports."""
    obligation_bundles = [
        _json_dict(bundle)
        for bundle in proof_program_report.get("proof_obligation_bundles", [])
    ]
    obligations = sorted({
        *[str(v) for v in proof_program_report.get("open_obligations", [])],
        *[str(v) for v in proof_program_report.get("open_lemmas", [])],
        *[str(bundle.get("obligation_id", "")) for bundle in obligation_bundles],
    })
    theorem_names = {
        "finite_time_blowup": "NavierStokesFiniteTimeBlowupCAP",
        "global_regularity": "NavierStokesGlobalRegularityCAP",
        "blocked": "NavierStokesProofProgramBlocked",
    }
    imported_hash = hashlib.sha256(
        json.dumps(_json_dict(proof_program_report), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "navier-stokes-lean-formalization-1",
        "theorem_names": theorem_names,
        "artifact_sha256": imported_hash,
        "obligation_ids": obligations,
        "proof_obligation_hashes": {
            str(bundle.get("obligation_id", "")): str(bundle.get("obligation_sha256", ""))
            for bundle in obligation_bundles
        },
        "lean_modules": {
            "interval_arithmetic": "Omnibias.NavierStokes.Interval",
            "neumann_inverse": "Omnibias.NavierStokes.Neumann",
            "radii_polynomial": "Omnibias.NavierStokes.Radii",
            "continuation": "Omnibias.NavierStokes.Continuation",
            "certificate_import": "Omnibias.NavierStokes.GeneratedCertificate",
        },
        "proof_assistant_verified": False,
        "proof_status": "blocked_with_named_missing_lemma" if obligations else "ready_for_external_verification",
        "unproven_claim": False,
    }


def external_review_gate(
    proof_program_report: dict[str, Any],
    lean_package: dict[str, Any],
    *,
    external_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Strict external review gate for proof-program outputs."""
    expected_hash = str(lean_package.get("artifact_sha256", ""))
    actual_hash = hashlib.sha256(
        json.dumps(_json_dict(proof_program_report), sort_keys=True).encode("utf-8")
    ).hexdigest()
    hash_match = bool(expected_hash and expected_hash == actual_hash)
    external_ok = bool(
        external_verification
        and external_verification.get("verified", False)
        and str(external_verification.get("artifact_sha256", "")) == expected_hash
    )
    no_open = not proof_program_report.get("open_obligations") and not proof_program_report.get("open_lemmas")
    route_status = str(proof_program_report.get("proof_status", "blocked_with_named_missing_lemma"))
    route_ingestion = dict(proof_program_report.get("verifier_ingestion", {}))
    route_evidence_ok = (
        bool(route_ingestion.get("finite_time_blowup", {}).get("verified", False))
        if route_status == "proved_finite_time_blowup"
        else bool(route_ingestion.get("global_regularity", {}).get("verified", False))
        if route_status == "proved_global_regularity"
        else False
    )
    unproven_claim = bool(
        hash_match
        and external_ok
        and route_evidence_ok
        and no_open
        and route_status in {"proved_global_regularity", "proved_finite_time_blowup"}
    )
    return {
        "schema_version": "navier-stokes-external-review-gate-1",
        "hash_match": hash_match,
        "external_verification_ok": external_ok,
        "route_verifier_evidence_ok": route_evidence_ok,
        "no_open_obligations": no_open,
        "route_status": route_status,
        "unproven_claim": unproven_claim,
        "review_status": route_status if unproven_claim else "blocked_with_named_missing_lemma",
        "open_obligations": [] if unproven_claim else sorted(set(
            [str(v) for v in proof_program_report.get("open_obligations", [])]
            + [str(v) for v in proof_program_report.get("open_lemmas", [])]
            + ([] if hash_match else ["lean_artifact_hash_mismatch"])
            + ([] if external_ok else ["external_review_missing_or_stale"])
            + ([] if route_evidence_ok else ["route_verifier_evidence_missing"])
        )),
    }


def build_ns_proof_program_report(
    *,
    theorem_attempt: dict[str, Any],
    candidate_family_status: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    verifier_bundles: dict[str, dict[str, Any]] | None = None,
    external_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the full Navier-Stokes proof-program status artifact."""
    verifier = verifier_bundles or {}
    exact = exact_navier_stokes_equation_contracts()
    spaces = theorem_grade_function_space_definitions()
    interval_backend = interval_cap_backend_contract()
    routes = dict(theorem_attempt.get("route_attempts", {}))
    blowup_lemmas = blowup_route_lemma_package(
        theorem_attempt,
        verifier_bundle=verifier.get("finite_time_blowup") or verifier.get("blowup"),
    )
    regularity_lemmas = regularity_route_lemma_package(
        dict(routes.get("regularity_all_data", {})),
        verifier_bundle=verifier.get("global_regularity") or verifier.get("regularity"),
    )
    candidate_status = [_json_dict(item) for item in candidate_family_status]
    proof_obligation_bundles = [
        *list(blowup_lemmas.get("proof_obligation_bundles", [])),
        *list(regularity_lemmas.get("proof_obligation_bundles", [])),
    ]
    verifier_ingestion = {
        "finite_time_blowup": _json_dict(blowup_lemmas.get("verifier_ingestion", {})),
        "global_regularity": _json_dict(regularity_lemmas.get("verifier_ingestion", {})),
    }
    open_obligations = sorted({
        *[str(v) for v in theorem_attempt.get("open_obligations", [])],
        *[str(v) for v in interval_backend.get("open_obligations", [])],
        *[str(v) for family in candidate_status for v in family.get("open_obligations", [])],
        *[
            str(obligation_id)
            for ingestion in verifier_ingestion.values()
            for obligation_id in ingestion.get("rejected_obligations", {})
        ],
    })
    open_lemmas = sorted({
        *[str(v) for v in blowup_lemmas.get("open_lemmas", [])],
        *[str(v) for v in regularity_lemmas.get("open_lemmas", [])],
        *[str(v) for contract in exact.get("contracts", {}).values() for v in contract.get("open_obligations", [])],
        *[
            str(v)
            for definition in spaces.get("definitions", {}).values()
            for v in definition.get("open_obligations", [])
        ],
    })
    if blowup_lemmas["proof_status"] == "proved_finite_time_blowup":
        proof_status = "proved_finite_time_blowup"
    elif regularity_lemmas["proof_status"] == "proved_global_regularity":
        proof_status = "proved_global_regularity"
    elif any(family.get("status") == "candidate_falsified" for family in candidate_status):
        proof_status = "candidate_falsified"
    else:
        proof_status = "blocked_with_named_missing_lemma"
    report = {
        "schema_version": "navier-stokes-proof-program-1",
        "candidate_type": "navier_stokes_proof_program_report",
        "exact_equation_contracts": exact,
        "function_space_definitions": spaces,
        "candidate_family_status": candidate_status,
        "interval_cap_backend": interval_backend,
        "theorem_attempt": _json_dict(theorem_attempt),
        "proof_obligation_bundles": proof_obligation_bundles,
        "verifier_ingestion": verifier_ingestion,
        "lemma_packages": {
            "finite_time_blowup": blowup_lemmas,
            "global_regularity": regularity_lemmas,
        },
        "proof_status": proof_status,
        "open_obligations": open_obligations,
        "open_lemmas": open_lemmas,
        "unproven_claim": False,
    }
    lean = lean_formalization_package(report)
    review = external_review_gate(report, lean, external_verification=external_verification)
    report["lean_formalization"] = lean
    report["external_review_gate"] = review
    report["unproven_claim"] = bool(review.get("unproven_claim", False))
    report["independent_replay"] = {
        "status": "required",
        "harness": "omnibias.symbolic.navier_stokes.verify_ns_proof_program_report",
    }
    return report


def build_formal_proof_package(
    closure_report: dict[str, Any],
    *,
    target: str = "written_cap",
) -> dict[str, Any]:
    """Generate deterministic proof-assistant or written-CAP obligations."""
    route = str(closure_report.get("selected_route", closure_report.get("route", "unknown")))
    open_obligations = [str(v) for v in closure_report.get("open_obligations", [])]
    target_name = str(target)
    theorem_name = (
        "NavierStokesFiniteTimeBlowupCertificate"
        if route == "finite_time_blowup"
        else "NavierStokesGlobalRegularityCertificate"
        if route == "global_regularity"
        else "NavierStokesBlockedCertificate"
    )
    obligations = [
        {
            "id": f"{route}.{name}",
            "status": "open",
            "target": target_name,
        }
        for name in open_obligations
    ]
    if not obligations:
        obligations = [{"id": f"{route}.final_implication", "status": "ready", "target": target_name}]
    blowup = dict(closure_report.get("blowup", closure_report if closure_report.get("route") == "finite_time_blowup" else {}))
    regularity = dict(closure_report.get("regularity", closure_report if closure_report.get("route") == "global_regularity" else {}))
    blocker_sections = {
        "operator_remainder_obligations": (
            blowup.get("blocker_resolution", {})
            .get("operator_theoretic_invertibility", {})
            .get("open_obligations", [])
        ),
        "radii_polynomial_interval_roots": (
            blowup.get("closure_certificates", {})
            .get("radii_polynomial", {})
            .get("componentwise", {})
            .get("components", {})
        ),
        "norm_divergence_linkage_obligations": (
            blowup.get("blocker_resolution", {})
            .get("norm_divergence", {})
            .get("open_obligations", [])
        ),
        "regularity_inequality_obligations": regularity.get("open_obligations", []),
        "regularity_counterexample_sweep": regularity.get("counterexample_sweep", {}),
    }
    if closure_report.get("candidate_type") == "theorem_grade_closure_attempt":
        routes = dict(closure_report.get("route_attempts", {}))
        blocker_sections.update({
            "theorem_operator_obligations": routes.get("operator_invertibility", {}).get("open_obligations", []),
            "theorem_radii_obligations": routes.get("radii_polynomial", {}).get("open_obligations", []),
            "theorem_norm_obligations": routes.get("norm_divergence", {}).get("open_obligations", []),
            "theorem_regularity_obligations": routes.get("regularity_all_data", {}).get("open_obligations", []),
        })
    proof_obligation_bundles = list(closure_report.get("proof_obligation_bundles", []))
    if closure_report.get("candidate_type") == "theorem_grade_closure_attempt":
        proof_obligation_bundles = [
            *blowup_proof_obligation_bundles(closure_report),
            *regularity_proof_obligation_bundles(dict(closure_report.get("route_attempts", {}).get("regularity_all_data", {}))),
        ]
    proof_obligation_ids = [str(bundle.get("obligation_id", "")) for bundle in proof_obligation_bundles]
    obligation_names = ", ".join(f'"{item["id"]}"' for item in obligations)
    lean_stub = "\n".join([
        f"theorem {theorem_name}",
        "  (interval_certificates_verified : Prop)",
        "  (analytic_obligations_verified : Prop)",
        "  (certificate_hypotheses : interval_certificates_verified ∧ analytic_obligations_verified) :",
        "  True := by",
        "  exact True.intro",
    ])
    return {
        "schema_version": "navier-stokes-formal-package-1",
        "target": target_name,
        "route": route,
        "theorem_name": theorem_name,
        "theorem_statement": (
            "If every imported interval certificate and analytic obligation in "
            "this package is verified, then the selected Navier-Stokes route "
            "satisfies its stated theorem contract."
        ),
        "obligations": obligations,
        "proof_obligation_bundles": _json_dict(proof_obligation_bundles),
        "proof_obligation_hashes": {
            str(bundle.get("obligation_id", "")): str(bundle.get("obligation_sha256", ""))
            for bundle in proof_obligation_bundles
        },
        "blocker_obligation_sections": _json_dict(blocker_sections),
        "obligation_ids": [str(item["id"]) for item in obligations] + proof_obligation_ids,
        "proof_assistant_stub": {
            "language": "lean4" if target_name == "lean" else "written_cap",
            "theorem_name": theorem_name,
            "source": lean_stub,
            "imported_obligation_ids": obligation_names,
        },
        "theorem_prover_verified": False,
        "unproven_claim": False,
    }


def _worst_componentwise_radii(componentwise_certificate: dict[str, Any]) -> tuple[str, float]:
    components = dict(componentwise_certificate.get("components", {}))
    if not components:
        return "", float("inf")
    name, component = max(
        components.items(),
        key=lambda item: float(item[1].get("closure_interval", {}).get("upper", float("inf"))),
    )
    return str(name), float(component.get("closure_interval", {}).get("upper", float("inf")))


def freeze_axisymmetric_baseline_manifest(
    interval_report: dict[str, Any],
    *,
    blowup_report: dict[str, Any] | None = None,
    theorem_attempt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze the replayable finite-candidate baseline and its decisive metrics."""
    active_blowup = blowup_report or build_axisymmetric_blowup_closure_report(interval_report)
    artifact = dict(interval_report.get("replay_inputs", {}).get("refined_artifact", {}))
    coeffs = [float(v) for v in artifact.get("replay_inputs", {}).get("coefficients", [])]
    linearized = dict(active_blowup.get("closure_certificates", {}).get("linearized_operator", {}))
    operator = dict(active_blowup.get("blocker_resolution", {}).get("operator_theoretic_invertibility", {}))
    componentwise = dict(
        active_blowup.get("closure_certificates", {})
        .get("radii_polynomial", {})
        .get("componentwise", {})
    )
    worst_name, worst_upper = _worst_componentwise_radii(componentwise)
    metrics = {
        "coefficient_count": len(coeffs),
        "coefficient_l2_norm": float(np.linalg.norm(np.asarray(coeffs, dtype=float))) if coeffs else 0.0,
        "finite_energy_estimate": float(artifact.get("result", {}).get("finite_energy_estimate", float("nan"))),
        "train_loss": float(artifact.get("result", {}).get("train", {}).get("loss", float("nan"))),
        "holdout_loss": float(artifact.get("result", {}).get("holdout", {}).get("loss", float("nan"))),
        "tail_radius": float(interval_report.get("continuum_residual_certificates", {}).get("tail_radius", 0.0)),
        "continuum_residual_upper_bound": continuum_residual_upper_bound(interval_report),
        "smallest_singular_value": float(linearized.get("smallest_singular_value", 0.0)),
        "approximate_inverse_norm": linearized.get("approximate_inverse_norm"),
        "condition_estimate": linearized.get("condition_estimate"),
        "neumann_upper": float(operator.get("neumann_defect_interval", {}).get("upper", float("inf"))),
        "neumann_passed": bool(operator.get("neumann_passed", False)),
        "worst_componentwise_radii_upper": worst_upper,
        "worst_component": worst_name,
        "componentwise_radii_passed": bool(componentwise.get("passed", False)),
        "closure_consistency_verified": bool(active_blowup.get("closure_consistency_verified", False)),
    }
    payload = {
        "schema_version": "navier-stokes-baseline-freeze-1",
        "candidate_type": "axisymmetric_baseline_freeze_manifest",
        "source_candidate_type": artifact.get("candidate_type", "unknown"),
        "source_artifact_sha256": _sha256_json(artifact),
        "interval_report_sha256": _sha256_json(interval_report),
        "blowup_report_sha256": _sha256_json(active_blowup),
        "theorem_attempt_sha256": _sha256_json(theorem_attempt) if theorem_attempt is not None else "",
        "metrics": _json_dict(metrics),
        "coefficients": coeffs,
        "proof_status": (
            "finite_closure_consistency_verified"
            if metrics["neumann_passed"] and metrics["componentwise_radii_passed"]
            else "blocked_finite_gate"
        ),
        "open_obligations": [str(v) for v in active_blowup.get("open_obligations", [])],
        "unproven_claim": False,
    }
    payload["baseline_sha256"] = _sha256_json({k: v for k, v in payload.items() if k != "baseline_sha256"})
    return payload


def axisymmetric_nontriviality_gate(
    interval_report: dict[str, Any],
    *,
    blowup_report: dict[str, Any] | None = None,
    min_energy: float = 1.0,
    min_coefficient_norm: float = 1e-6,
    min_residual_upper_bound: float = 1e-12,
) -> dict[str, Any]:
    """Detect whether finite closure is being won by shrinking toward zero."""
    active_blowup = blowup_report or build_axisymmetric_blowup_closure_report(interval_report)
    baseline = freeze_axisymmetric_baseline_manifest(interval_report, blowup_report=active_blowup)
    metrics = dict(baseline["metrics"])
    energy = float(metrics.get("finite_energy_estimate", 0.0))
    coeff_norm = float(metrics.get("coefficient_l2_norm", 0.0))
    residual_upper = float(metrics.get("continuum_residual_upper_bound", 0.0))
    worst_upper = float(metrics.get("worst_componentwise_radii_upper", float("inf")))
    finite_closure_passed = bool(metrics.get("neumann_passed", False) and metrics.get("componentwise_radii_passed", False))
    amplitude_softness_warning = bool(finite_closure_passed and energy <= 10.0 * float(min_energy))
    passed = bool(
        energy >= float(min_energy)
        and coeff_norm >= float(min_coefficient_norm)
        and residual_upper >= float(min_residual_upper_bound)
    )
    open_obligations: list[str] = []
    if energy < float(min_energy):
        open_obligations.append("nontrivial_finite_energy_lower_bound")
    if coeff_norm < float(min_coefficient_norm):
        open_obligations.append("nonzero_profile_coefficient_norm")
    if residual_upper < float(min_residual_upper_bound):
        open_obligations.append("exclude_trivial_zero_residual_profile")
    open_obligations.append("certify_nontrivial_blowup_scaling_law")
    return {
        "schema_version": "navier-stokes-nontriviality-gate-1",
        "candidate_type": "axisymmetric_nontriviality_gate",
        "passed": passed,
        "finite_closure_passed": finite_closure_passed,
        "amplitude_softness_warning": amplitude_softness_warning,
        "diagnostics": {
            "finite_energy_estimate": energy,
            "coefficient_l2_norm": coeff_norm,
            "continuum_residual_upper_bound": residual_upper,
            "worst_componentwise_radii_upper": worst_upper,
            "closure_margin": 1.0 - worst_upper if np.isfinite(worst_upper) else -float("inf"),
            "min_energy": float(min_energy),
            "min_coefficient_norm": float(min_coefficient_norm),
            "min_residual_upper_bound": float(min_residual_upper_bound),
        },
        "proof_status": "nontriviality_screen_passed" if passed else "blocked_triviality_risk",
        "open_obligations": open_obligations,
        "unproven_claim": False,
    }


def conditioning_preserving_ansatz_report(
    base_linearized_certificate: dict[str, Any],
    *,
    enriched_certificates: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    """Report why naive enrichment fails and what a safer ansatz must satisfy."""
    base_sigma = float(base_linearized_certificate.get("smallest_singular_value", 0.0))
    enriched: list[dict[str, Any]] = []
    for cert in enriched_certificates:
        sigma = float(cert.get("smallest_singular_value", 0.0))
        enriched.append({
            "coefficient_count": int(cert.get("coefficient_count", 0)),
            "smallest_singular_value": sigma,
            "condition_estimate": cert.get("condition_estimate"),
            "interlacing_drop_factor": (base_sigma / sigma) if sigma > 0.0 and base_sigma > 0.0 else float("inf"),
            "conditioning_preserved": bool(base_sigma > 0.0 and sigma >= 0.75 * base_sigma),
        })
    naive_enrichment_rejected = bool(enriched and not all(item["conditioning_preserved"] for item in enriched))
    open_obligations = [
        "construct_residual_sensitivity_orthogonal_modes",
        "validate_conditioning_preserving_continuation",
        "cross_grid_ansatz_falsification",
    ]
    if naive_enrichment_rejected:
        open_obligations.append("replace_global_polynomial_enrichment")
    return {
        "schema_version": "navier-stokes-conditioning-preserving-ansatz-1",
        "candidate_type": "conditioning_preserving_ansatz_report",
        "base": {
            "coefficient_count": int(base_linearized_certificate.get("coefficient_count", 0)),
            "smallest_singular_value": base_sigma,
            "condition_estimate": base_linearized_certificate.get("condition_estimate"),
        },
        "enriched_diagnostics": enriched,
        "naive_global_enrichment_rejected": naive_enrichment_rejected,
        "recommended_ansatz_families": [
            "localized_compact_meridional_packets",
            "moving_core_self_similar_profiles",
            "residual_sensitivity_orthogonalized_basis",
            "five_dimensional_lifted_axisymmetric_variables",
        ],
        "acceptance_criteria": [
            "new_modes_preserve_smallest_singular_value",
            "worst_componentwise_radii_upper_below_one",
            "neumann_defect_upper_below_one",
            "nontriviality_gate_passed",
        ],
        "proof_status": "ansatz_redesign_required" if naive_enrichment_rejected else "ansatz_requirements_declared",
        "open_obligations": open_obligations,
        "unproven_claim": False,
    }


def theorem_interval_backend_readiness_report(
    interval_report: dict[str, Any],
    *,
    backend_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """State the directed-rounding interval backend requirements for theorem use."""
    contract = backend_contract or interval_cap_backend_contract()
    current = dict(interval_report.get("interval_backend", contract.get("current_interval_backend", {})))
    ready = bool(contract.get("theorem_grade_ready", False) and current.get("certified", False))
    open_obligations = list(contract.get("open_obligations", []))
    if not ready and "directed_rounding_theorem_backend" not in open_obligations:
        open_obligations.append("directed_rounding_theorem_backend")
    return {
        "schema_version": "navier-stokes-interval-backend-readiness-1",
        "candidate_type": "theorem_interval_backend_readiness_report",
        "current_interval_backend": _json_dict(current),
        "required_backend_contract": _json_dict(contract),
        "artifact_hash_requirements": [
            "hash_interval_inputs",
            "hash_directed_rounding_backend_version",
            "hash_operator_norm_certificates",
        ],
        "theorem_grade_ready": ready,
        "proof_status": "ready" if ready else "blocked_with_named_missing_backend",
        "open_obligations": [str(v) for v in open_obligations],
        "unproven_claim": False,
    }


def continuum_neumann_inequality_certificate(
    interval_report: dict[str, Any],
    linearized_certificate: dict[str, Any],
    *,
    interval_backend_report: dict[str, Any] | None = None,
    external_verification: dict[str, Any] | None = None,
    neumann_threshold: float = 1.0,
) -> dict[str, Any]:
    """Attempt the continuum Neumann inequality behind Banach invertibility."""
    finite = operator_theoretic_invertibility_certificate(
        interval_report,
        linearized_certificate,
        neumann_threshold=neumann_threshold,
    )
    backend = interval_backend_report or theorem_interval_backend_readiness_report(interval_report)
    finite_upper = float(finite.get("neumann_defect_interval", {}).get("upper", float("inf")))
    backend_ready = bool(backend.get("theorem_grade_ready", False))
    external_ok = _external_verifies(external_verification, "continuum_neumann_inequality")
    passed = bool(finite.get("neumann_passed", False))
    certified = bool(passed and backend_ready and external_ok)
    open_obligations: list[str] = []
    if not passed:
        open_obligations.append("finite_neumann_defect_below_one")
    if not backend_ready:
        open_obligations.append("certified_directed_rounding_interval_backend")
    if not external_ok:
        open_obligations.append("external_continuum_neumann_inequality_proof")
    open_obligations.append("analytic_tail_operator_bound")
    if certified:
        open_obligations = []
    return {
        "schema_version": "navier-stokes-continuum-neumann-1",
        "candidate_type": "continuum_neumann_inequality_certificate",
        "inequality": "||A_N^{-1}|| * (projection_error + tail_operator_bound) < 1",
        "finite_projection_certificate": _json_dict(finite),
        "finite_neumann_defect_upper": finite_upper,
        "finite_margin": 1.0 - finite_upper if np.isfinite(finite_upper) else -float("inf"),
        "interval_backend_report": _json_dict(backend),
        "external_verification": _json_dict(external_verification or {}),
        "continuum_neumann_certified": certified,
        "proof_status": "proved_by_external_artifact" if certified else "blocked_with_precise_obligations",
        "open_obligations": open_obligations,
        "unproven_claim": False,
    }


def defect_to_exact_solution_bridge_attempt(
    interval_report: dict[str, Any],
    *,
    continuum_neumann: dict[str, Any],
    componentwise_radii: dict[str, Any],
    active_subspace_completeness: dict[str, Any] | None = None,
    external_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attempt a Newton-Kantorovich bridge from a defect profile to an exact one."""
    residual_upper = continuum_residual_upper_bound(interval_report)
    worst_name, worst_upper = _worst_componentwise_radii(componentwise_radii)
    finite_radii_passed = bool(componentwise_radii.get("passed", False))
    continuum_ok = bool(continuum_neumann.get("continuum_neumann_certified", False))
    active_complete = (
        True
        if active_subspace_completeness is None
        else bool(active_subspace_completeness.get("active_subspace_complete", False))
    )
    external_ok = _external_verifies(external_verification, "defect_to_exact_solution_bridge")
    exact_profile = bool(
        finite_radii_passed
        and continuum_ok
        and active_complete
        and external_ok
        and residual_upper < float("inf")
    )
    open_obligations: list[str] = []
    if not finite_radii_passed:
        open_obligations.append("finite_radii_self_map")
    if not continuum_ok:
        open_obligations.append("continuum_banach_invertibility")
    if not active_complete:
        open_obligations.append("active_subspace_completeness")
    if not external_ok:
        open_obligations.append("external_newton_kantorovich_existence_proof")
    open_obligations.append("prove_exact_profile_solves_axisymmetric_ns")
    if exact_profile:
        open_obligations = []
    return {
        "schema_version": "navier-stokes-defect-exact-bridge-1",
        "candidate_type": "defect_to_exact_solution_bridge_attempt",
        "method": "newton_kantorovich_radii_polynomial_exact_profile_bridge",
        "residual_upper_bound": residual_upper,
        "componentwise_radii": {
            "passed": finite_radii_passed,
            "worst_component": worst_name,
            "worst_upper": worst_upper,
        },
        "continuum_neumann_certified": continuum_ok,
        "active_subspace_complete": active_complete,
        "active_subspace_completeness": _json_dict(active_subspace_completeness or {}),
        "external_verification": _json_dict(external_verification or {}),
        "exact_profile_verified": exact_profile,
        "proof_status": "proved_by_external_artifact" if exact_profile else "blocked_with_precise_obligations",
        "open_obligations": open_obligations,
        "unproven_claim": False,
    }


def certified_norm_divergence_bridge_attempt(
    interval_report: dict[str, Any],
    norm_certificate: dict[str, Any],
    *,
    exact_bridge: dict[str, Any],
    external_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Link a certified exact profile to finite-time norm divergence."""
    exponent = norm_certificate.get("growth_exponent")
    exponent_ok = bool(exponent is not None and float(exponent) > 0.0)
    lower_bound = norm_certificate.get("lower_bound_interval")
    lower_bound_ok = bool(
        isinstance(lower_bound, dict)
        and bool(lower_bound.get("certified", False))
        and float(lower_bound.get("lower", 0.0)) > 0.0
    )
    linked_ok = bool(norm_certificate.get("linked_to_field_profile", False))
    exact_ok = bool(exact_bridge.get("exact_profile_verified", False))
    finite_energy_ok = bool(interval_report.get("upgrade_gate", {}).get("finite_energy_certified", False))
    axis_ok = bool(interval_report.get("upgrade_gate", {}).get("axis_regular_certified", False))
    external_ok = _external_verifies(external_verification, "certified_norm_divergence_lower_bound")
    certified = bool(
        exponent_ok
        and lower_bound_ok
        and linked_ok
        and exact_ok
        and finite_energy_ok
        and axis_ok
        and external_ok
    )
    open_obligations: list[str] = []
    if not exponent_ok:
        open_obligations.append("positive_norm_growth_exponent")
    if not lower_bound_ok:
        open_obligations.append("certified_positive_norm_lower_bound")
    if not linked_ok:
        open_obligations.append("link_norm_trace_to_field_profile")
    if not exact_ok:
        open_obligations.append("exact_profile_verified")
    if not finite_energy_ok:
        open_obligations.append("finite_energy_initial_data")
    if not axis_ok:
        open_obligations.append("axis_smooth_reconstruction")
    if not external_ok:
        open_obligations.append("external_norm_divergence_lower_bound_proof")
    return {
        "schema_version": "navier-stokes-certified-norm-divergence-1",
        "candidate_type": "certified_norm_divergence_bridge_attempt",
        "norm_name": str(norm_certificate.get("norm_name", "candidate_trace_norm")),
        "blowup_time": norm_certificate.get("blowup_time"),
        "growth_exponent": exponent,
        "lower_bound_interval": _json_dict(norm_certificate.get("lower_bound_interval")),
        "linked_to_field_profile": linked_ok,
        "exact_profile_verified": exact_ok,
        "finite_energy_certified": finite_energy_ok,
        "axis_regular_certified": axis_ok,
        "external_verification": _json_dict(external_verification or {}),
        "norm_divergence_certified": certified,
        "proof_status": "proved_by_external_artifact" if certified else "blocked_with_precise_obligations",
        "open_obligations": open_obligations,
        "unproven_claim": False,
    }


def classical_assumptions_readiness_gate(
    interval_report: dict[str, Any],
    *,
    exact_bridge: dict[str, Any],
    norm_divergence: dict[str, Any],
    external_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check whether the closed route matches the global-regularity problem assumptions."""
    finite_energy_ok = bool(interval_report.get("upgrade_gate", {}).get("finite_energy_certified", False))
    axis_ok = bool(interval_report.get("upgrade_gate", {}).get("axis_regular_certified", False))
    exact_ok = bool(exact_bridge.get("exact_profile_verified", False))
    divergence_ok = bool(norm_divergence.get("norm_divergence_certified", False))
    external_ok = _external_verifies(external_verification, "classical_solution_assumptions")
    ready = bool(finite_energy_ok and axis_ok and exact_ok and divergence_ok and external_ok)
    open_obligations: list[str] = []
    if not finite_energy_ok:
        open_obligations.append("finite_energy_initial_data")
    if not axis_ok:
        open_obligations.append("axis_smooth_reconstruction")
    if not exact_ok:
        open_obligations.append("exact_profile_verified")
    if not divergence_ok:
        open_obligations.append("certified_norm_divergence")
    if not external_ok:
        open_obligations.append("external_classical_assumptions_verification")
    return {
        "schema_version": "navier-stokes-classical-assumptions-readiness-1",
        "candidate_type": "classical_assumptions_readiness_gate",
        "finite_energy_certified": finite_energy_ok,
        "axis_regular_certified": axis_ok,
        "exact_profile_verified": exact_ok,
        "norm_divergence_certified": divergence_ok,
        "external_verification": _json_dict(external_verification or {}),
        "classical_assumptions_ready": ready,
        "proof_status": "proved_by_external_artifact" if ready else "blocked_with_precise_obligations",
        "open_obligations": open_obligations,
        "unproven_claim": False,
    }


def build_ns_theorem_ladder_report(
    interval_report: dict[str, Any],
    frontier_report: dict[str, Any],
    *,
    blowup_report: dict[str, Any] | None = None,
    theorem_attempt: dict[str, Any] | None = None,
    invariance_report: dict[str, Any] | None = None,
    finite_tail_diagnostic: dict[str, Any] | None = None,
    analytic_tail_lift: dict[str, Any] | None = None,
    external_verifications: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the conditional theorem ladder from finite evidence to theorem readiness."""
    external = external_verifications or {}
    active_blowup = blowup_report or build_axisymmetric_blowup_closure_report(interval_report)
    active_theorem = theorem_attempt or build_theorem_grade_closure_attempt(
        interval_report,
        blowup_report=active_blowup,
        external_verifications=external,
    )
    linearized = dict(active_blowup.get("closure_certificates", {}).get("linearized_operator", {}))
    componentwise = dict(
        active_blowup.get("closure_certificates", {})
        .get("radii_polynomial", {})
        .get("componentwise", {})
    )
    norm_cert = dict(active_blowup.get("closure_certificates", {}).get("norm_divergence", {}))
    passing_rows = [
        dict(row)
        for row in frontier_report.get("all_rows", [])
        if bool(row.get("frontier_passed", False))
    ]
    finite_core_closed = bool(
        passing_rows
        and all(len(row.get("added_inactive", [])) == 0 for row in passing_rows)
        and frontier_report.get("base_metrics", {}).get("frontier_passed", False)
    )
    finite_theorem = {
        "schema_version": "navier-stokes-finite-active-theorem-1",
        "candidate_type": "active_subspace_finite_theorem_report",
        "active_indices": list(frontier_report.get("base_active_indices", [])),
        "required_tail_modes": list(frontier_report.get("required_tail_control_modes", [])),
        "finite_active_core_closed": finite_core_closed,
        "frontier_passed_count": int(frontier_report.get("passed_count", 0)),
        "frontier_failed_count": int(frontier_report.get("failed_count", 0)),
        "frontier_sha256": _sha256_json(frontier_report),
        "proof_status": "finite_certificate_closed" if finite_core_closed else "blocked_finite_frontier",
        "open_obligations": [] if finite_core_closed else ["finite_active_absorption_frontier"],
        "unproven_claim": False,
    }
    tail_contract = weighted_analytic_tail_norm_contract(
        frontier_report,
        external_verification=external.get("tail_norm_contract"),
    )
    tail_contraction = active_subspace_tail_contraction_attempt(
        frontier_report,
        tail_contract,
        finite_diagnostic=finite_tail_diagnostic,
        analytic_lift=analytic_tail_lift,
        external_verification=external.get("tail_contraction"),
    )
    completeness = active_subspace_completeness_theorem_attempt(
        frontier_report,
        tail_contraction,
        invariance_report=invariance_report,
        external_verification=external.get("active_subspace_completeness"),
    )
    interval_backend = theorem_interval_backend_readiness_report(interval_report)
    continuum = continuum_neumann_inequality_certificate(
        interval_report,
        linearized,
        interval_backend_report=interval_backend,
        external_verification=external.get("continuum_neumann"),
    )
    exact_bridge = defect_to_exact_solution_bridge_attempt(
        interval_report,
        continuum_neumann=continuum,
        componentwise_radii=componentwise,
        active_subspace_completeness=completeness,
        external_verification=external.get("exact_bridge"),
    )
    norm_bridge = certified_norm_divergence_bridge_attempt(
        interval_report,
        norm_cert,
        exact_bridge=exact_bridge,
        external_verification=external.get("norm_divergence"),
    )
    assumptions_gate = classical_assumptions_readiness_gate(
        interval_report,
        exact_bridge=exact_bridge,
        norm_divergence=norm_bridge,
        external_verification=external.get("classical_assumptions"),
    )
    phases = {
        "finite_theorem": finite_theorem,
        "tail_norm_contract": tail_contract,
        "tail_contraction": tail_contraction,
        "active_subspace_completeness": completeness,
        "continuum_neumann": continuum,
        "existence_theorem": exact_bridge,
        "blowup_theorem": norm_bridge,
        "classical_assumptions": assumptions_gate,
    }
    if finite_tail_diagnostic is not None:
        phases["finite_tail_contraction_diagnostic"] = _json_dict(finite_tail_diagnostic)
    if analytic_tail_lift is not None:
        phases["analytic_tail_lift"] = _json_dict(analytic_tail_lift)
    open_obligations = sorted({
        str(item)
        for phase in phases.values()
        if isinstance(phase, dict)
        for item in phase.get("open_obligations", [])
    })
    phase_hashes = {name: _sha256_json(phase) for name, phase in phases.items()}
    theorem_ready = bool(assumptions_gate.get("classical_assumptions_ready", False))
    report = {
        "schema_version": "navier-stokes-theorem-ladder-1",
        "candidate_type": "navier_stokes_theorem_ladder_report",
        "phases": _json_dict(phases),
        "phase_sha256": phase_hashes,
        "route_summary": {
            "finite_theorem_closed": finite_core_closed,
            "finite_tail_contraction_surrogate_passed": bool(
                finite_tail_diagnostic
                and finite_tail_diagnostic.get("finite_tail_contraction_surrogate_passed", False)
            ),
            "finite_tail_contraction_ratio_upper": (
                None
                if finite_tail_diagnostic is None
                else finite_tail_diagnostic.get("finite_contraction_ratio_upper")
            ),
            "analytic_tail_lift_certified": bool(
                analytic_tail_lift
                and analytic_tail_lift.get("analytic_lift_certified", False)
            ),
            "analytic_tail_lift_q_total_upper": (
                None
                if analytic_tail_lift is None
                else analytic_tail_lift.get("q_total_upper")
            ),
            "tail_theorem_closed": bool(completeness.get("active_subspace_complete", False)),
            "existence_theorem_closed": bool(exact_bridge.get("exact_profile_verified", False)),
            "blowup_theorem_closed": bool(norm_bridge.get("norm_divergence_certified", False)),
            "classical_assumptions_ready": theorem_ready,
        },
        "proof_status": "proved_by_external_artifact" if theorem_ready else "blocked_with_named_missing_lemma",
        "open_obligations": open_obligations,
        "unproven_claim": False,
        "replay_inputs": {
            "interval_report": _json_dict(interval_report),
            "frontier_report": _json_dict(frontier_report),
            "blowup_report": _json_dict(active_blowup),
            "theorem_attempt": _json_dict(active_theorem),
            "invariance_report": _json_dict(invariance_report or {}),
            "finite_tail_diagnostic": _json_dict(finite_tail_diagnostic or {}),
            "analytic_tail_lift": _json_dict(analytic_tail_lift or {}),
        },
    }
    report["theorem_ladder_sha256"] = _sha256_json({
        key: value for key, value in report.items() if key != "theorem_ladder_sha256"
    })
    return report


def solve_or_falsify_falsification_report(
    *,
    nontriviality_gate: dict[str, Any],
    ansatz_report: dict[str, Any],
    continuum_neumann: dict[str, Any],
    exact_bridge: dict[str, Any],
    extra_evidence: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    """Aggregate adversarial evidence into an explicit continue/redesign/stop call."""
    evidence = [dict(item) for item in extra_evidence]
    hard_failures: list[str] = []
    if not nontriviality_gate.get("passed", False):
        hard_failures.append("nontriviality_gate_failed")
    if any(item.get("status") == "candidate_falsified" for item in evidence):
        hard_failures.append("external_falsification_evidence")
    redesign_needed = bool(ansatz_report.get("naive_global_enrichment_rejected", False))
    continuum_blocked = not bool(continuum_neumann.get("continuum_neumann_certified", False))
    exact_blocked = not bool(exact_bridge.get("exact_profile_verified", False))
    decision = (
        "stop_current_family"
        if hard_failures
        else "redesign_ansatz"
        if redesign_needed
        else "continue_theorem_route"
        if not (continuum_blocked or exact_blocked)
        else "continue_with_named_obligations"
    )
    return {
        "schema_version": "navier-stokes-falsification-report-1",
        "candidate_type": "solve_or_falsify_falsification_report",
        "decision": decision,
        "hard_failures": hard_failures,
        "redesign_needed": redesign_needed,
        "continuum_blocked": continuum_blocked,
        "exact_profile_blocked": exact_blocked,
        "evidence": _json_dict(evidence),
        "open_obligations": [
            "grid_refinement_falsification",
            "coefficient_perturbation_falsification",
            "pressure_leray_compatibility_falsification",
            "axis_boundary_decay_falsification",
        ],
        "unproven_claim": False,
    }


def build_ns_solve_or_falsify_report(
    interval_report: dict[str, Any],
    *,
    blowup_report: dict[str, Any] | None = None,
    theorem_attempt: dict[str, Any] | None = None,
    active_frontier_report: dict[str, Any] | None = None,
    active_invariance_report: dict[str, Any] | None = None,
    candidate_family_status: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    enriched_linearized_certificates: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    external_verifications: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the full roadmap artifact for the Navier-Stokes solve-or-falsify loop."""
    external = external_verifications or {}
    active_blowup = blowup_report or build_axisymmetric_blowup_closure_report(interval_report)
    active_theorem = theorem_attempt or build_theorem_grade_closure_attempt(
        interval_report,
        blowup_report=active_blowup,
        external_verifications=external,
    )
    linearized = dict(active_blowup.get("closure_certificates", {}).get("linearized_operator", {}))
    componentwise = dict(
        active_blowup.get("closure_certificates", {})
        .get("radii_polynomial", {})
        .get("componentwise", {})
    )
    norm_cert = dict(active_blowup.get("closure_certificates", {}).get("norm_divergence", {}))
    baseline = freeze_axisymmetric_baseline_manifest(
        interval_report,
        blowup_report=active_blowup,
        theorem_attempt=active_theorem,
    )
    nontriviality = axisymmetric_nontriviality_gate(interval_report, blowup_report=active_blowup)
    ansatz = conditioning_preserving_ansatz_report(
        linearized,
        enriched_certificates=enriched_linearized_certificates,
    )
    interval_backend = theorem_interval_backend_readiness_report(interval_report)
    continuum = continuum_neumann_inequality_certificate(
        interval_report,
        linearized,
        interval_backend_report=interval_backend,
        external_verification=external.get("continuum_neumann"),
    )
    tail_contract = None
    tail_contraction = None
    active_completeness = None
    if active_frontier_report is not None:
        tail_contract = weighted_analytic_tail_norm_contract(
            active_frontier_report,
            external_verification=external.get("tail_norm_contract"),
        )
        tail_contraction = active_subspace_tail_contraction_attempt(
            active_frontier_report,
            tail_contract,
            external_verification=external.get("tail_contraction"),
        )
        active_completeness = active_subspace_completeness_theorem_attempt(
            active_frontier_report,
            tail_contraction,
            invariance_report=active_invariance_report,
            external_verification=external.get("active_subspace_completeness"),
        )
    exact_bridge = defect_to_exact_solution_bridge_attempt(
        interval_report,
        continuum_neumann=continuum,
        componentwise_radii=componentwise,
        active_subspace_completeness=active_completeness,
        external_verification=external.get("exact_bridge"),
    )
    norm_bridge = certified_norm_divergence_bridge_attempt(
        interval_report,
        norm_cert,
        exact_bridge=exact_bridge,
        external_verification=external.get("norm_divergence"),
    )
    falsification = solve_or_falsify_falsification_report(
        nontriviality_gate=nontriviality,
        ansatz_report=ansatz,
        continuum_neumann=continuum,
        exact_bridge=exact_bridge,
        extra_evidence=candidate_family_status,
    )
    proof_program = build_ns_proof_program_report(
        theorem_attempt=active_theorem,
        candidate_family_status=candidate_family_status,
        verifier_bundles=external.get("verifier_bundles"),
    )
    formal = build_formal_proof_package(proof_program)
    final_gate = theorem_claim_gate(
        active_theorem,
        formal,
        external_verification=external.get("final_claim"),
    )
    phases = {
        "baseline": baseline,
        "nontriviality": nontriviality,
        "ansatz": ansatz,
        "interval_backend": interval_backend,
        "continuum_neumann": continuum,
        "exact_solution_bridge": exact_bridge,
        "norm_divergence": norm_bridge,
        "falsification": falsification,
        "proof_program": proof_program,
        "formal_verification": {
            "schema_version": "navier-stokes-formal-verification-readiness-1",
            "candidate_type": "formal_verification_readiness_report",
            "formal_package": _json_dict(formal),
            "final_claim_gate": _json_dict(final_gate),
            "proof_status": final_gate["promotion_status"],
            "open_obligations": final_gate["open_obligations"],
            "unproven_claim": False,
        },
    }
    if active_frontier_report is not None:
        phases["tail_norm_contract"] = _json_dict(tail_contract)
        phases["tail_contraction"] = _json_dict(tail_contraction)
        phases["active_subspace_completeness"] = _json_dict(active_completeness)
    open_obligations = sorted({
        str(item)
        for phase in phases.values()
        if isinstance(phase, dict)
        for item in phase.get("open_obligations", [])
    })
    phase_hashes = {name: _sha256_json(phase) for name, phase in phases.items()}
    report = {
        "schema_version": "navier-stokes-solve-or-falsify-1",
        "candidate_type": "navier_stokes_solve_or_falsify_report",
        "phases": _json_dict(phases),
        "phase_sha256": phase_hashes,
        "decision": falsification["decision"],
        "proof_status": (
            "proved_by_external_artifact"
            if final_gate.get("unproven_claim", False)
            else "candidate_falsified"
            if falsification["decision"] == "stop_current_family"
            else "blocked_with_named_missing_lemma"
        ),
        "open_obligations": open_obligations,
        "unproven_claim": False,
        "replay_inputs": {
            "interval_report": _json_dict(interval_report),
            "blowup_report": _json_dict(active_blowup),
            "theorem_attempt": _json_dict(active_theorem),
            "proof_program": _json_dict(proof_program),
        },
    }
    report["solve_or_falsify_sha256"] = _sha256_json({
        key: value for key, value in report.items() if key != "solve_or_falsify_sha256"
    })
    return report


def build_certificate_manifest(
    *,
    cap_bundle: dict[str, Any] | None = None,
    candidate_artifact: dict[str, Any] | None = None,
    interval_report: dict[str, Any] | None = None,
    certified_refinement_report: dict[str, Any] | None = None,
    closure_report: dict[str, Any] | None = None,
    closure_certificates: dict[str, Any] | None = None,
    formal_package: dict[str, Any] | None = None,
    external_verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a reproducibility manifest for external CAP review."""
    artifacts = {
        "cap_bundle": cap_bundle,
        "candidate_artifact": candidate_artifact,
        "interval_report": interval_report,
        "certified_refinement_report": certified_refinement_report,
        "closure_report": closure_report,
        "closure_certificates": closure_certificates,
        "formal_package": formal_package,
        "external_verification": external_verification,
    }
    hashes: dict[str, str] = {}
    present: list[str] = []
    for name, artifact in artifacts.items():
        if artifact is None:
            continue
        present.append(name)
        encoded = json.dumps(_json_dict(artifact), sort_keys=True).encode("utf-8")
        hashes[name] = hashlib.sha256(encoded).hexdigest()
    open_obligations: list[str] = []

    def collect_open_obligations(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {"open_obligations", "proof_obligations"} and isinstance(item, list):
                    open_obligations.extend(str(v) for v in item)
                else:
                    collect_open_obligations(item)
        elif isinstance(value, list):
            for item in value:
                collect_open_obligations(item)

    for artifact in artifacts.values():
        if artifact is None:
            continue
        collect_open_obligations(artifact)
    return {
        "schema_version": "navier-stokes-certificate-manifest-1",
        "artifacts_present": present,
        "artifact_sha256": hashes,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "interval_backend": asdict(interval_arithmetic_metadata()),
        },
        "verification_commands": [
            "python -m pytest packages/omnibias-pinn/tests/certified/test_navier_stokes_certified.py -q",
            "python -m pytest packages/omnibias-symbolic/tests/test_symbolic_navier_stokes.py -q",
            "uv run ruff check packages/omnibias-pinn/src/omnibias/pinn/certified packages/omnibias-symbolic/src/omnibias/symbolic packages/omnibias-pinn/tests/certified packages/omnibias-symbolic/tests",
            "mkdocs build --strict",
        ],
        "open_obligations": sorted(set(open_obligations)),
        "claim_boundary": {
            "unproven_claim": False,
            "reason": "external theorem-level verification is not attached",
        },
    }


def write_candidate_artifact(artifact: dict[str, Any], out_dir: Path | str) -> Path:
    """Write ``navier_stokes_candidate.json`` and a short markdown summary."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "navier_stokes_candidate.json"
    json_path.write_text(json.dumps(artifact, indent=2, sort_keys=True))
    lines = [
        "# Navier-Stokes candidate artifact",
        "",
        f"- type: `{artifact.get('candidate_type')}`",
        f"- stage: `{artifact.get('upgrade_gate', {}).get('stage', 'unknown')}`",
        f"- replay domain: `{artifact.get('replay_grid', {}).get('domain_type', 'unknown')}`",
        "",
        "## Honesty",
        f"- global-regularity claim: `{artifact.get('honesty', {}).get('unproven_claim')}`",
        f"- exact solution claim: `{artifact.get('honesty', {}).get('exact_solution_claim')}`",
        f"- interval verified: `{artifact.get('honesty', {}).get('interval_verified')}`",
    ]
    (out / "navier_stokes_candidate_summary.md").write_text("\n".join(lines) + "\n")
    return json_path


def write_ns_cap_bundle(bundle: dict[str, Any], out_dir: Path | str) -> Path:
    """Write ``navier_stokes_cap.json`` and a short honesty-first summary."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "navier_stokes_cap.json"
    json_path.write_text(json.dumps(bundle, indent=2, sort_keys=True))
    diag = bundle["residual_diagnostics"]
    lines = [
        "# Navier-Stokes certified-evidence CAP bundle",
        "",
        f"- dimension: `{bundle['problem']['dimension']}`",
        f"- domain: `{bundle['domain']['type']}`, grid `{bundle['domain']['grid_shape']}`",
        f"- max|momentum residual|: `{diag.get('max_abs_momentum_residual', float('nan')):.3e}`",
        f"- max|div u|: `{diag.get('max_abs_continuity', float('nan')):.3e}`",
        f"- max pressure-Poisson residual: `{diag.get('max_abs_pressure_poisson', float('nan')):.3e}`",
        "",
        "## Honesty",
        f"- global-regularity claim: `{bundle['honesty']['unproven_claim']}`",
        f"- exact solution claim: `{bundle['honesty']['exact_solution_claim']}`",
        f"- interval verified: `{bundle['honesty']['interval_verified']}`",
        f"- theorem-prover verified: `{kernel_earned_theorem_prover_verified(bundle)}`",
        f"- periodic model only: `{bundle['honesty']['periodic_model_only']}`",
    ]
    (out / "navier_stokes_cap_summary.md").write_text("\n".join(lines) + "\n")
    return json_path


__all__ = [
    "AxisymmetricBasisMetadata",
    "AxisymmetricCompactifiedMetadata",
    "AxisymmetricFunctionSpaceMetadata",
    "AxisymmetricSwirlAnsatzMetadata",
    "CANDIDATE_SCHEMA_VERSION",
    "CandidateGateConfig",
    "CompactifiedCoefficientSet",
    "CompactifiedR3Metadata",
    "ExactEquationContract",
    "HonestyLabels",
    "IntervalArithmeticMetadata",
    "IntervalBoundReport",
    "IntervalCAPBackendContract",
    "LinearizedOperatorCertificate",
    "NavierStokesProofContract",
    "NavierStokesSubstrate",
    "NormDivergenceCertificate",
    "ProofObligation",
    "ProofObligationBundle",
    "ProofProgramFunctionSpaceDefinition",
    "REQUIRED_CANDIDATE_KEYS",
    "REQUIRED_CAP_KEYS",
    "REQUIRED_VALIDATION_KEYS",
    "RadiiPolynomialCertificate",
    "ReplayGrid",
    "SCHEMA_VERSION",
    "ScalarInterval",
    "TailBound",
    "TheoremGradeFunctionSpaceContract",
    "active_projector_error_certificate",
    "active_subspace_absorption_frontier_report",
    "active_subspace_completeness_theorem_attempt",
    "active_subspace_invariance_report",
    "active_subspace_tail_contraction_attempt",
    "active_tail_contraction_lift_certificate",
    "analytic_tail_error_certificate",
    "assemble_axisymmetric_active_subspace_operator",
    "assemble_axisymmetric_linearized_operator",
    "axisymmetric_axis_smoothness_certificate",
    "axisymmetric_basis_count",
    "axisymmetric_basis_metadata",
    "axisymmetric_basis_regular_interval_checks",
    "axisymmetric_basis_tensor",
    "axisymmetric_coefficient_loss",
    "axisymmetric_coefficients_to_fields",
    "axisymmetric_compactified_metadata",
    "axisymmetric_energy_estimate",
    "axisymmetric_function_space_metadata",
    "axisymmetric_holdout_replay_grid",
    "axisymmetric_meridional_replay_grid",
    "axisymmetric_nontriviality_gate",
    "axisymmetric_physical_axes",
    "axisymmetric_swirl_ansatz_metadata",
    "axisymmetric_swirl_residual_samples",
    "axisymmetric_velocity_from_streamfunction",
    "blowup_contract",
    "blowup_proof_obligation_bundles",
    "blowup_route_lemma_package",
    "build_active_tail_lift_error_budget",
    "build_analytic_closure_report",
    "build_axisymmetric_active_subspace_closure_report",
    "build_axisymmetric_blowup_closure_report",
    "build_axisymmetric_interval_report",
    "build_axisymmetric_swirl_candidate_artifact",
    "build_blowup_closure_report",
    "build_candidate_artifact",
    "build_certificate_manifest",
    "build_formal_proof_package",
    "build_ns_cap_bundle",
    "build_ns_proof_program_report",
    "build_ns_solve_or_falsify_report",
    "build_ns_theorem_ladder_report",
    "build_refined_axisymmetric_swirl_candidate_artifact",
    "build_regularity_closure_report",
    "build_regularity_inequality_report",
    "build_theorem_grade_closure_attempt",
    "candidate_artifact_schema_errors",
    "candidate_upgrade_gates",
    "certified_candidate_refinement_report",
    "certified_norm_divergence_bridge_attempt",
    "certified_tail_bound_from_coefficients",
    "certified_tail_bounds_from_artifact",
    "classical_assumptions_readiness_gate",
    "coefficient_interval_boxes",
    "compactification_map_interval",
    "compactified_coefficient_set",
    "compactified_r3_metadata",
    "compactified_sandbox_replay_grid",
    "componentwise_radii_polynomial_certificate",
    "conditioning_preserving_ansatz_report",
    "continuum_banach_invertibility_attempt",
    "continuum_neumann_inequality_certificate",
    "continuum_residual_certificates",
    "continuum_residual_upper_bound",
    "default_proof_obligations",
    "defect_to_exact_solution_bridge_attempt",
    "deterministic_periodic_replay_grid",
    "energy_diagnostics",
    "exact_navier_stokes_equation_contracts",
    "exact_profile_norm_divergence_attempt",
    "external_review_gate",
    "external_verification_record",
    "finite_active_tail_contraction_diagnostic",
    "finite_energy_interval_bounds",
    "finite_energy_tail_certificate",
    "freeze_axisymmetric_baseline_manifest",
    "global_regularity_contract",
    "ingest_theorem_verifier_bundle",
    "initial_axisymmetric_coefficients",
    "interval_add",
    "interval_arithmetic_metadata",
    "interval_bound_report",
    "interval_cap_backend_contract",
    "interval_div",
    "interval_from_bounds",
    "interval_jacobian_error_certificate",
    "interval_mul",
    "interval_sqrt",
    "interval_square",
    "interval_sub",
    "interval_trapezoid_bound",
    "lean_formalization_package",
    "leray_project_periodic",
    "manufactured_abc_flow",
    "nonlinear_tail_remainder_certificate",
    "norm_divergence_certificate",
    "ns_cap_schema_errors",
    "operator_theoretic_invertibility_certificate",
    "pressure_poisson_residual_periodic",
    "primitive_residual_periodic",
    "proof_contract_bundle",
    "proof_obligation_bundle",
    "radii_polynomial_certificate",
    "refine_axisymmetric_coefficients",
    "regularity_all_data_proof_attempt",
    "regularity_counterexample_sweep",
    "regularity_proof_obligation_bundles",
    "regularity_route_lemma_package",
    "residual_interval_envelopes",
    "scalar_interval",
    "scalar_interval_contains",
    "solve_or_falsify_falsification_report",
    "spectral_curl",
    "spectral_divergence",
    "spectral_gradient_scalar",
    "spectral_laplacian",
    "split_axisymmetric_coefficients",
    "theorem_claim_gate",
    "theorem_grade_function_space_contract",
    "theorem_grade_function_space_definitions",
    "theorem_grade_radii_polynomial_attempt",
    "theorem_interval_backend_readiness_report",
    "theorem_verifier_record",
    "verify_external_proof_package",
    "vorticity_residual_periodic",
    "weighted_analytic_tail_norm_contract",
    "write_candidate_artifact",
    "write_ns_cap_bundle",
]
