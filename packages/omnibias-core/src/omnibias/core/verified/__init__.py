# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Verified (interval / Taylor-model) arithmetic backend for omnibias.

This subpackage lifts omnibias's *bit-stable* closed-form derivative tower into a
*rigorous* one: every quantity is returned as a guaranteed
:class:`~omnibias.core.verified.interval.Interval` enclosure produced by outward
directed rounding.  It is the dependency-free substrate the certified-evidence stack
needs to replace float64 padding with theorem-grade bounds.

Layers
------
* :mod:`~omnibias.core.verified.interval` -- rigorous interval arithmetic.
* :mod:`~omnibias.core.verified.transcend` -- directed ``exp``/``tanh``/``sigmoid``
  enclosures with optional mpmath; the libm fallback is conditional and refused
  in certificate mode.
* :mod:`~omnibias.core.verified.clamped_biharmonic` -- exact full Green-operator
  constants and a conditional rational contraction gate for a clamped cubic beam.
* :mod:`~omnibias.core.verified.coeffs` -- exact-integer tower coefficients +
  interval Horner.
* :mod:`~omnibias.core.verified.sigma` -- rigorous ``sigma^(k)`` tower enclosure.
* :mod:`~omnibias.core.verified.jet` -- verified directional jets.
* :mod:`~omnibias.core.verified.taylor_model` -- Taylor models (interval
  polynomial + rigorous remainder) for enclosing functions over a whole cell.
* :mod:`~omnibias.core.verified.taylor_model_mv` -- the multivariate
  generalisation: total-degree Taylor models over a box in ``R^n``.

``mpmath`` is an optional accelerator (``pip install omnibias-core[verified]``);
the package imports and runs without it.
"""

from __future__ import annotations

from omnibias.core.verified.affine import (
    AffineForm,
    from_intervals,
    new_noise_symbol,
)
from omnibias.core.verified.banded import (
    banded_tail_inverse_bound,
    finite_band_row_sum_bound,
    geometric_band_row_sum_bound,
)
from omnibias.core.verified.clamped_biharmonic import (
    ClampedCubicContraction,
    clamped_biharmonic_green_kernel,
    clamped_biharmonic_inverse_norm,
    clamped_cubic_contraction,
)
from omnibias.core.verified.coeffs import (
    bernoulli_number_exact,
    bernoulli_polynomial_exact,
    euler_number_exact,
    generalized_bernoulli_exact,
    hermite_coeffs_exact,
    horner_interval,
    sech_poly_coeffs_exact,
    sigmoid_poly_coeffs_exact,
    tanh_poly_coeffs_exact,
)
from omnibias.core.verified.complex_interval import ComplexInterval, ComplexLike
from omnibias.core.verified.complex_ode import (
    ComplexTaylorSeries,
    ComplexVectorField,
    integrate_complex_ivp,
)
from omnibias.core.verified.complex_rootfind import (
    ComplexFn,
    ComplexNewtonResult,
    ComplexNewtonStatus,
    parametric_complex_interval_newton,
)
from omnibias.core.verified.conditioning import (
    certified_condition_number,
    certified_damping,
    certified_max_eigenvalue,
    certified_min_eigenvalue,
    certified_regularization_error,
    conditioning_certificate,
    positive_definite_pivots_certificate,
)
from omnibias.core.verified.dfinite_continuation import (
    DFiniteContinuation,
    SerializedOperator,
    companion_matrix,
    continue_dfinite,
    continue_dfinite_path,
    evaluate_complex_polynomial,
    normalize_operator,
    realify_complex_matrix,
)
from omnibias.core.verified.eig import (
    BlockOperatorGapCertificate,
    PerronGapCertificate,
    SymmetricGapCertificate,
    birkhoff_contraction_ratio,
    birkhoff_projective_diameter,
    certified_block_operator_gap,
    certified_perron_spectral_gap,
    certified_symmetric_spectral_gap,
    collatz_wielandt_perron_bounds,
    rayleigh_quotient,
    symmetric_eigenvalue_residual_enclosure,
)
from omnibias.core.verified.eig_operator import (
    EigenvalueLowerBound,
    Inertia,
    LehmannCertificate,
    SpectralGapCertificate,
    certified_spectral_gap,
    count_eigenvalues_below,
    generalized_eigenvalue_enclosure,
    interval_ldlt_factor,
    interval_ldlt_inertia,
    interval_ldlt_pivots,
    is_positive_definite,
    lehmann_maehly_lower_bounds,
    operator_comparison_bounds,
    ritz_upper_bound,
    temple_lower_bound,
    temple_lower_bound_vector,
)
from omnibias.core.verified.enclosure_collapse import (
    RecommendedAction,
    WidthLaw,
    diagnose_width,
    measured_critical_width,
    measured_mean_value_width,
    measured_taylor_remainder,
    predicted_width,
    rounding_floor_witness,
    width_law,
)
from omnibias.core.verified.euler_maclaurin import (
    digamma_iv,
    euler_maclaurin_sum,
    log_gamma_iv,
)
from omnibias.core.verified.exp_polynomial import (
    evaluate_exp_polynomial,
    evaluate_rational_field,
)
from omnibias.core.verified.fourier import (
    NuLike,
    Symbol,
    ValidatedFourierSeries,
    Wavevector,
    hilbert_symbol,
    leray_symbol,
    riesz_symbol,
)
from omnibias.core.verified.ftc import (
    certified_ftc_residual,
    ftc_certificate,
)
from omnibias.core.verified.gaussian_affine import (
    GaussianAffineRestriction,
    InconsistentAffineConstraints,
    restrict_gaussian_heat,
)
from omnibias.core.verified.gaussian_affine_adjoint import (
    GaussianAffineAdjointGeometry,
    compile_gaussian_affine_adjoint,
)
from omnibias.core.verified.gaussian_affine_aging import (
    GaussianAffineAging,
    certify_gaussian_affine_aging,
    certify_gaussian_affine_predictor,
)
from omnibias.core.verified.gaussian_affine_approximation import (
    GaussianAffineApproximation,
    certify_gaussian_affine_approximation,
)
from omnibias.core.verified.gaussian_affine_evolution import (
    GaussianAffineEvolution,
    certify_gaussian_affine_evolution,
)
from omnibias.core.verified.gaussian_affine_parameters import (
    GaussianAffineParameterVJP,
    certify_gaussian_affine_parameter_vjp,
)
from omnibias.core.verified.gaussian_affine_sensitivity import (
    GaussianAffineTargetVJPCertificate,
    certify_gaussian_affine_target_vjp,
)
from omnibias.core.verified.gaussian_affine_stationarity import (
    GaussianAffineStationarity,
    certify_gaussian_affine_stationarity,
)
from omnibias.core.verified.gaussian_aging import (
    GaussianInverseAging,
    certify_gaussian_inverse_aging,
)
from omnibias.core.verified.gaussian_aging_derivative import (
    GaussianAgingDerivative,
    certify_gaussian_aging_derivative,
)
from omnibias.core.verified.gaussian_analytic import (
    GaussianSeriesCertificate,
    certify_gaussian_inverse_series,
)
from omnibias.core.verified.gaussian_bias_analytic import (
    GaussianBiasSeriesCertificate,
    certify_gaussian_bias_inverse_series,
)
from omnibias.core.verified.gaussian_chebyshev import (
    GaussianInverseApproximation,
    approximate_gaussian_inverse,
)
from omnibias.core.verified.gaussian_chebyshev_derivative import (
    GaussianInverseDerivative,
    certify_gaussian_inverse_derivative,
)
from omnibias.core.verified.gaussian_collinear_minimum import (
    GaussianCollinearMinimum,
    certify_gaussian_collinear_minimum,
)
from omnibias.core.verified.gaussian_collinear_reduction import (
    GaussianCollinearPiece,
    GaussianCollinearReduction,
    compile_gaussian_collinear_reduction,
)
from omnibias.core.verified.gaussian_critical import (
    GaussianCriticalChart,
    certify_gaussian_critical_chart,
)
from omnibias.core.verified.gaussian_entropy_bias import (
    GaussianEntropyBias,
    certify_gaussian_entropy_bias,
)
from omnibias.core.verified.gaussian_entropy_bias_face import (
    GaussianEntropyBiasFace,
    certify_gaussian_entropy_bias_face,
)
from omnibias.core.verified.gaussian_entropy_bias_region import (
    GaussianEntropyBiasRegion,
    certify_gaussian_entropy_bias_region,
)
from omnibias.core.verified.gaussian_entropy_face import (
    GaussianEntropyFace,
    certify_gaussian_entropy_face,
)
from omnibias.core.verified.gaussian_entropy_joint_face import (
    GaussianEntropyJointFace,
    certify_gaussian_entropy_joint_face,
)
from omnibias.core.verified.gaussian_entropy_joint_region import (
    GaussianEntropyJointRegion,
    certify_gaussian_entropy_joint_region,
)
from omnibias.core.verified.gaussian_entropy_minimum import (
    GaussianEntropyMinimum,
    certify_gaussian_entropy_minimum,
)
from omnibias.core.verified.gaussian_entropy_residual import (
    GaussianEntropyResidual,
    certify_gaussian_entropy_residual,
)
from omnibias.core.verified.gaussian_entropy_sensitivity import (
    GaussianEntropySensitivity,
    certify_gaussian_entropy_sensitivity,
)
from omnibias.core.verified.gaussian_entropy_solver import (
    GaussianEntropySolve,
    solve_gaussian_entropy,
)
from omnibias.core.verified.gaussian_entropy_unique import (
    GaussianEntropyUnique,
    certify_gaussian_entropy_unique,
)
from omnibias.core.verified.gaussian_evolution import (
    GaussianInverseEvolution,
    certify_gaussian_inverse_evolution,
)
from omnibias.core.verified.gaussian_grouped_directional import (
    GaussianGroupedCriticalFace,
    GaussianGroupedDirectional,
    certify_gaussian_grouped_directional,
    compile_gaussian_grouped_critical_face,
)
from omnibias.core.verified.gaussian_grouped_equilibrium import (
    GaussianGroupedEquilibrium,
    certify_gaussian_grouped_equilibrium,
)
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_joint_directional import (
    GaussianJointDirectional,
    certify_gaussian_joint_directional,
)
from omnibias.core.verified.gaussian_polyhedral import (
    GaussianPolyhedralFace,
    GaussianPolyhedralMinimum,
    GaussianPolyhedralParameterVJP,
    certify_gaussian_polyhedral_minimum,
    certify_gaussian_polyhedral_parameter_vjp,
    compile_gaussian_polyhedral_face,
)
from omnibias.core.verified.gaussian_polyhedral_directional import (
    GaussianCriticalFace,
    GaussianPolyhedralDirectional,
    certify_gaussian_polyhedral_directional,
    compile_gaussian_critical_face,
)
from omnibias.core.verified.gaussian_polyhedral_equilibrium import (
    GaussianPolyhedralEquilibrium,
    certify_gaussian_polyhedral_equilibrium,
)
from omnibias.core.verified.gaussian_polyhedral_path import (
    GaussianPathSign,
    GaussianPolyhedralPath,
    compile_gaussian_polyhedral_path,
)
from omnibias.core.verified.gaussian_polyhedral_radius import (
    GaussianPathSignBound,
    GaussianPolyhedralPathRadius,
    certify_gaussian_polyhedral_path_radius,
)
from omnibias.core.verified.gaussian_polyhedral_reduced_minimum import (
    GaussianPolyhedralReducedMinimum,
    certify_gaussian_polyhedral_reduced_minimum,
)
from omnibias.core.verified.gaussian_polyhedral_reduced_sensitivity import (
    GaussianPolyhedralReducedSensitivity,
    certify_gaussian_polyhedral_reduced_sensitivity,
)
from omnibias.core.verified.gaussian_polyhedral_reduction import (
    GaussianPolyhedralPiece,
    GaussianPolyhedralReduction,
    compile_gaussian_polyhedral_reduction,
)
from omnibias.core.verified.gaussian_polyhedral_unique import (
    GaussianPolyhedralUnique,
    certify_gaussian_polyhedral_unique,
)
from omnibias.core.verified.gaussian_posterior_domain import (
    GaussianPosteriorDomain,
    certify_gaussian_posterior_domain,
)
from omnibias.core.verified.gaussian_predictor import (
    GaussianInversePredictor,
    certify_gaussian_inverse_predictor,
)
from omnibias.core.verified.gaussian_projected_equilibrium import (
    GaussianProjectedEquilibrium,
    certify_gaussian_projected_equilibrium,
)
from omnibias.core.verified.gaussian_projection_minimum import (
    GaussianProjectionMinimum,
    certify_gaussian_projection_minimum,
)
from omnibias.core.verified.gaussian_sensitivity import (
    GaussianTargetVJPCertificate,
    certify_gaussian_target_vjp,
)
from omnibias.core.verified.gaussian_tube import (
    GaussianInverseTubeCertificate,
    certify_gaussian_inverse_tube,
)
from omnibias.core.verified.gaussian_two_center import (
    GaussianTwoCenterMinimum,
    certify_gaussian_two_center_minimum,
)
from omnibias.core.verified.global_min import BoxMinimum, minimize_box
from omnibias.core.verified.hardy_line import (
    hardy_angle,
    hardy_even,
    hardy_even_dalpha,
    hardy_even_deriv,
    hardy_even_deriv_matrix,
    hardy_even_deriv_n,
    hardy_even_deriv_n_iv,
    hardy_even_matrix,
    hardy_even_profile,
    hardy_even_profile_deriv,
    hardy_odd,
    hardy_odd_dalpha,
    hardy_odd_deriv,
    hardy_odd_deriv_matrix,
    hardy_odd_deriv_n,
    hardy_odd_deriv_n_iv,
    hardy_odd_matrix,
    hardy_omega_velocity_atom,
    hardy_omega_velocity_atom_iv,
    hardy_pair,
    hardy_radius,
    hardy_tail_constant,
    hilbert_hardy_even_profile,
    hilbert_hardy_even_profile_deriv,
    hilbert_of_hardy_even,
    hilbert_of_hardy_even_deriv_n,
    hilbert_of_hardy_odd,
    hilbert_of_hardy_odd_deriv_n,
    pochhammer,
    pochhammer_iv,
)
from omnibias.core.verified.hermite_basis import (
    CRAMER_UNIFORM_BOUND,
    HermiteExpansion,
    gaussian_weight,
    hermite_function,
    hermite_function_normalized,
    hermite_poly_coeffs_exact,
    neg_i_power,
)
from omnibias.core.verified.interval import Interval, IntervalLike, hull, sum_intervals
from omnibias.core.verified.interval_array import (
    IntervalArray,
    IntervalArrayLike,
    dot,
    sparse_matvec,
)
from omnibias.core.verified.invariant_subspace import (
    InvariantSubspaceCertificate,
    certified_invariant_subspace,
)
from omnibias.core.verified.inverse_gaussian_affine import (
    GaussianAffineInverseCertificate,
    GaussianAffineInverseResult,
    certify_gaussian_affine_inverse,
    inverse_gaussian_affine,
)
from omnibias.core.verified.inverse_gaussian_heat import (
    GaussianInverseCertificate,
    GaussianInverseResult,
    certify_gaussian_inverse,
    inverse_gaussian_heat,
)
from omnibias.core.verified.inverse_mean_map import (
    MeanInverseCertificate,
    MeanInverseResult,
    certify_mean_inverse,
    inverse_mean_map,
)
from omnibias.core.verified.jet import (
    affine_jet,
    antiderivative_jet,
    compose_jet,
    derivative_jet,
    jet_to_tower,
    layer_jet,
    lhopital_ratio_iv,
    mlp_jet,
    path_jet,
    series_mul,
    tower_to_jet,
)
from omnibias.core.verified.jet_mv import (
    MVJet,
    certified_partials,
    certified_partials_subdivided,
    certified_residual_bound,
    compose_jet_mv,
    identity_jet,
    jet_gradient,
    jet_hessian,
    jet_laplacian,
    jet_multiply,
    jet_partials,
    layer_jet_mv,
    mlp_jet_mv,
)
from omnibias.core.verified.kantorovich import (
    CONTINUUM_PDE_CLAIM_KEY,
    FINITE_RESIDUAL_CLAIM,
    KantorovichAccept,
    KrawczykCertificate,
    NKBounds,
    RadiiCertificate,
    certify_zero_radii,
    kantorovich_accept_step,
    krawczyk_certificate,
    krawczyk_search,
    newton_kantorovich_bounds,
    polynomial_sqrt2_maps,
    radii_polynomial_certificate,
    select_accepted_params,
)
from omnibias.core.verified.laguerre_basis import (
    LaguerreExpansion,
    laguerre_function,
    laguerre_poly_coeffs_exact,
    laguerre_weight,
)
from omnibias.core.verified.linalg import (
    identity_matrix,
    inf_norm_matrix,
    inf_norm_vector,
    interval_solve,
    interval_triangular_solve,
    mat_sub,
    matmul,
    matvec,
    neumann_inverse_norm_bound,
    to_interval_matrix,
)
from omnibias.core.verified.linalg_array import (
    IntervalLDLT,
    interval_ldlt_factor_array,
    interval_ldlt_inertia_array,
    interval_ldlt_pivots_array,
    is_positive_definite_array,
)
from omnibias.core.verified.line import (
    conjugate_poisson,
    conjugate_poisson_deriv,
    conjugate_poisson_deriv_matrix,
    conjugate_poisson_matrix,
    even_profile,
    even_profile_deriv,
    even_profile_tail_constant,
    hilbert_even_profile,
    hilbert_even_profile_deriv,
    hilbert_of_conjugate,
    hilbert_of_poisson,
    hilbert_tail_bound,
    poisson_kernel,
    poisson_kernel_deriv,
    poisson_matrix,
    poisson_primitive,
    poisson_primitive_matrix,
)
from omnibias.core.verified.lohner import (
    JacobianEnclosure,
    LohnerSet,
    constant_jacobian,
    interval_matrix_exp,
    linear_field,
    lohner_flow,
    lohner_step,
    naive_interval_flow,
    qr_gram_schmidt,
)
from omnibias.core.verified.mean_map import MeanEvaluation, MeanMap
from omnibias.core.verified.mlp_objective import IntervalMLP
from omnibias.core.verified.ode import TaylorSeries, VectorField, integrate_ivp
from omnibias.core.verified.pde_certificate import (
    BoundaryFace,
    Coeff,
    LinearPDE,
    PINNErrorCertificate,
    advection_diffusion,
    aposteriori_error_certificate,
    certified_boundary_residual,
    certified_interior_residual,
    helmholtz,
    laplace,
    poisson,
    radii_polynomial_residual_certificate,
    screened_poisson,
    spectral_residual_norm,
)
from omnibias.core.verified.periodic_heat import (
    BurgersJet,
    HeatEnclosure,
    enclose_burgers_jet,
    enclose_heat_derivative,
)
from omnibias.core.verified.polyhedral_projection import (
    PolyhedralProjection,
    PolyhedralProjector,
    compile_polyhedral_projector,
)
from omnibias.core.verified.quadrature import (
    GAUSS_LEGENDRE_MAX_N,
    DerivOracle,
    IntervalFn,
    QuadEstimate,
    clenshaw_curtis_integral,
    euler_maclaurin_quadrature,
    gauss_legendre_integral,
    gauss_legendre_nodes,
    midpoint_integral,
    romberg_integral,
    simpson_integral,
    tanh_sinh_estimate,
    trapezoid_integral,
    trapezoid_sum,
)
from omnibias.core.verified.radii_series import (
    SeriesProblem,
    SeriesRadiiResult,
    constant_symbol,
    constant_tail_inverse_bound,
    series_radii_certificate,
)
from omnibias.core.verified.radii_series import (
    Symbol as SeriesSymbol,
)
from omnibias.core.verified.radii_series import (
    evaluate_residual as evaluate_series_residual,
)
from omnibias.core.verified.radii_spectral import (
    BandedLinearPart,
    Bilinear,
    SpectralProblem,
    SpectralRadiiResult,
    banded_row_sum_bound_nu,
    constant_coefficient_band,
    evaluate_residual,
    laplacian_symbol,
    laplacian_tail_inverse_bound,
    quadratic_radii_certificate,
    tail_inverse_bound_from_banded,
)
from omnibias.core.verified.riesz import (
    blob,
    blob_gradient,
    leray_blob_field,
    leray_divergence_residual,
    newtonian_potential,
    potential_gradient,
    riesz_double_blob,
    riesz_tail_bound,
)
from omnibias.core.verified.rootfind import (
    bisection_bracket,
    certified_sign_change,
    interval_newton,
)
from omnibias.core.verified.sampled import ConcentrationReport, hoeffding_enclosure
from omnibias.core.verified.sequence_space import (
    ValidatedSeries,
    convolve,
    ell1_nu_norm,
    fourier_nu_norm,
    geometric_tail_bound,
)
from omnibias.core.verified.series import (
    certified_geometric_series_sum,
    certified_ratio_series_sum,
    geometric_series_closed_form,
    geometric_tail_enclosure,
)
from omnibias.core.verified.sigma import sigma_tower_interval, sigma_value_interval
from omnibias.core.verified.spectral import (
    cos_matrix,
    hilbert_circulant,
    sin_matrix,
    uniform_nodes,
)
from omnibias.core.verified.sqg import (
    sqg_blob,
    sqg_blob_gradient,
    sqg_blob_l2_inner,
    sqg_riesz,
    sqg_stream,
    sqg_stream_gradient,
    sqg_velocity,
    sqg_velocity_divergence_residual,
)
from omnibias.core.verified.taylor_model import TaylorModel
from omnibias.core.verified.taylor_model_mv import TaylorModelMV
from omnibias.core.verified.transcend import (
    E_IV,
    PI_IV,
    atan_iv,
    atan_point,
    backend_name,
    besseli_iv,
    besseli_point,
    certificate_mode,
    clear_libm_fallback_used,
    cos_iv,
    cos_point,
    cosh_iv,
    erf_iv,
    erf_point,
    exp_iv,
    gauss_cdf_iv,
    gauss_cdf_point,
    libm_fallback_used,
    ln_iv,
    ln_point,
    require_rigorous_backend,
    sech_iv,
    set_strict_backend,
    sigmoid_iv,
    sin_iv,
    sin_point,
    sinh_iv,
    softplus_iv,
    strict_backend,
    tanh_iv,
)

__all__ = [
    'AffineForm',
    'BandedLinearPart',
    'Bilinear',
    'BlockOperatorGapCertificate',
    'BoundaryFace',
    'BoxMinimum',
    'BurgersJet',
    'CONTINUUM_PDE_CLAIM_KEY',
    'CRAMER_UNIFORM_BOUND',
    'ClampedCubicContraction',
    'Coeff',
    'ComplexFn',
    'ComplexInterval',
    'ComplexLike',
    'ComplexNewtonResult',
    'ComplexNewtonStatus',
    'ComplexTaylorSeries',
    'ComplexVectorField',
    'ConcentrationReport',
    'DFiniteContinuation',
    'DerivOracle',
    'E_IV',
    'EigenvalueLowerBound',
    'FINITE_RESIDUAL_CLAIM',
    'GAUSS_LEGENDRE_MAX_N',
    'GaussianAffineAdjointGeometry',
    'GaussianAffineAging',
    'GaussianAffineApproximation',
    'GaussianAffineEvolution',
    'GaussianAffineInverseCertificate',
    'GaussianAffineInverseResult',
    'GaussianAffineParameterVJP',
    'GaussianAffineRestriction',
    'GaussianAffineStationarity',
    'GaussianAffineTargetVJPCertificate',
    'GaussianAgingDerivative',
    'GaussianBiasSeriesCertificate',
    'GaussianCollinearMinimum',
    'GaussianCollinearPiece',
    'GaussianCollinearReduction',
    'GaussianCriticalChart',
    'GaussianCriticalFace',
    'GaussianEntropyBias',
    'GaussianEntropyBiasFace',
    'GaussianEntropyBiasRegion',
    'GaussianEntropyFace',
    'GaussianEntropyJointFace',
    'GaussianEntropyJointRegion',
    'GaussianEntropyMinimum',
    'GaussianEntropyResidual',
    'GaussianEntropySensitivity',
    'GaussianEntropySolve',
    'GaussianEntropyUnique',
    'GaussianGroupedCriticalFace',
    'GaussianGroupedDirectional',
    'GaussianGroupedEquilibrium',
    'GaussianHeatMap',
    'GaussianInverseAging',
    'GaussianInverseApproximation',
    'GaussianInverseCertificate',
    'GaussianInverseDerivative',
    'GaussianInverseEvolution',
    'GaussianInversePredictor',
    'GaussianInverseResult',
    'GaussianInverseTubeCertificate',
    'GaussianJointDirectional',
    'GaussianPathSign',
    'GaussianPathSignBound',
    'GaussianPolyhedralDirectional',
    'GaussianPolyhedralEquilibrium',
    'GaussianPolyhedralFace',
    'GaussianPolyhedralMinimum',
    'GaussianPolyhedralParameterVJP',
    'GaussianPolyhedralPath',
    'GaussianPolyhedralPathRadius',
    'GaussianPolyhedralPiece',
    'GaussianPolyhedralReducedMinimum',
    'GaussianPolyhedralReducedSensitivity',
    'GaussianPolyhedralReduction',
    'GaussianPolyhedralUnique',
    'GaussianPosteriorDomain',
    'GaussianProjectedEquilibrium',
    'GaussianProjectionMinimum',
    'GaussianSeriesCertificate',
    'GaussianTargetVJPCertificate',
    'GaussianTwoCenterMinimum',
    'HeatEnclosure',
    'HermiteExpansion',
    'InconsistentAffineConstraints',
    'Inertia',
    'Interval',
    'IntervalArray',
    'IntervalArrayLike',
    'IntervalFn',
    'IntervalLDLT',
    'IntervalLike',
    'IntervalMLP',
    'InvariantSubspaceCertificate',
    'JacobianEnclosure',
    'KantorovichAccept',
    'KrawczykCertificate',
    'LaguerreExpansion',
    'LehmannCertificate',
    'LinearPDE',
    'LohnerSet',
    'MVJet',
    'MeanEvaluation',
    'MeanInverseCertificate',
    'MeanInverseResult',
    'MeanMap',
    'NKBounds',
    'NuLike',
    'PINNErrorCertificate',
    'PI_IV',
    'PerronGapCertificate',
    'PolyhedralProjection',
    'PolyhedralProjector',
    'QuadEstimate',
    'RadiiCertificate',
    'RecommendedAction',
    'SerializedOperator',
    'SeriesProblem',
    'SeriesRadiiResult',
    'SeriesSymbol',
    'SpectralGapCertificate',
    'SpectralProblem',
    'SpectralRadiiResult',
    'Symbol',
    'SymmetricGapCertificate',
    'TaylorModel',
    'TaylorModelMV',
    'TaylorSeries',
    'ValidatedFourierSeries',
    'ValidatedSeries',
    'VectorField',
    'Wavevector',
    'WidthLaw',
    'advection_diffusion',
    'affine_jet',
    'antiderivative_jet',
    'aposteriori_error_certificate',
    'approximate_gaussian_inverse',
    'atan_iv',
    'atan_point',
    'backend_name',
    'banded_row_sum_bound_nu',
    'banded_tail_inverse_bound',
    'bernoulli_number_exact',
    'bernoulli_polynomial_exact',
    'besseli_iv',
    'besseli_point',
    'birkhoff_contraction_ratio',
    'birkhoff_projective_diameter',
    'bisection_bracket',
    'blob',
    'blob_gradient',
    'certificate_mode',
    'certified_block_operator_gap',
    'certified_boundary_residual',
    'certified_condition_number',
    'certified_damping',
    'certified_ftc_residual',
    'certified_geometric_series_sum',
    'certified_interior_residual',
    'certified_invariant_subspace',
    'certified_max_eigenvalue',
    'certified_min_eigenvalue',
    'certified_partials',
    'certified_partials_subdivided',
    'certified_perron_spectral_gap',
    'certified_ratio_series_sum',
    'certified_regularization_error',
    'certified_residual_bound',
    'certified_sign_change',
    'certified_spectral_gap',
    'certified_symmetric_spectral_gap',
    'certify_gaussian_affine_aging',
    'certify_gaussian_affine_approximation',
    'certify_gaussian_affine_evolution',
    'certify_gaussian_affine_inverse',
    'certify_gaussian_affine_parameter_vjp',
    'certify_gaussian_affine_predictor',
    'certify_gaussian_affine_stationarity',
    'certify_gaussian_affine_target_vjp',
    'certify_gaussian_aging_derivative',
    'certify_gaussian_bias_inverse_series',
    'certify_gaussian_collinear_minimum',
    'certify_gaussian_critical_chart',
    'certify_gaussian_entropy_bias',
    'certify_gaussian_entropy_bias_face',
    'certify_gaussian_entropy_bias_region',
    'certify_gaussian_entropy_face',
    'certify_gaussian_entropy_joint_face',
    'certify_gaussian_entropy_joint_region',
    'certify_gaussian_entropy_minimum',
    'certify_gaussian_entropy_residual',
    'certify_gaussian_entropy_sensitivity',
    'certify_gaussian_entropy_unique',
    'certify_gaussian_grouped_directional',
    'certify_gaussian_grouped_equilibrium',
    'certify_gaussian_inverse',
    'certify_gaussian_inverse_aging',
    'certify_gaussian_inverse_derivative',
    'certify_gaussian_inverse_evolution',
    'certify_gaussian_inverse_predictor',
    'certify_gaussian_inverse_series',
    'certify_gaussian_inverse_tube',
    'certify_gaussian_joint_directional',
    'certify_gaussian_polyhedral_directional',
    'certify_gaussian_polyhedral_equilibrium',
    'certify_gaussian_polyhedral_minimum',
    'certify_gaussian_polyhedral_parameter_vjp',
    'certify_gaussian_polyhedral_path_radius',
    'certify_gaussian_polyhedral_reduced_minimum',
    'certify_gaussian_polyhedral_reduced_sensitivity',
    'certify_gaussian_polyhedral_unique',
    'certify_gaussian_posterior_domain',
    'certify_gaussian_projected_equilibrium',
    'certify_gaussian_projection_minimum',
    'certify_gaussian_target_vjp',
    'certify_gaussian_two_center_minimum',
    'certify_mean_inverse',
    'certify_zero_radii',
    'clamped_biharmonic_green_kernel',
    'clamped_biharmonic_inverse_norm',
    'clamped_cubic_contraction',
    'clear_libm_fallback_used',
    'clenshaw_curtis_integral',
    'collatz_wielandt_perron_bounds',
    'companion_matrix',
    'compile_gaussian_affine_adjoint',
    'compile_gaussian_collinear_reduction',
    'compile_gaussian_critical_face',
    'compile_gaussian_grouped_critical_face',
    'compile_gaussian_polyhedral_face',
    'compile_gaussian_polyhedral_path',
    'compile_gaussian_polyhedral_reduction',
    'compile_polyhedral_projector',
    'compose_jet',
    'compose_jet_mv',
    'conditioning_certificate',
    'conjugate_poisson',
    'conjugate_poisson_deriv',
    'conjugate_poisson_deriv_matrix',
    'conjugate_poisson_matrix',
    'constant_coefficient_band',
    'constant_jacobian',
    'constant_symbol',
    'constant_tail_inverse_bound',
    'continue_dfinite',
    'continue_dfinite_path',
    'convolve',
    'cos_iv',
    'cos_matrix',
    'cos_point',
    'cosh_iv',
    'count_eigenvalues_below',
    'derivative_jet',
    'diagnose_width',
    'digamma_iv',
    'dot',
    'ell1_nu_norm',
    'enclose_burgers_jet',
    'enclose_heat_derivative',
    'erf_iv',
    'erf_point',
    'euler_maclaurin_quadrature',
    'euler_maclaurin_sum',
    'euler_number_exact',
    'evaluate_complex_polynomial',
    'evaluate_exp_polynomial',
    'evaluate_rational_field',
    'evaluate_residual',
    'evaluate_series_residual',
    'even_profile',
    'even_profile_deriv',
    'even_profile_tail_constant',
    'exp_iv',
    'finite_band_row_sum_bound',
    'fourier_nu_norm',
    'from_intervals',
    'ftc_certificate',
    'gauss_cdf_iv',
    'gauss_cdf_point',
    'gauss_legendre_integral',
    'gauss_legendre_nodes',
    'gaussian_weight',
    'generalized_bernoulli_exact',
    'generalized_eigenvalue_enclosure',
    'geometric_band_row_sum_bound',
    'geometric_series_closed_form',
    'geometric_tail_bound',
    'geometric_tail_enclosure',
    'hardy_angle',
    'hardy_even',
    'hardy_even_dalpha',
    'hardy_even_deriv',
    'hardy_even_deriv_matrix',
    'hardy_even_deriv_n',
    'hardy_even_deriv_n_iv',
    'hardy_even_matrix',
    'hardy_even_profile',
    'hardy_even_profile_deriv',
    'hardy_odd',
    'hardy_odd_dalpha',
    'hardy_odd_deriv',
    'hardy_odd_deriv_matrix',
    'hardy_odd_deriv_n',
    'hardy_odd_deriv_n_iv',
    'hardy_odd_matrix',
    'hardy_omega_velocity_atom',
    'hardy_omega_velocity_atom_iv',
    'hardy_pair',
    'hardy_radius',
    'hardy_tail_constant',
    'helmholtz',
    'hermite_coeffs_exact',
    'hermite_function',
    'hermite_function_normalized',
    'hermite_poly_coeffs_exact',
    'hilbert_circulant',
    'hilbert_even_profile',
    'hilbert_even_profile_deriv',
    'hilbert_hardy_even_profile',
    'hilbert_hardy_even_profile_deriv',
    'hilbert_of_conjugate',
    'hilbert_of_hardy_even',
    'hilbert_of_hardy_even_deriv_n',
    'hilbert_of_hardy_odd',
    'hilbert_of_hardy_odd_deriv_n',
    'hilbert_of_poisson',
    'hilbert_symbol',
    'hilbert_tail_bound',
    'hoeffding_enclosure',
    'horner_interval',
    'hull',
    'identity_jet',
    'identity_matrix',
    'inf_norm_matrix',
    'inf_norm_vector',
    'integrate_complex_ivp',
    'integrate_ivp',
    'interval_ldlt_factor',
    'interval_ldlt_factor_array',
    'interval_ldlt_inertia',
    'interval_ldlt_inertia_array',
    'interval_ldlt_pivots',
    'interval_ldlt_pivots_array',
    'interval_matrix_exp',
    'interval_newton',
    'interval_solve',
    'interval_triangular_solve',
    'inverse_gaussian_affine',
    'inverse_gaussian_heat',
    'inverse_mean_map',
    'is_positive_definite',
    'is_positive_definite_array',
    'jet_gradient',
    'jet_hessian',
    'jet_laplacian',
    'jet_multiply',
    'jet_partials',
    'jet_to_tower',
    'kantorovich_accept_step',
    'krawczyk_certificate',
    'krawczyk_search',
    'laguerre_function',
    'laguerre_poly_coeffs_exact',
    'laguerre_weight',
    'laplace',
    'laplacian_symbol',
    'laplacian_tail_inverse_bound',
    'layer_jet',
    'layer_jet_mv',
    'lehmann_maehly_lower_bounds',
    'leray_blob_field',
    'leray_divergence_residual',
    'leray_symbol',
    'lhopital_ratio_iv',
    'libm_fallback_used',
    'linear_field',
    'ln_iv',
    'ln_point',
    'log_gamma_iv',
    'lohner_flow',
    'lohner_step',
    'mat_sub',
    'matmul',
    'matvec',
    'measured_critical_width',
    'measured_mean_value_width',
    'measured_taylor_remainder',
    'midpoint_integral',
    'minimize_box',
    'mlp_jet',
    'mlp_jet_mv',
    'naive_interval_flow',
    'neg_i_power',
    'neumann_inverse_norm_bound',
    'new_noise_symbol',
    'newton_kantorovich_bounds',
    'newtonian_potential',
    'normalize_operator',
    'operator_comparison_bounds',
    'parametric_complex_interval_newton',
    'path_jet',
    'pochhammer',
    'pochhammer_iv',
    'poisson',
    'poisson_kernel',
    'poisson_kernel_deriv',
    'poisson_matrix',
    'poisson_primitive',
    'poisson_primitive_matrix',
    'polynomial_sqrt2_maps',
    'positive_definite_pivots_certificate',
    'potential_gradient',
    'predicted_width',
    'qr_gram_schmidt',
    'quadratic_radii_certificate',
    'radii_polynomial_certificate',
    'radii_polynomial_residual_certificate',
    'rayleigh_quotient',
    'realify_complex_matrix',
    'require_rigorous_backend',
    'restrict_gaussian_heat',
    'riesz_double_blob',
    'riesz_symbol',
    'riesz_tail_bound',
    'ritz_upper_bound',
    'romberg_integral',
    'rounding_floor_witness',
    'screened_poisson',
    'sech_iv',
    'sech_poly_coeffs_exact',
    'select_accepted_params',
    'series_mul',
    'series_radii_certificate',
    'set_strict_backend',
    'sigma_tower_interval',
    'sigma_value_interval',
    'sigmoid_iv',
    'sigmoid_poly_coeffs_exact',
    'simpson_integral',
    'sin_iv',
    'sin_matrix',
    'sin_point',
    'sinh_iv',
    'softplus_iv',
    'solve_gaussian_entropy',
    'sparse_matvec',
    'spectral_residual_norm',
    'sqg_blob',
    'sqg_blob_gradient',
    'sqg_blob_l2_inner',
    'sqg_riesz',
    'sqg_stream',
    'sqg_stream_gradient',
    'sqg_velocity',
    'sqg_velocity_divergence_residual',
    'strict_backend',
    'sum_intervals',
    'symmetric_eigenvalue_residual_enclosure',
    'tail_inverse_bound_from_banded',
    'tanh_iv',
    'tanh_poly_coeffs_exact',
    'tanh_sinh_estimate',
    'temple_lower_bound',
    'temple_lower_bound_vector',
    'to_interval_matrix',
    'tower_to_jet',
    'trapezoid_integral',
    'trapezoid_sum',
    'uniform_nodes',
    'width_law',
]
