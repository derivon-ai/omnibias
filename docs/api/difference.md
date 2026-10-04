# omnibias-difference

Finite-difference primitives.

`omnibias.difference` supplies finite-difference stencils, Taylor-error
bounds and exact coefficient algebra.

- `StencilRequest`, `IrregularStencil`, `solve_irregular_stencil`:
  construct irregular stencils.
- `apply_irregular_stencil`: evaluate a stencil on sampled values.
- `certified_irregular_error`: bound truncation error under derivative bounds.
- `finite_difference_estimate`: a finite-step derivative estimate.

Finite steps are approximations. Separate truncation bounds, supplied
regularity assumptions and floating-point error. Use closed-form activation
jets when the represented network supports them.

Install this distribution with `pip install omnibias-difference`; select its
backend extras when needed. See [guarantees](../guarantees.md).

<!-- BEGIN GENERATED API INVENTORY -->

Version **0.1.0a2** · Python **>=3.10** · **3 - Alpha** · Apache-2.0

<details markdown="1">
<summary>Public modules and top-level exports</summary>

[Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-difference/src/omnibias/difference). Modules below are relative to `omnibias.difference`; underscored modules are internal.

`jax`, `recurrence`, `singularity`, `torch`, `umbral`, `validation`.

Exports from `omnibias.difference`:

`DerivBound`, `DerivativeEnclosure`, `DerivativeProofVerdict`, `DifferenceEstimate`, `FiniteDifferenceCertificate`, `IrregularStencil`, `RationalIdentityVerdict`, `ShefferClass`, `StencilRequest`, `TransferEstimate`, `accuracy_order`, `appell_sequence`, `apply_irregular_stencil`, `associated_sequence`, `bell_asymptotic_relative_error`, `bell_dobinski_enclosure`, `bell_number`, `bell_number_asymptotic`, `bell_number_asymptotic_refined`, `bernoulli_asymptotic`, `bernoulli_enclosure`, `bernoulli_number`, `bernoulli_polynomial`, `bernoulli_recurrence_identity`, `bernoulli_sign_certificate`, `binomial_coefficient`, `binomial_transform`, `catalan_asymptotic`, `cauchy_product`, `certified_derivative_enclosure`, `certified_fd_error`, `certified_fd_error_general`, `certified_irregular_error`, `check_derivative_certificate`, `check_identity_certificate`, `compose_series`, `compositional_inverse`, `connection_constants`, `delta_operator_apply`, `derivative_sign_certificate`, `dirichlet_beta_odd_enclosure`, `dominant_pole_coefficient_asymptotic`, `euler_asymptotic`, `euler_enclosure`, `euler_number`, `euler_polynomial`, `euler_recurrence_identity`, `eulerian_number`, `exponential_generating_coeffs`, `falling_factorial_coeffs`, `falling_to_monomial`, `finite_difference_estimate`, `forward_difference`, `inverse_binomial_transform`, `is_poised_exact`, `log_bell_number_asymptotic`, `log_bell_number_asymptotic_refined`, `monomial_to_falling`, `newton_forward_coeffs`, `newton_forward_value`, `offsets_exact`, `ordinary_from_exponential`, `pade_approximant`, `pade_certified_remainder`, `pade_evaluate`, `pade_evaluate_interval`, `physical_weights`, `pincherle_derivative`, `polya_screen`, `power_sum_coeffs`, `rational_ogf_coefficients`, `rational_ogf_growth_base`, `rational_series`, `rational_value_identity`, `recommended_bell_fallback_n`, `riordan_array`, `riordan_inverse`, `riordan_product`, `rising_factorial_coeffs`, `series_reciprocal`, `sheffer_classify`, `sheffer_sequence`, `shift_polynomial`, `sigma_deriv_bound`, `signs_exact`, `singular_template_coefficient`, `solve_irregular_stencil`, `stencil_offsets`, `stencil_signs`, `stirling_first_signed`, `stirling_first_signed_row`, `stirling_first_unsigned`, `stirling_second`, `stirling_second_asymptotic`, `stirling_second_row`, `thiele_evaluate`, `thiele_interpolation`, `transfer_theorem`, `umbral_composition`, `umbral_functional`, `zeta_int_enclosure`, `zeta_negative_odd_identity`.

</details>

<!-- END GENERATED API INVENTORY -->
