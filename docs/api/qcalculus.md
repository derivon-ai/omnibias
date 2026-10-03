# omnibias-qcalculus

q-calculus primitives.

`omnibias.qcalculus` contains q-numbers and polynomial/q-difference operations.

- `q_bracket`, `q_factorial`, `q_binomial`: coefficient algebra.
- `q_derivative`, `q_derivative_poly`: q-derivatives.
- `q_integral`, `q_antiderivative_poly`: q-integration.
- `q_exp`, `q_exp_big`: q-exponential families.

The limit `q → 1` relates these operations to ordinary calculus. Check each
function's supported q-domain, convergence assumptions and truncation controls
before using a numerical series result.

Install this distribution with `pip install omnibias-qcalculus`; select its
backend extras when needed. See [guarantees](../guarantees.md).

<!-- BEGIN GENERATED API INVENTORY -->

Version **0.1.0a1** · Python **>=3.10** · **3 - Alpha** · Apache-2.0

<details markdown="1">
<summary>Public modules and top-level exports</summary>

[Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-qcalculus/src/omnibias/qcalculus). Modules below are relative to `omnibias.qcalculus`; underscored modules are internal.

`jax`, `jax.hybrid`, `torch`, `torch.hybrid`, `umbral`.

Exports from `omnibias.qcalculus`:

`QOMBUConfig`, `QShefferClass`, `basic_hypergeometric`, `basic_hypergeometric_enclosure`, `q_antiderivative_poly`, `q_appell_sequence`, `q_associated_sequence`, `q_bernoulli`, `q_binomial`, `q_binomial_poly`, `q_binomial_transform`, `q_bracket`, `q_bracket_poly`, `q_delta_operator_apply`, `q_derivative`, `q_derivative_poly`, `q_euler`, `q_exp`, `q_exp_big`, `q_exp_enclosure`, `q_factorial`, `q_falling_factorial_coeffs`, `q_falling_to_monomial`, `q_integral`, `q_inverse_binomial_transform`, `q_monomial_to_falling`, `q_newton_forward_coeffs`, `q_newton_forward_value`, `q_ombu_forward`, `q_ombu_limit`, `q_pincherle_derivative`, `q_pochhammer`, `q_rising_factorial_coeffs`, `q_sheffer_classify`, `q_sheffer_sequence`, `q_stirling_first_signed`, `q_stirling_first_signed_row`, `q_stirling_first_unsigned`, `q_stirling_second`, `q_stirling_second_row`, `q_umbral_composition`.

</details>

<!-- END GENERATED API INVENTORY -->
