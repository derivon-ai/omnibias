# omnibias-core

Shared algebra and verified numerics.

Pure Python; independent of tensor frameworks.

- `omnibias.core.polynomials`: shared activation derivative coefficients.
- `omnibias.core.bell`: composition combinatorics.
- `omnibias.core.multi_index`: ordering and products for mixed Taylor jets.
- `omnibias.core.spec.ActivationSpec`: activation metadata and derivative providers.
- `omnibias.core.verified`: interval and Taylor-model enclosures.
- `omnibias.core.proof.certificate`: canonical certificates and digest checks.

Tensor implementations consume these coefficients. Extend a recurrence here
once, then update and test each backend. An integrity digest is separate from
a successful mathematical checker run.

Install this distribution with `pip install omnibias-core`; select its
backend extras when needed. See [guarantees](../guarantees.md).

<!-- BEGIN GENERATED API INVENTORY -->

Version **0.5.0rc1** · Python **>=3.10** · **4 - Beta** · Apache-2.0

<details markdown="1">
<summary>Public modules and top-level exports</summary>

[Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-core/src/omnibias/core). Modules below are relative to `omnibias.core`; underscored modules are internal.

`band_embed`, `bell`, `block_search`, `composed_curvature`, `confluence`, `contraction`, `cubature`, `frames`, `ftc`, `implicit`, `integral_kernel`, `jets`, `line_search`, `locus`, `mollifier`, `multi_index`, `multipack`, `parameter_jets`, `polynomials`, `proof`, `proof.catalog`, `proof.certificate`, `proof.condition`, `proof.discovery`, `proof.inequality`, `proof.lean_check`, `proof.lean_lock`, `proof.lift`, `proof.obligations`, `proof.obligations.rational_stencil`, `proof.observe`, `proof.realization_algebra_replay`, `proof.realization_replay`, `proof.replay`, `realization`, `realization.algebraic`, `realization.membership`, `realization.polynomial`, `realization.rank`, `realization.schema`, `realization.transition`, `realization.witness`, `refine`, `scale`, `scan`, `spec`, `spectral_design`, `tanh_method`, `transfer`, `transforms_pde`, `uncertainty`, `verified`, `verified.affine`, `verified.asymptotic_jet`, `verified.banded`, `verified.clamped_biharmonic`, `verified.coeffs`, `verified.complex_interval`, `verified.complex_ode`, `verified.complex_rootfind`, `verified.conditioning`, `verified.dfinite_continuation`, `verified.eig`, `verified.eig_operator`, `verified.enclosure_collapse`, `verified.euler_maclaurin`, `verified.fourier`, `verified.frobenius`, `verified.ftc`, `verified.hardy_line`, `verified.hermite_basis`, `verified.interval`, `verified.interval_array`, `verified.invariant_subspace`, `verified.jet`, `verified.jet_flow`, `verified.jet_mv`, `verified.kantorovich`, `verified.laguerre_basis`, `verified.linalg`, `verified.linalg_array`, `verified.line`, `verified.lohner`, `verified.ode`, `verified.pde_certificate`, `verified.quadrature`, `verified.radii_series`, `verified.radii_spectral`, `verified.riesz`, `verified.rootfind`, `verified.sampled`, `verified.sequence_space`, `verified.series`, `verified.sigma`, `verified.spectral`, `verified.sqg`, `verified.taylor_model`, `verified.taylor_model_mv`, `verified.tm_neuron`, `verified.transcend`, `weight_loss_jet`.

Exports from `omnibias.core`:

`ActivationSpec`, `AffineSet`, `BandPlan`, `BankSpec`, `BlockKind`, `BlockSpec`, `ComposedCurvatureConfig`, `ComposedCurvatureReport`, `DEQConfig`, `DEQNotContractive`, `DEQSolverUnknown`, `EffectiveOperator`, `EqualitySystem`, `FlowSystem`, `FrameSpec`, `GradientSecant`, `Indicator`, `JetLineSearchConfig`, `Layer`, `LineSearchResult`, `LinearizingTransform`, `MollifierSpec`, `MomentSystem`, `MultiPackSpec`, `NewtonResult`, `NthDerivativeFn`, `PackSpec`, `QuadratureRule`, `RefinePolicy`, `RefineReport`, `RefinedPack`, `ScaleBand`, `ScaledPack`, `TensorFn`, `TensorT`, `TransformKernels`, `TravellingWaveAnsatz`, `UnitTerm`, `WeightLossJetSpec`, `admissibility_constant`, `affine_locus`, `alpha_for_peak`, `apply_block_step`, `apply_rule`, `arrangement_w_block`, `assert_zero_perturbation`, `bell_complete`, `bell_number`, `bell_partial`, `central_stencil_weights`, `certified_band_gap`, `certified_error`, `certified_truncation_radius`, `certify_locus_point`, `chain_rule_mse_blocks`, `coarse_grain_linear`, `cole_hopf_jet`, `cole_hopf_u`, `compile_bank`, `contact_residual`, `default_block_config`, `design_band_plan`, `design_order`, `design_rule`, `dilated_sigma_n`, `eigh_symmetric`, `eval_gaussian_derivative`, `eval_sigma_derivative`, `eval_tanh_derivative`, `factorial_jet_multiply`, `factorial_jet_reciprocal`, `flow_coefficients`, `gram_matrix`, `hermite_coeffs`, `honesty_payload`, `hp_decision`, `incidence_matrix`, `index_position`, `is_admissible`, `is_holonomic`, `is_poised`, `last_linear_block`, `local_scale_from_derivatives`, `make_tempered_fastpath`, `make_tempered_transforms`, `mish_inner_coeffs`, `moments`, `multi_index_factorial`, `multi_indices`, `multiply_table`, `num_multi_indices`, `ombu_bias_block`, `one_layer_loss`, `one_layer_loss_grad`, `one_layer_loss_hessian`, `one_layer_loss_jet`, `one_layer_newton_direction`, `one_layer_output_jet`, `one_layer_param_count`, `overlap`, `pack_moment`, `pack_one_layer_params`, `peak_frequency`, `peano_kernel`, `polya_condition`, `polynomial_wolfe`, `refine_bank`, `reject_anderson`, `reject_deq_contraction`, `reject_full_parameter_jacobian`, `relative_bandwidth`, `report_exponents`, `rescale_pack`, `resolve_block_mask`, `response_profile`, `run_model_line_search`, `scalar_nest_hessian`, `select_composed_step`, `select_model_step`, `sigmoid_polynomial_coeffs`, `solve_dense`, `solve_rule`, `spectral_radius_inf_bound`, `stiffness_matrix`, `symmetrize`, `tail_bound`, `tanh_polynomial_coeffs`, `target_moments`, `taylor_coeffs_from_derivatives`, `tempered`, `unit_direction_from_mask`, `unpack_one_layer_params`, `vanishing_moments`, `verify_cole_hopf_burgers_jet`, `verify_exact`, `verify_transform`.

</details>

<!-- END GENERATED API INVENTORY -->
