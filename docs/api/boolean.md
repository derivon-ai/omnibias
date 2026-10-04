# omnibias-boolean

Boolean algebra primitives.

`omnibias.boolean` provides exact finite Boolean representations and transforms.

- `TruthTable`, `all_assignments`: enumerate finite inputs.
- `anf_from_truth_table`, `anf_monomials`: algebraic normal forms.
- `boolean_derivative`, `mixed_partial`: Boolean differential operations.
- `gf2_solve`, `solve_system`: finite algebraic solvers.

Optional tensor backends use the binary primitive for smooth gate training.
Exact discrete algebra and differentiable relaxation are separate modes.
Truth-table enumeration scales exponentially with the number of variables;
reserve it for bounded problems and reference checks.

Install this distribution with `pip install omnibias-boolean`; select its
backend extras when needed. See [guarantees](../guarantees.md).

<!-- BEGIN GENERATED API INVENTORY -->

Version **0.1.0a2** · Python **>=3.10** · **3 - Alpha** · Apache-2.0

<details markdown="1">
<summary>Public modules and top-level exports</summary>

[Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-boolean/src/omnibias/boolean). Modules below are relative to `omnibias.boolean`; underscored modules are internal.

`inequality`, `jax`, `jax.ops`, `jax.ops.design`, `jax.ops.gates`, `jax.ops.solver`, `jax.ops.spectrum`, `torch`, `torch.ops`, `torch.ops.design`, `torch.ops.gates`, `torch.ops.solver`, `torch.ops.spectrum`.

Exports from `omnibias.boolean`:

`BooleanAntiderivative`, `BooleanInequalityBackend`, `BooleanSolution`, `GF2Solution`, `GeneralSolution`, `TruthTable`, `absolute_indicator_iv`, `algebraic_degree`, `all_assignments`, `anf_from_multilinear_coeffs`, `anf_from_truth_table`, `anf_monomials`, `anf_to_string`, `assignment`, `autocorrelation_iv`, `bit_to_spin`, `boolean_derivative`, `boolean_derivative_reduced`, `boolean_derivative_set`, `boolean_integral`, `check_truth_table`, `constraint_from_predicate`, `constraints_are_linear`, `differential_bias_iv`, `eliminant`, `equation_from_callables`, `fourier_coeffs`, `fourier_coeffs_iv`, `fourier_influences`, `gf2_solve`, `index_of`, `influences`, `is_independent_of`, `is_satisfiable`, `linear_bias_iv`, `linear_system_rows`, `linearity_iv`, `max_linear_bias_iv`, `mixed_partial`, `mobius_iv`, `multilinear_coeffs`, `multilinear_eval`, `multilinear_eval_from_coeffs`, `nonlinearity_iv`, `num_vars`, `parseval_defect`, `parseval_defect_iv`, `pm1_values`, `reduced_index`, `restrict`, `solution_set`, `solve_for`, `solve_system`, `spin_to_bit`, `system_constraint`, `total_influence`, `truth_table_from_anf`, `truth_table_from_callable`, `truth_table_to_callable`, `values_from_multilinear_coeffs`, `verify_assignment`, `walsh_at_iv`, `walsh_hadamard`, `walsh_hadamard_iv`, `walsh_spectrum`, `walsh_spectrum_iv`.

</details>

<!-- END GENERATED API INVENTORY -->
