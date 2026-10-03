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
