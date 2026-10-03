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
