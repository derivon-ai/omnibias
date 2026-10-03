# omnibias-sos

Polynomial positivity certificates.

`omnibias.sos` provides sum-of-squares and constrained positivity certificates.

- `Polynomial`, `MonomialBasis`, `SOSProblem`: polynomial problem data.
- `certify_sos`, `certify_sos_rational`: construct checked decompositions.
- `certify_nonneg_on_set`: constrained positivity.
- `lean_check_sos`: check supported finite obligations with the optional Lean kernel.

A valid decomposition proves the stated polynomial claim on its specified
set. Failure to find a decomposition is inconclusive. Numerical solver output,
a sealed digest and a successful rigorous/formal check are distinct results.

Install this distribution with `pip install omnibias-sos`; select its
backend extras when needed. See [guarantees](../guarantees.md).

<!-- BEGIN GENERATED API INVENTORY -->

Version **0.1.0a1** · Python **>=3.10** · **3 - Alpha** · AGPL-3.0-or-later **or commercial**

<details markdown="1">
<summary>Public modules and top-level exports</summary>

[Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-sos/src/omnibias/sos). Modules below are relative to `omnibias.sos`; underscored modules are internal.

`certify`, `combinatorial`, `conditions`, `families`, `formal`, `honesty`, `inequality`, `monomials`, `positivstellensatz`, `problem`, `rounding`, `solve`.

Exports from `omnibias.sos`:

`DEFAULT_DENOMINATORS`, `Exponent`, `FINITE_DIM_SYSTEM`, `GALERKIN_TRUNCATION`, `GLOBAL_POLYNOMIAL`, `MonomialBasis`, `Polynomial`, `PolynomialInequalityBackend`, `PositivstellensatzCertificate`, `RationalPolynomial`, `SOSCertificate`, `SOSMultiplier`, `SOSProblem`, `SOSScope`, `arrangement_adapted_basis`, `certify_nonneg_on_set`, `certify_sos`, `certify_sos_rational`, `degree_reduction_report`, `gram_products`, `gram_to_poly`, `honesty_labels`, `is_nonneg_on_set`, `is_sos`, `is_theorem_prover_verified`, `lean_available`, `lean_check_sos`, `monomial_basis`, `named_adapted_problems`, `rational_gram`, `seal_positivstellensatz_certificate`, `seal_sos_certificate`.

</details>

<!-- END GENERATED API INVENTORY -->
