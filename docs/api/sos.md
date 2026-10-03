# omnibias-sos

Polynomial positivity certificates.

`omnibias.sos` provides sum-of-squares and constrained positivity certificates.

- `Polynomial`, `MonomialBasis`, `SOSProblem`: polynomial problem data.
- `certify_sos`, `certify_sos_rational`: construct checked decompositions.
- `certify_nonneg_on_set`: constrained positivity.
- `replay_sos_certificate`: replay the certificate's finite checks.

A valid decomposition proves the stated polynomial claim on its specified
set. Failure to find a decomposition is inconclusive. Numerical solver output,
a sealed digest and a successful rigorous/formal check are distinct results.

Install this distribution with `pip install omnibias-sos`; select its
backend extras when needed. See [guarantees](../guarantees.md).
