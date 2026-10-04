# Numerical maintenance contracts

Load this reference when changing numerical implementations. Package skills
identify the source and tests they own; this file owns the shared contracts.

- Polynomial coefficients have one definition in `omnibias.core.polynomials`;
  combinatorics belong in `core.bell` and `core.multi_index`. Core remains pure
  Python and cannot import a numerical backend.
- Keep corresponding tensor kernels mathematically paired. Preserve the existing
  accumulation order and tested dtype/device tolerances rather than assuming
  shared coefficients guarantee bit identity.
- A directional Taylor coefficient is `f^(k)/k!`; a mixed coefficient is
  `D^alpha f/alpha!`. Use conversion helpers at derivative-facing boundaries.
- Negative derivative orders raise `ValueError`; genuinely unsupported orders
  raise `NotImplementedError`. New tensors use the framework default dtype.
- Preserve parameter gradients through spatial operators. Avoid detached arrays,
  host conversions and data-dependent Python branches inside traced kernels.
- Compare changed values with an independent analytic or high-precision oracle;
  then check parameter gradients and affected backend parity. A comparison
  against another wrapper of the same implementation is not independent.
- Specialized Laplacians, mixed jets and support estimators have different cost
  and output contracts. Preserve mode and resource-budget reporting. An empirical
  estimator diagnostic is not automatically a rigorous probability bound.
- Distinguish full Hessians from Gauss–Newton, Fisher, KFAC and stochastic
  approximations. Optimizer closures and functional update APIs are different
  interfaces; inspect the implementation before substituting them.
- Preserve outward rounding, domain assumptions and refusal states. A digest
  establishes integrity; a formal verification flag requires the corresponding
  checker to pass. Missing toolchains cannot manufacture a pass.

For timing or accuracy claims, use the [benchmark protocol](../performance.md).
For commands, typing, exports and executable examples, use
[contribution instructions](https://github.com/derivon-ai/omnibias/blob/main/CONTRIBUTING.md). Check
[licensing](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md) before adding a dependency, extra or runtime
import across distribution boundaries.
