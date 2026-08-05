# omnibias-boolean

**Status: Alpha (0.1.0a1).**

Differentiable Boolean algebra on top of the omnibias closed-form derivative
towers. Two layers:

- **Exact pure-Python `_core`** (no torch / jax): truth tables, the
  algebraic-normal-form / Reed-Muller transform (GF(2) Mobius), the Walsh-Hadamard
  / Fourier spectrum and influences, Boolean *differential calculus* (the Boolean
  derivative / difference and its integral with a free "constant" `c`), and
  **Boolean equation solving** (the eliminant plus Loewenheim-style *reproductive*
  general solutions with a free parameter).
- **Verified spectra** (`omnibias.boolean._core.verified`, the `_iv` functions):
  outward-rounded interval twins of the Walsh / Fourier and real-multilinear
  transforms, plus certified linear-/differential-bias bounds (linearity,
  nonlinearity, linear bias, autocorrelation, differential bias, absolute
  indicator). These give the **real-valued** spectra of a differentiable gate
  relaxation or a noisy truth table rigorous two-sided bounds despite
  floating-point round-off.
- **S-box figure-of-merit analysis** (`omnibias.boolean.cipher`): the
  difference-distribution and linear-approximation tables, differential
  uniformity, linearity / nonlinearity, algebraic degree, and the exact
  higher-order Boolean derivative that bounds resistance to higher-order
  differential distinguishers. Reproduces the published AES (4 / 112 / 7) and
  PRESENT (4 / 4 / 3) figures of merit. This scores S-box *design metrics*; it is
  **not** cryptanalysis and does not break ciphers or recover keys.
- **Differentiable torch / jax backends**: soft logic gates (the product-t-norm
  multilinear extension, exact on the cube vertices), a **spectrum engine** that
  reads the Mobius / ANF coefficients off the *mixed partials* of the multilinear
  extension via `jet_partials`, a **soft-gate equation/system solver** that anneals
  `beta -> infinity` (reusing `omnibias.binary.BetaAnnealScheduler`) and then
  *verifies the exact Boolean system*, and differentiable **design** losses
  (algebraic degree / influence / target spectrum).

## What this is and is not

- The `_core` transforms are **exact** (integer / GF(2)).
- The backends are **differentiable relaxations**: heuristics with no completeness
  guarantee. Numerically exact derivatives are **not** logically complete
  reasoning. The solver is **propose-and-verify** -- it relaxes and optimizes, then
  checks the exact Boolean system.
- There is **no SAT/SMT engine, no theorem prover, and no cryptographic attack**
  here. See the [RSA-limitation cookbook](../../docs/cookbook/rsa-limitation.md)
  for an honest negative result on why a soft-gate relaxation does **not** shrink
  the factoring search space.

## The bridge

For a Boolean function `f` with multilinear extension `F` on `[0,1]^n`, the
`{0,1}` Mobius (Reed-Muller-over-reals) coefficient of a monomial `prod_{i in S} x_i`
equals the mixed partial `d^|S| F / prod_{i in S} d x_i` evaluated at `0`, and its
reduction mod 2 is the GF(2) ANF coefficient. So a single multivariate jet yields
the whole Boolean spectrum -- the discrete Boolean difference is the arithmetic
derivative of the relaxation.

## Public API (pure core)

```python
from omnibias.boolean import (
    truth_table_from_callable, anf_from_truth_table, algebraic_degree,
    walsh_spectrum, influences, boolean_derivative, boolean_integral,
    eliminant, solve_for, solve_system, multilinear_coeffs,
    # verified (interval) spectra + certified bias bounds
    walsh_hadamard_iv, mobius_iv, fourier_coeffs_iv,
    linearity_iv, nonlinearity_iv, linear_bias_iv, max_linear_bias_iv,
    autocorrelation_iv, differential_bias_iv, absolute_indicator_iv,
)
```

Backends: `omnibias.boolean.torch.ops` and `omnibias.boolean.jax.ops`
(`soft_and/or/not/xor`, `mobius_spectrum`, `walsh_spectrum`, `influences_diff`,
`BooleanSystem`, `solve`, `degree_penalty`, `target_spectrum_loss`).

## Tests

```bash
python -m pytest packages/omnibias-boolean/tests -q
```

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
