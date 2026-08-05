# omnibias-difference

**Status: Alpha (0.1.0a1).**

The **founding `delta -> 0` register** of omnibias: discrete calculus and analytic
combinatorics read off the closed-form derivative towers. This is *the* bias
collapse the library is named for -- `K` biases on a difference stencil (spread
`delta`), signs `s_k = (-1)^(K-k) C(K-1, k-1) / delta^(K-1)`, so

```
f_K(z) = sum_k s_k * sigma(z + b_k)  ->  sigma^(K-1)(z + b_mean)   as delta -> 0
```

The many biases coalesce onto one value and the finite difference *becomes a
derivative*. The closed-form tower evaluates that limit **exactly**, with no
`1/delta^(K-1)` catastrophic cancellation.

## Capabilities

- **Certified finite-difference -> derivative extraction.** The rigorous
  interval-tower enclosure of `sigma^(n)(z)` (`omnibias.core.verified`), the
  numerical finite-difference estimate, and a *certified* truncation-error bound
  proving the estimate collapses into the enclosure as `delta -> 0`.
- **Umbral / Sheffer sequence calculus.** The forward-difference operator, Newton
  forward-difference interpolation, the falling/rising-factorial <-> monomial
  change of basis (the Stirling transforms), and Appell / Sheffer sequences.
- **Asymptotic-coefficient reading.** Stirling numbers (both kinds) off the
  Bell / Faa di Bruno tower, Bernoulli numbers off the `tanh` tower, and Euler
  (secant) numbers off the `sech` tower -- exact `int` / `Fraction`, each sealed
  in a tightest outward-rounded interval, with asymptotic formulas.
- **Bit-identical torch / jax twins** for the finite-difference stencil operator
  (`omnibias.difference.torch` / `omnibias.difference.jax`).

## What this is and is not

- Extraction is **closed-form** (the exact towers + exact integer/rational
  coefficients). The finite-difference estimate and the mpmath comparison values
  are **numerical**. Every result says which register it is in.
- This is the `delta -> 0` **founding bias collapse** (a smooth *derivative*),
  **not** `beta -> inf` **temperature collapse**, the penalty of `omnibias-convex` /
  `-control` / `-routing` (a 0/1 feasibility step). Same word, different limit;
  see `docs/theory.md` and the `omnibias-dev-core-concepts` skill.
- The pure-Python core depends only on `omnibias-core`; the stencil twins need
  `omnibias-torch` / `omnibias-jax`.

## Public API

```python
from omnibias.difference import (
    certified_derivative_enclosure, finite_difference_estimate, certified_fd_error,
    stirling_second, stirling_first_signed, bell_number,
    bernoulli_number, bernoulli_polynomial,
    euler_number, eulerian_number,
    forward_difference, newton_forward_coeffs, monomial_to_falling, falling_to_monomial,
    bernoulli_asymptotic, euler_asymptotic, bell_number_asymptotic,
)
```

## Tests

```bash
python -m pytest packages/omnibias-difference/tests -q
```

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
