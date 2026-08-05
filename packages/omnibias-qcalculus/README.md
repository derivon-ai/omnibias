# omnibias-qcalculus

**Status: Alpha (0.1.0a1).**

Quantum / *q*-calculus on top of the omnibias closed-form tower and the founding
`omnibias-difference` register.

## Capabilities

- **Exact q-combinatorics** (`omnibias.qcalculus`): the q-number `[n]_q`, q-factorial
  `[n]_q!`, Gaussian / q-binomial `[n choose k]_q`, and the q-Pochhammer `(a; q)_n` --
  in exact `Fraction` arithmetic at numeric `q`, plus exact **integer-polynomial** forms
  in `q` (the Gaussian polynomials via the q-Pascal recurrence).
- **Jackson calculus**: the q-derivative `D_q f = (f(qx) - f(x)) / ((q-1)x)` and the
  q-integral (Jackson sum), both as exact polynomial operators and as numerical operators
  on callables.
- **q-special functions**: the two q-exponentials `e_q`, `E_q` (with the certified
  `e_q(z) E_q(-z) = 1` identity), and q-deformed Bernoulli / Euler numbers.
- **Basic hypergeometric series** `_r phi_s(a; b; q, z)` with a **certified** geometric
  tail enclosure (reusing `omnibias.core.verified`), alongside the plain numerical
  direct-summation baseline.
- **Backend twins**: bit-identical PyTorch / JAX Jackson q-derivatives of the activation
  dictionary.

## The `q -> 1` collapse (honesty)

Every q-object reduces to its ordinary-calculus counterpart as `q -> 1`: `[n]_q -> n`,
`D_q f -> f'`, `e_q -> exp`. This is a **distinct limit** from the `delta -> 0` founding
bias-collapse of `omnibias-difference` (a finite difference becoming a derivative) and
from the `beta -> inf` feasibility penalty of `omnibias-convex` -- same spirit, different
parameter. The three are never conflated.

## Honesty labels

- **closed-form / exact**: q-numbers, q-factorials, Gaussian binomials, q-Pochhammer,
  the polynomial q-derivative / q-antiderivative, and q-Bernoulli / q-Euler numbers
  (exact `Fraction`).
- **numerical**: the callable Jackson derivative / integral and the direct q-series sums.
- **numerical (certified)**: the basic-hypergeometric and q-exponential enclosures, whose
  geometric tails are rigorously bounded.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
