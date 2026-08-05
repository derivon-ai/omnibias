# omnibias-qcalculus

Quantum / *q*-calculus on the omnibias closed-form tower and the founding
`omnibias-difference` register.

- **Exact q-combinatorics** — the q-number `[n]_q`, q-factorial `[n]_q!`, Gaussian /
  q-binomial `[n choose k]_q`, and the q-Pochhammer `(a; q)_n`, in exact `Fraction`
  arithmetic at numeric `q` and as integer polynomials in `q`.
- **Jackson calculus** — the q-derivative `D_q f = (f(qx) - f(x)) / ((q-1)x)` and the
  q-integral (Jackson sum), exact on polynomials and numerical on callables.
- **q-special functions** — the two q-exponentials `e_q`, `E_q` (with the
  `e_q(z) E_q(-z) = 1` identity) and q-deformed Bernoulli / Euler numbers.
- **q-umbral / q-Sheffer calculus** (`umbral`) — the quantum twin of
  `omnibias.difference.umbral`: q-Stirling numbers and the q-falling / q-rising factorial
  bases, the q-binomial and q-Stirling transforms, q-Newton interpolation on the `[j]_q`
  nodes, q-Appell / q-associated / general `q_sheffer_sequence` generation, and the
  q-delta operator `Q = f(D_q)`. Exact `Fraction` arithmetic carrying the closed-form
  q-Sheffer recurrence `Q s_n = [n]_q s_{n-1}`, in the `omnibias.qcalculus.umbral`
  namespace (and flat-re-exported).
- **Basic hypergeometric series** `_r phi_s(a; b; q, z)` with a **certified** geometric
  tail enclosure, alongside the numerical direct-summation baseline.
- **Bit-identical torch / jax twins** of the Jackson q-derivative.

!!! note "The `q -> 1` collapse (a distinct limit)"
    Every q-object reduces to its ordinary-calculus counterpart as `q -> 1`
    (`[n]_q -> n`, `D_q f -> f'`, `e_q -> exp`). This is a **distinct** limit from the
    `delta -> 0` founding bias collapse of `omnibias-difference` and the `beta -> inf`
    **temperature collapse** elsewhere — same spirit, different parameter, never conflated.

Honesty labels: **closed-form / exact** for the q-combinatorics, polynomial Jackson
operators, and q-Bernoulli / q-Euler numbers; **numerical** for the callable Jackson
operators and direct q-series; **numerical (certified)** for the q-exponential and
basic-hypergeometric enclosures. Smoke: `docs/examples/qcalculus_validate.py`.

## Public API

::: omnibias.qcalculus
    options:
      show_root_heading: false
      heading_level: 3
      members_order: source

Status: Alpha (`0.1.0a1`).
