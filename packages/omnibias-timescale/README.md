# omnibias-timescale

**Status: Alpha (0.1.0a1).**

Time-scale (Hilger) calculus: one derivative that unifies the continuous and discrete
registers of omnibias.

## Capabilities

- **`TimeScale`** (`omnibias.timescale`): the real line `R`, the uniform mesh `hZ`, the
  quantum scale `q^Z ∪ {0}` (`q > 1`), or an arbitrary finite set -- with the forward /
  backward jump operators `sigma`, `rho`, the graininess `mu = sigma - t`, and the
  right/left dense/scattered classification.
- **`delta_derivative` / `nabla_derivative`**: the Hilger delta (forward) and nabla
  (backward) derivatives. The activation-aware `delta_derivative_tower` **dispatches**:
  the closed-form omnibias derivative tower on `R`, the omnibias-difference forward
  difference on `hZ`, and the omnibias-qcalculus Jackson q-derivative on `q^Z` -- three
  registers, one operator.
- **`delta_integral`**: the time-scale integral (an exact graininess-weighted sum on a
  discrete scale; a quadrature on `R`), satisfying the fundamental theorem
  `int_a^b f^Delta Delta t = f(b) - f(a)`.
- **`hilger_exponential`** and the **`circle_plus` / `circle_minus`** regressive group,
  with the cylinder transformation; and **`solve_linear_dynamic`** for `y^Delta = p y + r`.
- **Backend twins**: bit-identical PyTorch / JAX delta-derivatives of the activation
  dictionary.

## The `mu -> 0` limit (the founding sense)

As the graininess `mu -> 0` the time scale becomes `R` and the delta derivative becomes the
ordinary derivative: `f^Delta -> f'`, `hilger_exponential -> exp`. This is the **founding
`delta -> 0` bias collapse** (a finite difference of many biases becoming the smooth
derivative `sigma^(K-1)`), *generalized* to a variable mesh. It is the derivative sense of
"collapse" -- **not** the `beta -> inf` feasibility penalty, and distinct from the `q -> 1`
limit of `omnibias-qcalculus` (which fixes the mesh and varies the deformation parameter).

## Honesty labels

- **closed-form**: `delta_derivative_tower` on `R` (the omnibias tower) and the exact
  graininess-weighted `delta_integral` on a discrete scale.
- **numerical**: the difference / Jackson quotients on `hZ` / `q^Z` (finite differences of
  the libm activation value) and the `R` quadrature.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
