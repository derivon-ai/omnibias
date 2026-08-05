# omnibias-boolean

Differentiable Boolean algebra. The package has two layers:

- **Exact `_core`** (pure Python, no backend imports): truth tables, the GF(2)
  Mobius/ANF transform and algebraic degree, the Walsh-Hadamard / Fourier
  spectrum and influences, Boolean differential calculus (Boolean derivative,
  set-derivative, and integral with a free "+C" function), the real multilinear
  extension, and reproductive Boolean equation/unification solving (eliminant +
  Loewenheim general solution) with a GF(2) Gaussian-elimination fast-path.
- **Differentiable backends** (`torch` / `jax`): soft logic gates (the multilinear
  extension -- exact on the cube vertices), a jet-based spectrum/influence engine
  that reads ANF/Walsh coefficients as the mixed partials of the multilinear
  extension, a beta-annealed propose-and-verify equation/system solver, and
  spectral-design losses.

!!! note "Honesty guardrails"
    `_core` transforms are exact (integer / GF(2)); the backends are
    **differentiable relaxations** with no completeness guarantee. The solver is
    **propose-and-verify**: it relaxes and optimizes, then checks the exact
    Boolean system. See the
    [RSA-limitation study](../cookbook/rsa-limitation.md) for an explicit negative
    result -- this is not a cryptanalytic tool.

See the [Boolean-equations cookbook](../cookbook/boolean-equations.md) for the
discrete <-> continuous derivative bridge and a worked reproductive solution.

## Core (exact, pure Python)

::: omnibias.boolean
    options:
      show_root_heading: false
      heading_level: 3
      members_order: source

## Verified spectra (rigorous interval twin)

The `_iv` functions are the outward-rounded interval twins of the Walsh / Fourier
and real-multilinear transforms, plus certified linear- and differential-bias
figures of merit (linearity, nonlinearity, linear bias, autocorrelation,
differential bias, absolute indicator). They exist for the **real-valued** spectra
omnibias unlocks — the coefficients of a differentiable `tanh(beta x)` /
`sigmoid(beta x)` gate relaxation, a noisy/measured truth table, or any function
whose values are themselves intervals — and provably *contain* the true spectrum
despite floating-point round-off. (The exact `{0,1}` integer transforms above are
already bit-exact and need no enclosure.)

::: omnibias.boolean._core.verified
    options:
      show_root_heading: false
      heading_level: 3

## S-box figure-of-merit analysis

!!! note "Design metrics, not cryptanalysis"
    `omnibias.boolean.cipher` scores an S-box's published **figures of merit**; it
    does **not** break ciphers, recover keys, or mount attacks. See
    [scope & guarantees](../scope-and-guarantees.md) §6.

`omnibias.boolean.cipher` analyses a vector Boolean function (S-box): the
difference-distribution and linear-approximation tables, the differential
uniformity, the linearity / nonlinearity, the algebraic degree, and the exact
higher-order Boolean derivative that bounds an S-box's resistance to higher-order
differential distinguishers (a function of degree `d` is annihilated by every
`(d+1)`-th order derivative). The test-suite checks the published figures of merit
for the AES (differential uniformity 4, nonlinearity 112, degree 7) and PRESENT
(4 / 4 / 3) S-boxes. Each component `<b, S(x)>` is an ordinary Boolean function, so
the rigorous interval bias bounds from the verified layer apply to a
differentiable / noisy S-box.

::: omnibias.boolean.cipher
    options:
      show_root_heading: false
      heading_level: 3

## Ops (torch)

Soft gates, the jet-based spectrum/influence engine, the annealed solver, and the
spectral-design losses.

::: omnibias.boolean.torch.ops
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

The JAX backend (`omnibias.boolean.jax.ops`) is the bit-for-bit twin of the torch
ops; the cross-backend tests assert agreement to `rtol=1e-9` in float64.

Status: Alpha (`0.1.0a1`).
