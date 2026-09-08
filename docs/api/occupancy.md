# Fermi occupancy and thermodynamic potentials (04-03)

With `z = -beta (e - mu)` the Fermi-Dirac occupancy **is** the sigmoid,
so the whole shipped closed-form derivative tower becomes exact
non-interacting thermodynamics: occupancy derivatives, Fermi entropy,
grand-potential density, and a certified chemical potential all read off
one sigmoid / softplus evaluation at any order, with no finite
difference anywhere.

- Occupancy: `f(e) = sigma(z)`, `d^n f/de^n = (-beta)^n sigma^(n)(z)`,
  `d^n f/dmu^n = (+beta)^n sigma^(n)(z)` (`mu` enters affinely).
- Entropy: `s(z) = softplus(z) - z sigma(z)`, closing on itself via
  `s^(n)(z) = -(z sigma^(n)(z) + (n-1) sigma^(n-1)(z))` for `n >= 1`.
- Grand potential: `omega(z) = -(1/beta) softplus(z)`, and
  `d omega/d mu = -f`, reproducing `dOmega/dmu = -N` exactly.
- `occupancy_window` is the `band` role: an exact antiderivative window
  for a constant density of states.
- `sommerfeld_coefficient(n)` is the classical dimensionless Sommerfeld
  coefficient, closed form via `zeta_even` (`a_1 = pi^2/6`,
  `a_2 = 7 pi^4/360`).

Two collapse senses are named and must not be conflated: the founding
**bias collapse** (spread `delta -> 0`, `sigma^(K-1)`) never appears in
this module -- every identity above is a finite-order derivative read,
not a limit. `beta -> inf` is the founding **temperature collapse**
(feasibility sense, `omnibias.core.collapse.schema`'s founding spec:
`parameter="beta"`, `limit="inf"`, `surviving_object="indicator"`).
`zero_temperature_occupancy` evaluates that limit's value directly as a
named external reference, and this module explicitly does **not**
request a new collapse-registry slot for it: `register_collapse` given a
`beta` / `indicator` spec is refused as a rebrand of the founding
`temperature` collapse, which the module's own test suite asserts.

Status is **shipped**. G1-G5 are CI-gated
(`benchmarks/occupancy.py`). Scope is non-interacting fermions in a
single band with an externally supplied (or absent) density of states --
not a many-body solve, not density-functional theory, no thermodynamic
limit taken, and no phase-transition claim; see `honesty_payload()` on
both the core and verified modules for the permanently-false keys. See
theory spec [04-03](https://github.com/derivon-ai/omnibias/blob/main/theory/04-bridges/03-fermi-occupancy-and-thermodynamic-potentials.md).

Homes: `omnibias.core.occupancy`, `omnibias.core.verified.occupancy`,
plus bit-identical differentiable twins `omnibias.torch.occupancy` and
`omnibias.jax.occupancy`.

!!! note "Module / function name collision"
    `omnibias.core` (and each backend package) re-exports a function
    named `occupancy` at package level. Once the parent package has been
    imported, `<package>.occupancy` is that *function*, not this
    submodule -- ordinary Python attribute shadowing, not a bug. Import
    specific names directly, e.g.
    `from omnibias.core.occupancy import occupancy_derivatives`, which
    resolves against the fully-qualified module in `sys.modules` and is
    unaffected by the shadow.

## Core algebra

::: omnibias.core.occupancy
    options:
      show_root_heading: false
      heading_level: 3

## Verified twin

::: omnibias.core.verified.occupancy
    options:
      show_root_heading: false
      heading_level: 3

## PyTorch twin

::: omnibias.torch.occupancy
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.jax.occupancy
    options:
      show_root_heading: false
      heading_level: 3
