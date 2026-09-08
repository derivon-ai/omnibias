# 04-03 Fermi occupancy and thermodynamic potentials

## 1. Thesis and status

With `z = -beta (e - mu)` the Fermi-Dirac occupancy **is** the sigmoid, so
the whole shipped closed-form derivative tower becomes exact non-interacting
thermodynamics: occupancy derivatives, Fermi entropy, grand-potential
density, and a certified chemical potential all read off one sigmoid /
softplus evaluation, at any order, with no finite difference anywhere.

- **Status**: shipped (core + verified + torch/jax twins; G1-G5 earned)
- **Depends on**: 01-13 (the `band` / `integral` operator roles), the
  shared polynomial tower in `omnibias.core.polynomials`
- **Blocks**: none

## 2. Where it lands

`omnibias-core` (`omnibias.core.occupancy` and
`omnibias.core.verified.occupancy`), plus bit-identical differentiable
twins in `omnibias.torch.occupancy` and `omnibias.jax.occupancy`. This is a
submodule, not a new package: it re-composes the existing sigmoid/softplus
tower and the existing verified quadrature / Kantorovich substrate with no
new dependency tier and no new audience (AGENTS.md "earn independent
existence"; `NEW_PACKAGES_ALLOWED` in `test_theory_homes.py` is empty by
design).

## 3. Prior art in omnibias

- `omnibias.core.polynomials.sigmoid_polynomial_coeffs` -- the exact
  Eulerian-number tower for `sigma^(n)(z)`, shared by every backend.
- `omnibias.core.spec.ActivationSpec` -- `softplus` is already carried as
  the sigmoid's `integral` field.
- `omnibias.core.verified.sigma.sigma_tower_interval` -- rigorous
  enclosure of `sigma^(0..N)(z)` from one transcendental evaluation;
  `"softplus"` is already a supported name (`softplus^(k)(z) =
  sigma^(k-1)(z)` for `k >= 1`).
- `omnibias.core.verified.transcend.softplus_iv` -- the stable rigorous
  softplus enclosure this spec's entropy and grand-potential identities
  are built on.
- `omnibias.core.verified.quadrature.trapezoid_integral` -- certified
  quadrature that refuses to integrate without a derivative bound.
- `omnibias.core.verified.kantorovich.kantorovich_accept_step` -- the
  Newton-Kantorovich accept/reject policy (theory 08-04) reused here,
  unmodified, for a physics root instead of an optimizer step.
- `omnibias.core.verified.dirichlet.zeta_even` -- the closed-form
  `zeta(2m)` special value the Sommerfeld coefficients read off directly.
- `omnibias.core.collapse.schema` -- the founding-collapse registry that
  section 4 shows correctly *refuses* a `beta -> inf` / `indicator` spec.

**Confirmed gap.** No occupancy, entropy, grand-potential, or
finite-temperature electron-count primitive existed anywhere in the tree
before this spec; the tower and verified-quadrature pieces it composes were
all already present and unmodified.

## 4. Mathematics

Let `z = -beta (e - mu)`, `beta > 0` the inverse temperature and `mu` the
chemical potential. Every identity below is exact, closed form, and uses
only the existing sigmoid/softplus tower -- **no limit is taken** in any of
them.

**Occupancy derivatives.** `f(e) = sigma(z)`. Since `dz/de = -beta` and
`dz/dmu = +beta` (affine in both),

```
d^n f / de^n  = (-beta)^n sigma^(n)(z)
d^n f / dmu^n = (+beta)^n sigma^(n)(z)
```

**Fermi entropy.** `ln f = -softplus(-z)` and `softplus(-z) = softplus(z) -
z`, so the entropy per state is exactly

```
s(z) = softplus(z) - z sigma(z)
```

Differentiating once, `s'(z) = -z sigma'(z)`; the Leibniz rule on the
product `z * sigma'(z)` then closes the whole tower on itself, for `n >=
1`:

```
s^(n)(z) = -( z sigma^(n)(z) + (n - 1) sigma^(n-1)(z) )
```

Checks used as the module's own regression tests: `s(0) = ln 2` (the value
verified numerically against `-f ln f - (1-f) ln(1-f)` to machine
precision), `s'(0) = 0`, and `s -> 0` in both tails.

**Grand-potential density.** `omega(z) = -(1/beta) softplus(z)`, and
`d omega / d mu = -sigma(z) = -f`, reproducing `dOmega/dmu = -N` exactly
(one sigmoid evaluation, no finite difference).

**The `band` role: occupancy window.** `omega(z(e))` is a closed-form
antiderivative of `f` with respect to `e` (`d/de[-(1/beta) softplus(z(e))]
= sigma(z) * (-beta) * (-1/beta) = f(e)`), so the exact electron count for
a **constant** density of states over `[e_lo, e_hi]` is

```
integral_{e_lo}^{e_hi} f(e) de = (1/beta) [ softplus(z_lo) - softplus(z_hi) ]
```

nonnegative because `softplus` is increasing and `z` decreases in `e`.

**Sommerfeld coefficients, closed form.** The classical dimensionless
Sommerfeld coefficients are moments of the thermal broadening kernel
`beta sigma'(z)`. Numerically verified (via `mpmath`, 50 digits) before
committing to this spec:

```
integral_{-inf}^{inf} sigma'(z) z^{2n} dz = 2 (2n)! eta(2n),   eta(s) = (1 - 2^{1-s}) zeta(s)
```

so the standard coefficient `a_n = M_{2n} / (2n)! = 2 eta(2n)` gives `a_1 =
pi^2/6`, `a_2 = 7 pi^4/360` (the Ashcroft & Mermin values), read directly
off `omnibias.core.verified.dirichlet.zeta_even` -- a rational multiple of
`pi^{2n}`, comfortably on the `Re(s) > 1` side of the wall that
`zeta_enclosure` guards explicitly (`2n >= 2` always). Odd moments vanish
exactly by the symmetry `sigma'(-z) = sigma'(z)`.

**The limit that is named and not taken.** `beta -> inf` collapses `f` to
the T=0 step, a 0/1 indicator. That is *literally* the founding
**temperature collapse** (`parameter="beta"`, `limit="inf"`,
`surviving_object="indicator"`, the feasibility sense, in
`omnibias.core.collapse.schema`). `omnibias.core.occupancy.
zero_temperature_occupancy` evaluates that limit's value directly and
`honesty_payload()["requests_new_collapse_registry_slot"]` is `False`: a
direct test (`register_collapse` given `parameter="beta"`,
`surviving_object="indicator"`) confirms the registry refuses the slot, so
this module is not smuggling in a tenth named collapse. This is a
different limit from the founding **bias collapse** (spread `delta -> 0`,
`sigma^(K-1)`), which never appears anywhere in this spec: every identity
above is a finite-order derivative read.

## 5. Worked example

`beta = 2`, `mu = 0.5`, `e = 0.3` (all checked numerically to machine
precision while writing this spec):

```
z = -2 * (0.3 - 0.5) = 0.4
f(e) = sigma(0.4) = 0.598687...
s(e) = softplus(0.4) - 0.4 * f(e) = 0.673540...
d^2 f / de^2 = (-2)^2 sigma''(0.4) = -0.189686...
```

Occupancy window, `g0 = 2` constant density of states, `[e_lo, e_hi] =
[-3, 3]`: `occupancy_window = 3.497098...`, so the electron count is `2 *
3.497098 = 6.994196...`, matching a direct `scipy.integrate.quad` reference
to 1e-9.

A certified chemical potential: with `n_target = 6.994196...` fixed at the
true `mu = 0.5` and a Newton trial `mu_bar = 0.51` (a `1%` initial
displacement), `certified_chemical_potential` returns `accepted=True` with
a unique-zero ball of radius `~0.055` around `0.51` -- comfortably
containing the true `mu = 0.5`. At `r_max = 0.02` (a ball too tight for the
tower-derived Lipschitz bound to close) the same call honestly returns
`reason="empty"`: a reported halt, not a failure and not silently widened.

## 6. Proposed API

Already implemented; see `packages/omnibias-core/src/omnibias/core/
occupancy.py`, `.../verified/occupancy.py`,
`packages/omnibias-torch/src/omnibias/torch/occupancy.py`,
`packages/omnibias-jax/src/omnibias/jax/occupancy.py`.

```python
from omnibias.core.occupancy import FermiModel, occupancy, occupancy_derivatives

model = FermiModel(beta=2.0, mu=0.5)
f = occupancy(model, energy=0.3)
f_tower = occupancy_derivatives(model, energy=0.3, order=2)
```

`omnibias.core.verified.occupancy` mirrors every core function as an
`Interval`-valued enclosure and adds `electron_count_enclosure` (certified
`trapezoid_integral` fed a tower-derived second-derivative bound) and
`certified_chemical_potential` (Newton gated by `kantorovich_accept_step`,
with the Lipschitz bound on `d^2 N/d mu^2` also tower-derived, over the
whole trial ball). `omnibias.torch.occupancy` / `omnibias.jax.occupancy`
give bit-identical differentiable twins (`occupancy`,
`occupancy_derivative`, `entropy_per_state`, `grand_potential_density`,
`occupancy_window`) built from the framework-native `sigmoid` / `softplus`
plus Horner over `omnibias.core.polynomials` coefficients -- never
re-derived per backend -- differentiable in `mu` and `beta`, dtype from the
input tensor, `jit` / `vmap` safe.

## 7. Practical use cases

1. **Finite-temperature electronic-structure post-processing.** Given a
   caller-supplied density of states (tight-binding, DFT band structure,
   etc.), get exact-order Sommerfeld corrections and a certified chemical
   potential without a bisection loop or a finite-difference derivative.
2. **Differentiable statistical-mechanics layers.** `occupancy` /
   `entropy_per_state` as a torch/jax layer inside a model whose loss
   depends on an occupation number or a free-energy term, with exact
   gradients in both `mu` and `beta`.
3. **Sanity-checking finite-temperature simulation codes.** The closed-form
   Sommerfeld coefficients (`a_1 = pi^2/6`, `a_2 = 7 pi^4/360`) and the
   certified electron-count enclosure give a machine-precision reference
   to test a numerical solid-state code's low-temperature expansion
   against.
4. **Teaching / notebook use.** One call reproduces the textbook Sommerfeld
   expansion table exactly, rather than a numerically-fit approximation.

## 8. Acceptance gates

Baselines: nested-autodiff derivatives (`torch.autograd.grad` applied
`n` times) for G1, a fine bisection oracle for G3, and a finite-difference
Sommerfeld arm for G4.

- **G1 exactness.** The entropy identity and its `s^(n)` recursion match a
  reference computed from `-f ln f - (1-f) ln(1-f)` to `1e-12`; `d^n
  f/de^n` matches nested-autodiff for `n` up to 4 (the AD arm is
  increasingly finite-difference-noisy past that, which is exactly why the
  closed form exists); `d omega/d mu = -f` holds to `1e-10`.
- **G2 soundness.** Every `omnibias.core.verified.occupancy` enclosure
  contains both a dense deterministic grid of core-module float values and
  a random sample of them, across several `(beta, mu)` configurations --
  the repo's standing enclosure rule.
- **G3 certified chemical potential.** When the Kantorovich ball is
  nonempty, it contains a fine-bisection root of `N(mu) = n_target`; a
  ball that fails to form (too-tight `r_max` for the tower-derived
  Lipschitz bound) is asserted to return `reason="empty"`, a halt rather
  than a failure.
- **G4 certified Sommerfeld.** `sommerfeld_coefficient_enclosure(n)`
  contains `sommerfeld_coefficient(n)` for `n = 1..3`, and both certified
  and float agree with the classical `a_1 = pi^2/6`, `a_2 = 7 pi^4/360` to
  `1e-9` -- beating a finite-difference numerical-integration arm of the
  same moment on both accuracy and the absence of a truncation parameter.
- **G5 honesty non-vacuity.** The permanently-false keys in
  `honesty_payload()` stay false and a static source scan confirms they
  are never set true; a test asserts `omnibias.core.collapse.schema.
  register_collapse` refuses a `parameter="beta"` /
  `surviving_object="indicator"` spec, proving this module is not
  smuggling in a tenth named collapse.

## 9. Benchmark plan

`benchmarks/occupancy.py` mirrors the `omnibias.core.cubature` /
`omnibias.core.scale` shape: one dict per gate (G1-G5), a `gates_block`
aggregate, `provenance`, `write_json` to a committed
`docs/benchmarks/occupancy_smoke.json` by default, and `--full` writing a
multi-seed acceptance artifact under `$OMNIBIAS_SCRATCH`. Wired into the
`cross_backend` CI job beside the existing quadrature / scale-flow smokes.

## 10. Honesty and scope

- **Scope.** Non-interacting fermions in a single band with an externally
  supplied (or absent) density of states -- mirroring how
  `omnibias.core.verified.lattice_ground_state` bounds itself. Not a
  many-body solve, not density-functional theory, no thermodynamic limit,
  no phase-transition claim.
- **Two collapse senses, not conflated.** The founding bias collapse
  (`delta -> 0`) never appears in this module. `beta -> inf` is the
  founding temperature collapse and is evaluated only as a named external
  reference value (`zero_temperature_occupancy`); this module explicitly
  does not request a new collapse-registry slot for it (G5).
  Permanently-false honesty keys: `dft_solved_claim`,
  `many_body_solved_claim`, `interacting_system_claim`,
  `thermodynamic_limit_taken`, `phase_transition_proved`,
  `founding_bias_collapse`, `temperature_collapse`,
  `requests_new_collapse_registry_slot`, `theorem_prover_verified`.
- **Certificate tier.** `electron_count_enclosure` and
  `sommerfeld_moment_enclosure` are sound enclosures (`Interval`, outward
  rounded). `certified_chemical_potential` is a Kantorovich sound-existence
  ball, explicitly *not* `theorem_prover_verified` (no Lean pass is
  attempted); its claim string says so plainly ("not a continuum
  thermodynamic-limit claim").
- **Quadrature choice is deliberate, not incidental.** The certified
  integrals use the composite trapezoid rule (needing only a
  second-derivative bound) rather than Simpson or Gauss-Legendre
  specifically because the Kantorovich Lipschitz bound needs `d^2 N/d
  mu^2`, capping the combined mixed-partial sigmoid-tower order this
  module ever certifies at `4`; a higher-order quadrature rule would push
  that to `6` and demand a much finer (and slower) sub-box partition for
  the same tightness, for no accuracy benefit here.
- **Sommerfeld coefficients use `zeta_even`, not the general
  `zeta_enclosure`.** Both are valid; `zeta_even` is strictly tighter at
  these exact even-integer points (a closed rational multiple of `pi^{2n}`
  rather than a truncated, tail-bounded series), so it is used in
  preference, not as a departure from the `Re(s) > 1` wall.

## 11. Open questions and risks

- The interval Horner evaluation of a high-order sigmoid-tower polynomial
  over a *wide* box suffers a well-known dependency-problem
  overestimation; this module mitigates it with a sub-box hull
  (`_occ_mixed_bound_over_box`) rather than a smarter (Bernstein-form or
  affine-arithmetic) polynomial enclosure. That mitigation is empirically
  tuned (`_BOUND_TARGET_Z_WIDTH`), not derived from a proven convergence
  rate; a pathological `(beta, e_lo, e_hi)` combination could still need
  more sub-boxes than the hard cap (`_BOUND_MAX_SUBDIVISIONS`) allows, in
  which case the returned enclosure is sound but may be looser than a
  caller expects. This is a tightness risk, never a soundness one.
- `certified_chemical_potential` requires the caller to choose `r_max`;
  too generous a value can make the tower-derived Lipschitz bound over the
  whole trial ball needlessly pessimistic and return `reason="empty"` even
  when a tighter ball would succeed. No automatic radius search is
  attempted (mirroring `kantorovich_accept_step`'s own contract) -- a
  caller sweeping `r_max` is the documented workaround.
- **Falsifier.** If a genuine interacting or many-body extension were ever
  attempted on this substrate, every "non-interacting" honesty key above
  would need to flip, and the module would need a new name -- this spec
  makes no claim that the substrate generalizes there.

## 12. Implementation checklist

- [x] `packages/omnibias-core/src/omnibias/core/occupancy.py`
- [x] `packages/omnibias-core/src/omnibias/core/verified/occupancy.py`
- [x] `packages/omnibias-torch/src/omnibias/torch/occupancy.py` and
      `packages/omnibias-jax/src/omnibias/jax/occupancy.py`
- [x] `packages/omnibias-core/tests/test_occupancy.py` and
      `packages/omnibias-core/tests/verified/test_occupancy.py` (G1-G5)
- [x] `tests/test_occupancy_parity.py` (torch/jax parity + jit smoke)
- [x] `benchmarks/occupancy.py` plus `docs/benchmarks/occupancy_smoke.json`
- [x] Docs page (`docs/api/occupancy.md`) and cookbook page plus mkdocs nav
- [x] CI job smoke step
- [x] Regenerated `__all__` blocks in the four touched `__init__.py`
- [x] Index row in `theory/README.md`
