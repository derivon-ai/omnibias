# omnibias-formal

Status: Alpha (0.1.0a1).

Mathlib-backed formal checking for omnibias certificates. This package drives the
Lean project `formal/omnibias-analytic` (the deliberate counterpart to the
Mathlib-free `formal/omnibias-verified-kernel`) to discharge a certificate's
finite obligations against Mathlib.

!!! note "Two trust tiers, never conflated"
    **`theorem_prover_verified`** is earned by the tiny, hand-auditable,
    **Mathlib-free** kernel (`omnibias.core.proof.lean_check` +
    `formal/omnibias-verified-kernel`) over integer `ZInterval` arithmetic.
    **`mathlib_verified`** (this package) is earned by the **Mathlib-backed**
    project over `ℚ` / `ℝ` -- a larger, honestly-labelled trust base. It never sets
    `theorem_prover_verified`, and a green build never implies `unproven_claim`.

## Obligation classes

The bridge re-derives each obligation exactly over `ℚ` and emits Lean that
Mathlib re-checks. It refuses to emit anything it cannot itself confirm holds, so
a green build is meaningful.

- **Enclosed-quantity sign** -- a v1 `interval` certificate, a PDE
  `pinn_aposteriori_error` finite margin, a `taylor_model` centre value, or any
  raw `lo`/`hi` mapping: discharged against the proven, `sorry`-free `enclosed_pos`
  / `enclosed_neg` lemmas, emitting the rational endpoint directly (no
  integer-scaling hack) and closing the endpoint sign with `norm_num`.
- **Positive-definite** -- a `positive_definite` payload: every `LDLᵀ` pivot's
  rational lower endpoint is strictly positive (the `ℚ` analogue of the integer
  kernel's inertia obligation).
- **Newton-Kantorovich / Krawczyk contraction** -- a `radii_polynomial` (also the
  `quadratic_radii` route) or `krawczyk` payload: the genuine rational *polynomial*
  inequalities `p(r) < 0`, `κ(r) < 1`, and strict box containment that the integer
  kernel cannot even state. Existence remains a trusted input for these classes.
  Reusable capability lemmas live in
  `OmnibiasAnalytic/Check/{Positivity,Kantorovich}.lean`.
- **NK / Krawczyk existence** -- an `nk_existence` payload from
  `omnibias.formal.nk.nk_existence_certificate`: the locked quadratic plant
  `x² - 2` on `[5/4, 7/4]`. The generated obligation applies a sorry-free Check
  theorem that concludes a unique real root (Banach on the Newton operator).
  This is a compact-interval statement, not a continuum PDE.
- **Enclosure trace** -- an `enclosure_trace` payload from
  `omnibias.formal.trace.enclosure_trace_certificate`: a locked rational DAG
  (`tower` Horner, `nk` bounds plus unique root, `bernoulli` / named `zetaNeg1`,
  or 2×2 `ldlt`) that Lean replays. Finite `ℚ`-interval arithmetic, not a
  transcendental enclosure or analytic continuation.
- **Named unique zero** -- a `named_zero` payload from
  `omnibias.formal.named.named_zero_certificate`: a locked named polynomial
  (circle ∩ line on `[5/8, 7/8]²`, Hopf radial `r(1-r²)` on `[3/4, 5/4]`, or
  Chebyshev `T₃` on `[3/4, 1]`). The generated obligation applies a sorry-free
  Check theorem that concludes a unique real root on that compact box. Not a
  continuum PDE, Lohner return map, or continuum CCF blow-up.
- **Compact box** -- a `compact_box` payload from
  `omnibias.formal.compact.compact_box_certificate`: a named incompressible
  residual on `[1/2, 1]²`, or the characteristic polynomial of
  `[[13/2, 3/2], [3/2, 13/2]]` (ratio `5/8`). Finite named data, not a
  continuum regularity theorem or a continuum gauge claim.
- **Tower coefficients** -- a `tower_coeffs` payload from
  `omnibias.formal.tower.tower_coeffs_certificate`: the exact integer list of
  one family (`sigmoid` / `tanh` / `sech` / `hermite`) at a finite order, re-derived
  from `omnibias.core.verified.coeffs` and checked against the Lean recurrences
  in `OmnibiasAnalytic.Tower`. This is a coefficient identity, not an
  `iteratedDeriv` replay and not a finite-difference collapse. The link theorems
  (`iteratedDeriv_sigmoid` and siblings) live in the Lean library itself.

The bridge is tamper-evident (reuses `verify_certificate_digest`) and degrades
gracefully when no Lean toolchain is present.

## Verdict tier

`evaluate_with_mathlib` runs the ordinary `ProofMachine.evaluate` pipeline and
attaches the `mathlib_verified` tier via a consumer-side `MathlibVerdict` wrapper
(the core `Verdict` is never edited). If a conjecture *asserts* the
`mathlib_verified` claim but Mathlib does not verify the obligation, a `PROVED`
verdict is downgraded to `BLOCKED` -- the honesty gate, mirroring core's
`theorem_prover_verified` gate. It never sets `theorem_prover_verified` and never
implies `unproven_claim`.

## Scope

Every module in `formal/omnibias-analytic/` is `sorry`-free. The track discharges
finite rational inequalities, a unique root of a named polynomial on a compact
box, replay of a planted rational enclosure DAG, and named compact-box residual
/ finite-matrix gap plants. Infinite analytic
statements -- continuum regularity, asymptotics -- are not expressed here at
all, so they can never be silently discharged here either. A green build
certifies the emitted obligation and nothing beyond it.

## Driving the loop

The `omnibias-dev-formal-agent` skill drives the formal loop through one
deterministic helper, `drive_obligation`, which composes the bridge into a single
actionable pass: `classify_obligation` -> `generate_obligation` -> `lake build`
(via `check_certificate`) -> a `DriveReport`. The report distils a failing build
to its salient lines and carries a rule-based `next_action`; it is plumbing, not
an autonomous prover. The Mathlib tier (`DriveReport.tier`) is set only on a
genuine `lake` pass, is never conflated with `theorem_prover_verified`, and never
implies `unproven_claim`. With no Lean toolchain present it degrades gracefully
(`available=False`) and its `next_action` says how to get a real verdict.

::: omnibias.formal.drive
    options:
      show_root_heading: false
      heading_level: 3

## Bridge

::: omnibias.formal.mathlib_check
    options:
      show_root_heading: false
      heading_level: 3

## NK / Krawczyk existence

::: omnibias.formal.nk
    options:
      show_root_heading: false
      heading_level: 3

## Enclosure traces

::: omnibias.formal.trace
    options:
      show_root_heading: false
      heading_level: 3

## Named unique zeros

::: omnibias.formal.named
    options:
      show_root_heading: false
      heading_level: 3

## Compact boxes

::: omnibias.formal.compact
    options:
      show_root_heading: false
      heading_level: 3

## Tower coefficients

::: omnibias.formal.tower
    options:
      show_root_heading: false
      heading_level: 3

## Verdict augmentation

::: omnibias.formal.augment
    options:
      show_root_heading: false
      heading_level: 3
