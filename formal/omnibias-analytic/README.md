# omnibias analytic track (Lean 4, Mathlib-backed)

A **Mathlib-backed** Lean 4 library -- the deliberate counterpart to the
Mathlib-free [`omnibias-verified-kernel`](../omnibias-verified-kernel). It exists
so that certificate obligations can be discharged over `ℚ` / `ℝ` with Mathlib's
tactics.

## Modules

The `Check/**` modules are the sound, `sorry`-free certificate checkers:

| Module | Content |
| --- | --- |
| `OmnibiasAnalytic/Check/EnclosedSign.lean` | Rational enclosed-sign lemmas (`enclosed_pos`, `enclosed_neg`) over `ℚ` -- the Mathlib analogue of the integer `ZInterval` sign obligations, with **no integer-scaling hack**. |
| `OmnibiasAnalytic/Check/Positivity.lean` | Sum-of-squares / positivity capability lemmas (`sq_add_sq_nonneg`, `quad_pos` via `positivity` / `nlinarith`) the integer kernel cannot express. |
| `OmnibiasAnalytic/Check/Kantorovich.lean` | Newton-Kantorovich / Krawczyk finite-obligation lemmas (radii polynomial self-map, contraction, strict box containment) over `ℚ`. |
| `OmnibiasAnalytic/Generated.lean` | Placeholder overwritten by the Python bridge (`omnibias.formal.mathlib_check`) with the certificate under test. |

## Trust tier and honesty

A green `lake build` here feeds only the bridge's **`mathlib_verified`** flag.
It is a **distinct, larger** trust base than the minimal Mathlib-free kernel:

- It never feeds the minimal-kernel `theorem_prover_verified`.
- A green build means the emitted obligation *checks against Mathlib*, not that
  any statement beyond that finite obligation is proved.

**Scope.** This track discharges *finite, rational* obligations only. Every
module is `sorry`-free and CI audits that (`scripts`-free `grep` in
[`lean-analytic.yml`](../../.github/workflows/lean-analytic.yml)). Asymptotic,
continuum, and other infinite analytic statements are out of scope: they are not
expressed here, so they can never be silently "discharged" here either.

## Build

This project depends on Mathlib, pinned to the release tag matching its
`lean-toolchain` (`leanprover/lean4:v4.31.0`, the same toolchain as the kernel):

```bash
cd formal/omnibias-analytic
lake update            # resolves mathlib @ v4.31.0 and writes lake-manifest.json
lake exe cache get     # download Mathlib's prebuilt oleans (minutes, not hours)
lake build
```

Committing `lake-manifest.json` after a first successful `lake update` pins the
exact Mathlib commit for reproducibility. Because the dependency is pinned to a
fixed release tag, `lake update` is deterministic and CI regenerates the manifest
on demand (see [`.github/workflows/lean-analytic.yml`](../../.github/workflows/lean-analytic.yml)).

The Python bridge drives this automatically:

```python
from omnibias.formal import mathlib_check_available, check_certificate
```
