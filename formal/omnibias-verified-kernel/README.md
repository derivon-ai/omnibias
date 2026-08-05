# omnibias verified kernel (Lean 4, Mathlib-free)

A small, **Mathlib-free** Lean 4 library that *kernel-checks* the finite, rational
proof obligations emitted by the omnibias certificate format
(`omnibias.core.proof.certificate`).

Because it uses the Lean 4 core only, `lake build` elaborates and kernel-checks
every proof **without** a Mathlib cache -- cheap enough to run in CI on every
push. Every theorem here is `sorry`-free.

## What it proves

| Module | Content |
| --- | --- |
| `Omnibias/Interval.lean` | `ZInterval` integer-interval arithmetic (add / neg / sub / **mul**) with **proven** membership-soundness lemmas (`mem_add`, `mem_sub`, `mem_neg`, **`mem_mul`**) and positivity certificates, all discharged by `omega` (multiplication factors the one nonlinear step through two monotone `min`/`max` bounds). |
| `Omnibias/Certificate.lean` | The certificate obligations: spectral-gap positivity (`1 - r > 0` from a subdominant ratio `r < 1`) and the enclosed-quantity sign obligations (CLM blow-up / CCF closure), built on the interval kernel. |
| `Omnibias/LDLT.lean` | **Kernel-verified LDLᵀ positive-definiteness.** `allPivotsPos` decides that every pivot interval of a symmetric matrix box's interval LDLᵀ factorisation is strictly positive; `matrix_positive_definite_certified` proves the whole `n`-pivot inertia vector is positive, lifting the strict-local-minimum PD claim above its former single-scalar `eig_min > 0` shadow. |
| `Omnibias/Golden.lean` | Checked-in **golden certificates** with the exact integer constants of representative omnibias certificates (including a golden LDLᵀ inertia vector); CI re-verifies them every run. |
| `Omnibias/Generated.lean` | Placeholder overwritten by the Python bridge (`omnibias.core.proof.lean_check`) with the certificate under test. |

## Scope and honesty

This kernel discharges the obligations that are genuinely **finite and decidable**
once a certificate has been computed: rational sign / gap facts, the soundness of the
integer-interval enclosure algebra (now including **multiplication**), and the LDLᵀ
inertia (positive-definiteness) vector. It does **not** attempt infinite analytic
obligations -- limits, continuum statements, asymptotics -- which are out of scope
and are not expressed here at all.

The kernel-verified LDLᵀ obligation certifies the *finite* fact that every pivot
interval is strictly positive (zero negative inertia). That these intervals really are
the pivots of the symmetric matrix box's factorisation `S = L D Lᵀ` -- and hence, by
Sylvester's law of inertia (a congruence), that every point matrix in `S` is positive
definite -- is computed by the unverified Python interval LDLᵀ and is a **trusted input**
here, the same trust base as the earlier `eig_min` scalar, only finer-grained. The
congruence-to-PD step itself is a quadratic-form statement outside the Mathlib-free
kernel's core-only vocabulary, so it is deliberately not re-proved here; the newly proven
`ZInterval.mul` makes a division-free `L D Lᵀ` reconstruction *check* expressible for
future tightening. Taylor-model remainders remain future work.

## Build

```bash
cd formal/omnibias-verified-kernel
lake build
```

The Python bridge drives this automatically:

```python
from omnibias.core.proof.lean_check import lean_check_available, check_certificate
```
