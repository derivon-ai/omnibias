# Claim ladder and forbidden-claims register

A result in this repository licenses one sentence. The four rungs are
strictly stronger and none implies the next. Theory spec 06-02 is the
source; this page is the public copy. Status is **shipped**.

This register records what the **runtime** does not license. An agent may
still judge a parent solved; they must not forge honesty flags or Lean
tiers. The table is the executable contract, not an order to stop.

| Rung | Earned by | Licenses | Does not license |
|---|---|---|---|
| **1 Empirical** | `gates_block(...)["all_passed"]` on a benchmark | "on this problem, at this size, this method achieved X" | other problems, sizes, or the true value |
| **2 Sound enclosure** | outward-rounded interval arithmetic in `omnibias.core.verified` | "the true value of this quantity lies in `[a, b]`" | anything outside the enclosed domain |
| **3 Kernel-verified** | a genuine `lake build` of the Mathlib-free kernel | "this finite rational obligation is machine-checked" | any infinite or analytic statement |
| **4 Mathlib-verified** | a genuine pass of `formal/omnibias-analytic` | "this finite rational obligation is machine-checked against Mathlib" | the same limits as rung 3 |

Rungs 3 and 4 are distinct. Both Lean projects are `sorry`-free and
scoped to **finite, rational** obligations. `theorem_prover_verified`
cannot be sealed by a producer:

```python
from omnibias.core.proof.certificate import make_certificate

raised = False
try:
    make_certificate(
        claim="forged formal claim",
        payload={"type": "interval"},
        honesty={"theorem_prover_verified": True},
    )
except ValueError as exc:
    raised = "theorem_prover_verified" in str(exc)
assert raised
```

## Certificate conditions

| Condition | What may be reported | What it does not license |
|---|---|---|
| `libm_fallback` transcendental stamp | a conditional diagnostic enclosure, with its fallback and ULP inflation stated | a rung-2 sound enclosure or a sealed designated rigorous payload |

`libm_fallback` is retained for non-certificate diagnostics when a rigorous
transcendental backend is unavailable. `make_certificate` refuses that stamp
for designated interval, Taylor-model, positive-definite, spectral-gap, and
PINN a-posteriori-error payloads.

## Forbidden-claims register

These sentences are what a sealed certificate and the honesty flags do
not license today.

| Never write | Write instead |
|---|---|
| "we prove global regularity for Navier-Stokes" | a finite residual enclosure on a discretized problem; global regularity stays external |
| "we prove the Yang-Mills mass gap" | a spectral gap for one fixed transfer matrix at one spacing; the continuum is not taken |
| "we prove / disprove the Riemann Hypothesis" | Dirichlet series on `Re(s) > 1`; continuation and zeros stay external |
| "P = NP" | a relaxation plus a decoder with an honest, possibly non-tight gap |
| "a certificate implies a continuum result" | a finite certificate constrains a finite object |
| "the Lean kernel verified our analysis" | the kernel verified a finite rational obligation extracted from the certificate |
| "closed form" for an autodiff or finite-difference path | `CLOSED_FORM` / `AUTODIFF` / `NUMERICAL` / `SPECTRAL` / `HIGH_ORDER` |

Padé / Borel (spec 03-10) locates singularities of a truncated series. That
is not analytic continuation of a Dirichlet series past `Re(s) = 1`.

See also [scope and guarantees](scope-and-guarantees.md),
the [frontier sub-obligation ledger](frontier-ledger.md), and
[theory/06-program/02-honesty-and-claim-boundaries.md](https://github.com/derivon-ai/omnibias/blob/main/theory/06-program/02-honesty-and-claim-boundaries.md).
