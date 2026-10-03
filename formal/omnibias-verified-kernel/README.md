# Finite rational verification kernel

**An optional, independently checked last step for supported numerical certificates.**

A numerical routine proposes an enclosure or certificate. This Mathlib-free
Lean 4 library checks the finite rational obligations emitted by
`omnibias.core.proof.lean_check`. A successful check earns
`theorem_prover_verified`; a JSON flag, digest or solver success cannot earn it.

## Why keep it?

Ordinary PINN training needs neither Lean nor this directory. Keep this small
verification substrate when a downstream application needs more than a numerical
assertion: a machine-checked justification for the supported finite claim. It
belongs with the certificate infrastructure, rather than a particular scientific
application. It is not a new Python distribution or a mandatory runtime dependency.

```text
numerical proposal → finite rational obligations → Lean kernel check
                                                        ↓
                                            verified result or explicit failure
```

## What it checks

The library proves integer-interval arithmetic, enclosed-value signs,
spectral-ratio inequalities, positive pivot vectors, stencil identities, replay
traces and finite subdivision coverage. Every theorem is checked without `sorry`.

The kernel checks the supplied finite data. Analytic hypotheses, numerical
factorizations and Taylor remainder derivations remain trusted inputs unless
represented explicitly by a supported replay trace. A successful finite check
is not a proof of an entire PDE model, a trained network's global accuracy or a
physical system's safety.

## Run from a source checkout

Install the Lean toolchain specified by `lean-toolchain`, then run:

```bash
cd formal/omnibias-verified-kernel
lake build
```

The Python bridge searches for this directory in the checkout's parent hierarchy.
A regular `omnibias-core` wheel does not bundle a Lean checkout or toolchain.
The bridge returns an unavailable result when either is missing; it must never
silently substitute an unchecked success. Only a successful supported kernel
check sets `theorem_prover_verified`.

## Maintain the trust boundary

Keep rational arithmetic and replay schemas explicit. Test valid obligations,
invalid obligations and unavailable-toolchain behavior. Changes to emitted Lean
and its schema must be coordinated with the Python bridge. Do not weaken a failed
check into a warning or add unproved axioms to make a certificate pass.

See the [Python bridge](../../packages/omnibias-core/src/omnibias/core/proof/lean_check.py)
and [guarantees](../../docs/guarantees.md) for the caller-facing contract.
