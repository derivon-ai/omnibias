# Rational stencil obligations (01-11)

Collapse weights are **rationals**, so the consistency conditions of a
stencil are a finite conjunction of identities the Mathlib-free Lean
kernel can decide. Status is **gated**, not shipped. G1–G5 are
CI-gated (G1 kernel pass and G2 lake-fail run in the Lean job).

Lean certifies the **algebra**. It does not state `delta -> 0`, Taylor's
theorem, or any claim quantified over a function class. Say "the
collapse weights are verified", never "the collapse is verified".
`theorem_prover_verified` is earned only by a genuine `lake build`.
`mathlib_verified` stays false: this spec does not touch
`omnibias-analytic`.

No new package. The payload lives in
`omnibias.core.proof.obligations.rational_stencil` and the evaluator
lemmas live in `formal/omnibias-verified-kernel/Omnibias/RationalStencil.lean`.

## API

::: omnibias.core.proof.obligations.rational_stencil
    options:
      show_root_heading: false
      heading_level: 3
