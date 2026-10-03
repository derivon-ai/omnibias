# Finite rational verification kernel

A Mathlib-free Lean 4 library for the finite rational obligations emitted by
`omnibias.core.proof.lean_check`.

It proves integer-interval arithmetic, enclosed-value signs, spectral-ratio
inequalities, positive pivot vectors, stencil identities, replay traces and
finite subdivision coverage. Every theorem is checked without `sorry`.

The kernel checks the supplied finite data. Analytic hypotheses, numerical
factorizations and Taylor remainder derivations remain trusted inputs unless
they are represented explicitly by a supported replay trace.

```bash
cd formal/omnibias-verified-kernel
lake build
```

The Python bridge returns an unavailable result when the toolchain is absent;
only a successful kernel check sets `theorem_prover_verified`.
