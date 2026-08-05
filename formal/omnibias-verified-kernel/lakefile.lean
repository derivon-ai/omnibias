import Lake
open Lake DSL

/-
omnibias verified kernel.

A deliberately **Mathlib-free** Lean 4 core library: a sound rational/integer
interval arithmetic kernel with *proven* soundness lemmas (no `sorry`), plus a
certificate checker that discharges the **finite, rational** proof obligations
emitted by the omnibias certificate format v1 -- the Birkhoff-Hopf / Perron
spectral-gap positivity and the CLM/CCF rational sign obligations.

Being Mathlib-free means `lake build` elaborates and *kernel-checks* every proof
without downloading a Mathlib cache, so it is cheap enough for CI.  Infinite
analytic obligations -- limits, continuum statements, asymptotics -- are out of
scope and are not expressed here at all; this library only certifies the finite
obligations that are genuinely decidable.
-/

package «omnibias-verified-kernel» where
  leanOptions := #[⟨`autoImplicit, false⟩]

@[default_target]
lean_lib «Omnibias» where
