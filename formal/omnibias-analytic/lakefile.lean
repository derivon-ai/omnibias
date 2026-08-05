import Lake
open Lake DSL

/-
omnibias analytic track -- Mathlib-BACKED (deliberately NOT Mathlib-free).

Unlike `formal/omnibias-verified-kernel` (finite / decidable obligations, tiny
trust base, fast enough to kernel-check on every push) this project depends on
Mathlib so that real / rational obligations can be discharged with Mathlib's
tactics (`norm_num`, `positivity`, `nlinarith`, ...).

Trust tier. A green `lake build` here feeds the Python bridge's `mathlib_verified`
flag ONLY. It never feeds the minimal-kernel `theorem_prover_verified`. Every
module is `sorry`-free, and the scope is finite rational obligations -- the track
makes no analytic or asymptotic claim.

Cost. Mathlib is large; even with `lake exe cache get` this project builds in
minutes, so it lives OFF the fast CI path (see `.github/workflows/lean-analytic.yml`).
The Mathlib revision is pinned to the release tag matching this project's
`lean-toolchain` (`leanprover/lean4:v4.31.0`, the same toolchain as the kernel).
-/

package «omnibias-analytic» where
  leanOptions := #[⟨`autoImplicit, false⟩, ⟨`relaxedAutoImplicit, false⟩]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.31.0"

@[default_target]
lean_lib «OmnibiasAnalytic» where
