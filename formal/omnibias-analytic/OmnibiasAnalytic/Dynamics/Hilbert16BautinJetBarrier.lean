/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
A finite coefficient prefix never determines the next coefficient without an
independent recurrence. This formalizes only the logical finite-jet barrier;
the exact-Q Gröbner witnesses for the named quadratic family live in Python.
It does not assert realizability of the adversarial continuation by a vector
field, all-orders Bautin stabilization, G2, or Hilbert XVI.
-/

import Mathlib.Data.Rat.Defs

namespace OmnibiasAnalytic.Dynamics.Hilbert16BautinJetBarrier

/-- Two rational sequences can agree through any finite order and differ at
the next coefficient. -/
theorem finitePrefixHasDistinctContinuations (N : ℕ) :
    ∃ a b : ℕ → ℚ,
      (∀ k ≤ N, a k = b k) ∧ a (N + 1) ≠ b (N + 1) := by
  refine ⟨fun _ => 0, fun k => if k = N + 1 then 1 else 0, ?_, ?_⟩
  · intro k hk
    have hne : k ≠ N + 1 := Nat.ne_of_lt (Nat.lt_succ_of_le hk)
    simp [hne]
  · simp

end OmnibiasAnalytic.Dynamics.Hilbert16BautinJetBarrier
