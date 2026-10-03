/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Fold-wall disc identities: L = r², λ₁ = -2 r, λ₁² = 4 L, and
B_- = (x - r)². These theorems do not enclose Z, physical C2,
first-hit, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16FoldZeta

/-- Fold coefficient `λ₁ = -2 r`. -/
theorem fold_lambda_identity (r : ℝ) :
    (-2 * r) + 2 * r = 0 := by
  ring

/-- Fold coefficient `L = r²`. -/
theorem fold_L_identity (r : ℝ) :
    r ^ 2 - r ^ 2 = 0 := by
  ring

/-- Discriminant wall `λ₁² - 4 L = 0` on the fold. -/
theorem fold_disc_identity (r : ℝ) :
    (-2 * r) ^ 2 - 4 * (r ^ 2) = 0 := by
  ring

/-- Slow-line double root `B_- = (x - r)²`. -/
theorem fold_bminus_identity (x r : ℝ) :
    (x - r) ^ 2 - (r ^ 2 + (-2 * r) * x + x ^ 2) = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16FoldZeta
