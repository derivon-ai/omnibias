/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Kill-sequence identities on λ₁ = -2: L = r₁(2-r₁), r₁+r₂ = 2,
discriminant 4(1-L), and the two-root cubic. These theorems do not
enclose Z, bound T-h along the orbit, first-hit, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16KillZeta

/-- Kill coefficient `λ₁ = -2`. -/
theorem kill_lambda_identity :
    (-2 : ℝ) + 2 = 0 := by
  ring

/-- Product `L = r₁(2-r₁)` versus `2 r₁ - r₁²`. -/
theorem kill_product_identity (r1 : ℝ) :
    r1 * (2 - r1) - (2 * r1 - r1 ^ 2) = 0 := by
  ring

/-- Root sum `r₁ + r₂ = 2`. -/
theorem kill_sum_identity (r1 : ℝ) :
    r1 + (2 - r1) - 2 = 0 := by
  ring

/-- Discriminant `λ₁² - 4 L = 4(1-L)` at `λ₁ = -2`. -/
theorem kill_disc_gap_identity (L : ℝ) :
    ((-2 : ℝ) ^ 2 - 4 * L) - 4 * (1 - L) = 0 := by
  ring

/-- Two-root cubic on `r₁ + r₂ = 2`. -/
theorem kill_tworoot_identity (x r1 : ℝ) :
    (x - r1) * (x - (2 - r1))
      - (x ^ 2 - 2 * x + r1 * (2 - r1)) = 0 := by
  ring

/-- Discriminant is positive for `L < 1`. -/
theorem kill_disc_pos {L : ℝ} (hL : L < 1) :
    0 < 4 * (1 - L) := by
  nlinarith

end OmnibiasAnalytic.Dynamics.Hilbert16KillZeta
