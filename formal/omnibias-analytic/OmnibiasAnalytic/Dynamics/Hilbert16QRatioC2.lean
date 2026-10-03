/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Uniform C = 2 leading |q| ratio on the shrinking-root sequence λ₁ = -2.
These theorems do not bound k = 1 + O(ε), a ζ remainder, height-section
first-hit, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16QRatioC2

/-- Inner gap `2 + x² - (x-r₁)(r₂-x)` on the two-root embedding. -/
theorem ratio_gap_identity (x r1 r2 : ℝ) :
    2 + x ^ 2 - (x - r1) * (r2 - x)
      - (2 + 2 * x ^ 2 + (-(r1 + r2)) * x + r1 * r2) = 0 := by
  ring

/-- Discriminant `(r₂-r₁)² - 8(L+2)` on the two-root embedding. -/
theorem ratio_disc_identity (r1 r2 : ℝ) :
    ((r2 - r1) ^ 2 - 8 * (r1 * r2 + 2))
      - ((-(r1 + r2)) ^ 2 - 12 * (r1 * r2) - 16) = 0 := by
  ring

/-- Complete square at `λ₁ = -2`. -/
theorem kill_square_identity (x L : ℝ) :
    (2 + 2 * x ^ 2 - 2 * x + L)
      - (2 * (x - 1 / 2) ^ 2 + 3 / 2 + L) = 0 := by
  ring

/-- Outer gap on `r₁ + r₂ = 2`. -/
theorem ratio_outer_identity (x r1 : ℝ) :
    2 + x ^ 2 - (x - r1) * (x - (2 - r1))
      - (2 + 2 * x - r1 * (2 - r1)) = 0 := by
  ring

/-- Discriminant `λ₁² - 12 L - 16` is negative on `λ₁ = -2`, `L ≥ 0`. -/
theorem kill_disc_neg {L : ℝ} (hL : 0 ≤ L) :
    (-2 : ℝ) ^ 2 - 12 * L - 16 < 0 := by
  nlinarith

/-- Inner complete square is strictly positive for `L ≥ 0`. -/
theorem kill_square_pos (x L : ℝ) (hL : 0 ≤ L) :
    0 < 2 * (x - 1 / 2) ^ 2 + 3 / 2 + L := by
  nlinarith

end OmnibiasAnalytic.Dynamics.Hilbert16QRatioC2
