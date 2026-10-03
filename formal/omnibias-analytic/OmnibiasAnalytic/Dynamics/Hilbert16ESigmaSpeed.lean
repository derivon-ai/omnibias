/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Incoming GRAZING comparison speed identities: F(0) = -4ε³(1+ε),
φ(0) = -2ε + 4ε³ = -2ε(1-2ε²), and the comparison time
T = 1/(4ε³(1+ε)) at the sample ε = 1/16, T = 16384/17.
These theorems do not enclose a certified E_σ first-hit, G1, G4,
or Hilbert XVI.
-/

import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16ESigmaSpeed

/-- Comparison `F(0) = -4ε³(1+ε)` from `f(0)+4ε³ g(0)`. -/
theorem F_zero_identity (ε : ℝ) :
    ((4 * ε ^ 3) * (-1 + ε * (0 - 1)))
      + 4 * ε ^ 3 * (1 + ε) = 0 := by
  ring

/-- `φ(0) = -2ε + 4ε³`. -/
theorem phi_zero_identity (ε : ℝ) :
    ((0 : ℝ) ^ 2 - 2 * 0 - 2 * ε + 4 * ε ^ 3)
      - (-2 * ε + 4 * ε ^ 3) = 0 := by
  ring

/-- Factor `φ(0) = -2ε(1-2ε²)`. -/
theorem phi_zero_factor_identity (ε : ℝ) :
    ((0 : ℝ) ^ 2 - 2 * 0 - 2 * ε + 4 * ε ^ 3)
      + 2 * ε * (1 - 2 * ε ^ 2) = 0 := by
  ring

/-- Sample incoming comparison time at `ε = 1/16`. -/
theorem T_in_identity :
    (4 : ℝ) * (1 / 16) ^ 3 * (1 + 1 / 16) * (16384 / 17) - 1 = 0 := by
  ring

/-- `φ(0) < 0` for `0 < ε ≤ 1/16`. -/
theorem phi_zero_neg {ε : ℝ} (hε : 0 < ε) (h16 : ε ≤ 1 / 16) :
    -2 * ε + 4 * ε ^ 3 < 0 := by
  have hpos : 0 ≤ ε := le_of_lt hε
  have hmul : ε * ε ≤ ε * (1 / 16) :=
    mul_le_mul_of_nonneg_left h16 hpos
  have h256 : ε * (1 / 16) ≤ (1 / 16) * (1 / 16) :=
    mul_le_mul_of_nonneg_right h16 (by nlinarith)
  have hsq : ε ^ 2 ≤ (1 : ℝ) / 256 := by
    have : ((1 : ℝ) / 16) * (1 / 16) = 1 / 256 := by ring
    nlinarith
  nlinarith

end OmnibiasAnalytic.Dynamics.Hilbert16ESigmaSpeed
