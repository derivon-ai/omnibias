/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Kill-line comparison speed identities: F = f + 4ε³ g expands to the
cubic polynomial, F_V = ε φ, φ(-ε) = ε² + 4ε³, and the comparison
hitting time T = (ρ-ε)/(3ε³) at the sample ε = 1/16, ρ = 1/4, T = 256.
These theorems do not enclose a Lohner first-hit for every ε, GRAZING
E_σ, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16EOutSpeed

/-- Comparison `F = f + 4ε³ g` expands to the cubic polynomial. -/
theorem F_expand_identity (V ε : ℝ) :
    ((-2) * ε ^ 2 * V - ε * V ^ 2 + (ε / 3) * V ^ 3
        + (4 * ε ^ 3) * (-1 + ε * (V - 1)))
      - ((-2) * ε ^ 2 * V - ε * V ^ 2 + (ε / 3) * V ^ 3
        - 4 * ε ^ 3 - 4 * ε ^ 4 + 4 * ε ^ 4 * V) = 0 := by
  ring

/-- `F_V = ε φ` with `φ = V² - 2V - 2ε + 4ε³`. -/
theorem F_V_phi_identity (V ε : ℝ) :
    ((-2) * ε ^ 2 - 2 * ε * V + ε * V ^ 2 + 4 * ε ^ 4)
      - ε * (V ^ 2 - 2 * V - 2 * ε + 4 * ε ^ 3) = 0 := by
  ring

/-- Right-end value `φ(-ε, ε) = ε² + 4ε³`. -/
theorem phi_right_identity (ε : ℝ) :
    ((-ε) ^ 2 - 2 * (-ε) - 2 * ε + 4 * ε ^ 3)
      - (ε ^ 2 + 4 * ε ^ 3) = 0 := by
  ring

/-- Sample comparison time `T = (ρ-ε)/(3ε³)` at `ε = 1/16`, `ρ = 1/4`. -/
theorem T_cubic_identity :
    (3 : ℝ) * (1 / 16) ^ 3 * 256 - (1 / 4 - 1 / 16) = 0 := by
  ring

/-- `φ(-ε)` is strictly positive for `ε > 0`. -/
theorem phi_min_pos {ε : ℝ} (hε : 0 < ε) :
    0 < ε ^ 2 * (1 + 4 * ε) := by
  have hsq : 0 < ε ^ 2 := sq_pos_of_pos hε
  have hlin : 0 < 1 + 4 * ε := by nlinarith
  exact mul_pos hsq hlin

end OmnibiasAnalytic.Dynamics.Hilbert16EOutSpeed
