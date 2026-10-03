/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Matching-chart shrinking-eps identities: V(0) = -ε, h(0) = 4ε³,
the declared majorant T = n²/8, and the kill-line matching slope
V̇ + 3ε³ + (13/3)ε⁴ + 4ε⁵ = 0. These theorems do not enclose a
uniform-in-ε first-hit, GRAZING E_σ, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16EOutEps

/-- Matching initial `V(0) = -ε`. -/
theorem v0_matching_identity (ε : ℝ) :
    (-ε) + ε = 0 := by
  ring

/-- Matching initial `h(0) = 4ε³`. -/
theorem h0_matching_identity (ε : ℝ) :
    (4 * ε * ε * ε) - 4 * ε ^ 3 = 0 := by
  ring

/-- Declared majorant `T = n²/8` at the sample `n = 16`, `T = 32`. -/
theorem horizon_n2_identity :
    (16 : ℝ) ^ 2 - 8 * 32 = 0 := by
  ring

/-- Kill-line matching slope at `L = 0`, `λ₁ = -2`, `V = -ε`, `h = 4ε³`. -/
theorem vdot_kill_init_identity (ε : ℝ) :
    ((-2) * ε ^ 2 * (-ε) - ε * (-ε) ^ 2 + (ε / 3) * (-ε) ^ 3
        + (4 * ε ^ 3) * (-1 + ε * ((-ε) - 1)))
      + 3 * ε ^ 3 + (13 / 3) * ε ^ 4 + 4 * ε ^ 5 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16EOutEps
