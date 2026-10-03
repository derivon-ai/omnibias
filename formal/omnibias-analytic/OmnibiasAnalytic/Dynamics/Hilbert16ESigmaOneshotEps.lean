/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Shrinking-ε one-shot GRAZING E_σ identities: the finite pack
{1/16, 1/20, 1/25} sums to 61/400, the n = 20 compact is
T = 400 · (1/4) = 100, the n = 25 compact is T = 250, and
h(0) = 4ε³ at n = 25. These theorems do not enclose a
uniform-in-ε first-hit, Z_x C2, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16ESigmaOneshotEps

/-- Declared shrinking pack `1/16 + 1/20 + 1/25 = 61/400`. -/
theorem oeps_pack_sum_identity :
    ((1 / 16 : ℝ) + (1 / 20) + (1 / 25)) - (61 / 400) = 0 := by
  ring

/-- n = 20 compact `T = 400 · (1/4) = 100`. -/
theorem oeps_T20_identity :
    (400 : ℝ) * (1 / 4 : ℝ) - 100 = 0 := by
  ring

/-- n = 25 compact `T = 1000 · (1/4) = 250`. -/
theorem oeps_T25_identity :
    (1000 : ℝ) * (1 / 4 : ℝ) - 250 = 0 := by
  ring

/-- GRAZING start `h(0) = 4ε³` at `ε = 1/25`. -/
theorem oeps_h0_25_identity :
    (4 : ℝ) * (1 / 25 : ℝ) ^ 3 - (4 / 15625) = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16ESigmaOneshotEps
