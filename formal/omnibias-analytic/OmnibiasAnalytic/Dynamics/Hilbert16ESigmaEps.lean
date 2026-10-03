/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Shrinking-ε aligned GRAZING E_σ identities: the finite pack
{1/16, 1/20, 1/25} sums to 61/400, the GRAZING starts
h(0) = 4ε³ at n = 16 and n = 20, and the aligned compact is
T = 80 · (1/8) = 10. These theorems do not enclose a uniform-in-ε
first-hit, a Lohner run from V = 0, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16ESigmaEps

/-- Declared shrinking pack `1/16 + 1/20 + 1/25 = 61/400`. -/
theorem sigma_pack_sum_identity :
    ((1 / 16 : ℝ) + (1 / 20) + (1 / 25)) - (61 / 400) = 0 := by
  ring

/-- GRAZING start `h(0) = 4ε³` at `ε = 1/16`. -/
theorem sigma_h0_16_identity :
    (4 : ℝ) * (1 / 16 : ℝ) ^ 3 - (1 / 1024) = 0 := by
  ring

/-- GRAZING start `h(0) = 4ε³` at `ε = 1/20`. -/
theorem sigma_h0_20_identity :
    (4 : ℝ) * (1 / 20 : ℝ) ^ 3 - (1 / 2000) = 0 := by
  ring

/-- Aligned compact horizon `T = 80 · (1/8) = 10`. -/
theorem sigma_horizon_identity :
    (80 : ℝ) * (1 / 8 : ℝ) - 10 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16ESigmaEps
