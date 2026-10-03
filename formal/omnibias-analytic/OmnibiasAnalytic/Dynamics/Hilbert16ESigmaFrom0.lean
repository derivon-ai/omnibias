/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Comparison GRAZING E_σ identities from V = 0: kill-line cubic f
factors, reverse Vdot split, dh/dV ratio split, cubic Taylor
numerator of the log gap, and dE/dV ≥ 55/79 at the sample
(V*, ρ, den). These theorems do not enclose a Lohner first-hit
from V = 0, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16ESigmaFrom0

/-- Kill-line `f` at `L = 0` factors as `(ε/3) V (V²-3V-6ε)`. -/
theorem f_cubic_factor_identity (V ε : ℝ) :
    ((-2) * ε ^ 2 * V - ε * V ^ 2 + (ε / 3) * V ^ 3)
      - ((ε / 3) * V * (V ^ 2 - 3 * V - 6 * ε)) = 0 := by
  ring

/-- Reverse `V̇ = -f + h(1+ν-ν V)` with `g = -1+ν(V-1)`. -/
theorem vdot_rev_split_identity (V h ν : ℝ) :
    (-(((-2) * ν ^ 2 * V - ν * V ^ 2 + (ν / 3) * V ^ 3)
        + h * (-1 + ν * (V - 1))))
      - (-((-2) * ν ^ 2 * V - ν * V ^ 2 + (ν / 3) * V ^ 3)
        + h * (1 + ν - ν * V)) = 0 := by
  ring

/-- `V ν = (1+ν) - (1+ν-ν V)`, the cleared `dh/dV` ratio split. -/
theorem ratio_split_identity (V ν : ℝ) :
    V * ν - ((1 + ν) - (1 + ν - ν * V)) = 0 := by
  ring

/-- Derivative numerator of `x - x²/2 + x³/3 - log(1+x)`. -/
theorem log_taylor_num_identity (x : ℝ) :
    (1 - x + x ^ 2) * (1 + x) - 1 - x ^ 3 = 0 := by
  ring

/-- Sample `dE/dV` lower bound `1 - ρ V*/den = 55/79`. -/
theorem de_sigma_lo_identity :
    (55 / 79 : ℝ) * (79 / 80) - (79 / 80 - (1 / 4) * (6 / 5)) = 0 := by
  ring

/-- The sample lower bound is strictly positive. -/
theorem de_sigma_lo_pos :
    (0 : ℝ) < (55 / 79 : ℝ) := by
  nlinarith

end OmnibiasAnalytic.Dynamics.Hilbert16ESigmaFrom0
