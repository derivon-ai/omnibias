/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Incoming GRAZING-chart identities: the start V(0) = 0, reverse
ḣ = V h, the matching reverse slope V̇_rev = 4ε³(1+ε) at V = 0,
and the declared wall V = 1/4. These theorems do not enclose a
certified E_σ first-hit, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16ESigmaIn

/-- GRAZING start `V(0) = 0` independently of `ε`. -/
theorem v0_grazing_identity (ε : ℝ) :
    (0 : ℝ) * ε = 0 := by
  ring

/-- Reverse of forward `ḣ = -V h` is `V h`. -/
theorem hdot_rev_identity (V h : ℝ) :
    (-((-V) * h)) - V * h = 0 := by
  ring

/-- Kill-line reverse matching slope at `V = 0`, `h = 4ε³`. -/
theorem vdot_rev_init_identity (ε : ℝ) :
    (-((4 * ε ^ 3) * (-1 + ε * (0 - 1))))
      - 4 * ε ^ 3 * (1 + ε) = 0 := by
  ring

/-- Declared incoming wall `V = 1/4`. -/
theorem vin_wall_identity :
    (1 / 4 : ℝ) - 1 / 4 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16ESigmaIn
