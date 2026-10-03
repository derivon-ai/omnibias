/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Declared incoming-point E_σ identities: restart V = 3/4, h = 1/4,
E_σ at ν = 0 is V-1+σ ρ h, and the grouped HEIGHT-COMPARISON
polynomial. These theorems do not enclose the GRAZING band from
V = 0, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16ESigmaHit

/-- Declared incoming restart `V = 3/4`. -/
theorem restart_v_identity :
    (3 / 4 : ℝ) - 3 / 4 = 0 := by
  ring

/-- Declared incoming restart `h = 1/4`. -/
theorem restart_h_identity :
    (1 / 4 : ℝ) - 1 / 4 = 0 := by
  ring

/-- `E_σ` at `ν = 0` is `V-1+σ ρ h`. -/
theorem e_sigma_in_lead_identity (V h σ ρ C : ℝ) :
    (V - 1 + σ * ρ * h + (0 : ℝ) * ρ * ρ * h * h
        + C * 0 * 0 * σ * ρ * h * h)
      - (V - 1 + σ * ρ * h) = 0 := by
  ring

/-- Grouped HEIGHT-COMPARISON polynomial. -/
theorem e_sigma_in_group_identity (V h σ ρ ν C : ℝ) :
    (V - 1 + σ * ρ * h + ν * ρ * ρ * h * h + C * ν * ν * σ * ρ * h * h)
      - (V - 1 + σ * ρ * (h + C * ν * ν * h * h) + ν * ρ * ρ * h * h) = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16ESigmaHit
