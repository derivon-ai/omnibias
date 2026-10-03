/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Uniform comparison GRAZING E_σ identities: cancelled height
integral at ε = 0 is V²/2, E_σ at ν = 0 is V-1+σ ρ h, and
1 - ρ V* = 7/10 at V* = 6/5. These theorems do not enclose a
Lohner first-hit for every ε, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16ESigmaUnif

/-- Cancelled height integral at `ε = 0` is `V²/2`. -/
theorem I_limit_identity (V : ℝ) :
    (V ^ 2 / 1 - (1 + 0) * V ^ 2 / (2 * 1 ^ 2) + (1 + 0) * 0 * V ^ 3 / (3 * 1 ^ 3))
      - V ^ 2 / 2 = 0 := by
  ring

/-- Cleared cancelled height integrand versus the polynomial numerator. -/
theorem I_cancel_clear_identity (d V ε : ℝ) :
    (6 : ℝ) * d ^ 3 *
        (V ^ 2 / d - (1 + ε) * V ^ 2 / (2 * d ^ 2)
          + (1 + ε) * ε * V ^ 3 / (3 * d ^ 3))
      - (6 * d ^ 2 * V ^ 2 - 3 * (1 + ε) * d * V ^ 2
        + 2 * (1 + ε) * ε * V ^ 3) = 0 := by
  ring

/-- `E_σ` at `ν = 0` is `V-1+σ ρ h`. -/
theorem E_eps0_identity (V h σ ρ C : ℝ) :
    (V - 1 + σ * ρ * h + (0 : ℝ) * ρ * ρ * h * h
        + C * 0 * 0 * σ * ρ * h * h)
      - (V - 1 + σ * ρ * h) = 0 := by
  ring

/-- Sample `1 - ρ V* = 7/10` at `V* = 6/5`, `ρ = 1/4`. -/
theorem de_eps0_identity :
    (1 : ℝ) - (1 / 4) * (6 / 5) - 7 / 10 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16ESigmaUnif
