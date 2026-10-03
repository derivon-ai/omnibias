/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Matching-chart E_out identities: E_σ at ν = 0 reduces to V-1+σ ρ h,
E_out at ν = 0 reduces to V+ρ, the embedding V = -ε (ρ/ε) is -ρ, and
E_out groups as V + ρ (1 + ν h + C ν² h²). These theorems do not
enclose GRAZING E_σ first-hit, uniform ε → 0, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16EOutSection

/-- GRAZING `E_σ` at `ν = 0` is `V - 1 + σ ρ h`. -/
theorem e_sigma_nu0_identity (V h σ ρ C : ℝ) :
    (V - 1 + σ * ρ * h + (0 : ℝ) * ρ ^ 2 * h ^ 2
        + C * (0 : ℝ) ^ 2 * σ * ρ * h ^ 2)
      - (V - 1 + σ * ρ * h) = 0 := by
  ring

/-- Matching-chart `E_out` at `ν = 0` is `V + ρ`. -/
theorem e_out_lead_identity (V ρ C : ℝ) :
    (V + ρ + (0 : ℝ) * ρ * (0 : ℝ) + C * (0 : ℝ) ^ 2 * ρ * (0 : ℝ) ^ 2)
      - (V + ρ) = 0 := by
  ring

/-- Physical section `x = ρ/ε` maps to `V = -ρ` under `V = -ε x`. -/
theorem section_embed_identity (ε ρ : ℝ) (hε : ε ≠ 0) :
    -ε * (ρ / ε) + ρ = 0 := by
  field_simp [hε]

/-- Height-corrected `E_out` groups as `V + ρ (1 + ν h + C ν² h²)`. -/
theorem e_out_group_identity (V h ρ ν C : ℝ) :
    (V + ρ + ν * ρ * h + C * ν ^ 2 * ρ * h ^ 2)
      - (V + ρ * (1 + ν * h + C * ν ^ 2 * h ^ 2)) = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16EOutSection
