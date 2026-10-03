/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
C ≠ 0 height mixing: ℓ and V pick up C ν² h, so V_v + ℓ = 0,
V_h + C ν² v = 0, and the GRAZING first-order g jet is independent of
h. These theorems do not bound T-h along the orbit, first-hit, G1, G4,
or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16HeightMix

/-- `ℓ` versus the C = 0 factor plus `C ν² h`. -/
theorem ell_mix_identity (ν v C h : ℝ) :
    (1 + 2 * ν * v + C * ν ^ 2 * h) - (1 + 2 * ν * v) - C * ν ^ 2 * h = 0 := by
  ring

/-- `V` versus the slow-line factor minus `C ν² v h`. -/
theorem V_mix_identity (ν v C h : ℝ) :
    (1 - v - ν * v ^ 2 - C * ν ^ 2 * v * h)
      - (1 - v - ν * v ^ 2) + C * ν ^ 2 * v * h = 0 := by
  ring

/-- `V_v + ℓ = 0`. -/
theorem V_v_ell_identity (ν v C h : ℝ) :
    (-1 - 2 * ν * v - C * ν ^ 2 * h)
      + (1 + 2 * ν * v + C * ν ^ 2 * h) = 0 := by
  ring

/-- `V_h + C ν² v = 0`. -/
theorem V_h_c_identity (ν v C : ℝ) :
    (-C * ν ^ 2 * v) + C * ν ^ 2 * v = 0 := by
  ring

/-- GRAZING (2.4) jet: `(V̇ - f)/h = -1 + ν(V-1)`, cleared of `h`. -/
theorem g_lead_identity (ν V h : ℝ) :
    ((-h + ν * (V ^ 3 - 3 * V ^ 2 + (V - 1) * h)) - ν * (V ^ 3 - 3 * V ^ 2))
      - h * (-1 + ν * (V - 1)) = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16HeightMix
