/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Cubic (V,h) field identities: q + f = 0 on h = 0, ḣ = -V h,
g = -1 + ν(V-1) expands, and V̇ = f + h g. These theorems do not
enclose a physical E_σ first-hit, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16VhOrbit

/-- Height flux `q(-w)` plus cubic `f(-w)` cancels. -/
theorem q_plus_f_identity (L λ1 ε w : ℝ) :
    (L * ε ^ 3 + λ1 * ε ^ 2 * w + ε * w ^ 2 * (1 + w / 3))
      + (-L * ε ^ 3 + λ1 * ε ^ 2 * (-w) - ε * (-w) ^ 2 + (ε / 3) * (-w) ^ 3)
      = 0 := by
  ring

/-- Exact height equation on the cubic field. -/
theorem hdot_identity (V h : ℝ) :
    (-V * h) + V * h = 0 := by
  ring

/-- First-order transverse jet expands to `-1 - ν + ν V`. -/
theorem g_jet_identity (ν V : ℝ) :
    (-1 + ν * (V - 1)) - (-1 - ν + ν * V) = 0 := by
  ring

/-- Cubic splitting `V̇ = f + h g` is additive. -/
theorem Vdot_split_identity (L λ1 ε ν V h : ℝ) :
    ((-L * ε ^ 3 + λ1 * ε ^ 2 * V - ε * V ^ 2 + (ε / 3) * V ^ 3)
        + h * (-1 + ν * (V - 1)))
      - (-L * ε ^ 3 + λ1 * ε ^ 2 * V - ε * V ^ 2 + (ε / 3) * V ^ 3)
      - h * (-1 + ν * (V - 1)) = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16VhOrbit
