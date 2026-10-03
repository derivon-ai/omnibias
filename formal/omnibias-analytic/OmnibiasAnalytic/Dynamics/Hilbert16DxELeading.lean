/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Kill-line dx_e leading identities: algebraic prefactor (9/64)/(3/8) = 3/8,
slope half (1-2 theta)/2 = 3/8 at theta = 1/8, and threshold net
(3/16)(4-3) = 3/16. These theorems do not enclose the uniform-in-chi
bound, Stage C, first-hit, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16DxELeading

/-- Algebraic prefactor ``(9/64)*(8/3) = 3/8``. -/
theorem dx_prefactor_identity :
    (9 / 64 : ℝ) * (8 / 3) - 3 / 8 = 0 := by
  ring

/-- Integrand half-slope ``(1 - 2*(1/8))/2 = 3/8``. -/
theorem slope_half_identity :
    (1 - 2 * (1 / 8 : ℝ)) / 2 - 3 / 8 = 0 := by
  ring

/-- Threshold net floor ``(3/16)*(4 - 3) = 3/16``. -/
theorem net_floor_identity :
    (3 / 16 : ℝ) * (4 - 3) - 3 / 16 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16DxELeading
