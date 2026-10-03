/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Kill-line Stage-C leading T_h identities: midpoint product
(1-r1)(1-r2) + sep^2/4 = 0 on lambda1 = -2, worst T_h = 1 - 1/4 = 3/4,
and room 3/4 - 1/2 = 1/4. These theorems do not enclose an outgoing
orbit, first-hit, C2, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16StageCTh

/-- Midpoint product ``(1-(1-sep/2))*(1-(1+sep/2)) + (sep/2)^2 = 0``. -/
theorem mid_product_identity (sep : ℝ) :
    (1 - (1 - sep / 2)) * (1 - (1 + sep / 2)) + (sep / 2) * (sep / 2) = 0 := by
  ring

/-- Worst leading ``T_h = 1 - 1/4 = 3/4``. -/
theorem th_three_four_identity :
    (1 : ℝ) - 1 / 4 - 3 / 4 = 0 := by
  ring

/-- Declared floor room ``3/4 - 1/2 = 1/4``. -/
theorem th_half_room_identity :
    (3 / 4 : ℝ) - 1 / 2 - 1 / 4 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16StageCTh
