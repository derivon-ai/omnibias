/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Kill-line Stage-A wall identities: B_-(a) = theta(1+theta) sep^2 and
B_-'(bnd) = -sep(1-2 theta) at theta = 1/8, plus the worst-case left
wall 1 - (1/2 + 1/8) = 3/8. These theorems do not enclose
dx_e/dkappa, a chi threshold, Stage C, first-hit, G1, G4, or
Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16StageA

/-- Left-wall quadratic: `(a - r1)(a - r2) - theta(1+theta) sep^2`. -/
theorem B_wall_identity (sep : ℝ) :
    ((1 - sep / 2) - (1 / 8) * sep - (1 - sep / 2)) *
        ((1 - sep / 2) - (1 / 8) * sep - (1 + sep / 2)) -
      (1 / 8) * (1 + 1 / 8) * sep * sep = 0 := by
  ring

/-- Right-wall slope: `2(bnd - 1) + sep(1 - 2 theta)`. -/
theorem B_slope_identity (sep : ℝ) :
    2 * ((1 - sep / 2) + (1 / 8) * sep - 1) + sep * (1 - 2 * (1 / 8)) = 0 := by
  ring

/-- Worst-case left wall at `sep = 1`: `1 - (1/2 + 1/8) = 3/8`. -/
theorem a_min_kill_identity :
    (1 : ℝ) - (1 / 2 + 1 / 8) - 3 / 8 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16StageA
