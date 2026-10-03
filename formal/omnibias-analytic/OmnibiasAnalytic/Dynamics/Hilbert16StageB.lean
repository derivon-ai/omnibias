/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Kill-line Stage-B identities: r1 = 1 - sep/2 on lambda1 = -2, the
naive majorant (1/16)/(1/4) = 1/4, and the tracked exponent
2(1 - 2/16) = 7/4 at the compact edge. These theorems do not enclose
Stage A/C, C2, first-hit, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16StageB

/-- Kill-line incoming root: `2 (1 - sep/2) + sep - 2`. -/
theorem r1_half_identity (sep : ℝ) :
    2 * (1 - sep / 2) + sep - 2 = 0 := by
  ring

/-- Naive Stage-B majorant `(1/16) / (1/4) = 1/4`. -/
theorem dx_declared_identity :
    (1 : ℝ) / 16 / (1 / 4) - 1 / 4 = 0 := by
  ring

/-- Tracked exponent `2(1 - 2/16) = 7/4` at `eps = 1/16`. -/
theorem alpha_pos_identity :
    2 * (1 - 2 * (1 / 16 : ℝ)) - 7 / 4 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16StageB
