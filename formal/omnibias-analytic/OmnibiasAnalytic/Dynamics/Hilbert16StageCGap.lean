/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Kill-line Stage-C start-gap identities: wall 1/8 - 1/16 = 1/16,
h_1 cube (1/16)^3 = 1/4096, and exact wall T-h = 1/4096 at eps = 1/16.
These theorems do not enclose an outgoing orbit, first-hit, C2, G1, G4,
or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16StageCGap

/-- Wall start gap ``1/8 - 1/16 = 1/16``. -/
theorem start_gap_identity :
    (1 / 8 : ℝ) - 1 / 16 - 1 / 16 = 0 := by
  ring

/-- Height cube ``(1/16)^3 = 1/4096``. -/
theorem h1_cube_identity :
    (1 / 16 : ℝ) * (1 / 16) * (1 / 16) - 1 / 4096 = 0 := by
  ring

/-- Exact wall ``T-h`` ``(1/8)*(1/16)^2 - (1/16)^3 = 1/4096``. -/
theorem wall_gap_identity :
    (1 / 8 : ℝ) * ((1 / 16) * (1 / 16)) - (1 / 16) * (1 / 16) * (1 / 16) - 1 / 4096 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16StageCGap
