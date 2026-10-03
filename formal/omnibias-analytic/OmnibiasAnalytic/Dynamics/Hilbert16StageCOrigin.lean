/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Kill-line chart-O matching-chart identities: r1=0 at sep=2,
r1=1/4 at sep=3/2, and r1=1/8 at sep=7/4. These theorems do not
enclose every r1, complete first-hit on chart O, C2, G1, G4, or
Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16StageCOrigin

/-- Chart-O limit ``r1 = 1 - 2/2 = 0``. -/
theorem r1_origin_identity :
    (1 : ℝ) - 2 / 2 = 0 := by
  ring

/-- Pack member ``r1 = 1 - (3/2)/2 = 1/4``. -/
theorem r1_o_quarter_identity :
    (1 : ℝ) - (3 / 2) / 2 - 1 / 4 = 0 := by
  ring

/-- Pack member ``r1 = 1 - (7/4)/2 = 1/8``. -/
theorem r1_o_eighth_identity :
    (1 : ℝ) - (7 / 4) / 2 - 1 / 8 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16StageCOrigin
