/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Kill-line neck identities: 1/4+7/4=2, 70*(1/40)=7/4, and
2*(4-1/4)/(1/16)=120. These theorems do not enclose a Lohner tube,
eps > 1/16, the shrinking interface, complete first-hit on chart O,
C2, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16StageCUniform

/-- Neck grid ``1/4 + 7/4 = 2``. -/
theorem unif_neck_identity :
    (1 : ℝ) / 4 + 7 / 4 - 2 = 0 := by
  ring

/-- Phase count ``70 * (1/40) = 7/4``. -/
theorem unif_phases_identity :
    (70 : ℝ) * (1 / 40) - 7 / 4 = 0 := by
  ring

/-- Time majorant at ``eps = 1/16`` is ``120``. -/
theorem unif_time_identity :
    (2 : ℝ) * (4 - 1 / 4) / (1 / 16) - 120 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16StageCUniform
