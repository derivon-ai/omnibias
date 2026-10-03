/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Kill-line comparison identities: 2*1-1^2=1, (1/4)/(1/32)=8, and
310*(1/40)=31/4. These theorems do not enclose every eps, a Lohner
tube, complete first-hit on chart O, C2, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16StageCCompare

/-- Root excess at ``r1 = 1`` is ``2*1 - 1^2 = 1``. -/
theorem cmp_emax_identity :
    (2 : ℝ) * 1 - 1 ^ 2 - 1 = 0 := by
  ring

/-- Far matching section ``(1/4)/(1/32) = 8``. -/
theorem cmp_xfar_identity :
    ((1 : ℝ) / 4) / (1 / 32) - 8 = 0 := by
  ring

/-- Phase grid ``310 * (1/40) = 31/4``. -/
theorem cmp_phases_identity :
    (310 : ℝ) * (1 / 40) - (8 - 1 / 4) = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16StageCCompare
