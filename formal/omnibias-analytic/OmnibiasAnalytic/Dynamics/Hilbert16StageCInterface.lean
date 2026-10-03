/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Kill-line entrance identities: 1+(1/2)*(1/2-2)=1/4, 60*(1/40)=3/2, and
5*(1/4)/(1/16)^2=320. These theorems do not enclose a Lohner tube, the
height-section flag on L=1/n, eps > 1/16, complete first-hit on chart O,
C2, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16StageCInterface

/-- Entrance gap ``1 + (1/2)*(1/2-2) = 1/4``. -/
theorem iface_edge_identity :
    (1 : ℝ) + (1 / 2) * (1 / 2 - 2) - 1 / 4 = 0 := by
  ring

/-- Neck grid ``60 * (1/40) = 3/2``. -/
theorem iface_phases_identity :
    (60 : ℝ) * (1 / 40) - 3 / 2 = 0 := by
  ring

/-- Longest time majorant ``5 * (1/4) / (1/16)^2 = 320``. -/
theorem iface_time_identity :
    (5 : ℝ) * (1 / 4) / ((1 / 16) ^ 2) - 320 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16StageCInterface
