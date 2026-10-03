/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Matching-chart fold I-map Z_x identities: the cleared chain
ell Z_x - Z_v nu, the declared rate (1/4)(1/50)/(49/50) = 1/196,
and x=3/2 interior to [7/5, 8/5]. These theorems do not enclose
sep>0, first-hit, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16FoldZX

/-- Cleared matching chain `ell (Z_v nu / ell) - Z_v nu`. -/
theorem zx_chain_identity (ell zv nu : ℝ) :
    ell * (zv * nu) - zv * nu * ell = 0 := by
  ring

/-- Declared rate `1/196 = 1/(4*49)`. -/
theorem zx_declared_identity :
    (196 : ℝ) - 4 * 49 = 0 := by
  ring

/-- `x=3/2` interior product `(8/5 - 3/2)(3/2 - 7/5) = 1/100`. -/
theorem fold_x_interior_identity :
    ((8 : ℝ) / 5 - 3 / 2) * (3 / 2 - 7 / 5) - 1 / 100 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16FoldZX
