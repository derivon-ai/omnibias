/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Slow-line chain-rule identities: V_v + ell = 0, the cleared form of
ell Z_V + Z_v = 0, and the declared ell floor 1 + 2 (1/50) (-1/2) = 49/50.
These theorems do not enclose fold I-map Z_x, sep>0, first-hit, G1, G4,
or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16ZSlowV

/-- Polynomial derivative of `V = 1 - v - nu v^2` plus `ell = 1 + 2 nu v`. -/
theorem V_v_slow_identity (nu v : ℝ) :
    ((-1) - 2 * nu * v) + (1 + 2 * nu * v) = 0 := by
  ring

/-- Cleared chain rule `ell (-Z_v) + Z_v ell`. -/
theorem ZV_chain_identity (ell zv : ℝ) :
    ell * (-zv) + zv * ell = 0 := by
  ring

/-- Declared `ell` floor on `nu <= 1/50`, `v >= -1/2`, cleared of denominators. -/
theorem ell_min_identity :
    (50 : ℝ) * 1 + 2 * 1 * (-1) - 49 = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16ZSlowV
