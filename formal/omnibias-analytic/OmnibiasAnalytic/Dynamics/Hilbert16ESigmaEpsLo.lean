/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Lower aligned parametric-ε GRAZING E_σ identities: six slabs of
1/128 fill [1/64, 1/16], the geometric midpoint 1/32 sits in that
compact, and the L-pack 9/25 + 1/16 = 169/400. These theorems do not
enclose a uniform-in-ε first-hit for every ε, Lohner from V = 0 on
that compact, Z_x C2, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16ESigmaEpsLo

/-- Declared compact lower endpoint `1/64`. -/
theorem elo_lo_identity :
    (1 / 64 : ℝ) - (1 / 64 : ℝ) = 0 := by
  ring

/-- Six slabs of `1/128` fill `[1/64, 1/16]`. -/
theorem elo_width_identity :
    (6 : ℝ) * (1 / 128 : ℝ) - ((1 / 16 : ℝ) - (1 / 64 : ℝ)) = 0 := by
  ring

/-- Geometric midpoint: `(1/32-1/64)*(1/16-1/32) = 1/2048`. -/
theorem elo_contains_32_identity :
    ((1 / 32 : ℝ) - (1 / 64 : ℝ)) * ((1 / 16 : ℝ) - (1 / 32 : ℝ))
      - (1 / 2048 : ℝ) = 0 := by
  ring

/-- Declared L-pack `9/25 + 1/16 = 169/400`. -/
theorem elo_pack_L_identity :
    ((9 / 25 : ℝ) + (1 / 16 : ℝ)) - (169 / 400 : ℝ) = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16ESigmaEpsLo
