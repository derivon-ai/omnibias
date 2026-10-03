/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Compact aligned parametric-ε GRAZING E_σ identities: three slabs of
3/400 fill [1/25, 1/16], the interior pack point 1/20 sits in that
compact, and the L-pack 9/25 + 1/16 = 169/400. These theorems do not
enclose a uniform-in-ε first-hit for every ε, Lohner from V = 0 on
that compact, Z_x C2, G1, G4, or Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16ESigmaEpsSpan

/-- Declared compact lower endpoint `1/25`. -/
theorem espan_lo_identity :
    (1 / 25 : ℝ) - (1 / 25 : ℝ) = 0 := by
  ring

/-- Three slabs of `3/400` fill `[1/25, 1/16]`. -/
theorem espan_width_identity :
    (3 : ℝ) * (3 / 400 : ℝ) - ((1 / 16 : ℝ) - (1 / 25 : ℝ)) = 0 := by
  ring

/-- Interior pack point: `(1/20-1/25)*(1/16-1/20) = 1/8000`. -/
theorem espan_contains_identity :
    ((1 / 20 : ℝ) - (1 / 25 : ℝ)) * ((1 / 16 : ℝ) - (1 / 20 : ℝ))
      - (1 / 8000 : ℝ) = 0 := by
  ring

/-- Declared L-pack `9/25 + 1/16 = 169/400`. -/
theorem espan_pack_L_identity :
    ((9 / 25 : ℝ) + (1 / 16 : ℝ)) - (169 / 400 : ℝ) = 0 := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16ESigmaEpsSpan
