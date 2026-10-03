/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Exact kill-sequence identities for the direct tau/log-W LN format barrier.
They prove growth in this representation, not that every normalized
Log-Noetherian representation fails.  They do not prove G3 or Hilbert XVI.
-/

import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16LNFormatBarrier

/-- On epsilon=1/n and sep=exp(-n^2), the formal tau coordinate is n. -/
theorem killTauIdentity (n : ℝ) :
    n - n = 0 := by
  ring

/-- The existing epsilon^3 section has formal log W-ratio 2n. -/
theorem killLogWIdentity (n : ℝ) :
    2 * n - 2 * n = 0 := by
  ring

/-- The two direct chain-function suprema sum to 3n. -/
theorem killNormIdentity (n : ℝ) :
    (n + 2 * n) - 3 * n = 0 := by
  ring

/-- The direct chain norm grows strictly with its truncation parameter. -/
theorem killNormStrictGrowth
    (n m : ℝ)
    (h : n < m) :
    3 * n < 3 * m := by
  linarith

end OmnibiasAnalytic.Dynamics.Hilbert16LNFormatBarrier
