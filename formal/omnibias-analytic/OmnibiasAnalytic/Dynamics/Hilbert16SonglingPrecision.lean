/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

/-
Exact coefficient cancellation in the Songling field. This proves only the
rational identity whose binary64 enclosure loses the epsilon sign. It does not
replay a return map, certify four cycles, or prove Hilbert XVI.
-/

import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16SonglingPrecision

/-- Removing the dominant `-25-9*delta` baseline leaves exactly `8*epsilon`. -/
theorem xyCoefficientCancellation (epsilon delta : ℚ) :
    (-25 + 8 * epsilon - 9 * delta) - (-25 - 9 * delta) =
      8 * epsilon := by
  ring

end OmnibiasAnalytic.Dynamics.Hilbert16SonglingPrecision
