/-
Exact W-coordinate identities for the two-blow-up attempt.

These theorems do not prove a physical C2 remainder, G1, or Hilbert XVI.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16TwoBlowup

/-- The central-strip height reconstruction: epsilon * W = h^epsilon. -/
theorem w_recovers_height_power
    {ε W hpow : ℝ} (hε : ε ≠ 0) (hW : W = hpow / ε) :
    ε * W = hpow := by
  rw [hW]
  exact mul_div_cancel₀ hpow hε

/-- The outgoing variational factor is a ratio of W-labels. -/
theorem outgoing_as_w_ratio
    {ε Wmax We hmax_pow he_pow : ℝ}
    (hε : ε ≠ 0) (hWe : We ≠ 0)
    (hmax : hmax_pow = ε * Wmax) (he : he_pow = ε * We) :
    hmax_pow / he_pow = Wmax / We := by
  rw [hmax, he]
  field_simp [hε]

/-- Exact central-strip log-W velocity. -/
def logW_tau (u : ℝ) : ℝ := -u

theorem logW_tau_neg_u (u : ℝ) :
    logW_tau u = -u := rfl

/-- The product of the two event scales is the separation-scale event. -/
theorem two_scale_product
    {σf σs χ r1 sep : ℝ} (hsep : sep ≠ 0) (hσf : σf ≠ 0) :
    σf * (χ * r1 / sep) * (σs / σf) = σs * (χ * r1 / sep) := by
  field_simp [hsep, hσf]

end OmnibiasAnalytic.Dynamics.Hilbert16TwoBlowup
