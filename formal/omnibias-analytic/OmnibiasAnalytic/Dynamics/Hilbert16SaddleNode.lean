/-
Exact double-root algebra and the linear fact that vanishing separation
gives no χ-attenuation.

These theorems do not identify the quadratic with the linear model, prove
a physical C2 remainder, or discharge G1 or Hilbert XVI.
-/

import OmnibiasAnalytic.Dynamics.Hilbert16ChiScale
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16SaddleNode

open Hilbert16ChiScale

/-- At vanishing discriminant, L = rstar^2 and lambda1 = -2 rstar. -/
theorem double_root_quadratic (rstar x : ℝ) :
    rstar ^ 2 + (-2 * rstar) * x + x ^ 2 = (x - rstar) ^ 2 := by
  ring

/-- The fold wall at the equilibrium is zero. -/
theorem wall_at_equilibrium (rstar : ℝ) :
    (rstar - rstar) ^ 2 = 0 := by
  ring

/-- Vanishing separation produces no linear χ-attenuation. -/
theorem vanishing_sep_linear_exit (κ : ℝ) :
    Real.exp (-(0 : ℝ) * κ) = 1 := by
  simp

/-- The matching coordinate is zero when the separation is zero. -/
theorem chi_of_zero_sep (r κ : ℝ) :
    chi 0 r κ = 0 := by
  unfold chi
  ring

/-- On a fixed-χ locus, sigma * kappa is (sigma / sep) times a bounded factor. -/
theorem sigma_kappa_on_chi_locus
    {σ sep χ r1 : ℝ} (hsep : sep ≠ 0) :
    σ * (χ * r1 / sep) = σ / sep * χ * r1 := by
  field_simp [hsep]

/-- The first outgoing root shrinks linearly with L at fixed negative lambda1. -/
theorem first_root_from_L
    {L lam1 sep : ℝ} (hsep : lam1 ^ 2 - sep ^ 2 = 4 * L) :
    (-lam1 - sep) / 2 * (-lam1 + sep) = 2 * L := by
  have h : (-lam1 - sep) * (-lam1 + sep) = lam1 ^ 2 - sep ^ 2 := by ring
  have h4 : (-lam1 - sep) * (-lam1 + sep) = 4 * L := by
    rw [h, hsep]
  calc
    (-lam1 - sep) / 2 * (-lam1 + sep) =
        ((-lam1 - sep) * (-lam1 + sep)) / 2 := by ring
    _ = (4 * L) / 2 := by rw [h4]
    _ = 2 * L := by ring

end OmnibiasAnalytic.Dynamics.Hilbert16SaddleNode
