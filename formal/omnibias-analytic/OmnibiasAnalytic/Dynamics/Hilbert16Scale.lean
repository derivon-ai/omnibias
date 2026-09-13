/-
Confluent diagonal and fixed-product scale identities.

These theorems establish an actual removable divided-difference limit,
derivatives of the exponential scale paths, the second derivative of a
weighted first derivative, and the signed-root pole margin. The Python
interval/series evaluator and any physical passage remainder are outside
these statements. No Hilbert XVI or Dulac closure theorem is asserted.
-/

import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.Slope
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace OmnibiasAnalytic.Dynamics.Hilbert16Scale

open Filter Topology

/-- The derivative furnishing the power compensator's confluent diagonal. -/
theorem hasDerivAt_exponential_parameter (a L : ℝ) :
    HasDerivAt (fun b : ℝ => Real.exp (b * L)) (L * Real.exp (a * L)) a := by
  have h := (Real.hasDerivAt_exp (L * a)).comp a ((hasDerivAt_id a).const_mul L)
  convert h using 1 <;> try rfl
  · funext b
    change Real.exp (b * L) = Real.exp (L * b)
    rw [mul_comm b L]
  · rw [mul_comm a L]
    ring

/-- An actual limit of the divided difference, including either side. -/
theorem power_compensator_diagonal_limit (a L : ℝ) :
    Tendsto (slope (fun b : ℝ => Real.exp (b * L)) a) (𝓝[≠] a)
      (𝓝 (L * Real.exp (a * L))) :=
  (hasDerivAt_exponential_parameter a L).tendsto_slope

/-- The two scales keep epsilon=omega*u fixed at every real path parameter. -/
theorem fixed_product_path (omega u s : ℝ) :
    (omega * Real.exp (-s)) * (u * Real.exp s) = omega * u := by
  calc
    (omega * Real.exp (-s)) * (u * Real.exp s) =
        omega * u * (Real.exp (-s) * Real.exp s) := by ring
    _ = omega * u := by rw [← Real.exp_add]; simp

theorem hasDerivAt_omega_path (omega s : ℝ) :
    HasDerivAt (fun t : ℝ => omega * Real.exp (-t))
      (-(omega * Real.exp (-s))) s := by
  have h : HasDerivAt (fun t : ℝ => omega * Real.exp (-t))
      (omega * (Real.exp (-s) * -1)) s :=
    ((Real.hasDerivAt_exp (-s)).comp s (hasDerivAt_id s).neg).const_mul omega
  exact h.congr_deriv (by ring)

theorem hasDerivAt_u_path (u s : ℝ) :
    HasDerivAt (fun t : ℝ => u * Real.exp t) (u * Real.exp s) s :=
  (Real.hasDerivAt_exp s).const_mul u

/-- The derivative of E R along an actual scale path includes acceleration.

A and B are the first partials restricted to the path. Their stated actual
derivatives are the two Hessian contractions. Those identifications remain
explicit hypotheses; no callbacks or approximations are silently promoted.
-/
theorem weighted_second_derivative
    {W U A B : ℝ → ℝ} {s omega u rw ru rww rwu ruu : ℝ}
    (hW : HasDerivAt W (-omega) s) (hU : HasDerivAt U u s)
    (hA : HasDerivAt A (-omega * rww + u * rwu) s)
    (hB : HasDerivAt B (-omega * rwu + u * ruu) s)
    (hW0 : W s = omega) (hU0 : U s = u)
    (hA0 : A s = rw) (hB0 : B s = ru) :
    HasDerivAt (fun t => U t * B t - W t * A t)
      (omega ^ 2 * rww - 2 * omega * u * rwu + u ^ 2 * ruu
        + omega * rw + u * ru) s := by
  have h : HasDerivAt (fun t => U t * B t - W t * A t)
      (u * B s + U s * (-omega * rwu + u * ruu)
        - (-omega * A s + W s * (-omega * rww + u * rwu))) s :=
    (hU.mul hB).sub (hW.mul hA)
  apply h.congr_deriv
  rw [hW0, hU0, hA0, hB0]
  ring

/-- On R=omega*u the acceleration cancels the straight-Hessian term. -/
theorem product_second_derivative_cancellation (omega u : ℝ) :
    omega ^ 2 * 0 - 2 * omega * u * 1 + u ^ 2 * 0 + omega * u + u * omega = 0 := by
  ring

/-- The negative-delta root primitive has no pole before the endpoint margin. -/
theorem signed_root_pole_margin {delta v t : ℝ}
    (hd : delta ≤ 0) (ht : t ^ 2 ≤ v ^ 2) (hv : 0 < 1 + delta * v ^ 2) :
    0 < 1 + delta * t ^ 2 := by
  have h := mul_le_mul_of_nonpos_left ht hd
  linarith

end OmnibiasAnalytic.Dynamics.Hilbert16Scale
