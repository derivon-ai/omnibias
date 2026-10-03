/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/

import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
Elementary conditional lemmas for a logarithmic-normalized (LN) passage cell.

The monomial statements use natural exponents, where Mathlib proves the
derivatives directly.  The Cauchy estimate below is an explicit hypothesis:
this module does not derive it from holomorphy, construct a complex passage,
or assert any Hilbert XVI consequence.
-/

namespace OmnibiasAnalytic.Dynamics.Hilbert16LNCell

noncomputable section

/-- The pointwise logarithmic derivation `z d/dz`, expressed using `deriv`. -/
def logDerivation (f : ℝ → ℝ) (z : ℝ) : ℝ := z * deriv f z

/-- The actual derivative of a scalar multiple of a natural-power monomial. -/
theorem hasDerivAt_const_mul_pow_nat (c z : ℝ) (n : ℕ) :
    HasDerivAt (fun x : ℝ => c * x ^ n)
      (c * ((n : ℝ) * z ^ (n - 1))) z :=
  (hasDerivAt_pow n z).const_mul c

/-- A natural-power monomial is an eigenvector of `z d/dz`.

No real- or complex-exponent power is claimed here.
-/
theorem monomial_logarithmic_derivation_eigenvalue (c z : ℝ) (n : ℕ) :
    logDerivation (fun x : ℝ => c * x ^ n) z =
      (n : ℝ) * (c * z ^ n) := by
  unfold logDerivation
  rw [(hasDerivAt_const_mul_pow_nat c z n).deriv]
  cases n with
  | zero => simp
  | succ n =>
      simp only [Nat.cast_succ, Nat.succ_sub_one, pow_succ]
      ring

/-- Applying `z d/dz` twice multiplies a natural-power monomial by `n²`. -/
theorem monomial_second_logarithmic_derivation_eigenvalue (c z : ℝ) (n : ℕ) :
    logDerivation
        (fun x : ℝ => logDerivation (fun y : ℝ => c * y ^ n) x) z =
      (n : ℝ) ^ 2 * (c * z ^ n) := by
  have hinner :
      (fun x : ℝ => logDerivation (fun y : ℝ => c * y ^ n) x) =
        (fun x : ℝ => ((n : ℝ) * c) * x ^ n) := by
    funext x
    rw [monomial_logarithmic_derivation_eigenvalue]
    ring
  rw [hinner, monomial_logarithmic_derivation_eigenvalue]
  ring

/-- Conditional order-two Cauchy consequence for a supplied log-chart jet.

`hCauchy` is the analytic input: a caller must establish it from a genuine
holomorphic extension and a supremum bound.  This theorem only specializes
that input to `k = 2` and divides by the positive strip margin.
-/
theorem conditional_log_chart_cauchy_bound_order_two
    {jet : ℕ → ℝ} {M δ : ℝ}
    (_hM : 0 ≤ M) (hδ : 0 < δ)
    (hCauchy : ∀ k : ℕ,
      |jet k| * δ ^ k ≤ (Nat.factorial k : ℝ) * M) :
    |jet 2| ≤ 2 * M / δ ^ 2 := by
  have hδ2 : 0 < δ ^ 2 := sq_pos_of_pos hδ
  rw [le_div_iff₀ hδ2]
  simpa using hCauchy 2

/-- The conditional `k = 2` estimate after a real quadratic rescaling. -/
theorem conditional_log_chart_cauchy_bound_order_two_scaled
    {jet : ℕ → ℝ} {M δ scale : ℝ}
    (hM : 0 ≤ M) (hδ : 0 < δ)
    (hCauchy : ∀ k : ℕ,
      |jet k| * δ ^ k ≤ (Nat.factorial k : ℝ) * M) :
    |scale ^ 2 * jet 2| ≤ 2 * scale ^ 2 * M / δ ^ 2 := by
  have htwo :=
    conditional_log_chart_cauchy_bound_order_two hM hδ hCauchy
  calc
    |scale ^ 2 * jet 2| = scale ^ 2 * |jet 2| := by
      rw [abs_mul, abs_of_nonneg (sq_nonneg scale)]
    _ ≤ scale ^ 2 * (2 * M / δ ^ 2) :=
      mul_le_mul_of_nonneg_left htwo (sq_nonneg scale)
    _ = 2 * scale ^ 2 * M / δ ^ 2 := by ring

/-- A perturbation bounded by `δMax` stays in the logarithmic strip whenever
the unperturbed point has more than that much strict margin. -/
theorem delta_preserves_log_strip_margin
    {logCoordinate perturbation δMax stripRadius : ℝ}
    (hperturbation : |perturbation| ≤ δMax)
    (hmargin : |logCoordinate| + δMax < stripRadius) :
    |logCoordinate + perturbation| < stripRadius := by
  calc
    |logCoordinate + perturbation| ≤
        |logCoordinate| + |perturbation| := abs_add_le _ _
    _ ≤ |logCoordinate| + δMax := by linarith
    _ < stripRadius := hmargin

end

end OmnibiasAnalytic.Dynamics.Hilbert16LNCell
