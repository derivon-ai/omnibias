/-
Link theorems: the n-th derivative equals the tower polynomial.

For each family the identity is proved by induction on `n` plus the chain
rule. No finite-difference collapse, continuum PDE, or asymptotic is stated.
-/

import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Polynomial
import Mathlib.Analysis.Calculus.IteratedDeriv.Defs
import Mathlib.Analysis.Calculus.IteratedDeriv.Lemmas
import OmnibiasAnalytic.Tower.Poly
import OmnibiasAnalytic.Tower.Riccati

namespace OmnibiasAnalytic.Tower

noncomputable section

open Polynomial

/-- Chain rule for evaluating an integer polynomial along a real map. -/
theorem deriv_aeval_comp {p : ℤ[X]} {f : ℝ → ℝ} {x : ℝ}
    (hf : DifferentiableAt ℝ f x) :
    deriv (fun y => aeval (f y) p) x = aeval (f x) (derivative p) * deriv f x := by
  have hp : HasDerivAt (fun t : ℝ => aeval t p) (aeval (f x) (derivative p)) (f x) :=
    p.hasDerivAt_aeval (f x)
  have hf' : HasDerivAt f (deriv f x) x := hf.hasDerivAt
  exact (hp.comp x hf').deriv

/-- If `f' = m(f)` and `P_{n+1} = m P_n'`, then `f^{(n)} = P_n(f)`. -/
theorem iteratedDeriv_of_riccati
    {f : ℝ → ℝ} {P : ℕ → ℤ[X]} {m : ℤ[X]}
    (hf : ContDiff ℝ ⊤ f)
    (hP0 : P 0 = X)
    (hPsucc : ∀ n, P (n + 1) = m * derivative (P n))
    (hm : ∀ x, deriv f x = aeval (f x) m) (n : ℕ) (x : ℝ) :
    iteratedDeriv n f x = aeval (f x) (P n) := by
  have hfdiff : Differentiable ℝ f := hf.differentiable (by decide)
  revert x
  induction n with
  | zero =>
    intro x
    simp [iteratedDeriv_zero, hP0]
  | succ n ih =>
    intro x
    have hfun : iteratedDeriv n f = fun y => aeval (f y) (P n) := funext ih
    rw [iteratedDeriv_succ, hfun, deriv_aeval_comp (hfdiff x), hm x, hPsucc n]
    simp [aeval_mul, mul_comm]

theorem iteratedDeriv_sigmoid (n : ℕ) (x : ℝ) :
    iteratedDeriv n Real.sigmoid x = aeval (Real.sigmoid x) (sigmoidPoly n) := by
  refine iteratedDeriv_of_riccati contDiff_sigmoid sigmoidPoly_zero
    (fun k => sigmoidPoly_succ k) ?_ n x
  intro y
  rw [deriv_sigmoid]
  simp [aeval_mul, aeval_sub, aeval_one, aeval_X]

theorem iteratedDeriv_tanh (n : ℕ) (x : ℝ) :
    iteratedDeriv n Real.tanh x = aeval (Real.tanh x) (tanhPoly n) := by
  refine iteratedDeriv_of_riccati contDiff_tanh tanhPoly_zero
    (fun k => tanhPoly_succ k) ?_ n x
  intro y
  rw [deriv_tanh]
  simp [aeval_sub, aeval_one, aeval_X_pow]

/-- Product rule for `Q(tanh) * sech`. -/
theorem deriv_sech_tower (p : ℤ[X]) (x : ℝ) :
    deriv (fun y => aeval (Real.tanh y) p * sech y) x =
      aeval (Real.tanh x) ((1 - X ^ 2) * derivative p - X * p) * sech x := by
  have hQ : HasDerivAt (fun y => aeval (Real.tanh y) p)
      (aeval (Real.tanh x) (derivative p) * (1 - Real.tanh x ^ 2)) x :=
    (p.hasDerivAt_aeval (Real.tanh x)).comp x (hasDerivAt_tanh x)
  have hS : HasDerivAt sech (-Real.tanh x * sech x) x := hasDerivAt_sech x
  rw [(hQ.fun_mul hS).deriv]
  simp [aeval_sub, aeval_mul, aeval_one, aeval_X_pow, aeval_X]
  ring

theorem iteratedDeriv_sech (n : ℕ) (x : ℝ) :
    iteratedDeriv n sech x = aeval (Real.tanh x) (sechPoly n) * sech x := by
  revert x
  induction n with
  | zero =>
    intro x
    simp [iteratedDeriv_zero, sechPoly_zero]
  | succ n ih =>
    intro x
    have hfun : iteratedDeriv n sech =
        fun y => aeval (Real.tanh y) (sechPoly n) * sech y := funext ih
    rw [iteratedDeriv_succ, hfun, deriv_sech_tower, sechPoly_succ]

/-- One inductive step of the Hermite / Gaussian tower. -/
theorem deriv_gaussian_tower (n : ℕ) (x : ℝ) :
    deriv (fun y => ((-1 : ℝ) ^ n) * aeval y (hermitePoly n) * gaussian y) x =
      ((-1 : ℝ) ^ (n + 1)) * aeval x (hermitePoly (n + 1)) * gaussian x := by
  have hHe : DifferentiableAt ℝ (fun y : ℝ => aeval y (hermitePoly n)) x :=
    (hermitePoly n).differentiableAt_aeval
  have hG : DifferentiableAt ℝ gaussian x := (hasDerivAt_gaussian x).differentiableAt
  have hreassoc :
      (fun y => ((-1 : ℝ) ^ n) * aeval y (hermitePoly n) * gaussian y) =
        fun y => ((-1 : ℝ) ^ n) * (aeval y (hermitePoly n) * gaussian y) := by
    funext y; ring
  rw [hreassoc, deriv_const_mul_field, deriv_fun_mul hHe hG, deriv_gaussian,
    (hermitePoly n).hasDerivAt_aeval x |>.deriv]
  have hident :
      aeval x (derivative (hermitePoly n)) - x * aeval x (hermitePoly n) =
        -aeval x (hermitePoly (n + 1)) := by
    cases n with
    | zero =>
      simp [hermitePoly, aeval_X]
    | succ k =>
      have hd := hermitePoly_deriv k
      have hrec := hermitePoly_succ k
      simp [hd, hrec, aeval_sub, aeval_mul, aeval_X]
  simp [pow_succ]
  linear_combination ((-1 : ℝ) ^ n * gaussian x) * hident

theorem iteratedDeriv_gaussian (n : ℕ) (x : ℝ) :
    iteratedDeriv n gaussian x =
      ((-1 : ℝ) ^ n) * aeval x (hermitePoly n) * gaussian x := by
  revert x
  induction n with
  | zero =>
    intro x
    simp [iteratedDeriv_zero, hermitePoly_zero]
  | succ n ih =>
    intro x
    have hfun : iteratedDeriv n gaussian =
        fun y => ((-1 : ℝ) ^ n) * aeval y (hermitePoly n) * gaussian y := funext ih
    rw [iteratedDeriv_succ, hfun, deriv_gaussian_tower]

/-- Softplus is the antiderivative of sigmoid, so its tower is the shifted Eulerian tower. -/
theorem iteratedDeriv_softplus_succ (n : ℕ) (x : ℝ) :
    iteratedDeriv (n + 1) softplus x = iteratedDeriv n Real.sigmoid x := by
  have hfun : deriv softplus = Real.sigmoid := funext deriv_softplus
  rw [iteratedDeriv_succ', hfun]

theorem iteratedDeriv_softplus (n : ℕ) (x : ℝ) :
    iteratedDeriv (n + 1) softplus x = aeval (Real.sigmoid x) (sigmoidPoly n) := by
  rw [iteratedDeriv_softplus_succ, iteratedDeriv_sigmoid]

end

end OmnibiasAnalytic.Tower
