/-
Integer polynomial recurrences for the omnibias derivative tower.

These are the same recurrences as `omnibias.core.polynomials` /
`omnibias.core.verified.coeffs` (signed Eulerian, tanh / sech Riccati,
probabilist's Hermite). They are algebraic identities on `ℤ[X]`.

Coefficient *lists* are computable and match the Python exact-integer
generators. The polynomials themselves are the mathematical recurrences
(Mathlib's `X` is noncomputable).

Scope. This module does not state a finite-difference collapse, a continuum
PDE claim, or any asymptotic. A green build here means the recurrences and
the listed low-order coefficients hold, nothing beyond that.
-/

import Mathlib.Algebra.Polynomial.Derivative
import Mathlib.Tactic

namespace OmnibiasAnalytic.Tower

open Polynomial

/-! ### Computable coefficient lists (Python twins) -/

/-- Formal derivative of a coefficient list: `(c₁, 2 c₂, 3 c₃, …)`. -/
def derivCoeffs (cs : List ℤ) : List ℤ :=
  cs.tail.zipIdx.map fun ⟨c, i⟩ => (i + 1 : ℤ) * c

/-- Add `c` at index `i` of a pre-allocated list. -/
def addAt (cs : List ℤ) (i : ℕ) (c : ℤ) : List ℤ :=
  cs.set i (cs.getD i 0 + c)

/-- Convolve a derivative list with `(0, 1, -1)`: `X (1 - X) p'`. -/
def mulXOneMinusX (deriv : List ℤ) : List ℤ :=
  deriv.zipIdx.foldl (fun acc ⟨c, i⟩ =>
    addAt (addAt acc (i + 1) c) (i + 2) (-c))
    (List.replicate (deriv.length + 2) (0 : ℤ))

/-- Convolve a derivative list with `(1, 0, -1)`: `(1 - X^2) p'`. -/
def mulOneMinusXSq (deriv : List ℤ) : List ℤ :=
  deriv.zipIdx.foldl (fun acc ⟨c, i⟩ =>
    addAt (addAt acc i c) (i + 2) (-c))
    (List.replicate (deriv.length + 2) (0 : ℤ))

/-- Eulerian coefficients: `P_0 = [0, 1]`, then `X(1-X) P'`. -/
def sigmoidCoeffs : ℕ → List ℤ
  | 0 => [0, 1]
  | n + 1 => mulXOneMinusX (derivCoeffs (sigmoidCoeffs n))

/-- Tanh coefficients: `T_0 = [0, 1]`, then `(1-X^2) T'`. -/
def tanhCoeffs : ℕ → List ℤ
  | 0 => [0, 1]
  | n + 1 => mulOneMinusXSq (derivCoeffs (tanhCoeffs n))

/-- Sech coefficients: `Q_0 = [1]`, then `(1-X^2) Q' - X Q`. -/
def sechCoeffs : ℕ → List ℤ
  | 0 => [1]
  | n + 1 =>
    let prev := sechCoeffs n
    let fromDeriv :=
      derivCoeffs prev |>.zipIdx.foldl (fun acc ⟨c, i⟩ =>
        addAt (addAt acc i c) (i + 2) (-c))
        (List.replicate (prev.length + 1) (0 : ℤ))
    prev.zipIdx.foldl (fun acc ⟨c, i⟩ => addAt acc (i + 1) (-c)) fromDeriv

/-- Probabilist's Hermite coefficients. -/
def hermiteCoeffs : ℕ → List ℤ
  | 0 => [1]
  | 1 => [0, 1]
  | n + 2 =>
    let prev1 := hermiteCoeffs (n + 1)
    let prev2 := hermiteCoeffs n
    let acc := List.replicate (n + 3) (0 : ℤ)
    let acc := prev1.zipIdx.foldl (fun a ⟨c, k⟩ => addAt a (k + 1) c) acc
    prev2.zipIdx.foldl (fun a ⟨c, k⟩ => addAt a k (-((n + 1 : ℤ) * c))) acc

@[simp] theorem sigmoidCoeffs_zero : sigmoidCoeffs 0 = [0, 1] := rfl
@[simp] theorem tanhCoeffs_zero : tanhCoeffs 0 = [0, 1] := rfl
@[simp] theorem sechCoeffs_zero : sechCoeffs 0 = [1] := rfl
@[simp] theorem hermiteCoeffs_zero : hermiteCoeffs 0 = [1] := rfl
@[simp] theorem hermiteCoeffs_one : hermiteCoeffs 1 = [0, 1] := rfl

theorem sigmoidCoeffs_one : sigmoidCoeffs 1 = [0, 1, -1] := by native_decide
theorem sigmoidCoeffs_two : sigmoidCoeffs 2 = [0, 1, -3, 2] := by native_decide
theorem tanhCoeffs_one : tanhCoeffs 1 = [1, 0, -1] := by native_decide
theorem sechCoeffs_one : sechCoeffs 1 = [0, -1] := by native_decide
theorem sechCoeffs_two : sechCoeffs 2 = [-1, 0, 2] := by native_decide
theorem hermiteCoeffs_two : hermiteCoeffs 2 = [-1, 0, 1] := by native_decide

/-- Constant terms `Q_n(0)` are the Euler (secant) numbers `E_n` for `n ≤ 8`. -/
theorem sech_euler_numbers :
    List.map (fun n => (sechCoeffs n).headD 0) [0, 1, 2, 3, 4, 5, 6, 7, 8] =
      [1, 0, -1, 0, 5, 0, -61, 0, 1385] := by
  native_decide

/-! ### Mathematical recurrences on `ℤ[X]` -/

/-- Eulerian tower: `P_0 = X`, `P_{n+1} = X (1 - X) P_n'`. -/
noncomputable def sigmoidPoly : ℕ → ℤ[X]
  | 0 => X
  | n + 1 => (X * (1 - X)) * derivative (sigmoidPoly n)

/-- Tanh / Legendre-style tower: `T_0 = X`, `T_{n+1} = (1 - X^2) T_n'`. -/
noncomputable def tanhPoly : ℕ → ℤ[X]
  | 0 => X
  | n + 1 => (1 - X ^ 2) * derivative (tanhPoly n)

/-- Sech / Euler-number tower: `Q_0 = 1`, `Q_{n+1} = (1 - X^2) Q_n' - X Q_n`. -/
noncomputable def sechPoly : ℕ → ℤ[X]
  | 0 => 1
  | n + 1 => (1 - X ^ 2) * derivative (sechPoly n) - X * sechPoly n

/-- Probabilist's Hermite: `He_0 = 1`, `He_1 = X`, `He_{n+1} = X He_n - n He_{n-1}`. -/
noncomputable def hermitePoly : ℕ → ℤ[X]
  | 0 => 1
  | 1 => X
  | n + 2 => X * hermitePoly (n + 1) - C (n + 1 : ℤ) * hermitePoly n

@[simp] theorem sigmoidPoly_zero : sigmoidPoly 0 = X := rfl
@[simp] theorem tanhPoly_zero : tanhPoly 0 = X := rfl
@[simp] theorem sechPoly_zero : sechPoly 0 = 1 := rfl
@[simp] theorem hermitePoly_zero : hermitePoly 0 = 1 := rfl
@[simp] theorem hermitePoly_one : hermitePoly 1 = X := rfl

theorem sigmoidPoly_succ (n : ℕ) :
    sigmoidPoly (n + 1) = (X * (1 - X)) * derivative (sigmoidPoly n) := rfl

theorem tanhPoly_succ (n : ℕ) :
    tanhPoly (n + 1) = (1 - X ^ 2) * derivative (tanhPoly n) := rfl

theorem sechPoly_succ (n : ℕ) :
    sechPoly (n + 1) = (1 - X ^ 2) * derivative (sechPoly n) - X * sechPoly n :=
  rfl

theorem hermitePoly_succ (n : ℕ) :
    hermitePoly (n + 2) = X * hermitePoly (n + 1) - C (n + 1 : ℤ) * hermitePoly n :=
  rfl

/-- `P_1(s) = s - s^2`. -/
theorem sigmoidPoly_one : sigmoidPoly 1 = X - X ^ 2 := by
  simp [sigmoidPoly, derivative_X]
  ring

/-- `P_2(s) = s - 3 s^2 + 2 s^3`. -/
theorem sigmoidPoly_two : sigmoidPoly 2 = X - 3 * X ^ 2 + 2 * X ^ 3 := by
  rw [sigmoidPoly_succ, sigmoidPoly_one]
  simp [derivative_sub, derivative_X]
  ring

/-- `T_1(t) = 1 - t^2`. -/
theorem tanhPoly_one : tanhPoly 1 = 1 - X ^ 2 := by
  simp [tanhPoly, derivative_X]

/-- `He_2 = X^2 - 1`. -/
theorem hermitePoly_two : hermitePoly 2 = X ^ 2 - 1 := by
  simp [hermitePoly]
  ring

/-- `Q_1(t) = -t`. -/
theorem sechPoly_one : sechPoly 1 = -X := by
  simp [sechPoly]

/-- `Q_2(t) = 2 t^2 - 1`. -/
theorem sechPoly_two : sechPoly 2 = 2 * X ^ 2 - 1 := by
  rw [sechPoly_succ, sechPoly_one]
  simp [derivative_neg, derivative_X]
  ring

/-- Aliases used by the Mathlib bridge (computable lists). -/
def sigmoidCoeffList := sigmoidCoeffs
def tanhCoeffList := tanhCoeffs
def sechCoeffList := sechCoeffs
def hermiteCoeffList := hermiteCoeffs

private lemma hermite_deriv_step (n : ℕ)
    (ih1 : derivative (hermitePoly (n + 2)) = C (n + 2 : ℤ) * hermitePoly (n + 1))
    (ih0 : derivative (hermitePoly (n + 1)) = C (n + 1 : ℤ) * hermitePoly n) :
    derivative (hermitePoly (n + 3)) = C (n + 3 : ℤ) * hermitePoly (n + 2) := by
  rw [hermitePoly_succ (n + 1)]
  rw [derivative_sub, derivative_mul, derivative_X, one_mul, derivative_C_mul]
  rw [ih1, ih0]
  rw [hermitePoly_succ n]
  simp [Nat.cast_succ]
  ring

/-- Probabilist's Hermite derivative: (He_{n+1})' = (n+1) He_n. -/
theorem hermitePoly_deriv : ∀ n : ℕ,
    derivative (hermitePoly (n + 1)) = C (n + 1 : ℤ) * hermitePoly n
  | 0 => by
    simp [hermitePoly, derivative_X]
  | 1 => by
    simp [hermitePoly, derivative_sub, derivative_mul, derivative_X]
    ring
  | n + 2 =>
    hermite_deriv_step n (hermitePoly_deriv (n + 1)) (hermitePoly_deriv n)

end OmnibiasAnalytic.Tower
