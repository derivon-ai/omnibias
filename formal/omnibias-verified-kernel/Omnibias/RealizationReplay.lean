/- SPDX-License-Identifier: Apache-2.0
Copyright (C) 2026 Derivon -/
import Omnibias.Certificate

/-! Operand-bound finite rational replay. The operands of every multiplication,
polynomial evaluation, and selected minor occur in the generated proposition.
This proves finite identities, not an analytic realization or a continuum limit.
-/
namespace Omnibias.RealizationReplay

abbrev Matrix := List (List Rat)
abbrev Polynomial := List (List Nat × Rat)

def monomial (powers : List Nat) (point : List Rat) : Rat :=
  ((powers.zip point).map fun p => p.2 ^ p.1).foldl (· * ·) 1

def evaluate (p : Polynomial) (point : List Rat) : Rat :=
  (p.map fun t => t.2 * monomial t.1 point).foldl (· + ·) 0

def dot (a b : List Rat) : Rat :=
  ((a.zip b).map fun p => p.1 * p.2).foldl (· + ·) 0

def multiply (a b : Matrix) (cols : Nat) : Matrix :=
  a.map fun row => (List.range cols).map fun j =>
    dot row (b.map fun r => r.getD j 0)

def minor (a : Matrix) (rows cols : List Nat) : Matrix :=
  rows.map fun i => cols.map fun j => (a.getD i []).getD j 0

def determinant : Nat → Matrix → Rat
  | 0, _ => 1
  | n + 1, a => ((List.range (n + 1)).map fun j =>
      (if j % 2 = 0 then (1 : Rat) else -1) * (a.getD 0 []).getD j 0 *
        determinant n ((a.drop 1).map fun row => row.eraseIdx j)).foldl (· + ·) 0

def shape (a : Matrix) (rows cols : Nat) : Bool :=
  a.length == rows && a.all (fun row => row.length == cols)

abbrev RI := Rat × Rat
abbrev RIMatrix := List (List RI)

def imul (a b : RI) : RI :=
  let x := a.1 * b.1
  let y := a.1 * b.2
  let z := a.2 * b.1
  let w := a.2 * b.2
  (min (min x y) (min z w), max (max x y) (max z w))

def isub (a b : RI) : RI := (a.1 - b.2, a.2 - b.1)

def iadd (a b : RI) : RI := (a.1 + b.1, a.2 + b.2)

def irecip (a : RI) : RI := (1 / a.2, 1 / a.1)

def iget (a : RIMatrix) (i j : Nat) : RI := (a.getD i []).getD j (0, 0)

/- The positive pivot gate is checked before its reciprocal is used. This
finite checker binds every Schur complement to the supplied matrix endpoints.
It is not itself a formal theorem about an arbitrary real matrix in that box. -/
def positivePivots : Nat → RIMatrix → Bool
  | 0, _ => true
  | n + 1, a =>
    let d := iget a 0 0
    decide (0 < d.1 ∧ d.1 ≤ d.2) && positivePivots n
      ((List.range n).map fun i => (List.range n).map fun j =>
        isub (iget a (i + 1) (j + 1))
          (imul (imul (iget a (i + 1) 0) (iget a 0 (j + 1))) (irecip d)))

def validSymmetric (a : RIMatrix) (n : Nat) : Bool :=
  a.length == n && a.all (fun row => row.length == n) &&
  (List.range n).all (fun i => (List.range n).all fun j =>
    let x := iget a i j
    decide (x.1 ≤ x.2 ∧ x = iget a j i))

def strictIncluded (inner outer : List RI) : Bool :=
  inner.length == outer.length && ((inner.zip outer).all fun p =>
    decide (p.2.1 < p.1.1 ∧ p.1.1 ≤ p.1.2 ∧ p.1.2 < p.2.2))

def included (inner outer : List RI) : Bool :=
  inner.length == outer.length && ((inner.zip outer).all fun p =>
    decide (p.2.1 ≤ p.1.1 ∧ p.1.1 ≤ p.1.2 ∧ p.1.2 ≤ p.2.2))

def errorBudget (error : RI) (budget : Rat) : Bool :=
  decide (0 ≤ budget ∧ error.1 ≤ error.2 ∧ -budget ≤ error.1 ∧ error.2 ≤ budget)

def isum (xs : List RI) : RI := xs.foldl iadd (0, 0)

def krawczykResidual (h : RIMatrix) (r : Matrix) (n : Nat) : RIMatrix :=
  (List.range n).map fun i => (List.range n).map fun j =>
    let id := if i = j then (1 : Rat) else 0
    isub (id, id) (isum ((List.range n).map fun k =>
      let coeff := (r.getD i []).getD k 0
      imul (coeff, coeff) (iget h k j)))

def krawczykImage (center : List Rat) (gradient : List RI) (h : RIMatrix)
    (r : Matrix) (box : List RI) : List RI :=
  let n := center.length
  let residual := krawczykResidual h r n
  (List.range n).map fun i =>
    let c := center.getD i 0
    let base := isub (c, c) (isum ((List.range n).map fun j =>
      let coeff := (r.getD i []).getD j 0
      imul (coeff, coeff) (gradient.getD j (0, 0))))
    iadd base (isum ((List.range n).map fun j =>
      let cj := center.getD j 0
      imul (iget residual i j) (isub (box.getD j (0, 0)) (cj, cj))))

def rowContraction (residual : RIMatrix) : Bool :=
  residual.all fun row => decide ((row.map fun x => max (max x.1 (-x.1)) (max x.2 (-x.2))).foldl (· + ·) 0 < 1)

def krawczykDataValid (center : List Rat) (gradient : List RI) (h : RIMatrix)
    (r : Matrix) (box : List RI) : Bool :=
  let n := center.length
  decide (0 < n) && gradient.length == n && box.length == n && shape r n n &&
  h.length == n && h.all (fun row => row.length == n) &&
  gradient.all (fun x => decide (x.1 ≤ x.2)) &&
  h.all (fun row => row.all fun x => decide (x.1 ≤ x.2)) &&
  ((center.zip box).all fun p => decide (p.2.1 ≤ p.1 ∧ p.1 ≤ p.2.2))

end Omnibias.RealizationReplay
