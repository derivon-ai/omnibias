/- SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
Copyright (C) 2026 Derivon -/
import OmnibiasAnalytic.RealizationReplay

/-! Finite operand replay for polynomial substitution, Sturm arithmetic,
Laurent cancellation, and rational subdivision. Analytic interpretations
(Sturm's real-root theorem, interval soundness, a Laurent limit) are separate.
-/
namespace OmnibiasAnalytic.RealizationAlgebra
open OmnibiasAnalytic.RealizationReplay

abbrev UP := List Rat

def unorm (p : UP) : UP := (p.reverse.dropWhile (· == 0)).reverse
def uadd (a b : UP) : UP := unorm ((List.range (max a.length b.length)).map fun i => a.getD i 0 + b.getD i 0)
def uneg (a : UP) : UP := a.map (- ·)
def umul (a b : UP) : UP := unorm ((List.range (a.length + b.length)).map fun k =>
  ((List.range (k + 1)).map fun i => a.getD i 0 * b.getD (k - i) 0).foldl (· + ·) 0)
def upow : UP → Nat → UP
  | _, 0 => [1]
  | p, n + 1 => umul p (upow p n)
def ueval (p : UP) (x : Rat) : Rat := p.foldr (fun c v => c + x * v) 0
def uderivative (p : UP) : UP := unorm ((List.range (p.length - 1)).map fun i => (i + 1 : Nat) * p.getD (i + 1) 0)
def ucompose (p : Polynomial) (parameters : List UP) : UP :=
  (p.map fun term => ((term.1.zip parameters).map fun q => upow q.2 q.1).foldl umul [term.2]).foldl uadd []

def variations (values : List Rat) : Nat :=
  let signs := (values.filter (· != 0)).map fun x => decide (0 < x)
  ((signs.zip (signs.drop 1)).filter fun p => p.1 != p.2).length

def sturmArithmetic (p : UP) (chain quotients : List UP) (lo hi : Rat) : Bool :=
  decide (lo < hi ∧ ueval p lo ≠ 0 ∧ ueval p hi ≠ 0) &&
  chain.length ≥ 2 && quotients.length + 1 == chain.length &&
  chain.getD 0 [] == unorm p && chain.getD 1 [] == uderivative p &&
  (chain.getLastD []).length == 1 &&
  (List.range (chain.length - 1)).all (fun i =>
    let a := chain.getD i []
    let b := chain.getD (i + 1) []
    let c := chain.getD (i + 2) []
    b.length < a.length && a == uadd (umul (quotients.getD i []) b) (uneg c)) &&
  variations (chain.map fun q => ueval q lo) == variations (chain.map fun q => ueval q hi) + 1

abbrev LP := List (Int × Rat)
def lcoeff (p : LP) (k : Int) : Rat := ((p.filter fun t => t.1 == k).map Prod.snd).foldl (· + ·) 0
def lnorm (p : LP) : LP :=
  ((((p.map Prod.fst).eraseDups).map fun k => (k, lcoeff p k)).filter fun t => t.2 != 0).mergeSort
    (fun a b => decide (a.1 ≤ b.1))
def lmul (a b : LP) : LP := lnorm (a.flatMap fun x => b.map fun y => (x.1 + y.1, x.2 * y.2))
def lpow : LP → Nat → LP
  | _, 0 => [(0, 1)]
  | p, n + 1 => lmul p (lpow p n)
def lcompose (p : Polynomial) (parameters : List LP) : LP :=
  lnorm ((p.map fun term => ((term.1.zip parameters).map fun q => lpow q.2 q.1).foldl lmul [(0, term.2)]).flatten)
def laurentCheck (p : Polynomial) (parameters : List LP) (target rho error : Rat) : Bool :=
  let path := lcompose p parameters
  decide (0 < rho) && path.all (fun t => decide (0 ≤ t.1)) && lcoeff path 0 == target &&
  decide (((path.filter fun t => 0 < t.1).map fun t => max t.2 (-t.2) * rho ^ t.1.toNat).foldl (· + ·) 0 ≤ error)

def ipow (x : RI) (n : Nat) : RI :=
  let a := x.1 ^ n
  let b := x.2 ^ n
  if n % 2 = 0 then
    (if n ≠ 0 ∧ x.1 ≤ 0 ∧ 0 ≤ x.2 then 0 else min a b, max a b)
  else (a, b)
def ievaluate (p : Polynomial) (box : List RI) : RI :=
  isum (p.map fun t => ((t.1.zip box).map fun q => ipow q.2 q.1).foldl imul (t.2, t.2))

inductive BoxTree where
  | leaf (equation : Nat) (claimed : RI)
  | split (axis : Nat) (midpoint : Rat) (left right : BoxTree)

def boxReplay (polynomials : List Polynomial) (box : List RI) : BoxTree → Bool
  | .leaf i claimed => i < polynomials.length &&
      let bound := ievaluate (polynomials.getD i []) box
      bound == claimed && decide (bound.2 < 0 ∨ 0 < bound.1)
  | .split axis midpoint left right =>
      let old := box.getD axis (0, 0)
      axis < box.length && decide (old.1 < midpoint ∧ midpoint < old.2) &&
      boxReplay polynomials (box.set axis (old.1, midpoint)) left &&
      boxReplay polynomials (box.set axis (midpoint, old.2)) right

end OmnibiasAnalytic.RealizationAlgebra
