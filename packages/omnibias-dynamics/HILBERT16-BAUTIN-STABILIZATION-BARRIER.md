# Hilbert XVI H5: finite Bautin stabilization is not all-orders stabilization

H5 proposed using high-order exact jets and the exact-`Q` Buchberger engine
to prove stabilization of the Bautin ideal. The finite computation succeeds;
the all-orders inference does not.

## Certified finite result

For Bautin's normalized five-parameter quadratic family,
`omnibias.dynamics.bautin_stabilization_barrier` mechanically derives
\(V_1,V_2,V_3,V_4\), at homogeneous degrees \(4,6,8,10\). Exact polynomial
division proves

\[
V_4\in(V_1,V_2,V_3),
\]

with replayable rational cofactors and zero remainder. This is genuine
finite-order stabilization through degree ten.

The public `BautinBasis` fields now distinguish that result:

- `finite_order_stabilization_verified` may become true;
- `bautin_stabilization_verified_to_order` records the finite endpoint;
- `bautin_ideal_stabilization_proved` remains false without an all-orders
  theorem.

## Why the jet-only inference fails

A finite prefix does not determine its tail. The audit constructs two formal
coefficient continuations sharing the certified prefix through degree ten.
One keeps every later coefficient in the ideal; the other takes the next
coefficient to be \(a_2\). Exact Gröbner reduction gives

\[
a_2\notin(V_1,V_2,V_3)
\]

with nonzero remainder \(a_2\).

This is not claimed to be the return map of a quadratic vector field. It is a
falsifier of the proposed logic: no amount of finite `jet_mv` output can, by
itself, prove an infinite tail. To exclude the adversarial continuation one
must derive an independent recurrence, a finite-termination theorem, or
Bautin's all-orders theorem from the source field.

## Consequence for G2

The computed focal prefix concerns one monodromic origin. G2 asks for exact
center/Bautin generators of the physical return around arbitrary singular
graphics, plus closure and finite termination under the required return-map
operations. Neither an all-orders focal recurrence nor that physical singular
return map is derived here. Therefore:

```text
finite_order_stabilization_verified = true
bautin_ideal_stabilization_proved = false
actual_singular_return_map_derived = false
g2_passed = false
full_hilbert16_solved = false
```

Reproduce with:

```bash
python -m pytest packages/omnibias-dynamics/tests/test_bautin.py \
  packages/omnibias-dynamics/tests/test_bautin_stabilization_barrier.py -q
python benchmarks/hilbert16_bautin_stabilization.py
```
