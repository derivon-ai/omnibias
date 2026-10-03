# Exact Viro patchwork and the selected octic target

`omnibias.geometry.patchwork` provides the exact combinatorial half of a
planar Viro construction:

- all degree-\(d\) Newton lattice points;
- complete face-to-face unimodular triangulation validation;
- quadrant-reflected signs;
- midpoint T-curve segments with antipodal projective-boundary gluing;
- exact connected components and the rooted complement-region tree.

For even degree, that rooted tree is the oval nesting tree. No rasterization,
floating point, or visual component classifier participates in acceptance.

## Exact quartic replay

```python
from omnibias.geometry.patchwork import (
    SignDistribution,
    patchwork_curve,
    staircase_triangulation,
)

quartic = staircase_triangulation(4)
quartic_signs = SignDistribution.create(
    4,
    {point: (point[0] * point[1]) % 2 for point in quartic.vertices},
)
quartic_curve = patchwork_curve(quartic, quartic_signs)

assert len(quartic.triangles) == 16
assert len(quartic.edges) == 30
assert quartic_curve.component_count == 4
assert quartic_curve.rooted_tree == ((), (), (), ())
```

This is GP1, the known four-oval quartic control.

## Rational lower-hull and Farkas checks

`regular_height_system` assembles
\(h_p-\ell_\tau(p)\geq1\) for every declared triangle and every other lattice
point. A floating convex solve may propose heights, but the certificate
multiplies out every inequality over \(\mathbb Q\).

```python
from omnibias.geometry.patchwork_height_lp import (
    certify_regular_heights,
    verify_regular_height_certificate,
)

quartic_heights = {
    point: point[0] ** 2 + point[1] ** 2 + point[0] * point[1]
    for point in quartic.vertices
}
regular = certify_regular_heights(quartic, quartic_heights)
assert verify_regular_height_certificate(regular)
assert min(regular.feasibility.residuals) == 0
```

For a system \(Mx\geq b\), infeasibility is accepted only from
\(\lambda\geq0\), \(\lambda^\mathsf TM=0\), and
\(\lambda^\mathsf Tb>0\), all checked exactly. GP2 applies this to the
standard six-point nonregular planar triangulation through
`lower_hull_inequalities`, rather than only to a synthetic contradictory pair.

## Search and direct acceptance

`PatchworkSearchFamily` is deliberately `complete=False`. CSP soft arc
consistency and `anneal_descent` produce sign seeds; `run_discovery` accepts
only an exact target tree with exact regular heights. A finite miss is
`search_incomplete`.

The selected degree-eight target is

\[
\langle4\sqcup1\langle2\sqcup1\langle14\rangle\rangle\rangle.
\]

`certify_patchwork_realization` is a stricter, independent gate. It forms an
explicit rational \(F_t\), finds exact projective complex-smoothness
identities, and requires 22 rational polygonal annuli whose parent forest
matches the target. Viro asymptotics are not substituted for this direct
coefficient-level check.

The current smoke passes GP1--GP4 and reports `search_incomplete`; GP5 is
false because no target candidate has been found. The existing 16-oval octic
does recertify through polygonal annuli. Its finite Bezout coefficient
identities and signed Bernstein margins earn a genuine Mathlib-free
`polynomial_identity_q` Lean pass where the toolchain is installed. Lean does
not formalize the Harnack or ambient-isotopy implication. An eight-seed full
sweep evaluated 16,384 exact candidates and remained `search_incomplete`.

See [`patchwork_octic_smoke.json`](../benchmarks/patchwork_octic_smoke.json)
and
[`HILBERT16-ALGEBRAIC-TARGET.md`](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-dynamics/HILBERT16-ALGEBRAIC-TARGET.md).
